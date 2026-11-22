//! Persistent exact workers: rows or whole experts are claimed in chunks, and each dot keeps its single-thread order.
const std = @import("std");
const g = @import("core").gguf;
const rows = @import("cpu_rows.zig");
const gguf_rows = @import("cpu_gguf_rows.zig");

/// One routed expert's gate, up and down matrices; its F32 result lands in `out`, one hidden row.
pub const Expert = struct { gate: g.Tensor, up: g.Tensor, down: g.Tensor, out: [*]f32 };

const Job = struct {
    mode: enum { rows, experts },
    tensor: g.Tensor = undefined,
    input: [*]const f32 = undefined,
    output: [*]f32 = undefined,
    count: usize = 1,
    width: usize = 0,
    inter: usize = 0,
    n: usize = 0,
    rounded: bool = true,
    limit: f32 = 0,
    experts: [*]const Expert = undefined,
    units: usize = 0,
    chunk: usize = 1,
};

fn range(t: g.Tensor, input: [*]const f32, output: [*]f32, count: usize, width: usize, n: usize, rounded: bool, lo: usize, hi: usize) void {
    if (t.kind == .f32) {
        const dense = rows.Tensor{ .dtype = .f32, .rank = 2, .shape = .{ n, width, 1, 1, 1 }, .bytes = t.bytes };
        return rows.projectRange(dense, input[0 .. count * width], output[0 .. count * n], count, width, n, rounded, lo, hi);
    }
    gguf_rows.dispatchRange(t.kind, t.bytes.ptr, input, output, count, width, n, rounded, lo, hi);
}

/// The serial forward's SwiGLU expert for one row, with the same rounding and clamp at each step.
fn expert(j: *const Job, e: Expert) void {
    var gate_buf: [1024]f32 = undefined;
    var up_buf: [1024]f32 = undefined;
    const gate = gate_buf[0..j.inter];
    const up = up_buf[0..j.inter];
    range(e.gate, j.input, gate.ptr, 1, j.width, j.inter, true, 0, j.inter);
    range(e.up, j.input, up.ptr, 1, j.width, j.inter, true, 0, j.inter);
    for (gate, up) |*v, *u| {
        var silu = rows.round(v.* * rows.sigmoid(v.*));
        if (j.limit > 0) {
            silu = @min(silu, j.limit);
            u.* = std.math.clamp(u.*, -j.limit, j.limit);
        }
        v.* = rows.round(silu * u.*);
    }
    range(e.down, gate.ptr, e.out, 1, j.inter, j.width, true, 0, j.width);
}

fn runJob(j: *const Job, lo: usize, hi: usize) void {
    switch (j.mode) {
        .rows => range(j.tensor, j.input, j.output, j.count, j.width, j.n, j.rounded, lo, hi),
        .experts => for (lo..hi) |i| expert(j, j.experts[i]),
    }
}

fn chunkFor(units: usize, threads: usize) usize {
    // About four claims a thread, 16 to 256 rows each, but never fewer than two chunks a thread.
    const target = @max(@as(usize, 16), @min(@as(usize, 256), units / (threads * 4)));
    var chunk = @min(target, @max(@as(usize, 1), units / (threads * 2)));
    if (chunk > 1 and chunk % 2 != 0) chunk += 1;
    return chunk;
}

fn shape(t: g.Tensor) !struct { width: usize, n: usize } {
    if (t.rank != 2) return error.BadLingGgufMatrix;
    const width = std.math.cast(usize, t.shape[0]) orelse return error.BadLingGgufMatrix;
    const n = std.math.cast(usize, t.shape[1]) orelse return error.BadLingGgufMatrix;
    if (width == 0 or n == 0) return error.BadLingGgufMatrix;
    const bytes = if (t.kind == .f32) try std.math.mul(usize, try std.math.mul(usize, n, width), 4) else blk: {
        try gguf_rows.supported(t.kind);
        const block = t.kind.block();
        if (width % block.values != 0) return error.BadLingGgufMatrix;
        break :blk try std.math.mul(usize, n, try std.math.mul(usize, width / block.values, block.bytes));
    };
    if (t.bytes.len != bytes) return error.BadLingGgufMatrix;
    return .{ .width = width, .n = n };
}

/// Extract up to `out.len` set CPU indices from a bitmask of `usize` words, in ascending order.
pub fn extractAllowedCpus(words: []const usize, out: []usize) usize {
    const bits = @bitSizeOf(usize);
    var count: usize = 0;
    for (words, 0..) |word, wi| {
        var w = word;
        while (w != 0 and count < out.len) {
            const bit = @ctz(w);
            out[count] = wi * bits + bit;
            count += 1;
            w &= w - 1;
        }
    }
    return count;
}

fn pinSingleCpu(cpu: usize) void {
    if (@import("builtin").os.tag != .linux) return;
    const bits = @bitSizeOf(usize);
    var set: std.os.linux.cpu_set_t = @splat(0);
    if (cpu >= set.len * bits) return;
    set[cpu / bits] = @as(usize, 1) << @intCast(cpu % bits);
    std.os.linux.sched_setaffinity(0, &set) catch {};
}

/// Workers spin briefly between jobs, which come every few microseconds within a token, then park until woken.
pub const Pool = struct {
    const spin_limit: usize = 500_000;

    allocator: std.mem.Allocator,
    io: std.Io,
    threads: usize,
    /// Smaller matrices run on the calling thread; dispatch costs more than their work.
    min_weights: usize = 256 * 256,
    workers: [7]std.Thread = undefined,
    made: usize = 0,
    allowed_cpus: [8]usize = @splat(0),
    pin_enabled: bool = false,
    saved_mask: if (@import("builtin").os.tag == .linux) std.os.linux.cpu_set_t else void =
        if (@import("builtin").os.tag == .linux) @splat(0) else {},
    mutex: std.Io.Mutex = .init,
    ready: std.Io.Condition = .init,
    generation: std.atomic.Value(usize) align(64) = .init(0),
    remaining: std.atomic.Value(usize) align(64) = .init(0),
    next: std.atomic.Value(usize) align(64) = .init(0),
    stopping: std.atomic.Value(bool) align(64) = .init(false),
    parked: std.atomic.Value(bool) align(64) = .init(false),
    job: Job align(64) = undefined,

    pub fn create(a: std.mem.Allocator, io: std.Io, threads: usize) !*Pool {
        if (threads == 0 or threads > 8) return error.CpuThreadCount;
        const p = try a.create(Pool);
        p.* = .{ .allocator = a, .io = io, .threads = threads };
        errdefer p.destroy();
        if (@import("builtin").os.tag == .linux and threads > 1) {
            var mask: std.os.linux.cpu_set_t = @splat(0);
            const rc = std.os.linux.sched_getaffinity(0, @sizeOf(std.os.linux.cpu_set_t), &mask);
            if (std.os.linux.errno(rc) == .SUCCESS) {
                if (extractAllowedCpus(&mask, p.allowed_cpus[0..threads]) >= threads) {
                    p.saved_mask = mask;
                    p.pin_enabled = true;
                }
            }
        }
        for (0..threads - 1) |id| {
            p.workers[p.made] = try std.Thread.spawn(.{}, worker, .{ p, id });
            p.made += 1;
        }
        if (p.pin_enabled) pinSingleCpu(p.allowed_cpus[0]);
        return p;
    }

    pub fn destroy(p: *Pool) void {
        p.stopping.store(true, .release);
        p.mutex.lockUncancelable(p.io);
        p.parked.store(false, .release);
        p.ready.broadcast(p.io);
        p.mutex.unlock(p.io);
        for (p.workers[0..p.made]) |w| w.join();
        if (@import("builtin").os.tag == .linux and p.pin_enabled) {
            std.os.linux.sched_setaffinity(0, &p.saved_mask) catch {};
        }
        p.allocator.destroy(p);
    }

    fn claim(p: *Pool) void {
        const j = &p.job;
        while (true) {
            const lo = p.next.fetchAdd(j.chunk, .monotonic);
            if (lo >= j.units) break;
            runJob(j, lo, @min(j.units, lo + j.chunk));
        }
    }

    fn worker(p: *Pool, id: usize) void {
        if (p.pin_enabled) pinSingleCpu(p.allowed_cpus[id + 1]);
        var seen: usize = 0;
        while (true) {
            var spins: usize = 0;
            while (true) {
                if (p.stopping.load(.acquire)) return;
                const gen = p.generation.load(.acquire);
                if (gen != seen) {
                    seen = gen;
                    break;
                }
                if (spins < spin_limit) {
                    spins += 1;
                    std.atomic.spinLoopHint();
                } else {
                    p.mutex.lockUncancelable(p.io);
                    p.parked.store(true, .release);
                    while (!p.stopping.load(.acquire) and p.generation.load(.acquire) == seen) p.ready.waitUncancelable(p.io, &p.mutex);
                    p.mutex.unlock(p.io);
                    spins = 0;
                }
            }
            p.claim();
            _ = p.remaining.fetchSub(1, .acq_rel);
        }
    }

    /// Every worker takes part in every job, so the next job never starts while one still runs.
    fn dispatch(p: *Pool, job: Job) void {
        p.job = job;
        p.next.store(0, .monotonic);
        p.remaining.store(p.made, .release);
        _ = p.generation.fetchAdd(1, .acq_rel);
        if (p.parked.load(.acquire)) {
            p.mutex.lockUncancelable(p.io);
            p.parked.store(false, .release);
            p.ready.broadcast(p.io);
            p.mutex.unlock(p.io);
        }
        p.claim();
        while (p.remaining.load(.acquire) != 0) std.atomic.spinLoopHint();
    }

    /// The same bytes as gguf_rows.projectValidatedOn at one thread; quantized scales must already be validated.
    pub fn project(p: *Pool, t: g.Tensor, input: []const f32, output: []f32, count: usize, rounded: bool) !void {
        const s = try shape(t);
        if (count == 0 or input.len != try std.math.mul(usize, count, s.width) or output.len != try std.math.mul(usize, count, s.n))
            return error.BadLingGgufMatrix;
        const job = Job{ .mode = .rows, .tensor = t, .input = input.ptr, .output = output.ptr, .count = count, .width = s.width, .n = s.n, .rounded = rounded, .units = s.n, .chunk = chunkFor(s.n, p.threads) };
        if (p.made == 0 or s.width * s.n < p.min_weights) return runJob(&job, 0, s.n);
        p.dispatch(job);
    }

    /// One row's routed experts, one expert a claim; each writes only its own `out`.
    pub fn experts(p: *Pool, list: []const Expert, input: []const f32, hidden: usize, inter: usize, limit: f32) !void {
        if (inter == 0 or inter > 1024 or input.len != hidden) return error.BadLingGgufMatrix;
        for (list) |e| for ([_]g.Tensor{ e.gate, e.up, e.down }, [_]usize{ hidden, hidden, inter }) |t, width| {
            const s = try shape(t);
            if (s.width != width or s.n != hidden + inter - width) return error.BadLingGgufMatrix;
        };
        const job = Job{ .mode = .experts, .input = input.ptr, .width = hidden, .inter = inter, .limit = limit, .experts = list.ptr, .units = list.len, .chunk = 1 };
        if (p.made == 0) return runJob(&job, 0, list.len);
        p.dispatch(job);
    }
};
