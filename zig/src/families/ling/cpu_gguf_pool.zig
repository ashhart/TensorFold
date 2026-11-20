//! Persistent CPU workers: rows or whole experts are claimed in chunks, for exact and opt-in fast SDOT paths.
const std = @import("std");
const g = @import("core").gguf;
const rows = @import("cpu_rows.zig");
const gguf_rows = @import("cpu_gguf_rows.zig");
const sdot = @import("cpu_gguf_q4_sdot.zig");
const repack8 = @import("cpu_gguf_repack8.zig");

/// One routed expert's gate, up and down matrices; its F32 result lands in `out`, one hidden row.
pub const Expert = struct { gate: g.Tensor, up: g.Tensor, down: g.Tensor, out: [*]f32 };

/// One routed expert with optional 8x4 repacked SDOT views for opt-in fast mode.
pub const FastExpert = struct {
    gate: g.Tensor,
    up: g.Tensor,
    down: g.Tensor,
    gate_rep: ?repack8.Repacked = null,
    up_rep: ?repack8.Repacked = null,
    down_rep: ?repack8.Repacked = null,
    use_sdot_gate: bool = false,
    use_sdot_up: bool = false,
    use_sdot_down: bool = false,
    out: [*]f32,
};

const Job = struct {
    mode: enum { rows, experts, sdot_q4k, sdot_q6k, sdot_q50, sdot_q4k_x8, sdot_q6k_x8, fast_experts },
    tensor: g.Tensor = undefined,
    q4k_x8: [*]const repack8.BlockQ4Kx8 = undefined,
    q6k_x8: [*]const repack8.BlockQ6Kx8 = undefined,
    input: [*]const f32 = undefined,
    acts256: [*]const sdot.Activation256 = undefined,
    acts32: [*]const sdot.Activation32 = undefined,
    output: [*]f32 = undefined,
    count: usize = 1,
    width: usize = 0,
    blocks_per_row: usize = 0,
    inter: usize = 0,
    n: usize = 0,
    rounded: bool = true,
    limit: f32 = 0,
    experts: [*]const Expert = undefined,
    fast_experts: [*]const FastExpert = undefined,
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

fn fastExpert(j: *const Job, e: FastExpert) void {
    const hidden = j.width;
    const inter = j.inter;
    const hidden_blocks = hidden / 256;
    const inter_blocks = inter / 256;
    var gate_buf: [1024]f32 = undefined;
    var up_buf: [1024]f32 = undefined;
    const gate = gate_buf[0..inter];
    const up = up_buf[0..inter];

    if (e.use_sdot_gate) {
        if (e.gate_rep) |rep| switch (rep) {
            .q4_k => |b| repack8.projectRangeQ4Kx8(b.ptr, e.gate.bytes.ptr, j.acts256, gate.ptr, 1, hidden_blocks, inter, true, 0, inter / 8),
            .q6_k => |b| repack8.projectRangeQ6Kx8(b.ptr, e.gate.bytes.ptr, j.acts256, gate.ptr, 1, hidden_blocks, inter, true, 0, inter / 8),
        } else if (e.gate.kind == .q4_k) {
            sdot.projectRangeQ4K(e.gate.bytes.ptr, j.acts256, gate.ptr, 1, hidden_blocks, inter, true, 0, inter);
        } else {
            sdot.projectRangeQ6K(e.gate.bytes.ptr, j.acts256, gate.ptr, 1, hidden_blocks, inter, true, 0, inter);
        }
    } else range(e.gate, j.input, gate.ptr, 1, hidden, inter, true, 0, inter);

    if (e.use_sdot_up) {
        if (e.up_rep) |rep| switch (rep) {
            .q4_k => |b| repack8.projectRangeQ4Kx8(b.ptr, e.up.bytes.ptr, j.acts256, up.ptr, 1, hidden_blocks, inter, true, 0, inter / 8),
            .q6_k => |b| repack8.projectRangeQ6Kx8(b.ptr, e.up.bytes.ptr, j.acts256, up.ptr, 1, hidden_blocks, inter, true, 0, inter / 8),
        } else if (e.up.kind == .q4_k) {
            sdot.projectRangeQ4K(e.up.bytes.ptr, j.acts256, up.ptr, 1, hidden_blocks, inter, true, 0, inter);
        } else {
            sdot.projectRangeQ6K(e.up.bytes.ptr, j.acts256, up.ptr, 1, hidden_blocks, inter, true, 0, inter);
        }
    } else range(e.up, j.input, up.ptr, 1, hidden, inter, true, 0, inter);

    for (gate, up) |*v, *u| {
        var silu = rows.round(v.* * rows.sigmoid(v.*));
        if (j.limit > 0) {
            silu = @min(silu, j.limit);
            u.* = std.math.clamp(u.*, -j.limit, j.limit);
        }
        v.* = rows.round(silu * u.*);
    }

    if (e.use_sdot_down) {
        var down_acts: [8]sdot.Activation256 = undefined;
        const d_acts = down_acts[0..inter_blocks];
        if (sdot.quantize256(gate, d_acts, true)) |_| {
            if (e.down_rep) |rep| switch (rep) {
                .q4_k => |b| repack8.projectRangeQ4Kx8(b.ptr, e.down.bytes.ptr, d_acts.ptr, e.out, 1, inter_blocks, hidden, true, 0, hidden / 8),
                .q6_k => |b| repack8.projectRangeQ6Kx8(b.ptr, e.down.bytes.ptr, d_acts.ptr, e.out, 1, inter_blocks, hidden, true, 0, hidden / 8),
            } else if (e.down.kind == .q4_k) {
                sdot.projectRangeQ4K(e.down.bytes.ptr, d_acts.ptr, e.out, 1, inter_blocks, hidden, true, 0, hidden);
            } else {
                sdot.projectRangeQ6K(e.down.bytes.ptr, d_acts.ptr, e.out, 1, inter_blocks, hidden, true, 0, hidden);
            }
            return;
        } else |_| {}
    }
    range(e.down, gate.ptr, e.out, 1, inter, hidden, true, 0, hidden);
}

fn runJob(j: *const Job, lo: usize, hi: usize) void {
    switch (j.mode) {
        .rows => range(j.tensor, j.input, j.output, j.count, j.width, j.n, j.rounded, lo, hi),
        .experts => for (lo..hi) |i| expert(j, j.experts[i]),
        .sdot_q4k => sdot.projectRangeQ4K(j.tensor.bytes.ptr, j.acts256, j.output, j.count, j.blocks_per_row, j.n, j.rounded, lo, hi),
        .sdot_q6k => sdot.projectRangeQ6K(j.tensor.bytes.ptr, j.acts256, j.output, j.count, j.blocks_per_row, j.n, j.rounded, lo, hi),
        .sdot_q50 => sdot.projectRangeQ50(j.tensor.bytes.ptr, j.acts32, j.output, j.count, j.blocks_per_row, j.n, j.rounded, lo, hi),
        .sdot_q4k_x8 => repack8.projectRangeQ4Kx8(j.q4k_x8, j.tensor.bytes.ptr, j.acts256, j.output, j.count, j.blocks_per_row, j.n, j.rounded, lo, hi),
        .sdot_q6k_x8 => repack8.projectRangeQ6Kx8(j.q6k_x8, j.tensor.bytes.ptr, j.acts256, j.output, j.count, j.blocks_per_row, j.n, j.rounded, lo, hi),
        .fast_experts => for (lo..hi) |i| fastExpert(j, j.fast_experts[i]),
    }
}

fn chunkFor(units: usize, threads: usize, min_chunk: usize, max_chunk: usize, even: bool) usize {
    const target = @max(min_chunk, @min(max_chunk, units / (threads * 4)));
    var chunk = @min(target, @max(@as(usize, 1), units / (threads * 2)));
    if (even and chunk > 1 and chunk % 2 != 0) chunk += 1;
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
        for (0..threads - 1) |_| {
            p.workers[p.made] = try std.Thread.spawn(.{}, worker, .{p});
            p.made += 1;
        }
        return p;
    }

    pub fn destroy(p: *Pool) void {
        p.stopping.store(true, .release);
        p.mutex.lockUncancelable(p.io);
        p.parked.store(false, .release);
        p.ready.broadcast(p.io);
        p.mutex.unlock(p.io);
        for (p.workers[0..p.made]) |w| w.join();
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

    fn worker(p: *Pool) void {
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
        const job = Job{ .mode = .rows, .tensor = t, .input = input.ptr, .output = output.ptr, .count = count, .width = s.width, .n = s.n, .rounded = rounded, .units = s.n, .chunk = chunkFor(s.n, p.threads, 16, 256, true) };
        if (p.made == 0 or s.width * s.n < p.min_weights) return runJob(&job, 0, s.n);
        p.dispatch(job);
    }

    /// Opt-in 8x4 repacked SDOT projection across workers; output bytes match 1-thread repack8.
    pub fn projectSdotRepacked(p: *Pool, a: std.mem.Allocator, rep: repack8.Repacked, t: g.Tensor, input: []const f32, output: []f32, count: usize, rounded: bool) !void {
        try sdot.checkTarget();
        const s = try shape(t);
        if (count == 0 or s.width % 256 != 0 or s.n < 8 or input.len != try std.math.mul(usize, count, s.width) or output.len != try std.math.mul(usize, count, s.n))
            return error.BadLingGgufMatrix;
        const bpr = s.width / 256;
        const total_acts = count * bpr;
        var small_buf: [8]sdot.Activation256 = undefined;
        const acts = if (total_acts <= small_buf.len) small_buf[0..total_acts] else try a.alloc(sdot.Activation256, total_acts);
        defer if (total_acts > small_buf.len) a.free(acts);
        for (0..count) |r| try sdot.quantize256(input[r * s.width ..][0..s.width], acts[r * bpr ..][0..bpr], true);
        const groups = s.n / 8;
        const is_q4 = rep == .q4_k;
        const job = Job{
            .mode = if (is_q4) .sdot_q4k_x8 else .sdot_q6k_x8,
            .tensor = t,
            .q4k_x8 = if (is_q4) rep.q4_k.ptr else undefined,
            .q6k_x8 = if (!is_q4) rep.q6_k.ptr else undefined,
            .acts256 = acts.ptr,
            .output = output.ptr,
            .count = count,
            .width = s.width,
            .blocks_per_row = bpr,
            .n = s.n,
            .rounded = rounded,
            .units = groups,
            .chunk = chunkFor(groups, p.threads, 4, 32, false),
        };
        if (p.made == 0 or s.width * s.n < p.min_weights) return runJob(&job, 0, groups);
        p.dispatch(job);
    }

    /// Opt-in 1-row SDOT projection across workers for Q4_K, Q6_K and Q5_0; output bytes match 1-thread SDOT.
    pub fn projectSdot(p: *Pool, a: std.mem.Allocator, t: g.Tensor, input: []const f32, output: []f32, count: usize, rounded: bool) !void {
        try sdot.checkTarget();
        const s = try shape(t);
        if (count == 0 or input.len != try std.math.mul(usize, count, s.width) or output.len != try std.math.mul(usize, count, s.n))
            return error.BadLingGgufMatrix;
        switch (t.kind) {
            .q4_k, .q6_k => {
                const bpr = s.width / 256;
                const total_acts = count * bpr;
                var small_buf: [64]sdot.Activation256 = undefined;
                const acts = if (total_acts <= small_buf.len) small_buf[0..total_acts] else try a.alloc(sdot.Activation256, total_acts);
                defer if (total_acts > small_buf.len) a.free(acts);
                for (0..count) |r| {
                    if (t.kind == .q4_k) try sdot.quantize256(input[r * s.width ..][0..s.width], acts[r * bpr ..][0..bpr], true) else try sdot.quantize256(input[r * s.width ..][0..s.width], acts[r * bpr ..][0..bpr], false);
                }
                const job = Job{
                    .mode = if (t.kind == .q4_k) .sdot_q4k else .sdot_q6k,
                    .tensor = t,
                    .acts256 = acts.ptr,
                    .output = output.ptr,
                    .count = count,
                    .width = s.width,
                    .blocks_per_row = bpr,
                    .n = s.n,
                    .rounded = rounded,
                    .units = s.n,
                    .chunk = chunkFor(s.n, p.threads, 16, 256, true),
                };
                if (p.made == 0 or s.width * s.n < p.min_weights) return runJob(&job, 0, s.n);
                p.dispatch(job);
            },
            .q5_0 => {
                const bpr = s.width / 32;
                const total_acts = count * bpr;
                var small_buf: [64]sdot.Activation32 = undefined;
                const acts = if (total_acts <= small_buf.len) small_buf[0..total_acts] else try a.alloc(sdot.Activation32, total_acts);
                defer if (total_acts > small_buf.len) a.free(acts);
                for (0..count) |r| try sdot.quantize32(input[r * s.width ..][0..s.width], acts[r * bpr ..][0..bpr]);
                const job = Job{
                    .mode = .sdot_q50,
                    .tensor = t,
                    .acts32 = acts.ptr,
                    .output = output.ptr,
                    .count = count,
                    .width = s.width,
                    .blocks_per_row = bpr,
                    .n = s.n,
                    .rounded = rounded,
                    .units = s.n,
                    .chunk = chunkFor(s.n, p.threads, 16, 256, true),
                };
                if (p.made == 0 or s.width * s.n < p.min_weights) return runJob(&job, 0, s.n);
                p.dispatch(job);
            },
            else => return error.UnsupportedLingGgufQuant,
        }
    }

    /// One row's routed experts in fast mode, one expert a claim; each writes only its own `out`.
    pub fn expertsFast(p: *Pool, list: []const FastExpert, input: []const f32, acts256: [*]const sdot.Activation256, hidden: usize, inter: usize, limit: f32) !void {
        if (inter == 0 or inter > 1024 or input.len != hidden) return error.BadLingGgufMatrix;
        for (list) |e| for ([_]g.Tensor{ e.gate, e.up, e.down }, [_]usize{ hidden, hidden, inter }) |t, width| {
            const s = try shape(t);
            if (s.width != width or s.n != hidden + inter - width) return error.BadLingGgufMatrix;
        };
        const job = Job{ .mode = .fast_experts, .input = input.ptr, .acts256 = acts256, .width = hidden, .inter = inter, .limit = limit, .fast_experts = list.ptr, .units = list.len, .chunk = 1 };
        if (p.made == 0) return runJob(&job, 0, list.len);
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
