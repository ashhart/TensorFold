//! TF_DSV41_ENGRAM_GATE: the GPU, not the host, waits for a decode window's Engram rows (same bits as the old path).

const std = @import("std");
const quiet = @import("builtin").is_test; // a test's stderr is the build runner's channel
const cuda = @import("cuda");
const eh = @import("engram_host.zig");

const log = std.log.scoped(.dsv41);

pub const ctl_want = 0;
pub const ctl_ready = 1;
pub const ctl_err = 7;
pub const ctl_n = 8;

pub const Settings = struct {
    on: bool = false,
    rows: u32 = 128,
    timeout_ns: i64 = 30 * std.time.ns_per_s,
    wait_ns: u64 = 300 * std.time.ns_per_s,
};

fn get(name: [:0]const u8) ?[]const u8 {
    const v = std.mem.trim(u8, std.mem.span(std.c.getenv(name) orelse return null), " \t");
    return if (v.len == 0) null else v;
}

pub fn settings() !Settings {
    var s: Settings = .{};
    const raw = get("TF_DSV41_ENGRAM_GATE") orelse return s;
    s.on = !std.mem.eql(u8, raw, "0");
    if (get("TF_DSV41_ENGRAM_GATE_ROWS")) |v| s.rows = std.fmt.parseInt(u32, v, 10) catch return error.BadEngramGate;
    if (s.rows == 0) return error.BadEngramGate;
    if (get("TF_DSV41_ENGRAM_GATE_TIMEOUT_S")) |v| s.timeout_ns = @intFromFloat(1e9 * (std.fmt.parseFloat(f64, v) catch return error.BadEngramGate));
    if (get("TF_DSV41_ENGRAM_WAIT_S")) |v| s.wait_ns = @intFromFloat(1e9 * (std.fmt.parseFloat(f64, v) catch return error.BadEngramGate));
    return s;
}

pub const Item = @import("forward_types.zig").Item;

const Job = struct {
    /// 0: a warm job (reads only, nothing published)
    seq: u64,
    /// the window's rows and the rows to fill (a row-mode bucket's padding: copies of the last row)
    n: u32,
    fill: u32,
    ids: []u32,
    tails: []u32,
    /// per item: ids and tail lengths
    lens: []u32,
    finished: bool = false,
    failed: ?anyerror = null,

    fn free(j: *Job, a: std.mem.Allocator) void {
        a.free(j.ids);
        a.free(j.tails);
        a.free(j.lens);
        a.destroy(j);
    }

    fn item(j: *const Job, i: usize, at: *[2]usize) Item {
        const ni = j.lens[2 * i];
        const nt = j.lens[2 * i + 1];
        defer at.* = .{ at[0] + ni, at[1] + nt };
        return .{ .ids = j.ids[at[0]..][0..ni], .tail = j.tails[at[1]..][0..nt] };
    }

    fn items(j: *const Job) usize {
        return j.lens.len / 2;
    }
};

/// The gate's memory: control words and staging [2][layers][rows][cols] u16, host memory the GPU reads (0 in tests).
pub const Memory = struct {
    ctl: []i64,
    staging: []u16,
    ctl_dev: u64 = 0,
    stage_dev: u64 = 0,
};

pub const Gate = struct {
    a: std.mem.Allocator,
    host: *const eh.Host,
    /// the rank's table (engram_rows), read only under `lock`
    src: eh.Source,
    /// the Engram layers in the forward's order and their index in the host's tables
    layers: []const u32,
    rank: u32,
    world: u32,
    dim: usize,
    /// this rank's columns of a row (its heads x dim)
    cols: usize,
    rows: u32,
    timeout_ns: i64,
    wait_ns: u64,
    mem: Memory,
    lock: std.c.pthread_mutex_t = std.c.PTHREAD_MUTEX_INITIALIZER,
    mu: std.c.pthread_mutex_t = std.c.PTHREAD_MUTEX_INITIALIZER,
    work: std.c.pthread_cond_t = .{},
    done: std.c.pthread_cond_t = .{},
    jobs: std.ArrayList(*Job) = .empty,
    stop: bool = false,
    thread: ?std.Thread = null,
    seq: u64 = 0,
    /// the round the control word names (owned until the next arm)
    armed: ?*Job = null,
    stats: Stats = .{},
    /// the device side (`open`; null in host tests): the pinned buffers, and the stream a new round waits for
    gpu: ?*anyopaque = null,
    gpu_free: ?*const fn (a: std.mem.Allocator, gpu: *anyopaque) void = null,
    sync: ?Sync = null,

    pub const Stats = struct {
        rounds: u64 = 0,
        warm: u64 = 0,
        syncs: u64 = 0,
        failed: u64 = 0,
        /// TF_DSV41_WIN_PROF=1: host ns of each arm's parts (sync, previous wait, job, push), logged at deinit
        prof: bool = false,
        sync_ns: u64 = 0,
        wait_ns: u64 = 0,
        push_ns: u64 = 0,
    };
    pub const Sync = struct { ctx: *anyopaque, run: *const fn (ctx: *anyopaque) anyerror!void };

    /// The gate over `src` with its own worker; `mem` sized by `memBytes` (host tests pass plain slices).
    pub fn init(a: std.mem.Allocator, host: *const eh.Host, src: eh.Source, layers: []const u32, rank: u32, world: u32, dim: usize, s: Settings, mem: Memory) !*Gate {
        if (layers.len == 0 or layers.len > ctl_err - ctl_ready) return error.BadEngramGate;
        const g = try a.create(Gate);
        errdefer a.destroy(g);
        const cols = host.headShard(rank, world)[1] * dim;
        g.* = .{ .a = a, .host = host, .src = src, .layers = try a.dupe(u32, layers), .rank = rank, .world = world, .dim = dim, .cols = cols, .rows = s.rows, .timeout_ns = s.timeout_ns, .wait_ns = s.wait_ns, .mem = mem };
        g.stats.prof = if (std.c.getenv("TF_DSV41_WIN_PROF")) |v| std.mem.eql(u8, std.mem.span(v), "1") else false;
        if (mem.ctl.len < ctl_n or mem.staging.len < 2 * layers.len * s.rows * cols) return error.BadEngramGate;
        @memset(mem.ctl, 0);
        g.thread = try std.Thread.spawn(.{ .stack_size = 1 << 20 }, loop, .{g});
        return g;
    }

    /// The pinned, device-mapped memory for `layers` x `rows` x this rank's columns, then `init`.
    pub fn open(a: std.mem.Allocator, d: *const cuda.Driver, host: *const eh.Host, src: eh.Source, layers: []const u32, rank: u32, world: u32, dim: usize, s: Settings) !*Gate {
        const cols = host.headShard(rank, world)[1] * dim;
        var ctl = try cuda.HostBuffer.allocMapped(d, 8 * ctl_n);
        errdefer ctl.free();
        var st = try cuda.HostBuffer.allocMapped(d, 2 * 2 * layers.len * s.rows * cols);
        errdefer st.free();
        const mem: Memory = .{
            .ctl = @alignCast(std.mem.bytesAsSlice(i64, ctl.bytes)),
            .staging = @alignCast(std.mem.bytesAsSlice(u16, st.bytes)),
            .ctl_dev = try ctl.device(),
            .stage_dev = try st.device(),
        };
        @memset(st.bytes, 0);
        const g = try init(a, host, src, layers, rank, world, dim, s, mem);
        const Gpu = struct {
            ctl: cuda.HostBuffer,
            staging: cuda.HostBuffer,
            fn free(al: std.mem.Allocator, p: *anyopaque) void {
                const x: *@This() = @ptrCast(@alignCast(p));
                x.ctl.free();
                x.staging.free();
                al.destroy(x);
            }
        };
        const gp = try a.create(Gpu);
        gp.* = .{ .ctl = ctl, .staging = st };
        g.gpu = gp;
        g.gpu_free = Gpu.free;
        log.info("engram gate: on, {d} layers, windows <= {d} rows, {d} columns a row on this rank, GPU wait <= {d} s", .{ layers.len, s.rows, cols, @divTrunc(s.timeout_ns, std.time.ns_per_s) });
        return g;
    }

    pub fn close(g: *Gate) void {
        _ = std.c.pthread_mutex_lock(&g.mu);
        g.stop = true;
        _ = std.c.pthread_cond_broadcast(&g.work);
        _ = std.c.pthread_mutex_unlock(&g.mu);
        if (g.thread) |t| t.join();
        for (g.jobs.items) |j| j.free(g.a);
        g.jobs.deinit(g.a);
        if (g.armed) |j| j.free(g.a);
        if (g.gpu) |x| g.gpu_free.?(g.a, x);
        log.info("engram gate: {d} rounds, {d} warm jobs, {d} syncs, {d} failed", .{ g.stats.rounds, g.stats.warm, g.stats.syncs, g.stats.failed });
        if (g.stats.prof and g.stats.rounds > 0) {
            const n: f64 = @floatFromInt(g.stats.rounds);
            log.info("engram gate arm (rank {d}, us a round): sync {d:.1}, previous round's wait {d:.1}, job + push {d:.1}", .{ g.rank, @as(f64, @floatFromInt(g.stats.sync_ns)) / 1e3 / n, @as(f64, @floatFromInt(g.stats.wait_ns)) / 1e3 / n, @as(f64, @floatFromInt(g.stats.push_ns)) / 1e3 / n });
        }
        g.a.free(g.layers);
        g.a.destroy(g);
    }

    /// Whether a window of `rows` rows (its bucket's) goes through the gate.
    pub fn eligible(g: *const Gate, rows: usize) bool {
        return rows > 0 and rows <= g.rows;
    }

    /// The staging of layer slot `k` in the parity buffer of round `seq` (host side).
    pub fn stagingOf(g: *const Gate, seq: u64, k: usize) []u16 {
        const per = @as(usize, g.rows) * g.cols;
        return g.mem.staging[((seq & 1) * g.layers.len + k) * per ..][0..per];
    }

    /// Byte offset of layer slot `k` in parity 0, and the bytes between the two parities (the kernel's base / stride).
    pub fn layerBase(g: *const Gate, k: usize) u64 {
        return g.mem.stage_dev + 2 * @as(u64, k) * g.rows * g.cols;
    }
    pub fn parityStride(g: *const Gate) i64 {
        return @intCast(2 * g.layers.len * @as(usize, g.rows) * g.cols);
    }

    /// The layer slot of Engram layer L (its control word is ctl_ready + slot).
    pub fn slotOf(g: *const Gate, L: u32) ?usize {
        return std.mem.indexOfScalar(u32, g.layers, L);
    }

    fn job(g: *Gate, seq: u64, items: []const Item, fill: u32) !*Job {
        var ni: usize = 0;
        var nt: usize = 0;
        for (items) |it| {
            ni += it.ids.len;
            nt += it.tail.len;
        }
        const j = try g.a.create(Job);
        errdefer g.a.destroy(j);
        j.* = .{ .seq = seq, .n = @intCast(ni), .fill = fill, .ids = try g.a.alloc(u32, ni), .tails = try g.a.alloc(u32, nt), .lens = try g.a.alloc(u32, 2 * items.len) };
        var at: [2]usize = .{ 0, 0 };
        for (items, 0..) |it, i| {
            @memcpy(j.ids[at[0]..][0..it.ids.len], it.ids);
            @memcpy(j.tails[at[1]..][0..it.tail.len], it.tail);
            j.lens[2 * i] = @intCast(it.ids.len);
            j.lens[2 * i + 1] = @intCast(it.tail.len);
            at = .{ at[0] + it.ids.len, at[1] + it.tail.len };
        }
        return j;
    }

    fn push(g: *Gate, j: *Job) void {
        _ = std.c.pthread_mutex_lock(&g.mu);
        defer _ = std.c.pthread_mutex_unlock(&g.mu);
        g.jobs.append(g.a, j) catch {
            j.failed = error.OutOfMemory;
            j.finished = true;
            return;
        };
        _ = std.c.pthread_cond_broadcast(&g.work);
    }

    /// Reads ahead of a window (a commit's bonus row, a pass's drafts): cache only, nothing published.
    pub fn warm(g: *Gate, items: []const Item) void {
        const j = g.job(0, items, 0) catch return;
        g.stats.warm += 1;
        g.push(j);
    }

    /// Arms a window over `items`: syncs the earlier rounds' stream, queues the round job, sets the control word.
    pub fn arm(g: *Gate, items: []const Item, fill: u32) !void {
        var t = if (g.stats.prof) nowNs() else 0;
        if (g.armed) |prev| {
            if (g.sync) |s| {
                try s.run(s.ctx); // every window armed before has read its rows
                g.stats.syncs += 1;
            }
            if (g.stats.prof) g.stats.sync_ns += lap(&t);
            try g.wait(prev);
            if (g.stats.prof) g.stats.wait_ns += lap(&t);
            const failed = prev.failed;
            prev.free(g.a);
            g.armed = null;
            if (failed) |e| return e;
            try g.flagged();
        }
        g.seq += 1;
        const j = try g.job(g.seq, items, fill);
        if (j.n == 0 or j.n > fill or fill > g.rows) {
            j.free(g.a);
            return error.BadEngramGate;
        }
        g.armed = j;
        g.stats.rounds += 1;
        @as(*volatile i64, &g.mem.ctl[ctl_want]).* = @intCast(j.seq); // before the launches that read it
        g.push(j);
        if (g.stats.prof) g.stats.push_ns += lap(&t);
    }

    fn nowNs() u64 {
        var ts: std.c.timespec = undefined;
        _ = std.c.clock_gettime(.MONOTONIC, &ts);
        return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
    }

    /// The ns since `t`, `t` moved to now.
    fn lap(t: *u64) u64 {
        const x = nowNs();
        defer t.* = x;
        return x - t.*;
    }

    /// After a window: its reads done (TF_DSV41_ENGRAM_WAIT_S bound); a failed read or timed-out GPU wait is raised.
    pub fn check(g: *Gate) !void {
        const j = g.armed orelse return;
        try g.wait(j);
        if (j.failed) |e| return e;
        try g.flagged();
    }

    fn flagged(g: *Gate) !void {
        const p: *volatile i64 = &g.mem.ctl[ctl_err];
        const slot = p.*;
        if (slot == 0) return;
        p.* = 0;
        if (!quiet) log.warn("engram gate: the GPU waited past the timeout for layer {d}", .{g.layers[@intCast(slot - ctl_ready)]});
        return error.EngramGateTimeout;
    }

    fn wait(g: *Gate, j: *Job) !void {
        _ = std.c.pthread_mutex_lock(&g.mu);
        defer _ = std.c.pthread_mutex_unlock(&g.mu);
        if (j.finished) return;
        var ts: std.c.timespec = undefined;
        _ = std.c.clock_gettime(.REALTIME, &ts);
        const end_ns = @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec)) + g.wait_ns;
        const until: std.c.timespec = .{ .sec = @intCast(end_ns / std.time.ns_per_s), .nsec = @intCast(end_ns % std.time.ns_per_s) };
        while (!j.finished) {
            if (std.c.pthread_cond_timedwait(&g.done, &g.mu, &until) == .TIMEDOUT and !j.finished) {
                log.err("engram gate: round {d}'s reads not finished after {d} s (TF_DSV41_ENGRAM_WAIT_S)", .{ j.seq, g.wait_ns / std.time.ns_per_s });
                return error.EngramGateStalled;
            }
        }
    }

    // -- the worker ------------------------------------------------------------------------------------------------

    fn loop(g: *Gate) void {
        while (true) {
            _ = std.c.pthread_mutex_lock(&g.mu);
            while (g.jobs.items.len == 0 and !g.stop) _ = std.c.pthread_cond_wait(&g.work, &g.mu);
            if (g.jobs.items.len == 0) {
                _ = std.c.pthread_mutex_unlock(&g.mu);
                return;
            }
            const j = g.jobs.orderedRemove(0);
            _ = std.c.pthread_mutex_unlock(&g.mu);
            g.run(j) catch |e| {
                j.failed = e;
                g.stats.failed += 1;
                if (!quiet) log.warn("engram gate: round {d} failed ({t})", .{ j.seq, e });
            };
            if (j.seq != 0 and j.failed != null) for (0..g.layers.len) |k| g.publish(j, k); // the GPU must not wait on it
            _ = std.c.pthread_mutex_lock(&g.mu);
            j.finished = true;
            const warm_job = j.seq == 0;
            _ = std.c.pthread_cond_broadcast(&g.done);
            _ = std.c.pthread_mutex_unlock(&g.mu);
            if (warm_job) j.free(g.a); // nobody waits on a warm job
        }
    }

    fn run(g: *Gate, j: *Job) !void {
        _ = std.c.pthread_mutex_lock(&g.lock);
        defer _ = std.c.pthread_mutex_unlock(&g.lock);
        // every layer's reads in flight first (one batch), then rows layer by layer, published as each lands
        var at: [2]usize = .{ 0, 0 };
        for (0..j.items()) |i| {
            const it = j.item(i, &at);
            g.host.prefetch(g.a, g.src, it.ids, it.tail, g.rank, g.world);
        }
        if (j.seq == 0) return;
        for (g.layers, 0..) |L, k| {
            const out = g.stagingOf(j.seq, k);
            var row: usize = 0;
            at = .{ 0, 0 };
            for (0..j.items()) |i| {
                const it = j.item(i, &at);
                try g.host.windowRows(g.a, g.src, it.ids, it.tail, L, g.rank, g.world, g.dim, out[row * g.cols ..][0 .. it.ids.len * g.cols]);
                row += it.ids.len;
            }
            for (j.n..j.fill) |t| @memcpy(out[t * g.cols ..][0..g.cols], out[(j.n - 1) * g.cols ..][0..g.cols]);
            g.publish(j, k);
        }
    }

    fn publish(g: *Gate, j: *const Job, k: usize) void {
        @atomicStore(i64, &g.mem.ctl[ctl_ready + k], @intCast(j.seq), .release);
    }

    // -- the locked source -----------------------------------------------------------------------------------------

    /// The table under the gate's lock: every other Engram read of the forward while the gate lives.
    pub fn source(g: *Gate) eh.Source {
        return .{ .ptr = g, .vtable = &.{ .issue = issueFn, .rows = rowsFn } };
    }

    fn issueFn(p: *anyopaque, li: usize, idx: []const i64) anyerror!void {
        const g: *Gate = @ptrCast(@alignCast(p));
        _ = std.c.pthread_mutex_lock(&g.lock);
        defer _ = std.c.pthread_mutex_unlock(&g.lock);
        return g.src.vtable.issue(g.src.ptr, li, idx);
    }

    fn rowsFn(p: *anyopaque, li: usize, idx: []const i64, dim: usize, out: []u16) anyerror!void {
        const g: *Gate = @ptrCast(@alignCast(p));
        _ = std.c.pthread_mutex_lock(&g.lock);
        defer _ = std.c.pthread_mutex_unlock(&g.lock);
        return g.src.vtable.rows(g.src.ptr, li, idx, dim, out);
    }
};

// -- host test: the worker's staging against the old path's rows ------------------------------------------------------

const testing = std.testing;

/// A table whose row r holds (r * 31 + j) & 0xffff at column j; `fail`: every read fails.
const FakeRows = struct {
    fail: bool = false,
    reads: std.atomic.Value(u32) = .init(0),
    fn source(x: *FakeRows) eh.Source {
        return .{ .ptr = x, .vtable = &.{ .issue = issue, .rows = rows } };
    }
    fn issue(_: *anyopaque, _: usize, _: []const i64) anyerror!void {}
    fn rows(p: *anyopaque, li: usize, idx: []const i64, dim: usize, out: []u16) anyerror!void {
        const x: *FakeRows = @ptrCast(@alignCast(p));
        _ = x.reads.fetchAdd(1, .monotonic);
        if (x.fail) return error.EngramReadFailed;
        for (idx, 0..) |r, i| for (0..dim) |j| {
            out[i * dim + j] = @truncate(@as(u64, @bitCast(r)) *% 31 +% j +% li * 7);
        };
    }
};

fn fixtureHost(a: std.mem.Allocator, ids_buf: []u32) !struct { h: eh.Host, bytes: []align(4) u8 } {
    const fx: []const u8 = @embedFile("fixtures/engram-hash-ref.bin");
    const L = std.mem.readInt(u32, fx[8..12], .little);
    const M = std.mem.readInt(u32, fx[12..16], .little);
    const V = std.mem.readInt(u32, fx[16..20], .little);
    const head = 24 + 4 * L + 8 * L * M;
    const bytes = try a.alignedAlloc(u8, .@"4", head + 4 * @as(usize, V));
    @memcpy(bytes[0..head], fx[0..head]);
    @memset(bytes[head..], 0); // every token maps to 0: the hashes still differ by position and layer
    return .{ .h = try eh.Host.loadLegacyFixture(bytes, 8, 2, 16_000_000, ids_buf), .bytes = bytes };
}

test "engram gate: the worker stages the old path's rows (row mode padding included), publishes in order, and a failed read still raises every flag" {
    const a = std.heap.c_allocator;
    var ids_buf: [eh.max_layers]u32 = undefined;
    const fh = try fixtureHost(a, &ids_buf);
    defer a.free(fh.bytes);
    const h = &fh.h;
    const dim: usize = 32;
    const layers = h.layer_ids;
    var fake: FakeRows = .{};
    const s: Settings = .{ .on = true, .rows = 16 };
    const cols = h.headShard(0, 2)[1] * dim;
    const ctl = try a.alloc(i64, ctl_n);
    defer a.free(ctl);
    const staging = try a.alloc(u16, 2 * layers.len * s.rows * cols);
    defer a.free(staging);
    const g = try Gate.init(a, h, fake.source(), layers, 0, 2, dim, s, .{ .ctl = ctl, .staging = staging });
    defer g.close();
    // two slots' windows (row mode) in a bucket of 8 rows
    const ids_a = [_]u32{ 5, 9, 13 };
    const tail_a = [_]u32{ 1, 2 };
    const ids_b = [_]u32{ 7, 7 };
    const tail_b = [_]u32{3};
    const items = [_]Item{ .{ .ids = &ids_a, .tail = &tail_a }, .{ .ids = &ids_b, .tail = &tail_b } };
    for (0..3) |round| {
        try g.arm(&items, 8);
        try g.check();
        try testing.expectEqual(@as(i64, @intCast(round + 1)), ctl[ctl_want]);
        for (layers, 0..) |L, k| {
            try testing.expectEqual(ctl[ctl_want], ctl[ctl_ready + k]);
            const got = g.stagingOf(g.seq, k);
            const want = try a.alloc(u16, 8 * cols);
            defer a.free(want);
            try h.windowRows(a, fake.source(), &ids_a, &tail_a, L, 0, 2, dim, want[0 .. 3 * cols]);
            try h.windowRows(a, fake.source(), &ids_b, &tail_b, L, 0, 2, dim, want[3 * cols .. 5 * cols]);
            for (5..8) |t| @memcpy(want[t * cols ..][0..cols], want[4 * cols ..][0..cols]);
            try testing.expectEqualSlices(u16, want, got[0 .. 8 * cols]);
        }
    }
    g.warm(&items); // reads only: nothing published
    fake.fail = true;
    try g.arm(&items, 8);
    try testing.expectError(error.EngramReadFailed, g.check());
    for (0..layers.len) |k| try testing.expectEqual(@as(i64, 4), ctl[ctl_ready + k]); // up, so the GPU never hangs
    // the failed round is reported again at the next arm, which arms nothing; the GPU's timeout flag raises once
    fake.fail = false;
    try testing.expectError(error.EngramReadFailed, g.arm(&items, 8));
    ctl[ctl_err] = ctl_ready;
    try g.arm(&items, 8);
    try testing.expectError(error.EngramGateTimeout, g.check());
    try g.check();
    try testing.expect(!g.eligible(17) and g.eligible(16));
}
