//! GPU time by kind of command buffer, and the forward costs the depth rule and shared rounds plan with.
const std = @import("std");
const mtl = @import("metal");
const lanes = @import("lanes");
const fwd = @import("forward.zig");
const st = @import("state.zig");
const backend = @import("backend.zig");

pub const Kind = enum(u2) { prefill, step, verify, draft };

pub const Timing = struct {
    ms: [4]f64 = @splat(0),
    count: [4]usize = @splat(0),
    encode_ms: [4]f64 = @splat(0), // host time encoding each kind's command buffers
    dispatches: usize = 0,

    pub fn gpu(t: Timing, k: Kind) f64 {
        const n = t.count[@backingInt(k)];
        return if (n > 0) t.ms[@backingInt(k)] / @as(f64, @floatFromInt(n)) else 0;
    }

    pub fn encode(t: Timing, k: Kind) f64 {
        const n = t.count[@backingInt(k)];
        return if (n > 0) t.encode_ms[@backingInt(k)] / @as(f64, @floatFromInt(n)) else 0;
    }
};

pub const Costs = struct {
    window: [32]lanes.config.Cost = undefined, // a lone stream's window ms by rows (1-16, then wider to a 32-lane tree)
    windows: usize = 0,
    shared: [8]lanes.config.Cost = undefined, // a shared round's ms by total rows (2-row windows)
    shareds: usize = 0,
    head_ms: f64 = 0, // one chained head step
};

/// Rows of KV a timed forward attends to: a reply's early tokens, as rounds meet them.
const timed_len = 192;

/// The median GPU ms of `reps` forwards over `streams` windows of `rows` rows each.
fn timeWindows(b: *backend.Metal, caches: []st.Cache, rows: usize, reps: usize) !f64 {
    const ids = caches[0].windows[0].slice(u32, st.max_rows);
    for (ids, 0..) |*t, i| t.* = @intCast(1000 + 37 * i);
    var ms: [16]f64 = undefined;
    for (ms[0..reps]) |*m| {
        var segs: [st.max_rows]fwd.Seg = undefined;
        for (caches, 0..) |*c, i| {
            c.len = timed_len;
            c.start = i * rows;
            segs[i] = .{ .rows = rows, .cache = c };
        }
        const Job = struct {
            segs: []const fwd.Seg,
            pub fn encode(j: @This(), m2: *backend.Metal, e: *fwd.Enc) !void {
                const f = m2.forward();
                const c = j.segs[0].cache;
                f.body(e, j.segs, c.windows[0], 0);
                f.draw(e, j.segs, .all, m2.tokens, 0);
            }
        };
        try b.submit(.verify, 0, Job{ .segs = segs[0..caches.len] });
        try b.drain();
        m.* = b.last_ms;
    }
    return median(ms[0..reps]);
}

fn median(v: []f64) f64 {
    std.mem.sort(f64, v, {}, std.sort.asc(f64));
    return if (v.len % 2 == 1) v[v.len / 2] else (v[v.len / 2 - 1] + v[v.len / 2]) / 2;
}

/// GPU-timed fastest runs after a warm-up: windows of 1-33 rows, shared 2-row rounds, a head step; kept per build.
pub fn measure(b: *backend.Metal, io: std.Io) !void {
    const c = b.m.config;
    var shape: [160]u8 = undefined;
    const parts = [_][]const u8{ "nemotron-metal", std.mem.span(b.m.device.name()), try std.fmt.bufPrint(&shape, "{d}/{d}/{d}/{d}/{d}/{d}", .{ c.layers, c.hidden, c.vocab, c.experts, b.o.batch_rows, @intFromBool(b.head != null) }) };
    const k = lanes.cost_cache.key(b.gpa, io, &parts) catch null;
    if (k) |key| if (lanes.cost_cache.load(Costs, b.gpa, io, key)) |kept| {
        b.costs = kept;
        b.timing = .{};
        return;
    };
    var caches: [16]st.Cache = undefined;
    const n = @min(b.o.batch_rows / 2, caches.len);
    for (caches[0..n]) |*x| x.* = try st.Cache.init(b.m.device, c, timed_len + st.max_rows, b.head != null);
    defer for (caches[0..n]) |*x| x.deinit(&b.pool);
    // the GPU's clocks ramp up under load: time nothing until it has run for a while
    _ = try timeWindows(b, caches[0..1], backend.max_window, 12);
    const windows: WindowTimer = .{ .b = b, .caches = caches[0..1] };
    const heads: HeadTimer = .{ .b = b, .cache = &caches[0] };
    var ms: [wide.len + backend.max_window]f64 = undefined;
    var head = [1]f64{0};
    const head_ms: ?*f64 = if (b.head != null) &head[0] else null;
    try lanes.cost_rule.measure(windows, heads, &ms, head_ms, false);
    try lanes.cost_rule.smooth(windows, &ms);
    const ref = lanes.cost_cache.referenceKey(&parts);
    if (lanes.cost_cache.load(Reference, b.gpa, io, ref)) |r| {
        if (lanes.cost_rule.drifted(&ms, &r.window) or (b.head != null and lanes.cost_rule.drifted(&head, &r.head))) {
            try lanes.cost_rule.measure(windows, heads, &ms, head_ms, true);
        }
    }
    var costs = Costs{ .windows = ms.len, .head_ms = head[0] };
    for (ms, 0..) |m, i| costs.window[i] = .{ .width = @intCast(widthOf(i)), .ms = m };
    var streams: usize = 2;
    while (streams <= n) : (streams *= 2) {
        costs.shared[costs.shareds] = .{ .width = @intCast(2 * streams), .ms = try lanes.cost_rule.fastest(SharedTimer{ .b = b, .caches = caches[0..streams] }, 0) };
        costs.shareds += 1;
    }
    b.costs = costs;
    b.timing = .{};
    if (k) |key| lanes.cost_cache.save(Costs, b.gpa, io, key, costs);
    lanes.cost_cache.save(Reference, b.gpa, io, ref, .{ .window = ms, .head = head });
}

/// Windows past 16 rows, timed for the lanes' wider trees.
const wide = [_]usize{ 20, 24, 28, 31, 33 };

/// Window entry `i`'s rows: 1 to 16, then the wide ones.
fn widthOf(i: usize) usize {
    return if (i < backend.max_window) i + 1 else wide[i - backend.max_window];
}

/// The last table any build measured on this chip and model shape: windows and the head step.
const Reference = struct { window: [wide.len + backend.max_window]f64, head: [1]f64 };

const WindowTimer = struct {
    b: *backend.Metal,
    caches: []st.Cache,
    pub fn time(t: WindowTimer, i: usize) !f64 {
        return timeWindows(t.b, t.caches, widthOf(i), 1);
    }
};

const SharedTimer = struct {
    b: *backend.Metal,
    caches: []st.Cache,
    pub fn time(t: SharedTimer, _: usize) !f64 {
        return timeWindows(t.b, t.caches, 2, 1);
    }
};

/// One chained head step's GPU ms.
const HeadTimer = struct {
    b: *backend.Metal,
    cache: *st.Cache,
    const Chain = struct {
        cache: *st.Cache,
        pub fn encode(j: @This(), m: *backend.Metal, e: *fwd.Enc) !void {
            m.head.?.chain(e, j.cache, 1, m.scratch.x, 0, j.cache.windows[0], 0, .greedy, j.cache.windows[1], 4);
        }
    };
    pub fn time(t: HeadTimer, _: usize) !f64 {
        t.cache.mtp_len = timed_len;
        try t.b.submit(.draft, 0, Chain{ .cache = t.cache });
        try t.b.drain();
        return t.b.last_ms;
    }
};

/// The round loop's clock: a round's GPU ms (its verify, the draft before it) plus recent rounds' median host gap.
pub const RoundClock = struct {
    b: *backend.Metal,
    wall: lanes.backend.WallClock,
    gpu0: f64 = 0,
    gaps: [15]f64 = @splat(0), // recent rounds' wall ms past their GPU ms
    seen: usize = 0,

    pub fn clock(self: *RoundClock) lanes.backend.Clock {
        return .{ .ptr = self, .vtable = &.{ .start = start, .elapsed_ms = elapsed } };
    }

    fn start(ptr: *anyopaque) void {
        const self: *RoundClock = @ptrCast(@alignCast(ptr));
        self.wall.clock().start();
        self.gpu0 = self.b.gpu_ms;
    }

    fn elapsed(ptr: *anyopaque, mark: lanes.backend.Mark) f64 {
        const self: *RoundClock = @ptrCast(@alignCast(ptr));
        const wall = self.wall.clock().elapsedMs(mark);
        if (mark == .overhead) return wall;
        const gpu = self.b.gpu_ms - self.gpu0;
        self.gaps[self.seen % self.gaps.len] = @max(wall - gpu, 0);
        self.seen += 1;
        var recent = self.gaps;
        return gpu + median(recent[0..@min(self.seen, recent.len)]);
    }
};

/// Decode `steps` one-row steps of `cache`'s stream with every dispatch alone in a command buffer: GPU ms by kernel.
pub fn profile(b: *backend.Metal, cache: *st.Cache, steps: usize) !void {
    try b.drain();
    var prof = fwd.Profiler{ .queue = b.m.queue, .kernels = &b.m.kernels };
    var in: u64 = b.next - 1;
    for (0..steps) |_| {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const out = b.take(1);
        const state = try b.slotFor(cache, .full);
        const segs = [_]fwd.Seg{.{ .rows = 1, .cache = cache, .store = .full, .slot = state }};
        var e = fwd.Enc{ .e = prof.begin(), .prof = &prof };
        const f = b.forward();
        f.body(&e, &segs, b.tokens, backend.Metal.slot(in));
        f.draw(&e, &segs, .all, b.tokens, backend.Metal.slot(out));
        e.e.end();
        prof.cb.?.commit();
        prof.cb.?.wait();
        backend.Metal.hold(cache, 1, .full, state);
        cache.commit(&b.pool, 1);
        in = out;
    }
    b.landed = b.next;
    prof.report(steps);
    const head = &(b.head orelse return);
    var hp = fwd.Profiler{ .queue = b.m.queue, .kernels = &b.m.kernels };
    for (0..steps) |_| {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        var e = fwd.Enc{ .e = hp.begin(), .prof = &hp };
        head.chain(&e, cache, 1, b.scratch.x, 0, cache.windows[0], 0, .greedy, cache.windows[1], 4);
        e.e.end();
        hp.cb.?.commit();
        hp.cb.?.wait();
    }
    std.debug.print("the head's one-row step:\n", .{});
    hp.report(steps);
}
