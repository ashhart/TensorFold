//! Several Nemotron streams on Metal: shared forwards' GPU ms by streams x window rows, and the lane core's throughput.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const probe = @import("wide_probe.zig");

const nm = tf.nemotron;
const st = nm.state;
const fwd = nm.forward;
const lanes = tf.lanes;
const Probe = probe.Probe;

/// One forward over `n` streams' windows of `rows` rows each (their own continuations), never committed: GPU ms.
fn shared(b: *nm.backend.Metal, ps: []Probe, next: []const []const u32, rows: usize) !f64 {
    var segs: [st.max_rows]fwd.Seg = undefined;
    const ids = b.scratch.ids.slice(u32, ps.len * rows);
    for (ps, 0..) |*p, i| {
        for (0..rows) |r| ids[i * rows + r] = next[i][r];
        p.c.start = i * rows;
        segs[i] = .{ .rows = rows, .cache = p.c, .store = .lag, .slot = try b.slotFor(p.c, .lag) };
    }
    defer for (segs[0..ps.len]) |g| if (g.slot >= 0) b.pool.give(@intCast(g.slot));
    const Job = struct {
        segs: []const fwd.Seg,
        pub fn encode(j: @This(), q: *Probe, e: *fwd.Enc) !void {
            const f = q.b.forward();
            f.body(e, j.segs, q.b.scratch.ids, 0);
            f.draw(e, j.segs, .all, q.out, 0);
        }
    };
    return ps[0].run(&.{}, Job{ .segs = segs[0..ps.len] });
}

/// GPU ms of shared forwards: streams 1..32 (each its own prompt, `at` tokens in) x window rows, total rows <= 64.
pub fn costs(gpa: std.mem.Allocator, arena: std.mem.Allocator, io: std.Io, model: []const u8, path: []const u8, at: usize, reps: usize) !void {
    const prompts = try probe.readPrompts(arena, path);
    const n = @min(prompts.map.count(), 32);
    var longest: usize = 0;
    for (prompts.map.values()) |v| longest = @max(longest, v.len);
    const m = try nm.Model.load(gpa, io, model, true);
    defer m.deinit();
    const b = try nm.backend.Metal.init(gpa, m, .{ .chunk = 16, .capacity = longest + at + 64, .drafts = true, .streams = n, .batch_rows = 64 });
    defer b.deinit();
    const ps = try arena.alloc(Probe, n);
    const next = try arena.alloc([]u32, n);
    for (ps, next, prompts.map.values()[0..n]) |*p, *nx, ids| {
        p.* = try Probe.init(b);
        try p.prefill(ids);
        var t = try p.first();
        for (0..at) |_| t = try p.step(t);
        nx.* = try arena.alloc(u32, 8);
        // the next tokens of its greedy continuation: decoded on a scratch copy would move the cache, so repeat t
        var q = try Probe.init(b);
        defer q.deinit();
        try q.prefill(ids);
        var u = try q.first();
        for (0..at) |_| u = try q.step(u);
        for (nx.*) |*x| {
            x.* = u;
            u = try q.step(u);
        }
    }
    defer for (ps) |*p| p.deinit();
    std.debug.print("shared forwards, GPU ms (median of {d}), each stream {d} tokens into its own prompt:\n", .{ reps, at });
    for ([_]usize{ 1, 2, 4, 8, 16, 32 }) |streams| {
        if (streams > n) break;
        std.debug.print("  {d:>2} streams:", .{streams});
        for ([_]usize{ 1, 2, 3, 4 }) |rows| {
            if (streams * rows > 64) break;
            var ms: [32]f64 = undefined;
            _ = try shared(b, ps[0..streams], next[0..streams], rows);
            for (0..reps) |r| ms[r] = try shared(b, ps[0..streams], next[0..streams], rows);
            std.debug.print(" x{d} {d:.2}", .{ rows, probe.median(ms[0..reps]) });
        }
        std.debug.print("\n", .{});
    }
}

/// `n` streams of different prompts through the lane core at once (greedy, drafts on): aggregate decode tok/s.
pub fn serve(gpa: std.mem.Allocator, arena: std.mem.Allocator, io: std.Io, model: []const u8, path: []const u8, counts: []const usize, max_new: u32) !void {
    const prompts = try probe.readPrompts(arena, path);
    var longest: usize = 0;
    for (prompts.map.values()) |v| longest = @max(longest, v.len);
    var most: usize = 1;
    for (counts) |c| most = @max(most, c);
    const m = try nm.Model.load(gpa, io, model, true);
    defer m.deinit();
    const b = try nm.backend.Metal.init(gpa, m, .{ .chunk = 16, .capacity = longest + max_new + 64, .drafts = true, .streams = most, .batch_rows = 64 });
    defer b.deinit();
    try nm.timing.measure(b);
    var cfg = try lanes.Config.init(gpa, b.facts(), nm.backend.max_window, nm.backend.max_window - 1);
    defer cfg.deinit(gpa);
    const eos = m.config.eos[0..m.config.eos_count];
    for (counts) |count| {
        if (count > prompts.map.count()) break;
        var clock = lanes.backend.WallClock{ .io = io };
        var engine = lanes.Engine.init(gpa, &cfg, b.backend(), clock.clock());
        defer engine.deinit();
        const streams = try gpa.alloc(lanes.Stream, count);
        defer gpa.free(streams);
        for (streams, prompts.map.keys()[0..count], prompts.map.values()[0..count]) |*s, name, ids| {
            s.* = try lanes.Stream.init(gpa, .{ .id = name, .prompt = ids, .max_new = max_new, .eos = eos, .drafts = true });
        }
        defer for (streams) |*s| s.deinit(gpa);
        b.timing = .{};
        for (streams) |*s| try engine.addStream(s);
        const t0 = mtl.clock.seconds();
        var steps: usize = 0;
        while (engine.activeCount() > 0) : (steps += 1) try engine.step();
        const dt = mtl.clock.seconds() - t0;
        try b.drain();
        var tokens: usize = 0;
        var rounds: u64 = 0;
        for (streams) |*s| {
            tokens += s.emitted().len - 1;
            rounds += s.rounds;
        }
        const t = b.timing;
        std.debug.print("{d:>2} streams: {d:.1} tok/s together ({d:.1} each), {d} steps, {d:.2} tokens a stream-round; GPU ms a verify {d:.2}, a draft batch {d:.2}\n", .{ count, @as(f64, @floatFromInt(tokens)) / dt, @as(f64, @floatFromInt(tokens)) / dt / @as(f64, @floatFromInt(count)), steps, @as(f64, @floatFromInt(tokens)) / @as(f64, @floatFromInt(@max(rounds, 1))), t.gpu(.verify), t.gpu(.draft) });
    }
}
