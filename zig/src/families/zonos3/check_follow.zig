//! zonos3-generate --follow-check: the follower fed by the port's readings against the reference's weights and frames.
const std = @import("std");
const z3 = @import("zonos3");
const gen = @import("generate.zig");
const laci_checks = @import("check_laci.zig");
const steer_checks = @import("check_steering.zig");

const cfg = z3.config;
const W = cfg.frame_width;
const D = cfg.dim;
const sf = z3.span_follow;

/// The plan's deltas with the follower's weights: the run's steering callback.
const Steering = struct {
    plan: *const z3.steering.Plan,
    follower: *sf.Follower,
    weights: []f32,
    out: []f32,
    sum: []f32,
    steer: []z3.Steer,

    fn at(ctx: *anyopaque, frame: usize) anyerror![]const z3.Steer {
        const s: *Steering = @ptrCast(@alignCast(ctx));
        try s.follower.weights(frame, s.weights);
        return if (s.plan.weighed(frame, s.out, s.sum, s.weights)) s.steer else &.{};
    }
};

fn number(v: std.json.Value) f64 {
    return if (v == .float) v.float else @floatFromInt(v.integer);
}

/// GEN_DIR's request with its follower: `heads_path` is the language's heads file, `bundle_path` the steering bundle.
pub fn run(gpa: std.mem.Allocator, io: std.Io, e: *z3.Engine, dir: []const u8, prompt: []const [W]i32, replaced: []const z3.Replace, heads_path: []const u8, bundle_path: []const u8, max_frames: usize) !u8 {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const cwd = std.Io.Dir.cwd();
    const read = struct {
        fn json(al: std.mem.Allocator, i: std.Io, d: []const u8, name: []const u8) !std.json.ObjectMap {
            const text = try std.Io.Dir.cwd().readFileAlloc(i, try std.fs.path.join(al, &.{ d, name }), al, .limited(1 << 26));
            return (try std.json.parseFromSliceLeaky(std.json.Value, al, text, .{})).object;
        }
    };
    const follow = try read.json(a, io, dir, "follow.json");
    const side = try read.json(a, io, dir, "steer.json");
    // the request's steering resolved here, its span deltas against the reference's
    var b = try z3.steer_bundle.load(gpa, io, bundle_path);
    defer b.deinit();
    var base: [cfg.speaker_dim]f32 = undefined;
    var message: []const u8 = "";
    const got = z3.steer_bundle.resolve(a, &b, try steer_checks.dumped(a, side, &base), &message) catch |err| {
        std.debug.print("RESULT zonos3 follow FAIL {s}: {s}\n", .{ @errorName(err), message });
        return 1;
    };
    const plan = got.plan;
    const deltas = try (try z3.npy.parse(try cwd.readFileAlloc(io, try std.fs.path.join(a, &.{ dir, "follow_deltas.npy" }), a, .limited(1 << 26)))).floats(a);
    const width = plan.sources.len * D;
    var deltas_differ: usize = @intFromBool(deltas.len != plan.spans.len * width);
    if (deltas_differ == 0) for (plan.spans, 0..) |s, k| {
        for (s.delta, deltas[k * width ..][0..width]) |x, y| deltas_differ += @intFromBool(@as(u32, @bitCast(x)) != @as(u32, @bitCast(y)));
    };
    const tokens = follow.get("tokens").?.array.items;
    const ranges = try a.alloc([2]u32, tokens.len);
    for (ranges, tokens) |*r, v| r.* = .{ @intCast(v.array.items[0].integer), @intCast(v.array.items[1].integer) };
    const gate = follow.get("gate").?.array.items;
    var follower = try sf.Follower.init(gpa, ranges, .{ .lead = number(gate[0]), .lag = number(gate[1]) });
    defer follower.deinit();
    const heads = try z3.laci.Heads.parse(a, try cwd.readFileAlloc(io, heads_path, a, .limited(1 << 20)));
    var steering: Steering = .{ .plan = &plan, .follower = &follower, .weights = try a.alloc(f32, ranges.len), .out = try a.alloc(f32, width), .sum = try a.alloc(f32, width), .steer = try a.alloc(z3.Steer, plan.sources.len) };
    for (steering.steer, plan.sources, 0..) |*s, source, j| s.* = .{ .source = source, .delta = steering.out[j * D ..][0..D] };
    const t0 = gen.nowNs(io);
    var rows_ok = true;
    var frames: usize = 0;
    const laci_dump = try std.fs.path.join(a, &.{ dir, "laci.json" });
    if (cwd.access(io, laci_dump, .{})) |_| {
        // under LACI the session's readings feed the follower; the LACI check compares rows, readings and record
        const hook: laci_checks.Hook = .{ .steer = .{ .ctx = &steering, .at = Steering.at, .twin = false }, .follower = &follower };
        rows_ok = try laci_checks.engine(gpa, io, e, dir, prompt, replaced, heads_path, "", max_frames, hook) == 0;
    } else |_| {
        frames = try alone(gpa, io, e, dir, prompt, replaced, heads, &steering, max_frames, side.get("twin_steered").?.bool, &rows_ok);
    }
    const seconds = @as(f64, @floatFromInt(gen.nowNs(io) - t0)) / 1e9;
    // the reference's readings after its rollbacks, against the follower's
    var theirs: std.ArrayList([2]usize) = .empty;
    const observed = follow.get("observed").?.array.items;
    const truncated = follow.get("truncated").?.array.items;
    var cut: usize = 0;
    for (observed, 0..) |v, i| {
        while (cut < truncated.len and truncated[cut].array.items[0].integer == i) : (cut += 1) {
            const from: usize = @intCast(truncated[cut].array.items[1].integer);
            while (theirs.items.len > 0 and theirs.items[theirs.items.len - 1][0] >= from) _ = theirs.pop();
        }
        try theirs.append(a, .{ @intCast(v.array.items[0].integer), @intCast(v.array.items[1].integer) });
    }
    var readings_differ: usize = @intFromBool(theirs.items.len != follower.frames.items.len);
    if (readings_differ == 0) for (theirs.items, follower.frames.items, follower.pos.items) |t, f, p| {
        readings_differ += @intFromBool(t[0] != f or t[1] != p);
    };
    // every frame the follower decided, its weights against the reference's
    const decided = follow.get("weights").?.object;
    var weights_differ: usize = 0;
    var on: usize = 0;
    for (follower.decided.items, 0..) |frame, i| {
        var key: [24]u8 = undefined;
        const want = decided.get(try std.fmt.bufPrint(&key, "{d}", .{frame})) orelse {
            weights_differ += 1;
            continue;
        };
        const mine = follower.kept.items[i * ranges.len ..][0..ranges.len];
        var any = false;
        for (mine, want.array.items) |x, y| {
            weights_differ += @intFromBool(x != number(y));
            any = any or x > 0;
        }
        on += @intFromBool(any);
    }
    const pass = rows_ok and deltas_differ == 0 and readings_differ == 0 and weights_differ == 0 and follower.decided.items.len == decided.count();
    std.debug.print("RESULT zonos3 follow {s} spans={d} tokens={any} readings={d} readings_differ={d} frames_decided={d} reference_decided={d} frames_on={d} weights_differ={d} deltas_differ={d} rows_identical={} rows={d} seconds={d:.3}\n", .{ if (pass) "PASS" else "FAIL", ranges.len, ranges, follower.frames.items.len, readings_differ, follower.decided.items.len, decided.count(), on, weights_differ, deltas_differ, rows_ok, frames, seconds });
    return if (pass) 0 else 1;
}

/// Without LACI: the request's own passes, the main lane's queries read after each, a twin on lane 1 when guided.
fn alone(gpa: std.mem.Allocator, io: std.Io, e: *z3.Engine, dir: []const u8, prompt: []const [W]i32, replaced: []const z3.Replace, heads: z3.laci.Heads, steering: *Steering, max_frames: usize, twin_steered: bool, ok: *bool) !usize {
    var twin: [][W]i32 = &.{};
    defer gpa.free(twin);
    var twin_replaced: std.ArrayList(z3.Replace) = .empty;
    defer twin_replaced.deinit(gpa);
    var scale: f32 = 1;
    const twin_path = try std.fs.path.join(gpa, &.{ dir, "twin_ids.npy" });
    defer gpa.free(twin_path);
    if (std.Io.Dir.cwd().access(io, twin_path, .{})) |_| {
        scale = try gen.loadBranch(gpa, io, e, dir, "twin_ids", &twin, &twin_replaced);
        try e.twin();
        e.swap();
        try e.reset();
        e.swap();
    } else |_| {}
    var bos: usize = 0;
    while (prompt[bos][cfg.codebooks] != z3.prompt.bos) bos += 1;
    var eos: usize = prompt.len - 1;
    while (prompt[eos][cfg.codebooks] != z3.prompt.eos) eos -= 1;
    var g = try z3.generation.Run.begin(gpa, e, prompt, replaced, twin, twin_replaced.items, scale, .{ .temperature = 0 }, max_frames);
    defer g.deinit();
    var reader = try z3.span_follow.Reader.init(gpa, e, heads, bos, eos - bos + 1);
    defer reader.deinit();
    while (true) {
        // the last pass's reading is taken on every step, the reply's last included, as the reference observes it
        const finished = try g.sample();
        const frame = g.rows.items.len;
        if (frame >= 2) try steering.follower.observe(frame - 1, reader.position());
        if (finished) break;
        const steer = try Steering.at(steering, frame);
        var turns: [2]z3.Turn = undefined;
        turns[0] = .{ .lane = 0, .row = g.lastRow(), .out = @ptrCast(g.logits), .steer = steer, .probe = &reader.probed };
        var n: usize = 1;
        if (g.unconditional) |u| {
            turns[1] = .{ .lane = 1, .row = g.lastRow(), .out = u, .steer = if (twin_steered) steer else &.{} };
            n = 2;
        }
        try e.runLanes(turns[0..n]);
    }
    const r = try g.result();
    defer gpa.free(r.rows);
    const path = try std.fs.path.join(gpa, &.{ dir, "rows.npy" });
    defer gpa.free(path);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 28));
    defer gpa.free(bytes);
    const arr = try z3.npy.parse(bytes);
    const want = try arr.ints(gpa);
    defer gpa.free(want);
    ok.* = arr.shape[0] == r.rows.len and std.mem.eql(i32, @as([*]const i32, @ptrCast(r.rows.ptr))[0 .. r.rows.len * W], want);
    return r.rows.len;
}
