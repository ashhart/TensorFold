//! The 27B on the shared lane core: drafted, plain and oracle-tree rounds keep plain greedy's tokens and runner state.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const q = tf.qwen27;
const lanes = tf.lanes;
const lb = @import("qwen27_host").lanes_backend;
const target_graph = @import("synthetic_graph.zig");
const draft_graph = @import("dflash_graph.zig");

/// Trees from the reference reply: its next tokens, each level with a wrong sibling when `branch`.
const Oracle = struct {
    a: std.mem.Allocator,
    runner: *q.decode_round.Runner,
    tokens: []const u32,
    prompt: usize,
    branch: bool = false,
    fn propose(ptr: *anyopaque, pending: u32, limit: u32) !q.dflash.selector.Tree {
        const o: *Oracle = @ptrCast(@alignCast(ptr));
        const at = o.runner.offsets[0] - o.prompt;
        if (at >= o.tokens.len or pending != o.tokens[at]) return error.OraclePosition;
        const children: usize = if (o.branch) 2 else 1;
        const count = @min(@as(usize, limit) / children, o.tokens.len - at - 1);
        const ids = try o.a.alloc(u32, count * children);
        const parents = try o.a.alloc(i32, ids.len);
        const scores = try o.a.alloc(f64, ids.len);
        for (0..count) |depth| for (0..children) |child| {
            const i = depth * children + child;
            const token = o.tokens[at + 1 + depth];
            ids[i] = if (child + 1 == children) token else @intCast((token + 1) % o.runner.model.config.vocab);
            parents[i] = if (depth == 0) -1 else @intCast(depth * children - 1);
            scores[i] = 0;
        };
        return .{ .gpa = o.a, .tokens = ids, .parents = parents, .scores = scores };
    }
};

const Run = struct { tokens: []const u32, rounds: u64, drafted: u64, accepted: u64 };

fn run(a: std.mem.Allocator, io: std.Io, b: *lb.Metal, prompt: []const u32, count: u32, drafts: bool, sampling: ?lanes.Sampling) !Run {
    var cfg = try lanes.Config.init(a, b.facts(), lb.max_rows, lb.max_rows - 1);
    defer cfg.deinit(a);
    var clock = lanes.backend.WallClock{ .io = io };
    var engine = lanes.Engine.init(a, &cfg, b.backend(), clock.clock());
    defer engine.deinit();
    var s = try lanes.Stream.init(a, .{ .id = "q27", .prompt = prompt, .max_new = count, .drafts = drafts, .sampling = sampling });
    defer s.deinit(a);
    try engine.addStream(&s);
    while (engine.activeCount() > 0) try engine.step();
    return .{ .tokens = try a.dupe(u32, s.emitted()), .rounds = s.rounds, .drafted = s.drafted, .accepted = s.accepted };
}

/// The reply and the runner equal plain greedy's: the first `count` tokens, the state after the rows the core fed.
fn same(r: Run, runner: *q.decode_round.Runner, prompt: usize, expected: []const u32, states: []const [32]u8, count: usize) !void {
    if (!std.mem.eql(u32, expected[0..count], r.tokens)) return error.LaneTokensDiffer;
    const fed = runner.offsets[0] - prompt;
    if (fed + 1 < count or fed > count) return error.LaneFedRows;
    if (!std.mem.eql(u8, &states[fed + 1], &try (q.session.Session{ .runner = runner }).fingerprint())) return error.LaneStateDiffers;
}

pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const io = init.io;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const model = try target_graph.createLayers(a, 5);
    defer model.deinit();
    const drafter = try draft_graph.create(a, model);
    defer drafter.deinit();
    var runner = try q.decode_round.Runner.init(a, model, 1, 384);
    defer runner.deinit();
    const direct = q.session.Session{ .runner = &runner };
    var prompt: [130]u32 = undefined;
    for (&prompt, 0..) |*id, i| id.* = @intCast((i * 17 + 7) % 96);
    try direct.prefill(&prompt, 16);
    var expected: [34]u32 = undefined;
    var states: [35][32]u8 = undefined;
    states[0] = try direct.fingerprint();
    for (&expected, 0..) |*id, i| {
        id.* = try direct.greedy();
        states[i + 1] = try direct.fingerprint();
        if (i + 1 < expected.len) try direct.step(id.*);
    }
    const draft = try lb.Draft.attach(a, &runner, drafter, false);
    defer draft.deinit();
    const drafted = try lb.Metal.init(a, &runner, draft);
    defer drafted.deinit();
    const plain = try lb.Metal.init(a, &runner, null);
    defer plain.deinit();
    const counts = [_]u32{ 1, 2, 7, 16, 17, 33 };
    for (counts) |count| {
        try same(try run(a, io, drafted, &prompt, count, true, null), &runner, prompt.len, &expected, &states, count);
        try same(try run(a, io, drafted, &prompt, count, false, null), &runner, prompt.len, &expected, &states, count);
        try same(try run(a, io, drafted, &prompt, count, true, .{ .seed = 7, .temperature = 0 }), &runner, prompt.len, &expected, &states, count);
        try same(try run(a, io, plain, &prompt, count, true, null), &runner, prompt.len, &expected, &states, count);
    }
    std.debug.print("lane core: drafted, plain, sampled-at-0 and no-drafter streams at 1/2/7/16/17/33 equal plain greedy tokens and state\n", .{});
    var oracle = Oracle{ .a = a, .runner = &runner, .tokens = &expected, .prompt = prompt.len };
    draft.generation.proposer = .{ .ptr = &oracle, .call = Oracle.propose };
    defer draft.generation.proposer = null;
    for ([_]bool{ false, true }) |branch| {
        oracle.branch = branch;
        for (counts) |count| {
            const r = try run(a, io, drafted, &prompt, count, true, null);
            try same(r, &runner, prompt.len, &expected, &states, count);
            if (count == 33) std.debug.print("oracle branch={}: {d} rounds, {d}/{d} drafts landed\n", .{ branch, r.rounds, r.accepted, r.drafted });
            if (count == 33 and r.rounds > @as(u64, if (branch) 8 else 4)) return error.OracleRoundsTooMany;
        }
    }
    std.debug.print("lane core oracle trees: chains and non-prefix branch paths equal plain greedy tokens and state\n", .{});
}
