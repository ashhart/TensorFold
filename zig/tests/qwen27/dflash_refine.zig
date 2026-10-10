//! Probe: the drafter's top-1 at each block position along a greedy reply, masked versus k true tokens given.
const std = @import("std");
const mtl = @import("metal");
const q = @import("tensorfold").qwen27;
const df = q.dflash;
const help = "tf-qwen27-dflash-refine --model TARGET --draft DRAFTER --tokens IDS.json [--max-tokens 160]";

fn ids(a: std.mem.Allocator, io: std.Io, path: []const u8) ![]const u32 {
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(16 << 20));
    return (try std.json.parseFromSlice([]u32, a, text, .{})).value;
}

/// Each block position's best candidate by the drafter's own score.
fn best(l: df.operators.Lattice, out: []u32) void {
    for (out[0..l.depth], 0..) |*t, d| {
        var top: usize = 0;
        for (0..l.topk) |i| if (l.unary[d * l.topk + i] > l.unary[d * l.topk + top]) {
            top = i;
        };
        t.* = l.candidates[d * l.topk + top];
    }
}

pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    var target_dir: ?[]const u8 = null;
    var draft_dir: ?[]const u8 = null;
    var tokens_file: ?[]const u8 = null;
    var count: usize = 160;
    var i: usize = 1;
    while (i + 1 < args.len) : (i += 2) {
        if (std.mem.eql(u8, args[i], "--model")) target_dir = args[i + 1] else if (std.mem.eql(u8, args[i], "--draft")) draft_dir = args[i + 1] else if (std.mem.eql(u8, args[i], "--tokens")) tokens_file = args[i + 1] else if (std.mem.eql(u8, args[i], "--max-tokens")) count = try std.fmt.parseInt(usize, args[i + 1], 10) else return error.UnknownOption;
    }
    if (target_dir == null or draft_dir == null or tokens_file == null) {
        try std.Io.File.stdout().writeStreamingAll(io, help ++ "\n");
        return error.BadOptions;
    }
    const prompt = try ids(a, io, tokens_file.?);
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const target = try q.model.Model.load(init.gpa, io, target_dir.?, 128);
    defer target.deinit();
    var runner = try q.decode_round.Runner.init(init.gpa, target, 1, @intCast(prompt.len + count + 64));
    defer runner.deinit();
    const draft = try df.runtime_model.Model.load(init.gpa, io, target, draft_dir.?, .prepared_q4_reference);
    defer draft.deinit();
    try draft.setTreeBlock(16);
    var g = try df.generation.Generation.init(init.gpa, &runner, draft);
    defer g.deinit();
    // the true greedy reply
    try g.prefill(prompt, 128);
    const truth = try g.run(count, true, 15);
    defer init.gpa.free(truth.tokens);
    try runner.reset(0);
    try draft.reset();
    try g.prefill(prompt, 128);
    const block = 16;
    const ks = [_]usize{ 0, 1, 2, 4, 8 };
    var hits: [ks.len][block]u64 = @splat(@splat(0));
    var seen: [ks.len][block]u64 = @splat(@splat(0));
    var chain: [ks.len]u64 = @splat(0); // landed run length after the given tokens, summed
    const tops = [_]usize{ 1, 2, 4, 8, 16 };
    var covered: [tops.len][block]u64 = @splat(@splat(0)); // the true token among the drafter's top-k (all masked)
    var guess: [block]u32 = undefined;
    var t: usize = 0;
    while (t + block < truth.tokens.len) : (t += 1) {
        const anchor = truth.tokens[t];
        try draft.flush();
        for (ks, 0..) |k, ki| {
            var lattice = if (k == 0) try df.execution.forwardBlock(draft.backend.ops(), &draft.graph, draft.preparation, draft.session.context, anchor, block) else try df.execution.forwardGiven(draft.backend.ops(), &draft.graph, draft.preparation, draft.session.context, anchor, truth.tokens[t + 1 ..][0..k], block);
            defer lattice.deinit();
            best(lattice, &guess);
            if (k == 0) for (0..lattice.depth) |d| {
                var rank: usize = lattice.topk;
                const truth_token = truth.tokens[t + 1 + d];
                for (0..lattice.topk) |c| if (lattice.candidates[d * lattice.topk + c] == truth_token) {
                    var above: usize = 0;
                    for (0..lattice.topk) |o| above += @intFromBool(lattice.unary[d * lattice.topk + o] > lattice.unary[d * lattice.topk + c]);
                    rank = above;
                };
                for (tops, 0..) |top, ti| covered[ti][d] += @intFromBool(rank < top);
            };
            var run = true;
            for (k..lattice.depth) |d| {
                const ok = guess[d] == truth.tokens[t + 1 + d];
                seen[ki][d] += 1;
                hits[ki][d] += @intFromBool(ok);
                run = run and ok;
                chain[ki] += @intFromBool(run);
            }
        }
        try g.advance(anchor);
    }
    const n: f64 = @floatFromInt(@max(seen[0][0], 1));
    for (tops, 0..) |top, ti| {
        std.debug.print("top{d} covers:", .{top});
        for (0..block - 1) |d| std.debug.print(" p{d}={d:.2}", .{ d + 1, @as(f64, @floatFromInt(covered[ti][d])) / n });
        std.debug.print("\n", .{});
    }
    for (ks, 0..) |k, ki| {
        std.debug.print("given={d} landed_after_given={d:.2} top1:", .{ k, @as(f64, @floatFromInt(chain[ki])) / n });
        for (k..block - 1) |d| std.debug.print(" p{d}={d:.2}", .{ d + 1, @as(f64, @floatFromInt(hits[ki][d])) / @as(f64, @floatFromInt(@max(seen[ki][d], 1))) });
        std.debug.print("\n", .{});
    }
}
