//! Synthetic hybrid checks use one generated graph for CLI and served arithmetic.
const std = @import("std");
const mtl = @import("metal");
const q = @import("tensorfold").qwen27;
const graph = @import("synthetic_graph.zig");
pub fn main(init: std.process.Init) !void {
    var arena = std.heap.ArenaAllocator.init(init.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    if (args.len == 2 and (std.mem.eql(u8, args[1], "forward-diff") or std.mem.eql(u8, args[1], "forward-diff-red"))) {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        return @import("forward_diff.zig").run(a, std.mem.eql(u8, args[1], "forward-diff-red"));
    }
    if (args.len > 3) return error.Usage;
    const prompt_length = if (args.len >= 2) try std.fmt.parseInt(usize, args[1], 10) else 130;
    if (prompt_length == 0 or prompt_length > 640) return error.BadPromptLength;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const layer_count = if (args.len == 3) try std.fmt.parseInt(usize, args[2], 10) else 4;
    const m = try graph.createLayers(a, layer_count);
    defer m.deinit();
    var runner = try q.decode_round.Runner.init(a, m, 1, @intCast(@max(prompt_length + 8, 256)));
    defer runner.deinit();
    const session = q.session.Session{ .runner = &runner };
    var prompt: [640]u32 = undefined;
    for (&prompt, 0..) |*id, i| id.* = @intCast((i * 17 + 7) % 96);
    const control = q.session.Session{ .runner = &runner, .reference_tree = true };
    try control.prefill(prompt[0..prompt_length], 1);
    const reference = try session.fingerprint();
    const state_reference = try a.alloc([]u8, runner.gdn.budget.layers * 2);
    for (0..runner.gdn.budget.layers) |layer| {
        state_reference[layer * 2] = try a.dupe(u8, try runner.gdn.committed(layer, 0, true));
        state_reference[layer * 2 + 1] = try a.dupe(u8, try runner.gdn.committed(layer, 0, false));
    }

    var tokens: [8]u32 = undefined;
    for (&tokens, 0..) |*token, i| {
        token.* = try session.greedy();
        if (i + 1 < tokens.len) try control.step(token.*);
    }
    const final_reference = try session.fingerprint();
    var cases: usize = 0;
    for ([_]usize{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 32, 64, 128 }) |width| {
        try runner.reset(0);
        try session.prefill(prompt[0..prompt_length], width);
        if (!std.mem.eql(u8, &reference, &try session.fingerprint())) {
            std.debug.print("prompt state/logits differ at chunk {d}\n", .{width});
            for (0..runner.gdn.budget.layers) |layer| for ([_]bool{ true, false }, 0..) |conv, field| {
                const actual = try runner.gdn.committed(layer, 0, conv);
                var unequal: usize = 0;
                for (state_reference[layer * 2 + field], actual) |wanted, got| if (wanted != got) {
                    unequal += 1;
                };
                std.debug.print("committed layer{d} field{d} unequal_bytes{d}/{d}\n", .{ layer, field, unequal, actual.len });
            };

            return error.PromptDiffers;
        }
        for (tokens, 0..) |wanted, i| {
            const got = try session.greedy();
            if (got != wanted) return error.DecodeDiffers;
            if (i + 1 < tokens.len) try session.step(got);
        }
        cases += 1;
    }
    try runner.reset(0);
    try session.prefill(prompt[0..prompt_length], 16);
    var trace = @import("tensorfold").gpu_profile.Trace{ .queue = m.queue, .enabled = true };
    runner.profile(&trace);
    for (tokens, 0..) |wanted, i| {
        if (try session.greedy() != wanted) return error.ProfileTokensDiffer;
        if (i + 1 < tokens.len) try session.step(wanted);
    }
    if (!std.mem.eql(u8, &final_reference, &try session.fingerprint())) return error.ProfileStateDiffer;
    if (trace.calls[0] == 0 or trace.shape_count == 0) return error.EmptyProfile;
    runner.profile(null);
    std.debug.print("profiled split buffers: greedy tokens and final state/KV/logits byte-equal, {d} dense shapes\n", .{trace.shape_count});
    std.debug.print("qwen27 synthetic: {d} chunk widths byte-equal to chunk 1, all committed state/KV/logits and 8 greedy tokens, no model files\n", .{cases});
}
