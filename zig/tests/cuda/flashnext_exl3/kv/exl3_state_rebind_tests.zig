const std = @import("std");
const cuda = @import("cuda");
const flash = @import("flashnext");
const check = @import("../../check.zig");

const graph = flash.exl3.graph;
const iterations = 4;
const prefix = [_]u32{ 760, 6511, 314, 9338, 369, 11751, 13, 1049, 369, 3750, 364, 279, 242476, 300, 21262, 13 };

fn require(name: [:0]const u8, value: []const u8) !void {
    const setting = std.c.getenv(name);
    try check.expect(setting != null and std.mem.eql(u8, std.mem.span(setting.?), value), "rebind fixture requires {s}={s}", .{ name, value });
}

fn stateBytes(sequence: *const flash.prompt.Seq) usize {
    var total = sequence.tail.len;
    for ([_][]cuda.DeviceBuffer{ sequence.state, sequence.conv }) |buffers| for (buffers) |buffer| {
        total += buffer.len;
    };
    return total;
}

fn readState(sequence: *const flash.prompt.Seq, target: []u8) !void {
    var offset: usize = 0;
    for ([_][]cuda.DeviceBuffer{ sequence.state, sequence.conv }) |buffers| for (buffers) |buffer| {
        try buffer.download(0, target[offset..][0..buffer.len]);
        offset += buffer.len;
    };
    try sequence.tail.download(0, target[offset..][0..sequence.tail.len]);
}

pub fn run(engine: anytype, comptime operations: anytype) !void {
    try require("TF_FLASHNEXT_EXL3_GRAPH_ASYNC", "0");
    graph.rebind.state_rebind = true;
    graph.cache.capture_after_override = 1;
    defer {
        graph.rebind.state_rebind = false;
        graph.cache.capture_after_override = null;
    }
    try check.expect(engine.seq.paged == null and engine.world == 2, "rebind fixture requires two ranks and the dense control KV layout", .{});
    var draft: flash.mtp.Draft = undefined;
    draft.keep = .{};
    draft.verify_cache = .{};
    try check.expect(engine.seq.state.len == draft.spare.len, "rebind fixture requires the complete recurrent layer layout", .{});
    var made: usize = 0;
    defer {
        cuda.completion.drain(engine.gpu.d, engine.stream, "state rebind fixture consumers did not drain");
        graph.clear();
        for (draft.spare[0..made]) |*buffer| buffer.free();
    }
    for (engine.seq.state) |state| {
        draft.spare[made] = try cuda.DeviceBuffer.alloc(engine.gpu.d, state.len);
        made += 1;
    }
    var linear: usize = 0;
    for (engine.e.layers[0..engine.e.n_layers], 0..) |layer, index| {
        if (!layer.linear) continue;
        draft.keep.layers[index].state_out = draft.spare[linear].ptr;
        linear += 1;
    }
    try check.expect(linear == made, "rebind fixture maps every physical recurrent layer", .{});
    const allocator = engine.gpu.gpa;
    const state_bytes = stateBytes(&engine.seq);
    const reference_state = try allocator.alloc(u8, iterations * state_bytes);
    defer allocator.free(reference_state);
    const actual_state = try allocator.alloc(u8, state_bytes);
    defer allocator.free(actual_state);
    const reference_logits = try allocator.alloc(u8, iterations * 16 * flash.prompt.vocabulary * 2);
    defer allocator.free(reference_logits);
    const actual_logits = try allocator.alloc(u8, 16 * flash.prompt.vocabulary * 2);
    defer allocator.free(actual_logits);
    const previous_mode = graph.force;
    defer graph.force = previous_mode;
    for ([_]usize{ 3, 4, 16 }) |width| {
        const row_bytes = width * flash.prompt.vocabulary * 2;
        var before: graph.Stats = undefined;
        for ([_]bool{ false, true }) |captured| {
            try flash.exl3.dplan.toHost(&engine.e.exl3_dplan);
            graph.clear();
            graph.force = .off;
            try operations.reset(engine);
            for (draft.spare) |buffer| try buffer.fill8(0, engine.stream.handle);
            try operations.forward(engine, &prefix, .{ .logits = .last });
            graph.force = if (captured) .graph else .dplan;
            before = graph.stats;
            for (0..iterations) |step| {
                try operations.forward(engine, prefix[0..width], .{ .logits = .all, .logits_out = engine.out.ptr, .keep = &draft.keep });
                try engine.out.download(0, actual_logits[0..row_bytes]);
                const input_before = engine.seq.state[0].ptr;
                const output_before = draft.spare[0].ptr;
                draft.open = true;
                draft.settle(&engine.seq);
                try check.expect(engine.seq.state[0].ptr == output_before and draft.spare[0].ptr == input_before, "rank {d}: width {d} step {d} actually swaps the live state pair", .{ engine.rank, width, step });
                try readState(&engine.seq, actual_state);
                const wanted_state = reference_state[step * state_bytes ..][0..state_bytes];
                const wanted_logits = reference_logits[step * row_bytes ..][0..row_bytes];
                if (!captured) {
                    @memcpy(wanted_state, actual_state);
                    @memcpy(wanted_logits, actual_logits[0..row_bytes]);
                }
                const equal = std.mem.eql(u8, wanted_state, actual_state) and std.mem.eql(u8, wanted_logits, actual_logits[0..row_bytes]);
                const other_equal = try operations.swap(engine, @intFromBool(equal));
                try check.expect(equal and other_equal == 1, "rank {d}: width {d} step {d} exact eager/graph logits, recurrent state, convolution and PLE tail", .{ engine.rank, width, step });
                const digest = std.hash.Wyhash.hash(0, actual_logits[0..row_bytes]);
                const other_digest = try operations.swap(engine, digest);
                try check.expect(other_digest == digest, "rank {d}: width {d} step {d} cross-rank BF16 logits", .{ engine.rank, width, step });
            }
        }
        const after = graph.stats;
        const succeeded = after.rebinds - before.rebinds;
        const failures = after.rebind_failures - before.rebind_failures;
        const proved = succeeded == iterations - 1 and failures == 0 and after.rebind_attempts - before.rebind_attempts == succeeded and after.captures - before.captures == 1 and after.replays - before.replays == succeeded;
        const other_proved = try operations.swap(engine, @intFromBool(proved));
        try check.expect(proved and other_proved == 1, "rank {d}: width {d} needs one capture and three actual successful widened rebind replays, zero failures; got {d}/{d}", .{ engine.rank, width, succeeded, failures });
        check.pass("rank {d}/2 width {d}: real state flips; exact eager/graph logits and state; actual widened rebinds {d}, failures {d}", .{ engine.rank, width, succeeded, failures });
    }
}
