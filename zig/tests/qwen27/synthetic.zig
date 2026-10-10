//! Synthetic hybrid checks on one generated graph: prompt chunks of every width, decoded rows against the CLI tree.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const q = tf.qwen27;
const graph = @import("synthetic_graph.zig");

/// Committed GDN state per layer and field, then each attention layer's KV, then the last logits: what differs.
fn report(r: *q.decode_round.Runner, gdn: []const []const u8, kv: []const []const u8, logits: []const u8) !void {
    for (0..r.gdn.budget.layers) |layer| for ([_]bool{ true, false }, 0..) |conv, field| {
        const got = try r.gdn.committed(layer, 0, conv);
        var unequal: usize = 0;
        for (gdn[layer * 2 + field], got) |want, have| unequal += @intFromBool(want != have);
        std.debug.print("  gdn layer{d} field{d} unequal_bytes {d}/{d}\n", .{ layer, field, unequal, got.len });
    };
    for (try kvBytes(r.allocator, r), kv, 0..) |got, want, i| {
        var unequal: usize = 0;
        for (want, got) |x, y| unequal += @intFromBool(x != y);
        std.debug.print("  attention kv{d} unequal_bytes {d}/{d}\n", .{ i, unequal, got.len });
    }
    const got = r.model.frame.get(.logits).buffer.contents()[0..logits.len];
    std.debug.print("  last logits equal {}\n", .{std.mem.eql(u8, got, logits)});
}

/// Each attention layer's visible keys then values for slot 0, as owned copies.
fn kvBytes(a: std.mem.Allocator, r: *q.decode_round.Runner) ![][]u8 {
    const layers = r.caches.len / r.slots;
    const out = try a.alloc([]u8, layers * 2);
    for (0..layers) |layer| {
        const cache = r.caches[layer * r.slots];
        for ([_]q.projection.Ref{ cache.keys, cache.values }, [_]u32{ cache.key_stride, cache.value_stride }, 0..) |ref, stride, j| {
            const head_stride: usize = if (stride == 0) cache.capacity * 256 else stride;
            var bytes: std.ArrayList(u8) = .empty;
            for (0..r.model.config.kv_heads) |head| try bytes.appendSlice(a, ref.buffer.contents()[ref.offset + head * head_stride * 2 ..][0 .. @as(usize, r.offsets[0]) * 256 * 2]);
            out[layer * 2 + j] = bytes.items;
        }
    }
    return out;
}

/// Every prompt row's logits at chunk `width` (prompt arithmetic, all rows projected), concatenated.
fn rowLogits(a: std.mem.Allocator, r: *q.decode_round.Runner, ids: []const u32, width: usize) ![]u8 {
    const v = r.model.config.vocab * 2;
    const out = try a.alloc(u8, ids.len * v);
    try r.reset(0);
    r.model.kernels.prompt = true;
    defer r.model.kernels.prompt = false;
    var first: usize = 0;
    while (first < ids.len) {
        const end = @min(ids.len, first + width);
        var round = try q.round_plan.Round.init(a, &.{.{ .slot = 0, .start = r.offsets[0], .capacity = r.capacity, .ids = ids[first..end] }}, @intCast(r.model.config.conv_kernel), r.slots, @intCast(r.model.config.vocab));
        defer round.deinit();
        try r.advance(&round, .all);
        @memcpy(out[first * v .. end * v], r.model.frame.get(.logits).buffer.contents()[0 .. (end - first) * v]);
        first = end;
    }
    return out;
}

/// The first prompt row whose logits differ between chunk 1 and chunk `width`.
fn firstRow(a: std.mem.Allocator, r: *q.decode_round.Runner, ids: []const u32, width: usize) !void {
    const one = try rowLogits(a, r, ids, 1);
    const wide = try rowLogits(a, r, ids, width);
    const v = r.model.config.vocab * 2;
    for (0..ids.len) |row| if (!std.mem.eql(u8, one[row * v ..][0..v], wide[row * v ..][0..v])) {
        std.debug.print("  first differing row {d}: chunk {d}, row {d} of it\n", .{ row, row / width, row % width });
        return;
    };
    std.debug.print("  every row's logits equal: the difference is in state the logits do not show yet\n", .{});
}

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
    const ids = prompt[0..prompt_length];

    // prompt rows: a row's bits never depend on where its prompt was cut, so every chunk width equals chunk 1
    try session.prefill(ids, 1);
    const reference = try session.fingerprint();
    const gdn_reference = try a.alloc([]u8, runner.gdn.budget.layers * 2);
    for (0..runner.gdn.budget.layers) |layer| {
        gdn_reference[layer * 2] = try a.dupe(u8, try runner.gdn.committed(layer, 0, true));
        gdn_reference[layer * 2 + 1] = try a.dupe(u8, try runner.gdn.committed(layer, 0, false));
    }
    const kv_reference = try kvBytes(a, &runner);
    const logits_reference = try a.dupe(u8, runner.model.frame.get(.logits).buffer.contents()[0 .. m.config.vocab * 2]);
    var tokens: [8]u32 = undefined;
    for (&tokens, 0..) |*token, i| {
        token.* = try session.greedy();
        if (i + 1 < tokens.len) try session.step(token.*);
    }
    const final_reference = try session.fingerprint();
    var cases: usize = 0;
    for ([_]usize{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 32, 64, 128 }) |width| {
        try runner.reset(0);
        try session.prefill(ids, width);
        if (!std.mem.eql(u8, &reference, &try session.fingerprint())) {
            std.debug.print("prompt state/logits differ at chunk {d}:\n", .{width});
            try report(&runner, gdn_reference, kv_reference, logits_reference);
            try firstRow(a, &runner, ids, width);
            return error.PromptDiffers;
        }
        for (tokens, 0..) |wanted, i| {
            const got = try session.greedy();
            if (got != wanted) return error.DecodeDiffers;
            if (i + 1 < tokens.len) try session.step(got);
        }
        cases += 1;
    }

    // decoded rows: the served chain equals the CLI's tree path (verify, then keep) at every chunk width
    const tree = q.session.Session{ .runner = &runner, .reference_tree = true };
    try runner.reset(0);
    for (ids, 0..) |id, i| try tree.decodeChunk(&.{id}, i + 1 == ids.len);
    const decoded = try session.fingerprint();
    var decoded_cases: usize = 0;
    for ([_]usize{ 1, 2, 3, 7, 16, 32, 128 }) |width| {
        try runner.reset(0);
        var first: usize = 0;
        while (first < ids.len) {
            const end = @min(ids.len, first + width);
            try session.decodeChunk(ids[first..end], end == ids.len);
            first = end;
        }
        if (!std.mem.eql(u8, &decoded, &try session.fingerprint())) {
            std.debug.print("decoded rows differ from the CLI tree at chunk {d}\n", .{width});
            return error.DecodedRowsDiffer;
        }
        decoded_cases += 1;
    }

    try runner.reset(0);
    try session.prefill(ids, 16);
    var trace = tf.gpu_profile.Trace{ .queue = m.queue, .enabled = true };
    runner.profile(&trace);
    for (tokens, 0..) |wanted, i| {
        if (try session.greedy() != wanted) return error.ProfileTokensDiffer;
        if (i + 1 < tokens.len) try session.step(wanted);
    }
    if (!std.mem.eql(u8, &final_reference, &try session.fingerprint())) return error.ProfileStateDiffer;
    if (trace.calls[0] == 0 or trace.shape_count == 0) return error.EmptyProfile;
    runner.profile(null);
    std.debug.print("profiled split buffers: greedy tokens and final state/KV/logits byte-equal, {d} dense shapes\n", .{trace.shape_count});
    std.debug.print("qwen27 synthetic: {d} prompt chunk widths byte-equal to chunk 1 with 8 greedy tokens, {d} decoded-row chunk widths equal to the CLI tree, no model files\n", .{ cases, decoded_cases });
}
