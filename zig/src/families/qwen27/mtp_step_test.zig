//! The MTP head on real weights: TF_QWEN27_DIR, TF_QWEN27_MTP_DIR, and TF_QWEN27_MTP_PROMPT (space-separated ids).
const std = @import("std");
const platform = @import("sys");
const Model = @import("model.zig").Model;
const Runner = @import("decode_round.zig").Runner;
const Round = @import("round_plan.zig").Round;
const session = @import("session.zig");
const mtp = @import("mtp.zig");

fn prompt(a: std.mem.Allocator, io: std.Io, path: []const u8) ![]u32 {
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 20));
    defer a.free(text);
    var ids: std.ArrayList(u32) = .empty;
    var it = std.mem.tokenizeAny(u8, text, " \n\t,");
    while (it.next()) |word| try ids.append(a, try std.fmt.parseInt(u32, word, 10));
    return ids.toOwnedSlice(a);
}

/// The head's pick at each of its `rows` logits rows.
fn picks(m: *Model, sc: *const mtp.Scratch, rows: usize, out: []u32) !void {
    const vocab = m.config.vocab;
    const all = sc.logits.slice(u16, rows * vocab);
    for (out[0..rows], 0..) |*o, i| o.* = try session.argmax(all[i * vocab ..][0..vocab]);
}

test "qwen27 MTP head: its picks follow the target's next token, and a pass split at a cache row picks the same" {
    const dir = std.mem.span(platform.c.getenv("TF_QWEN27_DIR") orelse return error.SkipZigTest);
    const mtp_dir = std.mem.span(platform.c.getenv("TF_QWEN27_MTP_DIR") orelse return error.SkipZigTest);
    const ids_path = std.mem.span(platform.c.getenv("TF_QWEN27_MTP_PROMPT") orelse return error.SkipZigTest);
    const a = std.testing.allocator;
    const io = std.testing.io;
    const ids = try prompt(a, io, ids_path);
    defer a.free(ids);
    const m = try Model.loadDeferred(a, io, dir, 128);
    defer m.deinit();
    try m.loadMtp(io, mtp_dir);
    try m.loadShare(0);
    var runner = try Runner.init(a, m, 1, 256);
    defer runner.deinit();
    const s = session.Session{ .runner = &runner };
    // the target's greedy continuation of the prompt, so the head is scored on rows the target wrote
    var seq: [128]u32 = undefined;
    const n_prompt = @min(ids.len, 64);
    @memcpy(seq[0..n_prompt], ids[0..n_prompt]);
    try s.prefill(seq[0..n_prompt], 128);
    var n: usize = n_prompt;
    while (n < seq.len) : (n += 1) {
        seq[n] = try s.greedy();
        if (n + 1 < seq.len) try s.step(seq[n]);
    }
    // the whole sequence once more in one pass, layer 63's rows (before the final norm) kept in the tap plane
    try runner.reset(0);
    runner.capture_taps = true;
    runner.taps.ids[0] = m.config.layers - 1;
    runner.taps.count = 1;
    try s.decodeChunk(seq[0..n], true);
    var sc = try mtp.Scratch.init(m.device, m.frame.capacity, m.config.hidden, m.config.vocab, 128);
    defer sc.deinit();
    const hidden = @import("projection.zig").Ref{ .buffer = runner.taps.buffer };
    const rows = n - 1; // row j: the target's hidden at j and token j + 1; its pick is token j + 2's guess
    var whole: [128]u32 = undefined;
    {
        var round = try Round.init(a, &.{.{ .slot = 0, .start = 0, .capacity = runner.capacity, .ids = seq[1..n] }}, @intCast(m.config.conv_kernel), 1, @intCast(m.shard.global.vocab));
        defer round.deinit();
        _ = try runner.headPass(&round, &sc, hidden, .all);
        try picks(m, &sc, rows, &whole);
    }
    var hits: usize = 0;
    for (n_prompt - 1..rows - 1) |j| hits += @intFromBool(whole[j] == seq[j + 2]);
    const scored = rows - 1 - (n_prompt - 1);
    std.debug.print("MTP head: {d} of {d} generated rows pick the target's token ({d}%)\n", .{ hits, scored, hits * 100 / scored });
    try std.testing.expect(hits * 2 > scored);
    // the same rows as two passes: the first `cut` rows into the head's cache, then the rest reading them
    const cut = 40;
    var split: [128]u32 = undefined;
    for ([_][2]usize{ .{ 0, cut }, .{ cut, rows } }) |span| {
        var round = try Round.init(a, &.{.{ .slot = 0, .start = @intCast(span[0]), .capacity = runner.capacity, .ids = seq[span[0] + 1 .. span[1] + 1] }}, @intCast(m.config.conv_kernel), 1, @intCast(m.shard.global.vocab));
        defer round.deinit();
        _ = try runner.headPass(&round, &sc, .{ .buffer = runner.taps.buffer, .offset = span[0] * m.config.hidden * 2 }, .all);
        try picks(m, &sc, span[1] - span[0], split[span[0]..]);
    }
    var same: usize = 0;
    for (0..rows) |j| same += @intFromBool(whole[j] == split[j]);
    std.debug.print("MTP head: split at row {d}, {d} of {d} picks equal the whole pass's\n", .{ cut, same, rows });
    try std.testing.expect(same + 2 >= rows);
    // a chain: row k - 1 again (its real hidden and token k), then row k from the head's own row and its pick
    var first: usize = 0;
    var both: usize = 0;
    const vocab = m.config.vocab;
    for (n_prompt..rows - 2) |k| {
        var round = try Round.init(a, &.{.{ .slot = 0, .start = @intCast(k - 1), .capacity = runner.capacity, .ids = seq[k .. k + 1] }}, @intCast(m.config.conv_kernel), 1, @intCast(m.shard.global.vocab));
        defer round.deinit();
        const y = try runner.headPass(&round, &sc, .{ .buffer = runner.taps.buffer, .offset = (k - 1) * m.config.hidden * 2 }, .last);
        const d1 = try mtp.pick(sc.logits.slice(u16, vocab));
        var chain = try Round.init(a, &.{.{ .slot = 0, .start = @intCast(k), .capacity = runner.capacity, .ids = &.{d1.token} }}, @intCast(m.config.conv_kernel), 1, @intCast(m.shard.global.vocab));
        defer chain.deinit();
        _ = try runner.headPass(&chain, &sc, y, .last);
        const d2 = try mtp.pick(sc.logits.slice(u16, vocab));
        try std.testing.expect(d1.chance > 0 and d1.chance <= 1 and d2.chance > 0 and d2.chance <= 1);
        first += @intFromBool(d1.token == seq[k + 1]);
        both += @intFromBool(d1.token == seq[k + 1] and d2.token == seq[k + 2]);
    }
    const tried = rows - 2 - n_prompt;
    std.debug.print("MTP head: chained, depth 1 {d} of {d}, depths 1 and 2 {d} of {d}\n", .{ first, tried, both, tried });
    try std.testing.expect(both * 3 > tried);
}
