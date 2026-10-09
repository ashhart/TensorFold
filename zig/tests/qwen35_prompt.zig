//! A prompt filled in wide chunks (MLX's NAX qmm for the projections) against the same prompt in 32-row chunks (the row
//! decoder's projections): last-row logits, their argmax and a greedy continuation from each cache.
const std = @import("std");
const mtl = @import("metal");
const q = @import("tensorfold").qwen35;
const st = q.state;
const fwd = q.forward;
const c = q.config;

fn fill(m: *q.Model, s: *st.Scratch, cache: *st.Cache, ids: []const u32, chunk: usize) !void {
    var at: usize = 0;
    while (at < ids.len) {
        const rows = @min(chunk, ids.len - at);
        try fwd.run(m, s, &.{.{ .cache = cache, .rows = rows }}, ids[at..][0..rows], false, .last);
        at += rows;
    }
}

fn logits(cache: *const st.Cache) []const u16 {
    return @as([*]const u16, @ptrCast(@alignCast(cache.logits.contents())))[0..c.vocab];
}

fn value(x: u16) f32 {
    return @bitCast(@as(u32, x) << 16);
}

fn argmax(l: []const u16) u32 {
    var best: usize = 0;
    for (l, 0..) |x, i| if (value(x) > value(l[best])) {
        best = i;
    };
    return @intCast(best);
}

fn greedy(m: *q.Model, s: *st.Scratch, cache: *st.Cache, out: []u32) !void {
    for (out) |*t| {
        t.* = argmax(logits(cache));
        try fwd.run(m, s, &.{.{ .cache = cache, .rows = 1 }}, &.{t.*}, false, .all);
    }
}

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 4) return error.ExpectedModelTokensRows;
    const n = try std.fmt.parseInt(usize, args[2], 10);
    const wide = try std.fmt.parseInt(usize, args[3], 10);
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const m = try q.Model.load(init.gpa, init.io, args[1]);
    defer m.deinit();
    if (m.prompt == null) std.debug.print("no prompt kernels on this GPU: both fills take the row decoder\n", .{});
    const steps = 64;
    var s = try st.Scratch.init(init.gpa, m.device, m.config.g, @max(32, wide), n + steps + 1);
    defer s.deinit();
    var narrow_cache = try st.Cache.init(init.gpa, m.device, m.config.g, n + steps + 1);
    defer narrow_cache.deinit();
    var wide_cache = try st.Cache.init(init.gpa, m.device, m.config.g, n + steps + 1);
    defer wide_cache.deinit();
    const ids = try init.gpa.alloc(u32, n);
    defer init.gpa.free(ids);
    for (ids, 0..) |*id, i| id.* = @intCast(1000 + (i * 7919 + i / 13) % 90000);

    try fill(m, &s, &narrow_cache, ids, 32);
    try fill(m, &s, &wide_cache, ids, wide);
    const a, const b = .{ logits(&narrow_cache), logits(&wide_cache) };
    var max_diff: f32 = 0;
    var max_abs: f32 = 0;
    var equal: usize = 0;
    for (a, b) |x, y| {
        max_diff = @max(max_diff, @abs(value(x) - value(y)));
        max_abs = @max(max_abs, @abs(value(x)));
        equal += @intFromBool(x == y);
    }
    std.debug.print("{d}-token prompt, 32-row vs {d}-row chunks: last-row logits max |diff| {d:.4} (max |logit| {d:.2}), {d}/{d} bit-equal, argmax {d} vs {d}\n", .{ n, wide, max_diff, max_abs, equal, a.len, argmax(a), argmax(b) });

    var ga: [steps]u32 = undefined;
    var gb: [steps]u32 = undefined;
    try greedy(m, &s, &narrow_cache, &ga);
    try greedy(m, &s, &wide_cache, &gb);
    var same: usize = 0;
    while (same < steps and ga[same] == gb[same]) same += 1;
    std.debug.print("greedy continuation: first {d} of {d} tokens equal\n", .{ same, steps });
}
