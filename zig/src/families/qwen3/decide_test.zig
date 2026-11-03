//! Qwen3-Reranker: the real decide hook (branches packed into several runs) against scoreLabels alone, byte for byte.
const std = @import("std");
const hip = @import("hip");
const qwen3 = @import("qwen3");

const t = std.testing;
const gpa = t.allocator;

fn env(a: std.mem.Allocator, key: []const u8) !?[]u8 {
    return t.environ.getAlloc(a, key) catch |err| switch (err) {
        error.EnvironmentVariableMissing => null,
        else => err,
    };
}

test "Qwen3-Reranker decide: packed branches give each prompt's scoreLabels logits, across several runs" {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const dir = try env(a, "TF_QWEN3_RERANK_DIR") orelse return error.SkipZigTest;
    const ref_path = try env(a, "TF_QWEN3_RERANK_REF") orelse return error.SkipZigTest;
    const bytes = try std.Io.Dir.cwd().readFileAlloc(t.io, ref_path, a, .limited(64 << 20));
    const ref = try std.json.parseFromSliceLeaky(std.json.Value, a, bytes, .{});
    const window = 2048;
    // the length cases share a query and the default instruction: whole prompts of up to 2,048 tokens
    var prompts: std.ArrayList([]const u32) = .empty;
    for (ref.object.get("cases").?.array.items) |c| {
        if (!std.mem.startsWith(u8, c.object.get("kind").?.string, "len")) continue;
        const ids = c.object.get("ids").?.array.items;
        if (ids.len > window) continue;
        const one = try a.alloc(u32, ids.len);
        for (one, ids) |*o, v| o.* = @intCast(v.integer);
        try prompts.append(a, one);
    }
    try t.expect(prompts.items.len >= 6);
    var total: usize = 0;
    var shortest: usize = std.math.maxInt(usize);
    for (prompts.items) |p| {
        total += p.len;
        shortest = @min(shortest, p.len);
    }
    try t.expect(total > window); // several runs at this window
    var trunk_len: usize = shortest - 1;
    for (prompts.items[1..]) |p| {
        var n: usize = 0;
        while (n < trunk_len and p[n] == prompts.items[0][n]) n += 1;
        trunk_len = n;
    }
    try t.expect(trunk_len > 0);
    const trunk = prompts.items[0][0..trunk_len];
    const branches = try a.alloc([]const u32, prompts.items.len);
    const counts = try a.alloc(usize, prompts.items.len);
    var labels: std.ArrayList(u32) = .empty;
    const yes: u32 = @intCast(ref.object.get("yes").?.integer);
    const no: u32 = @intCast(ref.object.get("no").?.integer);
    for (prompts.items, branches, counts, 0..) |p, *b, *n, i| {
        b.* = p[trunk_len..];
        // unequal label counts: yes and no, then yes alone
        n.* = if (i % 2 == 0) 2 else 1;
        try labels.append(a, yes);
        if (i % 2 == 0) try labels.append(a, no);
    }
    const e = try qwen3.engine.Engine.load(gpa, t.io, dir, 0, .{});
    defer e.deinit();
    try e.size(window);
    var p = qwen3.prompt.Prompt.init(gpa, e, false);
    defer p.deinit();
    p.heads = qwen3.score.Heads.of(&e.m, t.io, dir);
    const packed_logits = try a.alloc(f64, labels.items.len);
    try qwen3.prompt.Prompt.decideLabels(&p, trunk, branches, counts, labels.items, packed_logits);
    var at: usize = 0;
    for (prompts.items, counts) |prompt, n| {
        const alone = try a.alloc(f64, n);
        _ = try qwen3.prompt.Prompt.scoreLabels(&p, prompt, labels.items[at..][0..n], alone, false);
        for (packed_logits[at..][0..n], alone) |x, y| try t.expectEqual(@as(u64, @bitCast(y)), @as(u64, @bitCast(x)));
        at += n;
    }
    std.debug.print("rerank decide: {d} branches ({d} tokens, trunk {d}) at a {d}-token window equal scoreLabels alone\n", .{ prompts.items.len, total, trunk_len, window });
}
