//! Qwen3-Reranker: the real decide hook against scoreLabels alone, byte for byte, on TF_QWEN3_MATRIX and _ATTENTION.
const std = @import("std");
const hip = @import("hip");
const qwen3 = @import("qwen3");
const checks = @import("forward_test.zig");

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
    const e = try qwen3.engine.Engine.load(gpa, t.io, dir, 0, try checks.choices());
    defer e.deinit();
    try e.size(window);
    var p = qwen3.prompt.Prompt.init(gpa, e, false);
    defer p.deinit();
    p.heads = qwen3.score.Heads.of(&e.m, t.io, dir);
    const packed_logits = try a.alloc(f64, labels.items.len);
    var problem: []const u8 = "";
    try qwen3.prompt.Prompt.decideLabels(&p, trunk, branches, counts, labels.items, packed_logits, &problem);
    var at: usize = 0;
    for (prompts.items, counts) |prompt, n| {
        const alone = try a.alloc(f64, n);
        _ = try qwen3.prompt.Prompt.scoreLabels(&p, prompt, labels.items[at..][0..n], alone, false);
        for (packed_logits[at..][0..n], alone) |x, y| try t.expectEqual(@as(u64, @bitCast(y)), @as(u64, @bitCast(x)));
        at += n;
    }
    std.debug.print("rerank decide ({t}, {s}): {d} branches ({d} tokens, trunk {d}) at a {d}-token window equal scoreLabels alone\n", .{ e.lib.dense.serve, checks.attentionPath(&e.lib), prompts.items.len, total, trunk_len, window });
}

test "Qwen3-Reranker decide: every scored reference case, the windows' too, packed equals scoreLabels alone" {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const dir = try env(a, "TF_QWEN3_RERANK_DIR") orelse return error.SkipZigTest;
    const ref_path = try env(a, "TF_QWEN3_RERANK_REF") orelse return error.SkipZigTest;
    const bytes = try std.Io.Dir.cwd().readFileAlloc(t.io, ref_path, a, .limited(64 << 20));
    const ref = try std.json.parseFromSliceLeaky(std.json.Value, a, bytes, .{});
    const window = 40960; // the config's window: the 40,960-token case fits, served with --context 0
    var prompts: std.ArrayList([]const u32) = .empty;
    for (ref.object.get("cases").?.array.items) |c| {
        if (c.object.get("diff") == null) continue; // ids-only cases (the special one, one past a window) score nothing
        const ids = c.object.get("ids").?.array.items;
        const one = try a.alloc(u32, ids.len);
        for (one, ids) |*o, v| o.* = @intCast(v.integer);
        try prompts.append(a, one);
    }
    try t.expect(prompts.items.len >= 32);
    var shortest: usize = std.math.maxInt(usize);
    for (prompts.items) |p| shortest = @min(shortest, p.len);
    var trunk_len: usize = shortest - 1;
    for (prompts.items[1..]) |p| {
        var n: usize = 0;
        while (n < trunk_len and p[n] == prompts.items[0][n]) n += 1;
        trunk_len = n;
    }
    const trunk = prompts.items[0][0..trunk_len];
    const yes: u32 = @intCast(ref.object.get("yes").?.integer);
    const no: u32 = @intCast(ref.object.get("no").?.integer);
    const branches = try a.alloc([]const u32, prompts.items.len);
    const counts = try a.alloc(usize, prompts.items.len);
    const labels = try a.alloc(u32, 2 * prompts.items.len);
    for (prompts.items, branches, counts, 0..) |p, *b, *n, i| {
        b.* = p[trunk_len..];
        n.* = 2;
        labels[2 * i ..][0..2].* = .{ yes, no };
    }
    const e = try qwen3.engine.Engine.load(gpa, t.io, dir, 0, try checks.choices());
    defer e.deinit();
    try e.size(window);
    var p = qwen3.prompt.Prompt.init(gpa, e, false);
    defer p.deinit();
    p.heads = qwen3.score.Heads.of(&e.m, t.io, dir);
    const packed_logits = try a.alloc(f64, labels.len);
    var problem: []const u8 = "";
    try qwen3.prompt.Prompt.decideLabels(&p, trunk, branches, counts, labels, packed_logits, &problem);
    // an empty branch is the client's error: refused with its reason before any GPU work
    var empty = [_][]const u32{&.{}};
    const refused = qwen3.prompt.Prompt.decideLabels(&p, trunk, &empty, counts[0..1], labels[0..2], packed_logits[0..2], &problem);
    try t.expectError(error.Refused, refused);
    try t.expectEqualStrings(qwen3.score.problemOf(error.BadPrompt), problem);
    var tokens: usize = 0;
    for (prompts.items, 0..) |prompt, i| {
        var alone: [2]f64 = undefined;
        _ = try qwen3.prompt.Prompt.scoreLabels(&p, prompt, labels[2 * i ..][0..2], &alone, false);
        for (packed_logits[2 * i ..][0..2], alone) |x, y| try t.expectEqual(@as(u64, @bitCast(y)), @as(u64, @bitCast(x)));
        tokens += prompt.len;
    }
    std.debug.print("rerank decide ({t}, {s}): all {d} scored cases ({d} tokens, trunk {d}) packed at a {d}-token window equal alone\n", .{ e.lib.dense.serve, checks.attentionPath(&e.lib), prompts.items.len, tokens, trunk_len, window });
}

/// The decide runs through the engine at a window below its own, so one load also gives several runs.
const Runner = struct {
    e: *qwen3.engine.Engine,
    hidden: usize,
    runs: usize = 0,

    pub fn run(r: *Runner, tokens: []const u32, trunk: u32, ends: []const u32, rows: []f32) !void {
        r.runs += 1;
        try r.e.run(tokens, trunk, ends, rows);
    }
};

test "Qwen3-Reranker decide: a trunk longer than one span, in one run and in two, equals scoreLabels alone" {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const dir = try env(a, "TF_QWEN3_RERANK_DIR") orelse return error.SkipZigTest;
    const ref_path = try env(a, "TF_QWEN3_RERANK_REF") orelse return error.SkipZigTest;
    const bytes = try std.Io.Dir.cwd().readFileAlloc(t.io, ref_path, a, .limited(64 << 20));
    const ref = try std.json.parseFromSliceLeaky(std.json.Value, a, bytes, .{});
    var prompt: []u32 = &.{};
    for (ref.object.get("cases").?.array.items) |c| {
        if (!std.mem.eql(u8, c.object.get("kind").?.string, "len8192")) continue;
        const ids = c.object.get("ids").?.array.items;
        prompt = try a.alloc(u32, ids.len);
        for (prompt, ids) |*o, v| o.* = @intCast(v.integer);
    }
    try t.expectEqual(@as(usize, 8272), prompt.len);
    // a 2,100-token trunk (past one 2,048-row span), then branches of 1, 100, 2,048 (a span) and the rest
    const trunk_len = 2100;
    const trunk = prompt[0..trunk_len];
    const lengths = [_]usize{ 1, 100, 2048, prompt.len - trunk_len };
    var branches: [4][]const u32 = undefined;
    for (&branches, lengths) |*b, n| b.* = prompt[trunk_len..][0..n];
    const counts = [_]usize{ 2, 2, 2, 2 };
    const yes: u32 = @intCast(ref.object.get("yes").?.integer);
    const no: u32 = @intCast(ref.object.get("no").?.integer);
    const labels = [_]u32{ yes, no, yes, no, yes, no, yes, no };
    const e = try qwen3.engine.Engine.load(gpa, t.io, dir, 0, try checks.choices());
    defer e.deinit();
    try e.size(40960);
    var p = qwen3.prompt.Prompt.init(gpa, e, false);
    defer p.deinit();
    p.heads = qwen3.score.Heads.of(&e.m, t.io, dir);
    var one: [8]f64 = undefined;
    var problem: []const u8 = "";
    try qwen3.prompt.Prompt.decideLabels(&p, trunk, &branches, &counts, &labels, &one, &problem);
    // at 4,200 tokens the first three take two runs: the trunk + 1 + 100, then the trunk + 2,048
    var runner: Runner = .{ .e = e, .hidden = e.m.spec.hidden };
    var two: [6]f64 = undefined;
    try qwen3.score.decidePacked(a, &runner, &p.heads.?, trunk, branches[0..3], counts[0..3], labels[0..6], &two, 4200, qwen3.engine.max_inputs);
    try t.expectEqual(@as(usize, 2), runner.runs);
    for (lengths, 0..) |n, i| {
        var alone: [2]f64 = undefined;
        _ = try qwen3.prompt.Prompt.scoreLabels(&p, prompt[0 .. trunk_len + n], labels[2 * i ..][0..2], &alone, false);
        for (one[2 * i ..][0..2], alone) |x, y| try t.expectEqual(@as(u64, @bitCast(y)), @as(u64, @bitCast(x)));
        if (i < 3) for (two[2 * i ..][0..2], alone) |x, y| try t.expectEqual(@as(u64, @bitCast(y)), @as(u64, @bitCast(x)));
    }
    std.debug.print("rerank decide ({t}, {s}): a {d}-token trunk, branches of 1, 100, 2048 and {d} tokens, in one run and two, equal alone\n", .{ e.lib.dense.serve, checks.attentionPath(&e.lib), trunk_len, lengths[3] });
}
