//! The Qwen3 CUDA engine against transformers: embedding vectors, the reranker's logits, a batch equal to alone.
const std = @import("std");
const engine = @import("engine.zig");
const checks = @import("../checks.zig");
const score = @import("../score.zig");

const t = std.testing;
const gpa = t.allocator;

/// The engine as score.decidePacked drives any runner: a trunk once a run, its branches after it.
const Runner = struct {
    e: *engine.Engine,
    hidden: usize,

    pub fn run(r: @This(), tokens: []const u32, trunk_len: u32, ends: []const u32, rows: []f32) !void {
        try r.e.run(tokens, trunk_len, ends, rows);
    }
};

test "Qwen3-Embedding: the card's and the long inputs on CUDA match transformers, a batch's rows equal alone's bits" {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const dir = try checks.env(a, "TF_QWEN3_EMBED_DIR") orelse return error.SkipZigTest;
    const ref = try checks.readJson(a, try checks.env(a, "TF_QWEN3_EMBED_REF") orelse return error.SkipZigTest);
    const bar = try checks.readJson(a, try checks.env(a, "TF_QWEN3_EMBED_BAR") orelse return error.SkipZigTest);
    try checks.checkReferences(ref, bar);
    try checks.checkInputs(ref, &checks.embed_inputs);
    try checks.checkFiles(a, t.io, dir, ref.object.get("pins").?);
    const inputs = ref.object.get("ids").?.array.items;
    const refs = ref.object.get("vectors").?.array.items;
    const bars = bar.object.get("vectors").?.array.items;
    try t.expectEqual(@as(usize, 9), inputs.len);
    try t.expectEqual(inputs.len, refs.len);
    try t.expectEqual(inputs.len, bars.len);
    const lengths = try a.alloc(usize, inputs.len);
    for (inputs, lengths) |one, *n| {
        try t.expect(one.array.items.len > 0);
        n.* = one.array.items.len;
    }
    try t.expect(lengths[inputs.len - 1] > 16384);
    const window = 32768;
    const e = try engine.Engine.load(gpa, t.io, dir, 0);
    defer e.deinit();
    const hidden = e.m.spec.hidden;
    for (refs, bars) |r, b| {
        try t.expectEqual(hidden, r.array.items.len);
        try t.expectEqual(hidden, b.array.items.len);
    }
    // the BF16 reference's own distance to fp32 sets the gate, rounded up, and never above 1e-3
    var bar_worst: f64 = 0;
    for (bars, refs) |b, r| {
        const bv = try a.alloc(f64, hidden);
        const rv = try a.alloc(f64, hidden);
        for (b.array.items, r.array.items, bv, rv) |x, y, *p, *q| {
            p.* = checks.number(x);
            q.* = checks.number(y);
        }
        bar_worst = @max(bar_worst, 1 - checks.cosine(bv, rv));
    }
    try t.expect(std.math.isFinite(bar_worst) and bar_worst > 0);
    const limit = @min(0.001, checks.roundUp(bar_worst));
    try e.size(window);
    var first: usize = 0;
    for (try checks.groupEnds(a, lengths, window)) |last| {
        defer first = last;
        var tokens: std.ArrayList(u32) = .empty;
        const ends = try a.alloc(u32, last - first);
        for (inputs[first..last], ends) |one, *end| {
            try tokens.appendSlice(a, try checks.tokensOf(a, one));
            end.* = @intCast(tokens.items.len);
        }
        const batch = try a.alloc(f32, (last - first) * hidden);
        try e.run(tokens.items, 0, ends, batch);
        var begin: u32 = 0;
        for (ends, 0..) |end, j| {
            defer begin = end;
            const i = first + j;
            const alone = try a.alloc(f32, hidden);
            try e.run(tokens.items[begin..end], 0, &.{end - begin}, alone);
            for (batch[j * hidden ..][0..hidden], alone) |x, y| try t.expectEqual(@as(u32, @bitCast(x)), @as(u32, @bitCast(y)));
            const got = try a.alloc(f64, hidden);
            const want = try a.alloc(f64, hidden);
            for (batch[j * hidden ..][0..hidden], refs[i].array.items, got, want) |x, r, *p, *q| {
                p.* = x;
                q.* = checks.number(r);
            }
            const distance = 1 - checks.cosine(got, want);
            std.debug.print("embedding input {d} ({d} tokens): cosine distance {e:.3} (limit {e:.3})\n", .{ i, end - begin, distance, limit });
            try t.expect(std.math.isFinite(distance) and distance <= limit);
        }
    }
}

test "Qwen3-Reranker: each case's yes - no logit matches transformers, and the packed decision equals the runs alone" {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const dir = try checks.env(a, "TF_QWEN3_RERANK_DIR") orelse return error.SkipZigTest;
    const ref = try checks.readJson(a, try checks.env(a, "TF_QWEN3_RERANK_REF") orelse return error.SkipZigTest);
    const cases = ref.object.get("cases").?.array.items;
    const yes: u32 = @intCast(ref.object.get("yes").?.integer);
    const no: u32 = @intCast(ref.object.get("no").?.integer);
    var prompts: std.ArrayList([]const u32) = .empty;
    var longest: usize = 0;
    for (cases) |c| {
        if (c.object.get("diff") == null) continue; // ids-only cases carry no score
        const prompt = try checks.tokensOf(a, c.object.get("ids").?);
        longest = @max(longest, prompt.len);
        try prompts.append(a, prompt);
    }
    try t.expect(prompts.items.len >= 32);
    const window = std.mem.alignForward(usize, @max(engine.least_window, longest), 512);
    const e = try engine.Engine.load(gpa, t.io, dir, 0);
    defer e.deinit();
    try e.size(window);
    const hidden = e.m.spec.hidden;
    var heads = score.Heads.of(&e.m, t.io, dir);
    defer heads.deinit(gpa);
    var worst_diff: f64 = 0;
    var worst_score: f64 = 0;
    for (cases) |c| {
        const want_diff = c.object.get("diff") orelse continue;
        const prompt = try checks.tokensOf(a, c.object.get("ids").?);
        const row = try a.alloc(f32, hidden);
        try e.run(prompt, 0, &.{@intCast(prompt.len)}, row);
        var logits: [2]f64 = undefined;
        try score.labelLogits(gpa, &heads, row, &.{ yes, no }, &logits);
        const diff = logits[0] - logits[1];
        const rank = 1 / (1 + @exp(-diff));
        const ref_score = checks.number(c.object.get("score").?);
        // a NaN would slip through @max: every value must be finite before it counts
        for ([_]f64{ diff, rank, checks.number(want_diff), ref_score }) |v| try t.expect(std.math.isFinite(v));
        std.debug.print("rerank {s} {d} tokens: diff {d:.4} ref {d:.4} score {d:.6} ref {d:.6}\n", .{ c.object.get("kind").?.string, prompt.len, diff, checks.number(want_diff), rank, ref_score });
        worst_diff = @max(worst_diff, @abs(diff - checks.number(want_diff)));
        worst_score = @max(worst_score, @abs(rank - ref_score));
    }
    std.debug.print("rerank: {d} cases, worst |diff| {d:.4} (gate 0.25), worst |score| {d:.5} (gate 0.02)\n", .{ prompts.items.len, worst_diff, worst_score });
    try t.expect(worst_diff <= 0.25);
    try t.expect(worst_score <= 0.02);
    // the decision path: branches packed into few runs, each prompt's logits the bits its own run gives
    var trunk_len: usize = std.math.maxInt(usize);
    for (prompts.items) |p| trunk_len = @min(trunk_len, p.len);
    trunk_len -= 1;
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
    for (prompts.items, branches, counts, 0..) |p, *b, *n, i| {
        b.* = p[trunk_len..];
        // unequal label counts: yes and no, then yes alone
        n.* = if (i % 2 == 0) 2 else 1;
        try labels.append(a, yes);
        if (i % 2 == 0) try labels.append(a, no);
    }
    const packed_logits = try a.alloc(f64, labels.items.len);
    try score.decidePacked(a, Runner{ .e = e, .hidden = hidden }, &heads, trunk, branches, counts, labels.items, packed_logits, window, engine.max_inputs);
    var at: usize = 0;
    for (prompts.items, 0..) |prompt, i| {
        const n = counts[i];
        const row = try a.alloc(f32, hidden);
        try e.run(prompt, 0, &.{@intCast(prompt.len)}, row);
        var alone: [2]f64 = undefined;
        try score.labelLogits(gpa, &heads, row, labels.items[at..][0..n], alone[0..n]);
        for (packed_logits[at..][0..n], alone[0..n]) |x, y| try t.expectEqual(@as(u64, @bitCast(y)), @as(u64, @bitCast(x)));
        at += n;
    }
    std.debug.print("rerank decide: {d} branches (trunk {d}) packed at a {d}-token window equal their runs alone\n", .{ prompts.items.len, trunk_len, window });
    // edges the cases' short common prefix never reaches: a trunk off a 512-key chunk, branches of 1 to over 512 rows
    var base = prompts.items[0];
    for (prompts.items) |p| {
        if (p.len > base.len) base = p;
    }
    const edge_trunk: usize = 700;
    const edge_lens = [_]usize{ 600, 513, 1, 530 };
    var edge_total: usize = edge_trunk;
    for (edge_lens) |n| edge_total += n;
    try t.expect(base.len >= edge_trunk + 600 and edge_total <= window);
    var edge_branches: [edge_lens.len][]const u32 = undefined;
    var edge_counts: [edge_lens.len]usize = undefined;
    var edge_labels: std.ArrayList(u32) = .empty;
    for (edge_lens, &edge_branches, &edge_counts, 0..) |n, *b, *c, i| {
        b.* = base[edge_trunk..][0..n];
        c.* = if (i % 2 == 0) 2 else 1;
        try edge_labels.append(a, yes);
        if (i % 2 == 0) try edge_labels.append(a, no);
    }
    const edge_packed = try a.alloc(f64, edge_labels.items.len);
    try score.decidePacked(a, Runner{ .e = e, .hidden = hidden }, &heads, base[0..edge_trunk], &edge_branches, &edge_counts, edge_labels.items, edge_packed, window, engine.max_inputs);
    var edge_at: usize = 0;
    for (edge_lens, edge_counts) |n, c| {
        const row = try a.alloc(f32, hidden);
        try e.run(base[0 .. edge_trunk + n], 0, &.{@intCast(edge_trunk + n)}, row);
        var alone: [2]f64 = undefined;
        try score.labelLogits(gpa, &heads, row, edge_labels.items[edge_at..][0..c], alone[0..c]);
        for (edge_packed[edge_at..][0..c], alone[0..c]) |x, y| try t.expectEqual(@as(u64, @bitCast(y)), @as(u64, @bitCast(x)));
        edge_at += c;
    }
    std.debug.print("rerank decide edges: trunk {d}, branches of {d}, {d}, {d} and {d} rows equal their runs alone\n", .{ edge_trunk, edge_lens[0], edge_lens[1], edge_lens[2], edge_lens[3] });
}
