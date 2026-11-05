//! The Qwen3 CUDA engine against transformers: embedding vectors, a batch's rows equal to alone's bits.
const std = @import("std");
const engine = @import("engine.zig");
const checks = @import("../checks.zig");

const t = std.testing;
const gpa = t.allocator;

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
