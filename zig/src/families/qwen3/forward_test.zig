//! lastRows on the GPU against transformers: the embedding's pooled rows and the reranker's yes/no logits.

// Skipped without TF_QWEN3_EMBED_* or TF_QWEN3_RERANK_*; TF_QWEN3_MATRIX and TF_QWEN3_ATTENTION pick the kernels.
const std = @import("std");
const hip = @import("hip");
const qwen3 = @import("qwen3");

const t = std.testing;
const gpa = t.allocator;

/// The device, its kernels and an arena: what the engine holds around `lastRows`.
const Gpu = struct {
    r: hip.Runtime,
    ctx: hip.Context,
    stream: hip.Stream,
    l: hip.Launcher,
    arena: hip.Arena,

    fn open(g: *Gpu, arena_bytes: usize) !void {
        g.r = try hip.Runtime.open();
        errdefer g.r.close();
        g.ctx = try hip.Context.init(&g.r, 0);
        errdefer g.ctx.deinit();
        g.stream = try hip.Stream.init(&g.r);
        errdefer g.stream.deinit();
        const c = try choices();
        g.l = try hip.Launcher.load(&g.r, c, hip.kernels.images);
        std.debug.print("qwen3 GPU checks: matrix={t}, dense products on {t}, {s}\n", .{ c.products.matrix, g.l.dense.serve, attentionPath(&g.l) });
        errdefer g.l.unload();
        g.arena = try hip.Arena.init(&g.r, arena_bytes);
    }

    fn close(g: *Gpu) void {
        g.arena.deinit();
        g.l.unload();
        g.stream.deinit();
        g.ctx.deinit();
        g.r.close();
    }

    fn ops(g: *Gpu) hip.ops.Ops {
        return .{ .l = &g.l, .bf16 = true, .stream = g.stream.handle, .arena = &g.arena };
    }

    /// Each input's final-normed last row, as BF16 bits (`ends.len` rows of `hidden`).
    fn lastRows(g: *Gpu, m: *const qwen3.Model, ids: []const i32, ends: []const u32) ![]u16 {
        const mark = g.arena.mark();
        defer g.arena.release(mark);
        var dev = try hip.DeviceBuffer.fromHost(&g.r, std.mem.sliceAsBytes(ids));
        defer dev.free();
        const hidden = m.spec.hidden;
        var out = try hip.DeviceBuffer.alloc(&g.r, ends.len * hidden * 2);
        defer out.free();
        try qwen3.lastRows(g.ops(), m, @intFromPtr(dev.ptr), 0, ends, .{ .ptr = @intFromPtr(out.ptr), .kind = .bf16 });
        try g.stream.synchronize();
        const rows = try gpa.alloc(u16, ends.len * hidden);
        errdefer gpa.free(rows);
        try out.download(0, std.mem.sliceAsBytes(rows));
        return rows;
    }
};

pub const Matrix = @FieldType(@FieldType(hip.Launcher.Choices, "products"), "matrix");

/// TF_QWEN3_MATRIX as the policy's matrix choice; unset is auto.
pub fn matrixChoice() !Matrix {
    const v = t.environ.getAlloc(gpa, "TF_QWEN3_MATRIX") catch |err| switch (err) {
        error.EnvironmentVariableMissing => return .auto,
        else => return err,
    };
    defer gpa.free(v);
    return std.meta.stringToEnum(Matrix, v) orelse error.BadMatrixChoice;
}

/// TF_QWEN3_ATTENTION as the policy's attention choice: `wide` (unset) keeps the 64-row tile, `f32` turns it off.
pub fn wideChoice() !bool {
    const v = t.environ.getAlloc(gpa, "TF_QWEN3_ATTENTION") catch |err| switch (err) {
        error.EnvironmentVariableMissing => return true,
        else => return err,
    };
    defer gpa.free(v);
    if (std.mem.eql(u8, v, "wide")) return true;
    if (std.mem.eql(u8, v, "f32")) return false;
    return error.BadAttentionChoice;
}

/// The engine's kernel choices for these checks: TF_QWEN3_MATRIX and TF_QWEN3_ATTENTION.
pub fn choices() !hip.Launcher.Choices {
    return .{ .wide = try wideChoice(), .products = .{ .matrix = try matrixChoice() } };
}

/// The prompt attention path the launcher serves: the 64-row tile, or the 16-row one under attention=f32.
pub fn attentionPath(l: *const hip.Launcher) []const u8 {
    return if (l.wide) "wide attention" else "16-row attention";
}

const checks = @import("checks.zig");

/// The GPU-free gates both Qwen3 checks share, one copy of the oracle's pins and references.
pub const env = checks.env;
pub const widen = checks.widen;
pub const readJson = checks.readJson;
pub const idsOf = checks.idsOf;
pub const number = checks.number;
pub const cosine = checks.cosine;
pub const embed_inputs = checks.embed_inputs;
pub const checkInputs = checks.checkInputs;
pub const checkReferences = checks.checkReferences;
pub const checkFiles = checks.checkFiles;
pub const groupEnds = checks.groupEnds;
pub const roundUp = checks.roundUp;

test "Qwen3-Embedding: the card's and the long inputs batched equal them alone, within the BF16 bar of fp32" {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const dir = try env(a, "TF_QWEN3_EMBED_DIR") orelse return error.SkipZigTest;
    const ref = try readJson(a, try env(a, "TF_QWEN3_EMBED_REF") orelse return error.SkipZigTest);
    const bar = try readJson(a, try env(a, "TF_QWEN3_EMBED_BAR") orelse return error.SkipZigTest);
    try checkReferences(ref, bar);
    try checkInputs(ref, &embed_inputs);
    try checkFiles(a, t.io, dir, ref.object.get("pins").?);
    var g: Gpu = undefined;
    try g.open(4 << 30);
    defer g.close();
    // the card's 2 queries and 2 documents, then 5 long inputs of 2K to 32K tokens, each vector full width
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
    var m = try qwen3.Model.load(gpa, t.io, &g.r, dir);
    defer m.deinit();
    for (refs, bars) |r, b| {
        try t.expectEqual(m.spec.hidden, r.array.items.len);
        try t.expectEqual(m.spec.hidden, b.array.items.len);
    }
    const hidden = m.spec.hidden;
    var bar_worst: f64 = 0;
    for (bars, refs) |b, r| {
        const bv = try a.alloc(f64, hidden);
        const rv = try a.alloc(f64, hidden);
        for (b.array.items, r.array.items, bv, rv) |x, y, *p, *q| {
            p.* = number(x);
            q.* = number(y);
        }
        bar_worst = @max(bar_worst, 1 - cosine(bv, rv));
    }
    try t.expect(std.math.isFinite(bar_worst) and bar_worst > 0);
    const limit = @min(0.001, roundUp(bar_worst));
    var first: usize = 0;
    for (try groupEnds(a, lengths, window)) |last| {
        defer first = last;
        var ids: std.ArrayList(i32) = .empty;
        const ends = try a.alloc(u32, last - first);
        for (inputs[first..last], ends) |one, *e| {
            try ids.appendSlice(a, try idsOf(a, one));
            e.* = @intCast(ids.items.len);
        }
        const batch = try g.lastRows(&m, ids.items, ends);
        defer gpa.free(batch);
        var begin: u32 = 0;
        for (ends, 0..) |e, j| {
            defer begin = e;
            const i = first + j;
            const alone = try g.lastRows(&m, ids.items[begin..e], &.{e - begin});
            defer gpa.free(alone);
            try t.expectEqualSlices(u16, batch[j * hidden ..][0..hidden], alone);
            const got = try a.alloc(f64, hidden);
            const want = try a.alloc(f64, hidden);
            for (batch[j * hidden ..][0..hidden], refs[i].array.items, got, want) |bits, r, *p, *q| {
                p.* = widen(bits);
                q.* = number(r);
            }
            const distance = 1 - cosine(got, want);
            std.debug.print("embedding input {d} ({d} tokens): cosine distance {e:.3} (limit {e:.3})\n", .{ i, e - begin, distance, limit });
            try t.expect(std.math.isFinite(distance) and distance <= limit);
        }
    }
}

test "Qwen3-Reranker: each case's yes - no logit from lastRows and lm_head rows matches transformers" {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const dir = try env(a, "TF_QWEN3_RERANK_DIR") orelse return error.SkipZigTest;
    const ref = try readJson(a, try env(a, "TF_QWEN3_RERANK_REF") orelse return error.SkipZigTest);
    var g: Gpu = undefined;
    try g.open(8 << 30);
    defer g.close();
    var m = try qwen3.Model.load(gpa, t.io, &g.r, dir);
    defer m.deinit();
    const yes: u32 = @intCast(ref.object.get("yes").?.integer);
    const no: u32 = @intCast(ref.object.get("no").?.integer);
    const head = try qwen3.readRows(gpa, t.io, dir, m.head, &.{ yes, no });
    defer gpa.free(head);
    const hidden = m.spec.hidden;
    var worst_diff: f64 = 0;
    var worst_score: f64 = 0;
    var cases: usize = 0;
    for (ref.object.get("cases").?.array.items) |c| {
        const want_diff = c.object.get("diff") orelse continue; // ids-only cases carry no score
        const ids = try idsOf(a, c.object.get("ids").?);
        const row = try g.lastRows(&m, ids, &.{@intCast(ids.len)});
        defer gpa.free(row);
        var ly: f64 = 0;
        var ln: f64 = 0;
        for (row, head[0..hidden], head[hidden..][0..hidden]) |bits, wy, wn| {
            ly += widen(bits) * wy;
            ln += widen(bits) * wn;
        }
        const diff = ly - ln;
        const score = 1 / (1 + @exp(-diff));
        const ref_score = number(c.object.get("score").?);
        // a NaN would slip through @max: every value must be finite before it counts
        for ([_]f64{ diff, score, number(want_diff), ref_score }) |v| try t.expect(std.math.isFinite(v));
        const d = @abs(diff - number(want_diff));
        const ds = @abs(score - ref_score);
        std.debug.print("rerank {s} {d} tokens: diff {d:.4} ref {d:.4} score {d:.6} ref {d:.6}\n", .{ c.object.get("kind").?.string, ids.len, diff, number(want_diff), score, number(c.object.get("score").?) });
        worst_diff = @max(worst_diff, d);
        worst_score = @max(worst_score, ds);
        cases += 1;
    }
    std.debug.print("rerank: {d} cases, worst |diff| {d:.4} (gate 0.25), worst |score| {d:.5} (gate 0.02)\n", .{ cases, worst_diff, worst_score });
    try t.expect(cases >= 32);
    try t.expect(worst_diff <= 0.25);
    try t.expect(worst_score <= 0.02);
}

test {
    _ = @import("projection_test.zig");
    _ = @import("decide_test.zig");
}
