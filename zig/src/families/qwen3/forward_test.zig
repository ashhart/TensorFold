//! lastRows on the GPU against transformers: the embedding's pooled rows and the reranker's yes/no logits.

// Skipped without TF_QWEN3_EMBED_DIR / _REF / _BAR (fp32, BF16 oracles) or TF_QWEN3_RERANK_DIR / _REF.
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
        g.l = try hip.Launcher.load(&g.r, .{}, hip.kernels.images);
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
        try qwen3.lastRows(g.ops(), m, @intFromPtr(dev.ptr), ends, .{ .ptr = @intFromPtr(out.ptr), .kind = .bf16 });
        try g.stream.synchronize();
        const rows = try gpa.alloc(u16, ends.len * hidden);
        errdefer gpa.free(rows);
        try out.download(0, std.mem.sliceAsBytes(rows));
        return rows;
    }
};

fn env(a: std.mem.Allocator, key: []const u8) !?[]u8 {
    return t.environ.getAlloc(a, key) catch |err| switch (err) {
        error.EnvironmentVariableMissing => null,
        else => err,
    };
}

fn widen(bits: u16) f64 {
    return @as(f32, @bitCast(@as(u32, bits) << 16));
}

fn readJson(a: std.mem.Allocator, path: []const u8) !std.json.Value {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(t.io, path, a, .limited(256 << 20));
    return std.json.parseFromSliceLeaky(std.json.Value, a, bytes, .{});
}

fn idsOf(a: std.mem.Allocator, v: std.json.Value) ![]i32 {
    const out = try a.alloc(i32, v.array.items.len);
    for (v.array.items, out) |x, *o| o.* = @intCast(x.integer);
    return out;
}

fn number(v: std.json.Value) f64 {
    return switch (v) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        else => std.math.nan(f64),
    };
}

fn cosine(x: []const f64, y: []const f64) f64 {
    var xy: f64 = 0;
    var xx: f64 = 0;
    var yy: f64 = 0;
    for (x, y) |p, q| {
        xy += p * q;
        xx += p * p;
        yy += q * q;
    }
    return xy / @sqrt(xx * yy);
}

/// The pins tools/zig/oracle_qwen3_embedding.py writes into each reference: the checkpoint and the Transformers tree.
const embed_revision = "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af";
const transformers_tree = "b08cead6d1b841ef9723106a81a92ec37d67aff3e43a198e56b7dc76538d377b";

fn text(v: std.json.Value, key: []const u8) ?[]const u8 {
    const x = (if (v == .object) v.object.get(key) else null) orelse return null;
    return if (x == .string) x.string else null;
}

fn same(expected: []const u8, got: ?[]const u8) !void {
    if (!std.mem.eql(u8, expected, got orelse "")) return error.ReferenceMismatch;
}

/// A silent check: a refused reference is an error, without the expect helpers' diff on stderr.
fn holds(ok: bool) !void {
    if (!ok) return error.ReferenceMismatch;
}

/// Refuses, before any GPU work, references that are not the fp32 target and BF16 bar of the same pinned inputs.
fn checkReferences(ref: std.json.Value, bar: std.json.Value) !void {
    try same("float32", text(ref, "dtype"));
    try same("bfloat16", text(bar, "dtype"));
    // the oracle checks every hash of the pin record; here both references carry one complete, identical record
    for ([_]std.json.Value{ ref, bar }) |r| {
        const pins = r.object.get("pins") orelse return error.UnpinnedReference;
        try same("Qwen/Qwen3-Embedding-8B", text(pins, "repo"));
        try same(embed_revision, text(pins, "revision"));
        try same("5.22.1", text(pins, "transformers"));
        try same(transformers_tree, text(pins, "transformers_tree"));
        for ([_][]const u8{ "checkpoint", "weights", "library" }) |key| {
            const files = pins.object.get(key) orelse return error.UnpinnedReference;
            try holds(files == .object and files.object.count() > 0);
            for (files.object.values()) |hash| try holds(hash == .string and hash.string.len == 64);
        }
    }
    try holds(jsonEqual(ref.object.get("pins").?, bar.object.get("pins").?));
    for ([_][]const u8{ "texts", "ids" }) |key| {
        const x = ref.object.get(key) orelse return error.IncompleteReference;
        const y = bar.object.get(key) orelse return error.IncompleteReference;
        try holds(std.meta.activeTag(x) == .array and std.meta.activeTag(y) == .array and x.array.items.len == y.array.items.len);
        for (x.array.items, y.array.items) |p, q| try holds(jsonEqual(p, q));
    }
    for ([_]std.json.Value{ ref, bar }) |r| for (r.object.get("vectors").?.array.items) |v| {
        var square: f64 = 0;
        for (v.array.items) |x| {
            const f = number(x);
            try holds(std.math.isFinite(f));
            square += f * f;
        }
        try holds(square > 0);
    };
}

fn jsonEqual(p: std.json.Value, q: std.json.Value) bool {
    if (std.meta.activeTag(p) != std.meta.activeTag(q)) return false;
    return switch (p) {
        .string => |x| std.mem.eql(u8, x, q.string),
        .integer => |x| x == q.integer,
        .array => |x| x.items.len == q.array.items.len and for (x.items, q.array.items) |a, b| {
            if (!jsonEqual(a, b)) break false;
        } else true,
        .object => |x| x.count() == q.object.count() and for (x.keys(), x.values()) |k, v| {
            const w = q.object.get(k) orelse break false;
            if (!jsonEqual(v, w)) break false;
        } else true,
        else => false,
    };
}

test "Qwen3-Embedding references: a bar of other inputs, an unpinned one or a non-finite vector is refused" {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const good =
        \\{"dtype": "float32", "pins": {"repo": "Qwen/Qwen3-Embedding-8B",
        \\ "revision": "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af", "transformers": "5.22.1",
        \\ "transformers_tree": "b08cead6d1b841ef9723106a81a92ec37d67aff3e43a198e56b7dc76538d377b",
        \\ "checkpoint": {"config.json": "dcccf7c7890c8debc4af8dace8d6acd8dd50bd56d50d4d8b60529381ba3de3d7"},
        \\ "weights": {"model-00001-of-00004.safetensors": "99b343597fe840706146144699a8b9188dd3387e43eb61faf0231b70b249d451"},
        \\ "library": {"models/qwen3/modeling_qwen3.py": "63a6213c15d6fb04e58a1c64fd05a4cdfb49476443fbeef524f0e0486043824b"}},
        \\ "texts": ["a"], "ids": [[1, 2]], "vectors": [[0.6, 0.8]]}
    ;
    const ref = try std.json.parseFromSliceLeaky(std.json.Value, a, good, .{});
    const bar_text = try std.mem.replaceOwned(u8, a, good, "float32", "bfloat16");
    try checkReferences(ref, try std.json.parseFromSliceLeaky(std.json.Value, a, bar_text, .{}));
    const bad = [_][2][]const u8{
        .{ "[[1, 2]]", "[[1, 3]]" },
        .{ "\"texts\": [\"a\"]", "\"texts\": [\"b\"]" },
        .{ "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af", "0000000000000000000000000000000000000000" },
        .{ "[[0.6, 0.8]]", "[[0.0, 0.0]]" },
        .{ "99b343597fe840706146144699a8b9188dd3387e43eb61faf0231b70b249d451", "00b343597fe840706146144699a8b9188dd3387e43eb61faf0231b70b249d451" },
        .{ "\"weights\": {\"model-00001-of-00004.safetensors\": \"99b343597fe840706146144699a8b9188dd3387e43eb61faf0231b70b249d451\"},", "" },
        .{ "\"transformers\": \"5.22.1\"", "\"transformers\": \"5.22.0\"" },
    };
    for (bad) |swap| {
        const other = try std.mem.replaceOwned(u8, a, bar_text, swap[0], swap[1]);
        try t.expect(std.meta.isError(checkReferences(ref, try std.json.parseFromSliceLeaky(std.json.Value, a, other, .{}))));
    }
}

test "Qwen3-Embedding: the card's inputs batched equal them alone, and sit within the BF16 bar of the fp32 vectors" {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const dir = try env(a, "TF_QWEN3_EMBED_DIR") orelse return error.SkipZigTest;
    const ref = try readJson(a, try env(a, "TF_QWEN3_EMBED_REF") orelse return error.SkipZigTest);
    const bar = try readJson(a, try env(a, "TF_QWEN3_EMBED_BAR") orelse return error.SkipZigTest);
    try checkReferences(ref, bar);
    var g: Gpu = undefined;
    try g.open(4 << 30);
    defer g.close();
    // the card's 2 queries and 2 documents in both references, each vector full width, before any GPU work
    const inputs = ref.object.get("ids").?.array.items;
    const refs = ref.object.get("vectors").?.array.items;
    const bars = bar.object.get("vectors").?.array.items;
    try t.expectEqual(@as(usize, 4), inputs.len);
    try t.expectEqual(@as(usize, 4), refs.len);
    try t.expectEqual(@as(usize, 4), bars.len);
    for (inputs) |one| try t.expect(one.array.items.len > 0);
    var m = try qwen3.Model.load(gpa, t.io, &g.r, dir);
    defer m.deinit();
    for (refs, bars) |r, b| {
        try t.expectEqual(m.spec.hidden, r.array.items.len);
        try t.expectEqual(m.spec.hidden, b.array.items.len);
    }
    var ids: std.ArrayList(i32) = .empty;
    const ends = try a.alloc(u32, inputs.len);
    for (inputs, ends) |one, *e| {
        try ids.appendSlice(a, try idsOf(a, one));
        e.* = @intCast(ids.items.len);
    }
    const batch = try g.lastRows(&m, ids.items, ends);
    defer gpa.free(batch);
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
    try t.expect(std.math.isFinite(bar_worst));
    const limit = @min(0.001, 2 * bar_worst);
    var begin: u32 = 0;
    for (ends, 0..) |e, i| {
        const alone = try g.lastRows(&m, ids.items[begin..e], &.{e - begin});
        defer gpa.free(alone);
        try t.expectEqualSlices(u16, batch[i * hidden ..][0..hidden], alone);
        const got = try a.alloc(f64, hidden);
        const want = try a.alloc(f64, hidden);
        for (batch[i * hidden ..][0..hidden], refs[i].array.items, got, want) |bits, r, *p, *q| {
            p.* = widen(bits);
            q.* = number(r);
        }
        const distance = 1 - cosine(got, want);
        std.debug.print("embedding input {d}: cosine distance {e:.3} (limit {e:.3})\n", .{ i, distance, limit });
        try t.expect(std.math.isFinite(distance) and distance <= limit);
        begin = e;
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
