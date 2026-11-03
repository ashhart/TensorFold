//! lastRows on the GPU against transformers: the embedding's pooled rows and the reranker's yes/no logits.

// Skipped without TF_QWEN3_EMBED_DIR / _REF / _BAR or TF_QWEN3_RERANK_DIR / _REF; TF_QWEN3_MATRIX picks matrix=.
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
        const matrix = try matrixChoice();
        g.l = try hip.Launcher.load(&g.r, .{ .products = .{ .matrix = matrix } }, hip.kernels.images);
        std.debug.print("qwen3 GPU checks: matrix={t}, dense products on {t}\n", .{ matrix, g.l.dense.serve });
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

pub fn env(a: std.mem.Allocator, key: []const u8) !?[]u8 {
    return t.environ.getAlloc(a, key) catch |err| switch (err) {
        error.EnvironmentVariableMissing => null,
        else => err,
    };
}

fn widen(bits: u16) f64 {
    return @as(f32, @bitCast(@as(u32, bits) << 16));
}

pub fn readJson(a: std.mem.Allocator, path: []const u8) !std.json.Value {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(t.io, path, a, .limited(256 << 20));
    return std.json.parseFromSliceLeaky(std.json.Value, a, bytes, .{});
}

pub fn idsOf(a: std.mem.Allocator, v: std.json.Value) ![]i32 {
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

/// One fidelity input as the oracle's INPUTS pins it: tokens, SHA256 of the UTF-8 text, SHA256 of the IDs as LE u32.
const InputPin = struct { tokens: usize, text: *const [64]u8, ids: *const [64]u8 };

/// The card's 4 inputs, then 5 long inputs of 2K to 32K tokens cut from pinned files.
const embed_inputs = [_]InputPin{
    .{ .tokens = 27, .text = "b11f8d3ae64c3c7029275c5d8d82172ef31a9a354223c12845b61047a0c972e8", .ids = "3c1537f6fc0a496f2b68c1aaa38484d8a82030338ae52bbd955b45b50c9efc24" },
    .{ .tokens = 23, .text = "bd6e4e8bb1b5c2e15db6003175c53f084690aeb6c09b34f1808161ef3353d524", .ids = "a08f60368f1da49a05dd929f406c503bcd0864ee464f25807866753bffe3062c" },
    .{ .tokens = 8, .text = "b6d0a6ed3a94d5e532ae83127502407102e51342d1db011caf4c5b9808a1e131", .ids = "c1e33d8973efc767e95f15a7b31af3c68eccc5ca89e0c2648f5337fec3c8f2c6" },
    .{ .tokens = 31, .text = "ba5bd6b5de9d9f827336e76d982b5dbae8aa3cc075c115f57405ba370bba8dbc", .ids = "4bd9685251a5e3587a645c0bf96dbc76ca70998073523b5537a1513c15f4c8ea" },
    .{ .tokens = 2044, .text = "f60c57fd5911f2c10704c555aef3a6e38543afa78604385ad5d8dde87050e0ff", .ids = "e10c109c63916e2cacd0143d8295fa5fa449c9eb09b7d3126405bfec9a6c9d92" },
    .{ .tokens = 4043, .text = "93c6978d85521f7b15d47dfedd386b72f063fd25ea5ada5ccf38fefdc88e7766", .ids = "a87db7cfe361a40e15c9b209796f6260bfb74017bc1e897e6e5be3d1a557cdbf" },
    .{ .tokens = 8177, .text = "3f5c76527fb4d032b31c827e5ec4d171a9feb94e7ab2ab96341028b1dd412505", .ids = "17c5b8a1cb0f288b40b9810549881b901eea58c83edb6e6d1b89ac1beab603eb" },
    .{ .tokens = 16383, .text = "c0a043846083a03ec205a34af84d7b1053a8d1a4ea4f3d3c028bb4c048ef1366", .ids = "528fda3905d2fc0340f8d5ca34f4bd91703613d76b40e5ac0835fab641acf173" },
    .{ .tokens = 32759, .text = "009256a9df975a6f67e5984760078c1dbcb0bd0b9aefbce0bdb8373ebd386dd7", .ids = "1dfcec470654b993960be543f8e84c461bc28d743aa5a50ee63bd81a9b8e15a8" },
};

/// Refuses references whose texts and token IDs are not exactly the pinned inputs, in order.
fn checkInputs(ref: std.json.Value, pins: []const InputPin) !void {
    const texts = ref.object.get("texts") orelse return error.IncompleteReference;
    const ids = ref.object.get("ids") orelse return error.IncompleteReference;
    try holds(texts == .array and ids == .array and texts.array.items.len == pins.len and ids.array.items.len == pins.len);
    for (texts.array.items, ids.array.items, pins) |x, one, pin| {
        try holds(x == .string and one == .array and one.array.items.len == pin.tokens);
        var h = std.crypto.hash.sha2.Sha256.init(.{});
        for (one.array.items) |id| {
            try holds(id == .integer and id.integer >= 0 and id.integer <= std.math.maxInt(u32));
            var le: [4]u8 = undefined;
            std.mem.writeInt(u32, &le, @intCast(id.integer), .little);
            h.update(&le);
        }
        try holds(std.mem.eql(u8, &std.fmt.bytesToHex(h.finalResult(), .lower), pin.ids));
        var th: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(x.string, &th, .{});
        try holds(std.mem.eql(u8, &std.fmt.bytesToHex(th, .lower), pin.text));
    }
}

test "Qwen3-Embedding references: inputs other than the pinned ones are refused" {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    // "a" and the IDs [1, 2] (bytes 01 00 00 00 02 00 00 00)
    const pin = [_]InputPin{.{ .tokens = 2, .text = "ca978112ca1bbdcafac231b39a23dc4da786eff8147c4e72b9807785afee48bb", .ids = "34fb5c825de7ca4aea6e712f19d439c1da0c92c37b423936c5f618545ca4fa1f" }};
    const parse = struct {
        fn of(al: std.mem.Allocator, body: []const u8) !std.json.Value {
            return std.json.parseFromSliceLeaky(std.json.Value, al, body, .{});
        }
    }.of;
    try checkInputs(try parse(a, "{\"texts\": [\"a\"], \"ids\": [[1, 2]]}"), &pin);
    for ([_][]const u8{
        "{\"texts\": [\"b\"], \"ids\": [[1, 2]]}",
        "{\"texts\": [\"a\"], \"ids\": [[1, 3]]}",
        "{\"texts\": [\"a\"], \"ids\": [[1]]}",
        "{\"texts\": [\"a\", \"a\"], \"ids\": [[1, 2], [1, 2]]}",
        "{\"texts\": [\"a\"], \"ids\": [[1, -2]]}",
        "{\"texts\": [\"a\"]}",
    }) |body| try t.expect(std.meta.isError(checkInputs(try parse(a, body), &pin)));
    try t.expectEqual(@as(usize, 9), embed_inputs.len);
}

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

/// The SHA256 of a file, as lowercase hex.
fn fileSha256(io: std.Io, path: []const u8) ![64]u8 {
    const f = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer f.close(io);
    const size = (try f.stat(io)).size;
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    var buf: [1 << 22]u8 = undefined;
    var at: u64 = 0;
    while (at < size) {
        const n = try f.readPositionalAll(io, buf[0..@intCast(@min(buf.len, size - at))], at);
        if (n == 0) return error.ShortRead;
        h.update(buf[0..n]);
        at += n;
    }
    return std.fmt.bytesToHex(h.finalResult(), .lower);
}

/// The references' pinned files are the checkpoint the test loads: each one's SHA256, and every indexed shard pinned.
fn checkFiles(a: std.mem.Allocator, io: std.Io, dir: []const u8, pins: std.json.Value) !void {
    for ([_][]const u8{ "checkpoint", "weights" }) |key| {
        const files = pins.object.get(key).?.object;
        for (files.keys(), files.values()) |name, hash| {
            const got = try fileSha256(io, try std.fs.path.join(a, &.{ dir, name }));
            try holds(std.mem.eql(u8, &got, hash.string));
        }
    }
    const index = try readJson(a, try std.fs.path.join(a, &.{ dir, "model.safetensors.index.json" }));
    const shards = index.object.get("weight_map").?.object;
    const weights = pins.object.get("weights").?.object;
    for (shards.values()) |file| try holds(file == .string and weights.get(file.string) != null);
    try holds(pins.object.get("checkpoint").?.object.get("model.safetensors.index.json") != null);
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

test "Qwen3-Embedding references: pins that do not hash to the checkpoint's files are refused, whoever made both" {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(t.io, .{ .sub_path = "s.safetensors", .data = "weights" });
    try tmp.dir.writeFile(t.io, .{ .sub_path = "model.safetensors.index.json", .data = "{\"weight_map\": {\"x\": \"s.safetensors\"}}" });
    const dir = try tmp.dir.realPathFileAlloc(t.io, ".", a);
    const shard = try fileSha256(t.io, try std.fs.path.join(a, &.{ dir, "s.safetensors" }));
    const index = try fileSha256(t.io, try std.fs.path.join(a, &.{ dir, "model.safetensors.index.json" }));
    const good = try std.fmt.allocPrint(a, "{{\"checkpoint\": {{\"model.safetensors.index.json\": \"{s}\"}}, \"weights\": {{\"s.safetensors\": \"{s}\"}}}}", .{ &index, &shard });
    try checkFiles(a, t.io, dir, try std.json.parseFromSliceLeaky(std.json.Value, a, good, .{}));
    const wrong = try std.mem.replaceOwned(u8, a, good, &shard, "0000000000000000000000000000000000000000000000000000000000000000");
    try t.expect(std.meta.isError(checkFiles(a, t.io, dir, try std.json.parseFromSliceLeaky(std.json.Value, a, wrong, .{}))));
    const unpinned = try std.fmt.allocPrint(a, "{{\"checkpoint\": {{\"model.safetensors.index.json\": \"{s}\"}}, \"weights\": {{}}}}", .{&index});
    try t.expect(std.meta.isError(checkFiles(a, t.io, dir, try std.json.parseFromSliceLeaky(std.json.Value, a, unpinned, .{}))));
}

/// The oracle's requests: consecutive inputs, at most 128 and `window` tokens in all (`ids` lengths in order).
fn groupEnds(a: std.mem.Allocator, lengths: []const usize, window: usize) ![]usize {
    var out: std.ArrayList(usize) = .empty;
    var used: usize = 0;
    var count: usize = 0;
    for (lengths, 0..) |n, i| {
        if (n > window) return error.InputPastWindow;
        if (count > 0 and (used + n > window or count == 128)) {
            try out.append(a, i);
            used = 0;
            count = 0;
        }
        used += n;
        count += 1;
    }
    try out.append(a, lengths.len);
    return out.items;
}

test "Qwen3-Embedding: the GPU test batches inputs as the oracle's check does" {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    try t.expectEqualSlices(usize, &.{ 2, 3, 4 }, try groupEnds(a, &.{ 3, 5, 9, 1 }, 9));
    try t.expectEqualSlices(usize, &.{ 1, 3 }, try groupEnds(a, &.{ 4, 2, 2 }, 4));
    try t.expectError(error.InputPastWindow, groupEnds(a, &.{10}, 9));
}

/// `x` in [1e-20, 10) rounded up to two significant digits, as the oracle's round_up: never below x.
fn roundUp(x: f64) f64 {
    var scale: f64 = 1;
    var k = 1 - @floor(std.math.log10(x));
    while (k > 0) : (k -= 1) scale *= 10;
    var up = @ceil(x * scale);
    while ((up - 1) / scale >= x) up -= 1;
    while (up / scale < x) up += 1;
    return up / scale;
}

test "Qwen3-Embedding gate: the BF16 bar's distance rounds up to two digits" {
    const cases = [_][2]f64{
        .{ 1.572e-4, 1.6e-4 }, .{ 1.624e-4, 1.7e-4 }, .{ 1.884e-4, 1.9e-4 },
        .{ 1.6e-4, 1.6e-4 },   .{ 0.1099, 0.11 },     .{ 9.96e-5, 1.0e-4 },
    };
    for (cases) |c| {
        try t.expectApproxEqRel(c[1], roundUp(c[0]), 1e-12);
        try t.expect(roundUp(c[0]) >= c[0]);
    }
    var tens: f64 = 100;
    while (tens <= 1e9) : (tens *= 10) for (10..100) |d| {
        const x = @as(f64, @floatFromInt(d)) / tens;
        try t.expectEqual(x, roundUp(x));
        try t.expectEqual(@as(f64, @floatFromInt(d + 1)) / tens, roundUp(std.math.nextAfter(f64, x, 1)));
    };
}

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
}
