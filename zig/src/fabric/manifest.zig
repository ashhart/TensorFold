//! A handoff's manifest (kv-handoff.md): what each exported layer's page rows hold, written as MCDMA's producer writes it.
const std = @import("std");
const wire = @import("wire.zig");
const Allocator = std.mem.Allocator;

/// What a dimension of a layer's page array holds.
pub const Dim = enum { block, token, head, head_dim, kv, kv_head_dim, latent };

pub const Dtype = enum {
    bfloat16,
    float16,
    float32,

    pub fn size(d: Dtype) u64 {
        return if (d == .float32) 4 else 2;
    }
};

pub const Kind = enum { attention, mla };

/// One layer's exported page array: `shape` with the block dimension cut to the exported rows.
pub const Layer = struct {
    index: u32,
    kind: Kind,
    shape: []const u64,
    dims: []const Dim,
    dtype: Dtype,
    heads: u64 = 0,
    total_heads: u64 = 0,
    head_size: u64 = 0,
    latent_size: u64 = 0,
    rope_size: u64 = 0,

    pub fn blockDim(l: Layer) usize {
        return std.mem.indexOfScalar(Dim, l.dims, .block).?;
    }

    pub fn rows(l: Layer) u64 {
        return l.shape[l.blockDim()];
    }

    /// Bytes of one page row: every dimension but the block one, times the element size.
    pub fn rowBytes(l: Layer) u64 {
        var n: u64 = l.dtype.size();
        for (l.shape, l.dims) |size, dim| {
            if (dim != .block) n *= size;
        }
        return n;
    }
};

pub const Manifest = struct {
    handoff: [16]u8,
    model: []const u8,
    prompt_tokens: u64,
    first_token: u64,
    token_sha256: [64]u8,
    block_size: u64,
    tp_rank: u64,
    tp_size: u64,
    layers: []const Layer,
    frames: u64,

    pub fn layerByIndex(m: Manifest, index: u32) ?*const Layer {
        for (m.layers) |*l| {
            if (l.index == index) return l;
        }
        return null;
    }
};

pub const Error = error{ BadManifest, OutOfMemory, WriteFailed };

/// json.dumps(manifest) with Python's defaults (", " and ": ", ASCII escapes) in export.py's key order.
pub fn write(m: Manifest, w: *std.Io.Writer) Error!void {
    try w.print("{{\"protocol\": {d}, \"handoff\": \"{s}\", \"model\": ", .{ wire.version, std.fmt.bytesToHex(m.handoff, .lower) });
    try writeString(w, m.model);
    try w.print(", \"prompt_tokens\": {d}, \"first_token\": {d}, \"token_sha256\": \"{s}\", \"block_size\": {d}, \"tp_rank\": {d}, \"tp_size\": {d}, \"layers\": [", .{ m.prompt_tokens, m.first_token, m.token_sha256, m.block_size, m.tp_rank, m.tp_size });
    for (m.layers, 0..) |l, i| {
        if (i > 0) try w.writeAll(", ");
        try w.print("{{\"index\": {d}, \"kind\": \"{s}\", \"shape\": [", .{ l.index, @tagName(l.kind) });
        for (l.shape, 0..) |s, j| try w.print("{s}{d}", .{ if (j > 0) ", " else "", s });
        try w.writeAll("], \"dims\": [");
        for (l.dims, 0..) |d, j| try w.print("{s}\"{s}\"", .{ if (j > 0) ", " else "", @tagName(d) });
        try w.print("], \"dtype\": \"{s}\", ", .{@tagName(l.dtype)});
        if (l.kind == .mla) {
            try w.print("\"latent_size\": {d}, \"rope_size\": {d}}}", .{ l.latent_size, l.rope_size });
        } else {
            try w.print("\"heads\": {d}, \"total_heads\": {d}, \"head_size\": {d}}}", .{ l.heads, l.total_heads, l.head_size });
        }
    }
    try w.print("], \"frames\": {d}}}", .{m.frames});
}

/// A JSON string as Python's json.dumps escapes it with ensure_ascii.
fn writeString(w: *std.Io.Writer, s: []const u8) Error!void {
    const view = std.unicode.Utf8View.init(s) catch return error.BadManifest;
    try w.writeByte('"');
    var it = view.iterator();
    while (it.nextCodepoint()) |cp| switch (cp) {
        '"' => try w.writeAll("\\\""),
        '\\' => try w.writeAll("\\\\"),
        '\n' => try w.writeAll("\\n"),
        '\r' => try w.writeAll("\\r"),
        '\t' => try w.writeAll("\\t"),
        8 => try w.writeAll("\\b"),
        12 => try w.writeAll("\\f"),
        0x20...0x21, 0x23...0x5B, 0x5D...0x7E => try w.writeByte(@intCast(cp)),
        0x10000...0x10FFFF => {
            const c = cp - 0x10000;
            try w.print("\\u{x:0>4}\\u{x:0>4}", .{ 0xD800 + (c >> 10), 0xDC00 + (c & 0x3FF) });
        },
        else => try w.print("\\u{x:0>4}", .{cp}),
    };
    try w.writeByte('"');
}

/// Parse and check a manifest; strings and arrays live in `arena`.
pub fn parse(arena: Allocator, bytes: []const u8) Error!Manifest {
    const root = std.json.parseFromSliceLeaky(std.json.Value, arena, bytes, .{}) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return error.BadManifest,
    };
    const o = asObject(root) orelse return error.BadManifest;
    if (try int(o, "protocol") != wire.version) return error.BadManifest;
    var m: Manifest = .{
        .handoff = undefined,
        .model = try str(o, "model"),
        .prompt_tokens = try int(o, "prompt_tokens"),
        .first_token = try int(o, "first_token"),
        .token_sha256 = undefined,
        .block_size = try int(o, "block_size"),
        .tp_rank = try int(o, "tp_rank"),
        .tp_size = try int(o, "tp_size"),
        .layers = &.{},
        .frames = try int(o, "frames"),
    };
    const id = try str(o, "handoff");
    if (id.len != 32) return error.BadManifest;
    _ = std.fmt.hexToBytes(&m.handoff, id) catch return error.BadManifest;
    const digest = try str(o, "token_sha256");
    if (digest.len != 64) return error.BadManifest;
    for (digest) |c| if (!std.ascii.isDigit(c) and !(c >= 'a' and c <= 'f')) return error.BadManifest;
    m.token_sha256 = digest[0..64].*;
    if (m.block_size == 0 or m.tp_size == 0 or m.tp_rank >= m.tp_size or m.first_token > m.prompt_tokens) return error.BadManifest;
    const list = (o.get("layers") orelse return error.BadManifest);
    if (list != .array) return error.BadManifest;
    const layers = try arena.alloc(Layer, list.array.items.len);
    for (list.array.items, layers) |item, *l| l.* = try parseLayer(arena, item);
    m.layers = layers;
    return m;
}

fn parseLayer(arena: Allocator, v: std.json.Value) Error!Layer {
    const o = asObject(v) orelse return error.BadManifest;
    const shape_v = o.get("shape") orelse return error.BadManifest;
    const dims_v = o.get("dims") orelse return error.BadManifest;
    if (shape_v != .array or dims_v != .array or shape_v.array.items.len != dims_v.array.items.len) return error.BadManifest;
    const shape = try arena.alloc(u64, shape_v.array.items.len);
    const dims = try arena.alloc(Dim, dims_v.array.items.len);
    var blocks: usize = 0;
    for (shape_v.array.items, dims_v.array.items, shape, dims) |sv, dv, *s, *d| {
        if (sv != .integer or sv.integer < 0 or dv != .string) return error.BadManifest;
        s.* = @intCast(sv.integer);
        d.* = std.meta.stringToEnum(Dim, dv.string) orelse return error.BadManifest;
        blocks += @intFromBool(d.* == .block);
    }
    const kind = std.meta.stringToEnum(Kind, try str(o, "kind")) orelse return error.BadManifest;
    if (blocks != 1 or (kind == .mla) != (std.mem.indexOfScalar(Dim, dims, .latent) != null)) return error.BadManifest;
    const index = try int(o, "index");
    if (index > std.math.maxInt(u32)) return error.BadManifest;
    var l: Layer = .{
        .index = @intCast(index),
        .kind = kind,
        .shape = shape,
        .dims = dims,
        .dtype = std.meta.stringToEnum(Dtype, try str(o, "dtype")) orelse return error.BadManifest,
    };
    if (kind == .mla) {
        l.latent_size = try int(o, "latent_size");
        l.rope_size = try int(o, "rope_size");
    } else {
        l.heads = try int(o, "heads");
        l.total_heads = try int(o, "total_heads");
        l.head_size = try int(o, "head_size");
    }
    return l;
}

fn asObject(v: std.json.Value) ?std.json.ObjectMap {
    return if (v == .object) v.object else null;
}

fn int(o: std.json.ObjectMap, key: []const u8) Error!u64 {
    const v = o.get(key) orelse return error.BadManifest;
    if (v != .integer or v.integer < 0) return error.BadManifest;
    return @intCast(v.integer);
}

fn str(o: std.json.ObjectMap, key: []const u8) Error![]const u8 {
    const v = o.get(key) orelse return error.BadManifest;
    return if (v == .string) v.string else error.BadManifest;
}

test "a manifest writes byte for byte as json.dumps and parses back" {
    const gpa = std.testing.allocator;
    const layers = [_]Layer{
        .{ .index = 0, .kind = .attention, .shape = &.{ 3, 2, 16, 16 }, .dims = &.{ .block, .head, .token, .kv_head_dim }, .dtype = .bfloat16, .heads = 2, .total_heads = 2, .head_size = 8 },
        .{ .index = 5, .kind = .mla, .shape = &.{ 2, 64, 576 }, .dims = &.{ .block, .token, .latent }, .dtype = .bfloat16, .latent_size = 512, .rope_size = 64 },
    };
    const m: Manifest = .{ .handoff = @splat(0xab), .model = "org/m\u{e9}\u{1f600}\"", .prompt_tokens = 40, .first_token = 0, .token_sha256 = wire.tokenSha256(&.{ 1, 2 }), .block_size = 16, .tp_rank = 1, .tp_size = 2, .layers = &layers, .frames = 3 };
    var out: std.Io.Writer.Allocating = .init(gpa);
    defer out.deinit();
    try write(m, &out.writer);
    const text = out.written();
    try std.testing.expect(std.mem.startsWith(u8, text, "{\"protocol\": 1, \"handoff\": \"abababab"));
    try std.testing.expect(std.mem.indexOf(u8, text, "\"model\": \"org/m\\u00e9\\ud83d\\ude00\\\"\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "\"shape\": [3, 2, 16, 16], \"dims\": [\"block\", \"head\", \"token\", \"kv_head_dim\"]") != null);
    try std.testing.expect(std.mem.endsWith(u8, text, "\"latent_size\": 512, \"rope_size\": 64}], \"frames\": 3}"));
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const back = try parse(arena.allocator(), text);
    try std.testing.expectEqualStrings("org/m\u{e9}\u{1f600}\"", back.model);
    try std.testing.expectEqual(@as(u64, 1024), back.layers[0].rowBytes());
    try std.testing.expectEqual(@as(u64, 3), back.layers[0].rows());
    try std.testing.expectEqual(@as(u64, 64 * 576 * 2), back.layers[1].rowBytes());
    try std.testing.expectEqual(m.handoff, back.handoff);
}

const zeros64 = "0000000000000000000000000000000000000000000000000000000000000000";

test "manifests that break the schema are refused" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try std.testing.expectError(error.BadManifest, parse(a, "[]"));
    try std.testing.expectError(error.BadManifest, parse(a, "{\"protocol\": 2}"));
    const two_blocks = "{\"protocol\": 1, \"handoff\": \"00000000000000000000000000000000\", \"model\": \"m\", \"prompt_tokens\": 1, \"first_token\": 0, \"token_sha256\": \"" ++ zeros64 ++ "\", \"block_size\": 16, \"tp_rank\": 0, \"tp_size\": 1, \"layers\": [{\"index\": 0, \"kind\": \"attention\", \"shape\": [1, 1], \"dims\": [\"block\", \"block\"], \"dtype\": \"bfloat16\", \"heads\": 1, \"total_heads\": 1, \"head_size\": 1}], \"frames\": 1}";
    try std.testing.expectError(error.BadManifest, parse(a, two_blocks));
}
