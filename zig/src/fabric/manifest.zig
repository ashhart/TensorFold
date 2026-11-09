//! A handoff's manifest: each layer's page rows (protocol 1, MCDMA's kv-handoff.md) or recurrent parts (protocol 2).
const std = @import("std");
const wire = @import("wire.zig");
const Allocator = std.mem.Allocator;

/// What a dimension of a layer's page array holds.
pub const Dim = enum { block, token, head, head_dim, kv, kv_head_dim, latent };

/// Protocol 2: what a dimension of a recurrent part holds. Its own set, so a page layer's dims stay protocol 1's.
pub const PartDim = enum { head, value, key, tap, channel };

pub const Dtype = enum {
    bfloat16,
    float16,
    float32,

    pub fn size(d: Dtype) u64 {
        return if (d == .float32) 4 else 2;
    }
};

pub const Kind = enum { attention, mla, recurrent };

/// Protocol 2: one tensor of a recurrent layer's state (a linear layer's recurrence, its convolution tail), whole.
pub const Part = struct {
    name: []const u8,
    shape: []const u64,
    dims: []const PartDim,
    dtype: Dtype,

    pub fn bytes(p: Part) u64 {
        var n: u64 = p.dtype.size();
        for (p.shape) |d| n *= d;
        return n;
    }
};

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
    parts: []const Part = &.{}, // recurrent only: its state, one frame a part, row_start naming the part

    pub fn blockDim(l: Layer) usize {
        return std.mem.indexOfScalar(Dim, l.dims, .block).?;
    }

    /// Page rows, or a recurrent layer's parts.
    pub fn rows(l: Layer) u64 {
        if (l.kind == .recurrent) return l.parts.len;
        return l.shape[l.blockDim()];
    }

    /// Bytes of one page row: every dimension but the block one, times the element size.
    pub fn rowBytes(l: Layer) u64 {
        if (l.kind == .recurrent) return 0; // parts differ in size: frameBytes
        var n: u64 = l.dtype.size();
        for (l.shape, l.dims) |size, dim| {
            if (dim != .block) n *= size;
        }
        return n;
    }

    /// A DATA frame's bytes for rows [row_start, row_start + count): whole page rows, or one recurrent part.
    pub fn frameBytes(l: Layer, row_start: u64, count: u64) ?u64 {
        if (l.kind != .recurrent) return std.math.mul(u64, count, l.rowBytes()) catch null;
        if (count != 1 or row_start >= l.parts.len) return null;
        return l.parts[@intCast(row_start)].bytes();
    }
};

pub const Manifest = struct {
    protocol: u16 = wire.version, // 2 when a layer is recurrent
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
    try w.print("{{\"protocol\": {d}, \"handoff\": \"{s}\", \"model\": ", .{ m.protocol, std.fmt.bytesToHex(m.handoff, .lower) });
    try writeString(w, m.model);
    try w.print(", \"prompt_tokens\": {d}, \"first_token\": {d}, \"token_sha256\": \"{s}\", \"block_size\": {d}, \"tp_rank\": {d}, \"tp_size\": {d}, \"layers\": [", .{ m.prompt_tokens, m.first_token, m.token_sha256, m.block_size, m.tp_rank, m.tp_size });
    for (m.layers, 0..) |l, i| {
        if (i > 0) try w.writeAll(", ");
        if (l.kind == .recurrent) {
            try writeRecurrent(l, w);
            continue;
        }
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

/// Protocol 2's recurrent layer: its index and kind, then each part's name, shape, dims and dtype.
fn writeRecurrent(l: Layer, w: *std.Io.Writer) Error!void {
    try w.print("{{\"index\": {d}, \"kind\": \"recurrent\", \"parts\": [", .{l.index});
    for (l.parts, 0..) |p, j| {
        if (j > 0) try w.writeAll(", ");
        try w.writeAll("{\"name\": ");
        try writeString(w, p.name);
        try w.writeAll(", \"shape\": [");
        for (p.shape, 0..) |x, k| try w.print("{s}{d}", .{ if (k > 0) ", " else "", x });
        try w.writeAll("], \"dims\": [");
        for (p.dims, 0..) |d, k| try w.print("{s}\"{s}\"", .{ if (k > 0) ", " else "", @tagName(d) });
        try w.print("], \"dtype\": \"{s}\"}}", .{@tagName(p.dtype)});
    }
    try w.writeAll("]}");
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

/// Parse and check a manifest as a protocol 1 reader; strings and arrays live in `arena`.
pub fn parse(arena: Allocator, bytes: []const u8) Error!Manifest {
    return parseUpTo(arena, bytes, wire.version);
}

/// As parse, accepting protocols 1 to `most`; only protocol 2 may hold a recurrent layer.
pub fn parseUpTo(arena: Allocator, bytes: []const u8, most: u16) Error!Manifest {
    const root = std.json.parseFromSliceLeaky(std.json.Value, arena, bytes, .{}) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return error.BadManifest,
    };
    const o = asObject(root) orelse return error.BadManifest;
    const protocol = try int(o, "protocol");
    if (protocol < wire.version or protocol > most) return error.BadManifest;
    var m: Manifest = .{
        .protocol = @intCast(protocol),
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
    for (list.array.items, layers) |item, *l| {
        l.* = try parseLayer(arena, item);
        if (l.kind == .recurrent and m.protocol < wire.recurrent_version) return error.BadManifest;
    }
    m.layers = layers;
    return m;
}

fn parseLayer(arena: Allocator, v: std.json.Value) Error!Layer {
    const o = asObject(v) orelse return error.BadManifest;
    if (std.mem.eql(u8, try str(o, "kind"), "recurrent")) return parseRecurrent(arena, o);
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
    // a peer's sizes: the row's bytes checked alone first (zero rows cannot mask it), then rows times row
    var row: u64 = l.dtype.size();
    for (shape, dims) |x, d| {
        if (d != .block) row = std.math.mul(u64, row, x) catch return error.BadManifest;
    }
    _ = std.math.mul(u64, row, l.rows()) catch return error.BadManifest;
    return l;
}

/// A recurrent layer: a non-empty list of parts, each a whole tensor (no block dimension) with a dtype.
fn parseRecurrent(arena: Allocator, o: std.json.ObjectMap) Error!Layer {
    const index = try int(o, "index");
    const list = o.get("parts") orelse return error.BadManifest;
    if (index > std.math.maxInt(u32) or list != .array or list.array.items.len == 0) return error.BadManifest;
    const parts = try arena.alloc(Part, list.array.items.len);
    for (list.array.items, parts) |item, *p| {
        const po = asObject(item) orelse return error.BadManifest;
        const shape_v = po.get("shape") orelse return error.BadManifest;
        const dims_v = po.get("dims") orelse return error.BadManifest;
        if (shape_v != .array or dims_v != .array or shape_v.array.items.len != dims_v.array.items.len or shape_v.array.items.len == 0) return error.BadManifest;
        const shape = try arena.alloc(u64, shape_v.array.items.len);
        const dims = try arena.alloc(PartDim, dims_v.array.items.len);
        for (shape_v.array.items, dims_v.array.items, shape, dims) |sv, dv, *x, *d| {
            if (sv != .integer or sv.integer <= 0 or dv != .string) return error.BadManifest;
            x.* = @intCast(sv.integer);
            d.* = std.meta.stringToEnum(PartDim, dv.string) orelse return error.BadManifest;
        }
        p.* = .{ .name = try str(po, "name"), .shape = shape, .dims = dims, .dtype = std.meta.stringToEnum(Dtype, try str(po, "dtype")) orelse return error.BadManifest };
        var n: u64 = p.dtype.size(); // a peer names these sizes: refuse one whose bytes a u64 cannot hold
        for (shape) |x| n = std.math.mul(u64, n, x) catch return error.BadManifest;
    }
    return .{ .index = @intCast(index), .kind = .recurrent, .shape = &.{}, .dims = &.{}, .dtype = .float32, .parts = parts };
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

test "a protocol 2 manifest with a recurrent layer writes, parses back, and a protocol 1 reader refuses it" {
    const gpa = std.testing.allocator;
    const parts = [_]Part{
        .{ .name = "recurrence", .shape = &.{ 4, 8, 8 }, .dims = &.{ .head, .value, .key }, .dtype = .float32 },
        .{ .name = "conv", .shape = &.{ 3, 40 }, .dims = &.{ .tap, .channel }, .dtype = .bfloat16 },
    };
    const layers = [_]Layer{
        .{ .index = 0, .kind = .recurrent, .shape = &.{}, .dims = &.{}, .dtype = .float32, .parts = &parts },
        .{ .index = 3, .kind = .attention, .shape = &.{ 2, 4, 1, 16 }, .dims = &.{ .block, .head, .token, .head_dim }, .dtype = .bfloat16, .heads = 4, .total_heads = 4, .head_size = 16 },
    };
    const m: Manifest = .{ .protocol = wire.recurrent_version, .handoff = @splat(0x2b), .model = "q", .prompt_tokens = 2, .first_token = 0, .token_sha256 = wire.tokenSha256(&.{ 5, 6 }), .block_size = 1, .tp_rank = 0, .tp_size = 1, .layers = &layers, .frames = 3 };
    var out: std.Io.Writer.Allocating = .init(gpa);
    defer out.deinit();
    try write(m, &out.writer);
    const text = out.written();
    try std.testing.expect(std.mem.startsWith(u8, text, "{\"protocol\": 2, "));
    try std.testing.expect(std.mem.indexOf(u8, text, "{\"index\": 0, \"kind\": \"recurrent\", \"parts\": [{\"name\": \"recurrence\", \"shape\": [4, 8, 8], \"dims\": [\"head\", \"value\", \"key\"], \"dtype\": \"float32\"}, ") != null);
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    try std.testing.expectError(error.BadManifest, parse(arena.allocator(), text)); // a protocol 1 reader
    const back = try parseUpTo(arena.allocator(), text, wire.recurrent_version);
    try std.testing.expectEqual(@as(u16, 2), back.protocol);
    try std.testing.expectEqual(@as(u64, 2), back.layers[0].rows());
    try std.testing.expectEqual(@as(?u64, 4 * 8 * 8 * 4), back.layers[0].frameBytes(0, 1));
    try std.testing.expectEqual(@as(?u64, 3 * 40 * 2), back.layers[0].frameBytes(1, 1));
    try std.testing.expectEqual(@as(?u64, null), back.layers[0].frameBytes(2, 1));
    try std.testing.expectEqual(@as(?u64, null), back.layers[0].frameBytes(0, 2));
    try std.testing.expectEqualStrings("conv", back.layers[0].parts[1].name);
    try std.testing.expectEqual(@as(?u64, 2 * 4 * 16 * 2), back.layers[1].frameBytes(0, 2));
}

const zeros64 = "0000000000000000000000000000000000000000000000000000000000000000";

test "a recurrent layer is refused under protocol 1, without parts, or with a block dimension" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const head = "\"handoff\": \"00000000000000000000000000000000\", \"model\": \"m\", \"prompt_tokens\": 1, \"first_token\": 0, \"token_sha256\": \"" ++ zeros64 ++ "\", \"block_size\": 1, \"tp_rank\": 0, \"tp_size\": 1, \"layers\": [";
    const part = "{\"name\": \"s\", \"shape\": [2], \"dims\": [\"value\"], \"dtype\": \"float32\"}";
    const good = "{\"index\": 0, \"kind\": \"recurrent\", \"parts\": [" ++ part ++ "]}";
    _ = try parseUpTo(a, "{\"protocol\": 2, " ++ head ++ good ++ "], \"frames\": 1}", 2);
    try std.testing.expectError(error.BadManifest, parseUpTo(a, "{\"protocol\": 1, " ++ head ++ good ++ "], \"frames\": 1}", 2));
    try std.testing.expectError(error.BadManifest, parseUpTo(a, "{\"protocol\": 2, " ++ head ++ "{\"index\": 0, \"kind\": \"recurrent\", \"parts\": []}], \"frames\": 1}", 2));
    const blocked = "{\"index\": 0, \"kind\": \"recurrent\", \"parts\": [{\"name\": \"s\", \"shape\": [2], \"dims\": [\"block\"], \"dtype\": \"float32\"}]}";
    try std.testing.expectError(error.BadManifest, parseUpTo(a, "{\"protocol\": 2, " ++ head ++ blocked ++ "], \"frames\": 1}", 2));
    try std.testing.expectError(error.BadManifest, parseUpTo(a, "{\"protocol\": 3, " ++ head ++ good ++ "], \"frames\": 1}", 2));
}

test "a page layer whose row or array byte size overflows is refused, and no frame size wraps" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const head = "{\"protocol\": 1, \"handoff\": \"00000000000000000000000000000000\", \"model\": \"m\", \"prompt_tokens\": 1, \"first_token\": 0, \"token_sha256\": \"" ++ zeros64 ++ "\", \"block_size\": 1, \"tp_rank\": 0, \"tp_size\": 1, \"layers\": [";
    const row = "{\"index\": 0, \"kind\": \"attention\", \"shape\": [1, 9223372036854775807, 2], \"dims\": [\"block\", \"head\", \"head_dim\"], \"dtype\": \"float32\", \"heads\": 1, \"total_heads\": 1, \"head_size\": 1}";
    try std.testing.expectError(error.BadManifest, parse(arena.allocator(), head ++ row ++ "], \"frames\": 1}"));
    const blocks = "{\"index\": 0, \"kind\": \"attention\", \"shape\": [9223372036854775807, 4], \"dims\": [\"block\", \"head_dim\"], \"dtype\": \"bfloat16\", \"heads\": 1, \"total_heads\": 1, \"head_size\": 4}";
    try std.testing.expectError(error.BadManifest, parse(arena.allocator(), head ++ blocks ++ "], \"frames\": 1}"));
    const zero_rows = "{\"index\": 0, \"kind\": \"attention\", \"shape\": [0, 9223372036854775807, 2], \"dims\": [\"block\", \"head\", \"head_dim\"], \"dtype\": \"float32\", \"heads\": 1, \"total_heads\": 1, \"head_size\": 1}";
    try std.testing.expectError(error.BadManifest, parse(arena.allocator(), head ++ zero_rows ++ "], \"frames\": 1}"));
    const empty = "{\"index\": 0, \"kind\": \"attention\", \"shape\": [0, 4, 16], \"dims\": [\"block\", \"head\", \"head_dim\"], \"dtype\": \"bfloat16\", \"heads\": 4, \"total_heads\": 4, \"head_size\": 16}";
    const kept = try parse(arena.allocator(), head ++ empty ++ "], \"frames\": 0}"); // an empty layer stays as protocol 1 had it
    try std.testing.expectEqual(@as(u64, 0), kept.layers[0].rows());
    try std.testing.expectEqual(@as(u64, 4 * 16 * 2), kept.layers[0].rowBytes());
    const l: Layer = .{ .index = 0, .kind = .attention, .shape = &.{ 4, 1024 }, .dims = &.{ .block, .head_dim }, .dtype = .bfloat16 };
    try std.testing.expectEqual(@as(?u64, null), l.frameBytes(0, std.math.maxInt(u64)));
    try std.testing.expectEqual(@as(?u64, 4096), l.frameBytes(0, 2));
}

test "a recurrent part whose byte size overflows is refused, not computed" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const text = "{\"protocol\": 2, \"handoff\": \"00000000000000000000000000000000\", \"model\": \"m\", \"prompt_tokens\": 1, \"first_token\": 0, \"token_sha256\": \"" ++ zeros64 ++ "\", \"block_size\": 1, \"tp_rank\": 0, \"tp_size\": 1, \"layers\": [{\"index\": 0, \"kind\": \"recurrent\", \"parts\": [{\"name\": \"r\", \"shape\": [9223372036854775807, 2], \"dims\": [\"head\", \"value\"], \"dtype\": \"float32\"}]}], \"frames\": 1}";
    try std.testing.expectError(error.BadManifest, parseUpTo(arena.allocator(), text, 2));
}

test "a page layer keeps protocol 1's dimension names: a part's names are refused there, under either protocol" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const head = "\"handoff\": \"00000000000000000000000000000000\", \"model\": \"m\", \"prompt_tokens\": 1, \"first_token\": 0, \"token_sha256\": \"" ++ zeros64 ++ "\", \"block_size\": 1, \"tp_rank\": 0, \"tp_size\": 1, \"layers\": [";
    inline for (.{ "value", "key", "tap", "channel" }) |name| {
        const layer = "{\"index\": 0, \"kind\": \"attention\", \"shape\": [1, 4], \"dims\": [\"block\", \"" ++ name ++ "\"], \"dtype\": \"bfloat16\", \"heads\": 1, \"total_heads\": 1, \"head_size\": 4}";
        try std.testing.expectError(error.BadManifest, parse(a, "{\"protocol\": 1, " ++ head ++ layer ++ "], \"frames\": 1}"));
        try std.testing.expectError(error.BadManifest, parseUpTo(a, "{\"protocol\": 2, " ++ head ++ layer ++ "], \"frames\": 1}", 2));
    }
    const part_block = "{\"index\": 0, \"kind\": \"recurrent\", \"parts\": [{\"name\": \"s\", \"shape\": [2], \"dims\": [\"token\"], \"dtype\": \"float32\"}]}";
    try std.testing.expectError(error.BadManifest, parseUpTo(a, "{\"protocol\": 2, " ++ head ++ part_block ++ "], \"frames\": 1}", 2));
}

test "manifests that break the schema are refused" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try std.testing.expectError(error.BadManifest, parse(a, "[]"));
    try std.testing.expectError(error.BadManifest, parse(a, "{\"protocol\": 2}"));
    const two_blocks = "{\"protocol\": 1, \"handoff\": \"00000000000000000000000000000000\", \"model\": \"m\", \"prompt_tokens\": 1, \"first_token\": 0, \"token_sha256\": \"" ++ zeros64 ++ "\", \"block_size\": 16, \"tp_rank\": 0, \"tp_size\": 1, \"layers\": [{\"index\": 0, \"kind\": \"attention\", \"shape\": [1, 1], \"dims\": [\"block\", \"block\"], \"dtype\": \"bfloat16\", \"heads\": 1, \"total_heads\": 1, \"head_size\": 1}], \"frames\": 1}";
    try std.testing.expectError(error.BadManifest, parse(a, two_blocks));
}
