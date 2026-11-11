//! A hand-off file for tests, not a wire format: tokens, manifest and parts; `feed` replays pulls into a sink.
const std = @import("std");
const fabric = @import("fabric");
const manifest = fabric.manifest;
const plan = @import("plan.zig");
const ex = @import("export.zig");
const p2 = @import("p2.zig");
const file = @import("../handoff_file.zig");

pub const magic = "DSV41HOF";
pub const version: u32 = 1;
const head_bytes = 8 + 3 * 4;

pub const Error = error{ BadBlob, BadManifest, OutOfMemory, WriteFailed };

/// The file for `parts` (shard 0, one producer) after `tokens`: the manifest is the plan's own, protocol 2.
pub fn write(arena: std.mem.Allocator, model: []const u8, tokens: []const u32, parts: []const ex.Part) Error![]u8 {
    const built = try p2.layers(arena, parts);
    const m: manifest.Manifest = .{ .protocol = 2, .handoff = @splat(0), .model = model, .prompt_tokens = tokens.len, .first_token = 0, .token_sha256 = fabric.wire.tokenSha256(tokens), .block_size = 1, .tp_rank = 0, .tp_size = 1, .layers = built.layers, .frames = parts.len };
    var json: std.Io.Writer.Allocating = .init(arena);
    try manifest.write(m, &json.writer);
    var out: std.ArrayList(u8) = .empty;
    try out.appendSlice(arena, magic);
    for ([_]u32{ version, @intCast(tokens.len), @intCast(json.written().len) }) |v| try out.appendSlice(arena, &std.mem.toBytes(std.mem.nativeToLittle(u32, v)));
    for (tokens) |t| try out.appendSlice(arena, &std.mem.toBytes(std.mem.nativeToLittle(u32, t)));
    try out.appendSlice(arena, json.written());
    for (parts) |p| try out.appendSlice(arena, p.bytes);
    return out.items;
}

pub const Blob = struct {
    tokens: []u32,
    manifest: manifest.Manifest,
    /// the parts' bytes, layer by layer
    data: []const u8,

    pub fn parse(arena: std.mem.Allocator, bytes: []const u8) Error!Blob {
        if (bytes.len < head_bytes or !std.mem.eql(u8, bytes[0..8], magic)) return error.BadBlob;
        const word = struct {
            fn at(b: []const u8, i: usize) u32 {
                return std.mem.readInt(u32, b[i..][0..4], .little);
            }
        }.at;
        if (word(bytes, 8) != version) return error.BadBlob;
        const n: usize = word(bytes, 12);
        const json: usize = word(bytes, 16);
        const tokens_end = std.math.add(usize, head_bytes, std.math.mul(usize, n, 4) catch return error.BadBlob) catch return error.BadBlob;
        const json_end = std.math.add(usize, tokens_end, json) catch return error.BadBlob;
        if (json_end > bytes.len) return error.BadBlob;
        const tokens = try arena.alloc(u32, n);
        for (tokens, 0..) |*t, i| t.* = word(bytes, head_bytes + i * 4);
        const m = manifest.parseUpTo(arena, bytes[tokens_end..json_end], 2) catch return error.BadManifest;
        var want: usize = 0;
        for (m.layers) |l| for (l.parts) |p| {
            want = std.math.add(usize, want, @intCast(p.bytes())) catch return error.BadBlob;
        };
        if (bytes.len - json_end != want) return error.BadBlob;
        if (m.prompt_tokens != n or !std.mem.eql(u8, &m.token_sha256, &fabric.wire.tokenSha256(tokens))) return error.BadBlob;
        return .{ .tokens = tokens, .manifest = m, .data = bytes[json_end..] };
    }

    /// Every part into `sink`, as a pull delivers them.
    pub fn feed(b: Blob, sink: fabric.decoder.Sink) !void {
        var at: usize = 0;
        for (b.manifest.layers) |*l| for (l.parts, 0..) |p, i| {
            const n: usize = @intCast(p.bytes());
            try sink.put_fn(sink.ptr, l, i, 1, b.data[at..][0..n]);
            at += n;
        };
    }
};

/// A producer's `.mkvh2` stream replayed through the decoder's checks (`expect` supplies name and tokens) as a Blob.
pub fn fromStream(arena: std.mem.Allocator, bytes: []const u8, expect: fabric.decoder.Expect, reason: *fabric.decoder.Reason) !Blob {
    const Entry = struct { layer: u32, part: u64, bytes: []const u8 };
    const Recorder = struct {
        arena: std.mem.Allocator,
        stream: []const u8,
        entries: std.ArrayList(Entry) = .empty,

        fn put(ptr: *anyopaque, layer: *const manifest.Layer, row_start: u64, rows: u64, data: []const u8) anyerror!void {
            const r: *@This() = @ptrCast(@alignCast(ptr));
            if (layer.kind != .recurrent or rows != 1 or row_start >= layer.parts.len) return error.BadPart;
            // a part inside the stream is kept where it is: copying a million tokens' parts would double their 3 GB
            const inside = @intFromPtr(data.ptr) >= @intFromPtr(r.stream.ptr) and @intFromPtr(data.ptr) + data.len <= @intFromPtr(r.stream.ptr) + r.stream.len;
            try r.entries.append(r.arena, .{ .layer = layer.index, .part = row_start, .bytes = if (inside) data else try r.arena.dupe(u8, data) });
        }
    };
    var rec: Recorder = .{ .arena = arena, .stream = bytes };
    const got = try file.pull(arena, bytes, expect, .{ .ptr = &rec, .put_fn = Recorder.put }, reason);
    var out: std.ArrayList(u8) = .empty;
    var total: usize = 0;
    for (got.manifest.layers) |l| for (l.parts) |p| {
        total += @intCast(p.bytes());
    };
    try out.ensureTotalCapacityPrecise(arena, total);
    for (got.manifest.layers) |l| for (l.parts, 0..) |p, i| {
        const e = for (rec.entries.items) |e| {
            if (e.layer == l.index and e.part == i) break e;
        } else return error.BadBlob;
        if (e.bytes.len != p.bytes()) return error.BadBlob;
        try out.appendSlice(arena, e.bytes);
    };
    return .{ .tokens = try arena.dupe(u32, expect.tokens), .manifest = got.manifest, .data = out.items };
}
