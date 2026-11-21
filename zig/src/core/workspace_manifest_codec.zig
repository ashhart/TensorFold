const std = @import("std");
const types = @import("workspace_manifest_types.zig");

const magic = "TFWM0001";

const Writer = struct {
    bytes: []u8,
    at: usize = 0,

    fn put(w: *Writer, data: []const u8) void {
        @memcpy(w.bytes[w.at..][0..data.len], data);
        w.at += data.len;
    }

    fn number(w: *Writer, comptime T: type, value: T) void {
        var data: [@sizeOf(T)]u8 = undefined;
        std.mem.writeInt(T, &data, value, .little);
        w.put(&data);
    }

    fn text(w: *Writer, value: []const u8) void {
        w.number(u32, @intCast(value.len));
        w.put(value);
    }
};

const Reader = struct {
    bytes: []const u8,
    at: usize = 0,

    fn take(r: *Reader, n: usize) ![]const u8 {
        if (r.at > r.bytes.len or n > r.bytes.len - r.at) return error.ManifestCorrupt;
        const out = r.bytes[r.at..][0..n];
        r.at += n;
        return out;
    }

    fn number(r: *Reader, comptime T: type) !T {
        const data = try r.take(@sizeOf(T));
        return std.mem.readInt(T, data[0..@sizeOf(T)], .little);
    }

    fn text(r: *Reader) ![]const u8 {
        const n = try r.number(u32);
        return r.take(n);
    }
};

fn add(total: *usize, n: usize) !void {
    total.* = std.math.add(usize, total.*, n) catch return error.ManifestTooLarge;
    if (total.* > types.max_manifest_bytes) return error.ManifestTooLarge;
}

fn textLength(value: []const u8) !usize {
    return std.math.add(usize, 4, value.len) catch return error.ManifestTooLarge;
}

fn encodedLength(m: *const types.Manifest) !usize {
    var total: usize = magic.len + 32 + 4 + 4 + 4;
    try add(&total, try textLength(m.workspace));
    try add(&total, try textLength(m.model));
    try add(&total, try textLength(m.template));
    try add(&total, try textLength(m.cache));
    for (m.files) |f| {
        try add(&total, try textLength(f.path));
        try add(&total, 8 + 32 + 4);
    }
    const prompt_bytes = std.math.mul(usize, m.prompt.len, 4) catch return error.ManifestTooLarge;
    try add(&total, prompt_bytes);
    return total;
}

pub fn encode(m: *const types.Manifest) ![]u8 {
    try types.validateStructure(m);
    const bytes = try m.gpa.alloc(u8, try encodedLength(m));
    errdefer m.gpa.free(bytes);
    var w: Writer = .{ .bytes = bytes };
    w.put(magic);
    w.text(m.workspace);
    w.text(m.model);
    w.text(m.template);
    w.text(m.cache);
    w.number(u32, m.base_prefix_end);
    w.number(u32, @intCast(m.files.len));
    for (m.files) |f| {
        w.text(f.path);
        w.number(u64, f.size);
        w.put(&f.sha256);
        w.number(u32, f.prompt_end);
    }
    w.number(u32, @intCast(m.prompt.len));
    for (m.prompt) |token| w.number(u32, token);
    std.debug.assert(w.at == bytes.len - 32);
    var provenance: types.Digest = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes[0..w.at], &provenance, .{});
    w.put(&provenance);
    std.debug.assert(w.at == bytes.len);
    return bytes;
}

pub fn decode(gpa: std.mem.Allocator, bytes: []const u8) !types.Manifest {
    if (bytes.len < magic.len + 32 or bytes.len > types.max_manifest_bytes) return error.ManifestCorrupt;
    const content = bytes[0 .. bytes.len - 32];
    var expected: types.Digest = undefined;
    std.crypto.hash.sha2.Sha256.hash(content, &expected, .{});
    if (!std.mem.eql(u8, &expected, bytes[bytes.len - 32 ..])) return error.ManifestCorrupt;
    var r: Reader = .{ .bytes = content };
    if (!std.mem.eql(u8, try r.take(magic.len), magic)) return error.ManifestCorrupt;
    const workspace = try gpa.dupeSentinel(u8, try r.text(), 0);
    errdefer gpa.free(workspace);
    const model = try gpa.dupe(u8, try r.text());
    errdefer gpa.free(model);
    const template = try gpa.dupe(u8, try r.text());
    errdefer gpa.free(template);
    const cache = try gpa.dupe(u8, try r.text());
    errdefer gpa.free(cache);
    const base_prefix_end = try r.number(u32);
    const file_count = try r.number(u32);
    if (file_count == 0 or file_count > types.max_files) return error.ManifestCorrupt;
    const files = try gpa.alloc(types.File, file_count);
    var initialized: usize = 0;
    errdefer {
        for (files[0..initialized]) |f| gpa.free(f.path);
        gpa.free(files);
    }
    for (files) |*f| {
        f.path = try gpa.dupe(u8, try r.text());
        initialized += 1;
        f.size = try r.number(u64);
        f.sha256 = (try r.take(32))[0..32].*;
        f.prompt_end = try r.number(u32);
    }
    const prompt_count = try r.number(u32);
    if (prompt_count == 0 or prompt_count > types.max_prompt_tokens) return error.ManifestCorrupt;
    const prompt = try gpa.alloc(u32, prompt_count);
    errdefer gpa.free(prompt);
    for (prompt) |*token| token.* = try r.number(u32);
    if (r.at != content.len) return error.ManifestCorrupt;
    const m: types.Manifest = .{
        .gpa = gpa,
        .workspace = workspace,
        .model = model,
        .template = template,
        .cache = cache,
        .files = files,
        .base_prefix_end = base_prefix_end,
        .prompt = prompt,
        .provenance = expected,
    };
    types.validateStructure(&m) catch return error.ManifestCorrupt;
    return m;
}
