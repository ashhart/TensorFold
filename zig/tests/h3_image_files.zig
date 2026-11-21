//! The H3-Image tools' safetensors files: aligned images, and a synthetic checkpoint of one shard and its index.
const std = @import("std");
const fixture = @import("checkpoint_fixture");

/// A safetensors image whose data starts on 16 bytes: the header padded with spaces.
pub fn image(a: std.mem.Allocator, entries: []const fixture.Entry) ![]u8 {
    const plain = try fixture.image(a, entries);
    const header: usize = @intCast(std.mem.readInt(u64, plain[0..8], .little));
    const pad = (16 - (8 + header) % 16) % 16;
    const out = try a.alloc(u8, plain.len + pad);
    std.mem.writeInt(u64, out[0..8], header + pad, .little);
    @memcpy(out[8..][0..header], plain[8..][0..header]);
    @memset(out[8 + header ..][0..pad], ' ');
    @memcpy(out[8 + header + pad ..], plain[8 + header ..]);
    return out;
}

/// `entries` as `dir`'s one shard, with the index naming it for every tensor.
pub fn checkpoint(io: std.Io, a: std.mem.Allocator, dir: []const u8, entries: []const fixture.Entry) !void {
    try std.Io.Dir.cwd().createDirPath(io, dir);
    const file = "model-00001-of-00001.safetensors";
    var index: std.ArrayList(u8) = .empty;
    try index.appendSlice(a, "{\"weight_map\": {");
    for (entries, 0..) |e, i| try index.print(a, "{s}\"{s}\": \"{s}\"", .{ if (i == 0) "" else ", ", e.name, file });
    try index.appendSlice(a, "}}");
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ dir, "model.safetensors.index.json" }), .data = index.items });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ dir, file }), .data = try image(a, entries) });
}
