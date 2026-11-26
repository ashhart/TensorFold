//! Streaming native-inspection validation over the same records used by the GLM and Flash Next loaders.
const std = @import("std");
const glm = @import("glm_tfq3_check").tfq3_check;
const flashnext = @import("flashnext_memory").tfq3_files.check;

pub const Kind = enum { dense, layer };
pub const Info = struct { kind: Kind, tensors: u32, layer: u32 = 0, experts: u32 = 0 };

/// Reuses the loader's complete PWV3 table validation, including all finite floats and halves.
pub fn tables(bytes: []const u8) !void {
    return glm.tables(bytes);
}

fn readAt(a: std.mem.Allocator, io: std.Io, f: std.Io.File, off: usize, len: usize) ![]u8 {
    const bytes = try a.alloc(u8, len);
    errdefer a.free(bytes);
    if ((try f.readPositionalAll(io, bytes, off)) != len) return error.Tfq3Truncated;
    return bytes;
}

fn filenameLayer(path: []const u8) !u32 {
    const base = std.fs.path.basename(path);
    if (!std.mem.startsWith(u8, base, "layer_") or !std.mem.endsWith(u8, base, ".tffpw")) return error.Tfq3Layer;
    const digits = base["layer_".len .. base.len - ".tffpw".len];
    if (digits.len == 0) return error.Tfq3Layer;
    for (digits) |ch| if (!std.ascii.isDigit(ch)) return error.Tfq3Layer;
    return std.fmt.parseInt(u32, digits, 10) catch error.Tfq3Layer;
}

/// Validates all metadata, sections, tensor extents and filename identity without mapping or reading the weight body.
pub fn shard(a: std.mem.Allocator, io: std.Io, path: []const u8, file_size: u64) !Info {
    const file_len = std.math.cast(usize, file_size) orelse return error.Tfq3Overflow;
    const f = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer f.close(io);
    var magic_bytes: [8]u8 = undefined;
    if ((try f.readPositionalAll(io, &magic_bytes, 0)) != magic_bytes.len) return error.Tfq3Truncated;
    const magic = std.mem.readInt(u32, magic_bytes[0..4], .little);
    if (magic == glm.magic_nonmoe) {
        var head: [@sizeOf(glm.NonMoeHeader)]u8 = undefined;
        if ((try f.readPositionalAll(io, &head, 0)) != head.len) return error.Tfq3Truncated;
        var words: usize = undefined;
        var halfs: usize = undefined;
        const layout = try glm.nonMoeHeader(&head, file_len, &words, &halfs);
        const meta = try readAt(a, io, f, layout.metas.off, layout.metas.len);
        defer a.free(meta);
        try glm.nonMoeRecords(meta, layout, words, halfs, true);
        const h = std.mem.bytesToValue(glm.NonMoeHeader, &head);
        return .{ .kind = .dense, .tensors = h.num_tensors };
    }
    if (magic != glm.magic_layer) return error.UnsupportedQuantization;
    var head: [@sizeOf(glm.LayerHeader)]u8 = undefined;
    if ((try f.readPositionalAll(io, &head, 0)) != head.len) return error.Tfq3Truncated;
    const h = std.mem.bytesToValue(glm.LayerHeader, &head);
    const named_layer = try filenameLayer(path);
    if (h.hidden_size == glm.expert_hidden and h.moe_inter == glm.expert_inter) {
        const want: glm.LayerWant = .{ .layer = named_layer, .experts = h.num_experts, .hidden = h.hidden_size, .inter = h.moe_inter };
        var layout = try glm.layerHeader(&head, file_len, want);
        const meta = try readAt(a, io, f, layout.meta.off, layout.meta.len);
        defer a.free(meta);
        try glm.layerExperts(meta, &layout, h.num_experts, h.hidden_size, h.moe_inter);
    } else if (h.hidden_size == flashnext.hidden_size and h.moe_inter == flashnext.moe_inter and h.num_experts == flashnext.num_experts) {
        var layout = try flashnext.header(&head, file_len, named_layer);
        const meta = try readAt(a, io, f, layout.meta.off, layout.meta.len);
        defer a.free(meta);
        try flashnext.experts(meta, &layout);
    } else return error.Tfq3Geometry;
    return .{ .kind = .layer, .tensors = h.num_tensors, .layer = h.layer_idx, .experts = h.num_experts };
}
