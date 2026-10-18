//! Versioned checkpoint, selected-layer and exact bounded-piece identity extends the rank/layout digest.
const std = @import("std");
const Piece = @import("sessions_gpu.zig").Piece;
fn bytes(h: *std.crypto.hash.sha2.Sha256, value: []const u8) void {
    var len: [8]u8 = undefined;
    std.mem.writeInt(u64, &len, value.len, .little);
    h.update(&len);
    h.update(value);
}
pub fn bind(h: *std.crypto.hash.sha2.Sha256, checkpoint: [32]u8, layers: []const u32, pieces: []const Piece, semantic_bytes: []const u8) void {
    h.update("TF-DSV41-CUDA-SESSION-v2-fp8-584-132");
    h.update(&checkpoint);
    bytes(h, std.mem.sliceAsBytes(layers));
    bytes(h, semantic_bytes);
    for (pieces) |p| {
        bytes(h, p.role);
        var shape: [16]u8 = undefined;
        std.mem.writeInt(u64, shape[0..8], p.len, .little);
        std.mem.writeInt(u64, shape[8..16], p.stride, .little);
        h.update(&shape);
    }
}

fn hashValue(h: *std.crypto.hash.sha2.Sha256, item: anytype) void {
    const T = @TypeOf(item);
    switch (@typeInfo(T)) {
        .@"struct" => if (@hasDecl(T, "items")) hashValue(h, item.items()) else {
            inline for (@typeInfo(T).@"struct".field_names) |field| {
                bytes(h, field);
                hashValue(h, @field(item, field));
            }
        },
        .pointer => |p| {
            if (p.size == .one) {
                hashValue(h, item.*);
                return;
            }
            if (p.size != .slice) @compileError("session identity requires an owned value or slice");
            var len: [8]u8 = undefined;
            std.mem.writeInt(u64, &len, item.len, .little);
            h.update(&len);
            for (item) |element| hashValue(h, element);
        },
        .array => {
            for (item) |element| hashValue(h, element);
        },
        .optional => {
            const present = [_]u8{@intFromBool(item != null)};
            h.update(&present);
            if (item) |v| hashValue(h, v);
        },
        .@"enum" => bytes(h, @tagName(item)),
        .bool => {
            const flag = [_]u8{@intFromBool(item)};
            h.update(&flag);
        },
        .int, .float => h.update(std.mem.asBytes(&item)),
        else => @compileError("unsupported session identity value"),
    }
}
pub fn semantics(item: anytype) [32]u8 {
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    hashValue(&h, item);
    var out: [32]u8 = undefined;
    h.final(&out);
    return out;
}
