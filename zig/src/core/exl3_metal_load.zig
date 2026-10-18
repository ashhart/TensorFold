//! Resident Metal views from the same exl3_load.Parts used by the CUDA uploader.
const mtl = @import("metal");
const projection = @import("exl3_projection.zig");
const loader = @import("exl3_load.zig");

pub const Owned = struct {
    matrix: projection.Matrix,
    buffers: [4]?mtl.Buffer,

    pub fn deinit(o: *Owned) void {
        for (&o.buffers) |*b| {
            if (b.*) |v| v.deinit();
            b.* = null;
        }
    }
};

pub fn fromParts(device: mtl.Device, parts: loader.Parts, g: loader.Group) !Owned {
    var o: Owned = .{ .matrix = undefined, .buffers = @splat(null) };
    errdefer o.deinit();
    o.buffers[0] = try copy(device, parts.trellis.bytes);
    o.buffers[1] = try copy(device, parts.suh.bytes);
    o.buffers[2] = try copy(device, parts.svh.bytes);
    if (parts.bias) |b| o.buffers[3] = try copy(device, b.bytes);
    o.matrix = .{ .trellis = .{ .buffer = o.buffers[0].? }, .suh = .{ .buffer = o.buffers[1].? }, .svh = .{ .buffer = o.buffers[2].? }, .bias = if (o.buffers[3]) |b| .{ .buffer = b } else null, .k = g.k, .n = g.n, .width = .{ .halves = g.halves }, .codebook = switch (g.codebook) {
        .inst3 => .inst3,
        .mcg => .mcg,
        .mul1 => .mul1,
    }, .suh_format = if (parts.suh_packed) .packed_signs else .f16, .svh_format = if (parts.suh_packed) .packed_signs else .f16 };
    try o.matrix.validate();
    return o;
}

fn copy(d: mtl.Device, bytes: []const u8) !mtl.Buffer {
    const b = try d.buffer(bytes.len, mtl.ResourceOptions.shared);
    @memcpy(b.contents()[0..bytes.len], bytes);
    return b;
}
