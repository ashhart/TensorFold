//! Compile the real shared affine encoder against proposal rows; this entry is never executed by host gates.
const mtl = @import("metal");
const rows = @import("core").proposal_rows;
const affine = @import("affine_mm");
export fn proposal_affine_encode(v: *const rows.Owned, e: *const mtl.ComputeEncoder, p: *const affine.Pipes, b: *const [3]mtl.Buffer, x: *const mtl.Buffer, sums: *const mtl.Buffer, y: *const mtl.Buffer, count: u32) u32 {
    const format: affine.Format = .{ .bits = v.bits, .group = @intCast(v.group), .out_f32 = v.output_f32 };
    v.affine(affine, format, e.*, p.*, .{ .w = .{ .buf = b[0], .off = @as(usize, 0) }, .s = .{ .buf = b[1], .off = @as(usize, 0) }, .b = .{ .buf = b[2], .off = @as(usize, 0) } }, .{ .buf = x.*, .off = @as(usize, 0) }, .{ .buf = sums.*, .off = @as(usize, 0) }, .{ .buf = y.*, .off = @as(usize, 0) }, count) catch return 1;
    return 0;
}
