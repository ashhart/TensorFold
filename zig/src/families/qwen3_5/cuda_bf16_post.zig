//! Plain HF's FP32 GDN scales use the shared FP32-weight RMS and the existing BF16 SiLU product.
const E = @import("cuda_engine.zig");
pub fn run(e: *E.Engine, weight: u64, rows: usize, sums: u64) !void {
    const width = e.g.valueWidth();
    // QKV is dead after conv commit and scan, and always holds at least valueWidth elements a row.
    try e.bf16.?.norms.addRmsH(e.stream, e.y, null, weight, .f32, 0, e.qkv, null, rows * e.g.v_heads, 128, e.g.eps);
    try e.run(e.fns.mlp_act, .{ (rows * width + 1023) / 1024, 1, 1 }, .{ 256, 1, 1 }, .{ e.z, e.qkv, e.mix, @as(c_longlong, @intCast(rows * width)), e.tables.ptr + E.Engine.params_at, @as(c_int, @intCast(width)), sums });
}
