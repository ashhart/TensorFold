//! MiniMax M3.1's device serve glue (the family-owned launches the mainline EXL3 modules do not
//! carry): the swigluoai activation that stands between the grouped gate/up GEMM partials and the
//! down GEMM, graded against serve.zig's swigluoai. The rest of the forward is mainline code:
//! per-token EXL3 decode lines (`cuda.exl3.Dense`, mul1 odd half steps and the head's 8 halves
//! included), the grouped routed experts (`cuda.exl3_experts`), and the MSA indexer launches
//! (`cuda_sparse.zig`).

const std = @import("std");
const cuda = @import("cuda");
const serve = @import("serve.zig");

/// The family's glue kernel from the minimax_m3 image.
pub const Kernels = struct {
    swigluoai: cuda.Function,

    pub fn load(m: cuda.Module) !Kernels {
        var k: Kernels = undefined;
        k.swigluoai = try m.function("tf_m3_swigluoai");
        return k;
    }
};

/// The M3.1 activation rows: (rows, n) fp16 gate and up rows -> (rows, n) fp16, the fork's
/// activation.cu:124 contract (`limit` 7, alpha hardwired 1.702 in the kernel). The pack's dense,
/// shared and routed widths are multiples of 512, so the launch never carries a tail block.
pub fn swigluoai(
    ks: Kernels,
    stream: cuda.Stream,
    g: u64,
    u: u64,
    out: u64,
    rows: usize,
    n: usize,
    limit: f32,
) !void {
    var a: cuda.Args = .{};
    inline for (.{ g, u, out, @as(i64, @intCast(n)), limit }) |v| a.add(v);
    const blocks = (n + 512 - 1) / 512; // 128 threads a block, 4 elements a lane
    try cuda.launch.launch(ks.swigluoai, .{ .grid = .{ .x = @intCast(rows), .y = @intCast(blocks) }, .block = .{ .x = 128 } }, stream, &a);
}

test "the glue's activation lands on serve.zig's swigluoai contract at the pack's numbers" {
    // the contract's fingerprints: u clamps both ways then +1, g never passes the limit,
    // the sigmoid runs at alpha 1.702 (the pack's swiglu_alpha); fp16 store rounds after.
    try std.testing.expectEqual(@as(f32, 0), serve.swigluoai(0, -7, 1.702, 7.0)); // u+1 = 0 at the floor
    try std.testing.expect(serve.swigluoai(7, 7, 1.702, 7.0) > 0);
    try std.testing.expect(serve.swigluoai(-1, 7, 1.702, 7.0) < serve.swigluoai(1, 7, 1.702, 7.0));
    try std.testing.expectEqual(serve.swigluoai(12, 0, 1.702, 7.0), serve.swigluoai(7, 0, 1.702, 7.0)); // g clipped above
}