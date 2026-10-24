//! Reviewed lane/decode serving bytes and calls; changing them needs a reviewed implementation and receipt.
const std = @import("std");
fn hash(bytes: []const u8) [64]u8 {
    var h: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &h, .{});
    return std.fmt.bytesToHex(h, .lower);
}
test "dense serving: owned CUDA images retain their reviewed lane and decode bytes" {
    try std.testing.expectEqualStrings("aa13e9f274fe6535a5fa54c3d3a68273435e2a716cb4065c073c55f26f01ec1b", &hash(@embedFile("kernels/cuda/experts.cu")));
    try std.testing.expectEqualStrings("f195f5678edf7d3267ec30e442b1e0048543006e595ad21801afd905df621baf", &hash(@embedFile("kernels/cuda/experts_pack.cu")));
    try std.testing.expectEqualStrings("db560d705729142c66df8dee18fc635f20f09e4f09b8431de265bd6fac68e892", &hash(@embedFile("kernels/cuda/experts_prefill.cu")));
    try std.testing.expectEqualStrings("a6bf09198ffd685f7392d4d433af9739c91d67f7039309d3fd5fe6785f810ab9", &hash(@embedFile("kernels/cuda/gdn.cu")));
    try std.testing.expectEqualStrings("90ffa4386a570b4630fa6bf6c5fe681a059cfd06687886b2a32d5a228a7f8855", &hash(@embedFile("kernels/cuda/lane_gemv.cu")));
    try std.testing.expectEqualStrings("feb6dfc9de6d051be1ad89e2b6f261ab9a00cdabe792f481921b2b1c18e770a6", &hash(@embedFile("kernels/cuda/prefill_attention.cu")));
    const group = @embedFile("kernels/cuda/qmm_group.cu");
    try std.testing.expectEqualStrings("3b27b8e964f6e7453e962cd7a259253ae7e1940297439b4b518f05da04dd1041", &hash(group[0..16132]));
    try std.testing.expectEqualStrings("25172d5cd98fe2768a216183336f7eb9c23a66b0a56cfff77b9825ef66a481a6", &hash(group));
    try std.testing.expectEqualStrings("2e4f98ea301a4d87639c55b93c37577a9fd7f3cda0cbae055c7ba4958f019bfd", &hash(@embedFile("kernels/cuda/qmm_prefill.cu")));
    try std.testing.expectEqualStrings("9f58d6a6acbce8394be1aedbe56e1f72f94907fe0bbddc7659e5ead1d289251b", &hash(@embedFile("kernels/cuda/qwen35.cu")));
    try std.testing.expectEqualStrings("9aacbd32219ffe56556c847e4751137175acf62e8040a0a786b2eff1fce4fdaf", &hash(@embedFile("kernels/cuda/qwen35.cuh")));
    try std.testing.expectEqualStrings("60ffcb12aaf8155d8b82274299a0e21cb007d06e540c47cfeaeb6de2e69a6317", &hash(@embedFile("kernels/cuda/qwen35_draft.cu")));
    try std.testing.expectEqualStrings("873d87646b38a43339817c2aa367a8b279786921b7b5c62bd7754aa80aed0073", &hash(@embedFile("kernels/cuda/qwen35_keep.cu")));
    try std.testing.expectEqualStrings("d1a379a98cf7504a7ed3b3f2d24c8ab6a39fe5511bcded6449e598ce5c3c79e7", &hash(@embedFile("kernels/cuda/scan_rows.cu")));
}
test "dense serving: owned launches and projections retain their reviewed argument sequence" {
    const source = @embedFile("src/families/qwen3_5/cuda_engine.zig") ++ @embedFile("src/families/qwen3_5/cuda_forward.zig");
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(std.testing.allocator);
    var at: usize = 0;
    while (std.mem.indexOfPos(u8, source, at, "try e.")) |start| {
        const rest = source[start..];
        const end = std.mem.indexOfScalar(u8, rest, ';') orelse return error.BadSource;
        const call = rest[0 .. end + 1];
        at = start + 6;
        if (!(std.mem.startsWith(u8, call, "try e.run(e.fns.") or std.mem.startsWith(u8, call, "try e.proj(") or
            std.mem.startsWith(u8, call, "try e.norm("))) continue;
        at = start + end + 1;
        for (call) |c| if (!std.ascii.isWhitespace(c)) {
            try out.append(std.testing.allocator, c);
        };
    }
    try std.testing.expectEqualStrings("b2ed0c08dd1c3c01cbd9eabdc85a977dd26871b0ee64a0d484c40fc8d2941baf", &hash(out.items));
}
