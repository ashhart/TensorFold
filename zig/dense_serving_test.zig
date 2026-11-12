//! Reviewed lane/decode serving bytes and calls; changing them needs a reviewed implementation and receipt.
const std = @import("std");
fn hash(bytes: []const u8) [64]u8 {
    var h: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &h, .{});
    return std.fmt.bytesToHex(h, .lower);
}
test "dense serving: owned CUDA images retain their reviewed lane and decode bytes" {
    try std.testing.expectEqualStrings("2a9b3989f3baaef5eadb56abfb8520700449bd910f2b00345703c22dbf758b8c", &hash(@embedFile("kernels/cuda/experts.cu")));
    try std.testing.expectEqualStrings("578e9a607f678f0547aaeb312d18252ea0608bad7fe71bb2d5241d28adf31bac", &hash(@embedFile("kernels/cuda/experts_pack.cu")));
    try std.testing.expectEqualStrings("5f325dc6d8134624b30b5e99c5764f75beb3c344b5f949d38bfaa41bc8dfe5dc", &hash(@embedFile("kernels/cuda/experts_prefill.cu")));
    try std.testing.expectEqualStrings("a9caeec5b270dcab933a5eae0f500d7aabcd31cde156238225ae064ac92ba2e8", &hash(@embedFile("kernels/cuda/gdn.cu")));
    try std.testing.expectEqualStrings("dba5aae94115571ed48933d80f646fff31b2a7da226569867c74a1dc061d2ca2", &hash(@embedFile("kernels/cuda/lane_gemv.cu")));
    // 191061381's bytes: 82f3dde92 and 191061381 (#2609) add pattn_kernel instances; 94a9a48d6 moves helpers out
    try std.testing.expectEqualStrings("905fff2c9018d778557a60ed2c81c24ce220adf5b092c8c32f8a1782963f9f28", &hash(@embedFile("kernels/cuda/prefill_attention.cu")));
    const group = @embedFile("kernels/cuda/qmm_group.cu");
    try std.testing.expectEqualStrings("ceff4ee13e97b5aa93cbe7a1edc9a524c78aaa145dd1e5c82dcb05a3011aae23", &hash(group[0..16095]));
    try std.testing.expectEqualStrings("8780fd710be3ce2f04e53fa1e69eb062809d3bec6391d9bd49983d0588e374d0", &hash(group));
    try std.testing.expectEqualStrings("45ab2be277c7c17e6cdc6041db45611f2554c6e88f09a295e88751b1c51be1f5", &hash(@embedFile("kernels/cuda/qmm_prefill.cu")));
    // 01b45cd1d's bytes: tf_q35_norm_shift and tf_q35_head_norm_shift for BF16 checkpoints, under the contract (#2383)
    try std.testing.expectEqualStrings("7438579264a99fd0ea99ac242afc68015c8b13a8cd3f10c43b93644376b87205", &hash(@embedFile("kernels/cuda/qwen35.cu")));
    try std.testing.expectEqualStrings("7d916f0bb75d636ed15d8f30a7108cf05a6c5a833bd76beb8b627e6567a641db", &hash(@embedFile("kernels/cuda/qwen35.cuh")));
    // 67711a6aa7's bytes: the opt-in DSpine draft kernels (#2633); default serving unchanged
    try std.testing.expectEqualStrings("df06cb0ef07d12ca2296f7d1691678d0649a148f4abcbb3cb69f0d839e45295c", &hash(@embedFile("kernels/cuda/qwen35_draft.cu")));
    try std.testing.expectEqualStrings("403cf841b89ad61da0818bdad8976ed4b286b3282c8c22df7171921b7e06eb6d", &hash(@embedFile("kernels/cuda/qwen35_keep.cu")));
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
    // 4dd870b84d's sequence: the first input norm keys on g.first + i (#2579); BF16 norms take norm_shift (#2383)
    try std.testing.expectEqualStrings("7be514892c77ce544d809f73d5a7e2dec2fd33bcb40c68ab67ab6e409a1c02f8", &hash(out.items));
}
