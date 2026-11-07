//! Reviewed lane/decode serving bytes and calls; changing them needs a reviewed implementation and receipt.
const std = @import("std");
fn hash(bytes: []const u8) [64]u8 {
    var h: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &h, .{});
    return std.fmt.bytesToHex(h, .lower);
}
test "dense serving: owned CUDA images retain their reviewed lane and decode bytes" {
    try std.testing.expectEqualStrings("f713095ee10bb47477bfe5fa28b95ea7c8971d79a9af867ad0aa5ffd8965bd76", &hash(@embedFile("kernels/cuda/experts.cu")));
    try std.testing.expectEqualStrings("f195f5678edf7d3267ec30e442b1e0048543006e595ad21801afd905df621baf", &hash(@embedFile("kernels/cuda/experts_pack.cu")));
    try std.testing.expectEqualStrings("db560d705729142c66df8dee18fc635f20f09e4f09b8431de265bd6fac68e892", &hash(@embedFile("kernels/cuda/experts_prefill.cu")));
    try std.testing.expectEqualStrings("a9caeec5b270dcab933a5eae0f500d7aabcd31cde156238225ae064ac92ba2e8", &hash(@embedFile("kernels/cuda/gdn.cu")));
    try std.testing.expectEqualStrings("dba5aae94115571ed48933d80f646fff31b2a7da226569867c74a1dc061d2ca2", &hash(@embedFile("kernels/cuda/lane_gemv.cu")));
    try std.testing.expectEqualStrings("278cdf9971af9f08da0aa0fd0016202798dd964ca5bb1df668e7fae8e73f993d", &hash(@embedFile("kernels/cuda/prefill_attention.cu")));
    const group = @embedFile("kernels/cuda/qmm_group.cu");
    try std.testing.expectEqualStrings("f57ba89df5bfc399343772e75feb421b5fe9465089c2189a4391822002b268dd", &hash(group[0..16132]));
    try std.testing.expectEqualStrings("611488f3779dfb5605fd5117add40aaa3e962168e936326d6b02b094cc995e1a", &hash(group));
    try std.testing.expectEqualStrings("0db14eec2fe47677b0e3dab85df5b3682889b20418a3d113051004749b2a3fc7", &hash(@embedFile("kernels/cuda/qmm_prefill.cu")));
    try std.testing.expectEqualStrings("9f39fae379e506dffe7047925a5d13b57041d623be9d36fac25efba5487b7f10", &hash(@embedFile("kernels/cuda/qwen35.cu")));
    try std.testing.expectEqualStrings("7d916f0bb75d636ed15d8f30a7108cf05a6c5a833bd76beb8b627e6567a641db", &hash(@embedFile("kernels/cuda/qwen35.cuh")));
    try std.testing.expectEqualStrings("acfa75d58a121eb389a69eb1f317a8a6b8ddc3adf7489ff295d7de42099ae70e", &hash(@embedFile("kernels/cuda/qwen35_draft.cu")));
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
    try std.testing.expectEqualStrings("b2ed0c08dd1c3c01cbd9eabdc85a977dd26871b0ee64a0d484c40fc8d2941baf", &hash(out.items));
}
