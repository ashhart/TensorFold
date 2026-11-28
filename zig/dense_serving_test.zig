//! Reviewed lane/decode serving bytes and calls; changing them needs a reviewed implementation and receipt.
const std = @import("std");
fn hash(bytes: []const u8) [64]u8 {
    var h: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &h, .{});
    return std.fmt.bytesToHex(h, .lower);
}
test "dense serving: owned CUDA images retain their reviewed lane and decode bytes" {
    try std.testing.expectEqualStrings("adc0f015f08b49e65df25fe17ef2489dd924d8ba6cfccd1f008be5cd1edbd464", &hash(@embedFile("kernels/cuda/experts.cu")));
    try std.testing.expectEqualStrings("578e9a607f678f0547aaeb312d18252ea0608bad7fe71bb2d5241d28adf31bac", &hash(@embedFile("kernels/cuda/experts_pack.cu")));
    try std.testing.expectEqualStrings("5f325dc6d8134624b30b5e99c5764f75beb3c344b5f949d38bfaa41bc8dfe5dc", &hash(@embedFile("kernels/cuda/experts_prefill.cu")));
    try std.testing.expectEqualStrings("e5ef9dcce75c478e59e476d2fa8747baee2317bd283b0756f125ea050dad9228", &hash(@embedFile("kernels/cuda/gdn.cu")));
    try std.testing.expectEqualStrings("dba5aae94115571ed48933d80f646fff31b2a7da226569867c74a1dc061d2ca2", &hash(@embedFile("kernels/cuda/lane_gemv.cu")));
    // #2239 adds head-64 attention for Qwen3-ASR; #3790 reuses it for GLM; existing instances keep their SASS
    try std.testing.expectEqualStrings("9d9faedafd8e06afd667839a187118b6b220c794c2948532f0a849f901627a23", &hash(@embedFile("kernels/cuda/prefill_attention.cu")));
    const group = @embedFile("kernels/cuda/qmm_group.cu");
    try std.testing.expectEqualStrings("ceff4ee13e97b5aa93cbe7a1edc9a524c78aaa145dd1e5c82dcb05a3011aae23", &hash(group[0..16095]));
    try std.testing.expectEqualStrings("8780fd710be3ce2f04e53fa1e69eb062809d3bec6391d9bd49983d0588e374d0", &hash(group));
    // #2825 hoists copy addresses before the K loop; copy order and math match prefill_kernel.
    try std.testing.expectEqualStrings("e3724114b87cae6f6f761b2f86357f05b1693d84a0233e4d58ce46349fa853f6", &hash(@embedFile("kernels/cuda/qmm_prefill.cu")));
    // KIA14 adds all_rows (false keeps selection/arithmetic); #4706 adds TF_QWEN_PDL prologues
    try std.testing.expectEqualStrings("7a42deb533fcb6558edb3edc765f3f6f53f7caa63759fc3c618dcdbb40382808", &hash(@embedFile("kernels/cuda/qwen35.cu")));
    // #1879's bytes: a comment names cuda_scratch.zig, where max_chunks moved; the code is unchanged
    try std.testing.expectEqualStrings("f9e4a040cf219aac8db211df25d264dba390fce4a7e8da45557652c6ff0f28ba", &hash(@embedFile("kernels/cuda/qwen35.cuh")));
    // 67711a6aa7's bytes: the opt-in DSpine draft kernels (#2633); default serving unchanged
    try std.testing.expectEqualStrings("df06cb0ef07d12ca2296f7d1691678d0649a148f4abcbb3cb69f0d839e45295c", &hash(@embedFile("kernels/cuda/qwen35_draft.cu")));
    try std.testing.expectEqualStrings("403cf841b89ad61da0818bdad8976ed4b286b3282c8c22df7171921b7e06eb6d", &hash(@embedFile("kernels/cuda/qwen35_keep.cu")));
    try std.testing.expectEqualStrings("d1a379a98cf7504a7ed3b3f2d24c8ab6a39fe5511bcded6449e598ce5c3c79e7", &hash(@embedFile("kernels/cuda/scan_rows.cu")));
}
test "dense serving: owned launches and projections retain their reviewed argument sequence" {
    const source = @embedFile("src/families/qwen3_5/cuda_engine.zig") ++
        @embedFile("src/families/qwen3_5/cuda_scan.zig") ++
        @embedFile("src/families/qwen3_5/cuda_forward.zig");
    const prefixes = [_][]const u8{ "try e.run(e.fns.", "try e.runPdl(e.fns.", "try e.run(scan,", "try e.proj(", "try e.norm(", "try mhc." };
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(std.testing.allocator);
    var at: usize = 0;
    while (true) {
        var start: ?usize = null;
        for (prefixes) |prefix| if (std.mem.indexOfPos(u8, source, at, prefix)) |candidate| {
            if (start) |previous| {
                if (candidate < previous) start = candidate;
            } else start = candidate;
        };
        const call_start = start orelse break;
        const rest = source[call_start..];
        const end = std.mem.indexOfScalar(u8, rest, ';') orelse return error.BadSource;
        const call = rest[0 .. end + 1];
        at = call_start + end + 1;
        for (call) |c| if (!std.ascii.isWhitespace(c)) {
            try out.append(std.testing.allocator, c);
        };
    }
    // The scan, retained-row MTP tap, opt-in TF_QWEN_PDL glue, and schedule-driven norm and mHC launches are pinned.
    try std.testing.expectEqualStrings("194b2a2b7597b12289250bb9100950c4c2c962f0a2487945511060c9ef7d1b29", &hash(out.items));
}
