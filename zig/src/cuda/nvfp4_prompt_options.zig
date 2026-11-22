//! NVFP4 prompt symbol and staging choices; defaults preserve the existing automatic depth.
const std = @import("std");

pub fn promptSmem(parts: usize, wn: usize) u32 {
    const stage: u32 = @intCast((64 * 8 + wn * 2 * parts * 36) * 16);
    const stages: u32 = if (stage * 4 <= 49152) 4 else if (stage * 3 <= 49152) 3 else 2;
    return stages * stage;
}

pub fn promptSmemOpt(parts: usize, wn: usize, stages3: bool) u32 {
    if (!stages3) return promptSmem(parts, wn);
    return 3 * @as(u32, @intCast((64 * 8 + wn * 2 * parts * 36) * 16));
}

pub fn promptNames(native_fp: bool, stages3: bool) [3][:0]const u8 {
    if (native_fp) return if (stages3)
        .{ "tf_nvfp4_prompt_relu2_native_s3", "tf_nvfp4_prompt_swiglu_native_s3", "tf_nvfp4_prompt_down_native_s3" }
    else
        .{ "tf_nvfp4_prompt_relu2_native", "tf_nvfp4_prompt_swiglu_native", "tf_nvfp4_prompt_down_native" };
    return if (stages3)
        .{ "tf_nvfp4_prompt_relu2_s3", "tf_nvfp4_prompt_swiglu_s3", "tf_nvfp4_prompt_down_s3" }
    else
        .{ "tf_nvfp4_prompt_relu2", "tf_nvfp4_prompt_swiglu", "tf_nvfp4_prompt_down" };
}

test "NVFP4 prompt options: arithmetic and depth are independent, and deep tiles fit their actual stages" {
    try std.testing.expectEqual(@as(u32, 3 * 17408), promptSmemOpt(1, 8, true));
    try std.testing.expectEqual(@as(u32, 3 * 17408), promptSmemOpt(2, 4, true));
    try std.testing.expectEqual(promptSmem(1, 4), promptSmemOpt(1, 4, false));
    const expected = [4][3][:0]const u8{
        .{ "tf_nvfp4_prompt_relu2", "tf_nvfp4_prompt_swiglu", "tf_nvfp4_prompt_down" },
        .{ "tf_nvfp4_prompt_relu2_native", "tf_nvfp4_prompt_swiglu_native", "tf_nvfp4_prompt_down_native" },
        .{ "tf_nvfp4_prompt_relu2_s3", "tf_nvfp4_prompt_swiglu_s3", "tf_nvfp4_prompt_down_s3" },
        .{ "tf_nvfp4_prompt_relu2_native_s3", "tf_nvfp4_prompt_swiglu_native_s3", "tf_nvfp4_prompt_down_native_s3" },
    };
    for (expected, 0..) |want, i| {
        const got = promptNames(i & 1 != 0, i & 2 != 0);
        for (got, want) |name, w| try std.testing.expectEqualStrings(w, name);
    }
}
