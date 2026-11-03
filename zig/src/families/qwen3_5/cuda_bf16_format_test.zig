//! Dense BF16 admission through the production CUDA geometry parser.
const std = @import("std");
const geometry = @import("cuda_geometry.zig");
const a = std.testing.allocator;
pub const fixture =
    \\{"model_type":"qwen3_5","dtype":"bfloat16","tie_word_embeddings":false,
    \\ "text_config":{"model_type":"qwen3_5_text","attention_bias":false,"attn_output_gate":true,"hidden_act":"silu",
    \\ "head_dim":256,"linear_key_head_dim":128,"linear_value_head_dim":128,"linear_conv_kernel_dim":4,
    \\ "hidden_size":5120,"intermediate_size":17408,"num_hidden_layers":4,"vocab_size":248320,
    \\ "linear_num_key_heads":16,"linear_num_value_heads":48,"num_attention_heads":24,"num_key_value_heads":4,
    \\ "rms_norm_eps":1e-6,"max_position_embeddings":262144,"tie_word_embeddings":false,
    \\ "layer_types":["linear_attention","linear_attention","linear_attention","full_attention"],
    \\ "rope_parameters":{"rope_type":"default","rope_theta":10000000,"partial_rotary_factor":0.25}}}
;
test "BF16 scoring: plain BF16 Qwen config reaches the native geometry loader" {
    const g = try geometry.parse(a, fixture);
    try std.testing.expectEqual(@as(usize, 5120), g.hidden);
    try std.testing.expectEqual(@as(usize, 48), g.v_heads);
    try std.testing.expectEqual(geometry.Kind.bf16, g.kinds.head);
    for (0..g.layers) |i| for (std.enums.values(geometry.Slot)) |slot| {
        try std.testing.expectEqual(geometry.Kind.bf16, g.kinds.of(i, slot));
    };
}
test "BF16 scoring: an unquantized checkpoint of another dtype is refused" {
    for ([_][]const u8{ "float16", "float32" }) |dtype| {
        const text = try std.mem.replaceOwned(u8, a, fixture, "bfloat16", dtype);
        defer a.free(text);
        try std.testing.expectError(error.UnsupportedQwenQuantization, geometry.parse(a, text));
    }
}

test "BF16 scoring: native admission refuses unqualified image, reduced precision and learning modes" {
    const format = @import("cuda_bf16_format.zig");
    try format.options(true, false, false, false);
    try std.testing.expectError(error.QwenBf16VisionUnqualified, format.options(true, true, false, false));
    try std.testing.expectError(error.UnsupportedQwenBf16Mode, format.options(true, false, true, false));
    try std.testing.expectError(error.UnsupportedQwenBf16Mode, format.options(true, false, false, true));
    try format.options(false, true, true, true);
}

test "BF16 scoring: actual native open refuses vision and input quantization before touching CUDA" {
    const native = @import("cuda_native.zig");
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "config.json", .data = fixture });
    const path = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    defer a.free(path);
    try std.testing.expectError(error.QwenBf16VisionUnqualified, native.open(a, std.testing.io, undefined, path, null, .{ .context = 8192, .drafts = true, .vision = .{ .image_tokens = 4096, .max_images = 1 } }));
    try std.testing.expectError(error.UnsupportedQwenBf16Mode, native.open(a, std.testing.io, undefined, path, null, .{ .context = 8192, .drafts = true, .prompt_int8 = true }));
    try std.testing.expect(native.explain(null, error.QwenBf16VisionUnqualified) != null);
}

test "BF16 scoring: production stats launcher preserves argument widths and refuses invalid frames before CUDA" {
    const cuda = @import("cuda");
    const Fake = struct {
        var calls: usize = 0;
        var rows: u32 = 0;
        var words: [5]u64 = undefined;
        fn launch(_: cuda.abi.Function, gx: c_uint, _: c_uint, _: c_uint, bx: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
            if (bx != 256) return 1;
            calls += 1;
            rows = gx;
            inline for ([_]type{ u64, u32, u64, u32, u64 }, 0..) |T, i| words[i] = @as(*const T, @ptrCast(@alignCast(args.?[i].?))).*;
            return 0;
        }
    };
    var d: cuda.Driver = undefined;
    d.api.cuLaunchKernel = Fake.launch;
    var k: cuda.qwen35_bf16_rows.Kernels = undefined;
    k.stats_fn = .{ .d = &d, .handle = @ptrFromInt(1) };
    const stream = cuda.Stream{ .d = &d, .handle = null };
    Fake.calls = 0;
    for ([_]usize{ 1, 4, 5, 128 }) |rows| {
        try k.stats(stream, 0x1000, 248320, 0x2000, 20, 0x3000, rows);
        try std.testing.expectEqual(@as(u32, @intCast(rows)), Fake.rows);
        try std.testing.expectEqualSlices(u64, &.{ 0x1000, 248320, 0x2000, 20, 0x3000 }, &Fake.words);
    }
    const before = Fake.calls;
    try std.testing.expectError(error.RowShape, k.stats(stream, 0x1000, 248320, 0x2000, 20, 0x3000, 129));
    try std.testing.expectError(error.RowShape, k.stats(stream, 0x1000, 248320, 0x2000, 21, 0x3000, 1));
    try std.testing.expectError(error.RowShape, k.stats(stream, 0x1000, 248320, 0x2000, 20, 0x3001, 1));
    try std.testing.expectEqual(before, Fake.calls);
}

test "BF16 scoring: FP32 GDN scale admission retains stored precision and old format refusals" {
    const core = @import("core");
    const format = @import("cuda_bf16_format.zig");
    const weights: [128]f32 = @splat(1.00001);
    var t: core.checkpoint.Tensor = .{ .dtype = .f32, .rank = 1, .shape = @splat(0), .bytes = std.mem.sliceAsBytes(&weights) };
    t.shape[0] = 128;
    try std.testing.expect(try format.normIsF32(t, true, 128));
    try std.testing.expectError(error.UnexpectedTensor, format.normIsF32(t, false, 128));
    t.shape[0] = 127;
    try std.testing.expectError(error.UnexpectedTensor, format.normIsF32(t, true, 128));
    t.shape[0] = 128;
    t.dtype = .bf16;
    try std.testing.expect(!try format.normIsF32(t, true, 128));
    t.dtype = .f16;
    try std.testing.expectError(error.UnexpectedTensor, format.normIsF32(t, true, 128));
}
