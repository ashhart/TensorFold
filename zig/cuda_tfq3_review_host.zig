//! CUDA TFQ3 review regressions: format identity, lifecycle failures and checkpoint arithmetic.
const std = @import("std");
const cuda = @import("cuda");
const fmt = cuda.tfq3_format;
const t = std.testing;

fn bf16(x: f32) f32 {
    const u: u32 = @bitCast(x);
    return @bitCast((u + 0x7fff + ((u >> 16) & 1)) & 0xffff0000);
}

test "CUDA TFQ3 review: routed SiLU has its BF16 point before multiplication" {
    const sl: f32 = 0.5 / (1 + @exp(@as(f32, -0.5)));
    const want = bf16(bf16(sl) * 1.5);
    try t.expectEqual(@as(f32, 0.46484375), want);
    try t.expect(bf16(sl * 1.5) != want);
    const source = @embedFile("kernels/cuda/tfq3_common.cuh");
    try t.expect(std.mem.indexOf(u8, source, "if (act_mode != 0u) sg =") == null);
    try t.expect(std.mem.indexOf(u8, source, "sg = __bfloat162float(__float2bfloat16(sg));") != null);
}

fn moeShard() ![]u8 {
    const bytes = try t.allocator.alloc(u8, 4 * fmt.section_align);
    @memset(bytes, 0);
    const words = fmt.tensorWords(128, 128, 4);
    const h: fmt.TfpwHeader = .{
        .magic = fmt.magic_tfpw,
        .version = 3,
        .layer_idx = 0,
        .num_experts = 1,
        .num_tensors = 3,
        .hidden_size = 128,
        .moe_inter = 128,
        .reserved0 = 0,
        .raw_fp8_bytes = 0,
        .stream_words = 3 * words,
        .scale_halfs = 3,
        .off_meta = 144,
        .off_scales = fmt.section_align,
        .off_weights = 2 * fmt.section_align,
        .reserved1 = @splat(0),
        .meta_tag = @splat(0),
    };
    @memcpy(bytes[0..144], std.mem.asBytes(&h));
    var m = std.mem.zeroes(fmt.TfpwExpertMeta);
    m.b_bits = 4;
    m.up_word_off = words;
    m.down_word_off = 2 * words;
    m.up_sc_off = 1;
    m.down_sc_off = 2;
    @memcpy(bytes[144..208], std.mem.asBytes(&m));
    return bytes;
}

test "CUDA TFQ3 review: decoder layer identity is checked, MTP is unconstrained" {
    const bytes = try moeShard();
    defer t.allocator.free(bytes);
    _ = try fmt.validateTfpw(bytes, 1, 128, 128, 0);
    try t.expectError(error.Tfq3LayerMismatch, fmt.validateTfpw(bytes, 1, 128, 128, 1));
    _ = try fmt.validateTfpw(bytes, 1, 128, 128, null);
}

fn enumerate(gpa: std.mem.Allocator, dir: []const u8) !void {
    const paths = try fmt.listShards(gpa, t.io, dir, "");
    defer fmt.freePaths(gpa, paths);
    try t.expectEqual(@as(usize, 2), paths.len);
    try t.expectEqualStrings("a.tffpw", std.fs.path.basename(paths[0]));
    try t.expectEqualStrings("b.tffpw", std.fs.path.basename(paths[1]));
}

test "CUDA TFQ3 review: shard enumeration owns every joined path on allocation failure" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(t.io, ".", t.allocator);
    defer t.allocator.free(dir);
    try tmp.dir.writeFile(t.io, .{ .sub_path = "b.tffpw", .data = "b" });
    try tmp.dir.writeFile(t.io, .{ .sub_path = "a.tffpw", .data = "a" });
    try tmp.dir.writeFile(t.io, .{ .sub_path = "ignored.bin", .data = "x" });
    try t.checkAllAllocationFailures(t.allocator, enumerate, .{dir});
}

test "CUDA TFQ3 review: Geometry.read keeps an MLX checkpoint with orphan tables" {
    const config =
        \\{"model_type":"qwen3_5","quantization":{"group_size":64,"bits":4,"mode":"affine"},"tie_word_embeddings":false,
        \\ "text_config":{"model_type":"qwen3_5_text","attention_bias":false,"attn_output_gate":true,"hidden_act":"silu",
        \\ "head_dim":256,"linear_key_head_dim":128,"linear_value_head_dim":128,"linear_conv_kernel_dim":4,
        \\ "hidden_size":5120,"intermediate_size":17408,"num_hidden_layers":4,"vocab_size":248320,"linear_num_key_heads":16,
        \\ "linear_num_value_heads":48,"num_attention_heads":24,"num_key_value_heads":4,"rms_norm_eps":1e-06,
        \\ "max_position_embeddings":262144,"tie_word_embeddings":false,
        \\ "layer_types":["linear_attention","linear_attention","linear_attention","full_attention"],
        \\ "rope_parameters":{"rope_type":"default","rope_theta":10000000,"partial_rotary_factor":0.25}}}
    ;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(t.io, ".", t.allocator);
    defer t.allocator.free(dir);
    try tmp.dir.writeFile(t.io, .{ .sub_path = "config.json", .data = config });
    const Geometry = @import("qwen35").geometry.Geometry;
    const before = try Geometry.read(t.allocator, t.io, dir);
    var tables: [fmt.tables_file_bytes]u8 = @splat(0);
    std.mem.writeInt(u32, tables[0..4], fmt.magic_tables, .little);
    std.mem.writeInt(u32, tables[4..8], 3, .little);
    try tmp.dir.writeFile(t.io, .{ .sub_path = "pw_v3_tables.bin", .data = &tables });
    const after = try Geometry.read(t.allocator, t.io, dir);
    try t.expect(!after.tfq3);
    try t.expectEqualDeep(before, after);
}

test "CUDA TFQ3 review: output sign periods repeat through narrowed tail views" {
    const base: cuda.tfq3.DenseWeight = .{
        .words = 0,
        .scales = 0,
        .tables = 0,
        .n = 248320,
        .k = 128,
        .bits = 4,
        .sr_split = 8192,
    };
    const tail = try base.sliceRows(247936, 384);
    try t.expectEqual(@as(u32, 8192), tail.sr_split);
    try t.expectEqual(@as(u32, 247936), tail.row_off);
    for ([_]u32{ 2047, 2048, 8191, 8192, 16383, 16384, 247936 }) |row| {
        const period: u32 = 8192;
        const local = row % period;
        if (row >= 2 * period) try t.expect(row - period != local);
    }
    const source = @embedFile("kernels/cuda/tfq3_common.cuh");
    try t.expect(std.mem.indexOf(u8, source, "r0 % sr_split") != null);
}
