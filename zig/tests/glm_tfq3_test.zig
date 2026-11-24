//! Host and Metal unit tests for GLM-5.3-Flash TFQ3 format loading, run merging, and GPU decode/prefill parity.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const affine = tf.glm.affine;
const quant = tf.glm.quant;
const wide = tf.glm.wide;
const tfq3 = tf.glm.tfq3;

const testing = std.testing;

fn makeMeta(name: []const u8, r: u32, c: u32, bits: u32, sc_off: u32, word_off: u64, word_count: u64) tfq3.NonMoeTensorMeta {
    var m: tfq3.NonMoeTensorMeta = std.mem.zeroes(tfq3.NonMoeTensorMeta);
    @memcpy(m.name[0..name.len], name);
    m.R = r;
    m.C = c;
    m.b_bits = bits;
    m.sc_off = sc_off;
    m.word_off = word_off;
    m.word_count = word_count;
    return m;
}

test "tfq3: affine admission and tally summary recognize 4b and 5b TFQ3 runs" {
    try testing.expect(affine.reads(.kda_in, .{ .bits = 4, .group = 512 }));
    try testing.expect(affine.reads(.shared_expert, .{ .bits = 5, .group = 512 }));
    try testing.expect(!affine.reads(.kda_in, .{ .bits = 3, .group = 512 }));
    try testing.expect(!affine.reads(.embed, .{ .bits = 4, .group = 512 }));
    try testing.expect(!affine.reads(.kv_b, .{ .bits = 4, .group = 512 }));

    const dummy: affine.Ref = .{ .buf = .{ .id = undefined }, .off = 0 };
    const a4: affine.Affine = .{ .w = dummy, .s = dummy, .b = dummy, .n = 4096, .k = 4096, .bits = 4, .group = 512 };
    const a5: affine.Affine = .{ .w = dummy, .s = dummy, .b = dummy, .n = 4096, .k = 4096, .bits = 5, .group = 512 };
    var t: affine.Tally = .{};
    t.add(a4);
    t.add(a4);
    t.add(a5);
    var buf: [128]u8 = undefined;
    try testing.expectEqualStrings("TFQ3 4b x2, TFQ3 5b x1", t.summary(&buf));
}

test "tfq3: planRuns merges contiguous KDA q/k/v, MLA q_b+wq_b, and Shared gate/up" {
    const q = makeMeta("q_proj", 8192, 4096, 4, 0, 0, 100);
    const k = makeMeta("k_proj", 8192, 4096, 4, 2048, 100, 100);
    const v = makeMeta("v_proj", 8192, 4096, 4, 4096, 200, 100);
    const pws = [_]?*const tfq3.NonMoeTensorMeta{ &q, &k, &v, null, null, null };
    const ws = [_]quant.Width{
        .{ .bits = 4, .group = 512 },
        .{ .bits = 4, .group = 512 },
        .{ .bits = 4, .group = 512 },
        .{ .bits = 8, .group = 64 },
        .{ .bits = 8, .group = 64 },
        .{ .bits = 8, .group = 64 },
    };
    var starts: [affine.max_runs + 1]usize = undefined;
    const runs = tfq3.planRuns(6, &pws, &ws, true, true, &starts);
    try testing.expectEqual(@as(usize, 2), runs);
    try testing.expectEqualSlices(usize, &.{ 0, 3, 6 }, starts[0 .. runs + 1]);

    const qb = makeMeta("q_b_proj", 16384, 1536, 4, 0, 0, 400);
    const wqb = makeMeta("indexer.wq_b", 4096, 1536, 4, 1536, 400, 100);
    const mla_pws = [_]?*const tfq3.NonMoeTensorMeta{ &qb, &wqb };
    const mla_ws = [_]quant.Width{ .{ .bits = 4, .group = 512 }, .{ .bits = 4, .group = 512 } };
    try testing.expectEqual(@as(usize, 1), tfq3.planRuns(2, &mla_pws, &mla_ws, true, false, &starts));

    const gap_k = makeMeta("k_proj", 8192, 4096, 4, 2048, 105, 100);
    const gap_pws = [_]?*const tfq3.NonMoeTensorMeta{ &q, &gap_k };
    try testing.expectEqual(@as(usize, 2), tfq3.planRuns(2, &gap_pws, mla_ws[0..2], true, false, &starts));
}

test "tfq3: dry-run loader validates magic, version, nonmoe lookup, and layer has_5b" {
    const gpa = testing.allocator;
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try std.fmt.allocPrint(gpa, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    defer gpa.free(dir);

    var tbl: [16384]u8 = @splat(0);
    std.mem.writeInt(u32, tbl[0..4], tfq3.magic_v3_tables, .little);
    std.mem.writeInt(u32, tbl[4..8], tfq3.format_version, .little);
    try tmp.dir.writeFile(io, .{ .sub_path = "pw_v3_tables.bin", .data = &tbl });

    var nm_bytes: [16384]u8 = @splat(0);
    const nm_hdr: tfq3.NonMoeHeader = .{
        .magic = tfq3.magic_nonmoe,
        .version = tfq3.format_version,
        .num_tensors = 1,
        .pad0 = 0,
        .off_meta = 128,
        .off_scales = 4096,
        .off_weights = 8192,
        .total_scale_halfs = 64,
        .total_weight_words = 128,
        .pad1 = 0,
    };
    @memcpy(nm_bytes[0..@sizeOf(tfq3.NonMoeHeader)], std.mem.asBytes(&nm_hdr));
    const nm_m = makeMeta("lm_head", 154880, 4096, 4, 0, 0, 128);
    @memcpy(nm_bytes[128 .. 128 + @sizeOf(tfq3.NonMoeTensorMeta)], std.mem.asBytes(&nm_m));
    try tmp.dir.writeFile(io, .{ .sub_path = "nonmoe.tffpw", .data = &nm_bytes });

    var l_bytes: [16384]u8 = @splat(0);
    const lh: tfq3.LayerHeader = .{
        .magic = tfq3.magic_layer,
        .version = tfq3.format_version,
        .layer_idx = 3,
        .num_experts = 2,
        .num_tensors = 6,
        .hidden_size = 4096,
        .moe_inter = 2048,
        .reserved0 = 0,
        .raw_fp8_bytes = 0,
        .stream_words = 128,
        .scale_halfs = 64,
        .off_meta = 256,
        .off_scales = 4096,
        .off_weights = 8192,
        .reserved1 = .{ 0, 0, 0, 0 },
        .meta_tag = @splat(0),
    };
    @memcpy(l_bytes[0..@sizeOf(tfq3.LayerHeader)], std.mem.asBytes(&lh));
    var ems = [_]tfq3.ExpertMeta{ std.mem.zeroes(tfq3.ExpertMeta), std.mem.zeroes(tfq3.ExpertMeta) };
    ems[0].b_bits = 5;
    ems[1].b_bits = 4;
    @memcpy(l_bytes[256 .. 256 + 2 * @sizeOf(tfq3.ExpertMeta)], std.mem.sliceAsBytes(&ems));
    try tmp.dir.writeFile(io, .{ .sub_path = "layer_03.tffpw", .data = &l_bytes });

    ems[0].b_bits = 4;
    ems[1].b_bits = 3;
    @memcpy(l_bytes[256 .. 256 + 2 * @sizeOf(tfq3.ExpertMeta)], std.mem.sliceAsBytes(&ems));
    try tmp.dir.writeFile(io, .{ .sub_path = "layer_07.tffpw", .data = &l_bytes });

    var arena_st = std.heap.ArenaAllocator.init(gpa);
    defer arena_st.deinit();
    var bufs: std.ArrayList(mtl.Buffer) = .empty;
    defer bufs.deinit(gpa);
    var maps: std.ArrayList(tfq3.MappedShard) = .empty;
    defer maps.deinit(gpa);
    var total: usize = 0;

    try testing.expect(tfq3.hasLayer(dir, 3));
    try testing.expect(!tfq3.hasLayer(dir, 99));
    const tb = try tfq3.loadV3Tables(gpa, undefined, &bufs, &maps, &total, dir, true);
    try testing.expect(tb != null);
    const nm = (try tfq3.loadNonMoe(gpa, arena_st.allocator(), undefined, &bufs, &maps, &total, dir, tb, true)).?;
    try testing.expect(nm.find("lm_head") != null);
    try testing.expect(nm.find("missing") == null);
    const l3 = try tfq3.loadLayer(gpa, undefined, &bufs, &maps, &total, dir, 3, tb, true);
    const l7 = try tfq3.loadLayer(gpa, undefined, &bufs, &maps, &total, dir, 7, tb, true);
    try testing.expect(l3.has_5b);
    try testing.expect(!l7.has_5b);
    try testing.expectEqual(@as(usize, 4 * 16384), total);

    std.mem.writeInt(u32, tbl[0..4], 0xDEADBEEF, .little);
    try tmp.dir.writeFile(io, .{ .sub_path = "pw_v3_tables.bin", .data = &tbl });
    try testing.expectError(error.UnsupportedQuantization, tfq3.loadV3Tables(gpa, undefined, &bufs, &maps, &total, dir, true));
}

test "tfq3: source expansion selects M5 NAX vs pre-M5 TF_SIMD_FRAGS cleanly" {
    const gpa = testing.allocator;
    const m5 = try tfq3.buildSource(gpa, true, wide.glue_math);
    defer gpa.free(m5);
    const m4 = try tfq3.buildSource(gpa, false, wide.glue_math);
    defer gpa.free(m4);
    try testing.expect(std.mem.indexOf(u8, m5, "#define TF_SIMD_FRAGS 1") == null);
    try testing.expect(std.mem.indexOf(u8, m4, "#define TF_SIMD_FRAGS 1") != null);
    try testing.expect(std.mem.indexOf(u8, m5, "#include \"../nax.h\"") == null);
    try testing.expect(std.mem.indexOf(u8, m4, "pw_stage_b(") != null);
}

test "tfq3: Metal compile and decode vs prefill numerical parity on GPU" {
    if (comptime !@hasDecl(mtl.Device, "init")) return error.SkipZigTest;
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    defer queue.deinit();
    const gpa = testing.allocator;

    var pipes = try tfq3.Pipes.init(gpa, device, wide.glue_math);
    defer pipes.deinit();

    const alt_src = try tfq3.buildSource(gpa, !device.tensorUnits(), wide.glue_math);
    defer gpa.free(alt_src);
    const alt_lib = try mtl.Library.fromSource(device, alt_src, mtl.CompileOptions.mlx());
    defer alt_lib.deinit();

    const n: u32 = 128;
    const k: u32 = 128;
    const rows: u32 = 16;
    const w_buf = try device.buffer(8 * 8 * 32 * 4, mtl.ResourceOptions.shared);
    defer w_buf.deinit();
    const s_buf = try device.buffer(16, mtl.ResourceOptions.shared);
    defer s_buf.deinit();
    const t_buf = try device.buffer(16384, mtl.ResourceOptions.shared);
    defer t_buf.deinit();
    const x_buf = try device.buffer(rows * k * 2, mtl.ResourceOptions.shared);
    defer x_buf.deinit();
    const y_dec = try device.buffer(rows * n * 2, mtl.ResourceOptions.shared);
    defer y_dec.deinit();
    const y_pre = try device.buffer(rows * n * 2, mtl.ResourceOptions.shared);
    defer y_pre.deinit();

    for (w_buf.slice(u32, 8 * 8 * 32), 0..) |*w, i| w.* = @truncate((i *% 2654435761) ^ 0xA55A1234);
    s_buf.slice(u16, 8)[0] = @bitCast(@as(f16, 0.25));
    const t_bytes = t_buf.slice(u8, 16384);
    @memset(t_bytes, 0);
    const lut4: [*]align(1) u16 = @ptrCast(t_bytes.ptr + 1088);
    for (0..2048) |i| {
        const v: f16 = @floatFromInt(@as(i32, @intCast(i % 15)) - 7);
        lut4[i] = @bitCast(v * 0.125);
    }
    const c64_4: [*]align(1) u16 = @ptrCast(t_bytes.ptr + 9792);
    for (0..64) |i| {
        c64_4[i * 4 + 0] = @bitCast(@as(f16, if ((i & 1) != 0) -1.0 else 1.0));
        c64_4[i * 4 + 1] = @bitCast(@as(f16, if ((i & 2) != 0) -1.0 else 1.0));
        c64_4[i * 4 + 2] = @bitCast(@as(f16, 0.01));
        c64_4[i * 4 + 3] = @bitCast(@as(f16, -0.01));
    }
    for (x_buf.slice(u16, rows * k), 0..) |*x, i| {
        const f: f32 = @as(f32, @floatFromInt(@as(i32, @intCast((i * 11) % 19)) - 9)) * 0.1;
        x.* = @intCast(@as(u32, @bitCast(f)) >> 16);
    }

    const cb = queue.commandBuffer();
    const enc = cb.compute(.serial);
    var r0: u32 = 0;
    while (r0 < rows) : (r0 += 4) {
        tfq3.dispatch.dense(&pipes, enc, .{ .buf = w_buf }, .{ .buf = s_buf }, .{ .buf = t_buf }, 4, .{ .buf = x_buf, .off = r0 * k * 2 }, .{ .buf = y_dec, .off = r0 * n * 2 }, n, k, 4, k, n);
    }
    tfq3.dispatch.dense(&pipes, enc, .{ .buf = w_buf }, .{ .buf = s_buf }, .{ .buf = t_buf }, 4, .{ .buf = x_buf }, .{ .buf = y_pre }, n, k, rows, k, n);
    enc.end();
    cb.commit();
    cb.wait();
    try testing.expect(cb.failure() == null);

    const a = y_dec.slice(u16, rows * n);
    const b = y_pre.slice(u16, rows * n);
    var max_diff: f32 = 0;
    for (a, b) |av, bv| {
        const af: f32 = @bitCast(@as(u32, av) << 16);
        const bf: f32 = @bitCast(@as(u32, bv) << 16);
        max_diff = @max(max_diff, @abs(af - bf));
    }
    try testing.expect(max_diff < 0.05);
}
