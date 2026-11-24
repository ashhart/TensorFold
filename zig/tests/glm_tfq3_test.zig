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

test {
    _ = tfq3.check; // the file checks' refusal fixtures, also run off macOS by test-glm-tfq3-host
}

fn nonMoeFile() [32768 + 8192]u8 {
    var out: [32768 + 8192]u8 = @splat(0);
    const h: tfq3.NonMoeHeader = .{ .magic = tfq3.magic_nonmoe, .version = tfq3.format_version, .num_tensors = 1, .pad0 = 0, .off_meta = 128, .off_scales = 16384, .off_weights = 32768, .total_scale_halfs = 1, .total_weight_words = 2048, .pad1 = 0 };
    @memcpy(out[0..@sizeOf(tfq3.NonMoeHeader)], std.mem.asBytes(&h));
    const m = makeMeta("lm_head", 128, 128, 4, 0, 0, 2048);
    @memcpy(out[128..][0..@sizeOf(tfq3.NonMoeTensorMeta)], std.mem.asBytes(&m));
    return out;
}

/// layer_NN.tffpw with header index `idx`: one expert at `bits`, gate, up and down back to back.
fn layerFile(gpa: std.mem.Allocator, idx: u32, bits: u32) ![]u8 {
    const words: usize = 2048 * 4096 * @as(usize, bits) / 32;
    const out = try gpa.alloc(u8, 49152 + 3 * words * 4);
    @memset(out, 0);
    var h = std.mem.zeroes(tfq3.LayerHeader);
    h.magic = tfq3.magic_layer;
    h.version = tfq3.format_version;
    h.layer_idx = idx;
    h.num_experts = 1;
    h.num_tensors = 3;
    h.hidden_size = 4096;
    h.moe_inter = 2048;
    h.stream_words = 3 * words;
    h.scale_halfs = 3 * 512;
    h.off_meta = 16384;
    h.off_scales = 32768;
    h.off_weights = 49152;
    var e = std.mem.zeroes(tfq3.ExpertMeta);
    e.b_bits = bits;
    e.up_word_off = words;
    e.down_word_off = 2 * words;
    e.up_sc_off = 512;
    e.down_sc_off = 1024;
    @memcpy(out[0..@sizeOf(tfq3.LayerHeader)], std.mem.asBytes(&h));
    @memcpy(out[16384..][0..@sizeOf(tfq3.ExpertMeta)], std.mem.asBytes(&e));
    return out;
}

test "tfq3: the loader checks every file on the dry path and refuses a wrong layer, short records and a bad table" {
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
    const nm_bytes = nonMoeFile();
    try tmp.dir.writeFile(io, .{ .sub_path = "nonmoe.tffpw", .data = &nm_bytes });
    for ([_][3]u32{ .{ 3, 3, 5 }, .{ 7, 3, 4 }, .{ 9, 9, 4 } }) |f| { // file index, header index, bits
        const bytes = try layerFile(gpa, f[1], f[2]);
        defer gpa.free(bytes);
        var name: [32]u8 = undefined;
        try tmp.dir.writeFile(io, .{ .sub_path = try std.fmt.bufPrint(&name, "layer_{d:0>2}.tffpw", .{f[0]}), .data = bytes });
    }

    var arena_st = std.heap.ArenaAllocator.init(gpa);
    defer arena_st.deinit();
    var bufs: std.ArrayList(mtl.Buffer) = .empty;
    defer bufs.deinit(gpa);
    var maps: std.ArrayList(tfq3.MappedShard) = .empty;
    defer maps.deinit(gpa);
    var total: usize = 0;
    const want = struct {
        fn at(i: usize) tfq3.check.LayerWant {
            return .{ .layer = i, .experts = 1, .hidden = 4096, .inter = 2048 };
        }
    };

    try testing.expect(tfq3.hasLayer(dir, 3));
    try testing.expect(!tfq3.hasLayer(dir, 99));
    const tb = try tfq3.loadV3Tables(gpa, undefined, &bufs, &maps, &total, dir, true);
    try testing.expect(tb != null);
    const nm = (try tfq3.loadNonMoe(gpa, arena_st.allocator(), undefined, &bufs, &maps, &total, dir, tb, true)).?;
    try testing.expect(nm.find("lm_head") != null);
    try testing.expect(nm.find("missing") == null);
    try testing.expect((try tfq3.loadLayer(gpa, undefined, &bufs, &maps, &total, dir, want.at(3), tb, true)).has_5b);
    try testing.expect(!(try tfq3.loadLayer(gpa, undefined, &bufs, &maps, &total, dir, want.at(9), tb, true)).has_5b);
    // E326: layer_07.tffpw holding layer 3's header is refused, not loaded as layer 7
    try testing.expectError(error.Tfq3Layer, tfq3.loadLayer(gpa, undefined, &bufs, &maps, &total, dir, want.at(7), tb, true));
    try testing.expectError(error.UnsupportedQuantization, tfq3.loadLayer(gpa, undefined, &bufs, &maps, &total, dir, want.at(3), null, true));
    try testing.expectEqual(@as(usize, 0), maps.items.len + bufs.items.len); // the dry path keeps no mapping
    const counted = total;

    var short = nm_bytes;
    std.mem.writeInt(u64, short[16..24], 16384, .little); // off_meta: E326's record past a 64-byte file
    try tmp.dir.writeFile(io, .{ .sub_path = "nonmoe.tffpw", .data = short[0..64] });
    try testing.expectError(error.Tfq3Truncated, tfq3.loadNonMoe(gpa, arena_st.allocator(), undefined, &bufs, &maps, &total, dir, tb, true));
    std.mem.writeInt(u32, tbl[0..4], 0xDEADBEEF, .little);
    try tmp.dir.writeFile(io, .{ .sub_path = "pw_v3_tables.bin", .data = &tbl });
    try testing.expectError(error.UnsupportedQuantization, tfq3.loadV3Tables(gpa, undefined, &bufs, &maps, &total, dir, true));
    try testing.expectEqual(counted, total); // refused files add no bytes
}

/// Main's stack rule's inputs: every name at `w`, except names ending in `odd`, at `odd_w`.
const FakeWidths = struct {
    w: quant.Width,
    odd: []const u8 = "-",
    odd_w: quant.Width = .{ .bits = 8, .group = 64 },

    pub fn of(f: FakeWidths, name: []const u8) !quant.Width {
        return if (std.mem.endsWith(u8, name, f.odd)) f.odd_w else f.w;
    }
};

test "tfq3: with no TFQ3 files every stack plans main's bytes (E326's per-stack witnesses)" {
    const plan = tf.glm.load_plan;
    const wts = tf.glm.weights;
    const L = struct { widths: FakeWidths, tfq3_ctx: tfq3.LoadCtx = .{} };
    const q4: L = .{ .widths = .{ .w = .{ .bits = 4, .group = 64 } } };
    try testing.expectEqual(@as(usize, 57_361_152), try plan.stackBytes(&q4, "a.", .{}, &wts.kda_in, 24896, &.{ 8192, 8192, 8192, 128, 128, 64 }, 4096, 64));
    try testing.expectEqual(@as(usize, 9_437_952), try plan.stackBytes(&q4, "s.", .{}, &wts.gate_up, 4096, &.{ 2048, 2048 }, 4096, 1));
    try testing.expectEqual(@as(usize, 56_623_872), try plan.stackBytes(&q4, "d.", .{}, &wts.gate_up, 24576, &.{ 12288, 12288 }, 4096, 1));
    try testing.expectEqual(@as(usize, 5_161_728), try plan.stackBytes(&q4, "a.", .{}, &wts.mla_x, 2240, &.{ 1536, 512, 128, 64 }, 4096, 64));
    try testing.expectEqual(@as(usize, 17_695_488), try plan.stackBytes(&q4, "a.", .{}, &wts.mla_qr, 20480, &.{ 16384, 4096 }, 1536, 1));
    // a mixed MLX stack (v_proj at 8/64): main's rule, a padded run a part at the costliest row
    const mixed: L = .{ .widths = .{ .w = .{ .bits = 4, .group = 64 }, .odd = "v_proj" } };
    try testing.expectEqual(@as(usize, (24896 + 5 * 64) * (4096 + 2 * 64 * 2) + 6 * 768), try plan.stackBytes(&mixed, "a.", .{}, &wts.kda_in, 24896, &.{ 8192, 8192, 8192, 128, 128, 64 }, 4096, 64));
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

const Shape = struct { n: u32, k: u32, bits: u8 };

/// Every table the kernels read: LUT halves at three widths and the c64 half4 rows, finite.
fn fillTables(t: []u8) void {
    @memset(t, 0);
    std.mem.writeInt(u32, t[0..4], tfq3.magic_v3_tables, .little);
    std.mem.writeInt(u32, t[4..8], tfq3.format_version, .little);
    for ([_][2]usize{ .{ 64, 512 }, .{ 1088, 2048 }, .{ 5184, 2048 } }) |lut| for (0..lut[1]) |i| {
        const v: f16 = @as(f16, @floatFromInt(@as(i32, @intCast(i % 15)) - 7)) * 0.125;
        std.mem.writeInt(u16, t[lut[0] + 2 * i ..][0..2], @bitCast(v), .little);
    };
    for ([_]usize{ 9280, 9792, 10304 }) |off| for (0..64) |i| {
        const q = [4]f16{ if (i & 1 != 0) -1.0 else 1.0, if (i & 2 != 0) -1.0 else 1.0, 0.01, -0.01 };
        for (q, 0..) |v, j| std.mem.writeInt(u16, t[off + 8 * i + 2 * j ..][0..2], @bitCast(v), .little);
    };
}

/// `rows` rows of x through tfq3.dispatch.dense `batch` rows a call (16: prefill, 4 and 1: decode) into y.
fn runDense(pipes: *const tfq3.Pipes, queue: anytype, b: [4]mtl.Buffer, y: mtl.Buffer, sh: Shape, rows: u32, batch: u32) !void {
    const cb = queue.commandBuffer();
    const enc = cb.compute(.serial);
    var r0: u32 = 0;
    while (r0 < rows) : (r0 += batch) {
        const in: affine.Ref = .{ .buf = b[3], .off = @as(usize, r0) * sh.k * 2 };
        const out: affine.Ref = .{ .buf = y, .off = @as(usize, r0) * sh.n * 2 };
        tfq3.dispatch.dense(pipes, enc, .{ .buf = b[0] }, .{ .buf = b[1] }, .{ .buf = b[2] }, sh.bits, in, out, sh.n, sh.k, batch, sh.k, sh.n);
    }
    enc.end();
    cb.commit();
    cb.wait();
    try testing.expect(cb.failure() == null);
}

/// Each run against the 16-row prefill, every differing pair named with its first differing output, then one failure.
fn sameRuns(units: bool, sh: Shape, ys: [4]mtl.Buffer, len: usize) !void {
    const names = [_][]const u8{ "16-row prefill", "repeated 16-row prefill", "4-row decode", "1-row decode" };
    const want = ys[0].slice(u16, len);
    var differ = false;
    for (ys[1..], names[1..]) |y, name| {
        const got = y.slice(u16, len);
        const at = std.mem.indexOfDiff(u16, want, got) orelse continue;
        differ = true;
        std.debug.print("TFQ3 {s} mode, {d}-bit {d}x{d}: {s} differs from the {s} at output {d} (0x{x:0>4} vs 0x{x:0>4})\n", .{ if (units) "tensor-unit" else "simdgroup", sh.bits, sh.n, sh.k, name, names[0], at, got[at], want[at] });
    }
    if (differ) return error.TestExpectedEqual;
}

test "tfq3: decode at 1 and 4 rows, prefill and a repeat give the same output bytes, both widths, split and not" {
    if (comptime !@hasDecl(mtl.Device, "init")) return error.SkipZigTest;
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    defer queue.deinit();
    const gpa = testing.allocator;
    const rows: u32 = 16;
    // the device's own mode, then on an M5 the simdgroup mode too; a mode count from a u1 (1 + 1) overflowed on M5
    const modes: []const bool = if (device.tensorUnits()) &.{ true, false } else &.{false};
    for (modes) |units| {
        var pipes = try tfq3.Pipes.initUnits(gpa, device, units, wide.glue_math);
        defer pipes.deinit();
        for ([_]Shape{ .{ .n = 128, .k = 128, .bits = 4 }, .{ .n = 128, .k = 128, .bits = 5 }, .{ .n = 128, .k = 1024, .bits = 4 }, .{ .n = 128, .k = 1024, .bits = 5 } }) |sh| {
            const words = sh.n * sh.k * sh.bits / 32;
            const halfs = (sh.n / 128) * (sh.k / 128);
            const b = [4]mtl.Buffer{ try device.buffer(words * 4, mtl.ResourceOptions.shared), try device.buffer(@max(16, halfs * 2), mtl.ResourceOptions.shared), try device.buffer(16384, mtl.ResourceOptions.shared), try device.buffer(rows * sh.k * 2, mtl.ResourceOptions.shared) };
            defer for (b) |x| x.deinit();
            for (b[0].slice(u32, words), 0..) |*w, i| w.* = @truncate((i *% 2654435761) ^ 0xA55A1234);
            for (b[1].slice(u16, halfs)) |*s| s.* = @bitCast(@as(f16, 0.25));
            fillTables(b[2].slice(u8, 16384));
            for (b[3].slice(u16, rows * sh.k), 0..) |*x, i| {
                const f: f32 = @as(f32, @floatFromInt(@as(i32, @intCast((i * 11) % 19)) - 9)) * 0.1;
                x.* = @intCast(@as(u32, @bitCast(f)) >> 16);
            }
            var ys: [4]mtl.Buffer = undefined;
            for (&ys) |*y| y.* = try device.buffer(rows * sh.n * 2, mtl.ResourceOptions.shared);
            defer for (ys) |y| y.deinit();
            for (ys, [_]u32{ rows, rows, 4, 1 }) |y, batch| try runDense(&pipes, queue, b, y, sh, rows, batch);
            try sameRuns(units, sh, ys, rows * sh.n);
        }
    }
}

const Stored = struct { name: []const u8, dtype: []const u8, shape: [2]usize, bytes: usize };

/// A safetensors shard of headers alone: each tensor's bytes are a hole, which a dry load never reads.
fn writeShard(a: std.mem.Allocator, dir: std.Io.Dir, name: []const u8, tensors: []const Stored) !void {
    var json: []const u8 = "{";
    var at: usize = 0;
    for (tensors, 0..) |s, i| {
        const dims = if (s.shape[1] == 0) try std.fmt.allocPrint(a, "{d}", .{s.shape[0]}) else try std.fmt.allocPrint(a, "{d},{d}", .{ s.shape[0], s.shape[1] });
        json = try std.fmt.allocPrint(a, "{s}{s}\"{s}\":{{\"dtype\":\"{s}\",\"shape\":[{s}],\"data_offsets\":[{d},{d}]}}", .{ json, if (i == 0) "" else ",", s.name, s.dtype, dims, at, at + s.bytes });
        at += s.bytes;
    }
    json = try std.fmt.allocPrint(a, "{s}}}", .{json});
    const f = try dir.createFile(testing.io, name, .{ .read = true });
    defer f.close(testing.io);
    var len: [8]u8 = undefined;
    std.mem.writeInt(u64, &len, json.len, .little);
    try f.writePositionalAll(testing.io, &len, 0);
    try f.writePositionalAll(testing.io, json, 8);
    try f.setLength(testing.io, 8 + json.len + at);
}

/// A one-Mac checkpoint: real-shaped 4/64 embedding and head, TFQ3 tables, one `record` of `rows` x `cols`.
fn writeCheckpoint(a: std.mem.Allocator, dir: std.Io.Dir, record: []const u8, rows: u32, cols: u32) !void {
    const io = testing.io;
    const V: usize = 154880;
    const D: usize = 4096;
    var tensors: [7]Stored = undefined;
    for ([_][]const u8{ "model.language_model.embed_tokens", "lm_head" }, 0..) |m, i| {
        tensors[3 * i] = .{ .name = try std.fmt.allocPrint(a, "{s}.weight", .{m}), .dtype = "U32", .shape = .{ V, D / 8 }, .bytes = V * D / 2 };
        tensors[3 * i + 1] = .{ .name = try std.fmt.allocPrint(a, "{s}.scales", .{m}), .dtype = "BF16", .shape = .{ V, D / 64 }, .bytes = V * D / 32 };
        tensors[3 * i + 2] = .{ .name = try std.fmt.allocPrint(a, "{s}.biases", .{m}), .dtype = "BF16", .shape = .{ V, D / 64 }, .bytes = V * D / 32 };
    }
    tensors[6] = .{ .name = "model.language_model.norm.weight", .dtype = "BF16", .shape = .{ D, 0 }, .bytes = D * 2 };
    try writeShard(a, dir, "model.safetensors", &tensors);
    var map: []const u8 = "";
    for (tensors, 0..) |s, i| map = try std.fmt.allocPrint(a, "{s}{s}\"{s}\":\"model.safetensors\"", .{ map, if (i == 0) "" else ",", s.name });
    try dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = try std.fmt.allocPrint(a, "{{\"weight_map\":{{{s}}}}}", .{map}) });
    try dir.writeFile(io, .{ .sub_path = "config.json", .data = "{\"quantization\": {\"group_size\": 64, \"bits\": 4, \"mode\": \"affine\"}}" });
    var tbl: [16384]u8 = @splat(0);
    std.mem.writeInt(u32, tbl[0..4], tfq3.magic_v3_tables, .little);
    std.mem.writeInt(u32, tbl[4..8], tfq3.format_version, .little);
    try dir.writeFile(io, .{ .sub_path = "pw_v3_tables.bin", .data = &tbl });
    // nonmoe.tffpw: records at 128, scales at 16384, weights on the next 16 KiB boundary (a hole past the header)
    const words: u64 = @as(u64, rows) * cols * 4 / 32;
    const halfs: u64 = (rows / 128) * (cols / 128);
    const at_weights: u64 = 16384 + std.mem.alignForward(u64, halfs * 2, 16384);
    var head: [256]u8 = @splat(0);
    const h: tfq3.NonMoeHeader = .{ .magic = tfq3.magic_nonmoe, .version = tfq3.format_version, .num_tensors = 1, .pad0 = 0, .off_meta = 128, .off_scales = 16384, .off_weights = at_weights, .total_scale_halfs = halfs, .total_weight_words = words, .pad1 = 0 };
    @memcpy(head[0..@sizeOf(tfq3.NonMoeHeader)], std.mem.asBytes(&h));
    const m = makeMeta(record, rows, cols, 4, 0, 0, words);
    @memcpy(head[128..][0..@sizeOf(tfq3.NonMoeTensorMeta)], std.mem.asBytes(&m));
    const f = try dir.createFile(io, "nonmoe.tffpw", .{ .read = true });
    defer f.close(io);
    try f.writePositionalAll(io, &head, 0);
    try f.setLength(io, at_weights + words * 4);
}

/// weights.load's dry plan and construction of `c`'s embedding and head over that checkpoint.
fn dryLoad(dir: []const u8, c: tf.glm.config.Config) !affine.Affine {
    const w = try tf.glm.weights.load(testing.allocator, undefined, dir, &c, 1, true);
    defer {
        w.deinit();
        testing.allocator.destroy(w);
    }
    return w.head;
}

test "tfq3: a dry load refuses a head record the model cannot read whole, before planning, and maps one it can" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const dir = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    const one: tf.glm.config.Config = .{ .run = 0, .mtp = 0 };
    // E329b: a valid 128x128 lm_head record beside real-shaped 4/64 fallback tensors: refused by name, not copied
    try writeCheckpoint(a, tmp.dir, "lm_head", 128, 128);
    try testing.expectError(error.Tfq3RecordShape, dryLoad(dir, one));
    // control: no head record, the ordinary 4/64 head
    try writeCheckpoint(a, tmp.dir, "model.language_model.layers.0.mlp.down_proj", 4096, 12288);
    try testing.expectEqual(@as(u16, 64), (try dryLoad(dir, one)).group);
    // the head's own shape: mapped from nonmoe.tffpw, as planned
    try writeCheckpoint(a, tmp.dir, "lm_head", 154880, 4096);
    try testing.expectEqual(@as(u16, 512), (try dryLoad(dir, one)).group);
    // TP2 would read rank 1's half of the head: refused before any plan
    try testing.expectError(error.Tfq3TensorParallel, dryLoad(dir, .{ .run = 0, .mtp = 0, .tp = 2, .tp_rank = 1 }));
}
