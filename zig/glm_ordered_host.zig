//! Execute actual pre-M5 dispatch selection against a recording boundary, without a GPU.
const std = @import("std");
const mtl = @import("metal");
const simd = @import("src/core/simd_qmm_bits.zig");
const projection = @import("src/families/glm/ordered_projection.zig");
const wide = @import("src/families/glm/wide.zig");
const affine = @import("src/families/glm/affine.zig");
const st = @import("src/families/glm/state.zig");
const cfg = @import("src/families/glm/config.zig");
const Ref = affine.Ref;
const t = std.testing;
fn ref(id: usize, off: usize) Ref {
    return .{ .buf = .{ .id = id }, .off = off };
}
const Kernels = struct { ordered: ?simd.Pipelines, wide: ?wide.Pipes = null };
fn args(p: mtl.Pass) simd.Args {
    return std.mem.bytesToValue(simd.Args, p.values[4][0..@sizeOf(simd.Args)]);
}

test "GLM ordered: all bits and window/chunk heights use warmed shared trees with explicit strides" {
    var p = try simd.Pipelines.load(.{});
    defer p.deinit();
    try p.warm();
    const built = mtl.builds;
    var k: Kernels = .{ .ordered = p };
    for (simd.bit_widths) |bits| for ([_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 2048, 8192, 16384 }) |rows| {
        var rec: mtl.Record = .{};
        const q: affine.Affine = .{ .w = ref(1, 16), .s = ref(2, 32), .b = ref(3, 64), .n = 24896, .k = 4096, .bits = bits, .group = 64 };
        try t.expect(projection.encode(&k, .{ .r = &rec }, q, .{ .in = ref(4, 256), .out = ref(5, 512), .ldx = 24896, .ldo = 24960, .rows = rows }));
        try t.expectEqual(@as(usize, 1), rec.n);
        const a = args(rec.passes[0]);
        try t.expectEqual(@as(i32, 24896), a.ldx);
        try t.expectEqual(@as(i32, 24960), a.ld);
        try t.expectEqual(@as(i32, @intCast(rows)), a.r);
        try t.expectEqual(@as(usize, 256), rec.passes[0].buffers[0].off);
        try t.expectEqual(@as(usize, 16), rec.passes[0].buffers[1].off);
        try t.expectEqual(@as(usize, 512), rec.passes[0].buffers[5].off);
        try t.expect(rec.passes[0].threads.width <= 512);
        try t.expectEqual(built, mtl.builds);
    };
}

test "GLM ordered: disabled policy does not record a replacement and g32 tails retain the row operator" {
    var k: Kernels = .{ .ordered = null };
    var rec: mtl.Record = .{};
    var q: affine.Affine = .{ .w = ref(1, 0), .s = ref(2, 0), .b = ref(3, 0), .n = 8, .k = 128, .bits = 4, .group = 32 };
    const io: wide.Io = .{ .in = ref(4, 0), .out = ref(5, 0), .ldx = 128, .ldo = 32, .rows = 3 };
    try t.expect(!projection.encode(&k, .{ .r = &rec }, q, io));
    try t.expectEqual(@as(usize, 0), rec.n);
    k.ordered = try simd.Pipelines.load(.{});
    defer k.ordered.?.deinit();
    try k.ordered.?.warm();
    var w: wide.Pipes = undefined;
    w.rows[2] = .{ .id = 17 };
    w.rows_wide[2] = .{ .id = 17 };
    k.wide = w;
    q.more = 1;
    q.n = 24;
    q.runs[0] = .{ .w = ref(6, 128), .s = ref(7, 256), .b = ref(8, 512), .row = 8, .n = 16, .bits = 8, .group = 64 };
    try t.expect(projection.encode(&k, .{ .r = &rec }, q, io));
    try t.expectEqual(@as(usize, 2), rec.n);
    try t.expectEqual(@as(u64, 17), rec.passes[0].pipe.id);
    try t.expectEqual(@as(usize, 16), rec.passes[1].buffers[5].off);
    try t.expectEqual(@as(usize, 128), rec.passes[1].buffers[1].off);
    try t.expectEqual(@as(i32, 32), args(rec.passes[1]).ld);
}

test "GLM ordered: contiguous ABI stays zero and invalid input pitches are refused" {
    var p = try simd.Pipelines.load(.{});
    defer p.deinit();
    try t.expectEqual(@as(i32, 0), (try simd.call(&p, 8, 8, 128, 3, 8, false, 0)).args.ldx);
    try t.expectError(error.UnsupportedSimdStride, simd.callAt(&p, 8, 8, 128, 3, 120, 8, false, 0));
    try t.expectError(error.UnsupportedSimdStride, simd.callAt(&p, 8, 8, 128, 3, 132, 8, false, 0));
}

test "GLM ordered: dry admission counts recurrence planes only on the selected path" {
    const c: cfg.Config = .{};
    var old: st.Arena = .{ .device = .{}, .gpa = t.allocator, .dry = true };
    var new: st.Arena = .{ .device = .{}, .gpa = t.allocator, .dry = true };
    _ = try st.init(&old, &c, 4096);
    _ = try st.initOrdered(&new, &c, 4096, true);
    try t.expectEqual(@as(usize, st.max_rows) * (14 * c.kdaWidth() + 4 * c.kda_heads), new.bytes - old.bytes);
}

test "GLM ordered: M5 and paired ranks keep their arithmetic regardless of prompt mode" {
    const policy = @import("src/families/glm/ordered_policy.zig");
    try t.expect(policy.enabled(false, false));
    try t.expect(!policy.enabled(true, false));
    try t.expect(!policy.enabled(false, true));
    try t.expect(!policy.enabled(true, true));
}

test "GLM ordered: KDA blocks carry FP32 state and keep the original convolution tail and projection pitch" {
    const wts = @import("src/families/glm/weights.zig");
    const K = struct {
        ordered: ?simd.Pipelines,
        wide: ?wide.Pipes = null,
        kda_prep: mtl.Pipeline = .{ .id = 101 },
        kda_scan: mtl.Pipeline = .{ .id = 102 },
        kda_post: mtl.Pipeline = .{ .id = 103 },
        kda_prep_tp: mtl.Pipeline = .{},
        kda_scan_tp: mtl.Pipeline = .{},
        kda_post_tp: mtl.Pipeline = .{},
    };
    var k: K = .{ .ordered = try simd.Pipelines.load(.{}) };
    defer k.ordered.?.deinit();
    try k.ordered.?.warm();
    const c: cfg.Config = .{};
    var state: st.State = .{ .cap = 4096 };
    state.kda[0] = .{ .st = .{ ref(10, 256), ref(11, 512) }, .cs = .{ ref(12, 768), ref(13, 1024) }, .cur = 1, .proj = ref(14, 1280) };
    const X = struct { c: *const cfg.Config, k: *const K, s: *st.State, pskip: u32 = 0 };
    const x: X = .{ .c = &c, .k = &k, .s = &state };
    const q: affine.Affine = .{ .w = ref(21, 0), .s = ref(22, 0), .b = ref(23, 0), .n = c.kdaWidth(), .k = c.kda_dim, .bits = 8, .group = 64 };
    const w: wts.Kda = .{ .in_proj = q, .f_b = q, .g_b = q, .o_proj = q, .conv_w = ref(24, 0), .a_log = ref(25, 0), .a = ref(26, 0), .dt_bias = ref(27, 0), .o_norm = ref(28, 0) };
    var rec: mtl.Record = .{};
    @import("src/families/glm/ordered_kda.zig").encode(&x, .{ .r = &rec }, 0, &w, state.kda[0].proj, c.kdaProj(), ref(29, 64), ref(30, 256), 17, 16);
    try t.expectEqual(@as(usize, 10), rec.n);
    for ([_]usize{ 0, 5 }) |at| {
        try t.expectEqual(@as(i32, @intCast(c.kdaProj())), args(rec.passes[at]).ldx);
        try t.expectEqual(@as(u64, 101), rec.passes[at + 2].pipe.id);
        try t.expectEqual(@as(usize, 13), rec.passes[at + 2].buffers[2].id);
        try t.expectEqual(@as(u64, 102), rec.passes[at + 3].pipe.id);
        try t.expectEqual(@as(usize, 10), rec.passes[at + 3].buffers[6].id);
        try t.expectEqual(@as(u64, 103), rec.passes[at + 4].pipe.id);
        try t.expectEqual(@as(usize, 13), rec.passes[at + 4].buffers[7].id);
        try t.expectEqual(@as(usize, 12), rec.passes[at + 4].buffers[8].id);
    }
    try t.expectEqual(@as(usize, 11), rec.passes[3].buffers[5].id);
    try t.expectEqual(@as(usize, 10), rec.passes[8].buffers[5].id);
    try t.expectEqual(@as(i32, 16), std.mem.bytesToValue(i32, rec.passes[3].values[8][0..4]));
    try t.expectEqual(@as(i32, 1), std.mem.bytesToValue(i32, rec.passes[8].values[8][0..4]));
    try t.expectEqual(@as(usize, 1280 + 16 * c.kdaProj() * 2 + 3 * c.kdaWidth() * 2), rec.passes[5].buffers[0].off);
    try t.expectEqual(@as(u1, 1), state.kda[0].cur);
}

test {
    _ = @import("tests/glm_ordered_kda_ref.zig");
}

test "GLM ordered: every scalar resource record matches the specialization's actual staging source" {
    const source = @import("kernel_sources").core_simd_qmm_bits;
    const prefix = "tf_sqw_scalar_q";
    var seen: usize = 0;
    for (@import("src/core/contract/glm_ordered_records.zig").records) |record| {
        const at = std.mem.indexOf(u8, record.name, prefix) orelse continue;
        var words = std.mem.tokenizeScalar(u8, record.name[at + prefix.len ..], '_');
        _ = try std.fmt.parseInt(usize, words.next().?, 10);
        const s = try std.fmt.parseInt(usize, words.next().?, 10);
        const nr = try std.fmt.parseInt(usize, words.next().?, 10);
        const rows = try std.fmt.parseInt(usize, words.next().?, 10);
        const xb = if (rows == 1) 32 else @max(s, 16);
        try t.expectEqual(rows * xb * 76 * 4, record.tg_memory);
        var text: [80]u8 = undefined;
        const macro = try std.fmt.bufPrint(&text, "TF_SQWS(B, {d}, {d}, {d}, {d})", .{ s, nr, xb, rows });
        try t.expect(std.mem.indexOf(u8, source, macro) != null);
        seen += 1;
    }
    try t.expectEqual(@as(usize, 72), seen);
}

test "GLM ordered: reflected scalar refusal falls back to tiles, zero tile capacity is refused" {
    defer mtl.scalar_limit = 512;
    defer mtl.tile_limit = 512;
    mtl.scalar_limit = 0;
    mtl.tile_limit = 256;
    var p = try simd.Pipelines.load(.{});
    defer p.deinit();
    try p.warm();
    var k: Kernels = .{ .ordered = p };
    var rec: mtl.Record = .{};
    const q: affine.Affine = .{ .w = ref(1, 0), .s = ref(2, 0), .b = ref(3, 0), .n = 8192, .k = 4096, .bits = 3, .group = 64 };
    try t.expectError(error.SimdThreads, simd.call(&p, 3, 8192, 4096, 2, 8192, true, 0));
    try t.expect(projection.encode(&k, .{ .r = &rec }, q, .{ .in = ref(4, 0), .out = ref(5, 0), .ldx = 4096, .ldo = 8192, .rows = 2 }));
    try t.expectEqual(@as(usize, 1), rec.n);
    try t.expect(!rec.passes[0].pipe.scalar);
    try t.expect(rec.passes[0].threads.width > 0 and rec.passes[0].threads.width <= 256);
    mtl.tile_limit = 0;
    try t.expectError(error.SimdThreads, simd.call(&p, 2, 8192, 4096, 16, 8192, false, 0));
    try t.expectError(error.SimdThreads, simd.call(&p, 4, 8192, 4096, 16, 8192, false, 0));
}
