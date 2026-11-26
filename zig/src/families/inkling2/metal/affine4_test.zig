//! Host-only tests for Inkling 2 Mini MLX 4-bit affine g64 tensor validation, conversions, and dispatch geometry.
const std = @import("std");
const mtl = @import("metal");
const st = @import("core").safetensors;
const cfg = @import("../config.zig");
const model = @import("model.zig");
const affine4 = @import("affine4.zig");
const model_affine4 = @import("model_affine4.zig");
const t = std.testing;

const FakeTensor = struct {
    dtype: st.DType,
    shape: [st.max_rank]usize = @splat(1),
    rank: usize = 0,
    bytes: usize,
};

const FakeCheckpoint = struct {
    map: std.StringHashMapUnmanaged(FakeTensor) = .empty,

    fn deinit(self: *FakeCheckpoint, a: std.mem.Allocator) void {
        self.map.deinit(a);
    }

    fn put(self: *FakeCheckpoint, a: std.mem.Allocator, name: []const u8, dtype: st.DType, dims: []const usize) !void {
        var count: usize = 1;
        for (dims) |d| count *= d;
        var ft = FakeTensor{ .dtype = dtype, .rank = dims.len, .bytes = count * dtype.size() };
        for (dims, 0..) |d, i| ft.shape[i] = d;
        try self.map.put(a, name, ft);
    }

    pub fn get(self: *const FakeCheckpoint, name: []const u8) !FakeTensor {
        return self.map.get(name) orelse error.MissingTensor;
    }

    pub fn has(self: *const FakeCheckpoint, name: []const u8) bool {
        return self.map.contains(name);
    }
};

test "affine4 tensor validation accepts all Inkling 2 Mini shapes and refuses bad companions, bits, groups, and dtypes" {
    const a = t.allocator;
    const shapes = [_][2]u32{
        .{ 201024, 2304 }, // embed_tokens, lm_head
        .{ 3072, 2304 }, // q_proj, gate_proj
        .{ 1024, 2304 }, // k_proj, v_proj
        .{ 384, 2304 }, // r_proj
        .{ 24, 2304 }, // sink_proj
        .{ 2304, 3072 }, // o_proj
        .{ 32, 2304 }, // k_sconv/v_sconv weight_proj
    };
    for (shapes) |sh| {
        var ck: FakeCheckpoint = .{};
        defer ck.deinit(a);
        try ck.put(a, "proj.weight", .u32, &.{ sh[0], sh[1] / 8 });
        try ck.put(a, "proj.scales", .bf16, &.{ sh[0], sh[1] / 64 });
        try ck.put(a, "proj.biases", .bf16, &.{ sh[0], sh[1] / 64 });
        const triple = try affine4.readTriple(&ck, "proj", sh[0], sh[1]);
        try t.expectEqual(@as(u8, 4), triple.bits);
        try t.expectEqual(@as(usize, 64), triple.group);
    }

    // Missing .biases or .scales
    {
        var ck: FakeCheckpoint = .{};
        defer ck.deinit(a);
        try ck.put(a, "p.weight", .u32, &.{ 24, 288 });
        try ck.put(a, "p.scales", .bf16, &.{ 24, 36 });
        try t.expectError(error.MissingTensor, affine4.readTriple(&ck, "p", 24, 2304));
    }
    {
        var ck: FakeCheckpoint = .{};
        defer ck.deinit(a);
        try ck.put(a, "p.weight", .u32, &.{ 24, 288 });
        try ck.put(a, "p.biases", .bf16, &.{ 24, 36 });
        try t.expectError(error.MissingTensor, affine4.readTriple(&ck, "p", 24, 2304));
    }

    // Wrong bits (8-bit and 3-bit packed words)
    for ([_]usize{ 2304 / 4, 2304 * 3 / 32 }) |bad_words| {
        var ck: FakeCheckpoint = .{};
        defer ck.deinit(a);
        try ck.put(a, "p.weight", .u32, &.{ 24, bad_words });
        try ck.put(a, "p.scales", .bf16, &.{ 24, 36 });
        try ck.put(a, "p.biases", .bf16, &.{ 24, 36 });
        try t.expectError(error.UnexpectedAffineTensor, affine4.readTriple(&ck, "p", 24, 2304));
    }

    // Wrong group_size (32 and 128)
    for ([_]usize{ 2304 / 32, 2304 / 128 }) |bad_groups| {
        var ck: FakeCheckpoint = .{};
        defer ck.deinit(a);
        try ck.put(a, "p.weight", .u32, &.{ 24, 288 });
        try ck.put(a, "p.scales", .bf16, &.{ 24, bad_groups });
        try ck.put(a, "p.biases", .bf16, &.{ 24, bad_groups });
        try t.expectError(error.UnexpectedAffineTensor, affine4.readTriple(&ck, "p", 24, 2304));
    }

    // Wrong dtypes & truncated bytes
    {
        var ck: FakeCheckpoint = .{};
        defer ck.deinit(a);
        try ck.put(a, "p.weight", .bf16, &.{ 24, 288 });
        try ck.put(a, "p.scales", .bf16, &.{ 24, 36 });
        try ck.put(a, "p.biases", .bf16, &.{ 24, 36 });
        try t.expectError(error.UnexpectedAffineTensor, affine4.readTriple(&ck, "p", 24, 2304));
    }
    {
        var ck: FakeCheckpoint = .{};
        defer ck.deinit(a);
        try ck.put(a, "p.weight", .u32, &.{ 24, 288 });
        try ck.put(a, "p.scales", .f32, &.{ 24, 36 });
        try ck.put(a, "p.biases", .bf16, &.{ 24, 36 });
        try t.expectError(error.UnexpectedAffineTensor, affine4.readTriple(&ck, "p", 24, 2304));
    }
    {
        var ck: FakeCheckpoint = .{};
        defer ck.deinit(a);
        try ck.put(a, "p.weight", .u32, &.{ 24, 288 });
        try ck.put(a, "p.scales", .bf16, &.{ 24, 36 });
        try ck.put(a, "p.biases", .bf16, &.{ 24, 36 });
        ck.map.getPtr("p.weight").?.bytes -= 4;
        try t.expectError(error.UnexpectedAffineTensor, affine4.readTriple(&ck, "p", 24, 2304));
    }
}

test "official BF16 dense validation refuses stray .scales or .biases affine companions" {
    const a = t.allocator;
    var ck: FakeCheckpoint = .{};
    defer ck.deinit(a);
    try ck.put(a, "m.weight", .bf16, &.{ 24, 2304 });
    const ok = try affine4.readBf16NoCompanions(&ck, "m", 24, 2304);
    try t.expectEqual(st.DType.bf16, ok.dtype);

    try ck.put(a, "m.scales", .bf16, &.{ 24, 36 });
    try t.expectError(error.UnexpectedAffineCompanion, affine4.readBf16NoCompanions(&ck, "m", 24, 2304));

    _ = ck.map.remove("m.scales");
    try ck.put(a, "m.biases", .bf16, &.{ 24, 36 });
    try t.expectError(error.UnexpectedAffineCompanion, affine4.readBf16NoCompanions(&ck, "m", 24, 2304));

    _ = ck.map.remove("m.biases");
    try ck.put(a, "m.weight", .u32, &.{ 24, 288 });
    try t.expectError(error.UnexpectedInkling2Tensor, affine4.readBf16NoCompanions(&ck, "m", 24, 2304));
}

test "admitConfig admits official NVFP4 and MLX affine4 Mini configs and refuses EXL3" {
    const mini_json = @embedFile("../fixtures/mini_config.json");
    var why: cfg.Why = .{};
    var c = try cfg.parse(t.allocator, mini_json, &why);
    try model.admitConfig(&c);
    c.quant = .mlx_affine4;
    try model.admitConfig(&c);
    c.quant = .exl3_kortexa_v1;
    try t.expectError(error.UnsupportedInkling2Quant, model.admitConfig(&c));
}

test "host dequantAffine4ToBf16, convertF32BytesToBf16, and splitMessageQueries match exact layouts" {
    const a = t.allocator;
    // 2 rows x 128 cols (2 groups of 64, 16 u32 words per row), starting at an unaligned byte offset (+1)
    const rows = 2;
    const cols = 128;
    var w_buf: [1 + rows * (cols / 8) * 4]u8 = undefined;
    var s_buf: [1 + rows * (cols / 64) * 2]u8 = undefined;
    var b_buf: [1 + rows * (cols / 64) * 2]u8 = undefined;
    const w_bytes = w_buf[1..];
    const s_bytes = s_buf[1..];
    const b_bytes = b_buf[1..];
    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();
    for (0..rows * (cols / 8)) |wi| std.mem.writeInt(u32, w_bytes[wi * 4 ..][0..4], rng.int(u32), .little);
    for (0..rows * (cols / 64)) |gi| {
        std.mem.writeInt(u16, s_bytes[gi * 2 ..][0..2], affine4.toBf16(rng.float(f32) * 0.25 + 0.01), .little);
        std.mem.writeInt(u16, b_bytes[gi * 2 ..][0..2], affine4.toBf16((rng.float(f32) - 0.5) * 0.5), .little);
    }
    var dst: [rows * cols]u16 = undefined;
    affine4.dequantAffine4ToBf16(w_bytes, s_bytes, b_bytes, rows, cols, &dst);
    for (0..rows) |r| {
        for (0..cols) |c| {
            const wi = r * (cols / 8) + c / 8;
            const word = std.mem.readInt(u32, w_bytes[wi * 4 ..][0..4], .little);
            const q: f32 = @floatFromInt((word >> @intCast((c % 8) * 4)) & 0xf);
            const gi = r * (cols / 64) + c / 64;
            const sc = affine4.fromBf16(std.mem.readInt(u16, s_bytes[gi * 2 ..][0..2], .little));
            const bi = affine4.fromBf16(std.mem.readInt(u16, b_bytes[gi * 2 ..][0..2], .little));
            try t.expectEqual(affine4.toBf16(q * sc + bi), dst[r * cols + c]);
        }
    }

    // F32->BF16 lossless roundtrip from unaligned bytes
    var f32_buf: [1 + 6 * 4]u8 = undefined;
    const sample_bf = [_]u16{ 0x0000, 0x8000, 0x3f80, 0xbf80, 0x4528, 0x7f80 };
    for (sample_bf, 0..) |h, i| {
        std.mem.writeInt(u32, f32_buf[1 + i * 4 ..][0..4], @bitCast(affine4.fromBf16(h)), .little);
    }
    var bf_out: [6]u16 = undefined;
    affine4.convertF32BytesToBf16(f32_buf[1..], &bf_out);
    try t.expectEqualSlices(u16, &sample_bf, &bf_out);

    // splitMessageQueries inverse of checkpoint.py ([12, 4, 2304] -> q/mlp & k/v)
    const msg_bytes = try a.alloc(u8, 12 * 4 * 2304 * 4);
    defer a.free(msg_bytes);
    for (0..12) |core| for (0..4) |role| for (0..2304) |d| {
        const val: f32 = @floatFromInt(core * 10000 + role * 3000 + d);
        const off = ((core * 4 + role) * 2304 + d) * 4;
        std.mem.writeInt(u32, msg_bytes[off..][0..4], @bitCast(val), .little);
    };
    const cross_q = try a.alloc(f32, 24 * 2304);
    defer a.free(cross_q);
    const cross_kv = try a.alloc(f32, 24 * 2304);
    defer a.free(cross_kv);
    affine4.splitMessageQueries(msg_bytes, cross_q, cross_kv);
    for (0..12) |core| {
        try t.expectEqual(@as(f32, @floatFromInt(core * 10000 + 0 * 3000 + 17)), cross_q[(2 * core) * 2304 + 17]);
        try t.expectEqual(@as(f32, @floatFromInt(core * 10000 + 3 * 3000 + 17)), cross_q[(2 * core + 1) * 2304 + 17]);
        try t.expectEqual(@as(f32, @floatFromInt(core * 10000 + 1 * 3000 + 17)), cross_kv[(2 * core) * 2304 + 17]);
        try t.expectEqual(@as(f32, @floatFromInt(core * 10000 + 2 * 3000 + 17)), cross_kv[(2 * core + 1) * 2304 + 17]);
    }
    try t.expect(model_affine4.stageCapacity() > 40 * 1024 * 1024);
    try t.expectEqual(@as(usize, 0), model_affine4.stageCapacity() % 256);
    try t.expectEqual(@as(f32, 2688.0), affine4.fromBf16(affine4.toBf16(affine4.nvfp4_global_denom)));
}

test "affine4 encoders guard empty launches and compute exact 8-col x 4-row grid geometry" {
    const FakePipes = struct { embed_norm: u32 = 1, dense: u32 = 2, logits: u32 = 3 };
    const Rec = struct {
        pipe: u32 = 0,
        groups: mtl.Size = mtl.Size.of(0, 0, 0),
        threads: mtl.Size = mtl.Size.of(0, 0, 0),
        count: usize = 0,
        pub fn setPipeline(r: *@This(), p: u32) void {
            r.pipe = p;
        }
        pub fn setBuffer(_: *@This(), _: mtl.Buffer, _: usize, _: usize) void {}
        pub fn setValue(_: *@This(), _: anytype, _: usize) void {}
        pub fn dispatchGroups(r: *@This(), g: mtl.Size, th: mtl.Size) void {
            r.groups = g;
            r.threads = th;
            r.count += 1;
        }
    };
    const dummy = affine4.Ref{ .buf = .{ .id = @ptrFromInt(0x1000) } };
    const pipes = FakePipes{};
    var rec = Rec{};
    affine4.encodeEmbedNorm(*Rec, &pipes, &rec, dummy, dummy, dummy, dummy, dummy, dummy, 0);
    affine4.encodeDense(*Rec, &pipes, &rec, dummy, dummy, dummy, dummy, dummy, .{ .rows = 0, .n = 24, .k = 2304, .ldy = 24 });
    affine4.encodeLogits(*Rec, &pipes, &rec, dummy, dummy, dummy, dummy, dummy, 1, 0);
    try t.expectEqual(@as(usize, 0), rec.count);

    affine4.encodeEmbedNorm(*Rec, &pipes, &rec, dummy, dummy, dummy, dummy, dummy, dummy, 5);
    try t.expectEqual(@as(u32, 1), rec.pipe);
    try t.expectEqual(mtl.Size.of(5, 1, 1), rec.groups);
    try t.expectEqual(mtl.Size.of(256, 1, 1), rec.threads);

    affine4.encodeDense(*Rec, &pipes, &rec, dummy, dummy, dummy, dummy, dummy, .{ .rows = 5, .n = 25, .k = 2304, .ldy = 25 });
    try t.expectEqual(@as(u32, 2), rec.pipe);
    try t.expectEqual(mtl.Size.of(4, 2, 1), rec.groups);
    try t.expectEqual(mtl.Size.of(256, 1, 1), rec.threads);

    affine4.encodeLogits(*Rec, &pipes, &rec, dummy, dummy, dummy, dummy, dummy, 9, 200058);
    try t.expectEqual(@as(u32, 3), rec.pipe);
    try t.expectEqual(mtl.Size.of(25008, 3, 1), rec.groups);
    try t.expectEqual(mtl.Size.of(256, 1, 1), rec.threads);
}
