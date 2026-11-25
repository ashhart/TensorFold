//! The one-row EXL3 kernel on a Metal GPU: its decoded W_q against the oracle, its dots against the serial reference.
const std = @import("std");
const mtl = @import("metal");
const ex = @import("exl3_projection.zig");
const ref = @import("exl3_projection_row_ref.zig");
const fmt = ex.format;
const shared = mtl.ResourceOptions.shared;

pub const Gpu = struct {
    device: mtl.Device,
    queue: mtl.Queue,
    p: ex.Pipelines,

    pub fn init() !Gpu {
        const device = try mtl.Device.init();
        errdefer device.deinit();
        const queue = try device.queue();
        errdefer queue.deinit();
        return .{ .device = device, .queue = queue, .p = try ex.Pipelines.load(device) };
    }
    pub fn deinit(g: *Gpu) void {
        g.p.deinit();
        g.queue.deinit();
        g.device.deinit();
    }
    /// The same pipelines with 1 to 16 rows sent through tf_exl3_matmul_hoisted, as before the row kernels.
    pub fn main(g: *const Gpu) ex.Pipelines {
        var p = g.p;
        p.row = null;
        p.rows = null;
        return p;
    }
};

pub fn complete(cb: mtl.CommandBuffer, e: mtl.ComputeEncoder) !void {
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure() != null) return error.Exl3GpuFailure;
}

pub fn bf16(x: f32) u16 {
    const bits: u32 = @bitCast(x);
    return @truncate((bits +% (0x7fff + ((bits >> 16) & 1))) >> 16);
}

/// A stored-order matrix: trellis words, suh and svh in their own shared buffers.
pub const Weight = struct {
    matrix: ex.Matrix,
    bufs: [3]mtl.Buffer,

    pub fn init(device: mtl.Device, k: usize, n: usize, halves: u16, book: fmt.Codebook, rng: std.Random, ones: bool) !Weight {
        const width = fmt.Width{ .halves = halves };
        const count = (k / 16) * (n / 16) * width.tileWords();
        const t = try device.buffer(count * 2, shared);
        errdefer t.deinit();
        const su = try device.buffer(k * 2, shared);
        errdefer su.deinit();
        const sv = try device.buffer(n * 2, shared);
        errdefer sv.deinit();
        for (t.slice(i16, count)) |*v| v.* = rng.int(i16);
        for (su.slice(f16, k)) |*v| v.* = if (ones) 1 else @floatCast(rng.float(f32) * 2 - 1);
        for (sv.slice(f16, n)) |*v| v.* = if (ones) 1 else @floatCast(rng.float(f32) * 2 - 1);
        const m = ex.Matrix{ .trellis = .{ .buffer = t }, .suh = .{ .buffer = su }, .svh = .{ .buffer = sv }, .k = k, .n = n, .width = width, .codebook = book };
        return .{ .matrix = m, .bufs = .{ t, su, sv } };
    }
    pub fn deinit(w: Weight) void {
        for (w.bufs) |b| b.deinit();
    }
    pub fn words(w: Weight) []i16 {
        return w.bufs[0].slice(i16, w.bufs[0].length() / 2);
    }
};

/// x is block row j of the Hadamard matrix (suh 1), so the rotated row is c * e_j and sums[col] = c * W_q[j][col].
fn probeWq(g: *Gpu, m: ex.Matrix, want: []align(1) const u16) !void {
    const n = m.n;
    const slot = try ex.scratchBytes(128, n, 1);
    const scratch = try g.device.buffer(128 * slot, shared);
    defer scratch.deinit();
    const x = try g.device.buffer(128 * 128 * 2, shared);
    defer x.deinit();
    const y = try g.device.buffer(128 * n * 4, shared);
    defer y.deinit();
    for (0..128) |j| for (0..128) |i| {
        x.slice(u16, 128 * 128)[j * 128 + i] = if (@popCount(i & j) & 1 == 1) bf16(-1) else bf16(1);
    };
    const cb = g.queue.commandBuffer();
    const e = cb.compute(.concurrent);
    for (0..128) |j| try ex.encode(&g.p, e, m, .{ .buffer = x, .offset = j * 256 }, .{ .buffer = y, .offset = j * n * 4 }, .{ .buffer = scratch, .offset = j * slot }, .{ .count = 1, .x_stride = 128, .y_stride = n, .output = .f32 });
    try complete(cb, e);
    for (0..128) |j| {
        const bytes = scratch.contents()[j * slot ..][0..slot];
        const rotated = std.mem.bytesAsSlice(f16, bytes[0..256]);
        for (rotated, 0..) |v, i| if (i != j) try std.testing.expectEqual(@as(f16, 0), v);
        try std.testing.expect(rotated[j] != 0);
        const sums = std.mem.bytesAsSlice(f32, bytes[256..][0 .. n * 4]);
        for (sums, 0..) |got, col| {
            const w: f16 = @bitCast(want[j * n + col]);
            const expect: f32 = @as(f32, rotated[j]) * @as(f32, w);
            try std.testing.expectEqual(@as(u32, @bitCast(expect)), @as(u32, @bitCast(got)));
        }
    }
}

test "the one-row kernel decodes every codebook and width's W_q bit for bit against the scalar decoder" {
    var g = try Gpu.init();
    defer g.deinit();
    var prng = std.Random.DefaultPrng.init(0x7771);
    for ([_]fmt.Codebook{ .inst3, .mcg, .mul1 }) |book| {
        for ([_]u16{ 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16 }) |halves| {
            if (halves % 2 == 1 and book != .mul1) continue;
            const w = try Weight.init(g.device, 128, 128, halves, book, prng.random(), true);
            defer w.deinit();
            var want: [128 * 128]f16 = undefined;
            fmt.unpack(w.matrix.width, book, w.words(), 8, 8, &want);
            try probeWq(&g, w.matrix, std.mem.bytesAsSlice(u16, std.mem.sliceAsBytes(&want)));
        }
    }
}

/// The checkpoint block that closes the oracle file: halves, its trellis, suh, svh and W_q fp16 bits.
const Real = struct { halves: u16, trellis: []const u8, suh: []const u8, svh: []const u8, wq: []const u8 };

fn realBlock(oracle: []const u8) Real {
    const r = struct {
        fn u32At(b: []const u8, at: usize) usize {
            return std.mem.readInt(u32, b[at..][0..4], .little);
        }
    };
    var at: usize = 16;
    for (0..std.mem.readInt(u64, oracle[8..16], .little)) |_| {
        const w = fmt.Width{ .halves = std.mem.readInt(u16, oracle[at..][0..2], .little) };
        at += 12 + r.u32At(oracle, at + 4) * r.u32At(oracle, at + 8) * (w.tileWords() * 2 + 256 * 4);
    }
    at += 4 + r.u32At(oracle, at) * 6;
    const halves = std.mem.readInt(u16, oracle[at..][0..2], .little);
    at += 10;
    var parts: [4][]const u8 = undefined;
    for (&parts) |*p| {
        const len = r.u32At(oracle, at);
        p.* = oracle[at + 4 ..][0..len];
        at += 4 + len;
    }
    return .{ .halves = halves, .trellis = parts[0], .suh = parts[1], .svh = parts[2], .wq = parts[3] };
}

test "the one-row path reads the oracle's real mcg 8.00 bpw block bit for bit; float64 dequant agreement is reported" {
    const gpa = std.testing.allocator;
    const oracle = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, "zig/tests/fixtures/exl3_oracle.bin", gpa, .limited(1 << 22));
    defer gpa.free(oracle);
    const real = realBlock(oracle);
    try std.testing.expectEqual(@as(usize, 128 * 128 * 2), real.wq.len);
    var g = try Gpu.init();
    defer g.deinit();
    var prng = std.Random.DefaultPrng.init(0x7265616c);
    const w = try Weight.init(g.device, 128, 128, real.halves, .mcg, prng.random(), true);
    defer w.deinit();
    @memcpy(w.bufs[0].contents()[0..real.trellis.len], real.trellis);
    try probeWq(&g, w.matrix, std.mem.bytesAsSlice(u16, real.wq));

    @memcpy(w.bufs[1].contents()[0..256], real.suh);
    @memcpy(w.bufs[2].contents()[0..256], real.svh);
    const x = try g.device.buffer(256, shared);
    defer x.deinit();
    for (x.slice(u16, 128)) |*v| v.* = bf16(prng.random().float(f32) * 2 - 1);
    const y = try g.device.buffer(512, shared);
    defer y.deinit();
    const scratch = try g.device.buffer(try ex.scratchBytes(128, 128, 1), shared);
    defer scratch.deinit();
    const cb = g.queue.commandBuffer();
    const e = cb.compute(.concurrent);
    try ex.encode(&g.p, e, w.matrix, .{ .buffer = x }, .{ .buffer = y }, .{ .buffer = scratch }, .{ .count = 1, .x_stride = 128, .y_stride = 128, .output = .f32 });
    try complete(cb, e);
    const dq = try gpa.alloc(f64, 3 * 128 * 128);
    defer gpa.free(dq);
    var suh: [128]f16 = undefined;
    var svh: [128]f16 = undefined;
    @memcpy(std.mem.sliceAsBytes(&suh), real.suh);
    @memcpy(std.mem.sliceAsBytes(&svh), real.svh);
    fmt.dequantize(w.matrix.width, .mcg, w.words(), 8, 8, &suh, &svh, dq[0 .. 128 * 128], dq[128 * 128 ..]);
    var worst: f64 = 0;
    var top: f64 = 0;
    for (y.slice(f32, 128), 0..) |got, col| {
        var want: f64 = 0;
        for (x.slice(u16, 128), 0..) |xv, i| want += @as(f64, @as(f32, @bitCast(@as(u32, xv) << 16))) * dq[i * 128 + col];
        worst = @max(worst, @abs(got - want));
        top = @max(top, @abs(want));
    }
    std.debug.print("exl3 row: real mcg block against float64 dequant: max |gpu - f64| {e:.3}, max |y| {e:.3}\n", .{ worst, top });
}

/// A DeepSeek-V4.1 dense projection at decode; `view` takes group 3's 1024 columns of an 8192-column parent.
pub const Shape = struct { name: []const u8, k: usize, n: usize, halves: u16, book: fmt.Codebook, view: bool = false, full: bool = false };

pub const v41 = [_]Shape{
    .{ .name = "wq_a", .k = 5120, .n = 1280, .halves = 4, .book = .mcg },
    .{ .name = "wkv", .k = 5120, .n = 512, .halves = 5, .book = .mul1, .full = true },
    .{ .name = "wq_b", .k = 1280, .n = 32768, .halves = 6, .book = .inst3 },
    .{ .name = "wo_a group", .k = 4096, .n = 1024, .halves = 8, .book = .mul1, .view = true },
    .{ .name = "wo_b", .k = 8192, .n = 5120, .halves = 10, .book = .mcg },
    .{ .name = "shared w1/w3", .k = 5120, .n = 2304, .halves = 12, .book = .mul1 },
    .{ .name = "shared w2", .k = 2304, .n = 5120, .halves = 16, .book = .mcg },
    .{ .name = "ix_wq_b", .k = 1280, .n = 4096, .halves = 3, .book = .mul1 },
};

test "the one-row dot equals the serial reference and tf_exl3_matmul_hoisted bit for bit at the DeepSeek-V4.1 shapes" {
    var g = try Gpu.init();
    defer g.deinit();
    const main = g.main();
    var prng = std.Random.DefaultPrng.init(0x763431);
    for (v41) |s| {
        const parent_n: usize = if (s.view) 8192 else s.n;
        const w = try Weight.init(g.device, s.k, parent_n, s.halves, s.book, prng.random(), false);
        defer w.deinit();
        var m = w.matrix;
        const tn0: usize = if (s.view) 3 * 1024 / 16 else 0;
        if (s.view) {
            m.n = s.n;
            m.tile_k_stride = (parent_n / 16) * m.tileBytes();
            m.trellis.offset = tn0 * m.tileBytes();
            m.svh.offset = tn0 * 16 * 2;
        }
        const x = try g.device.buffer(s.k * 2, shared);
        defer x.deinit();
        for (x.slice(u16, s.k)) |*v| v.* = bf16(prng.random().float(f32) * 2 - 1);
        const sb = try ex.scratchBytes(s.k, s.n, 1);
        const scratch = try g.device.buffer(2 * sb, shared);
        defer scratch.deinit();
        const y = try g.device.buffer(2 * s.n * 4, shared);
        defer y.deinit();
        const rows: ex.Rows = .{ .count = 1, .x_stride = s.k, .y_stride = s.n, .output = .f32 };
        const cb = g.queue.commandBuffer();
        const e = cb.compute(.concurrent);
        try ex.encode(&g.p, e, m, .{ .buffer = x }, .{ .buffer = y }, .{ .buffer = scratch }, rows);
        try ex.encode(&main, e, m, .{ .buffer = x }, .{ .buffer = y, .offset = s.n * 4 }, .{ .buffer = scratch, .offset = sb }, rows);
        try complete(cb, e);
        const bytes = scratch.contents()[0 .. 2 * sb];
        try std.testing.expectEqualSlices(u8, bytes[sb..][0..sb], bytes[0..sb]);
        try std.testing.expectEqualSlices(u32, y.slice(u32, 2 * s.n)[s.n..], y.slice(u32, s.n));
        const rotated: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, bytes[0 .. s.k * 2]));
        const sums: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, bytes[s.k * 2 ..][0 .. s.n * 4]));
        const blocks = s.n / 128;
        const picks = [_]usize{ 0, blocks / 2, blocks - 1 };
        var want: [128]f32 = undefined;
        for (0..blocks) |b| {
            if (!s.full and std.mem.indexOfScalar(usize, &picks, b) == null) continue;
            ref.dot(m.width, s.book, w.words(), parent_n / 16, rotated, tn0 + b * 8, &want);
            for (want, sums[b * 128 ..][0..128], 0..) |r, got, col| {
                if (@as(u32, @bitCast(r)) != @as(u32, @bitCast(got))) {
                    std.debug.print("{s}: column {d}: gpu {e} where the serial reference gives {e}\n", .{ s.name, b * 128 + col, got, r });
                    return error.Exl3RowMismatch;
                }
            }
        }
    }
}
