//! Ceiling-power-of-two dense dispatch and tight-buffer Metal checks for Inkling 2 Mini.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const ink2_src = @import("inkling2_metal");
const ck = @import("../../../core/checkpoint_metal.zig");
const dense = @import("dense.zig");
const fwd = @import("forward.zig");
const t = std.testing;

const FakeModel = struct { dense: [6]u32 = .{ 1, 2, 4, 8, 16, 32 } };

const Launch = struct {
    variant_rows: u32 = 0,
    x_off: usize = 0,
    w_off: usize = 0,
    y_off: usize = 0,
    args: fwd.DenseArgs = .{ .rows = 0, .n = 0, .k = 0, .ldy = 0 },
    groups: mtl.Size = mtl.Size.of(0, 0, 0),
    threads: mtl.Size = mtl.Size.of(0, 0, 0),
};

const Recorder = struct {
    launches: [64]Launch = undefined,
    count: usize = 0,
    current: Launch = .{},

    pub fn setPipeline(r: *Recorder, rows: u32) void {
        r.current.variant_rows = rows;
    }
    pub fn setBuffer(r: *Recorder, _: mtl.Buffer, off: usize, slot: usize) void {
        switch (slot) {
            0 => r.current.x_off = off,
            1 => r.current.w_off = off,
            3 => r.current.y_off = off,
            else => unreachable,
        }
    }
    pub fn setValue(r: *Recorder, val: anytype, slot: usize) void {
        std.debug.assert(slot == 2);
        r.current.args = .{ .rows = val.rows, .n = val.n, .k = val.k, .ldy = val.ldy };
    }
    pub fn dispatchGroups(r: *Recorder, groups: mtl.Size, threads: mtl.Size) void {
        r.current.groups = groups;
        r.current.threads = threads;
        r.launches[r.count] = r.current;
        r.count += 1;
        r.current = .{};
    }
};

fn dummyBuf(tag: usize) mtl.Buffer {
    return .{ .id = @ptrFromInt(tag * 0x1000) };
}

fn expectedVariantRows(rows: usize) u32 {
    if (rows <= 1) return 1;
    if (rows <= 2) return 2;
    if (rows <= 4) return 4;
    if (rows <= 8) return 8;
    if (rows <= 16) return 16;
    return 32;
}

fn toBf16(x: f32) u16 {
    const bits: u32 = @bitCast(x);
    if (std.math.isNan(x)) return @truncate((bits >> 16) | 0x40);
    return @truncate((bits +% 0x7fff +% ((bits >> 16) & 1)) >> 16);
}

fn fromBf16(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}

test "dense ceiling pow2 dispatch reads weights once up to 32 rows and twice up to 64" {
    const n: usize = 3072;
    const k: usize = 2304;
    const w_off: usize = 512;
    const x_off: usize = 256;
    const y_off: usize = 128;
    const weight = ck.Tensor{
        .buffer = dummyBuf(1),
        .offset = w_off,
        .bytes = n * k * 2,
        .dtype = .bf16,
        .shape = .{ n, k, 1, 1, 1 },
        .rank = 2,
    };
    const model = FakeModel{};
    var zero_rec = Recorder{};
    fwd.encodeDense(*Recorder, &model, &zero_rec, weight, .{ .buf = dummyBuf(2), .off = x_off }, .{ .buf = dummyBuf(3), .off = y_off }, 0);
    try t.expectEqual(@as(usize, 0), zero_rec.count);

    for (1..65) |rows| {
        var rec = Recorder{};
        fwd.encodeDense(*Recorder, &model, &rec, weight, .{ .buf = dummyBuf(2), .off = x_off }, .{ .buf = dummyBuf(3), .off = y_off }, rows);
        const want_dispatches: usize = if (rows <= 32) 1 else 2;
        try t.expectEqual(want_dispatches, rec.count);
        var covered: usize = 0;
        for (rec.launches[0..rec.count]) |launch| {
            const left = rows - covered;
            const take = @min(left, 32);
            try t.expectEqual(expectedVariantRows(take), launch.variant_rows);
            try t.expectEqual(@as(u32, @intCast(take)), launch.args.rows);
            try t.expectEqual(@as(u32, n), launch.args.n);
            try t.expectEqual(@as(u32, k), launch.args.k);
            try t.expectEqual(@as(u32, n), launch.args.ldy);
            try t.expectEqual(x_off + covered * k * 2, launch.x_off);
            try t.expectEqual(w_off, launch.w_off);
            try t.expectEqual(y_off + covered * n * 2, launch.y_off);
            try t.expectEqual((n + 7) / 8, launch.groups.width);
            try t.expectEqual(@as(usize, 256), launch.threads.width);
            covered += take;
        }
        try t.expectEqual(rows, covered);
    }
}

test "bf16_matvec guards zero rows and bounds X row reads for partial tiles" {
    const src = sources.spark25_glue;
    try t.expect(std.mem.indexOf(u8, src, "a.rows == 0") != null);
    try t.expect(std.mem.indexOf(u8, src, "min(r, active - 1u)") != null);
    try t.expectEqual(@as(usize, 1), std.mem.count(u8, src, "for (uint i = 8 * lane; i < a.k; i += 256) {"));
}

test "dense tiled serves, grid, and launch encoding" {
    // Reject invalid shapes and rows below tiled_min_rows.
    try t.expect(!dense.Tiled.valid(.{ .rows = 0, .n = 24, .k = 2304, .ldy = 24 }));
    try t.expect(!dense.Tiled.valid(.{ .rows = 8, .n = 0, .k = 2304, .ldy = 0 }));
    try t.expect(!dense.Tiled.valid(.{ .rows = 8, .n = 12, .k = 2304, .ldy = 12 }));
    try t.expect(!dense.Tiled.valid(.{ .rows = 8, .n = 24, .k = 128, .ldy = 24 }));
    try t.expect(!dense.Tiled.valid(.{ .rows = 8, .n = 24, .k = 2304, .ldy = 16 }));
    for (1..dense.tiled_min_rows) |rows| {
        try t.expect(!dense.Tiled.serves(.{ .rows = @intCast(rows), .n = 3072, .k = 2304, .ldy = 3072 }));
    }
    const served_shapes = [_]dense.Call{
        .{ .rows = 8, .n = 24, .k = 2304, .ldy = 24 },
        .{ .rows = 9, .n = 384, .k = 2304, .ldy = 384 },
        .{ .rows = 17, .n = 1024, .k = 2304, .ldy = 1024 },
        .{ .rows = 45, .n = 2304, .k = 3072, .ldy = 2304 },
        .{ .rows = 256, .n = 3072, .k = 2304, .ldy = 3072 },
    };
    for (served_shapes) |c| {
        try t.expect(dense.Tiled.serves(c));
        var rec = Recorder{};
        dense.Tiled.encodeLaunch(*Recorder, @as(u32, 99), &rec, .{ .buf = dummyBuf(2), .off = 128 }, .{ .buf = dummyBuf(1), .off = 256 }, .{ .buf = dummyBuf(3), .off = 64 }, c);
        try t.expectEqual(@as(usize, 1), rec.count);
        const launch = rec.launches[0];
        try t.expectEqual(@as(u32, 99), launch.variant_rows);
        try t.expectEqual(@as(usize, 128), launch.x_off);
        try t.expectEqual(@as(usize, 256), launch.w_off);
        try t.expectEqual(@as(usize, 64), launch.y_off);
        try t.expectEqual(c.rows, launch.args.rows);
        try t.expectEqual(c.n, launch.args.n);
        try t.expectEqual(c.k, launch.args.k);
        try t.expectEqual(c.ldy, launch.args.ldy);
        try t.expectEqual((c.rows + 15) / 16, launch.groups.width);
        try t.expectEqual((c.n + 15) / 16, launch.groups.height);
        try t.expectEqual(@as(usize, 1), launch.groups.depth);
        try t.expectEqual(@as(usize, dense.tiled_threads), launch.threads.width);
    }
}

test "dense_tiled shader guards bounds, tail rows, and 256-thread reservation" {
    const src = ink2_src.dense_tiled;
    try t.expect(std.mem.indexOf(u8, src, "max_total_threads_per_threadgroup(256)") != null);
    try t.expect(std.mem.indexOf(u8, src, "if (r0 >= a.rows || n0 >= a.n) return;") != null);
    try t.expect(std.mem.indexOf(u8, src, "min(r0 + 2u * g, a.rows - 1u)") != null);
    try t.expect(std.mem.indexOf(u8, src, "min(r0 + g, a.rows - 1u)") != null);
    try t.expectEqual(@as(usize, 1), std.mem.count(u8, src, "acc0[r] += dot(w0_lo, x0) + dot(w0_hi, x1);"));
    try t.expectEqual(@as(usize, 1), std.mem.count(u8, src, "acc1[r] += dot(w1_lo, x0) + dot(w1_hi, x1);"));
}

const GpuModel = struct { dense: [6]mtl.Pipeline };

fn loadDensePipelines(device: mtl.Device) !GpuModel {
    const lib = try mtl.Library.fromSource(device, sources.spark25_glue, .{
        .language = mtl.CompileOptions.mlx().language,
        .math_functions = .precise,
    });
    defer lib.deinit();
    var m: GpuModel = undefined;
    var made: usize = 0;
    errdefer for (m.dense[0..made]) |p| p.deinit();
    for ([_]u32{ 1, 2, 4, 8, 16, 32 }, 0..) |rows, i| {
        var buf: [32]u8 = undefined;
        const name = try std.fmt.bufPrintSentinel(&buf, "sp_matvec_{d}", .{rows}, 0);
        m.dense[i] = try mtl.Pipeline.init(device, lib, name, false);
        made += 1;
    }
    return m;
}

fn checkShapeExact(device: mtl.Device, queue: mtl.Queue, model: *const GpuModel, rows: usize, n: usize, k: usize, nonfinite: bool) !void {
    const a = t.allocator;
    const x_vals = try a.alloc(u16, rows * k);
    defer a.free(x_vals);
    const w_vals = try a.alloc(u16, n * k);
    defer a.free(w_vals);
    var prng = std.Random.DefaultPrng.init(0x5a17 ^ rows ^ (n << 8));
    const rng = prng.random();
    for (x_vals) |*v| v.* = toBf16(rng.float(f32) * 2 - 1);
    for (w_vals) |*v| v.* = toBf16(rng.float(f32) * 2 - 1);
    if (nonfinite) {
        x_vals[0] = toBf16(std.math.nan(f32));
        x_vals[k - 1] = toBf16(std.math.inf(f32));
        if (rows > 2) x_vals[2 * k + 7] = toBf16(-std.math.inf(f32));
    }

    // Tight-sized X buffer (exact rows * k * 2 bytes, zero trailing slack) and guarded Y buffer.
    const x_tight = try device.buffer(rows * k * 2, mtl.ResourceOptions.shared);
    defer x_tight.deinit();
    @memcpy(x_tight.slice(u16, rows * k), x_vals);

    const w_buf = try device.buffer(n * k * 2, mtl.ResourceOptions.shared);
    defer w_buf.deinit();
    @memcpy(w_buf.slice(u16, n * k), w_vals);

    const guard = 64;
    const y_bytes = rows * n * 2;
    const y_ref = try device.buffer(y_bytes, mtl.ResourceOptions.shared);
    defer y_ref.deinit();
    const y_got = try device.buffer(y_bytes + 2 * guard, mtl.ResourceOptions.shared);
    defer y_got.deinit();
    @memset(y_got.slice(u8, y_bytes + 2 * guard), 0xff);

    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = queue.commandBuffer();
    const e = cb.compute(.serial);
    for (0..rows) |r| {
        e.setPipeline(model.dense[0]);
        e.setBuffer(x_tight, r * k * 2, 0);
        e.setBuffer(w_buf, 0, 1);
        e.setValue(fwd.DenseArgs{ .rows = 1, .n = @intCast(n), .k = @intCast(k), .ldy = @intCast(n) }, 2);
        e.setBuffer(y_ref, r * n * 2, 3);
        e.dispatchGroups(mtl.Size.of((n + 7) / 8, 1, 1), mtl.Size.of(256, 1, 1));
    }
    const weight = ck.Tensor{
        .buffer = w_buf,
        .offset = 0,
        .bytes = n * k * 2,
        .dtype = .bf16,
        .shape = .{ n, k, 1, 1, 1 },
        .rank = 2,
    };
    fwd.encodeDense(mtl.ComputeEncoder, model, e, weight, .{ .buf = x_tight, .off = 0 }, .{ .buf = y_got, .off = guard }, rows);
    e.end();
    cb.commit();
    cb.wait();
    try t.expect(cb.failure() == null);

    const got_raw = y_got.slice(u8, y_bytes + 2 * guard);
    for (got_raw[0..guard]) |b| try t.expectEqual(@as(u8, 0xff), b);
    for (got_raw[guard + y_bytes ..]) |b| try t.expectEqual(@as(u8, 0xff), b);
    const got16 = std.mem.bytesAsSlice(u16, @as([]align(2) const u8, @alignCast(got_raw[guard .. guard + y_bytes])));
    try t.expectEqualSlices(u16, y_ref.slice(u16, rows * n), got16);
    if (!nonfinite and (rows == 5 or rows == 17 or rows == 31)) {
        for (0..rows) |r| for (0..n) |j| {
            var acc: f64 = 0;
            var mag: f64 = 0;
            for (0..k) |i| {
                const prod = @as(f64, fromBf16(x_vals[r * k + i])) * fromBf16(w_vals[j * k + i]);
                acc += prod;
                mag += @abs(prod);
            }
            const got_f = @as(f64, fromBf16(got16[r * n + j]));
            const bound = 0x1p-22 * mag + 0x1p-8 * @abs(acc) + 0x1p-133;
            try t.expect(@abs(got_f - acc) <= bound);
        };
    }
}

test "sp_matvec non-power-of-two rows are bit-equal to matvec_1 on tight X buffers" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    const model = try loadDensePipelines(device);
    defer for (model.dense) |p| p.deinit();

    // Zero rows on every pipeline must neither fault nor write to the poisoned output buffer.
    {
        const x_one = try device.buffer(512 * 2, mtl.ResourceOptions.shared);
        defer x_one.deinit();
        const w_one = try device.buffer(64 * 512 * 2, mtl.ResourceOptions.shared);
        defer w_one.deinit();
        const y_one = try device.buffer(64 * 2, mtl.ResourceOptions.shared);
        defer y_one.deinit();
        @memset(x_one.slice(u16, 512), toBf16(1.0));
        @memset(w_one.slice(u16, 64 * 512), toBf16(1.0));
        @memset(y_one.slice(u8, 64 * 2), 0xff);
        for (model.dense) |pipe| {
            const pool = mtl.objc.Pool.push();
            defer pool.pop();
            const cb = queue.commandBuffer();
            const e = cb.compute(.serial);
            e.setPipeline(pipe);
            e.setBuffer(x_one, 0, 0);
            e.setBuffer(w_one, 0, 1);
            e.setValue(fwd.DenseArgs{ .rows = 0, .n = 64, .k = 512, .ldy = 64 }, 2);
            e.setBuffer(y_one, 0, 3);
            e.dispatchGroups(mtl.Size.of(8, 1, 1), mtl.Size.of(256, 1, 1));
            e.end();
            cb.commit();
            cb.wait();
            try t.expect(cb.failure() == null);
            for (y_one.slice(u8, 64 * 2)) |b| try t.expectEqual(@as(u8, 0xff), b);
        }
    }

    const widths = [_]usize{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 16, 17, 24, 31, 32, 33, 63, 64 };
    for (widths) |rows| {
        try checkShapeExact(device, queue, &model, rows, 72, 512, false);
        try checkShapeExact(device, queue, &model, rows, 24, 256, false);
    }
    for ([_]usize{ 3, 5, 6, 7, 9, 15, 17, 31 }) |rows| {
        try checkShapeExact(device, queue, &model, rows, 72, 512, true);
    }
}

fn checkTiledExact(device: mtl.Device, queue: mtl.Queue, matvec1: mtl.Pipeline, tiled: dense.Tiled, rows: usize, n: usize, k: usize, nonfinite: bool) !void {
    const a = t.allocator;
    const x_vals = try a.alloc(u16, rows * k);
    defer a.free(x_vals);
    const w_vals = try a.alloc(u16, n * k);
    defer a.free(w_vals);
    var prng = std.Random.DefaultPrng.init(0x711ed ^ rows ^ (n << 8));
    const rng = prng.random();
    for (x_vals) |*v| v.* = toBf16(rng.float(f32) * 2 - 1);
    for (w_vals) |*v| v.* = toBf16(rng.float(f32) * 2 - 1);
    if (nonfinite) {
        x_vals[0] = toBf16(std.math.nan(f32));
        x_vals[k - 1] = toBf16(std.math.inf(f32));
        if (rows > 2) x_vals[2 * k + 7] = toBf16(-std.math.inf(f32));
    }

    const x_tight = try device.buffer(rows * k * 2, mtl.ResourceOptions.shared);
    defer x_tight.deinit();
    @memcpy(x_tight.slice(u16, rows * k), x_vals);
    const w_buf = try device.buffer(n * k * 2, mtl.ResourceOptions.shared);
    defer w_buf.deinit();
    @memcpy(w_buf.slice(u16, n * k), w_vals);

    const guard = 64;
    const y_bytes = rows * n * 2;
    const y_ref = try device.buffer(y_bytes, mtl.ResourceOptions.shared);
    defer y_ref.deinit();
    const y_got = try device.buffer(y_bytes + 2 * guard, mtl.ResourceOptions.shared);
    defer y_got.deinit();
    @memset(y_got.slice(u8, y_bytes + 2 * guard), 0xff);

    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = queue.commandBuffer();
    const e = cb.compute(.serial);
    for (0..rows) |r| {
        e.setPipeline(matvec1);
        e.setBuffer(x_tight, r * k * 2, 0);
        e.setBuffer(w_buf, 0, 1);
        e.setValue(fwd.DenseArgs{ .rows = 1, .n = @intCast(n), .k = @intCast(k), .ldy = @intCast(n) }, 2);
        e.setBuffer(y_ref, r * n * 2, 3);
        e.dispatchGroups(mtl.Size.of((n + 7) / 8, 1, 1), mtl.Size.of(256, 1, 1));
    }
    tiled.launch(e, .{ .buf = x_tight, .off = 0 }, .{ .buf = w_buf, .off = 0 }, .{ .buf = y_got, .off = guard }, .{ .rows = @intCast(rows), .n = @intCast(n), .k = @intCast(k), .ldy = @intCast(n) });
    e.end();
    cb.commit();
    cb.wait();
    try t.expect(cb.failure() == null);

    const got_raw = y_got.slice(u8, y_bytes + 2 * guard);
    for (got_raw[0..guard]) |b| try t.expectEqual(@as(u8, 0xff), b);
    for (got_raw[guard + y_bytes ..]) |b| try t.expectEqual(@as(u8, 0xff), b);
    const got16 = std.mem.bytesAsSlice(u16, @as([]align(2) const u8, @alignCast(got_raw[guard .. guard + y_bytes])));
    const ref16 = y_ref.slice(u16, rows * n);
    if (!nonfinite) {
        try t.expectEqualSlices(u16, ref16, got16);
    } else {
        for (ref16, got16) |r_bits, g_bits| {
            const rf = fromBf16(r_bits);
            const gf = fromBf16(g_bits);
            if (std.math.isNan(rf) and std.math.isNan(gf)) continue;
            try t.expectEqual(r_bits, g_bits);
        }
    }
}

test "tf_ink2_dense_tiled is bit-equal to matvec_1 on tight X buffers across all row and column tails" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    const model = try loadDensePipelines(device);
    defer for (model.dense) |p| p.deinit();
    const tiled = try dense.Tiled.loadSource(device, ink2_src.dense_tiled);
    defer tiled.deinit();

    for ([_]usize{ 1, 5, 7, 8, 9, 15, 16, 17, 32, 45 }) |rows| {
        try checkTiledExact(device, queue, model.dense[0], tiled, rows, 24, 256, false);
        try checkTiledExact(device, queue, model.dense[0], tiled, rows, 72, 512, false);
    }
    for ([_]usize{ 5, 7, 9, 17 }) |rows| {
        try checkTiledExact(device, queue, model.dense[0], tiled, rows, 24, 512, true);
    }
}
