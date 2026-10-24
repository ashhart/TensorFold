//! Gemma 4's Metal kernels on a Mac without tensor units: what builds, and the row path's bits.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");

const g4 = tf.gemma4;
const row = tf.row_projection;

/// The bf16 of a float, as the checkpoint stores it.
fn bf16(f: f32) u16 {
    return @truncate(@as(u32, @bitCast(f)) >> 16);
}

/// A bf16's value, as a float.
fn value(b: u16) f32 {
    return @bitCast(@as(u32, b) << 16);
}

/// One path's outputs against the host reference: the exact count and the worst difference.
fn report(what: []const u8, y: mtl.Buffer, rows: usize, n: usize, kk: usize, groups: usize, codes: []const u8, xs: []const u16, scales: mtl.Buffer, biases: mtl.Buffer) void {
    const scs = scales.slice(u16, n * groups);
    const bis = biases.slice(u16, n * groups);
    const ys = y.slice(u16, rows * n);
    var worst: f32 = 0;
    var exact: usize = 0;
    var within: usize = 0;
    for (0..rows) |r| {
        for (0..n) |o| {
            var acc: f64 = 0;
            for (0..groups) |g| {
                var sum: f64 = 0;
                var dot: f64 = 0;
                for (0..64) |t| {
                    const xv: f64 = value(xs[r * kk + g * 64 + t]);
                    sum += xv;
                    dot += xv * @as(f64, @floatFromInt(codes[o * kk + g * 64 + t]));
                }
                acc += value(scs[o * groups + g]) * dot + sum * value(bis[o * groups + g]);
            }
            const got: f32 = value(ys[r * n + o]);
            const want: f32 = @floatCast(acc);
            const diff = @abs(got - want);
            if (diff > worst) worst = diff;
            if (got == want) exact += 1;
            // one bf16 ulp: the fp32 accumulation order is the only difference.
            const ulp = @abs(want) * @as(f32, 0x1p-8) + @as(f32, 0x1p-20);
            if (diff <= ulp) within += 1;
        }
    }
    std.debug.print("{s}: {d} of {d} exact in bf16, {d} within a bf16 ulp of the fp64 reference, the worst difference {e:.6}\n", .{ what, exact, rows * n, within, worst });
}

/// The bias term alone: the group sum's order on its own, the scales at zero.
fn biasOrder(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, ps: *g4.weights.Projections, n: usize, xs: []const u16) !void {
    const kk: usize = 64;
    const groups: usize = 1;
    const words = try a.alloc(u32, n * (kk / 8));
    defer a.free(words);
    @memset(words, 0);
    const scales = try a.alloc(u16, n * groups);
    defer a.free(scales);
    @memset(scales, 0);
    const biases = try a.alloc(u16, n * groups);
    defer a.free(biases);
    for (biases, 0..) |*v, i| v.* = bf16(0.003 + @as(f32, @floatFromInt(i)) * 1e-5);
    var owned: std.ArrayList(mtl.Buffer) = .empty;
    defer owned.deinit(a);
    defer for (owned.items) |b| b.deinit();
    const placeholder = try device.buffer(4, mtl.ResourceOptions.shared);
    defer placeholder.deinit();
    const l = try g4.weights.packRows(a, device, &owned, words, scales, biases, .{ .w = placeholder, .sb = placeholder, .n = n, .k = kk, .format = .{ .bits = 4, .group = 64 } });
    const rows: usize = 1;
    const x = try device.buffer(rows * kk * 2, mtl.ResourceOptions.shared);
    defer x.deinit();
    @memcpy(x.slice(u16, rows * kk), xs[0 .. rows * kk]);
    const y = try device.buffer(rows * n * 2, mtl.ResourceOptions.shared);
    defer y.deinit();
    const cb = queue.commandBuffer();
    const e = cb.compute(.serial);
    try g4.weights.projectRow(ps, device, e, l.raw.?, l, .{ .buf = x }, y, rows);
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |m| {
        std.debug.print("the GPU command failed: {s}\n", .{m});
        return error.GpuFailure;
    }
    var sum: f32 = 0;
    for (0..64) |t| sum += value(xs[t]);
    const ys = y.slice(u16, rows * n);
    var same: usize = 0;
    var worst: f32 = 0;
    for (0..n) |j| {
        // the lane path's group sum: the group's 64 values added in order, then the bias's one fma.
        const want: f32 = value(biases[j]) * sum;
        const got: f32 = value(ys[j]);
        if (got == value(bf16(want))) same += 1;
        const diff = @abs(got - value(bf16(want)));
        if (diff > worst) worst = diff;
    }
    std.debug.print("the bias term alone: {d} of {d} equal the lane path's group sum in bits, the worst difference {e:.6}\n", .{ same, n, worst });
}

pub fn main(_: std.process.Init) !void {
    const a = std.heap.page_allocator;
    const device = try mtl.Device.init();
    defer device.deinit();
    std.debug.print("the device: {s} (generation {d}), tensor units: {}\n", .{ device.name(), device.generation(), device.tensorUnits() });
    const queue = try device.queue();
    defer queue.deinit();

    // the glue and the prompt attention: the MPP ones, which a chip without tensor units runs at simdgroup rate.
    var k = try g4.kernels.Kind.init(a, device, g4.config.dense31b, false, false, 4);
    defer k.deinit();
    var kg = try g4.kernels.Kind.init(a, device, g4.config.dense31b, true, false, 4);
    defer kg.deinit();
    std.debug.print("the glue and the prompt attention: built, the sliding and the global layers\n", .{});

    // the attention's op: the same nax.h fragment layout the M5 uses, checked against the host's sums.
    try tf.frags.check(device, queue, a);
    std.debug.print("the attention's op: bit for bit the host's sums on this GPU\n", .{});

    // the image tower's dense: the same nax.h branch, at the vision's width.
    {
        const text = try g4.affine_mm.source(device, a, .{ .bits = g4.config.widths.fast.bits, .group = g4.config.widths.fast.group });
        defer a.free(text);
        const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
        defer lib.deinit();
        var mm: g4.affine_mm.Pipes = undefined;
        for (&mm, g4.affine_mm.names) |*p, name| p.* = try mtl.Pipeline.init(device, lib, name, false);
        defer for (&mm) |*p| p.deinit();
        std.debug.print("the image tower's dense: built\n", .{});
    }

    // the lane projection: 4-bit groups of 64 in the reg layout, the weights' own layout.
    const layout = tf.lane_projection.Layout{ .n = 256, .k = 128, .format = .{ .bits = 4, .group = 64 }, .precompute_sums = true, .reg = true };
    if (tf.lane_projection.Projection.init(a, device, layout)) |p| {
        defer p.deinit();
        std.debug.print("the lane projection: built\n", .{});
    } else |e| {
        std.debug.print("the lane projection: not built ({s})\n", .{@errorName(e)});
    }

    // the row path's weights: 4-bit groups of 64, fp32 sums, as the checkpoint stores them.
    const n: usize = 256;
    const kk: usize = 128;
    const groups = kk / 64;
    const codes = kk / 8;
    var prng = std.Random.DefaultPrng.init(11);
    const rnd = prng.random();
    const host_codes = try a.alloc(u8, n * kk);
    defer a.free(host_codes);
    for (host_codes) |*c| c.* = @intCast(rnd.int(u8) & 0x0f);
    const words = try a.alloc(u32, n * codes);
    defer a.free(words);
    for (0..n) |r| {
        for (0..codes) |wrd| {
            var word: u32 = 0;
            for (0..8) |t| word |= @as(u32, host_codes[r * kk + wrd * 8 + t]) << @intCast(4 * t);
            words[r * codes + wrd] = word;
        }
    }
    const host_scales = try a.alloc(u16, n * groups);
    defer a.free(host_scales);
    const host_biases = try a.alloc(u16, n * groups);
    defer a.free(host_biases);
    for (host_scales, 0..) |*v, i| v.* = bf16(0.01 + @as(f32, @floatFromInt(i % 7)) * 0.001);
    for (host_biases, 0..) |*v, i| v.* = bf16(-0.002 + @as(f32, @floatFromInt(i % 5)) * 0.0005);

    // the family's own packing on this chip: the checkpoint's arrays as stored.
    var owned: std.ArrayList(mtl.Buffer) = .empty;
    defer owned.deinit(a);
    defer for (owned.items) |b| b.deinit();
    const placeholder = try device.buffer(4, mtl.ResourceOptions.shared);
    defer placeholder.deinit();
    const linear = try g4.weights.packRows(a, device, &owned, words, host_scales, host_biases, .{ .w = placeholder, .sb = placeholder, .n = n, .k = kk, .format = .{ .bits = 4, .group = 64 } });
    if (linear.raw == null) return error.NoRowWeights;
    std.debug.print("the family's row weights: packed\n", .{});

    // the input: bf16 rows [rows, k].
    const rows: usize = 8;
    const x = try device.buffer(rows * kk * 2, mtl.ResourceOptions.shared);
    defer x.deinit();
    {
        const xs = x.slice(u16, rows * kk);
        for (xs, 0..) |*v, i| v.* = bf16(@as(f32, @floatFromInt(@as(i32, @intCast(i % 17)) - 8)) * 0.125);
    }
    const y = try device.buffer(rows * n * 2, mtl.ResourceOptions.shared);
    defer y.deinit();

    var projections = g4.weights.Projections{ .geo = g4.config.dense31b };
    defer projections.deinit(a);

    // the family's projection on this chip: the simdgroup tiles.
    {
        const cb = queue.commandBuffer();
        const e = cb.compute(.serial);
        try g4.weights.projectRow(&projections, device, e, linear.raw.?, linear, .{ .buf = x }, y, rows);
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |m| {
            std.debug.print("the GPU command failed: {s}\n", .{m});
            return error.GpuFailure;
        }
    }
    report("the family's row path", y, rows, n, kk, groups, host_codes, x.slice(u16, rows * kk), owned.items[1], owned.items[2]);

    // the plain row kernel on the same weights: the path the tiles match.
    {
        var rp = try row.Pipelines.load(device);
        defer rp.deinit();
        const weights = row.Weights{ .w = owned.items[0], .scales = owned.items[1], .biases = owned.items[2], .n = n, .k = kk, .group = 64, .bits = 4, .sum = .f32 };
        try weights.validate();
        const rcall = try row.call(&rp, weights, rows, false);
        const cb = queue.commandBuffer();
        const e = cb.compute(.serial);
        e.setPipeline(rcall.pipeline);
        e.setBuffer(x, 0, 0);
        e.setBuffer(weights.w, 0, 1);
        e.setBuffer(weights.scales, 0, 2);
        e.setBuffer(weights.biases, 0, 3);
        e.setValue(rcall.dims, 4);
        e.setBuffer(y, 0, 5);
        e.dispatchGroups(.{ .width = rcall.groups }, .{ .width = rcall.threads });
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |m| {
            std.debug.print("the GPU command failed: {s}\n", .{m});
            return error.GpuFailure;
        }
    }
    report("the plain row kernel", y, rows, n, kk, groups, host_codes, x.slice(u16, rows * kk), owned.items[1], owned.items[2]);

    // the bias term alone: the group sum's order, the one part of the contract the row path shows by itself.
    try biasOrder(a, device, queue, &projections, n, x.slice(u16, rows * kk));
}
