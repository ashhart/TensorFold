//! The reg kernel, every output byte: 1-16 rows against the cooperative one, 129-2048 rows against 128-row dispatches.
const std = @import("std");
const mtl = @import("metal");
const lane = @import("core_lane");
const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;

fn bf16(v: f32) u16 {
    const bits: u32 = @bitCast(v);
    return @intCast((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
}

fn run(queue: mtl.Queue, p: lane.Projection, x: mtl.Buffer, w: mtl.Buffer, sb: mtl.Buffer, sums: ?lane.Sums, y: mtl.Buffer, rows: u32) !void {
    const cb = queue.commandBuffer();
    const e = cb.compute(.serial);
    if (sums) |s| try p.encodeSums(e, .{ .buf = x }, .{ .buf = w }, .{ .buf = sb }, s, .{ .buf = y }, rows) else try p.encode(e, .{ .buf = x }, .{ .buf = w }, .{ .buf = sb }, .{ .buf = y }, rows);
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure() != null) return error.GpuFailure;
}

/// `rows` rows through `p` in encodes of at most `step` rows, one command buffer, as quant_gpu's prompt loop does.
fn chunks(queue: mtl.Queue, p: lane.Projection, x: mtl.Buffer, w: mtl.Buffer, sb: mtl.Buffer, sums: ?lane.Sums, y: mtl.Buffer, rows: u32, step: u32) !void {
    const cb = queue.commandBuffer();
    const e = cb.compute(.serial);
    var first: u32 = 0;
    while (first < rows) : (first += step) {
        const n: u32 = @min(rows - first, step);
        const xr: lane.Ref = .{ .buf = x, .off = @as(usize, first) * p.k * 2 };
        const yr: lane.Ref = .{ .buf = y, .off = @as(usize, first) * p.y_stride * p.output_bytes };
        if (sums) |s| try p.encodeSums(e, xr, .{ .buf = w }, .{ .buf = sb }, .{ .ref = .{ .buf = s.ref.buf, .off = s.ref.off + @as(usize, first) * 4 }, .stride = s.stride }, yr, n) else try p.encode(e, xr, .{ .buf = w }, .{ .buf = sb }, yr, n);
    }
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure() != null) return error.GpuFailure;
}

/// Long reg layouts (`Layout.rows`) against 128-row dispatches, at the 27B's prompt shapes, splits and ANE share cuts.
fn longCheck(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, order: lane.RegOrder, rng: std.Random) !usize {
    var cases: usize = 0;
    // n, k, prompt split; ANE share at 3072 columns: tiles skipped each half (gate/up), first GPU input group (down)
    for ([_][5]usize{ .{ 34816, 5120, 1, 0, 0 }, .{ 34816, 5120, 1, 96, 0 }, .{ 5120, 17408, 2, 0, 0 }, .{ 5120, 17408, 2, 0, 48 }, .{ 10240, 5120, 1, 0, 0 }, .{ 12288, 5120, 2, 0, 0 }, .{ 6240, 5120, 4, 0, 0 }, .{ 5120, 6144, 2, 0, 0 }, .{ 2048, 5120, 8, 0, 0 } }) |shape| {
        const n = shape[0];
        const k = shape[1];
        const skip = shape[3];
        const half = n / 64;
        const ranges = [_][2]usize{ .{ skip, half - skip }, .{ half + skip, half - skip } };
        const cut: ?[2]usize = if (shape[4] > 0) .{ shape[4], k / 64 - shape[4] } else null;
        const base = lane.Layout{ .n = n, .k = k, .format = .{ .bits = 4, .group = 64 }, .sk = shape[2], .precompute_sums = true, .reg = true };
        var short = base;
        short.ranges = if (skip > 0) &ranges else &.{};
        short.groups = cut;
        short.output = if (cut != null) .f32 else .bf16;
        var long = short;
        long.rows = lane.max_reg_rows;
        const ob: usize = if (cut != null) 4 else 2;
        const raw = try a.alloc(u32, base.weightWords());
        for (raw) |*v| v.* = rng.int(u32);
        const scales = try a.alloc(u16, n * (k / 64));
        const biases = try a.alloc(u16, n * (k / 64));
        for (scales, biases) |*s, *b| {
            s.* = bf16(0.002 + rng.float(f32) * 0.02);
            b.* = bf16(-rng.float(f32) * 0.15);
        }
        const w = try device.buffer(base.weightWords() * 4, opts);
        defer w.deinit();
        const sb = try device.buffer(base.metadataElements() * 2, opts);
        defer sb.deinit();
        try lane.pack(base, raw, scales, biases, w.slice(u32, raw.len), sb.slice(u16, base.metadataElements()));
        try order.run(&.{.{ .w = .{ .buf = w }, .n = n, .k = k }});
        const most = lane.max_reg_rows;
        const x = try device.buffer(most * k * 2, opts);
        defer x.deinit();
        const xs = x.slice(u16, most * k);
        for (xs) |*v| v.* = bf16((rng.float(f32) - 0.5) * 4);
        // producer sums at the chunk's stride: each group's 64 values in order from zero
        const given = try device.buffer((k / 64) * most * 4, opts);
        defer given.deinit();
        const g = given.slice(f32, (k / 64) * most);
        for (0..most) |m| for (0..k / 64) |grp| {
            var acc: f32 = 0;
            for (0..64) |i| acc += @as(f32, @bitCast(@as(u32, xs[m * k + grp * 64 + i]) << 16));
            g[grp * most + m] = acc;
        };
        const want = try device.buffer(most * n * ob, opts);
        defer want.deinit();
        const got = try device.buffer(most * n * ob, opts);
        defer got.deinit();
        const p_short = try lane.Projection.init(a, device, short);
        defer p_short.deinit();
        const p_long = try lane.Projection.init(a, device, long);
        defer p_long.deinit();
        for ([_]u32{ 129, 1065, most }) |rows| for ([_]bool{ false, true }) |external| {
            const sums: ?lane.Sums = if (external) .{ .ref = .{ .buf = given }, .stride = most } else null;
            @memset(want.contents()[0..want.length()], 0x5a);
            // tiles skipped on both sides stay unwritten, so they start equal
            @memset(got.contents()[0..got.length()], if (skip > 0) 0x5a else 0xa5);
            try chunks(queue, p_short, x, w, sb, sums, want, rows, lane.reg_rows);
            try chunks(queue, p_long, x, w, sb, sums, got, rows, lane.max_reg_rows);
            if (!std.mem.eql(u8, got.contents()[0 .. rows * n * ob], want.contents()[0 .. rows * n * ob])) {
                std.debug.print("LONG DIFFER n={d} k={d} sk={d} skip={d} cut={d} rows={d} external={}\n", .{ n, k, shape[2], skip, shape[4], rows, external });
                return error.LongRegBytesDiffer;
            }
            cases += 1;
        };
    }
    return cases;
}

pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    if (!device.tensorUnits()) return error.TensorUnitsRequired;
    const queue = try device.queue();
    defer queue.deinit();
    const order = try lane.RegOrder.init(device);
    defer order.deinit();
    var prng = std.Random.DefaultPrng.init(0x7e9);
    const rng = prng.random();
    var cases: usize = 0;
    for ([_][2]usize{ .{ 96, 256 }, .{ 2048, 5120 }, .{ 5120, 6144 } }) |shape| for ([_]usize{ 1, 2, 4, 8 }) |sk| {
        const n = shape[0];
        const k = shape[1];
        if (k / 64 < sk) continue;
        const base = lane.Layout{ .n = n, .k = k, .format = .{ .bits = 4, .group = 64 }, .sk = sk, .precompute_sums = true };
        var coop = base;
        coop.cooperative = true;
        var reg = base;
        reg.reg = true;
        const raw = try a.alloc(u32, base.weightWords());
        for (raw) |*v| v.* = rng.int(u32);
        const scales = try a.alloc(u16, n * (k / 64));
        const biases = try a.alloc(u16, n * (k / 64));
        for (scales, biases) |*s, *b| {
            s.* = bf16(0.002 + rng.float(f32) * 0.02);
            b.* = bf16(-rng.float(f32) * 0.15);
        }
        const tiled = try device.buffer(base.weightWords() * 4, opts);
        defer tiled.deinit();
        const ordered = try device.buffer(base.weightWords() * 4, opts);
        defer ordered.deinit();
        const sb = try device.buffer(base.metadataElements() * 2, opts);
        defer sb.deinit();
        try lane.pack(base, raw, scales, biases, tiled.slice(u32, raw.len), sb.slice(u16, base.metadataElements()));
        @memcpy(ordered.slice(u32, raw.len), tiled.slice(u32, raw.len));
        try order.run(&.{.{ .w = .{ .buf = ordered }, .n = n, .k = k }});
        const x = try device.buffer(16 * k * 2, opts);
        defer x.deinit();
        const xs = x.slice(u16, 16 * k);
        for (xs) |*v| v.* = bf16((rng.float(f32) - 0.5) * 4);
        // producer sums at stride 32: each group's 64 values in order from zero, rows past the input zero
        const stride = 32;
        const given = try device.buffer((k / 64) * stride * 4, opts);
        defer given.deinit();
        const g = given.slice(f32, (k / 64) * stride);
        const want = try device.buffer(16 * n * 2, opts);
        defer want.deinit();
        const got = try device.buffer(16 * n * 2, opts);
        defer got.deinit();
        const p_coop = try lane.Projection.init(a, device, coop);
        defer p_coop.deinit();
        const p_reg = try lane.Projection.init(a, device, reg);
        defer p_reg.deinit();
        for (1..17) |rows| {
            @memset(g, 0);
            for (0..rows) |m| for (0..k / 64) |grp| {
                var acc: f32 = 0;
                for (0..64) |i| acc += @as(f32, @bitCast(@as(u32, xs[m * k + grp * 64 + i]) << 16));
                g[grp * stride + m] = acc;
            };
            try run(queue, p_coop, x, tiled, sb, null, want, @intCast(rows));
            for ([_]bool{ false, true }) |external| {
                @memset(got.contents()[0..got.length()], 0xa5);
                try run(queue, p_reg, x, ordered, sb, if (external) .{ .ref = .{ .buf = given }, .stride = stride } else null, got, @intCast(rows));
                if (!std.mem.eql(u8, got.contents()[0 .. rows * n * 2], want.contents()[0 .. rows * n * 2])) {
                    std.debug.print("DIFFER n={d} k={d} sk={d} rows={d} external={}\n", .{ n, k, sk, rows, external });
                    return error.RegBytesDiffer;
                }
                cases += 1;
            }
        }
    };
    std.debug.print("REG CHECK PASS cases={d}: every output byte equals the cooperative kernel's\n", .{cases});
    const long = try longCheck(a, device, queue, order, rng);
    std.debug.print("LONG CHECK PASS cases={d}: one dispatch of up to {d} rows equals {d}-row dispatches byte for byte\n", .{ long, lane.max_reg_rows, lane.reg_rows });
}
