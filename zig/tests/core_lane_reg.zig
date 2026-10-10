//! The reg kernel against the cooperative one: every output byte, 1-16 rows, every K split, own or producer sums.
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
}
