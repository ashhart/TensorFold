//! Multi-row-block prompt lanes against today's kernel at each width, split and chunk cut, byte for byte.
const std = @import("std");
const mtl = @import("metal");
const lane = @import("core_lane");
const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;

/// n, k, the 4-bit prompt split, the 6/8-bit loader split, tiles skipped each half, first GPU input group.
const shapes = [_][6]usize{
    .{ 34816, 5120, 1, 2, 0, 0 },  .{ 34816, 5120, 1, 2, 96, 0 }, .{ 5120, 17408, 2, 8, 0, 0 },
    .{ 5120, 17408, 2, 8, 0, 48 }, .{ 10240, 5120, 1, 4, 0, 0 },  .{ 12288, 5120, 2, 4, 0, 0 },
    .{ 6240, 5120, 4, 8, 0, 0 },   .{ 5120, 6144, 2, 8, 0, 0 },   .{ 2048, 5120, 8, 8, 0, 0 },
};
const row_counts = [_]u32{ 1, 15, 16, 17, 33, 64, 65, 129, 1065, 2048 };

fn bf16(v: f32) u16 {
    const b: u32 = @bitCast(v);
    return @intCast((b + 0x7fff + ((b >> 16) & 1)) >> 16);
}

/// `rows` rows through `p` in the cuts' encodes, one command buffer, as a prompt's chunks arrive.
fn cuts(q: mtl.Queue, p: lane.Projection, x: mtl.Buffer, w: mtl.Buffer, sb: mtl.Buffer, xs: ?lane.Sums, y: mtl.Buffer, rows: u32, cut: u32) !void {
    const cb = q.commandBuffer();
    const e = cb.compute(.serial);
    var first: u32 = 0;
    while (first < rows) {
        const n: u32 = if (first == 0 and cut > 0 and cut < rows) cut else rows - first;
        const xr = lane.Ref{ .buf = x, .off = @as(usize, first) * p.k * 2 };
        const yr = lane.Ref{ .buf = y, .off = @as(usize, first) * p.y_stride * p.output_bytes };
        if (xs) |s| {
            try p.encodeSums(e, xr, .{ .buf = w }, .{ .buf = sb }, .{ .ref = .{ .buf = s.ref.buf, .off = @as(usize, first) * 4 }, .stride = s.stride }, yr, n);
        } else try p.encode(e, xr, .{ .buf = w }, .{ .buf = sb }, yr, n);
        e.barrier();
        first += n;
    }
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure() != null) return error.GpuFailure;
}

/// A planted wrong sum address must differ, so a pass means the comparison can see a block's rows.
fn control(a: std.mem.Allocator, d: mtl.Device, q: mtl.Queue, l: lane.Layout, p: lane.Projection, x: mtl.Buffer, w: mtl.Buffer, sb: mtl.Buffer, xs: lane.Sums, want: mtl.Buffer, got: mtl.Buffer, bytes: usize) !void {
    const text = try lane.source(a, l);
    defer a.free(text);
    const bad = try std.mem.replaceOwned(u8, a, text, "XS[g * stride + base + mb]", "XS[g * stride + r0 + mb]");
    defer a.free(bad);
    if (std.mem.eql(u8, bad, text)) return error.ControlNotPlanted;
    const adapted = try lane.frags.source(d, a, bad);
    defer a.free(adapted);
    const lib = try mtl.Library.fromSource(d, adapted, mtl.CompileOptions.mlx());
    defer lib.deinit();
    const pipe = try mtl.Pipeline.init(d, lib, "tf_lane_block", false);
    defer pipe.deinit();
    var broken = p;
    broken.pipe = pipe;
    @memset(got.contents()[0..got.length()], 0xa5);
    try cuts(q, broken, x, w, sb, xs, got, 33, 0);
    if (std.mem.eql(u8, got.contents()[0..bytes], want.contents()[0..bytes])) return error.ControlMissed;
}

/// An M3 has no tensor units: every block layout is refused, so its prompt rows keep today's kernels.
pub fn refused(a: std.mem.Allocator, device: mtl.Device) !usize {
    var count: usize = 0;
    for ([_]u8{ 4, 6, 8 }) |bits| for ([_]u8{ 2, 4 }) |tm| {
        const l = lane.Layout{ .n = 64, .k = 256, .format = .{ .bits = bits, .group = 64 }, .precompute_sums = true, .reg = bits == 4, .row_blocks = bits != 4, .rows = lane.max_reg_rows, .tm = tm };
        try l.validate();
        if (lane.Projection.init(a, device, l)) |p| {
            p.deinit();
            return error.BlockLayoutAdmitted;
        } else |err| if (err != error.TensorUnitsRequired) return err;
        count += 1;
    };
    return count;
}

pub fn check(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, order: lane.RegOrder, rng: std.Random) !usize {
    var count: usize = 0;
    for ([_]u8{ 4, 6, 8 }) |bits| for (shapes) |shape| {
        const n = shape[0];
        const k = shape[1];
        const skip = shape[4];
        if (bits != 4 and (skip > 0 or shape[5] > 0)) continue;
        const sk = if (bits == 4) shape[2] else shape[3];
        const ng = k / 64;
        const half = n / 64;
        const ranges = [_][2]usize{ .{ skip, half - skip }, .{ half + skip, half - skip } };
        const cut: ?[2]usize = if (shape[5] > 0) .{ shape[5], ng - shape[5] } else null;
        const ob: usize = if (cut != null) 4 else 2;
        const today = lane.Layout{ .n = n, .k = k, .format = .{ .bits = bits, .group = 64 }, .sk = sk, .precompute_sums = bits != 6, .reg = bits == 4, .row_blocks = bits != 4, .rows = lane.max_reg_rows, .ranges = if (skip > 0) &ranges else &.{}, .groups = cut, .output = if (cut != null) .f32 else .bf16 };
        const p_today = try lane.Projection.init(a, device, today);
        defer p_today.deinit();
        const raw = try a.alloc(u32, today.sourceWords());
        defer a.free(raw);
        for (raw) |*v| v.* = rng.int(u32);
        const scales = try a.alloc(u16, n * ng);
        defer a.free(scales);
        const biases = try a.alloc(u16, n * ng);
        defer a.free(biases);
        for (scales, biases) |*s, *b| {
            s.* = bf16(0.002 + rng.float(f32) * 0.02);
            b.* = bf16(-0.01 - rng.float(f32) * 0.15);
        }
        const w = try device.buffer(today.weightWords() * 4, opts);
        defer w.deinit();
        const sb = try device.buffer(today.metadataElements() * 2, opts);
        defer sb.deinit();
        try lane.pack(today, raw, scales, biases, w.slice(u32, today.weightWords()), sb.slice(u16, today.metadataElements()));
        if (bits == 4) try order.run(&.{.{ .w = .{ .buf = w }, .n = n, .k = k }});
        const most = lane.max_reg_rows;
        const x = try device.buffer(most * k * 2, opts);
        defer x.deinit();
        const input = x.slice(u16, most * k);
        for (input) |*v| v.* = bf16((rng.float(f32) - 0.5) * 4);
        const given = try device.buffer(ng * most * 4, opts);
        defer given.deinit();
        const sums = given.slice(f32, ng * most);
        for (0..ng) |g| for (0..most) |m| {
            var acc: f32 = 0;
            for (0..64) |j| acc += @as(f32, @bitCast(@as(u32, input[m * k + g * 64 + j]) << 16));
            sums[g * most + m] = acc;
        };
        const xs = lane.Sums{ .ref = .{ .buf = given }, .stride = most };
        const want = try device.buffer(most * n * ob, opts);
        defer want.deinit();
        const got = try device.buffer(most * n * ob, opts);
        defer got.deinit();
        for ([_]u8{ 2, 4 }) |asked| {
            const tm = lane.blockTm(sk, asked);
            if (tm != asked) continue;
            var wide = today;
            wide.precompute_sums = true;
            wide.tm = tm;
            const p_wide = try lane.Projection.init(a, device, wide);
            defer p_wide.deinit();
            for (row_counts) |rows| {
                const bytes = rows * n * ob;
                @memset(want.contents()[0..want.length()], 0x5a);
                try cuts(queue, p_today, x, w, sb, if (bits == 6) null else xs, want, rows, 0);
                for ([_]bool{ false, true }) |external| for ([_]u32{ 0, rows / 2 + 1 }) |at| {
                    // skipped tiles stay unwritten on both sides, so they start equal
                    @memset(got.contents()[0..got.length()], if (skip > 0) 0x5a else 0xa5);
                    try cuts(queue, p_wide, x, w, sb, if (external) xs else null, got, rows, at);
                    if (!std.mem.eql(u8, got.contents()[0..bytes], want.contents()[0..bytes])) {
                        std.debug.print("BLOCK DIFFER bits={d} n={d} k={d} sk={d} tm={d} skip={d} cut={d} rows={d} external={} at={d}\n", .{ bits, n, k, sk, tm, skip, shape[5], rows, external, at });
                        return error.BlockBytesDiffer;
                    }
                    count += 1;
                };
                if (rows == 33 and skip == 0 and cut == null) try control(a, device, queue, wide, p_wide, x, w, sb, xs, want, got, bytes);
            }
        }
    };
    return count;
}
