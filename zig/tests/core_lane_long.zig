//! Whole prompt lanes at each width against a frozen 16-row kernel, including strided oQ member formats.
const std = @import("std");
const mtl = @import("metal");
const lane = @import("core_lane");
const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
const original = @embedFile("fixtures/lane_prompt/main_fb8fdf3_lane.metal");

fn bf16(v: f32) u16 {
    const b: u32 = @bitCast(v);
    return @intCast((b + 0x7fff + ((b >> 16) & 1)) >> 16);
}

fn pipe(device: mtl.Device, text: []const u8) !mtl.Pipeline {
    const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
    defer lib.deinit();
    return mtl.Pipeline.init(device, lib, "tf_lane", false);
}

fn frozen(a: std.mem.Allocator, d: mtl.Device, l: lane.Layout) !mtl.Pipeline {
    const current = try lane.source(a, l);
    defer a.free(current);
    const end = std.mem.indexOf(u8, current, "#include <metal_stdlib>") orelse return error.SourceHeader;
    const old = try std.mem.concat(a, u8, &.{ current[0..end], original });
    defer a.free(old);
    return pipe(d, old);
}

fn chunks(q: mtl.Queue, p: lane.Projection, x: mtl.Buffer, w: mtl.Buffer, sb: mtl.Buffer, xs: ?lane.Sums, y: mtl.Buffer, rows: u32) !void {
    const cb = q.commandBuffer();
    const e = cb.compute(.serial);
    var first: u32 = 0;
    while (first < rows) : (first += p.max_rows) {
        const xr = lane.Ref{ .buf = x, .off = @as(usize, first) * p.k * 2 };
        const yr = lane.Ref{ .buf = y, .off = @as(usize, first) * p.y_stride * 2 };
        const n = @min(rows - first, p.max_rows);
        if (xs) |s| {
            try p.encodeSums(e, xr, .{ .buf = w }, .{ .buf = sb }, .{ .ref = .{ .buf = s.ref.buf, .off = first * 4 }, .stride = s.stride }, yr, n);
        } else try p.encode(e, xr, .{ .buf = w }, .{ .buf = sb }, yr, n);
    }
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure() != null) return error.GpuFailure;
}

fn clear(y: mtl.Buffer, rows: usize, n: usize, stride: usize, candidate: bool) void {
    @memset(y.contents()[0..y.length()], 0x5a);
    if (candidate) for (0..rows) |m| @memset(y.contents()[m * stride * 2 ..][0 .. n * 2], 0xa5);
}

fn same(a: mtl.Buffer, b: mtl.Buffer) bool {
    return std.mem.eql(u8, a.contents()[0..a.length()], b.contents()[0..b.length()]);
}

fn controls(a: std.mem.Allocator, d: mtl.Device, q: mtl.Queue, l: lane.Layout, p: lane.Projection, x: mtl.Buffer, w: mtl.Buffer, sb: mtl.Buffer, sums: lane.Sums, want: mtl.Buffer, got: mtl.Buffer) !void {
    const text = try lane.source(a, l);
    defer a.free(text);
    const from = [_][]const u8{ "int(tg.y) * 16", "XS += row0;", "XS[(g) * mdims[1] + (row)]" };
    const to = [_][]const u8{ "0", "", "XS[(g) * 16 + (row)]" };
    for (from, to, 0..) |needle, replacement, i| {
        const bad = try std.mem.replaceOwned(u8, a, text, needle, replacement);
        defer a.free(bad);
        if (std.mem.eql(u8, bad, text)) return error.ControlNotPlanted;
        const bad_pipe = try pipe(d, bad);
        defer bad_pipe.deinit();
        var broken = p;
        broken.pipe = bad_pipe;
        clear(got, 17, l.n, l.y_stride.?, true);
        try chunks(q, broken, x, w, sb, sums, got, 17);
        if (same(want, got)) return error.ControlMissed;
        std.debug.print("LANE LONG CONTROL {d} caught\n", .{i});
    }
}

pub fn check(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, rng: std.Random) !usize {
    var count: usize = 0;
    const formats = [_]lane.Format{
        .{ .bits = 8, .group = 64 }, .{ .bits = 6, .group = 32 }, .{ .bits = 5, .group = 64 },
        .{ .bits = 3, .group = 32 }, .{ .bits = 2, .group = 64 }, .{ .bits = 8, .group = 128 },
        .{ .bits = 5, .group = 64 }, // the generic precomputed-sums layout, beside the normal inline-sums one
    };
    for (formats, 0..) |format, fi| {
        // Non-reg entries retain splitFor's loader split; only reg entries use promptSplit's smaller values.
        for ([_][3]usize{ .{ 34816, 5120, 2 }, .{ 5120, 17408, 8 }, .{ 10240, 5120, 4 }, .{ 12288, 5120, 4 }, .{ 6240, 5120, 8 }, .{ 5120, 6144, 8 }, .{ 2048, 5120, 8 }, .{ 48, 512, 2 } }, 0..) |shape, si| {
            const n = shape[0];
            const k = shape[1];
            const stride = n + @as(usize, if (si == 7) 32 else 0);
            const most = lane.max_reg_rows;
            const sum_stride = most + 16;
            const ng = k / format.group;
            // Group128 SK8 exceeds the unchanged admission bound; compare its supported SK4 variant on both sides.
            const sk = if (format.group == 128 and shape[2] == 8) 4 else shape[2];
            const layout = lane.Layout{ .n = n, .k = k, .format = format, .sk = sk, .rows = most, .row_blocks = true, .precompute_sums = fi == 0 or fi == 6, .y_stride = stride };
            const p_long = try lane.Projection.init(a, device, layout);
            defer p_long.deinit();
            var short = layout;
            short.rows = 16;
            const p_short = try lane.Projection.init(a, device, short);
            defer p_short.deinit();
            const old_pipe = try frozen(a, device, short);
            defer old_pipe.deinit();
            var p_old = p_short;
            p_old.pipe = old_pipe;
            const raw = try a.alloc(u32, layout.sourceWords());
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
            const w = try device.buffer(layout.weightWords() * 4, opts);
            defer w.deinit();
            const sb = try device.buffer(layout.metadataElements() * 2, opts);
            defer sb.deinit();
            try lane.pack(layout, raw, scales, biases, w.slice(u32, layout.weightWords()), sb.slice(u16, layout.metadataElements()));
            const x = try device.buffer(most * k * 2, opts);
            defer x.deinit();
            const input = x.slice(u16, most * k);
            for (input) |*v| v.* = bf16((rng.float(f32) - 0.5) * 4);
            const given = try device.buffer(ng * sum_stride * 4, opts);
            defer given.deinit();
            const sums = given.slice(f32, ng * sum_stride);
            @memset(sums, -999);
            for (0..ng) |g| for (0..most) |m| {
                var acc: f32 = 0;
                for (0..format.group) |j| acc += @as(f32, @bitCast(@as(u32, input[m * k + g * format.group + j]) << 16));
                sums[g * sum_stride + m] = acc;
            };
            const want = try device.buffer(most * stride * 2, opts);
            defer want.deinit();
            const got = try device.buffer(most * stride * 2, opts);
            defer got.deinit();
            for ([_]u32{ 16, 17, 1065, most }) |rows| for ([_]bool{ false, true }) |external| {
                if (external and fi != 0) continue;
                const xs: ?lane.Sums = if (external) .{ .ref = .{ .buf = given }, .stride = sum_stride } else null;
                clear(want, rows, n, stride, false);
                clear(got, rows, n, stride, true);
                try chunks(queue, p_old, x, w, sb, xs, want, rows);
                try chunks(queue, p_short, x, w, sb, xs, got, rows);
                if (!same(want, got)) return error.ShortLaneBytesDiffer;
                clear(got, rows, n, stride, true);
                try chunks(queue, p_long, x, w, sb, xs, got, rows);
                if (!same(want, got)) {
                    std.debug.print("LANE LONG DIFFER bits={d} group={d} n={d} k={d} sk={d} rows={d} external={}\n", .{ format.bits, format.group, n, k, sk, rows, external });
                    return error.LongLaneBytesDiffer;
                }
                if (fi == 0 and si == 0 and rows == 17 and external) try controls(a, device, queue, layout, p_long, x, w, sb, xs.?, want, got);
                count += 1;
            };
        }
    }
    std.debug.print("LANE SHORT CHECK PASS cases={d}: current short variant equals frozen main\n", .{count});
    return count;
}
