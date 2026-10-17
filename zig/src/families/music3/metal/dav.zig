//! The DAV on Metal: latents [L, 128] to 44.1 kHz stereo; fp32 convs as tap GEMMs from bf16 halves, in chunks.
const std = @import("std");
const wts = @import("../weights.zig");
const gpu = @import("gpu.zig");
const k = @import("kernels.zig");

const Gpu = gpu.Gpu;
const halo_rows = 27; // the widest conv's padding: 3 x dilation 9

pub const Options = struct { chunk: usize = 512, context: usize = 32, level: bool = true };

fn u(x: usize) u32 {
    return @intCast(x);
}

/// Every conv's taps as bf16 halves, hi then lo, in place of its fp32 taps (once, at load).
pub fn splitWeights(g: *Gpu, s: anytype, d: *wts.Dav) !void {
    var convs: std.ArrayList(*wts.Conv) = .empty;
    defer convs.deinit(g.gpa);
    try convs.appendSlice(g.gpa, &.{ &d.in_proj, &d.conv0, &d.conv_out });
    for (&d.blocks) |*B| {
        try convs.append(g.gpa, &B.up);
        for (&B.res) |*R| try convs.appendSlice(g.gpa, &.{ &R.c1, &R.c2 });
    }
    for (convs.items) |c| {
        const n = c.k * c.cout * c.cin;
        const packed_w = try s.alloc(n * 4);
        k.split(g, c.w, packed_w, packed_w + n * 2, n);
        try g.flush();
        g.free(c.w);
        s.bytes -= n * 4;
        c.w = packed_w;
    }
}

pub const Dav = struct {
    gpa: std.mem.Allocator,
    g: *Gpu,
    w: *const wts.Dav,
    o: Options = .{},

    const Bufs = struct { lat: u64, x0: u64, p: u64, r: u64, y: u64, z: u64, out: u64, sh: u64, sl: u64, cap: usize, zcap: usize };

    fn bias(dv: *Dav, y: u64, b: u64, C: usize, n: usize, set: bool) void {
        dv.g.flat(.m3_bias_rows, n, &.{ y, b }, [3]u32{ u(C), u(n), @intFromBool(set) });
    }

    /// The halves of x [n] into the split scratch, for the GEMMs after it.
    fn halves(dv: *Dav, b: Bufs, x: u64, n: usize) !void {
        if (n > b.cap) return error.DavBuffers;
        k.split(dv.g, x, b.sh, b.sl, n);
    }

    /// y [T, Cout] (+)= x [T, Cin] W_j^T: x at element `xe` of the split scratch, W_j tap j of conv c (split at load).
    fn gemm(dv: *Dav, b: Bufs, xe: usize, c: wts.Conv, j: usize, y: u64, T: usize, cout: usize, accumulate: bool) void {
        const n = c.k * c.cout * c.cin;
        const we = j * c.cout * c.cin;
        k.linearSplit(dv.g, b.sh + xe * 2, b.sl + xe * 2, c.w + we * 2, c.w + (n + we) * 2, y, .{ .m = T, .n = cout, .k = c.cin, .accumulate = accumulate });
    }

    /// y [T, Cout] = bias + sum_j W_j x[t + j dil - pad] with x padded (data at halo_rows rows in).
    fn conv(dv: *Dav, b: Bufs, c: wts.Conv, xpad: u64, T: usize, dil: usize, y: u64) !void {
        const pad = (c.k - 1) / 2 * dil;
        try dv.halves(b, xpad, (T + 2 * halo_rows) * c.cin);
        dv.bias(y, c.b, c.cout, T * c.cout, true);
        for (0..c.k) |j| dv.gemm(b, (halo_rows - pad + j * dil) * c.cin, c, j, y, T, c.cout, true);
    }

    fn zeroHalos(dv: *Dav, pad: u64, T: usize, C: usize) void {
        dv.g.flat(.m3_zero, halo_rows * C, &.{pad}, u(halo_rows * C));
        dv.g.flat(.m3_zero, halo_rows * C, &.{pad + (halo_rows + T) * C * 4}, u(halo_rows * C));
    }

    /// y = snake(x) into the padded buffer `pad` (C channels, T rows), its halos zeroed.
    fn snakeInto(dv: *Dav, x: u64, alpha: u64, T: usize, C: usize, pad: u64) void {
        dv.zeroHalos(pad, T, C);
        dv.g.flat(.m3_snake, T * C, &.{ x, alpha, data(pad, C) }, [2]u32{ u(C), u(T * C) });
    }

    fn data(p: u64, C: usize) u64 {
        return p + halo_rows * C * 4;
    }

    /// One stereo half's latent rows [a, a + T) -> audio [T 512] into b.out.
    fn decodeRows(dv: *Dav, b: Bufs, a0: usize, T0: usize, half: usize) !usize {
        const g = dv.g;
        const w = dv.w;
        g.flat(.m3_latent_half, T0 * 64, &.{ b.lat, b.x0 }, [3]u32{ u(a0), u(T0), u(half) });
        // in_proj (k 1) straight into the padded buffer, then conv0 (k 7)
        var T = T0;
        {
            const C = w.in_proj.cout;
            dv.zeroHalos(b.p, T, C);
            dv.bias(data(b.p, C), w.in_proj.b, C, T * C, true);
            try dv.halves(b, b.x0, T * w.in_proj.cin);
            dv.gemm(b, 0, w.in_proj, 0, data(b.p, C), T, C, true);
        }
        try dv.conv(b, w.conv0, b.p, T, 1, b.r);
        var C = w.conv0.cout;
        for (w.blocks) |B| {
            // snake, the transposed conv (stride s, kernel 2s) as one GEMM into its taps, then the overlap-add
            dv.snakeInto(b.r, B.alpha, T, C, b.p);
            const up = B.up;
            if (T * 2 * B.stride * up.cout > b.zcap) return error.DavBuffers;
            try dv.halves(b, data(b.p, C), T * up.cin);
            dv.gemm(b, 0, up, 0, b.z, T, up.k * up.cout, false);
            const Tout = T * B.stride;
            const OA = extern struct { Tin: u32, stride: u32, pad: u32, C: u32, Tout: u32 };
            g.flat(.m3_overlap_add, Tout * up.cout, &.{ b.z, up.b, b.r }, OA{ .Tin = u(T), .stride = u(B.stride), .pad = u((B.stride + 1) / 2), .C = u(up.cout), .Tout = u(Tout) });
            T = Tout;
            C = up.cout;
            if ((T + 2 * halo_rows) * C > b.cap) return error.DavBuffers;
            for (B.res) |R| {
                dv.snakeInto(b.r, R.a1, T, C, b.p);
                try dv.conv(b, R.c1, b.p, T, R.dilation, b.y);
                g.flat(.m3_snake, T * C, &.{ b.y, R.a2, b.y }, [2]u32{ u(C), u(T * C) });
                // r += c2(y) (k 1) + its bias
                try dv.halves(b, b.y, T * C);
                dv.gemm(b, 0, R.c2, 0, b.r, T, C, true);
                dv.bias(b.r, R.c2.b, C, T * C, false);
            }
            try g.commit();
        }
        dv.snakeInto(b.r, w.alpha_out, T, C, b.p);
        try dv.conv(b, w.conv_out, b.p, T, 1, b.out);
        g.flat(.m3_tanh, T, &.{b.out}, u(T));
        return T;
    }

    /// Latents [L, 128] (host) -> stereo [2, 512 L] (host, f32), then ComfyUI's level rule (/ max(1, 5 std)).
    pub fn decode(dv: *Dav, latents: []const f32, L: usize) ![]f32 {
        const g = dv.g;
        const chunk = dv.o.chunk;
        const ctxr = dv.o.context;
        const rows_max = @min(L, chunk + 2 * ctxr);
        const t_max = rows_max * 512;
        // the largest T x C over the stages: 96 x 512 rows (equal to 192 x 256)
        const cap = (t_max + 2 * halo_rows) * 96 + (rows_max * 8 + 2 * halo_rows) * 1536;
        const zcap = t_max / 2 * 4 * 96;
        var list: [9]u64 = @splat(0);
        defer {
            g.flush() catch {};
            for (list) |p| g.free(p);
        }
        const sizes = [_]usize{ L * 128 * 4, rows_max * 64 * 4, cap * 4, cap * 4, cap * 4, zcap * 4, t_max * 4, cap * 2, cap * 2 };
        for (&list, sizes) |*p, sz| p.* = try g.alloc(sz);
        const b: Bufs = .{ .lat = list[0], .x0 = list[1], .p = list[2], .r = list[3], .y = list[4], .z = list[5], .out = list[6], .sh = list[7], .sl = list[8], .cap = cap, .zcap = zcap };
        @memcpy(g.host(b.lat, latents.len * 4), std.mem.sliceAsBytes(latents));
        const audio = try dv.gpa.alloc(f32, 2 * L * 512);
        errdefer dv.gpa.free(audio);
        var start: usize = 0;
        while (start < L) : (start += chunk) {
            const end = @min(start + chunk, L);
            const a0 = start -| ctxr;
            const a1 = @min(L, end + ctxr);
            for (0..2) |half| {
                const T = try dv.decodeRows(b, a0, a1 - a0, half);
                std.debug.assert(T == (a1 - a0) * 512);
                const off = (start - a0) * 512;
                const n = (end - start) * 512;
                try g.get(b.out + off * 4, std.mem.sliceAsBytes(audio[half * L * 512 + start * 512 ..][0..n]));
            }
        }
        // level: std over both channels (unbiased), divide by max(1, 5 std)
        var mean: f64 = 0;
        for (audio) |v| mean += v;
        mean /= @floatFromInt(audio.len);
        var ss: f64 = 0;
        for (audio) |v| {
            const dd = @as(f64, v) - mean;
            ss += dd * dd;
        }
        const sd = @sqrt(ss / @as(f64, @floatFromInt(audio.len - 1)));
        const level = @max(1.0, sd * 5.0);
        if (dv.o.level and level > 1.0) for (audio) |*v| {
            v.* = @floatCast(@as(f64, v.*) / level);
        };
        return audio;
    }
};
