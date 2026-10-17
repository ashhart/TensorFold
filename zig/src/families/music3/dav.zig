//! The DAV in fp32: latents [L, 128] to stereo 44.1 kHz; convs as tap SGEMMs on rows with zero halos, in chunks.
const std = @import("std");
const cuda = @import("cuda");
const wts = @import("weights.zig");
const kern = @import("kernels.zig");

const Buf = cuda.DeviceBuffer;
const halo_rows = 27; // the widest conv's padding: 3 x dilation 9

pub const Options = struct { chunk: usize = 512, context: usize = 32, level: bool = true, tf32: bool = false };

pub const Dav = struct {
    gpa: std.mem.Allocator,
    k: *kern.Kernels,
    w: *const wts.Dav,
    d: *const cuda.Driver,
    s: cuda.Stream,
    o: Options = .{},

    fn i(x: usize) i32 {
        return @intCast(x);
    }

    fn view(dv: *Dav, p: u64) Buf {
        return .{ .d = dv.d, .ptr = p, .len = std.math.maxInt(usize) / 2, .borrowed = true };
    }

    const Bufs = struct { lat: u64, x0: u64, p: u64, r: u64, y: u64, z: u64, out: u64, cap: usize, zcap: usize };

    /// y [T, Cout] = bias + sum_j W_j x[t + j dil - pad] with x padded (data at halo_rows rows in).
    fn conv(dv: *Dav, c: wts.Conv, xpad: u64, T: usize, dil: usize, y: u64) !void {
        const k = dv.k;
        const pad = (c.k - 1) / 2 * dil;
        var a: cuda.Args = .{};
        a.add(y);
        a.add(c.b);
        a.add(i(c.cout));
        a.add(T * c.cout);
        try k.flat(.bias_rows, dv.s, T * c.cout, &a);
        for (0..c.k) |j| {
            const row = halo_rows - pad + j * dil;
            try k.blas.gemm(dv.s, .t, .n, c.cout, T, c.cin, 1.0, c.w + j * c.cout * c.cin * 4, c.cin, 0, xpad + row * c.cin * 4, c.cin, 0, 1.0, y, c.cout, 0, 1, .f32, .f32);
        }
    }

    /// y = snake(x) into the padded buffer `pad` (C channels, T rows), its halos zeroed.
    fn snakeInto(dv: *Dav, x: u64, alpha: u64, T: usize, C: usize, pad: u64) !void {
        const k = dv.k;
        var z1: cuda.Args = .{};
        z1.add(pad);
        z1.add(halo_rows * C);
        try k.flat(.zero_rows, dv.s, halo_rows * C, &z1);
        var z2: cuda.Args = .{};
        z2.add(pad + (halo_rows + T) * C * 4);
        z2.add(halo_rows * C);
        try k.flat(.zero_rows, dv.s, halo_rows * C, &z2);
        var a: cuda.Args = .{};
        a.add(x);
        a.add(alpha);
        a.add(pad + halo_rows * C * 4);
        a.add(i(C));
        a.add(T * C);
        try k.flat(.snake, dv.s, T * C, &a);
    }

    fn data(p: u64, C: usize) u64 {
        return p + halo_rows * C * 4;
    }

    /// One stereo half's latent rows [a, a + T) -> audio [T 512] into b.out.
    fn decodeRows(dv: *Dav, b: Bufs, a0: usize, T0: usize, half: usize) !usize {
        const k = dv.k;
        const s = dv.s;
        const w = dv.w;
        {
            var a: cuda.Args = .{};
            a.add(b.lat);
            a.add(i(a0));
            a.add(i(T0));
            a.add(i(half));
            a.add(b.x0);
            try k.flat(.latent_half, s, T0 * 64, &a);
        }
        // in_proj (k 1) straight into the padded buffer, then conv0 (k 7)
        var T = T0;
        {
            const C = w.in_proj.cout;
            var z1: cuda.Args = .{};
            z1.add(b.p);
            z1.add(halo_rows * C);
            try k.flat(.zero_rows, s, halo_rows * C, &z1);
            var z2: cuda.Args = .{};
            z2.add(b.p + (halo_rows + T) * C * 4);
            z2.add(halo_rows * C);
            try k.flat(.zero_rows, s, halo_rows * C, &z2);
            const c = w.in_proj;
            var a: cuda.Args = .{};
            a.add(data(b.p, C));
            a.add(c.b);
            a.add(i(C));
            a.add(T * C);
            try k.flat(.bias_rows, s, T * C, &a);
            try k.blas.gemm(s, .t, .n, C, T, c.cin, 1.0, c.w, c.cin, 0, b.x0, c.cin, 0, 1.0, data(b.p, C), C, 0, 1, .f32, .f32);
        }
        try dv.conv(w.conv0, b.p, T, 1, b.r);
        var C = w.conv0.cout;
        for (w.blocks) |B| {
            // snake, the transposed conv (stride s, kernel 2s) as one GEMM into its taps, then the overlap-add
            try dv.snakeInto(b.r, B.alpha, T, C, b.p);
            const up = B.up;
            if (T * 2 * B.stride * up.cout > b.zcap) return error.DavBuffers;
            try k.blas.gemm(s, .t, .n, up.k * up.cout, T, up.cin, 1.0, up.w, up.cin, 0, data(b.p, C), up.cin, 0, 0.0, b.z, up.k * up.cout, 0, 1, .f32, .f32);
            const Tout = T * B.stride;
            {
                var a: cuda.Args = .{};
                a.add(b.z);
                a.add(up.b);
                a.add(i(T));
                a.add(i(B.stride));
                a.add(i((B.stride + 1) / 2));
                a.add(i(up.cout));
                a.add(i(Tout));
                a.add(b.r);
                try k.flat(.overlap_add, s, Tout * up.cout, &a);
            }
            T = Tout;
            C = up.cout;
            if ((T + 2 * halo_rows) * C > b.cap) return error.DavBuffers;
            for (B.res) |R| {
                try dv.snakeInto(b.r, R.a1, T, C, b.p);
                try dv.conv(R.c1, b.p, T, R.dilation, b.y);
                {
                    var a: cuda.Args = .{};
                    a.add(b.y);
                    a.add(R.a2);
                    a.add(b.y);
                    a.add(i(C));
                    a.add(T * C);
                    try k.flat(.snake, s, T * C, &a);
                }
                // r += c2(y) (k 1) + its bias
                try k.blas.gemm(s, .t, .n, C, T, C, 1.0, R.c2.w, C, 0, b.y, C, 0, 1.0, b.r, C, 0, 1, .f32, .f32);
                var a: cuda.Args = .{};
                a.add(b.r);
                a.add(R.c2.b);
                a.add(i(C));
                a.add(T * C);
                try k.flat(.add_bias, s, T * C, &a);
            }
        }
        try dv.snakeInto(b.r, w.alpha_out, T, C, b.p);
        try dv.conv(w.conv_out, b.p, T, 1, b.out);
        var a: cuda.Args = .{};
        a.add(b.out);
        a.add(T);
        try k.flat(.tanh, s, T, &a);
        return T;
    }

    /// Latents [L, 128] (host) -> stereo [2, 512 L] (host, f32), then ComfyUI's level rule (/ max(1, 5 std)).
    pub fn decode(dv: *Dav, latents: []const f32, L: usize) ![]f32 {
        const chunk = dv.o.chunk;
        const ctxr = dv.o.context;
        const rows_max = @min(L, chunk + 2 * ctxr);
        const t_max = rows_max * 512;
        var list: std.ArrayList(Buf) = .empty;
        defer {
            for (list.items) |*bf| bf.free();
            list.deinit(dv.gpa);
        }
        const P = struct {
            fn mk(l: *std.ArrayList(Buf), gpa: std.mem.Allocator, d: *const cuda.Driver, bytes: usize) !u64 {
                const bf = try Buf.alloc(d, @max(bytes, 256));
                try l.append(gpa, bf);
                return bf.ptr;
            }
        };
        // the largest T x C over the stages: 96 x 512 rows (equal to 192 x 256)
        const cap = (t_max + 2 * halo_rows) * 96 + (rows_max * 8 + 2 * halo_rows) * 1536;
        const zcap = t_max / 2 * 4 * 96;
        const b: Bufs = .{
            .lat = try P.mk(&list, dv.gpa, dv.d, L * 128 * 4),
            .x0 = try P.mk(&list, dv.gpa, dv.d, rows_max * 64 * 4),
            .p = try P.mk(&list, dv.gpa, dv.d, cap * 4),
            .r = try P.mk(&list, dv.gpa, dv.d, cap * 4),
            .y = try P.mk(&list, dv.gpa, dv.d, cap * 4),
            .z = try P.mk(&list, dv.gpa, dv.d, zcap * 4),
            .out = try P.mk(&list, dv.gpa, dv.d, t_max * 4),
            .cap = cap,
            .zcap = zcap,
        };
        try dv.view(b.lat).upload(0, std.mem.sliceAsBytes(latents));
        try dv.k.blas.tf32(dv.o.tf32);
        defer dv.k.blas.tf32(false) catch {};
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
                try dv.view(b.out + off * 4).downloadAsync(0, std.mem.sliceAsBytes(audio[half * L * 512 + start * 512 ..][0..n]), dv.s.handle);
            }
            try dv.s.synchronize();
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
