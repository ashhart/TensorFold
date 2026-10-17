//! The DiT: condition, then Euler steps over windows (both CFG branches a forward) on cuBLASLt fp16 and our kernels.
const std = @import("std");
const cuda = @import("cuda");
const cfg = @import("config.zig");
const wts = @import("weights.zig");
const kern = @import("kernels.zig");

const Buf = cuda.DeviceBuffer;
const D = cfg.dit_width;

pub const Options = struct {
    window_batch: usize = 16, // windows a forward (each with both CFG branches)
    attn_group: usize = 8, // sequences a score buffer
    comfy_rope: bool = true, // rope angles rounded to fp16 before cos / sin, as ComfyUI computes them
    flash: bool = true, // the tensor-core flash attention kernel (else cuBLAS score GEMMs, a softmax, cuBLAS PV)
};

pub const Dit = struct {
    gpa: std.mem.Allocator,
    k: *kern.Kernels,
    w: *const wts.Dit,
    d: *const cuda.Driver,
    s: cuda.Stream,
    o: Options,

    fn i(x: usize) i32 {
        return @intCast(x);
    }

    const Scratch = struct {
        list: std.ArrayList(Buf) = .empty,
        gpa: std.mem.Allocator,
        d: *const cuda.Driver,
        fn mk(sc: *Scratch, bytes: usize) !u64 {
            const b = try Buf.alloc(sc.d, @max(bytes, 256));
            try sc.list.append(sc.gpa, b);
            return b.ptr;
        }
        fn deinit(sc: *Scratch) void {
            for (sc.list.items) |*b| b.free();
            sc.list.deinit(sc.gpa);
        }
    };

    fn view(dt: *Dit, p: u64) Buf {
        return .{ .d = dt.d, .ptr = p, .len = std.math.maxInt(usize) / 2, .borrowed = true };
    }

    /// The condition [L, 2048] fp16 (device) from frame states bf16 [T, 8, 4096] (host).
    pub fn condition(dt: *Dit, sc: *Scratch, states: []const u16, frames: usize, L: usize) !u64 {
        const k = dt.k;
        const s = dt.s;
        const st_d = try sc.mk(states.len * 2);
        try dt.view(st_d).upload(0, std.mem.sliceAsBytes(states));
        const w8 = try sc.mk(32);
        try dt.view(w8).upload(0, std.mem.sliceAsBytes(&dt.w.cond_w8));
        const mixed = try sc.mk(frames * 4096 * 2);
        var a: cuda.Args = .{};
        a.add(st_d);
        a.add(w8);
        a.add(dt.w.cond_scale);
        a.add(mixed);
        a.add(i(frames));
        try k.flat(.cond_mix, s, frames * 4096, &a);
        const col = try sc.mk(frames * 3 * 4096 * 2);
        var b: cuda.Args = .{};
        b.add(mixed);
        b.add(col);
        b.add(i(frames));
        b.add(i(4096));
        try k.flat(.im2col3, s, frames * 3 * 4096, &b);
        const conv = try sc.mk(frames * D * 2);
        try k.lt.linear(s, col, dt.w.cond_conv, dt.w.cond_bias, conv, frames, D, 3 * 4096, false);
        const idx = try cfg.nearest(dt.gpa, frames, L);
        defer dt.gpa.free(idx);
        const idx_d = try sc.mk(L * 4);
        try dt.view(idx_d).upload(0, std.mem.sliceAsBytes(idx));
        const cond = try sc.mk(L * D * 2);
        var c: cuda.Args = .{};
        c.add(conv);
        c.add(idx_d);
        c.add(cond);
        c.add(i(D));
        try k.go(.gather_rows_h, s, .{ L, 1, 1 }, 256, &c);
        return cond;
    }

    /// cos, sin [Lseq, 16, 2] (f32 values of fp16 numbers) for positions 0..Lseq (0 is the timestep token).
    fn ropeTable(dt: *Dit, sc: *Scratch, Lseq: usize) !u64 {
        const host = try dt.gpa.alloc(f32, Lseq * 32);
        defer dt.gpa.free(host);
        for (0..Lseq) |j| for (0..16) |q| {
            var ang: f32 = @as(f32, @floatFromInt(j)) * dt.w.inv_freq[q];
            if (dt.o.comfy_rope) ang = @floatCast(@as(f16, @floatCast(ang)));
            host[(j * 16 + q) * 2] = @floatCast(@as(f16, @floatCast(@cos(ang))));
            host[(j * 16 + q) * 2 + 1] = @floatCast(@as(f16, @floatCast(@sin(ang))));
        };
        const p = try sc.mk(host.len * 4);
        try dt.view(p).upload(0, std.mem.sliceAsBytes(host));
        return p;
    }

    pub const Progress = struct {
        ctx: *anyopaque,
        f: *const fn (ctx: *anyopaque, stage: []const u8, done: usize, total: usize) void,
    };

    pub const Probe = struct { cfg: f32, t: f32 }; // one forward at timestep t, returning u + (c - u) cfg

    /// Frame states (bf16 [frames, 32768]) to latents [L, 128]; `noise` or the seed's own; `probe`: one forward only.
    pub fn denoise(dt: *Dit, states: []const u16, frames: usize, steps: usize, dit_cfg: f32, seed: u64, noise: ?[]const f32, probe: ?Probe, progress: ?Progress) ![]f32 {
        const k = dt.k;
        const s = dt.s;
        const L = cfg.framesToLatents(frames);
        var sc: Scratch = .{ .gpa = dt.gpa, .d = dt.d };
        defer sc.deinit();
        const cond = try dt.condition(&sc, states, frames, L);
        const spans = try cfg.windows(dt.gpa, L);
        defer dt.gpa.free(spans);
        const x = try sc.mk(L * 128 * 4);
        if (noise) |nz| {
            if (nz.len != L * 128) return error.NoiseShape;
            try dt.view(x).upload(0, std.mem.sliceAsBytes(nz));
        } else {
            var a: cuda.Args = .{};
            a.add(x);
            a.add(seed);
            a.add(L * 128);
            try k.flat(.noise, s, L * 128 / 2 + 1, &a);
        }
        const vsum = try sc.mk(L * 128 * 4);
        const vcnt = try sc.mk(L * 128 * 4);
        for ([_]u64{ vsum, vcnt }) |p| {
            var b = dt.view(p);
            b.len = L * 128 * 4;
            try b.fill8(0, s.handle);
        }
        // per-batch buffers at the largest batch
        const nb_max = @min(dt.o.window_batch, spans.len);
        var lw_max: usize = 0;
        for (spans) |sp| lw_max = @max(lw_max, sp.len);
        const S_max = 2 * nb_max;
        const Lseq_max = lw_max + 1;
        const T_max = S_max * Lseq_max;
        const ld_max = (Lseq_max + 7) / 8 * 8;
        const full = try sc.mk(S_max * lw_max * cfg.dit_in * 2);
        const full2 = try sc.mk(S_max * lw_max * cfg.dit_in * 2);
        const h = try sc.mk(T_max * D * 2);
        const hn = try sc.mk(T_max * D * 2);
        const qkv = try sc.mk(T_max * 3 * D * 2);
        const qh = if (dt.o.flash) 0 else try sc.mk(T_max * D * 2);
        const kh = if (dt.o.flash) 0 else try sc.mk(T_max * D * 2);
        const vh = if (dt.o.flash) 0 else try sc.mk(T_max * D * 2);
        const oh = if (dt.o.flash) 0 else try sc.mk(T_max * D * 2);
        const att = try sc.mk(T_max * D * 2);
        const ffi = try sc.mk(T_max * 2 * cfg.dit_mlp * 2);
        const glu = try sc.mk(T_max * cfg.dit_mlp * 2);
        const G = dt.o.attn_group;
        const scores = if (dt.o.flash) 0 else try sc.mk(G * 32 * Lseq_max * ld_max * 4);
        const probs = if (dt.o.flash) 0 else try sc.mk(G * 32 * Lseq_max * ld_max * 2);
        const po = try sc.mk(S_max * lw_max * 128 * 2);
        const pp = try sc.mk(S_max * lw_max * 128 * 2);
        const tf = try sc.mk(256 * 2);
        const te1 = try sc.mk(D * 2);
        const te2 = try sc.mk(D * 2);
        const te = try sc.mk(D * 2);
        const wins_d = try sc.mk(2 * nb_max * 4);
        const lens_d = try sc.mk(S_max * 4);
        const rope_tabs = try dt.gpa.alloc(?u64, Lseq_max + 1);
        defer dt.gpa.free(rope_tabs);
        @memset(rope_tabs, null);
        const sig = try cfg.sigmas(dt.gpa, steps);
        defer dt.gpa.free(sig);
        const n_steps = if (probe != null) 1 else steps;
        for (0..n_steps) |step| {
            const sigma = sig[step];
            const t_val: f32 = if (probe) |p| p.t else 1.0 - sigma;
            const guide = if (probe) |p| p.cfg else dit_cfg;
            // the timestep embedding (the same for every window and branch)
            {
                var a: cuda.Args = .{};
                a.add(dt.w.time_features);
                a.add(t_val);
                a.add(tf);
                try k.go(.time_features, s, .{ 1, 1, 1 }, 128, &a);
                try dt.small(dt.w.time_in, dt.w.time_in_b, tf, 256, D, te1);
                var b: cuda.Args = .{};
                b.add(te1);
                b.add(te2);
                b.add(i(D));
                try k.go(.silu_h, s, .{ 1, 1, 1 }, 1024, &b);
                try dt.small(dt.w.time_out, dt.w.time_out_b, te2, D, D, te);
            }
            var b0: usize = 0;
            while (b0 < spans.len) : (b0 += nb_max) {
                const nb = @min(nb_max, spans.len - b0);
                const batch = spans[b0 .. b0 + nb];
                var lw: usize = 0;
                for (batch) |sp| lw = @max(lw, sp.len);
                const S = 2 * nb;
                const Lseq = lw + 1;
                const T = S * Lseq;
                const ld = (Lseq + 7) / 8 * 8;
                var wins: [64]i32 = undefined;
                var lens: [128]i32 = undefined;
                for (batch, 0..) |sp, wi| {
                    wins[2 * wi] = @intCast(sp.start);
                    wins[2 * wi + 1] = @intCast(sp.len);
                    lens[wi] = @intCast(sp.len + 1);
                    lens[nb + wi] = @intCast(sp.len + 1);
                }
                try dt.view(wins_d).uploadAsync(0, std.mem.sliceAsBytes(wins[0 .. 2 * nb]), s.handle);
                try dt.view(lens_d).uploadAsync(0, std.mem.sliceAsBytes(lens[0..S]), s.handle);
                if (rope_tabs[Lseq] == null) rope_tabs[Lseq] = try dt.ropeTable(&sc, Lseq);
                const cs = rope_tabs[Lseq].?;
                // inputs: [x, 0, condition] + preprocess(...), project_in into rows 1.. of each sequence
                {
                    var a: cuda.Args = .{};
                    a.add(x);
                    a.add(cond);
                    a.add(dt.w.cond_bias);
                    a.add(wins_d);
                    a.add(i(nb));
                    a.add(i(lw));
                    a.add(full);
                    try k.go(.dit_input, s, .{ lw, S, 1 }, 256, &a);
                }
                try k.lt.linear(s, full, dt.w.pre, 0, full2, S * lw, cfg.dit_in, cfg.dit_in, false);
                try dt.addH(full2, full, S * lw * cfg.dit_in);
                try k.hgemm(s, .t, .n, D, lw, cfg.dit_in, 1.0, dt.w.project_in, cfg.dit_in, 0, full2, cfg.dit_in, lw * cfg.dit_in, 0.0, h + D * 2, D, Lseq * D, S, .f16);
                {
                    var a: cuda.Args = .{};
                    a.add(te);
                    a.add(h);
                    a.add(i(Lseq));
                    try k.go(.put_time, s, .{ S, 1, 1 }, 256, &a);
                }
                for (&dt.w.blocks, 0..) |*B, bi| {
                    // the previous block's ff_out bias goes in with this block's pre-norm
                    try dt.layerNorm(h, if (bi > 0) dt.w.blocks[bi - 1].ff_out_b else 0, B.pre_g, B.pre_b, hn, T);
                    try k.lt.linear(s, hn, B.qkv, 0, qkv, T, 3 * D, D, false);
                    if (!dt.o.flash) {
                        var a: cuda.Args = .{};
                        a.add(qkv);
                        a.add(cs);
                        a.add(i(Lseq));
                        a.add(qh);
                        a.add(kh);
                        a.add(vh);
                        try k.go(.dit_rope, s, .{ Lseq, S, 1 }, 256, &a);
                    }
                    if (dt.o.flash) {
                        var a: cuda.Args = .{};
                        a.add(qkv);
                        a.add(cs);
                        a.add(i(Lseq));
                        a.add(lens_d);
                        a.add(@as(f32, 0.125));
                        a.add(att);
                        try k.go(.flash_attn, s, .{ (Lseq + 127) / 128, S * 32, 1 }, 256, &a);
                    } else {
                        var g0: usize = 0;
                        while (g0 < S) : (g0 += G) {
                            const g = @min(G, S - g0);
                            const off = g0 * 32 * Lseq * 64 * 2;
                            const bh = g * 32;
                            try k.hgemm(s, .t, .n, Lseq, Lseq, 64, 0.125, kh + off, 64, Lseq * 64, qh + off, 64, Lseq * 64, 0.0, scores, ld, Lseq * ld, bh, .f32);
                            var a: cuda.Args = .{};
                            a.add(scores);
                            a.add(i(ld));
                            a.add(i(Lseq));
                            a.add(lens_d);
                            a.add(i(g0));
                            a.add(probs);
                            try k.go(.softmax_rows, s, .{ Lseq, bh, 1 }, 256, &a);
                            try k.hgemm(s, .n, .n, 64, Lseq, Lseq, 1.0, vh + off, 64, Lseq * 64, probs, ld, Lseq * ld, 0.0, oh + off, 64, Lseq * 64, bh, .f16);
                        }
                        var a: cuda.Args = .{};
                        a.add(oh);
                        a.add(i(Lseq));
                        a.add(att);
                        try k.go(.heads_to_rows, s, .{ Lseq, S, 1 }, 256, &a);
                    }
                    try k.lt.linear(s, att, B.out, 0, h, T, D, D, true);
                    try dt.layerNorm(h, 0, B.ff_g, B.ff_b, hn, T);
                    try k.lt.linear(s, hn, B.ff_in, 0, ffi, T, 2 * cfg.dit_mlp, D, false);
                    {
                        var a: cuda.Args = .{};
                        a.add(ffi);
                        a.add(B.ff_in_b);
                        a.add(glu);
                        a.add(T);
                        try k.flat(.glu_h, s, T * 1024, &a);
                    }
                    try k.lt.linear(s, glu, B.ff_out, 0, h, T, D, cfg.dit_mlp, true);
                }
                {
                    var a: cuda.Args = .{};
                    a.add(h);
                    a.add(dt.w.blocks[cfg.dit_blocks - 1].ff_out_b);
                    a.add(T * D);
                    try k.flat(.bias_h, s, T * D, &a);
                }
                // project_out on rows 1.., the 1x1 postprocess, the guided velocity into the sums
                try k.hgemm(s, .t, .n, 128, lw, D, 1.0, dt.w.project_out, D, 0, h + D * 2, D, Lseq * D, 0.0, po, 128, lw * 128, S, .f16);
                try k.lt.linear(s, po, dt.w.post, 0, pp, S * lw, 128, 128, false);
                {
                    var a: cuda.Args = .{};
                    a.add(po);
                    a.add(pp);
                    a.add(wins_d);
                    a.add(i(nb));
                    a.add(i(lw));
                    a.add(guide);
                    a.add(i(L));
                    a.add(vsum);
                    a.add(vcnt);
                    try k.flat(.dit_gather, s, L * 128, &a);
                }
            }
            if (probe != null) {
                // the model output itself: vsum / vcnt, with no Euler step
                const out = try dt.gpa.alloc(f32, L * 128);
                errdefer dt.gpa.free(out);
                const cnt = try dt.gpa.alloc(f32, L * 128);
                defer dt.gpa.free(cnt);
                try dt.view(vsum).downloadAsync(0, std.mem.sliceAsBytes(out), s.handle);
                try dt.view(vcnt).downloadAsync(0, std.mem.sliceAsBytes(cnt), s.handle);
                try s.synchronize();
                for (out, cnt) |*o, c| o.* /= c;
                return out;
            }
            {
                var a: cuda.Args = .{};
                a.add(x);
                a.add(vsum);
                a.add(vcnt);
                a.add(sig[step + 1] - sigma);
                a.add(L * 128);
                try k.flat(.euler, s, L * 128, &a);
            }
            if (progress) |p| {
                try s.synchronize();
                p.f(p.ctx, "dit", step + 1, steps);
            }
        }
        const out = try dt.gpa.alloc(f32, L * 128);
        errdefer dt.gpa.free(out);
        try dt.view(x).downloadAsync(0, std.mem.sliceAsBytes(out), s.handle);
        try s.synchronize();
        return out;
    }

    fn small(dt: *Dit, w: u64, b: u64, x: u64, K: usize, N: usize, y: u64) !void {
        var a: cuda.Args = .{};
        a.add(w);
        a.add(b);
        a.add(x);
        a.add(i(K));
        a.add(i(N));
        a.add(y);
        try dt.k.go(.linear_h1, dt.s, .{ (N + 7) / 8, 1, 1 }, 256, &a);
    }

    fn addH(dt: *Dit, a_: u64, b_: u64, n: usize) !void {
        var a: cuda.Args = .{};
        a.add(a_);
        a.add(b_);
        a.add(n);
        try dt.k.flat(.add_h, dt.s, n, &a);
    }

    fn layerNorm(dt: *Dit, x: u64, bias: u64, g: u64, b: u64, y: u64, rows: usize) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(bias);
        a.add(g);
        a.add(b);
        a.add(y);
        try dt.k.go(.layernorm_h, dt.s, .{ rows, 1, 1 }, 256, &a);
    }
};
