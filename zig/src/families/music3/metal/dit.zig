//! The DiT on Metal: condition, then Euler steps over windows (both CFG branches a forward), fp16 on the tensor units.
const std = @import("std");
const cfg = @import("../config.zig");
const wts = @import("../weights.zig");
const gpu = @import("gpu.zig");
const k = @import("kernels.zig");

const Gpu = gpu.Gpu;
const D = cfg.dit_width;

pub const Options = struct {
    window_batch: usize = 16, // windows a forward (each with both CFG branches)
    comfy_rope: bool = true, // rope angles rounded to fp16 before cos / sin, as ComfyUI computes them
    flash: bool = true, // CUDA's choice of attention kernel; Metal has one (m3_dit_attn)
};

pub const Progress = @import("../common.zig").Progress;

pub const Probe = struct { cfg: f32, t: f32 }; // one forward at timestep t, returning u + (c - u) cfg

fn u(x: usize) u32 {
    return @intCast(x);
}

/// Device buffers for one denoise, freed together once the GPU is done.
const Scratch = struct {
    g: *Gpu,
    list: std.ArrayList(u64) = .empty,
    gpa: std.mem.Allocator,
    fn mk(sc: *Scratch, bytes: usize) !u64 {
        const p = try sc.g.alloc(bytes);
        try sc.list.append(sc.gpa, p);
        return p;
    }
    fn up(sc: *Scratch, bytes: []const u8) !u64 {
        const p = try sc.mk(bytes.len);
        @memcpy(sc.g.host(p, bytes.len), bytes);
        return p;
    }
    fn deinit(sc: *Scratch) void {
        sc.g.flush() catch {};
        for (sc.list.items) |p| sc.g.free(p);
        sc.list.deinit(sc.gpa);
    }
};

/// ff_in's rows in 32s, 16 value rows then their 16 gate rows, for m3_linear_glu_h (once, at load, in place).
pub fn pairGlu(gpa: std.mem.Allocator, g: *Gpu, d: *wts.Dit) !void {
    const half = cfg.dit_mlp;
    const row = D * 2;
    const tmp = try gpa.alloc(u8, 2 * half * row);
    defer gpa.free(tmp);
    for (&d.blocks) |*B| {
        const w = g.host(B.ff_in, 2 * half * row);
        @memcpy(tmp, w);
        for (0..half) |p| {
            const at = (p / 16) * 32 + p % 16;
            @memcpy(w[at * row ..][0..row], tmp[p * row ..][0..row]);
            @memcpy(w[(at + 16) * row ..][0..row], tmp[(half + p) * row ..][0..row]);
        }
    }
}

pub const Dit = struct {
    gpa: std.mem.Allocator,
    g: *Gpu,
    w: *const wts.Dit,
    o: Options,

    /// The condition [L, 2048] fp16 (device) from frame states bf16 [T, 8, 4096] (host).
    fn condition(dt: *Dit, sc: *Scratch, states: []const u16, frames: usize, L: usize) !u64 {
        const g = dt.g;
        const st_d = try sc.up(std.mem.sliceAsBytes(states));
        const w8 = try sc.up(std.mem.sliceAsBytes(&dt.w.cond_w8));
        const mixed = try sc.mk(frames * 4096 * 2);
        g.flat(.m3_cond_mix, frames * 4096, &.{ st_d, w8, mixed }, [2]f32{ dt.w.cond_scale, @floatFromInt(frames) });
        const col = try sc.mk(frames * 3 * 4096 * 2);
        g.flat(.m3_im2col3, frames * 3 * 4096, &.{ mixed, col }, u(frames));
        const conv = try sc.mk(frames * D * 2);
        k.linear(g, .m3_linear_h, col, dt.w.cond_conv, dt.w.cond_bias, conv, .{ .m = frames, .n = D, .k = 3 * 4096 });
        const idx = try cfg.nearest(dt.gpa, frames, L);
        defer dt.gpa.free(idx);
        const idx_d = try sc.up(std.mem.sliceAsBytes(idx));
        const cond = try sc.mk(L * D * 2);
        g.go(.m3_gather_rows_h, .{ L, 1, 1 }, 256, &.{ conv, idx_d, cond }, {});
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
        return sc.up(std.mem.sliceAsBytes(host));
    }

    /// Frame states (bf16 [frames, 32768]) to latents [L, 128]; `noise` or the seed's own; `probe`: one forward only.
    pub fn denoise(dt: *Dit, states: []const u16, frames: usize, steps: usize, dit_cfg: f32, seed: u64, noise: ?[]const f32, probe: ?Probe, progress: ?Progress) ![]f32 {
        const g = dt.g;
        const L = cfg.framesToLatents(frames);
        var sc: Scratch = .{ .g = g, .gpa = dt.gpa };
        defer sc.deinit();
        const cond = try dt.condition(&sc, states, frames, L);
        const spans = try cfg.windows(dt.gpa, L);
        defer dt.gpa.free(spans);
        const x = try sc.mk(L * 128 * 4);
        if (noise) |nz| {
            if (nz.len != L * 128) return error.NoiseShape;
            @memcpy(g.host(x, nz.len * 4), std.mem.sliceAsBytes(nz));
        } else {
            g.flat(.m3_noise, L * 128 / 2 + 1, &.{x}, [2]u64{ seed, L * 128 });
        }
        const vsum = try sc.mk(L * 128 * 4);
        const vcnt = try sc.mk(L * 128 * 4);
        const nb_max = @min(dt.o.window_batch, spans.len);
        var lw_max: usize = 0;
        for (spans) |sp| lw_max = @max(lw_max, sp.len);
        const S_max = 2 * nb_max;
        const Lseq_max = lw_max + 1;
        const T_max = S_max * Lseq_max;
        const full = try sc.mk(S_max * lw_max * cfg.dit_in * 2);
        const full2 = try sc.mk(S_max * lw_max * cfg.dit_in * 2);
        const h = try sc.mk(T_max * D * 2);
        const hn = try sc.mk(T_max * D * 2);
        const qkv = try sc.mk(T_max * 3 * D * 2);
        const att = try sc.mk(T_max * D * 2);
        const glu = try sc.mk(T_max * cfg.dit_mlp * 2);
        const po = try sc.mk(S_max * lw_max * 128 * 2);
        const pp = try sc.mk(S_max * lw_max * 128 * 2);
        const tf = try sc.mk(256 * 2);
        const te1 = try sc.mk(D * 2);
        const te2 = try sc.mk(D * 2);
        const te = try sc.mk(D * 2);
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
            g.go(.m3_time_features, .{ 1, 1, 1 }, 128, &.{ dt.w.time_features, tf }, t_val);
            dt.small(dt.w.time_in, dt.w.time_in_b, tf, 256, D, te1);
            g.go(.m3_silu_h, .{ 1, 1, 1 }, 1024, &.{ te1, te2 }, u(D));
            dt.small(dt.w.time_out, dt.w.time_out_b, te2, D, D, te);
            var b0: usize = 0;
            while (b0 < spans.len) : (b0 += nb_max) {
                const nb = @min(nb_max, spans.len - b0);
                const batch = spans[b0 .. b0 + nb];
                var lw: usize = 0;
                for (batch) |sp| lw = @max(lw, sp.len);
                const S = 2 * nb;
                const Lseq = lw + 1;
                const T = S * Lseq;
                var wins: [64]i32 = undefined;
                var lens: [128]i32 = undefined;
                for (batch, 0..) |sp, wi| {
                    wins[2 * wi] = @intCast(sp.start);
                    wins[2 * wi + 1] = @intCast(sp.len);
                    lens[wi] = @intCast(sp.len + 1);
                    lens[nb + wi] = @intCast(sp.len + 1);
                }
                const wins_d = try sc.up(std.mem.sliceAsBytes(wins[0 .. 2 * nb]));
                const lens_d = try sc.up(std.mem.sliceAsBytes(lens[0..S]));
                if (rope_tabs[Lseq] == null) rope_tabs[Lseq] = try dt.ropeTable(&sc, Lseq);
                const cs = rope_tabs[Lseq].?;
                // inputs: [x, 0, condition] + preprocess(...), project_in into rows 1.. of each sequence
                g.go(.m3_dit_input, .{ lw, S, 1 }, 256, &.{ x, cond, dt.w.cond_bias, wins_d, full }, [2]u32{ u(nb), u(lw) });
                k.linear(g, .m3_linear_h, full, dt.w.pre, 0, full2, .{ .m = S * lw, .n = cfg.dit_in, .k = cfg.dit_in });
                g.flat(.m3_add_h, S * lw * cfg.dit_in, &.{ full2, full }, u(S * lw * cfg.dit_in));
                k.linear(g, .m3_linear_h, full2, dt.w.project_in, 0, h + D * 2, .{ .m = lw, .n = D, .k = cfg.dit_in, .ldy = D, .batch = S, .sx = lw * cfg.dit_in, .sy = Lseq * D });
                g.go(.m3_put_time, .{ S, 1, 1 }, 256, &.{ te, h }, u(Lseq));
                for (&dt.w.blocks, 0..) |*B, bi| {
                    // the previous block's ff_out bias goes in with this block's pre-norm
                    dt.layerNorm(h, if (bi > 0) dt.w.blocks[bi - 1].ff_out_b else 0, B.pre_g, B.pre_b, hn, T);
                    // q and k leave the GEMM roped, rounded to fp16 first as the reference rounds them
                    k.linear(g, .m3_linear_rope_h, hn, B.qkv, 0, qkv, .{ .m = T, .n = 3 * D, .k = D, .cs = cs, .lseq = Lseq });
                    const A = extern struct { Lseq: u32, scale: f32 };
                    g.go(.m3_dit_attn, .{ (Lseq + 63) / 64, 32, S }, 128, &.{ qkv, lens_d, att }, A{ .Lseq = u(Lseq), .scale = 0.125 });
                    k.linear(g, .m3_linear_h, att, B.out, 0, h, .{ .m = T, .n = D, .k = D, .accumulate = true });
                    dt.layerNorm(h, 0, B.ff_g, B.ff_b, hn, T);
                    // ff_in's rows paired at load (dit.pairGlu): the GEMM writes the GLU of value and gate
                    k.linear(g, .m3_linear_glu_h, hn, B.ff_in, B.ff_in_b, glu, .{ .m = T, .n = 2 * cfg.dit_mlp, .k = D, .ldy = cfg.dit_mlp });
                    k.linear(g, .m3_linear_h, glu, B.ff_out, 0, h, .{ .m = T, .n = D, .k = cfg.dit_mlp, .accumulate = true });
                    try g.commit();
                }
                g.flat(.m3_bias_h, T * D, &.{ h, dt.w.blocks[cfg.dit_blocks - 1].ff_out_b }, u(T * D));
                // project_out on rows 1.., the 1x1 postprocess, the guided velocity into the sums
                k.linear(g, .m3_linear_h, h + D * 2, dt.w.project_out, 0, po, .{ .m = lw, .n = 128, .k = D, .ldx = D, .batch = S, .sx = Lseq * D, .sy = lw * 128 });
                k.linear(g, .m3_linear_h, po, dt.w.post, 0, pp, .{ .m = S * lw, .n = 128, .k = 128 });
                const GA = extern struct { nwin: u32, Lw: u32, L: u32, cfg: f32 };
                g.flat(.m3_dit_gather, L * 128, &.{ po, pp, wins_d, vsum, vcnt }, GA{ .nwin = u(nb), .Lw = u(lw), .L = u(L), .cfg = guide });
            }
            if (probe != null) {
                // the model output itself: vsum / vcnt, with no Euler step
                try g.flush();
                const out = try dt.gpa.alloc(f32, L * 128);
                const s_ = std.mem.bytesAsSlice(f32, g.host(vsum, L * 128 * 4));
                const c_ = std.mem.bytesAsSlice(f32, g.host(vcnt, L * 128 * 4));
                for (out, s_, c_) |*o, s, c| o.* = s / c;
                return out;
            }
            g.flat(.m3_euler, L * 128, &.{ x, vsum, vcnt }, [2]f32{ sig[step + 1] - sigma, @floatFromInt(L * 128) });
            try g.commit();
            if (progress) |p| {
                try g.flush();
                p.f(p.ctx, "dit", step + 1, steps);
            }
        }
        const out = try dt.gpa.alloc(f32, L * 128);
        errdefer dt.gpa.free(out);
        try g.get(x, std.mem.sliceAsBytes(out));
        return out;
    }

    fn small(dt: *Dit, w: u64, b: u64, x: u64, K: usize, N: usize, y: u64) void {
        dt.g.go(.m3_linear_h1, .{ (N + 7) / 8, 1, 1 }, 256, &.{ w, b, x, y }, [2]u32{ u(K), u(N) });
    }

    fn layerNorm(dt: *Dit, x: u64, bias: u64, g_: u64, b: u64, y: u64, rows: usize) void {
        dt.g.go(.m3_layernorm_h, .{ rows, 1, 1 }, 256, &.{ x, bias, g_, b, y }, @as(u32, @intFromBool(bias != 0)));
    }
};
