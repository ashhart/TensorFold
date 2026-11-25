//! H3 audio VAE decoder (DAC-lineage BigVGAN vocoder) — pure Zig f32 CPU port.
//! Reference: minimax-h3-mlx audio VAE / audio_vae/dac_bigvgan.py (checkpoint FL2VA).
const std = @import("std");
extern "c" fn fopen(path: [*:0]const u8, mode: [*:0]const u8) ?*anyopaque;
extern "c" fn fwrite(ptr: *const anyopaque, size: usize, count: usize, file: *anyopaque) usize;
extern "c" fn fclose(file: *anyopaque) c_int;
const core = @import("core");
pub const Checkpoint = core.checkpoint_metal.Checkpoint;

pub const AudioVAE = struct {
    gpa: std.mem.Allocator,
    // dec_in_proj: 1x1 conv 32 -> 2048
    dec_in_w: []f32 = &.{}, // [2048*32]
    dec_in_b: []f32 = &.{},
    // conv_pre: 2048 -> 1024, k7 pad3
    conv_pre_w: []f32 = &.{}, // [1024*2048*7] (out, in, k)
    conv_pre_b: []f32 = &.{},
    // ups: 7 ConvTranspose1d (in, out, k, stride, pad)
    ups_w: [7][]f32 = undefined, // [in*out*k]
    ups_b: [7][]f32 = undefined,
    ups_in: [7]usize = undefined,
    ups_out: [7]usize = undefined,
    ups_k: [7]usize = undefined,
    ups_s: [7]usize = undefined,
    // resblocks: 7 * 3 blocks; each: convs1 (k, dil) + convs2 (k, dil=1), ch
    rb_ch: [7]usize = undefined,
    rb_c1_w: [7][9][]f32 = undefined, // [out*in*k]
    rb_c1_b: [7][9][]f32 = undefined,
    rb_c1_k: [7][9]usize = undefined,
    rb_c1_d: [7][9]usize = undefined,
    rb_c2_w: [7][9][]f32 = undefined,
    rb_c2_b: [7][9][]f32 = undefined,
    rb_c2_k: [7][9]usize = undefined,
    // snake alphas/betas per resblock activation (2 per block: a1, a2) + activation_post
    rb_alpha: [7][18][]f32 = undefined, // activations j: a1,a2 × 3 blocks
    rb_beta: [7][18][]f32 = undefined,
    // activation1d filters per resblock activation (up/down, k12, ratio 2)
    rb_up_f: [7][18][]f32 = undefined, // [12]
    rb_dn_f: [7][18][]f32 = undefined,
    // activation_post (final): snake 8ch + filters
    post_alpha: []f32 = &.{},
    post_beta: []f32 = &.{},
    post_up_f: []f32 = &.{},
    post_dn_f: []f32 = &.{},
    // conv_post: 8 -> 1, k7 pad3, no bias
    conv_post_w: []f32 = &.{}, // [8*7]
    ch0: usize = 2048, // upsample_initial_channel
    latent_ch: usize = 32,
    vae_latent: usize = 32,
    n_ups: usize = 7,

    fn readF32(cp: *const Checkpoint, name: []const u8, count: usize, gpa: std.mem.Allocator) ![]f32 {
        const tv = cp.tensors.get(name) orelse return error.MissingTensor;
        const th = @constCast(tv.host(f32));
        const dst = try gpa.alloc(f32, count);
        @memcpy(dst[0..count], th[0..count]);
        return dst;
    }

    fn wn(cp: *const Checkpoint, g: []const u8, v: []const u8, gpa: std.mem.Allocator) ![]f32 {
        // weight_norm fuse: w = v * (g / ||v||) with norm over (out, k) per out-channel
        const tv = cp.tensors.get(v) orelse return error.MissingTensor;
        const tg = cp.tensors.get(g) orelse return error.MissingTensor;
        const vv = @constCast(tv.host(f32));
        const gg = @constCast(tg.host(f32));
        const out_ch = gg.len; // g: [out,1,1]
        const per = vv.len / out_ch;
        const w = try gpa.alloc(f32, vv.len);
        for (0..out_ch) |o| {
            var s: f32 = 0;
            for (0..per) |i| {
                const x = vv[o * per + i];
                s += x * x;
            }
            const scale = gg[o] / @sqrt(s + 1e-12);
            for (0..per) |i| w[o * per + i] = vv[o * per + i] * scale;
        }
        return w;
    }

    fn wnT(cp: *const Checkpoint, g: []const u8, v: []const u8, k: usize, gpa: std.mem.Allocator) ![]f32 {
        // ConvTranspose1d weight_norm fuse: v [in, out, k], g [out]; norm per out over (in, k)
        const tv = cp.tensors.get(v) orelse return error.MissingTensor;
        const tg = cp.tensors.get(g) orelse return error.MissingTensor;
        const vv = @constCast(tv.host(f32));
        const gg = @constCast(tg.host(f32));
        const in_ch = gg.len; // weight_norm(dim=0): g per IN channel for ConvTranspose [in, out, k]
        const out_ch = vv.len / (in_ch * k);
        const w = try gpa.alloc(f32, vv.len);
        // weight_norm(dim=0) on [in, out, k]: scale per IN channel over (out, k)
        for (0..in_ch) |i| {
            var sum: f32 = 0;
            for (0..out_ch) |o| {
                for (0..k) |kk| {
                    const x = vv[i * out_ch * k + o * k + kk];
                    sum += x * x;
                }
            }
            const scale = gg[i] / @sqrt(sum + 1e-12);
            for (0..out_ch) |o| {
                for (0..k) |kk| w[i * out_ch * k + o * k + kk] = vv[i * out_ch * k + o * k + kk] * scale;
            }
        }
        return w;
    }

    pub fn load(gpa: std.mem.Allocator, device: anytype, dir: []const u8) !*AudioVAE {
        const s = try gpa.create(AudioVAE);
        errdefer gpa.destroy(s);
        s.* = .{ .gpa = gpa };
        var arena_state = std.heap.ArenaAllocator.init(gpa);
        defer arena_state.deinit();
        const arena = arena_state.allocator();
        var cp = Checkpoint.init(gpa);
        const path = try std.fmt.allocPrintSentinel(arena, "{s}/model.safetensors", .{dir}, 0);
        try cp.addFile(device, path, "");
        s.dec_in_w = try readF32(&cp, "dec_in_proj.weight", 2048 * 32, gpa);
        s.dec_in_b = try readF32(&cp, "dec_in_proj.bias", 2048, gpa);
        s.conv_pre_w = try wn(&cp, "decoder.conv_pre.weight_g", "decoder.conv_pre.weight_v", gpa);
        s.conv_pre_b = try readF32(&cp, "decoder.conv_pre.bias", 1024, gpa);
        const rates = [7]usize{ 5, 5, 2, 2, 2, 2, 2 };
        const ks = [7]usize{ 9, 9, 4, 4, 4, 4, 4 };
        var ch_in: usize = 1024;
        for (0..7) |i| {
            const ch_out = ch_in / 2;
            s.ups_in[i] = ch_in;
            s.ups_out[i] = ch_out;
            s.ups_k[i] = ks[i];
            s.ups_s[i] = rates[i];
            var nb: [64]u8 = undefined;
            var nb2: [64]u8 = undefined;
            const wg = try std.fmt.bufPrint(&nb, "decoder.ups.{d}.0.weight_g", .{i});
            const wv = try std.fmt.bufPrint(&nb2, "decoder.ups.{d}.0.weight_v", .{i});
            s.ups_w[i] = try wnT(&cp, wg, wv, ks[i], gpa);
            const bi = try std.fmt.bufPrint(&nb, "decoder.ups.{d}.0.bias", .{i});
            s.ups_b[i] = try readF32(&cp, bi, ch_out, gpa);
            ch_in = ch_out;
        }
        s.ch0 = 2048;
        var rch: usize = 512;
        for (0..7) |i| {
            s.rb_ch[i] = rch;
            const dil = [3]usize{ 1, 3, 5 };
            const rks = [3]usize{ 3, 7, 11 };
            // checkpoint: resblocks.{i*3+n} = AMPBlock(kernel=rks[n], dilations=dil) with its own
            // convs1.{0..2}/convs2.{0..2} (index = dilation) and activations.{0..5} (2 per dilation).
            for (0..3) |n| {
                for (0..3) |d| {
                    const idx = n * 3 + d;
                    var nb: [96]u8 = undefined;
                    var nb2: [96]u8 = undefined;
                    s.rb_c1_k[i][idx] = rks[n];
                    s.rb_c1_d[i][idx] = dil[d];
                    s.rb_c2_k[i][idx] = rks[n];
                    const rbk = i * 3 + n;
                    s.rb_c1_w[i][idx] = try wn(&cp, try std.fmt.bufPrint(&nb, "decoder.resblocks.{d}.convs1.{d}.weight_g", .{ rbk, d }), try std.fmt.bufPrint(&nb2, "decoder.resblocks.{d}.convs1.{d}.weight_v", .{ rbk, d }), gpa);
                    s.rb_c1_b[i][idx] = try readF32(&cp, try std.fmt.bufPrint(&nb, "decoder.resblocks.{d}.convs1.{d}.bias", .{ rbk, d }), rch, gpa);
                    s.rb_c2_w[i][idx] = try wn(&cp, try std.fmt.bufPrint(&nb, "decoder.resblocks.{d}.convs2.{d}.weight_g", .{ rbk, d }), try std.fmt.bufPrint(&nb2, "decoder.resblocks.{d}.convs2.{d}.weight_v", .{ rbk, d }), gpa);
                    s.rb_c2_b[i][idx] = try readF32(&cp, try std.fmt.bufPrint(&nb, "decoder.resblocks.{d}.convs2.{d}.bias", .{ rbk, d }), rch, gpa);
                    for (0..2) |a| {
                        const act_i = idx * 2 + a;
                        const key_act = d * 2 + a;
                        s.rb_alpha[i][act_i] = try readF32(&cp, try std.fmt.bufPrint(&nb, "decoder.resblocks.{d}.activations.{d}.act.alpha", .{ rbk, key_act }), rch, gpa);
                        s.rb_beta[i][act_i] = try readF32(&cp, try std.fmt.bufPrint(&nb, "decoder.resblocks.{d}.activations.{d}.act.beta", .{ rbk, key_act }), rch, gpa);
                        s.rb_up_f[i][act_i] = try readF32(&cp, try std.fmt.bufPrint(&nb, "decoder.resblocks.{d}.activations.{d}.upsample.filter", .{ rbk, key_act }), 12, gpa);
                        s.rb_dn_f[i][act_i] = try readF32(&cp, try std.fmt.bufPrint(&nb, "decoder.resblocks.{d}.activations.{d}.downsample.lowpass.filter", .{ rbk, key_act }), 12, gpa);
                    }
                }
            }
            rch /= 2;
        }
        s.post_alpha = try readF32(&cp, "decoder.activation_post.act.alpha", 8, gpa);
        s.post_beta = try readF32(&cp, "decoder.activation_post.act.beta", 8, gpa);
        s.post_up_f = try readF32(&cp, "decoder.activation_post.upsample.filter", 12, gpa);
        s.post_dn_f = try readF32(&cp, "decoder.activation_post.downsample.lowpass.filter", 12, gpa);
        s.conv_post_w = try wn(&cp, "decoder.conv_post.weight_g", "decoder.conv_post.weight_v", gpa);
        return s;
    }

    /// snake activation with per-channel log-scale alpha/beta; x layout [C, T]
    fn snake(x: []f32, C: usize, T: usize, alpha: []const f32, beta: []const f32) void {
        for (0..C) |c| {
            const a = @exp(alpha[c]);
            const b = @exp(beta[c]);
            const inv = 1.0 / (b + 1e-9);
            const row = x[c * T ..][0..T];
            for (row) |*p| {
                const s = @sin(a * p.*);
                p.* += inv * s * s;
            }
        }
    }

    /// replicate pad helper: x [C, T] -> y [C, T+pl+pr]
    fn padReplicate(x: []const f32, C: usize, T: usize, pl: usize, pr: usize, y: []f32) void {
        const Tp = T + pl + pr;
        for (0..C) |c| {
            const xr = x[c * T ..][0..T];
            const yr = y[c * Tp ..][0..Tp];
            for (0..Tp) |t| {
                const s = if (t < pl) xr[0] else if (t >= pl + T) xr[T - 1] else xr[t - pl];
                yr[t] = s;
            }
        }
    }

    /// Activation1d up step: x [C, T] -> out[0..C*2T]; scratch y = [C, T+10], o = [C, 2T+30]
    fn actUp(x: []const f32, C: usize, T: usize, filter: []const f32, y: []f32, o: []f32, out: []f32) void {
        const k = filter.len; // 12
        const pad: usize = 5;
        const Tp = T + 2 * pad;
        const raw = (Tp - 1) * 2 + k; // 2T+30
        const cut = 15; // pad_left = pad_right = 15
        padReplicate(x, C, T, pad, pad, y);
        @memset(o[0 .. C * raw], 0);
        for (0..C) |c| {
            const tr = y[c * Tp ..][0..Tp];
            const orr = o[c * raw ..][0..raw];
            for (0..Tp) |t| {
                const v = tr[t];
                const base = t * 2;
                for (0..k) |kk| orr[base + kk] += v * filter[kk];
            }
            for (0..raw) |i| orr[i] *= 2.0;
            @memcpy(out[c * T * 2 ..][0..T * 2], orr[cut .. cut + T * 2]);
        }
    }

    /// Activation1d down step: x [C, T] -> out[0..C*T/2]; scratch y [C, T+11]
    fn actDown(x: []const f32, C: usize, T: usize, filter: []const f32, y: []f32, out: []f32) void {
        const k = filter.len; // 12
        const pl: usize = 5;
        const pr: usize = 6;
        const Tp = T + pl + pr;
        const outT = (Tp - k) / 2 + 1; // T/2
        padReplicate(x, C, T, pl, pr, y);
        for (0..C) |c| {
            const tr = y[c * Tp ..][0..Tp];
            const orr = out[c * outT ..][0..outT];
            for (0..outT) |t| {
                var acc: f32 = 0;
                const base = t * 2;
                for (0..k) |kk| acc += tr[base + kk] * filter[kk];
                orr[t] = acc;
            }
        }
    }

    /// SnakeBeta wrapped in alias-free up/down: in x [C, T] -> result in OUT [C, T/2*2] = [C, T]
    /// (up x2, act, down x2). scratch: y (pad), o (raw up), b (after act), y2, res
    fn act1d(x: []const f32, C: usize, T: usize, alpha: []const f32, beta: []const f32, upf: []const f32, dnf: []const f32, sc: *Scratch) void {
        const T2 = T * 2;
        actUp(x, C, T, upf, sc.y, sc.o, sc.b); // b = [C, 2T]
        snake(sc.b[0 .. C * T2], C, T2, alpha, beta);
        actDown(sc.b[0 .. C * T2], C, T2, dnf, sc.y, sc.res); // res = [C, T]
        @memcpy(sc.b[0 .. C * T], sc.res[0 .. C * T]);
    }



    pub const Scratch = struct {
        y: []f32 = &.{},
        o: []f32 = &.{},
        b: []f32 = &.{},
        res: []f32 = &.{},
    };

    /// Conv1d stride1: w [out, in, k], dilation d, replicate pad d*(k-1)/2 both sides.
    fn conv1d(x: []const f32, inC: usize, T: usize, w: []const f32, b: []const f32, outC: usize, k: usize, d: usize, y: []f32, tmp: []f32) void {
        const p = d * (k - 1) / 2;
        const Tp = T + 2 * p;
        @memset(tmp[0 .. inC * Tp], 0);
        for (0..inC) |ci| @memcpy(tmp[ci * Tp + p ..][0..T], x[ci * T ..][0..T]);
        for (0..outC) |o| {
            const wr = w[o * inC * k ..][0 .. inC * k];
            const orr = y[o * T ..][0..T];
            const bias = if (b.len > 0) b[o] else 0;
            for (0..T) |t| {
                var acc: f32 = bias;
                for (0..inC) |i| {
                    const trr = tmp[i * Tp ..][0..Tp];
                    const wr2 = wr[i * k ..][0..k];
                    for (0..k) |kk| acc += trr[t + kk * d] * wr2[kk];
                }
                orr[t] = acc;
            }
        }
    }

    /// ConvTranspose1d, torch w [in, out, k], stride s, padding p=(k-s)/2, no out pad.
    fn convT(x: []const f32, inC: usize, T: usize, w: []const f32, b: []const f32, outC: usize, k: usize, s: usize, p: usize, y: []f32, raw_scratch: []f32) void {
        const raw = (T - 1) * s + k;
        const out_len = raw - 2 * p;
        @memset(raw_scratch[0 .. outC * raw], 0);
        for (0..inC) |i| {
            const xr = x[i * T ..][0..T];
            for (0..T) |t| {
                const v = xr[t];
                if (v == 0) continue;
                const wr = w[i * outC * k ..][0 .. outC * k];
                const base = t * s;
                for (0..outC) |o| {
                    const wr2 = wr[o * k ..][0..k];
                    const orr = raw_scratch[o * raw ..][0..raw];
                    for (0..k) |kk| orr[base + kk] += v * wr2[kk];
                }
            }
        }
        for (0..outC) |o| {
            const orr = raw_scratch[o * raw ..][0..raw];
            const orr2 = y[o * out_len ..][0..out_len];
            for (0..out_len) |t| orr2[t] = orr[t + p] + b[o];
        }
    }

    const ups_rates = [7]usize{ 5, 5, 2, 2, 2, 2, 2 };
    const ups_ks = [7]usize{ 9, 9, 4, 4, 4, 4, 4 };

    /// Decode one batch element: z [32, T_lat] -> wave [T_out] f32 (T_out = T_lat * 800).
    pub fn decodeOne(self: *const AudioVAE, gpa: std.mem.Allocator, z: []const f32, T_lat: usize) ![]f32 {
        const vae_ch = self.vae_latent;
        var T = T_lat;
        var ch: usize = 2048;
        const cur = try gpa.alloc(f32, vae_ch * T_lat * 256);
        defer gpa.free(cur);
        for (0..T) |t| {
            for (0..ch) |o| {
                var acc: f32 = self.dec_in_b[o];
                const wr = self.dec_in_w[o * vae_ch ..][0..vae_ch];
                for (0..vae_ch) |i| acc += z[i * T_lat + t] * wr[i];
                cur[o * T + t] = acc;
            }
        }
        ch = 1024;
        {
            const x1 = try gpa.alloc(f32, ch * T);
            defer gpa.free(x1);
            const tmp = try gpa.alloc(f32, 2048 * (T + 6));
            defer gpa.free(tmp);
            conv1d(cur[0 .. 2048 * T], 2048, T, self.conv_pre_w, self.conv_pre_b, 1024, 7, 1, x1, tmp);
            @memcpy(cur[0 .. ch * T], x1);
        }
        var sc = Scratch{};
        defer { if (sc.y.len > 0) gpa.free(sc.y); if (sc.o.len > 0) gpa.free(sc.o); if (sc.b.len > 0) gpa.free(sc.b); if (sc.res.len > 0) gpa.free(sc.res); }
        for (0..7) |i| {
            const s = self.ups_s[i];
            const k = self.ups_k[i];
            const inC = self.ups_in[i];
            const outC = self.ups_out[i];
            const p = (k - s) / 2;
            const T_up = T * s;
            const raw = (T - 1) * s + k;
            const raws = try gpa.alloc(f32, outC * raw);
            defer gpa.free(raws);
            const up_buf = try gpa.alloc(f32, outC * T_up);
            defer gpa.free(up_buf);
            convT(cur[0 .. inC * T], inC, T, self.ups_w[i], self.ups_b[i], outC, k, s, p, up_buf, raws);
            @memcpy(cur[0 .. outC * T_up], up_buf);
            const rT = T_up;
            const rch = outC;
            // (re)allocate act1d scratch for this stage
            if (sc.y.len > 0) gpa.free(sc.y);
            if (sc.o.len > 0) gpa.free(sc.o);
            if (sc.b.len > 0) gpa.free(sc.b);
            if (sc.res.len > 0) gpa.free(sc.res);
            sc.y = try gpa.alloc(f32, rch * (rT * 2 + 16));
            sc.o = try gpa.alloc(f32, rch * (rT * 2 + 32));
            sc.b = try gpa.alloc(f32, rch * (rT * 2 + 32));
            sc.res = try gpa.alloc(f32, rch * (rT + 4));
            const xs = try gpa.alloc(f32, rch * rT);
            defer gpa.free(xs);
            @memset(xs, 0);
            const xt = try gpa.alloc(f32, rch * rT);
            defer gpa.free(xt);
            const tmp11 = try gpa.alloc(f32, rch * (rT + 2 * 5 * 10 + 8));
            defer gpa.free(tmp11);
            const xcur = try gpa.alloc(f32, rch * rT);
            defer gpa.free(xcur);
            for (0..3) |n| { // blocks: kernel rks[n], dilations 1/3/5
                @memcpy(xcur[0 .. rch * rT], cur[0 .. rch * rT]);
                for (0..3) |d| {
                    const j = n * 3 + d;
                    act1d(xcur[0 .. rch * rT], rch, rT, self.rb_alpha[i][j * 2], self.rb_beta[i][j * 2], self.rb_up_f[i][j * 2], self.rb_dn_f[i][j * 2], &sc);
                    const kk1 = self.rb_c1_k[i][j];
                    const dd1 = self.rb_c1_d[i][j];
                    const p1 = dd1 * (kk1 - 1) / 2;
                    const t1 = try gpa.alloc(f32, rch * (rT + 2 * p1));
                    defer gpa.free(t1);
                    conv1d(sc.b[0 .. rch * rT], rch, rT, self.rb_c1_w[i][j], self.rb_c1_b[i][j], rch, kk1, dd1, xt, t1);
                    act1d(xt, rch, rT, self.rb_alpha[i][j * 2 + 1], self.rb_beta[i][j * 2 + 1], self.rb_up_f[i][j * 2 + 1], self.rb_dn_f[i][j * 2 + 1], &sc);
                    const kk2 = self.rb_c2_k[i][j];
                    const p2 = (kk2 - 1) / 2;
                    const t2 = try gpa.alloc(f32, rch * (rT + 2 * p2));
                    defer gpa.free(t2);
                    conv1d(sc.b[0 .. rch * rT], rch, rT, self.rb_c2_w[i][j], self.rb_c2_b[i][j], rch, kk2, 1, xt, t2);
                    for (0..rch * rT) |q| { xs[q] += xt[q]; xcur[q] += xt[q]; }
                }
            }
            for (0..rch * rT) |q| cur[q] += xs[q] / 3.0;
            T = rT;
            ch = rch;
        }
        if (sc.y.len > 0) gpa.free(sc.y);
        if (sc.o.len > 0) gpa.free(sc.o);
        if (sc.b.len > 0) gpa.free(sc.b);
        if (sc.res.len > 0) gpa.free(sc.res);
        sc.y = try gpa.alloc(f32, ch * (T * 2 + 16));
        sc.o = try gpa.alloc(f32, ch * (T * 2 + 32));
        sc.b = try gpa.alloc(f32, ch * (T * 2 + 32));
        sc.res = try gpa.alloc(f32, ch * (T + 4));
        act1d(cur[0 .. ch * T], ch, T, self.post_alpha, self.post_beta, self.post_up_f, self.post_dn_f, &sc);
        const wav = try gpa.alloc(f32, T);
        const wtmp = try gpa.alloc(f32, ch * (T + 6));
        defer gpa.free(wtmp);
        conv1d(sc.b[0 .. ch * T], ch, T, self.conv_post_w, &[_]f32{}, 1, 7, 1, wav, wtmp);
        for (wav) |*v| v.* = std.math.clamp(v.*, -1.0, 1.0);
        return wav;
    }
};
