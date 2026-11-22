//! H3 video VAE CNN encoder (fp32 CPU reference port).
//! Spec: video_vae/vae_cnn.py EncoderFCN3D + conv.py causal Conv3d.
const std = @import("std");
const mtl = @import("metal");

extern "c" fn fopen(path: [*:0]const u8, mode: [*:0]const u8) ?*anyopaque;

extern "c" fn fwrite(ptr: [*]const u8, size: usize, nmemb: usize, stream: *anyopaque) usize;
extern "c" fn fclose(stream: *anyopaque) c_int;
extern "c" fn getenv(name: [*:0]const u8) ?[*:0]u8;
var gn_call_count: usize = 0;

pub const Config = struct {
    ch: usize = 128,
    ch_mult: [6]usize = .{ 1, 2, 2, 4, 4, 8 },
    num_res_blocks: usize = 2,
    z_channels: usize = 24,
    in_channels: usize = 3,
    groups: usize = 32,
    gn_eps: f32 = 1e-6,
};

/// Conv3D with causal temporal padding (pad*2 frames prepended, none
/// appended) and spatial zero padding. stride: (st, sh, sw).
pub fn conv3d(
    gpa: std.mem.Allocator,
    input: []const f32, // (C_in, T, H, W)
    weight: []const f32, // (C_out, C_in, kt, kh, kw)
    bias: []const f32, // (C_out)
    c_in: usize,
    c_out: usize,
    t: usize,
    h: usize,
    w: usize,
    kt: usize,
    kh: usize,
    kw: usize,
    st: usize,
    sh: usize,
    sw: usize,
    causal_t: bool,
    spatial_pad: usize,
    pad_right: usize,
    pad_bottom: usize,
) ![]f32 {
    // temporal causal padding: 2*pt frames prepended when causal
    const pt = kt / 2; // padding per side for non-causal
    const t_pad_pre: usize = if (causal_t) pt * 2 else pt;
    const t_pad_post: usize = if (causal_t) 0 else pt;
    const tp = t + t_pad_pre + t_pad_post;
    const hp = h + spatial_pad + pad_bottom;
    const wp = w + spatial_pad + pad_right;

    const padded = try gpa.alloc(f32, c_in * tp * hp * wp);
    defer gpa.free(padded);
    @memset(padded, 0);
    // copy input into padded (temporal offset t_pad_pre, spatial offset pad)
    for (0..c_in) |c| {
        for (0..t) |tt| {
            for (0..h) |hh| {
                const src = ((c * t) + tt) * h * w + hh * w;
                const dst = ((c * tp) + (tt + t_pad_pre)) * hp * wp + (hh + spatial_pad) * wp + spatial_pad;
                @memcpy(padded[dst .. dst + w], input[src .. src + w]);
            }
        }
    }

    const out_t = (tp - kt) / st + 1;
    const out_h = (hp - kh) / sh + 1;
    const out_w = (wp - kw) / sw + 1;
    const out = try gpa.alloc(f32, c_out * out_t * out_h * out_w);
    @memset(out, 0);

    for (0..c_out) |co| {
        for (0..out_t) |ot| {
            for (0..out_h) |oh| {
                for (0..out_w) |ow| {
                    var sum: f32 = bias[co];
                    for (0..c_in) |ci| {
                        for (0..kt) |ktt| {
                            for (0..kh) |khh| {
                                for (0..kw) |kww| {
                                    const ii = ((ci * tp) + (ot * st + ktt)) * hp * wp + (oh * sh + khh) * wp + (ow * sw + kww);
                                    const ww_idx = ((co * c_in + ci) * kt + ktt) * kh * kw + khh * kw + kww;
                                    sum += padded[ii] * weight[ww_idx];
                                }
                            }
                        }
                    }
                    out[((co * out_t) + ot) * out_h * out_w + oh * out_w + ow] = sum;
                }
            }
        }
    }
    return out;
}

/// GroupNorm (32 groups) + SiLU over (C, T, H, W).
pub fn groupNormSilu(gpa: std.mem.Allocator, x: []f32, c: usize, t: usize, h: usize, w: usize, weight: []const f32, bias: []const f32, groups: usize, eps: f32) !void {
    const spatial = t * h * w;
    const per_group = c / groups;
    const group_size = per_group * spatial;
    var y = try gpa.alloc(f32, x.len);
    defer gpa.free(y);
    for (0..groups) |g| {
        // GroupNorm normalises over (C/G, T, H, W) per sample
        var mean: f64 = 0;
        var m2: f64 = 0;
        for (0..per_group) |ci| {
            const ch = g * per_group + ci;
            for (0..spatial) |s| {
                const v: f64 = x[ch * spatial + s];
                mean += v;
                m2 += v * v;
            }
        }
        mean /= @floatFromInt(group_size);
        m2 /= @floatFromInt(group_size);
        const rstd: f32 = @floatCast(1.0 / @sqrt(m2 - mean * mean + eps));
        for (0..per_group) |ci| {
            const ch = g * per_group + ci;
            for (0..spatial) |s| {
                var v = (x[ch * spatial + s] - @as(f32, @floatCast(mean))) * rstd;
                v = v * weight[ch] + bias[ch];
                v = v / (1.0 + @exp(-v)); // SiLU: v * sigmoid(v)
                y[ch * spatial + s] = v;
            }
        }
    }
    @memcpy(x, y);
    if (getenv("DUMP_GN1") != null and c == 128 and t == 3 and h == 16) {
        gn_call_count += 1;
        if (gn_call_count > 2) return; // only first two calls (norm1, norm2 of L0 b0)
        var pz_buf: [512]u8 = undefined;
        const formatted = try std.fmt.bufPrint(&pz_buf, "zig-gn1-{d}.f32", .{gn_call_count});
        pz_buf[formatted.len] = 0;
        const file = fopen(@ptrCast(&pz_buf), "wb") orelse return error.CannotWrite;
        _ = fwrite(@ptrCast(x.ptr), 4, x.len, file);
        _ = fclose(file);
    }
}

test "conv3d identity smoke" {
    var gpa_state = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa_state.deinit();
    const gpa = gpa_state.allocator();
    // 1x1x1 kernel as identity
    var wgt = [_]f32{1.0};
    var bias = [_]f32{0.0};
    var input = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const out = try conv3d(gpa, &input, &wgt, &bias, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, false, 0);
    defer gpa.free(out);
    try std.testing.expectEqualSlices(f32, &input, out);
}

/// ResnetBlock3D: GN+silu → conv3x3 → GN+silu → conv3x3 → (+nin_shortcut)
pub const ResnetWeights = struct {
    norm1_w: []f32,
    norm1_b: []f32,
    conv1_w: []f32,
    conv1_b: []f32,
    norm2_w: []f32,
    norm2_b: []f32,
    conv2_w: []f32,
    conv2_b: []f32,
    nin_w: ?[]f32 = null,
    nin_b: ?[]f32 = null,
};

pub fn resnetBlock(
    gpa: std.mem.Allocator,
    x: []f32, // (C_in, T, H, W) — modified in place? returns new
    w: ResnetWeights,
    c_in: usize,
    c_out: usize,
    t: usize,
    h: usize,
    wd: usize, // width
    config: Config,
) ![]f32 {
    const h_buf = try gpa.dupe(f32, x);
    // norm1 + silu
    try groupNormSilu(gpa, h_buf, c_in, t, h, wd, w.norm1_w, w.norm1_b, config.groups, config.gn_eps);
    // conv1
    const h1 = try conv3d(gpa, h_buf, w.conv1_w, w.conv1_b, c_in, c_out, t, h, wd, 3, 3, 3, 1, 1, 1, true, 1, 1, 1);
    gpa.free(h_buf);
    // norm2 + silu
    try groupNormSilu(gpa, h1, c_out, t, h, wd, w.norm2_w, w.norm2_b, config.groups, config.gn_eps);
    // conv2
    const h2 = try conv3d(gpa, h1, w.conv2_w, w.conv2_b, c_out, c_out, t, h, wd, 3, 3, 3, 1, 1, 1, true, 1, 1, 1);
    gpa.free(h1);
    // shortcut
    var shortcut: []f32 = undefined;
    if (c_in != c_out) {
        shortcut = try conv1x1(gpa, x, w.nin_w.?, w.nin_b.?, c_in, c_out, t, h, wd);
    } else {
        shortcut = try gpa.dupe(f32, x);
    }
    // x + h2
    for (0..shortcut.len) |i| shortcut[i] += h2[i];
    gpa.free(h2);
    return shortcut;
}

/// 1x1x1 conv (pointwise).
pub fn conv1x1(
    gpa: std.mem.Allocator,
    input: []const f32,
    weight: []const f32, // (C_out, C_in, 1, 1, 1)
    bias: []const f32,
    c_in: usize,
    c_out: usize,
    t: usize,
    h: usize,
    wd: usize,
) ![]f32 {
    const n = t * h * wd;
    const out = try gpa.alloc(f32, c_out * n);
    for (0..c_out) |co| {
        for (0..n) |i| {
            var sum: f32 = bias[co];
            for (0..c_in) |ci| {
                sum += input[ci * n + i] * weight[co * c_in + ci];
            }
            out[co * n + i] = sum;
        }
    }
    return out;
}

/// Metal dispatch of the causal conv3x3x3 kernel. input/weight/bias
/// are host fp32 buffers; kernel is the encoder's h3_vae_enc_conv.
pub fn conv3dMetal(
    device: mtl.Device,
    queue: mtl.Queue,
    pipeline: mtl.Pipeline,
    gpa: std.mem.Allocator,
    input: []const f32,
    weight: []const f32,
    bias: []const f32,
    c_in: usize,
    c_out: usize,
    t: usize,
    h: usize,
    w: usize,
    causal_t: bool,
    spatial_pad: usize,
    pad_right: usize,
    pad_bottom: usize,
) ![]f32 {
    const kt = 3;
    const pt = kt / 2;
    const t_pre: usize = if (causal_t) pt * 2 else pt;
    const tp = t + t_pre;
    const hp = h + spatial_pad + pad_bottom;
    const wp = w + spatial_pad + pad_right;
    const out_t = (tp - kt) + 1;
    const out_h = (hp - 3) + 1;
    const out_w = (wp - 3) + 1;

    const in_buf = try device.buffer(input.len * 4, mtl.ResourceOptions.shared);
    defer in_buf.deinit();
    @memcpy(in_buf.slice(f32, input.len), input);
    const w_buf = try device.buffer(weight.len * 4, mtl.ResourceOptions.shared);
    defer w_buf.deinit();
    @memcpy(w_buf.slice(f32, weight.len), weight);
    const b_buf = try device.buffer(bias.len * 4, mtl.ResourceOptions.shared);
    defer b_buf.deinit();
    @memcpy(b_buf.slice(f32, bias.len), bias);
    const out_buf = try device.buffer(c_out * out_t * out_h * out_w * 4, mtl.ResourceOptions.shared);

    const cb = queue.commandBuffer();
    const enc_ = cb.compute(.concurrent);
    enc_.setPipeline(pipeline);
    enc_.setBuffer(in_buf, 0, 0);
    enc_.setBuffer(w_buf, 0, 1);
    enc_.setBuffer(b_buf, 0, 2);
    enc_.setBuffer(out_buf, 0, 3);
    const causal_u: u32 = if (causal_t) 1 else 0;
    enc_.setValue([12]u32{
        @intCast(c_in), @intCast(t), @intCast(h), @intCast(w), @intCast(c_out),
        1, 1, 1, causal_u, @intCast(spatial_pad), @intCast(pad_right), @intCast(pad_bottom),
    }, 4);
    enc_.dispatchThreads(mtl.Size.of(c_out * out_t, out_h, out_w), mtl.Size.of(1, 1, 1));
    enc_.end();
    cb.commit();
    cb.wait();
    const out = try gpa.alloc(f32, c_out * out_t * out_h * out_w);
    @memcpy(out, out_buf.slice(f32, out.len));
    out_buf.deinit();
    return out;
}

/// Downsample: spatial pad right/bottom 1 (zeros), conv k3 stride (1,2,2),
/// causal temporal.
pub fn downsample(
    gpa: std.mem.Allocator,
    x: []const f32,
    weight: []const f32,
    bias: []const f32,
    c: usize,
    t: usize,
    h: usize,
    wd: usize,
) ![]f32 {
    return conv3d(gpa, x, weight, bias, c, c, t, h, wd, 3, 3, 3, 1, 2, 2, true, 0, 1, 1);
}


pub const Encoder = struct {
    config: Config,

    pub fn init(config: Config) Encoder {
        return .{ .config = config };
    }

    /// Full encoder forward. x: (3, T, H, W) fp32. Returns (48, T, H/16, W/16).
    pub fn forward(
        self: *Encoder,
        gpa: std.mem.Allocator,
        x: []const f32,
        t_in: usize,
        h_in: usize,
        w_in: usize,
        weights: *const AllWeights,
    ) ![]f32 {
        const cfg = self.config;
        var channels: usize = cfg.ch; // 128
        const t = t_in;
        var h = h_in;
        var w = w_in;

        // conv_in
        std.debug.print("ENC: conv_in {d}x{d}x{d}", .{ t, h, w });
        var act = try conv3d(gpa, x, weights.conv_in_w, weights.conv_in_b, cfg.in_channels, channels, t, h, w, 3, 3, 3, 1, 1, 1, true, 1, 1, 1);
        if (getenv("DUMP_CONV_IN") != null) {
            var pz_buf: [512]u8 = undefined;
            const formatted = try std.fmt.bufPrint(&pz_buf, "zig-conv-in.f32", .{});
            pz_buf[formatted.len] = 0;
            const file = fopen(@ptrCast(&pz_buf), "wb") orelse return error.CannotWrite;
            _ = fwrite(@ptrCast(act.ptr), 4, act.len, file);
            _ = fclose(file);
        }

        // levels
        for (0..6) |level| {
            std.debug.print("ENC: level {d} ch={d} {d}x{d}x{d}", .{ level, channels, t, h, w });
            const c_out_level = cfg.ch * cfg.ch_mult[level];
            for (0..cfg.num_res_blocks) |bi| {
                const c_in_block = if (bi == 0) channels else c_out_level;
                const rw = weights.levels[level].blocks[bi];
                std.debug.print("ENC: L{d} block {d} c_in={d} c_out={d}", .{ level, bi, c_in_block, c_out_level });
                const next = try resnetBlock(gpa, act, rw, c_in_block, c_out_level, t, h, w, cfg);
                std.debug.print(" done", .{});
                if (getenv("DUMP_L0B0") != null and level == 0 and bi == 0) {
                    var pz_buf: [512]u8 = undefined;
                    const formatted = try std.fmt.bufPrint(&pz_buf, "zig-l0b0.f32", .{});
                    pz_buf[formatted.len] = 0;
                    const file = fopen(@ptrCast(&pz_buf), "wb") orelse return error.CannotWrite;
                    _ = fwrite(@ptrCast(next.ptr), 4, next.len, file);
                    _ = fclose(file);
                }
                gpa.free(act);
                act = next;
            }
            channels = c_out_level;
            if (weights.levels[level].down_w) |dw| {
                std.debug.print("ENC: L{d} down", .{level});
                const downed = try downsample(gpa, act, dw, weights.levels[level].down_b.?, channels, t, h, w);
                std.debug.print(" done", .{});
                gpa.free(act);
                act = downed;
                h = (h - 1) / 2 + 1;
                w = (w - 1) / 2 + 1;
            }
        }

        // norm_out + silu
        try groupNormSilu(gpa, act, channels, t, h, w, weights.norm_out_w, weights.norm_out_b, cfg.groups, cfg.gn_eps);

        // conv_out
        const out_ch = 2 * cfg.z_channels; // 48
        const moments = try conv3d(gpa, act, weights.conv_out_w, weights.conv_out_b, channels, out_ch, t, h, w, 3, 3, 3, 1, 1, 1, true, 1, 1, 1);
        gpa.free(act);

        // quant_conv 1x1x1 48→48
        const moments2 = try conv1x1(gpa, moments, weights.quant_w, weights.quant_b, out_ch, out_ch, t, h, w);
        gpa.free(moments);
        return moments2;
    }
};

pub const LevelWeights = struct {
    blocks: [2]ResnetWeights,
    down_w: ?[]f32 = null,
    down_b: ?[]f32 = null,
};

pub const AllWeights = struct {
    conv_in_w: []f32,
    conv_in_b: []f32,
    levels: [6]LevelWeights,
    norm_out_w: []f32,
    norm_out_b: []f32,
    conv_out_w: []f32,
    conv_out_b: []f32,
    quant_w: []f32, // (48, 48)
    quant_b: []f32,
};
