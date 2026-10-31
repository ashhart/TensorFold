//! Idol's image encoder on Metal: FLUX.2's VAE encoder in fp32, a 1024 x 1024 picture to the condition's 4,101 rows.
const std = @import("std");
const mtl = @import("metal");
const ks = @import("vae_kernels.zig");
const wts = @import("vae_weights.zig");
const ref = @import("reference.zig");
const Ref = ks.Ref;

/// The picture the encoder sees, its latent grid, and the condition rows: 5 zero rows, then a row a 2 x 2 patch.
pub const size = 1024;
pub const latent = size / 8;
pub const pad = 5;
pub const rows = pad + (latent / 2) * (latent / 2);
pub const channels = 4 * wts.latent_channels;

pub const Timing = struct { wall: f64, gpu: f64 };

/// Where `encode` copies each stage's output when asked, [rows, columns, channels] fp32 (null: not wanted).
pub const Stages = struct { down: [4]?[]f32 = @splat(null), mid: ?[]f32 = null, moments: ?[]f32 = null };

/// The widest picture tensor: the first stage's 1024 x 1024 x 128.
const most = size * size * wts.widths[0];

pub const Encoder = struct {
    gpa: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    k: ks.Kernels,
    w: wts.Weights,
    scratch: mtl.Buffer,
    image: Ref,
    noise: Ref,
    t: [4]Ref,
    moments: Ref,
    out: Ref,
    stats: Ref,
    part: Ref,

    /// The vae/ folder `dir` (FLUX.2-klein-4B's, or FLUX.2-dev's rounded to bf16 as the SDK loads it).
    pub fn load(gpa: std.mem.Allocator, io: std.Io, device: mtl.Device, dir: []const u8) !*Encoder {
        const v = try gpa.create(Encoder);
        errdefer gpa.destroy(v);
        v.* = undefined;
        v.gpa = gpa;
        v.device = device;
        v.queue = try device.queue();
        errdefer v.queue.deinit();
        v.k = try ks.Kernels.load(device);
        errdefer v.k.deinit();
        v.w = try wts.load(gpa, io, device, dir);
        errdefer v.w.deinit();
        const L = latent * latent;
        const parts = [_]usize{ size * size * 3, L * wts.latent_channels, most, most, most, most, L * 2 * wts.latent_channels, rows * channels, 2 * ks.groups, ks.groups * ks.chunks(size * size) };
        var total: usize = 0;
        for (parts) |n| total += std.mem.alignForward(usize, n * 4, 256);
        v.scratch = try device.buffer(total, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        var at: usize = 0;
        const views = [_]*Ref{ &v.image, &v.noise, &v.t[0], &v.t[1], &v.t[2], &v.t[3], &v.moments, &v.out, &v.stats, &v.part };
        for (views, parts) |view, n| {
            view.* = .{ .buf = v.scratch, .off = at };
            at += std.mem.alignForward(usize, n * 4, 256);
        }
        return v;
    }

    pub fn deinit(v: *Encoder) void {
        v.scratch.deinit();
        v.w.deinit();
        v.k.deinit();
        v.queue.deinit();
        v.gpa.destroy(v);
    }

    /// The condition rows [4101, 128] of `image` [1024, 1024, 3] in [0, 1] with the posterior's `noise` [128, 128, 32].
    pub fn encode(v: *Encoder, image: []const f32, noise: []const f32, out: []f32, stages: Stages) !Timing {
        if (image.len != size * size * 3) return error.PictureSize;
        if (noise.len != latent * latent * wts.latent_channels or out.len != rows * channels) return error.LatentRows;
        @memcpy(floats(v.image, image.len), image);
        @memcpy(floats(v.noise, noise.len), noise);
        const started = mtl.clock.seconds();
        var gpu: f64 = 0;
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        var cb = v.queue.commandBuffer();
        var e = cb.compute(.serial);
        var R: usize = size;
        ks.convIn(e, v.k.get("iv_conv_in"), v.image, v.w.conv_in.w, v.w.conv_in.b, v.t[0], R, R);
        var x: usize = 0;
        for (v.w.stages, 0..) |s, i| {
            for (s.resnets) |r| x = v.resnet(e, r, x, R);
            if (s.down) |d| {
                const y = (x + 1) % v.t.len;
                ks.conv(e, v.k.get("iv_conv"), v.t[x], d.w, d.b, null, v.t[y], .{ .rows = R / 2, .cols = R / 2, .in_rows = R, .in_cols = R, .cin = d.cin, .cout = d.cout, .ldw = d.ldw, .taps = 9 });
                x = y;
                R /= 2;
            }
            if (stages.down[i]) |dst| {
                gpu += try finish(cb, e);
                @memcpy(dst, floats(v.t[x], dst.len));
                cb = v.queue.commandBuffer();
                e = cb.compute(.serial);
            }
        }
        x = v.resnet(e, v.w.mid[0], x, R);
        x = v.attention(e, x, R * R);
        x = v.resnet(e, v.w.mid[1], x, R);
        if (stages.mid) |dst| {
            gpu += try finish(cb, e);
            @memcpy(dst, floats(v.t[x], dst.len));
            cb = v.queue.commandBuffer();
            e = cb.compute(.serial);
        }
        const n = (x + 1) % v.t.len;
        const o = (x + 2) % v.t.len;
        v.norm(e, v.t[x], v.t[n], R * R, wts.widths[3], v.w.norm_out, true);
        v.conv(e, v.w.conv_out, v.t[n], v.t[o], R, null);
        v.conv(e, v.w.quant, v.t[o], v.moments, R, null);
        ks.tokens(e, v.k.get("iv_tokens"), v.moments, v.noise, v.w.bn_mean, v.w.bn_std, v.out, R, R, pad);
        gpu += try finish(cb, e);
        if (stages.moments) |dst| @memcpy(dst, floats(v.moments, dst.len));
        @memcpy(out, floats(v.out, out.len));
        return .{ .wall = mtl.clock.seconds() - started, .gpu = gpu };
    }

    fn finish(cb: mtl.CommandBuffer, e: mtl.ComputeEncoder) !f64 {
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |msg| {
            std.log.err("idol vae: a pass failed on the GPU: {s}", .{msg});
            return error.GpuFailure;
        }
        return cb.gpuSeconds();
    }

    /// One residual block on t[x] at R x R, its output in another of t; returns that one's index.
    fn resnet(v: *const Encoder, e: mtl.ComputeEncoder, r: wts.Resnet, x: usize, R: usize) usize {
        const n = (x + 1) % v.t.len;
        const h = (x + 2) % v.t.len;
        const s = (x + 3) % v.t.len;
        v.norm(e, v.t[x], v.t[n], R * R, r.conv1.cin, r.norm1, true);
        v.conv(e, r.conv1, v.t[n], v.t[h], R, null);
        v.norm(e, v.t[h], v.t[n], R * R, r.conv1.cout, r.norm2, true);
        const skip = if (r.shortcut) |sc| blk: {
            v.conv(e, sc, v.t[x], v.t[s], R, null);
            break :blk v.t[s];
        } else v.t[x];
        v.conv(e, r.conv2, v.t[n], v.t[h], R, skip);
        return h;
    }

    /// The mid block's self-attention on t[x] (`n` positions): GroupNorm, q | k | v, one head of 512, out + t[x].
    fn attention(v: *const Encoder, e: mtl.ComputeEncoder, x: usize, n: usize) usize {
        const xn = (x + 1) % v.t.len;
        const qkv = v.t[(x + 2) % v.t.len];
        const att = v.t[(x + 3) % v.t.len];
        const W = ks.width;
        v.norm(e, v.t[x], v.t[xn], n, W, v.w.attn_norm, false);
        ks.conv(e, v.k.get("iv_conv"), v.t[xn], v.w.qkv.w, v.w.qkv.b, null, qkv, .{ .rows = n, .cols = 1, .in_rows = n, .in_cols = 1, .cin = W, .cout = 3 * W, .ldw = 3 * W });
        ks.attend(e, v.k.get("iv_attention"), qkv, att, n);
        ks.conv(e, v.k.get("iv_conv"), att, v.w.attn_out.w, v.w.attn_out.b, v.t[x], v.t[xn], .{ .rows = n, .cols = 1, .in_rows = n, .in_cols = 1, .cin = W, .cout = W, .ldw = W });
        return xn;
    }

    /// y = GroupNorm(x) with `nm` (then SiLU when `silu`), x and y [pixels, channels], statistics in two passes.
    fn norm(v: *const Encoder, e: mtl.ComputeEncoder, x: Ref, y: Ref, n: usize, ch: usize, nm: wts.Norm, silu: bool) void {
        for ([_]bool{ false, true }) |centered| {
            ks.groupPart(e, v.k.get("iv_group_part"), x, v.stats, v.part, n, ch, centered);
            ks.groupFinish(e, v.k.get("iv_group_finish"), v.part, v.stats, n, ch, centered);
        }
        ks.normed(e, v.k.get("iv_norm"), x, v.stats, nm.g, nm.b, y, n, ch, silu);
    }

    /// y = conv(x) + b (+ r) at R x R, a same-size 3x3 (Winograd) or a 1x1.
    fn conv(v: *const Encoder, e: mtl.ComputeEncoder, c: wts.Conv, x: Ref, y: Ref, R: usize, r: ?Ref) void {
        const s: ks.ConvShape = .{ .rows = R, .cols = R, .in_rows = R, .in_cols = R, .cin = c.cin, .cout = c.cout, .ldw = c.ldw };
        switch (c.kind) {
            .wino => ks.winograd(e, v.k.get("iv_wino"), x, c.w, c.b, r, y, s),
            .one => ks.conv(e, v.k.get("iv_conv"), x, c.w, c.b, r, y, s),
            .strided, .input => unreachable,
        }
    }
};

fn floats(r: Ref, n: usize) []f32 {
    return @as([*]f32, @ptrCast(@alignCast(r.addr())))[0..n];
}

/// The encoder's gate against the float64 ruler: worst row and 1 - cosine within twice the SDK's own, rows of 128.
pub const Verdict = struct { ours: ref.Match, sdk: ref.Match, pass: bool };

pub fn gate(ours: []const f32, sdk: []const f32, ruler: []const f32) Verdict {
    const o = ref.match(ours[pad * channels ..], ruler[pad * channels ..], channels);
    const s = ref.match(sdk[pad * channels ..], ruler[pad * channels ..], channels);
    const zeros = for (ours[0 .. pad * channels]) |z| {
        if (z != 0) break false;
    } else true;
    return .{ .ours = o, .sdk = s, .pass = zeros and ref.within(o, s) };
}

test "the encoder's scratch fits a GPU buffer and the rows are the denoiser's condition rows" {
    try std.testing.expectEqual(@as(usize, 4101), rows);
    try std.testing.expectEqual(@as(usize, 128), channels);
    try std.testing.expect(4 * most * 4 < 1 << 33);
}

test "a planted mismatch in one condition row fails the encoder's gate, as does a padding row not zero" {
    const a = std.testing.allocator;
    const n = rows * channels;
    const ruler = try a.alloc(f32, n);
    defer a.free(ruler);
    const sdk = try a.alloc(f32, n);
    defer a.free(sdk);
    const ours = try a.alloc(f32, n);
    defer a.free(ours);
    var rng = std.Random.DefaultPrng.init(11);
    for (ruler, sdk, ours, 0..) |*r, *s, *o, i| {
        const z = i < pad * channels;
        r.* = if (z) 0 else rng.random().floatNorm(f32);
        s.* = r.* * (1 + 4e-3 * rng.random().floatNorm(f32));
        o.* = r.* * (1 + 1e-6 * rng.random().floatNorm(f32));
    }
    try std.testing.expect(gate(ours, sdk, ruler).pass);
    const row = 2000;
    for (ours[row * channels ..][0..channels]) |*o| o.* *= 1.02;
    const v = gate(ours, sdk, ruler);
    try std.testing.expect(!v.pass);
    try std.testing.expect(v.ours.worst_row > 0.019);
    for (ours[row * channels ..][0..channels], ruler[row * channels ..][0..channels]) |*o, r| o.* = r;
    try std.testing.expect(gate(ours, sdk, ruler).pass);
    ours[3 * channels + 7] = 1e-3;
    try std.testing.expect(!gate(ours, sdk, ruler).pass);
}
