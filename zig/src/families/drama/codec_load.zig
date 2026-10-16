//! The pinned DAC codec's geometry and its device weight image, built once on the host from codec.pth.

const std = @import("std");
const pth = @import("codec_pth.zig");

const Allocator = std.mem.Allocator;

// modded_dac_vq.yaml and modded_dac.py's geometry for the pinned codec.
pub const latent = 1024;
pub const codebook_dim = 8;
pub const semantic_size = 4096;
pub const residual_size = 1024;
pub const residual_books = 9;
pub const codebooks = 1 + residual_books;
pub const layers = 8;
pub const heads = 16;
pub const head_dim = 64;
pub const ffn = 3072;
pub const window = 128;
pub const rope_base = 10000.0;
pub const rms_eps: f32 = 1e-5;
pub const layer_norm_eps: f32 = 1e-6;
pub const convnext_hidden = 4 * latent;
pub const upsample_stages = 2; // downsample_factor [2, 2]
pub const decoder_dim = 1536;
pub const rates = [_]usize{ 8, 8, 4, 2 };
pub const dilations = [_]usize{ 1, 3, 9 };
pub const latent_rows_per_frame = 1 << upsample_stages;
const hop = blk: {
    var h: usize = 1;
    for (rates) |r| h *= r;
    break :blk h;
};
pub const samples_per_frame = latent_rows_per_frame * hop; // DAC.frame_length = hop_length * 4 = 2048
pub const out_channels = decoder_dim >> rates.len;

pub const Snake = struct { alpha: u64, inv: u64 };

pub const Layer = struct {
    wqkv: u64,
    wo: u64,
    w13: u64,
    w2: u64,
    attention_norm: u64,
    ffn_norm: u64,
    attention_scale: u64,
    ffn_scale: u64,
};

pub const Upsample = struct {
    conv: u64,
    conv_b: u64,
    dw: u64,
    dw_b: u64,
    norm_w: u64,
    norm_b: u64,
    pw1: u64,
    pw1_b: u64,
    pw2: u64,
    pw2_b: u64,
    gamma: u64,
};

pub const Unit = struct { snake1: Snake, conv7: u64, conv7_b: u64, snake2: Snake, conv1: u64, conv1_b: u64 };

pub const Block = struct { snake: Snake, conv: u64, conv_b: u64, units: [dilations.len]Unit };

/// Byte offsets into the one device weight buffer.
pub const Weights = struct {
    semantic_book: u64,
    residual_books: u64,
    out_w: u64,
    out_b: u64,
    layers: [layers]Layer,
    norm: u64,
    upsample: [upsample_stages]Upsample,
    conv0: u64,
    conv0_b: u64,
    blocks: [rates.len]Block,
    out_snake: Snake,
    out_conv: u64,
    out_conv_b: u64,
    rope: u64,
};

/// The weight image built on the host before one upload; every tensor 256-byte aligned.
pub const Stage = struct {
    gpa: Allocator,
    bytes: std.ArrayList(u8) = .empty,

    fn reserve(st: *Stage, n: usize) !usize {
        const at = std.mem.alignForward(usize, st.bytes.items.len, 256);
        try st.bytes.appendNTimes(st.gpa, 0, at + n - st.bytes.items.len);
        return at;
    }

    /// FP32 values as FP16, round to nearest even (torch's .to(float16)).
    fn halves(st: *Stage, values: []const f32) !u64 {
        const at = try st.reserve(values.len * 2);
        const dst = st.bytes.items[at..][0 .. values.len * 2];
        for (values, 0..) |v, i| std.mem.writeInt(u16, dst[i * 2 ..][0..2], halfBits(v), .little);
        return at;
    }

    fn floats(st: *Stage, values: []const f32) !u64 {
        const at = try st.reserve(values.len * 4);
        @memcpy(st.bytes.items[at..][0 .. values.len * 4], std.mem.sliceAsBytes(values));
        return at;
    }
};

fn halfBits(v: f32) u16 {
    return @bitCast(@as(f16, @floatCast(v)));
}

/// float32 rounded to bfloat16 (round to nearest even) and back, as torch rounds it.
fn bf16Round(v: f32) f32 {
    const b: u32 = @bitCast(v);
    const r = b +% (0x7FFF + ((b >> 16) & 1));
    return @bitCast(r & 0xFFFF0000);
}

/// torch._weight_norm over dim 0 with the CUDA kernel's 256-strided FP32 FMA tree, before rounding to FP16.
fn foldWeightNorm(v: []f32, g: []const f32) !void {
    if (g.len == 0 or v.len % g.len != 0) return error.Invalid;
    const row = v.len / g.len;
    var width: usize = 1;
    while (width < @min(row, 256)) : (width *= 2) {}
    var partial: [256]f32 = undefined;
    for (g, 0..) |gi, i| {
        const r = v[i * row ..][0..row];
        for (0..width) |lane| {
            var sum: f32 = 0;
            var column = lane;
            while (column < row) : (column += 256) sum = @mulAdd(f32, r[column], r[column], sum);
            partial[lane] = sum;
        }
        // For short rows, the omitted upper lanes contain only zeros.
        var stride = width / 2;
        while (stride != 0) : (stride /= 2) {
            for (0..stride) |lane| partial[lane] += partial[lane + stride];
        }
        const norm: f32 = @sqrt(partial[0]);
        if (!(norm > 0)) return error.DegenerateWeightNorm;
        const rnorm: f32 = 1.0 / norm;
        for (r) |*e| e.* = gi * e.* * rnorm;
    }
}

/// Conv1d weight [cout][cin][k] as the implicit GEMM's [cout][k * cin].
fn convTaps(gpa: Allocator, w: []const f32, cout: usize, cin: usize, k: usize) ![]f32 {
    if (w.len != cout * cin * k) return error.Invalid;
    const out = try gpa.alloc(f32, w.len);
    for (0..cout) |co| {
        for (0..cin) |ci| {
            for (0..k) |t| out[(co * k + t) * cin + ci] = w[(co * cin + ci) * k + t];
        }
    }
    return out;
}

/// ConvTranspose1d [cin][cout][k] as the GEMM's [s * cout][taps * cin]; column j * cout + co is sample q * s + j.
fn transposedTaps(gpa: Allocator, w: []const f32, cin: usize, cout: usize, s: usize, k: usize) ![]f32 {
    if (w.len != cin * cout * k or (k != s and k != 2 * s)) return error.Invalid;
    const taps = k / s;
    const out = try gpa.alloc(f32, w.len);
    for (0..s) |j| {
        for (0..cout) |co| {
            for (0..taps) |tap| {
                const kk = if (taps == 2 and tap == 0) j + s else j;
                for (0..cin) |ci| out[((j * cout + co) * taps + tap) * cin + ci] = w[(ci * cout + co) * k + kk];
            }
        }
    }
    return out;
}

/// precompute_freqs_cis(seq_len, 64, 10000) for frames positions, cos and sin rounded to BF16, as [frames][32][2].
fn ropeTable(gpa: Allocator, frames: usize) ![]f32 {
    const pairs = head_dim / 2;
    var freqs: [pairs]f32 = undefined;
    for (&freqs, 0..) |*f, i| {
        const x: f32 = @as(f32, @floatFromInt(2 * i)) / @as(f32, head_dim);
        const p: f32 = @floatCast(std.math.pow(f64, rope_base, @as(f64, x)));
        f.* = 1.0 / p;
    }
    const out = try gpa.alloc(f32, frames * pairs * 2);
    for (0..frames) |t| {
        for (freqs, 0..) |f, i| {
            const angle: f32 = @as(f32, @floatFromInt(t)) * f;
            const a: f64 = angle;
            out[(t * pairs + i) * 2] = bf16Round(@floatCast(@cos(a)));
            out[(t * pairs + i) * 2 + 1] = bf16Round(@floatCast(@sin(a)));
        }
    }
    return out;
}

pub const Loader = struct {
    gpa: Allocator,
    names: Allocator,
    archive: *const pth.Archive,
    stage: *Stage,

    fn name(l: *Loader, comptime fmt: []const u8, args: anytype) ![]const u8 {
        return std.fmt.allocPrint(l.names, fmt, args);
    }

    fn read(l: *Loader, tensor: []const u8, shape: []const u64) ![]f32 {
        return l.archive.readF32(l.gpa, tensor, shape);
    }

    /// A tensor as FP16 in its checkpoint layout.
    fn half(l: *Loader, tensor: []const u8, shape: []const u64) !u64 {
        const v = try l.read(tensor, shape);
        defer l.gpa.free(v);
        return l.stage.halves(v);
    }

    /// A weight-normed tensor folded over dim 0 (caller frees); g is [shape[0], 1, 1].
    fn folded(l: *Loader, g_name: []const u8, v_name: []const u8, shape: []const u64) ![]f32 {
        const v = try l.read(v_name, shape);
        errdefer l.gpa.free(v);
        const g = try l.read(g_name, &[_]u64{ shape[0], 1, 1 });
        defer l.gpa.free(g);
        try foldWeightNorm(v, g);
        return v;
    }

    /// A decoder conv (parametrizations.weight_norm) at `prefix`.conv, folded, as GEMM taps in FP16.
    fn decoderConv(l: *Loader, prefix: []const u8, cout: usize, cin: usize, k: usize) !u64 {
        const v = try l.folded(
            try l.name("{s}.conv.parametrizations.weight.original0", .{prefix}),
            try l.name("{s}.conv.parametrizations.weight.original1", .{prefix}),
            &[_]u64{ cout, cin, k },
        );
        defer l.gpa.free(v);
        const t = try convTaps(l.gpa, v, cout, cin, k);
        defer l.gpa.free(t);
        return l.stage.halves(t);
    }

    fn snake(l: *Loader, tensor: []const u8, channels: usize) !Snake {
        const alpha = try l.read(tensor, &[_]u64{ 1, channels, 1 });
        defer l.gpa.free(alpha);
        const inv = try l.gpa.alloc(f32, channels);
        defer l.gpa.free(inv);
        // (alpha + 1e-9).reciprocal() in FP32
        for (alpha, inv) |a, *r| r.* = 1.0 / (a + @as(f32, 1e-9));
        return .{ .alpha = try l.stage.floats(alpha), .inv = try l.stage.floats(inv) };
    }

    pub fn weights(l: *Loader, max_frames: usize) !Weights {
        var w: Weights = undefined;
        const gpa = l.gpa;

        // quantizer RVQ: codebooks, folded out_proj (legacy weight_norm: weight_g, weight_v)
        w.semantic_book = try l.half("quantizer.semantic_quantizer.quantizers.0.codebook.weight", &[_]u64{ semantic_size, codebook_dim });
        {
            const books = try gpa.alloc(f32, residual_books * residual_size * codebook_dim);
            defer gpa.free(books);
            for (0..residual_books) |i| {
                const v = try l.read(try l.name("quantizer.quantizer.quantizers.{d}.codebook.weight", .{i}), &[_]u64{ residual_size, codebook_dim });
                defer gpa.free(v);
                @memcpy(books[i * residual_size * codebook_dim ..][0..v.len], v);
            }
            w.residual_books = try l.stage.halves(books);
        }
        {
            const ow = try gpa.alloc(f32, codebooks * latent * codebook_dim);
            defer gpa.free(ow);
            const ob = try gpa.alloc(f32, codebooks * latent);
            defer gpa.free(ob);
            for (0..codebooks) |i| {
                const prefix = if (i == 0) "quantizer.semantic_quantizer.quantizers.0" else try l.name("quantizer.quantizer.quantizers.{d}", .{i - 1});
                const v = try l.folded(try l.name("{s}.out_proj.weight_g", .{prefix}), try l.name("{s}.out_proj.weight_v", .{prefix}), &[_]u64{ latent, codebook_dim, 1 });
                defer gpa.free(v);
                @memcpy(ow[i * latent * codebook_dim ..][0..v.len], v);
                const b = try l.read(try l.name("{s}.out_proj.bias", .{prefix}), &[_]u64{latent});
                defer gpa.free(b);
                @memcpy(ob[i * latent ..][0..latent], b);
            }
            w.out_w = try l.stage.halves(ow);
            w.out_b = try l.stage.halves(ob);
        }

        // post_module transformer
        for (&w.layers, 0..) |*layer, i| {
            const p = try l.name("quantizer.post_module.layers.{d}", .{i});
            layer.wqkv = try l.half(try l.name("{s}.attention.wqkv.weight", .{p}), &[_]u64{ 3 * latent, latent });
            layer.wo = try l.half(try l.name("{s}.attention.wo.weight", .{p}), &[_]u64{ latent, latent });
            {
                const w13 = try gpa.alloc(f32, 2 * ffn * latent);
                defer gpa.free(w13);
                const w1 = try l.read(try l.name("{s}.feed_forward.w1.weight", .{p}), &[_]u64{ ffn, latent });
                defer gpa.free(w1);
                const w3 = try l.read(try l.name("{s}.feed_forward.w3.weight", .{p}), &[_]u64{ ffn, latent });
                defer gpa.free(w3);
                @memcpy(w13[0 .. ffn * latent], w1);
                @memcpy(w13[ffn * latent ..], w3);
                layer.w13 = try l.stage.halves(w13);
            }
            layer.w2 = try l.half(try l.name("{s}.feed_forward.w2.weight", .{p}), &[_]u64{ latent, ffn });
            layer.attention_norm = try l.half(try l.name("{s}.attention_norm.weight", .{p}), &[_]u64{latent});
            layer.ffn_norm = try l.half(try l.name("{s}.ffn_norm.weight", .{p}), &[_]u64{latent});
            layer.attention_scale = try l.half(try l.name("{s}.attention_layer_scale.gamma", .{p}), &[_]u64{latent});
            layer.ffn_scale = try l.half(try l.name("{s}.ffn_layer_scale.gamma", .{p}), &[_]u64{latent});
        }
        w.norm = try l.half("quantizer.post_module.norm.weight", &[_]u64{latent});

        // upsample stages (plain convs: rvq.py's own CausalConvNet classes carry no weight norm)
        for (&w.upsample, 0..) |*u, i| {
            const p = try l.name("quantizer.upsample.{d}", .{i});
            {
                const v = try l.read(try l.name("{s}.0.conv.weight", .{p}), &[_]u64{ latent, latent, 2 });
                defer gpa.free(v);
                const t = try transposedTaps(gpa, v, latent, latent, 2, 2);
                defer gpa.free(t);
                u.conv = try l.stage.halves(t);
            }
            u.conv_b = try l.half(try l.name("{s}.0.conv.bias", .{p}), &[_]u64{latent});
            u.dw = try l.half(try l.name("{s}.1.dwconv.conv.weight", .{p}), &[_]u64{ latent, 1, 7 });
            u.dw_b = try l.half(try l.name("{s}.1.dwconv.conv.bias", .{p}), &[_]u64{latent});
            u.norm_w = try l.half(try l.name("{s}.1.norm.weight", .{p}), &[_]u64{latent});
            u.norm_b = try l.half(try l.name("{s}.1.norm.bias", .{p}), &[_]u64{latent});
            u.pw1 = try l.half(try l.name("{s}.1.pwconv1.weight", .{p}), &[_]u64{ convnext_hidden, latent });
            u.pw1_b = try l.half(try l.name("{s}.1.pwconv1.bias", .{p}), &[_]u64{convnext_hidden});
            u.pw2 = try l.half(try l.name("{s}.1.pwconv2.weight", .{p}), &[_]u64{ latent, convnext_hidden });
            u.pw2_b = try l.half(try l.name("{s}.1.pwconv2.bias", .{p}), &[_]u64{latent});
            u.gamma = try l.half(try l.name("{s}.1.gamma", .{p}), &[_]u64{latent});
        }

        // decoder
        w.conv0 = try l.decoderConv("decoder.model.0", decoder_dim, latent, 7);
        w.conv0_b = try l.half("decoder.model.0.conv.bias", &[_]u64{decoder_dim});
        var ch: usize = decoder_dim;
        for (&w.blocks, rates, 0..) |*b, rate, i| {
            const out_ch = ch / 2;
            const p = try l.name("decoder.model.{d}.block", .{i + 1});
            b.snake = try l.snake(try l.name("{s}.0.alpha", .{p}), ch);
            {
                const v = try l.folded(
                    try l.name("{s}.1.conv.parametrizations.weight.original0", .{p}),
                    try l.name("{s}.1.conv.parametrizations.weight.original1", .{p}),
                    &[_]u64{ ch, out_ch, 2 * rate },
                );
                defer gpa.free(v);
                const t = try transposedTaps(gpa, v, ch, out_ch, rate, 2 * rate);
                defer gpa.free(t);
                b.conv = try l.stage.halves(t);
            }
            b.conv_b = try l.half(try l.name("{s}.1.conv.bias", .{p}), &[_]u64{out_ch});
            for (&b.units, 0..) |*u, j| {
                const q = try l.name("{s}.{d}.block", .{ p, j + 2 });
                u.snake1 = try l.snake(try l.name("{s}.0.alpha", .{q}), out_ch);
                u.conv7 = try l.decoderConv(try l.name("{s}.1", .{q}), out_ch, out_ch, 7);
                u.conv7_b = try l.half(try l.name("{s}.1.conv.bias", .{q}), &[_]u64{out_ch});
                u.snake2 = try l.snake(try l.name("{s}.2.alpha", .{q}), out_ch);
                u.conv1 = try l.decoderConv(try l.name("{s}.3", .{q}), out_ch, out_ch, 1);
                u.conv1_b = try l.half(try l.name("{s}.3.conv.bias", .{q}), &[_]u64{out_ch});
            }
            ch = out_ch;
        }
        const tail = rates.len + 1;
        w.out_snake = try l.snake(try l.name("decoder.model.{d}.alpha", .{tail}), out_channels);
        {
            const v = try l.folded(
                try l.name("decoder.model.{d}.conv.parametrizations.weight.original0", .{tail + 1}),
                try l.name("decoder.model.{d}.conv.parametrizations.weight.original1", .{tail + 1}),
                &[_]u64{ 1, out_channels, 7 },
            );
            defer gpa.free(v);
            // [1][96][7] -> [7][96], the rows the output kernel walks
            var t: [7 * out_channels]f32 = undefined;
            for (0..out_channels) |c| {
                for (0..7) |tap| t[tap * out_channels + c] = v[c * 7 + tap];
            }
            w.out_conv = try l.stage.halves(&t);
        }
        w.out_conv_b = try l.half(try l.name("decoder.model.{d}.conv.bias", .{tail + 1}), &[_]u64{1});

        const rope = try ropeTable(gpa, max_frames);
        defer gpa.free(rope);
        w.rope = try l.stage.halves(rope);
        return w;
    }
};

test "transposed-conv taps reproduce ConvTranspose1d with the causal trim" {
    // cin 2, cout 1, stride 2, kernel 4: direct definition vs the GEMM's two-tap form
    const cin = 2;
    const cout = 1;
    const s = 2;
    const k = 4;
    const w = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 }; // [cin][cout][k]
    const x = [_][cin]f32{ .{ 1, -1 }, .{ 2, 0.5 }, .{ -3, 4 } }; // [L][cin]
    const taps = try transposedTaps(std.testing.allocator, &w, cin, cout, s, k);
    defer std.testing.allocator.free(taps);
    for (0..x.len) |q| {
        for (0..s) |j| {
            // reference: y[p] = sum over i, kk with p = i * s + kk, of x[i] * w[ci][0][kk]
            const pos = q * s + j;
            var want: f32 = 0;
            for (0..x.len) |i| {
                for (0..k) |kk| {
                    if (i * s + kk != pos) continue;
                    for (0..cin) |ci| want += x[i][ci] * w[ci * k + kk];
                }
            }
            var got: f32 = 0;
            for (0..2) |tap| {
                if (tap == 0 and q == 0) continue;
                const row = if (tap == 0) q - 1 else q;
                for (0..cin) |ci| got += x[row][ci] * taps[(j * cout * 2 + tap) * cin + ci];
            }
            try std.testing.expectEqual(want, got);
        }
    }
}

test "weight norm folds per output row and bf16 rounding ties to even" {
    var v = [_]f32{ 3, 4, 0, 5 };
    try foldWeightNorm(&v, &.{ 10, 2 });
    try std.testing.expectEqualSlices(f32, &.{ 6, 8, 0, 2 }, &v);
    // 1 + 2^-8 is halfway between bf16 neighbours 1 and 1 + 2^-7: ties to the even 1
    try std.testing.expectEqual(@as(f32, 1.0), bf16Round(1.0 + 1.0 / 256.0));
    try std.testing.expectEqual(@as(f32, 1.0 + 1.0 / 64.0), bf16Round(1.0 + 3.0 / 256.0));
}

test "weight norm preserves FP16 rounding across the CUDA reduction boundary" {
    // Synthetic values: an FP64 sum changes this folded weight by one FP16 ULP.
    var state: u32 = 689124668;
    var v: [257]f32 = undefined;
    for (&v) |*value| {
        state = state *% 1664525 +% 1013904223;
        value.* = @as(f32, @floatFromInt(state >> 8)) / 8388608.0 - 1.0;
    }
    try foldWeightNorm(&v, &.{1});
    try std.testing.expectEqual(@as(u16, 0xab13), halfBits(v[105]));
}
