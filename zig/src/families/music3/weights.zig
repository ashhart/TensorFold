//! The three Comfy-Org files into any store with alloc / upload / stage / unstage: the AR (bf16 or int8), DiT, DAV.
const std = @import("std");
const builtin = @import("builtin");
const core = @import("core");
const cfg = @import("config.zig");

const st = core.safetensors;
const H = cfg.hidden;

pub const text_encoder = "text_encoders/minimax_music3_text_encoder_pruned_bf16.safetensors";
pub const diffusion = "diffusion_models/minimax_music3_dit_fp16.safetensors";
pub const decoder = "vae/minimax_music3_dav.safetensors";

pub const Format = enum { bf16, int8 };

/// A projection W [n, k]: bf16 rows, or int8 rows with fp32 scales (sc != 0).
pub const Mat = struct { w: u64, sc: u64 = 0, n: usize, k: usize, group: usize = 0 }; // group: inputs a scale (0: a row)

pub const BodyLayer = struct { in_norm: u64, post_norm: u64, q_norm: u64, k_norm: u64, qkv: Mat, o: Mat, gate_up: Mat, down: Mat };
pub const DepthLayer = struct { in_norm: u64, post_norm: u64, qkv: Mat, o: Mat, gate_up: Mat, down: Mat };

pub const Ar = struct {
    format: Format,
    embed_prefill: u64, // [151675, 4096]
    embed_audio: u64, // [16384, 4096]
    extra: u64, // [7168, 4096]
    body: [cfg.layers]BodyLayer,
    norm: u64,
    lm_head: Mat, // [16385, 4096]
    projection: Mat,
    pos_emb: u64, // [16, 4096]
    depth: [cfg.depth_layers]DepthLayer,
    depth_norm: u64,
    heads: [cfg.codebooks - 1]Mat, // [1024, 4096]
    inv_freq: u64, // f32 [64]
    tokenizer_json: []u8,
};

pub const DitBlock = struct { pre_g: u64, pre_b: u64, ff_g: u64, ff_b: u64, qkv: u64, out: u64, ff_in: u64, ff_in_b: u64, ff_out: u64, ff_out_b: u64 };

pub const Dit = struct {
    cond_w8: [8]f32, // softmax(cond_layer_logits) rounded to fp16
    cond_scale: f32,
    cond_conv: u64, // fp16 [2048, 3 * 4096] (taps outer)
    cond_bias: u64, // fp16 [2048]
    pre: u64, // [2304, 2304]
    post: u64, // [128, 128]
    time_features: u64, // [128]
    time_in: u64,
    time_in_b: u64,
    time_out: u64,
    time_out_b: u64,
    project_in: u64, // [2048, 2304]
    project_out: u64, // [128, 2048]
    inv_freq: [16]f32,
    blocks: [cfg.dit_blocks]DitBlock,
};

/// A DAV conv: taps [K][Cout][Cin] fp32, bias [Cout]; transposed convs keep [K][Cout][Cin] of W[Cin][Cout][K].
pub const Conv = struct { w: u64, b: u64, cin: usize, cout: usize, k: usize };
pub const ResUnit = struct { a1: u64, c1: Conv, a2: u64, c2: Conv, dilation: usize };
pub const DavBlock = struct { alpha: u64, up: Conv, stride: usize, res: [3]ResUnit };
pub const Dav = struct { in_proj: Conv, conv0: Conv, blocks: [4]DavBlock, alpha_out: u64, conv_out: Conv };

fn need(f: *const st.File, name: []const u8, dtype: st.DType, shape: []const usize) !st.Tensor {
    const t = f.get(name) orelse {
        std.log.err("music3: {s} has no {s}", .{ f.path, name });
        return error.MissingTensor;
    };
    if (!t.is(dtype, shape)) {
        std.log.err("music3: {s} is {s} {any}, expected {s} {any}", .{ name, @tagName(t.dtype), t.shape[0..t.rank], @tagName(dtype), shape });
        return error.BadTensor;
    }
    return t;
}

/// Drop a tensor's pages from the page cache once it is on the device (unified memory holds both otherwise).
fn drop(f: *const st.File, name: []const u8) void {
    const e = f.names.get(name) orelse return;
    if (builtin.os.tag == .linux) {
        _ = std.os.linux.fadvise(f.file.handle, @intCast(f.data + e.begin), @intCast(e.end - e.begin), std.os.linux.POSIX_FADV.DONTNEED);
    }
}

/// The quantizer the AR loader calls for int8 projections: bf16 rows staged on the device, int8 + scales out.
pub const Quantizer = struct {
    ctx: *anyopaque,
    f: *const fn (ctx: *anyopaque, src: u64, n: usize, k: usize, group: usize, q: u64, sc: u64) anyerror!void,
};

fn ArLoader(comptime S: type) type {
    return struct {
        s: S,
        f: *const st.File,
        format: Format,
        quant: ?Quantizer,

        const L = @This();

        fn plain(l: *L, name: []const u8, shape: []const usize) !u64 {
            const t = try need(l.f, name, .bf16, shape);
            const p = try l.s.upload(t.bytes);
            drop(l.f, name);
            return p;
        }

        fn mat(l: *L, name: []const u8, n: usize, k: usize) !Mat {
            return l.matG(name, n, k, 0);
        }

        /// `group` inputs a scale (int8 only; 0 is a scale a row).
        fn matG(l: *L, name: []const u8, n: usize, k: usize, group: usize) !Mat {
            const t = try need(l.f, name, .bf16, &.{ n, k });
            defer drop(l.f, name);
            if (l.format == .bf16) return .{ .w = try l.s.upload(t.bytes), .n = n, .k = k };
            const src = try l.s.stage(t.bytes);
            const q = try l.s.alloc(n * k);
            const sc = try l.s.alloc(n * (if (group > 0) k / group else 1) * 4);
            try l.quant.?.f(l.quant.?.ctx, src, n, k, group, q, sc);
            return .{ .w = q, .sc = sc, .n = n, .k = k, .group = group };
        }
    };
}

/// `depth_group`: int8 groups of that many inputs a scale for the depth decoder and its heads (0: a scale a row).
pub fn loadAr(gpa: std.mem.Allocator, io: std.Io, s: anytype, dir: []const u8, format: Format, depth_group: usize, quant: ?Quantizer, log: bool) !Ar {
    var nb: [8][160]u8 = undefined;
    const path = try std.fs.path.join(gpa, &.{ dir, text_encoder });
    defer gpa.free(path);
    var f = try st.File.open(gpa, io, path);
    defer f.close(io);
    var a: Ar = undefined;
    a.format = format;
    const tj = f.get("tokenizer_json") orelse return error.NoTokenizer;
    a.tokenizer_json = try gpa.dupe(u8, tj.bytes);
    defer s.unstage();
    var l: ArLoader(@TypeOf(s)) = .{ .s = s, .f = &f, .format = format, .quant = quant };
    a.embed_prefill = try l.plain("model.embed_tokens_prefill.weight", &.{ cfg.prompt_vocab, H });
    a.embed_audio = try l.plain("model.embed_tokens_audio.weight", &.{ cfg.c0_codes, H });
    a.extra = try l.plain("model.audio_extra_embedding.weight", &.{ cfg.codebook_size * (cfg.codebooks - 1), H });
    for (0..cfg.layers) |i| {
        const p = "model.layers.{d}.";
        a.body[i] = .{
            .in_norm = try l.plain(try std.fmt.bufPrint(&nb[0], p ++ "input_layernorm.weight", .{i}), &.{H}),
            .post_norm = try l.plain(try std.fmt.bufPrint(&nb[1], p ++ "post_attention_layernorm.weight", .{i}), &.{H}),
            .q_norm = try l.plain(try std.fmt.bufPrint(&nb[2], p ++ "self_attn.q_norm.weight", .{i}), &.{cfg.head_dim}),
            .k_norm = try l.plain(try std.fmt.bufPrint(&nb[3], p ++ "self_attn.k_norm.weight", .{i}), &.{cfg.head_dim}),
            .qkv = try l.mat(try std.fmt.bufPrint(&nb[4], p ++ "self_attn.qkv_proj.weight", .{i}), cfg.qkv_width, H),
            .o = try l.mat(try std.fmt.bufPrint(&nb[5], p ++ "self_attn.o_proj.weight", .{i}), H, H),
            .gate_up = try l.mat(try std.fmt.bufPrint(&nb[6], p ++ "mlp.gate_up_proj.weight", .{i}), 2 * cfg.mlp, H),
            .down = try l.mat(try std.fmt.bufPrint(&nb[7], p ++ "mlp.down_proj.weight", .{i}), H, cfg.mlp),
        };
        if (log and i % 9 == 8) std.log.info("music3: AR layer {d} loaded ({d:.1} GB on the device)", .{ i, @as(f64, @floatFromInt(s.bytes)) / 1e9 });
    }
    a.norm = try l.plain("model.norm.weight", &.{H});
    a.lm_head = try l.mat("model.lm_head_pruned.weight", cfg.c0_codes + 1, H);
    a.projection = try l.matG("model.audio_decoder.projection.weight", H, H, depth_group);
    a.pos_emb = try l.plain("model.audio_decoder.pos_embedding.weight", &.{ 16, H });
    for (0..cfg.depth_layers) |i| {
        const p = "model.audio_decoder.layers.{d}.";
        a.depth[i] = .{
            .in_norm = try l.plain(try std.fmt.bufPrint(&nb[0], p ++ "input_layernorm.weight", .{i}), &.{H}),
            .post_norm = try l.plain(try std.fmt.bufPrint(&nb[1], p ++ "post_attention_layernorm.weight", .{i}), &.{H}),
            .qkv = try l.matG(try std.fmt.bufPrint(&nb[2], p ++ "self_attn.qkv_proj.weight", .{i}), 3 * H, H, depth_group),
            .o = try l.matG(try std.fmt.bufPrint(&nb[3], p ++ "self_attn.o_proj.weight", .{i}), H, H, depth_group),
            .gate_up = try l.matG(try std.fmt.bufPrint(&nb[4], p ++ "mlp.gate_up_proj.weight", .{i}), 2 * cfg.depth_mlp, H, depth_group),
            .down = try l.matG(try std.fmt.bufPrint(&nb[5], p ++ "mlp.down_proj.weight", .{i}), H, cfg.depth_mlp, depth_group),
        };
    }
    a.depth_norm = try l.plain("model.audio_decoder.norm.weight", &.{H});
    for (0..cfg.codebooks - 1) |i| a.heads[i] = try l.matG(try std.fmt.bufPrint(&nb[0], "model.audio_decoder.audio_heads.{d}.weight", .{i}), cfg.codebook_size, H, depth_group);
    // rope: inv_freq_i = 1 / theta^(2i / 128) in float32, as ComfyUI's precompute_freqs_cis
    var inv: [64]f32 = undefined;
    for (&inv, 0..) |*v, i| v.* = 1.0 / std.math.pow(f32, cfg.rope_theta, @as(f32, @floatFromInt(2 * i)) / 128.0);
    a.inv_freq = try s.upload(std.mem.sliceAsBytes(&inv));
    if (log) std.log.info("music3: AR weights on the device: {d:.2} GB ({s})", .{ @as(f64, @floatFromInt(s.bytes)) / 1e9, @tagName(format) });
    return a;
}

fn f16v(t: st.Tensor, i: usize) f32 {
    return @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, t.bytes[i * 2 ..][0..2], .little))));
}

pub fn loadDit(gpa: std.mem.Allocator, io: std.Io, s: anytype, dir: []const u8, log: bool) !Dit {
    const path = try std.fs.path.join(gpa, &.{ dir, diffusion });
    defer gpa.free(path);
    var f = try st.File.open(gpa, io, path);
    defer f.close(io);
    var d: Dit = undefined;
    const P = struct {
        fn up(sto: anytype, file: *const st.File, name: []const u8, shape: []const usize) !u64 {
            const t = try need(file, name, .f16, shape);
            const p = try sto.upload(t.bytes);
            drop(file, name);
            return p;
        }
    };
    {
        const t = try need(&f, "cond_layer_logits", .f16, &.{8});
        var m: f32 = -std.math.inf(f32);
        var v: [8]f32 = undefined;
        for (&v, 0..) |*x, i| {
            x.* = f16v(t, i);
            m = @max(m, x.*);
        }
        var sum: f32 = 0;
        for (&v) |*x| {
            x.* = @exp(x.* - m);
            sum += x.*;
        }
        for (&d.cond_w8, v) |*o, x| o.* = @floatCast(@as(f16, @floatCast(x / sum)));
        const sc = try need(&f, "cond_layer_scale", .f16, &.{1});
        d.cond_scale = f16v(sc, 0);
    }
    {
        // latent_conditioners.0: [2048, 4096, 3] -> [2048][3][4096] for the im2col GEMM
        const t = try need(&f, "latent_conditioners.0.weight", .f16, &.{ 2048, 4096, 3 });
        const host = try gpa.alloc(u16, 2048 * 3 * 4096);
        defer gpa.free(host);
        const src = std.mem.bytesAsSlice(u16, t.bytes);
        for (0..2048) |o| for (0..4096) |c| for (0..3) |k| {
            host[(o * 3 + k) * 4096 + c] = std.mem.littleToNative(u16, src[(o * 4096 + c) * 3 + k]);
        };
        d.cond_conv = try s.upload(std.mem.sliceAsBytes(host));
        drop(&f, "latent_conditioners.0.weight");
        d.cond_bias = try P.up(s, &f, "latent_conditioners.0.bias", &.{2048});
    }
    const pre = "diffusion_transformer.";
    d.pre = try P.up(s, &f, pre ++ "preprocess_conv.weight", &.{ 2304, 2304, 1 });
    d.post = try P.up(s, &f, pre ++ "postprocess_conv.weight", &.{ 128, 128, 1 });
    d.time_features = try P.up(s, &f, pre ++ "timestep_features.weight", &.{ 128, 1 });
    d.time_in = try P.up(s, &f, pre ++ "to_timestep_embed.0.weight", &.{ 2048, 256 });
    d.time_in_b = try P.up(s, &f, pre ++ "to_timestep_embed.0.bias", &.{2048});
    d.time_out = try P.up(s, &f, pre ++ "to_timestep_embed.2.weight", &.{ 2048, 2048 });
    d.time_out_b = try P.up(s, &f, pre ++ "to_timestep_embed.2.bias", &.{2048});
    d.project_in = try P.up(s, &f, pre ++ "transformer.project_in.weight", &.{ 2048, 2304 });
    d.project_out = try P.up(s, &f, pre ++ "transformer.project_out.weight", &.{ 128, 2048 });
    {
        const t = try need(&f, pre ++ "transformer.rotary_pos_emb.inv_freq", .f16, &.{16});
        for (&d.inv_freq, 0..) |*v, i| v.* = f16v(t, i);
    }
    var nb: [10][160]u8 = undefined;
    for (0..cfg.dit_blocks) |i| {
        const p = pre ++ "transformer.layers.{d}.";
        d.blocks[i] = .{
            .pre_g = try P.up(s, &f, try std.fmt.bufPrint(&nb[0], p ++ "pre_norm.gamma", .{i}), &.{2048}),
            .pre_b = try P.up(s, &f, try std.fmt.bufPrint(&nb[1], p ++ "pre_norm.beta", .{i}), &.{2048}),
            .ff_g = try P.up(s, &f, try std.fmt.bufPrint(&nb[2], p ++ "ff_norm.gamma", .{i}), &.{2048}),
            .ff_b = try P.up(s, &f, try std.fmt.bufPrint(&nb[3], p ++ "ff_norm.beta", .{i}), &.{2048}),
            .qkv = try P.up(s, &f, try std.fmt.bufPrint(&nb[4], p ++ "self_attn.to_qkv.weight", .{i}), &.{ 6144, 2048 }),
            .out = try P.up(s, &f, try std.fmt.bufPrint(&nb[5], p ++ "self_attn.to_out.weight", .{i}), &.{ 2048, 2048 }),
            .ff_in = try P.up(s, &f, try std.fmt.bufPrint(&nb[6], p ++ "ff.ff.0.proj.weight", .{i}), &.{ 16384, 2048 }),
            .ff_in_b = try P.up(s, &f, try std.fmt.bufPrint(&nb[7], p ++ "ff.ff.0.proj.bias", .{i}), &.{16384}),
            .ff_out = try P.up(s, &f, try std.fmt.bufPrint(&nb[8], p ++ "ff.ff.2.weight", .{i}), &.{ 2048, 8192 }),
            .ff_out_b = try P.up(s, &f, try std.fmt.bufPrint(&nb[9], p ++ "ff.ff.2.bias", .{i}), &.{2048}),
        };
    }
    if (log) std.log.info("music3: DiT on the device (fp16)", .{});
    return d;
}

/// A weight-normed conv: w = g v / |v| over dims 1, 2 (f64), repacked to [K][Cout][Cin]; transposed v is [I, O, K].
fn davConv(gpa: std.mem.Allocator, s: anytype, f: *const st.File, base: []const u8, transposed: bool, normed: bool) !Conv {
    var nb: [3][200]u8 = undefined;
    const vname = try std.fmt.bufPrint(&nb[0], "{s}.{s}", .{ base, if (normed) "weight_v" else "weight" });
    const v = f.get(vname) orelse {
        std.log.err("music3: DAV has no {s}", .{vname});
        return error.MissingTensor;
    };
    if (v.dtype != .f32 or v.rank != 3) return error.BadTensor;
    const d0 = v.shape[0];
    const d1 = v.shape[1];
    const K = v.shape[2];
    const cin = if (transposed) d0 else d1;
    const cout = if (transposed) d1 else d0;
    const vs = std.mem.bytesAsSlice(f32, v.bytes);
    var gs: ?[]align(1) const f32 = null;
    if (normed) {
        const g = f.get(try std.fmt.bufPrint(&nb[1], "{s}.weight_g", .{base})) orelse return error.MissingTensor;
        gs = std.mem.bytesAsSlice(f32, g.bytes);
    }
    const out = try gpa.alloc(f32, K * cout * cin);
    defer gpa.free(out);
    for (0..d0) |a| {
        var scale: f64 = 1;
        if (gs) |g| {
            var ss: f64 = 0;
            for (0..d1 * K) |j| {
                const x: f64 = vs[a * d1 * K + j];
                ss += x * x;
            }
            scale = @as(f64, g[a]) / @sqrt(ss);
        }
        for (0..d1) |b| for (0..K) |k| {
            const w: f32 = @floatCast(@as(f64, vs[(a * d1 + b) * K + k]) * scale);
            const o = if (transposed) b else a;
            const i = if (transposed) a else b;
            out[(k * cout + o) * cin + i] = w;
        };
    }
    const bname = try std.fmt.bufPrint(&nb[2], "{s}.bias", .{base});
    const bt = f.get(bname) orelse return error.MissingTensor;
    if (bt.dtype != .f32 or bt.numel() != cout) return error.BadTensor;
    return .{ .w = try s.upload(std.mem.sliceAsBytes(out)), .b = try s.upload(bt.bytes), .cin = cin, .cout = cout, .k = K };
}

pub fn loadDav(gpa: std.mem.Allocator, io: std.Io, s: anytype, dir: []const u8) !Dav {
    const path = try std.fs.path.join(gpa, &.{ dir, decoder });
    defer gpa.free(path);
    var f = try st.File.open(gpa, io, path);
    defer f.close(io);
    var d: Dav = undefined;
    d.in_proj = try davConv(gpa, s, &f, "dec_in_proj", false, false);
    d.conv0 = try davConv(gpa, s, &f, "decoder.model.0", false, true);
    const strides = [_]usize{ 8, 8, 4, 2 };
    const dil = [_]usize{ 1, 3, 9 };
    var nb: [6][160]u8 = undefined;
    for (0..4) |bi| {
        const B = &d.blocks[bi];
        B.stride = strides[bi];
        B.alpha = try alpha(s, &f, try std.fmt.bufPrint(&nb[0], "decoder.model.{d}.block.0.alpha", .{bi + 1}));
        B.up = try davConv(gpa, s, &f, try std.fmt.bufPrint(&nb[1], "decoder.model.{d}.block.1", .{bi + 1}), true, true);
        for (0..3) |r| {
            const p = "decoder.model.{d}.block.{d}.block.";
            B.res[r] = .{
                .dilation = dil[r],
                .a1 = try alpha(s, &f, try std.fmt.bufPrint(&nb[2], p ++ "0.alpha", .{ bi + 1, r + 2 })),
                .c1 = try davConv(gpa, s, &f, try std.fmt.bufPrint(&nb[3], p ++ "1", .{ bi + 1, r + 2 }), false, true),
                .a2 = try alpha(s, &f, try std.fmt.bufPrint(&nb[4], p ++ "2.alpha", .{ bi + 1, r + 2 })),
                .c2 = try davConv(gpa, s, &f, try std.fmt.bufPrint(&nb[5], p ++ "3", .{ bi + 1, r + 2 }), false, true),
            };
        }
    }
    d.alpha_out = try alpha(s, &f, "decoder.model.5.alpha");
    d.conv_out = try davConv(gpa, s, &f, "decoder.model.6", false, true);
    return d;
}

fn alpha(s: anytype, f: *const st.File, name: []const u8) !u64 {
    const t = f.get(name) orelse {
        std.log.err("music3: DAV has no {s}", .{name});
        return error.MissingTensor;
    };
    if (t.dtype != .f32) return error.BadTensor;
    return s.upload(t.bytes);
}
