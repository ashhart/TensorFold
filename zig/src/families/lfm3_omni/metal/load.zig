//! The Metal engine's weights from the checkpoint and its device buffers, sized by its options.
const std = @import("std");
const st = @import("core").safetensors;
const stack = @import("stack.zig");
const melmod = @import("../mel.zig");
const cfg = @import("../config.zig");
const common = @import("../common.zig");
const Engine = @import("engine.zig").Engine;

const Arena = stack.Arena;
const hist_cap = common.hist_cap;

pub fn model(e: *Engine, dir: []const u8) !void {
    const gpa = e.gpa;
    const mpath = try std.fs.path.join(gpa, &.{ dir, "model.safetensors" });
    defer gpa.free(mpath);
    var f = try st.File.openPrefixes(gpa, e.io, mpath, &.{ "model.audio_encoder.", "model.audio_projector.", "model.language_model.", "model.codebook_embedding.", "model.depth_decoder." });
    defer f.close(e.io);
    const a = &e.a;
    const max_tokens = melmod.tokens(melmod.frames(e.o.max_input_s * melmod.sample_rate));
    e.enc = try stack.load(gpa, a, &f, cfg.encoderSpec(max_tokens));
    e.bb = try stack.load(gpa, a, &f, cfg.backboneSpec(e.o.prefill_rows, e.o.max_ctx));
    e.depth = try stack.load(gpa, a, &f, cfg.depthSpec());
    const p = "model.audio_encoder.downsample.";
    const C = cfg.conv_channels;
    e.ds = .{
        .conv_in_w = try up(a, &f, p ++ "conv_in.weight", &.{ C, 1, 3, 3 }),
        .conv_in_b = try up(a, &f, p ++ "conv_in.bias", &.{C}),
        .conv_mid_w = try up(a, &f, p ++ "conv_mid.weight", &.{ C, 1, 3, 3 }),
        .conv_mid_b = try up(a, &f, p ++ "conv_mid.bias", &.{C}),
        .lin_mid_w = try up(a, &f, p ++ "lin_mid.weight", &.{ C, C, 1, 1 }),
        .lin_mid_b = try up(a, &f, p ++ "lin_mid.bias", &.{C}),
        .conv_end_w = try up(a, &f, p ++ "conv_end.weight", &.{ C, 1, 3, 3 }),
        .conv_end_b = try up(a, &f, p ++ "conv_end.bias", &.{C}),
        .lin_end_w = try up(a, &f, p ++ "lin_end.weight", &.{ C, C, 1, 1 }),
        .lin_end_b = try up(a, &f, p ++ "lin_end.bias", &.{C}),
        .lin_out_w = try up(a, &f, p ++ "lin_out.weight", &.{ cfg.enc_d, C * cfg.mel_bins / 8, 1 }),
        .lin_out_b = try up(a, &f, p ++ "lin_out.bias", &.{cfg.enc_d}),
    };
    const q = "model.audio_projector.";
    e.ln_w = try up(a, &f, q ++ "layer_norm.weight", &.{cfg.enc_d});
    e.ln_b = try up(a, &f, q ++ "layer_norm.bias", &.{cfg.enc_d});
    e.l1_w = try up(a, &f, q ++ "linear_1.weight", &.{ cfg.hidden, cfg.enc_d });
    e.l1_b = try up(a, &f, q ++ "linear_1.bias", &.{cfg.hidden});
    e.l2_w = try up(a, &f, q ++ "linear_2.weight", &.{ cfg.hidden, cfg.hidden });
    e.l2_b = try up(a, &f, q ++ "linear_2.bias", &.{cfg.hidden});
    e.embed = try up(a, &f, "model.language_model.embed_tokens.weight", &.{ cfg.vocab, cfg.hidden });
    e.cb_embed = try up(a, &f, "model.codebook_embedding.weight", &.{ cfg.codebooks * cfg.codebook_size, cfg.hidden });
    const dd = "model.depth_decoder.";
    e.depth_w = try up(a, &f, dd ++ "depth_linear.weight", &.{ cfg.codebooks * cfg.depth_d, cfg.hidden });
    e.depth_b = try up(a, &f, dd ++ "depth_linear.bias", &.{cfg.codebooks * cfg.depth_d});
    e.mask_emb = try up(a, &f, dd ++ "mask_embedding", &.{cfg.depth_d});
    const tbytes = cfg.codebook_size * cfg.depth_d * 2;
    e.depth_tables = try a.alloc(cfg.codebooks * tbytes);
    e.depth_norms = try a.alloc(cfg.codebooks * cfg.depth_d * 2);
    var name: [128]u8 = undefined;
    for (0..cfg.codebooks) |i| {
        a.write(e.depth_tables, i * tbytes, try stack.tensor(&f, try std.fmt.bufPrint(&name, dd ++ "depth_embeddings.{d}.embedding.weight", .{i}), &.{ cfg.codebook_size, cfg.depth_d }));
        a.write(e.depth_norms, i * cfg.depth_d * 2, try stack.tensor(&f, try std.fmt.bufPrint(&name, dd ++ "depth_embeddings.{d}.embedding_norm.weight", .{i}), &.{cfg.depth_d}));
    }
    f.release();

    const cpath = try std.fs.path.join(gpa, &.{ dir, "audio_tokenizer", "model.safetensors" });
    defer gpa.free(cpath);
    var c = try st.File.openPrefixes(gpa, e.io, cpath, &.{ "decoder.", "quantizer.", "from_quantizer." });
    defer c.close(e.io);
    e.codec = try stack.load(gpa, a, &c, cfg.codecSpec(6 * e.o.codec_chunk, 6 * e.o.max_frames + 6 * e.o.codec_chunk));
    e.po_w = try up(a, &c, "quantizer.project_out.weight", &.{ cfg.quantizer_dim, cfg.codebooks * 4 });
    e.po_b = try up(a, &c, "quantizer.project_out.bias", &.{cfg.quantizer_dim});
    e.fq_w = try up(a, &c, "from_quantizer.weight", &.{ cfg.codec_d, cfg.quantizer_dim });
    e.fq_b = try up(a, &c, "from_quantizer.bias", &.{cfg.codec_d});
    e.lin_w = try up(a, &c, "decoder.lin.weight", &.{ 2 * cfg.spec_bins, cfg.codec_d });
    e.lin_b = try up(a, &c, "decoder.lin.bias", &.{2 * cfg.spec_bins});
    const wt = c.get("decoder.istft.window") orelse return error.MissingTensor;
    if (!wt.is(.f32, &.{cfg.n_fft})) return error.BadTensor;
    @memcpy(std.mem.asBytes(&e.window), wt.bytes);
    const basis = try common.synthesisBasis(gpa, &e.window);
    defer gpa.free(basis);
    e.basis = try a.upload(std.mem.sliceAsBytes(basis));
    c.release();
}

fn up(a: *Arena, f: *const st.File, name: []const u8, shape: []const usize) !u64 {
    return a.upload(try stack.tensor(f, name, shape));
}

pub fn scratch(e: *Engine) !void {
    const a = &e.a;
    const F = melmod.frames(e.o.max_input_s * melmod.sample_rate);
    const T = melmod.tokens(F);
    const C = cfg.conv_channels;
    const w1 = (F + 1) / 2;
    const w2 = (w1 + 1) / 2;
    e.mel_f32 = try a.alloc(cfg.mel_bins * F * 4);
    e.mel_pcm = try a.alloc((e.o.max_input_s * melmod.sample_rate + melmod.n_fft) * 4);
    e.mel_fr = try a.alloc(F * melmod.n_fft * 4);
    e.mel_s = try a.alloc(F * 2 * melmod.bins * 4);
    e.mel_p = try a.alloc(F * melmod.bins * 4);
    e.mel_m = try a.alloc(F * melmod.mels * 4);
    {
        const gpa = e.gpa;
        const win = try gpa.alloc(f32, melmod.n_fft);
        defer gpa.free(win);
        for (win, e.mel.window) |*w, x| w.* = @floatCast(x);
        e.mel_win = try a.upload(std.mem.sliceAsBytes(win));
        const dft = try gpa.alloc(f32, melmod.n_fft * 2 * melmod.bins);
        defer gpa.free(dft);
        for (0..melmod.n_fft) |n| for (0..melmod.bins) |kk| {
            dft[n * 2 * melmod.bins + kk] = @floatCast(e.mel.cos[n][kk]);
            dft[n * 2 * melmod.bins + melmod.bins + kk] = @floatCast(e.mel.sin[n][kk]);
        };
        e.mel_dft = try a.upload(std.mem.sliceAsBytes(dft));
        e.mel_fb = try a.upload(std.mem.sliceAsBytes(&e.mel.filters));
    }
    e.dframe = try a.alloc((3 * cfg.codebooks + 2) * 4);
    e.hist = try a.zeroed(hist_cap * cfg.codebooks * 4);
    e.mel_bf = try a.alloc(cfg.mel_bins * F * 2);
    e.a1 = try a.alloc(C * 64 * w1 * 2);
    e.a2 = try a.alloc(C * 32 * w2 * 2);
    e.a3 = try a.alloc(C * 32 * w2 * 2);
    e.acc = try a.alloc(@max(C * 32 * w2, T * cfg.hidden) * 4);
    e.enc_x = try a.alloc(T * cfg.enc_d * 2);
    e.enc_out = try a.alloc(T * cfg.enc_d * 2);
    e.ln_out = try a.alloc(T * cfg.enc_d * 2);
    e.h1 = try a.alloc(T * cfg.hidden * 2);
    e.audio_emb = try a.alloc(T * cfg.hidden * 2);
    const R = e.o.prefill_rows;
    e.ids = try a.alloc(R * 4);
    e.bb_x = try a.alloc(R * cfg.hidden * 2);
    e.bb_out = try a.alloc(R * cfg.hidden * 2);
    e.logits = try a.alloc(cfg.vocab * 2);
    e.pick = try a.alloc(4);
    e.amax_ws = try a.alloc(8 * (cfg.vocab / 1024 + 1));
    const sup = [_]i32{ cfg.audio_token_id, cfg.audio_out_token_id, cfg.image_token_id, cfg.video_token_id };
    e.sup = try a.upload(std.mem.sliceAsBytes(&sup));
    const NQ = cfg.codebooks;
    e.din_f32 = try a.alloc(NQ * cfg.depth_d * 4);
    e.din = try a.alloc(NQ * cfg.depth_d * 2);
    e.dx = try a.alloc(NQ * cfg.depth_d * 2);
    e.dout = try a.alloc(NQ * cfg.depth_d * 2);
    e.dz = try a.alloc(NQ * cfg.depth_d * 2);
    e.dlog = try a.alloc(NQ * cfg.codebook_size * 2);
    e.dlog_f32 = try a.alloc(NQ * cfg.codebook_size * 4);
    e.dcodes = try a.alloc(NQ * 4);
    e.dmask = try a.alloc(NQ * 4);
    const CC = e.o.codec_chunk;
    e.c_codes = try a.alloc(NQ * CC * 4);
    e.c_lat = try a.alloc(CC * NQ * 4 * 2);
    e.c_p1 = try a.alloc(CC * cfg.quantizer_dim * 2);
    e.c_p2 = try a.alloc(CC * cfg.codec_d * 2);
    e.c_up = try a.alloc(6 * CC * cfg.codec_d * 2);
    e.c_out = try a.alloc(6 * CC * cfg.codec_d * 2);
    e.c_f32 = try a.alloc(6 * CC * 2 * cfg.spec_bins * 4);
    e.c_lin = try a.alloc(6 * CC * 2 * cfg.spec_bins * 2);
    e.c_spec = try a.alloc(6 * CC * 2 * cfg.spec_bins * 4);
    e.c_frames = try a.alloc(6 * CC * cfg.n_fft * 4);
    e.host_frames = try e.gpa.alloc(f32, 6 * CC * cfg.n_fft);
    e.host_logits = try e.gpa.alloc(f32, NQ * cfg.codebook_size);
}
