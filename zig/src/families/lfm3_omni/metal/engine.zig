//! LFM3-Omni-2B on Metal in bf16 as released, one request at a time, with the CUDA engine's public calls.
const std = @import("std");
const metallib = @import("lfm3_omni_metallib");
const gpu = @import("gpu.zig");
const kern = @import("kernels.zig");
const stack = @import("stack.zig");
const load = @import("load.zig");
const melmod = @import("../mel.zig");
const cfg = @import("../config.zig");
const common = @import("../common.zig");

const Stack = stack.Stack;
const Arena = stack.Arena;
const Gpu = gpu.Gpu;

pub const Options = struct {
    /// Longest input clip (seconds of 16 kHz audio) the encoder's buffers take.
    max_input_s: usize = 60,
    /// Backbone rows: the window (prompt and reply) and a prompt chunk.
    max_ctx: usize = 4096,
    prefill_rows: usize = 512,
    /// Longest reply in speech frames (12.5 a second) and the codec's largest chunk.
    max_frames: usize = 1024,
    codec_chunk: usize = 64,
    /// Accepted for the CUDA engine's flags; Metal encodes each row's dispatches into one command buffer.
    graphs: bool = true,
};

pub const Sampling = common.Sampling;
pub const Sink = common.Sink;
pub const Streaming = common.Streaming;
pub const Forced = common.Forced;
pub const Stats = common.Stats;
pub const OverlapAdd = common.OverlapAdd;
pub const nowNs = common.nowNs;
const bf16 = common.bf16;
const pickCode = common.pickCode;

const Downsample = struct {
    conv_in_w: u64,
    conv_in_b: u64,
    conv_mid_w: u64,
    conv_mid_b: u64,
    lin_mid_w: u64,
    lin_mid_b: u64,
    conv_end_w: u64,
    conv_end_b: u64,
    lin_end_w: u64,
    lin_end_b: u64,
    lin_out_w: u64,
    lin_out_b: u64,
};

/// The CUDA engine's stream, here the device's open command buffer: `synchronize` waits for the GPU.
pub const Stream = struct {
    g: *Gpu,
    pub fn synchronize(s: Stream) !void {
        try s.g.flush();
    }
};

pub const Engine = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    o: Options,
    g: *Gpu,
    stream: Stream,
    k: kern.Kernels,
    a: Arena,
    mel: *melmod.Mel,
    window: [cfg.n_fft]f32,

    enc: Stack,
    bb: Stack,
    depth: Stack,
    codec: Stack,
    ds: Downsample,
    // projector
    ln_w: u64,
    ln_b: u64,
    l1_w: u64,
    l1_b: u64,
    l2_w: u64,
    l2_b: u64,
    // backbone tables
    embed: u64,
    cb_embed: u64,
    // depth decoder
    depth_w: u64,
    depth_b: u64,
    mask_emb: u64,
    depth_tables: u64, // [12 * 2017, 1024]
    depth_norms: u64, // [12, 1024]
    // codec
    po_w: u64,
    po_b: u64,
    fq_w: u64,
    fq_b: u64,
    lin_w: u64,
    lin_b: u64,
    basis: u64, // f32 [1282, 1280]

    // the log-mel front end on the device
    mel_pcm: u64,
    mel_win: u64,
    mel_dft: u64,
    mel_fb: u64,
    mel_fr: u64,
    mel_s: u64,
    mel_p: u64,
    mel_m: u64,
    // the device's frame: order | codes | masked (i32 [3 * 12]) and the repetition window's frames (a ring)
    dframe: u64,
    hist: u64,
    // scratch
    mel_f32: u64,
    mel_bf: u64,
    a1: u64,
    a2: u64,
    a3: u64,
    acc: u64,
    enc_x: u64,
    enc_out: u64,
    ln_out: u64,
    h1: u64,
    audio_emb: u64,
    audio_tokens: usize = 0,
    /// The newest backbone row's hidden state (bf16 on the device).
    last: u64 = 0,
    ids: u64,
    bb_x: u64,
    bb_out: u64,
    logits: u64,
    pick: u64,
    sup: u64,
    amax_ws: u64, // the greedy pick's parts
    din_f32: u64,
    din: u64,
    dx: u64,
    dout: u64,
    dz: u64,
    dlog: u64,
    dlog_f32: u64,
    dcodes: u64,
    dmask: u64,
    c_codes: u64,
    c_lat: u64,
    c_p1: u64,
    c_p2: u64,
    c_up: u64,
    c_out: u64,
    c_f32: u64,
    c_lin: u64,
    c_spec: u64,
    c_frames: u64,
    host_frames: []f32,
    host_logits: []f32,
    /// LFM3_OMNI_PROFILE=1: every phase waited for alone, its GPU time summed (logged after each reply).
    prof: bool = false,
    prof_s: [4]f64 = @splat(0),
    prof_n: [4]usize = @splat(0),

    pub const Phase = enum { row, frame, text, codec };

    /// Ends a phase begun when the device's GPU time was `t0` (profile only).
    pub fn phase(e: *Engine, p: Phase, t0: f64) !void {
        if (!e.prof) return;
        try e.g.flush();
        e.prof_s[@backingInt(p)] += e.g.gpu_s - t0;
        e.prof_n[@backingInt(p)] += 1;
    }

    pub fn create(gpa: std.mem.Allocator, io: std.Io, model_dir: []const u8, o: Options) !*Engine {
        const e = try gpa.create(Engine);
        errdefer gpa.destroy(e);
        e.* = undefined;
        e.gpa = gpa;
        e.io = io;
        e.o = o;
        e.audio_tokens = 0;
        e.last = 0;
        e.prof = if (std.c.getenv("LFM3_OMNI_PROFILE")) |v| !std.mem.eql(u8, std.mem.span(v), "0") else false;
        e.prof_s = @splat(0);
        e.prof_n = @splat(0);
        try cfg.check(gpa, io, model_dir);
        e.g = try Gpu.open(gpa, metallib.bytes);
        errdefer e.g.close();
        e.stream = .{ .g = e.g };
        e.k = .{ .g = e.g };
        e.a = .{ .g = e.g };
        e.mel = try gpa.create(melmod.Mel);
        errdefer gpa.destroy(e.mel);
        e.mel.init();
        try load.model(e, model_dir);
        try load.scratch(e);
        return e;
    }

    pub fn destroy(e: *Engine) void {
        e.g.flush() catch {};
        e.gpa.free(e.host_frames);
        e.gpa.free(e.host_logits);
        for ([_]*Stack{ &e.enc, &e.bb, &e.depth, &e.codec }) |s| e.gpa.free(s.layers);
        e.g.close();
        e.gpa.destroy(e.mel);
        e.gpa.destroy(e);
    }

    pub fn deviceBytes(e: *const Engine) usize {
        return e.g.bytes;
    }


    // ---- host copies (unified memory: the GPU's work so far finishes first)

    pub fn h2d(e: *Engine, dst: u64, bytes: []const u8) !void {
        if (bytes.len % 4 == 0 and bytes.len <= 4096 and bytes.len > 0) return e.g.set(dst, bytes);
        try e.g.put(dst, bytes);
    }

    pub fn d2h(e: *Engine, out: []u8, src: u64) !void {
        try e.g.get(src, out);
    }

    fn d2d(e: *Engine, dst: u64, src: u64, n: usize) void {
        e.g.copy(src, dst, n);
    }

    // ---- speech in

    /// Log-mel features of 16 kHz `pcm` into `out` (host, [128, frames]).
    pub fn features(e: *Engine, pcm: []const f32, out: []f32) void {
        e.mel.compute(pcm, out);
    }

    /// Log-mel features of 16 kHz `pcm` on the device (into `mel_f32`, [128, frames]); returns the frames.
    pub fn featuresDevice(e: *Engine, pcm: []const f32) !usize {
        const k = &e.k;
        const nf = melmod.frames(pcm.len);
        if (nf > melmod.frames(e.o.max_input_s * melmod.sample_rate)) return error.InputTooLong;
        try e.g.put(e.mel_pcm, std.mem.sliceAsBytes(pcm));
        k.melFrames(e.mel_pcm, pcm.len, e.mel_win, e.mel_fr, nf, melmod.n_fft, melmod.hop);
        k.matmul(false, false, nf, 2 * melmod.bins, melmod.n_fft, e.mel_fr, e.mel_dft, e.mel_s, .f32, .f32);
        k.power(e.mel_s, e.mel_p, nf, melmod.bins);
        k.matmul(false, false, nf, melmod.mels, melmod.bins, e.mel_p, e.mel_fb, e.mel_m, .f32, .f32);
        k.logmel(e.mel_m, e.mel_f32, nf, melmod.mels, -12.0);
        return nf;
    }

    /// The speech encoder and projector over features [128, nf] (null: `featuresDevice`'s), a row of `audio_emb` per 8.
    pub fn encode(e: *Engine, feats: ?[]const f32, nf: usize) !usize {
        const k = &e.k;
        const C = cfg.conv_channels;
        if (nf > melmod.frames(e.o.max_input_s * melmod.sample_rate)) return error.InputTooLong;
        if (feats) |fh| try e.g.put(e.mel_f32, std.mem.sliceAsBytes(fh));
        k.f32ToBf16(e.mel_f32, e.mel_bf, cfg.mel_bins * nf);
        const w1 = (nf + 1) / 2;
        const w2 = (w1 + 1) / 2;
        const w3 = (w2 + 1) / 2;
        k.conv2dS2(e.mel_bf, 1, cfg.mel_bins, nf, e.ds.conv_in_w, e.ds.conv_in_b, e.a1, C, 64, w1, true);
        k.conv2dS2(e.a1, C, 64, w1, e.ds.conv_mid_w, e.ds.conv_mid_b, e.a2, C, 32, w2, false);
        k.matmul(false, false, C, 32 * w2, C, e.ds.lin_mid_w, e.a2, e.acc, .bf16, .f32);
        k.biasAct(e.acc, e.ds.lin_mid_b, e.a3, C, 32 * w2, true, .silu);
        k.conv2dS2(e.a3, C, 32, w2, e.ds.conv_end_w, e.ds.conv_end_b, e.a2, C, 16, w3, false);
        k.matmul(false, false, C, 16 * w3, C, e.ds.lin_end_w, e.a2, e.acc, .bf16, .f32);
        k.biasAct(e.acc, e.ds.lin_end_b, e.a1, C, 16 * w3, true, .silu);
        // [C * 16, w3] rows to tokens: Z [w3, 768] = A^T Wo^T
        k.matmul(true, true, w3, cfg.enc_d, C * 16, e.a1, e.ds.lin_out_w, e.acc, .bf16, .f32);
        k.biasAct(e.acc, e.ds.lin_out_b, e.enc_x, w3, cfg.enc_d, false, .none);
        e.enc.reset();
        try e.enc.forward(k, e.enc_x, w3, e.enc_out);
        k.layernorm(e.enc_out, e.ln_w, e.ln_b, e.ln_out, cfg.enc_d, w3, 1e-5);
        k.linearF32(e.l1_w, e.ln_out, cfg.enc_d, cfg.enc_d, cfg.hidden, e.acc, w3);
        k.biasAct(e.acc, e.l1_b, e.h1, w3, cfg.hidden, false, .gelu);
        k.linearF32(e.l2_w, e.h1, cfg.hidden, cfg.hidden, cfg.hidden, e.acc, w3);
        k.biasAct(e.acc, e.l2_b, e.audio_emb, w3, cfg.hidden, false, .none);
        e.audio_tokens = w3;
        return w3;
    }

    // ---- the backbone

    /// Prompt rows from an empty window, the audio rows from `audio_at` from the encoder; `hidden_out` gets every row.
    pub fn prefill(e: *Engine, ids: []const u32, audio_at: usize, hidden_out: ?[]f32) !void {
        const k = &e.k;
        const R = e.o.prefill_rows;
        e.bb.reset();
        var at: usize = 0;
        const ibuf = try e.gpa.alloc(i32, R);
        defer e.gpa.free(ibuf);
        const hb = try e.gpa.alloc(u16, R * cfg.hidden);
        defer e.gpa.free(hb);
        while (at < ids.len) {
            const t = @min(R, ids.len - at);
            for (0..t) |j| {
                const g = at + j;
                const audio = g >= audio_at and g < audio_at + e.audio_tokens;
                ibuf[j] = if (audio) -1 else @intCast(ids[g]);
            }
            // a chunk's ids in order with the previous chunk's work
            var w: usize = 0;
            while (w < t) : (w += 1024) e.g.set(e.ids + w * 4, std.mem.sliceAsBytes(ibuf[w..@min(t, w + 1024)]));
            k.embed(e.embed, e.ids, e.bb_x, t, cfg.hidden);
            const lo = @max(at, audio_at);
            const hi = @min(at + t, audio_at + e.audio_tokens);
            if (hi > lo) e.d2d(e.bb_x + (lo - at) * cfg.hidden * 2, e.audio_emb + (lo - audio_at) * cfg.hidden * 2, (hi - lo) * cfg.hidden * 2);
            try e.bb.forward(k, e.bb_x, t, e.bb_out);
            if (hidden_out) |ho| {
                try e.d2h(std.mem.sliceAsBytes(hb[0 .. t * cfg.hidden]), e.bb_out);
                for (0..t * cfg.hidden) |j| ho[at * cfg.hidden + j] = bf16(hb[j]);
            }
            e.last = e.bb_out + (t - 1) * cfg.hidden * 2;
            at += t;
        }
    }

    /// One backbone row from the embedding in `bb_x` row 0; the state lands in `bb_out` row 0. Committed, not waited.
    pub fn stepRow(e: *Engine) !void {
        const t0 = e.g.gpu_s;
        e.last = e.bb_out;
        try e.bb.forward(&e.k, e.bb_x, 1, e.bb_out);
        try e.g.commit();
        try e.phase(.row, t0);
    }

    /// The greedy text token from the last hidden state (placeholders excluded), left on the device in `pick`.
    pub fn textPick(e: *Engine) !u32 {
        const t0 = e.g.gpu_s;
        defer e.phase(.text, t0) catch {};
        e.k.linear(e.embed, e.last, cfg.hidden, cfg.hidden, cfg.vocab, e.logits, cfg.vocab, 1);
        e.k.argmax(e.logits, cfg.vocab, e.sup, 4, e.pick, e.amax_ws);
        var v: i32 = 0;
        try e.d2h(std.mem.asBytes(&v), e.pick);
        return @intCast(v);
    }

    /// Text logits (fp32 on the host) from the last hidden state.
    pub fn textLogits(e: *Engine, out: []f32) !void {
        e.k.linear(e.embed, e.last, cfg.hidden, cfg.hidden, cfg.vocab, e.logits, cfg.vocab, 1);
        const hb = try e.gpa.alloc(u16, cfg.vocab);
        defer e.gpa.free(hb);
        try e.d2h(std.mem.sliceAsBytes(hb), e.logits);
        for (out, hb) |*o, h| o.* = bf16(h);
    }

    /// Sets the newest hidden state from the host (bf16-exact fp32 values), for checks of the heads alone.
    pub fn setHidden(e: *Engine, h: []const f32) !void {
        var hb: [cfg.hidden]u16 = undefined;
        for (&hb, h) |*o, v| o.* = @intCast(@as(u32, @bitCast(v)) >> 16);
        try e.h2d(e.bb_out, std.mem.sliceAsBytes(&hb));
        e.last = e.bb_out;
    }

    pub fn lastHidden(e: *Engine, out: []f32) !void {
        var hb: [cfg.hidden]u16 = undefined;
        try e.d2h(std.mem.sliceAsBytes(&hb), e.last);
        for (out, hb) |*o, h| o.* = bf16(h);
    }

    // ---- the depth decoder

    /// The depth decoder's slots from the last hidden state (run once a frame).
    pub fn depthStart(e: *Engine) !void {
        e.k.linearF32(e.depth_w, e.last, cfg.hidden, cfg.hidden, cfg.codebooks * cfg.depth_d, e.din_f32, 1);
        e.k.biasAct(e.din_f32, e.depth_b, e.din, 1, cfg.codebooks * cfg.depth_d, false, .none);
    }

    /// One pass's logits [12, 2017] in `dlog` given the codes and the masked set at `codes` / `masked` on the device.
    fn depthLogits(e: *Engine, codes: u64, masked: u64) !void {
        const k = &e.k;
        const NQ = cfg.codebooks;
        k.depthIn(e.din, e.mask_emb, e.depth_tables, codes, masked, e.dx, NQ, cfg.depth_d, cfg.codebook_size);
        e.depth.reset();
        try e.depth.forward(k, e.dx, NQ, e.dout);
        k.rmsnormRows(e.dout, 0, e.depth_norms, cfg.depth_d, e.dz, cfg.depth_d, cfg.depth_d, cfg.depth_d, NQ, 1e-5, .head);
        k.linearBatched(e.depth_tables, cfg.codebook_size * cfg.depth_d, e.dz, cfg.depth_d, cfg.depth_d, cfg.codebook_size, e.dlog, cfg.codebook_size, 1, NQ);
    }

    /// One pass: every codebook's logits (host, fp32 [12, 2017]) given the known codes and the masked set.
    pub fn depthPass(e: *Engine, codes: *const [cfg.codebooks]i32, masked: *const [cfg.codebooks]bool) ![]f32 {
        var m: [cfg.codebooks]i32 = undefined;
        for (&m, masked) |*x, y| x.* = @intFromBool(y);
        e.g.set(e.dcodes, std.mem.sliceAsBytes(codes));
        e.g.set(e.dmask, std.mem.sliceAsBytes(&m));
        try e.depthLogits(e.dcodes, e.dmask);
        e.k.bf16ToF32(e.dlog, e.dlog_f32, cfg.codebooks * cfg.codebook_size);
        try e.d2h(std.mem.sliceAsBytes(e.host_logits), e.dlog_f32);
        return e.host_logits;
    }

    /// A speech frame's codes from the last hidden state: the codebooks revealed over the mask schedule's passes.
    pub fn sampleFrame(e: *Engine, smp: Sampling, rng: std.Random, history: []const [cfg.codebooks]i32) ![cfg.codebooks]i32 {
        try e.depthStart();
        var codes: [cfg.codebooks]i32 = @splat(0);
        var masked: [cfg.codebooks]bool = @splat(true);
        var order: [cfg.codebooks]usize = undefined;
        for (&order, 0..) |*x, i| x.* = i;
        if (!smp.identity_order) rng.shuffle(usize, &order);
        var at: usize = 0;
        for (cfg.maskgit_schedule) |n| {
            const logits = try e.depthPass(&codes, &masked);
            for (order[at .. at + n]) |q| codes[q] = pickCode(logits[q * cfg.codebook_size ..][0..cfg.codebook_size], q, smp, rng, history);
            for (order[at .. at + n]) |q| masked[q] = false;
            at += n;
        }
        return codes;
    }

    /// A speech frame picked on the device into `dframe`'s codes, pushed to `hist` at `slot` unless it ends speech.
    pub fn frameDevice(e: *Engine, smp: Sampling, order: *const [cfg.codebooks]i32, frame: usize, hn: usize, slot: usize) !void {
        const NQ = cfg.codebooks;
        const k = &e.k;
        var init: [3 * NQ]i32 = undefined;
        @memcpy(init[0..NQ], order);
        @memset(init[NQ .. 2 * NQ], 0);
        @memset(init[2 * NQ .. 3 * NQ], 1);
        e.g.set(e.dframe, std.mem.sliceAsBytes(&init));
        const codes = e.dframe + NQ * 4;
        const masked = e.dframe + 2 * NQ * 4;
        try e.depthStart();
        var at: usize = 0;
        for (cfg.maskgit_schedule) |n| {
            try e.depthLogits(codes, masked);
            k.depthPick(.{ .logits = e.dlog, .order = e.dframe, .p0 = at, .n = n, .hist = e.hist, .hn = hn, .penalty = smp.audio_repetition_penalty, .temperature = smp.audio_temperature, .seed = smp.seed, .frame = frame, .codes = codes, .masked = masked, .nq = NQ, .size = cfg.codebook_size });
            at += n;
        }
        k.frameEnd(codes, e.hist, NQ, cfg.audio_eos_code, slot, true);
    }

    // ---- the codec

    /// Decodes `frames` (each 12 codes) after those already decoded; windowed frames for the overlap-add (host).
    pub fn codecChunk(e: *Engine, frames: []const [cfg.codebooks]i32, lin_out: ?[]f32) ![]const f32 {
        const k = &e.k;
        const n = frames.len;
        if (n == 0) return &.{};
        if (n > e.o.codec_chunk) return error.CodecChunkTooLong;
        try e.h2d(e.c_codes, std.mem.sliceAsBytes(frames));
        k.fsqDecode(e.c_codes, e.c_lat, n, cfg.codebooks);
        k.linearF32(e.po_w, e.c_lat, cfg.codebooks * 4, cfg.codebooks * 4, cfg.quantizer_dim, e.c_f32, n);
        k.biasAct(e.c_f32, e.po_b, e.c_p1, n, cfg.quantizer_dim, false, .none);
        k.linearF32(e.fq_w, e.c_p1, cfg.quantizer_dim, cfg.quantizer_dim, cfg.codec_d, e.c_f32, n);
        k.biasAct(e.c_f32, e.fq_b, e.c_p2, n, cfg.codec_d, false, .none);
        const r = cfg.frame_stack * n;
        k.repeatRows(e.c_p2, e.c_up, n, cfg.codec_d, cfg.frame_stack);
        try e.codec.forward(k, e.c_up, r, e.c_out);
        k.linearF32(e.lin_w, e.c_out, cfg.codec_d, cfg.codec_d, 2 * cfg.spec_bins, e.c_f32, r);
        k.biasAct(e.c_f32, e.lin_b, e.c_lin, r, 2 * cfg.spec_bins, false, .none);
        if (lin_out) |lo| {
            const hb = try e.gpa.alloc(u16, r * 2 * cfg.spec_bins);
            defer e.gpa.free(hb);
            try e.d2h(std.mem.sliceAsBytes(hb), e.c_lin);
            for (lo[0..hb.len], hb) |*o, h| o.* = bf16(h);
        }
        k.polar(e.c_lin, e.c_spec, r, cfg.spec_bins);
        k.matmul(false, false, r, cfg.n_fft, 2 * cfg.spec_bins, e.c_spec, e.basis, e.c_frames, .f32, .f32);
        const out = e.host_frames[0 .. r * cfg.n_fft];
        try e.d2h(std.mem.sliceAsBytes(out), e.c_frames);
        return out;
    }

    pub fn codecReset(e: *Engine) !void {
        e.codec.reset();
    }

    /// A reply after `prefill`, until <|im_end|> in text or `max_new` slots (reply.zig).
    pub const reply = @import("reply.zig").reply;
};
