//! Transcribe's audio encoder on CUDA as NeMo's vLLM SALM path runs it; n samples give ceil((n / 160) / 8) rows.
const std = @import("std");
const cuda = @import("cuda");
pub const Config = @import("config.zig");
pub const mel = @import("mel.zig");
pub const sortformer = @import("sortformer.zig");
pub const Dump = @import("dump.zig").Dump;
const W = @import("weights.zig");
const enc = @import("encoder.zig");
const Allocator = std.mem.Allocator;
const S = Config.speakers;

test {
    _ = Config;
    _ = mel;
    _ = sortformer;
}

/// The rows a clip of `samples` gives the prompt.
pub fn rows(samples: usize) usize {
    return (samples / Config.hop + Config.sub - 1) / Config.sub;
}

/// One online window over mel frames: its core [s, e) and the context-extended span [cs, ce).
pub const Window = struct { s: usize, e: usize, cs: usize, ce: usize };

pub fn windows(gpa: Allocator, total: usize) ![]Window {
    const n = @max(1, (total + Config.window_frames - 1) / Config.window_frames);
    const out = try gpa.alloc(Window, n);
    for (out, 0..) |*w, i| {
        const s = i * Config.window_frames;
        const e = @min(s + Config.window_frames, total);
        w.* = .{ .s = s, .e = e, .cs = s -| Config.context_frames, .ce = @min(e + Config.context_frames, total) };
    }
    return out;
}

fn ceilDiv(a: usize, b: usize) usize {
    return (a + b - 1) / b;
}

pub const Encoder = struct {
    gpa: Allocator,
    d: *const cuda.Driver,
    c: Config,
    w: W.Weights,
    r: enc.Runner,
    kernel: []u16, // the sinusoid speaker kernel [8, asr d] in bf16

    /// The audio encoder of the checkpoint at `dir` (config.json's perception block and its perception.* tensors).
    pub fn init(gpa: Allocator, io: std.Io, d: *const cuda.Driver, dir: []const u8) !*Encoder {
        const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
        defer gpa.free(path);
        const text = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 24));
        defer gpa.free(text);
        const parsed = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
        defer parsed.deinit();
        const c = try Config.parse(parsed.value.object);
        const e = try gpa.create(Encoder);
        errdefer gpa.destroy(e);
        var w = try W.Weights.load(gpa, io, d, dir, c);
        errdefer w.deinit(gpa);
        var r = try enc.Runner.init(gpa, d, c);
        errdefer r.deinit();
        e.* = .{ .gpa = gpa, .d = d, .c = c, .w = w, .r = r, .kernel = try sinusoid(gpa, c.asr.d) };
        return e;
    }

    pub fn deinit(e: *Encoder) void {
        e.gpa.free(e.kernel);
        e.r.deinit();
        e.w.deinit(e.gpa);
        e.gpa.destroy(e);
    }

    /// Device bytes the weights and scratch hold.
    pub fn bytes(e: *const Encoder) u64 {
        return e.w.bytes + e.r.scratch.len;
    }

    fn upload(e: *const Encoder, s: cuda.Stream, dst: u64, src: []const u8) !void {
        try e.d.check(e.d.api.cuMemcpyHtoDAsync_v2(dst, src.ptr, src.len, s.handle), "cuMemcpyHtoDAsync");
        try s.synchronize();
    }

    fn download(e: *const Encoder, s: cuda.Stream, dst: []u8, src: u64) !void {
        try s.synchronize();
        try e.d.check(e.d.api.cuMemcpyDtoH_v2(dst.ptr, src, dst.len), "cuMemcpyDtoH");
    }

    fn copy(e: *const Encoder, s: cuda.Stream, dst: u64, src: u64, len: usize) !void {
        try e.d.check(e.d.api.cuMemcpyDtoDAsync_v2(dst, src, len, s.handle), "cuMemcpyDtoDAsync");
    }

    /// FeatureStacking's input for frames [cs, ce) of `feats` [frames, 128] at b.in: rows of 8 frames, zero padded.
    fn stacked(e: *const Encoder, s: cuda.Stream, feats: []const u16, cs: usize, ce: usize) !usize {
        const n = ceilDiv(ce - cs, Config.sub);
        const host = try e.gpa.alloc(u16, n * Config.sub * Config.mels);
        defer e.gpa.free(host);
        @memset(host, 0);
        @memcpy(host[0 .. (ce - cs) * Config.mels], feats[cs * Config.mels .. ce * Config.mels]);
        try e.upload(s, e.r.b.in, std.mem.sliceAsBytes(host));
        return n;
    }

    /// `wav`'s rows (bf16 [rows(wav.len), text hidden]) at `out`; stage tensors into `dump` when given.
    pub fn encode(e: *Encoder, s: cuda.Stream, wav: []const f32, out: u64, dump: ?*Dump) !usize {
        const gpa = e.gpa;
        const c = e.c;
        const b = e.r.b;
        var f = try mel.features(gpa, wav, e.w.host.window, e.w.host.fb);
        defer f.deinit(gpa);
        if (dump) |dm| {
            try dm.bits("", "mel", f.mel, f.frames, Config.mels);
            try dm.bits("", "mel_asr_norm", f.asr, f.frames, Config.mels);
        }
        const wins = try windows(gpa, f.frames);
        defer gpa.free(wins);
        const encoded = ceilDiv(f.valid, Config.sub);
        var total_out: usize = 0;
        const cores = try gpa.alloc([2]usize, wins.len); // each window's (first core row, core rows)
        defer gpa.free(cores);
        var room: usize = 0;
        for (wins) |w| room += ceilDiv(w.ce - w.cs, Config.sub);
        const d = c.asr.d;
        // device rows for the whole clip: the ASR encoder's core rows, their norm, the speaker infusion, the fusion
        var clip = try cuda.DeviceBuffer.alloc(e.d, 4 * room * d * 2);
        defer clip.free();
        const asr_rows = clip.ptr;
        const normed = asr_rows + room * d * 2;
        const infusion = normed + room * d * 2;
        const fused = infusion + room * d * 2;
        var tag_buf: [32]u8 = undefined;
        // the ASR branch, window by window, its core rows kept
        for (wins, 0..) |w, wi| {
            const tag = try std.fmt.bufPrint(&tag_buf, "asr.w{d}", .{wi});
            const n = try e.stacked(s, f.asr, w.cs, w.ce);
            const len = ceilDiv(std.math.clamp(f.valid -| w.cs, 0, w.ce - w.cs), Config.sub);
            try e.r.stackIn(s, &e.w.asr, c.asr, n, b.x);
            if (dump) |dm| try dm.device(s, tag, ".pre_encode", b.x, n, d);
            try e.r.run(s, &e.w.asr, c.asr, 0, n, len, b.a, dump, tag);
            if (dump) |dm| try dm.device(s, tag, ".final_norm", b.a, n, d);
            const left = (w.s - w.cs) / Config.sub;
            const core = @min(ceilDiv(w.e, Config.sub) - ceilDiv(w.s, Config.sub), n - left);
            try e.copy(s, asr_rows + total_out * d * 2, b.a + left * d * 2, core * d * 2);
            cores[wi] = .{ total_out, core };
            total_out += core;
        }
        if (dump) |dm| try dm.device(s, "", "asr_encoded", asr_rows, total_out, d);
        // the Sortformer, streaming: each window sees the speaker cache and FIFO before its own rows
        const diar = try gpa.alloc(f32, total_out * S);
        defer gpa.free(diar);
        var st: sortformer.State = .{ .gpa = gpa, .width = c.sf.d };
        defer st.deinit();
        for (wins, 0..) |w, wi| {
            const tag = try std.fmt.bufPrint(&tag_buf, "sf.w{d}", .{wi});
            try e.diarize(s, &st, &f, w, wi, wi + 1 == wins.len, cores[wi][1], diar[cores[wi][0] * S ..][0 .. cores[wi][1] * S], dump, tag);
        }
        if (dump) |dm| try dm.floats("", "sortformer_probs", diar, total_out, S);
        // fusion: LayerNorm(asr) + LayerNorm_8(speakers above threshold) @ the sinusoid kernel, then the projection
        try e.r.vk.layerNorm(s, asr_rows, e.w.asr_norm, e.w.asr_norm_bias, normed, total_out, d, c.asr.eps);
        const inf_host = try gpa.alloc(u16, total_out * d);
        defer gpa.free(inf_host);
        const feats = try gpa.alloc(f32, total_out * S);
        defer gpa.free(feats);
        for (0..total_out) |t| {
            var x: [S]f32 = undefined;
            for (0..S) |k| x[k] = if (diar[t * S + k] > Config.speaker_threshold) 1 else 0;
            @memcpy(feats[t * S ..][0..S], &x);
            const nt = layerNorm8(x, e.w.host.diar_norm, e.w.host.diar_norm_bias, c.asr.eps);
            for (0..d) |col| {
                var acc: f32 = 0;
                for (0..S) |k| acc += nt[k] * mel.widen(e.kernel[k * d + col]);
                inf_host[t * d + col] = mel.bf16(acc);
            }
        }
        if (dump) |dm| {
            try dm.floats("", "speaker_features", feats, total_out, S);
            try dm.device(s, "", "asr_norm_out", normed, total_out, d);
            try dm.bits("", "speaker_infusion", inf_host, total_out, d);
        }
        try e.upload(s, infusion, std.mem.sliceAsBytes(inf_host));
        try e.r.vk.pointwise(s, infusion, normed, fused, total_out * d, .none);
        if (dump) |dm| try dm.device(s, "", "fused", fused, total_out, d);
        if (encoded > total_out) return error.AudioRows;
        try cuda.vit.linear(&e.r.g, &e.r.vk, s, b.acc, fused, e.w.proj, e.w.proj_bias, out, encoded, c.text_hidden, d);
        if (dump) |dm| try dm.device(s, "", "audio_embeddings", out, encoded, c.text_hidden);
        return encoded;
    }

    /// One Sortformer window: encoder and head over cache, FIFO and window, core predictions to `core_out` [core, 8].
    fn diarize(e: *Encoder, s: cuda.Stream, st: *sortformer.State, f: *const mel.Features, w: Window, wi: usize, last: bool, core: usize, core_out: []f32, dump: ?*Dump, tag: []const u8) !void {
        _ = wi;
        const gpa = e.gpa;
        const c = e.c;
        const b = e.r.b;
        const k = &e.r.k;
        const g = &e.r.g;
        const vk = &e.r.vk;
        const sd = c.sf.d;
        const t = Config.tf_d;
        const n = try e.stacked(s, f.mel, w.cs, w.ce);
        const pre_len = ceilDiv(std.math.clamp(f.valid -| w.cs, 0, w.ce - w.cs), Config.sub);
        try e.r.stackIn(s, &e.w.sf, c.sf, n, b.h);
        const pre = try gpa.alloc(u16, n * sd);
        defer gpa.free(pre);
        try e.download(s, std.mem.sliceAsBytes(pre), b.h);
        if (dump) |dm| try dm.bits(tag, ".pre_encode", pre, n, sd);
        const spk_n = st.spkRows();
        const fifo_n = st.fifoRows();
        const rows_ = spk_n + fifo_n + n;
        const len = spk_n + fifo_n + pre_len;
        const cat = try gpa.alloc(u16, rows_ * sd);
        defer gpa.free(cat);
        @memcpy(cat[0 .. spk_n * sd], st.spkcache.items);
        @memcpy(cat[spk_n * sd ..][0 .. fifo_n * sd], st.fifo.items);
        @memcpy(cat[(spk_n + fifo_n) * sd ..], pre);
        if (dump) |dm| try dm.bits(tag, ".encoder_in", cat, rows_, sd);
        try e.upload(s, b.x, std.mem.sliceAsBytes(cat));
        try e.r.run(s, &e.w.sf, c.sf, 1, rows_, len, b.a, dump, tag);
        if (dump) |dm| try dm.device(s, tag, ".encoder_out", b.a, rows_, sd);
        // the head: encoder_proj, padding zeroed, the subpixel conv (8 frames a row), relu, hidden, relu, speakers
        try cuda.vit.linear(g, vk, s, b.acc, b.a, e.w.encoder_proj, e.w.encoder_proj_bias, b.h, rows_, t, sd);
        try k.zeroRows(s, b.h, rows_, t, len);
        try k.im2col3(s, b.h, b.qkv, rows_, t, len);
        try cuda.vit.linear(g, vk, s, b.acc, b.qkv, e.w.upsample, e.w.upsample_bias, b.wide, rows_, Config.sub * t, 3 * t);
        const hr = rows_ * Config.sub;
        try k.reluInPlace(s, b.wide, hr * t);
        try cuda.vit.linear(g, vk, s, b.acc, b.wide, e.w.hidden, e.w.hidden_bias, b.qkv, hr, t, t);
        try k.reluInPlace(s, b.qkv, hr * t);
        try cuda.vit.linear(g, vk, s, b.acc, b.qkv, e.w.spks, e.w.spks_bias, b.h, hr, S, t);
        const logits = try gpa.alloc(u16, hr * S);
        defer gpa.free(logits);
        try e.download(s, std.mem.sliceAsBytes(logits), b.h);
        if (dump) |dm| try dm.bits(tag, ".logits_hr", logits, hr, S);
        // sigmoid in bf16, frames past the valid length zeroed
        const probs = try gpa.alloc(f32, hr * S);
        defer gpa.free(probs);
        for (probs, logits, 0..) |*p, l, i| {
            const x = mel.widen(l);
            p.* = if (i / S < len * Config.sub) mel.rbf(1.0 / (1.0 + @exp(-x))) else 0;
        }
        if (dump) |dm| try dm.floats(tag, ".probs_hr", probs, hr, S);
        const lc = (w.s - w.cs) / Config.sub;
        const rc = ceilDiv(w.ce - w.e, Config.sub);
        const lr = try sortformer.downsample(gpa, probs);
        defer gpa.free(lr);
        for (len..lr.len / S) |r| @memset(lr[r * S ..][0..S], 0);
        const chunk_len = n - lc - rc;
        const start = (spk_n + fifo_n + lc) * Config.sub;
        const chunk_lr = try sortformer.downsample(gpa, probs[start * S ..][0 .. chunk_len * Config.sub * S]);
        defer gpa.free(chunk_lr);
        if (!last) try st.update(pre, lr, lc, rc, e.w.host.sil);
        if (dump) |dm| {
            try dm.bits(tag, ".spkcache_after", st.spkcache.items, st.spkRows(), sd);
            try dm.bits(tag, ".fifo_after", st.fifo.items, st.fifoRows(), sd);
        }
        // _align_diar_frames: the core rows, the last pooled row repeated when the window gives fewer
        const have = chunk_lr.len / S;
        for (0..core) |r| @memcpy(core_out[r * S ..][0..S], chunk_lr[@min(r, have - 1) * S ..][0..S]);
    }
};

/// LayerNorm over the eight speaker features (bf16 0 or 1), fp32 inside and rounded once.
fn layerNorm8(x: [S]f32, w: [S]f32, b: [S]f32, eps: f32) [S]f32 {
    var mean: f32 = 0;
    for (x) |v| mean += v;
    mean /= S;
    var v2: f32 = 0;
    for (x) |v| v2 += (v - mean) * (v - mean);
    const rstd = 1.0 / @sqrt(v2 / S + eps);
    var out: [S]f32 = undefined;
    for (&out, x, w, b) |*o, v, wi, bi| o.* = mel.rbf((v - mean) * rstd * wi + bi);
    return out;
}

/// _build_sinusoid_position_encoding(8, d) in fp32, cast to bf16 with the module.
fn sinusoid(gpa: Allocator, d: usize) ![]u16 {
    const out = try gpa.alloc(u16, S * d);
    const step: f32 = @floatCast(-(@log(10000.0) / @as(f64, @floatFromInt(d))));
    for (0..S) |p| for (0..d / 2) |j| {
        const div: f32 = @floatCast(@exp(@as(f64, @as(f32, @floatFromInt(2 * j)) * step)));
        const a: f32 = @as(f32, @floatFromInt(p)) * div;
        out[p * d + 2 * j] = mel.bf16(@floatCast(@sin(@as(f64, a))));
        out[p * d + 2 * j + 1] = mel.bf16(@floatCast(@cos(@as(f64, a))));
    };
    return out;
}
