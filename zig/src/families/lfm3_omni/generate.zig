//! lfm3-omni-generate: LFM3-Omni-2B speech in, text and speech out, its server, stage dumps and teacher forcing.
const std = @import("std");
const core = @import("core");
const om = @import("lfm3_omni");

const cfg = om.config;
const Engine = om.Engine;

const usage =
    \\usage: lfm3-omni-generate --model DIR --serve HOST:PORT [--system TEXT | --system-file FILE] [sampling flags]
    \\       lfm3-omni-generate --model DIR --wav IN.wav [--out REPLY.wav] [--system TEXT | --system-file FILE]
    \\                          [--audio-temperature 0.4] [--audio-repetition-penalty 1.2] [--audio-top-k 0]
    \\                          [--identity-order 1] [--seed 0] [--max-new 512] [--first-chunk 1] [--chunk 8]
    \\                          [--dump DIR] [--force DIR] [--repeat 1]
    \\
    \\--serve: POST /v1/audio/chat {"audio": base64 WAV, "system"?, "stream"? (true), "audio_temperature"?, "seed"?,
    \\  "max_new"?} answers with server-sent events (text deltas, base64 pcm16 24 kHz audio, then done) or one JSON.
    \\The input is any PCM or float WAV, resampled to 16 kHz; the reply is 16-bit PCM mono at 24 kHz.
    \\--dump DIR (an existing directory) writes the stages as raw little-endian arrays: mel.f32 [128, frames],
    \\  enc.f32 and audio_emb.f32 [tokens, ...], input_ids.i32, prefill_hidden.f32 [rows, 2048], first_logits.f32,
    \\  depth_logits0.f32 [12, 2017], gen_seq.i32, gen_codes.i32 [frames, 12], step_hidden.f32, codec_lin.f32
    \\  [6 * frames, 1282] and reply.f32 (the whole reply through the codec at once).
    \\--codec-ref DIR: with --dump, codec_lin.f32 and reply.f32 decode DIR's gen_codes.i32 ([12, frames]) instead.
    \\--depth-check DIR: the depth decoder alone on DIR's depth_h.f32 [n, 2048] and depth_codes.i32 [n, 12]: three
    \\  passes a row (identity order), every pass's logits written to depth_check.f32 [n, 3, 12, 2017]; no reply.
    \\--graphs 0 launches the backbone's rows and the depth decoder's frames one kernel at a time (no CUDA graphs).
    \\--host-mel 1 computes the log-mel features on the host (a float64 DFT) instead of the device.
    \\--force DIR reads a reference's gen_seq.i32 and gen_codes.i32 ([12, frames]) and feeds them back, scoring the
    \\  engine's own picks (greedy text, its audio sampling settings) at every slot.
    \\The last stderr line is "RESULT lfm3-omni OK ..." with the timings.
    \\
;

const Args = struct {
    model: []const u8 = "",
    wav: []const u8 = "",
    out: ?[]const u8 = null,
    system: []const u8 = "Respond with interleaved text and audio.",
    smp: om.engine.Sampling = .{},
    stream: om.engine.Streaming = .{},
    dump: ?[]const u8 = null,
    force: ?[]const u8 = null,
    codec_ref: ?[]const u8 = null,
    depth_check: ?[]const u8 = null,
    repeat: usize = 1,
    host_mel: bool = false,
    serve: ?[]const u8 = null,
    graphs: bool = true,
};

pub fn main(init: std.process.Init) !u8 {
    const arena = init.arena.allocator();
    const gpa = std.heap.c_allocator;
    const io = init.io;
    const argv = try init.minimal.args.toSlice(arena);
    var a: Args = .{};
    var i: usize = 1;
    while (i < argv.len) : (i += 2) {
        if (i + 1 >= argv.len) {
            std.debug.print("{s}", .{usage});
            return 2;
        }
        const k = argv[i];
        const v = argv[i + 1];
        if (eq(k, "--model")) a.model = v else if (eq(k, "--wav")) a.wav = v else if (eq(k, "--out")) a.out = v else if (eq(k, "--system")) a.system = v else if (eq(k, "--system-file")) {
            a.system = std.mem.trim(u8, try std.Io.Dir.cwd().readFileAlloc(io, v, arena, .limited(1 << 20)), " \n\r\t");
        } else if (eq(k, "--audio-temperature")) a.smp.audio_temperature = try std.fmt.parseFloat(f32, v) else if (eq(k, "--audio-repetition-penalty")) a.smp.audio_repetition_penalty = try std.fmt.parseFloat(f32, v) else if (eq(k, "--audio-top-k")) a.smp.audio_top_k = try std.fmt.parseInt(usize, v, 10) else if (eq(k, "--identity-order")) a.smp.identity_order = !eq(v, "0") else if (eq(k, "--seed")) a.smp.seed = try std.fmt.parseInt(u64, v, 10) else if (eq(k, "--max-new")) a.smp.max_new = try std.fmt.parseInt(usize, v, 10) else if (eq(k, "--first-chunk")) a.stream.first = try std.fmt.parseInt(usize, v, 10) else if (eq(k, "--chunk")) a.stream.chunk = try std.fmt.parseInt(usize, v, 10) else if (eq(k, "--dump")) a.dump = v else if (eq(k, "--force")) a.force = v else if (eq(k, "--codec-ref")) a.codec_ref = v else if (eq(k, "--depth-check")) a.depth_check = v else if (eq(k, "--serve")) a.serve = v else if (eq(k, "--graphs")) a.graphs = !eq(v, "0") else if (eq(k, "--host-mel")) a.host_mel = !eq(v, "0") else if (eq(k, "--repeat")) a.repeat = try std.fmt.parseInt(usize, v, 10) else {
            std.debug.print("unknown flag {s}\n{s}", .{ k, usage });
            return 2;
        }
    }
    if (a.depth_check) |dc| {
        const e = try Engine.create(gpa, io, a.model, .{});
        defer e.destroy();
        try depthCheck(arena, io, e, dc);
        return 0;
    }
    if (a.serve) |hp| {
        if (a.model.len == 0) {
            std.debug.print("{s}", .{usage});
            return 2;
        }
        const colon = std.mem.lastIndexOfScalar(u8, hp, ':') orelse return error.ServeIsHostColonPort;
        const tpath0 = try std.fs.path.join(arena, &.{ a.model, "tokenizer.json" });
        var tok0 = try core.tokenizer.loadTokenizer(io, gpa, tpath0);
        defer tok0.deinit();
        const e = try Engine.create(gpa, io, a.model, .{ .graphs = a.graphs });
        defer e.destroy();
        try warm(gpa, io, e, &tok0, a);
        try om.serve.run(gpa, io, e, &tok0, .{ .host = hp[0..colon], .port = try std.fmt.parseInt(u16, hp[colon + 1 ..], 10), .system = a.system, .smp = a.smp, .stream = a.stream });
        return 0;
    }
    if (a.model.len == 0 or a.wav.len == 0) {
        std.debug.print("{s}", .{usage});
        return 2;
    }
    const tpath = try std.fs.path.join(arena, &.{ a.model, "tokenizer.json" });
    var tok = try core.tokenizer.loadTokenizer(io, gpa, tpath);
    defer tok.deinit();
    const t_load = om.engine.nowNs(io);
    const e = try Engine.create(gpa, io, a.model, .{ .graphs = a.graphs });
    defer e.destroy();
    std.log.info("lfm3-omni: loaded in {d:.1} s, {d:.2} GB on the device", .{ @as(f64, @floatFromInt(om.engine.nowNs(io) - t_load)) / 1e9, @as(f64, @floatFromInt(e.deviceBytes())) / 1e9 });
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, a.wav, arena, .limited(1 << 30));
    const au = try om.wav.parse(arena, bytes);
    const pcm = try om.wav.resample(arena, au.samples, au.rate, om.mel.sample_rate);
    var r: usize = 0;
    while (r < a.repeat) : (r += 1) try run(gpa, io, a, e, &tok, pcm, r + 1 == a.repeat);
    return 0;
}

/// One short turn on a second of quiet noise, so the first request finds cuBLAS's plans and the kernels loaded.
fn warm(gpa: std.mem.Allocator, io: std.Io, e: *Engine, tok: *const core.tokenizer.Tokenizer, a: Args) !void {
    var ar_state = std.heap.ArenaAllocator.init(gpa);
    defer ar_state.deinit();
    const ar = ar_state.allocator();
    const pcm = try ar.alloc(f32, om.mel.sample_rate);
    var prng = std.Random.DefaultPrng.init(1);
    for (pcm) |*x| x.* = (prng.random().float(f32) - 0.5) * 0.002;
    const nf = try e.featuresDevice(pcm);
    const n = try e.encode(null, nf);
    const p = try om.prompt.build(ar, tok, a.system, n);
    try e.prefill(p.ids, p.audio_at, null);
    var st: om.engine.Stats = .{};
    var smp = a.smp;
    smp.max_new = 24;
    var sink_ctx: u8 = 0;
    try e.reply(smp, .{}, .{ .ctx = &sink_ctx }, null, &st, om.engine.nowNs(io));
}

fn eq(x: []const u8, y: []const u8) bool {
    return std.mem.eql(u8, x, y);
}

const Out = struct {
    gpa: std.mem.Allocator,
    e: *Engine,
    text: std.ArrayList(u32) = .empty,
    seq: std.ArrayList(i32) = .empty,
    frames: std.ArrayList([cfg.codebooks]i32) = .empty,
    pcm: std.ArrayList(f32) = .empty,
    hidden: ?*std.ArrayList(f32) = null,

    fn onText(ctx: *anyopaque, id: u32) anyerror!void {
        const o: *Out = @ptrCast(@alignCast(ctx));
        try o.text.append(o.gpa, id);
        try o.seq.append(o.gpa, @intCast(id));
    }
    fn onFrame(ctx: *anyopaque, codes: *const [cfg.codebooks]i32) anyerror!void {
        const o: *Out = @ptrCast(@alignCast(ctx));
        try o.frames.append(o.gpa, codes.*);
        try o.seq.append(o.gpa, cfg.audio_out_token_id);
    }
    fn onPcm(ctx: *anyopaque, x: []const f32) anyerror!void {
        const o: *Out = @ptrCast(@alignCast(ctx));
        try o.pcm.appendSlice(o.gpa, x);
    }
    fn onHidden(ctx: *anyopaque) anyerror!void {
        const o: *Out = @ptrCast(@alignCast(ctx));
        if (o.hidden) |h| {
            var row: [cfg.hidden]f32 = undefined;
            try o.e.lastHidden(&row);
            try h.appendSlice(o.gpa, &row);
        }
    }
    fn deinit(o: *Out) void {
        o.text.deinit(o.gpa);
        o.seq.deinit(o.gpa);
        o.frames.deinit(o.gpa);
        o.pcm.deinit(o.gpa);
    }
};

fn writeRaw(io: std.Io, dir: []const u8, name: []const u8, data: []const u8) !void {
    var buf: [1024]u8 = undefined;
    const p = try std.fmt.bufPrint(&buf, "{s}/{s}", .{ dir, name });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = p, .data = data });
}

fn readRaw(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, name: []const u8) ![]u8 {
    var buf: [1024]u8 = undefined;
    const p = try std.fmt.bufPrint(&buf, "{s}/{s}", .{ dir, name });
    return std.Io.Dir.cwd().readFileAlloc(io, p, gpa, .limited(1 << 30));
}

fn run(gpa: std.mem.Allocator, io: std.Io, a: Args, e: *Engine, tok: *const core.tokenizer.Tokenizer, pcm: []const f32, last: bool) !void {
    var ar_state = std.heap.ArenaAllocator.init(gpa);
    defer ar_state.deinit();
    const ar = ar_state.allocator();
    var st: om.engine.Stats = .{};
    const t_req = om.engine.nowNs(io);
    // speech in
    var nf = om.mel.frames(pcm.len);
    const feats = try ar.alloc(f32, cfg.mel_bins * nf);
    if (a.host_mel) e.features(pcm, feats) else nf = try e.featuresDevice(pcm);
    st.mel_ms = @as(f64, @floatFromInt(om.engine.nowNs(io) - t_req)) / 1e6;
    const t_enc = om.engine.nowNs(io);
    const n_audio = try e.encode(if (a.host_mel) feats else null, nf);
    const dump = if (last) a.dump else null;
    if (dump) |d| {
        try e.stream.synchronize();
        if (!a.host_mel) try e.d2h(std.mem.sliceAsBytes(feats), e.mel_f32);
        try writeRaw(io, d, "mel.f32", std.mem.sliceAsBytes(feats));
        try writeRaw(io, d, "enc.f32", std.mem.sliceAsBytes(try deviceRows(ar, e, e.enc_out, n_audio * cfg.enc_d)));
        try writeRaw(io, d, "audio_emb.f32", std.mem.sliceAsBytes(try deviceRows(ar, e, e.audio_emb, n_audio * cfg.hidden)));
    }
    st.encode_ms = @as(f64, @floatFromInt(om.engine.nowNs(io) - t_enc)) / 1e6;
    st.audio_tokens = n_audio;
    const p = try om.prompt.build(ar, tok, a.system, n_audio);
    st.prompt_rows = p.ids.len;
    const t_pre = om.engine.nowNs(io);
    var all_hidden: ?[]f32 = null;
    if (dump != null) all_hidden = try ar.alloc(f32, p.ids.len * cfg.hidden);
    try e.prefill(p.ids, p.audio_at, all_hidden);
    st.prefill_ms = @as(f64, @floatFromInt(om.engine.nowNs(io) - t_pre)) / 1e6;
    var hidden: std.ArrayList(f32) = .empty;
    var out: Out = .{ .gpa = gpa, .e = e };
    defer out.deinit();
    if (dump) |d| {
        try writeRaw(io, d, "input_ids.i32", std.mem.sliceAsBytes(p.ids));
        try writeRaw(io, d, "prefill_hidden.f32", std.mem.sliceAsBytes(all_hidden.?));
        const lg = try ar.alloc(f32, cfg.vocab);
        try e.textLogits(lg);
        try writeRaw(io, d, "first_logits.f32", std.mem.sliceAsBytes(lg));
        out.hidden = &hidden;
        try Out.onHidden(&out);
    }
    var forced: ?om.engine.Forced = null;
    if (a.force) |fd| {
        const seq_b = try readRaw(ar, io, fd, "gen_seq.i32");
        const seq: []const u32 = @alignCast(std.mem.bytesAsSlice(u32, seq_b));
        forced = .{ .seq = seq, .frames = try readFrames(ar, io, fd) };
    }
    const sink: om.engine.Sink = .{ .ctx = &out, .text = Out.onText, .frame = Out.onFrame, .pcm = Out.onPcm, .hidden = if (dump != null) Out.onHidden else null };
    try e.reply(a.smp, a.stream, sink, if (forced) |*f| f else null, &st, t_req);
    // the reply's text (special tokens left out)
    var words: std.ArrayList(u32) = .empty;
    for (out.text.items) |id| if (id < cfg.bos_token_id) try words.append(ar, id);
    const text = try tok.decode(ar, words.items, true);
    if (a.out) |o| {
        if (last) try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = o, .data = try om.wav.encode16(ar, out.pcm.items, cfg.sample_rate) });
    }
    if (dump) |d| {
        try writeRaw(io, d, "gen_seq.i32", std.mem.sliceAsBytes(out.seq.items));
        try writeRaw(io, d, "gen_codes.i32", std.mem.sliceAsBytes(out.frames.items));
        try writeRaw(io, d, "step_hidden.f32", std.mem.sliceAsBytes(hidden.items));
        try writeRaw(io, d, "reply_stream.f32", std.mem.sliceAsBytes(out.pcm.items));
        var frames: []const [cfg.codebooks]i32 = out.frames.items;
        if (a.codec_ref) |cr| frames = try readFrames(ar, io, cr);
        try dumpDepthAndCodec(ar, io, e, d, p.ids, p.audio_at, frames, a.stream);
        hidden.deinit(gpa);
    }
    if (forced) |f| std.debug.print("FORCED text {d}/{d} codes {d}/{d} frames {d}/{d}\n", .{ f.text_hits, f.text_total, f.code_hits, f.code_total, f.frame_hits, f.frame_total });
    std.debug.print("TEXT {s}\n", .{text});
    std.debug.print("RESULT lfm3-omni OK input_s={d:.2} audio_tokens={d} prompt_rows={d} mel_ms={d:.1} encode_ms={d:.1} prefill_ms={d:.1} first_text_ms={d:.1} first_frame_ms={d:.1} first_pcm_ms={d:.1} text_tokens={d} frames={d} speech_s={d:.2} step_ms={d:.1} depth_ms={d:.1} codec_ms={d:.1} total_ms={d:.1}\n", .{ @as(f64, @floatFromInt(pcm.len)) / 16000.0, st.audio_tokens, st.prompt_rows, st.mel_ms, st.encode_ms, st.prefill_ms, st.first_text_ms, st.first_frame_ms, st.first_pcm_ms, st.text_tokens, st.frames, @as(f64, @floatFromInt(st.samples)) / 24000.0, st.step_ms, st.depth_ms, st.codec_ms, st.total_ms });
}

/// The depth decoder on given hidden states: per row, the passes of the identity order with the given codes revealed.
fn depthCheck(ar: std.mem.Allocator, io: std.Io, e: *Engine, dir: []const u8) !void {
    const hb = try readRaw(ar, io, dir, "depth_h.f32");
    const cb = try readRaw(ar, io, dir, "depth_codes.i32");
    const h: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, hb));
    const c: []const i32 = @alignCast(std.mem.bytesAsSlice(i32, cb));
    const n = h.len / cfg.hidden;
    var out: std.ArrayList(f32) = .empty;
    for (0..n) |r| {
        try e.setHidden(h[r * cfg.hidden ..][0..cfg.hidden]);
        try e.depthStart();
        var codes: [cfg.codebooks]i32 = @splat(0);
        var masked: [cfg.codebooks]bool = @splat(true);
        var at: usize = 0;
        for (cfg.maskgit_schedule) |k| {
            try out.appendSlice(ar, try e.depthPass(&codes, &masked));
            for (at..at + k) |q| {
                codes[q] = c[r * cfg.codebooks + q];
                masked[q] = false;
            }
            at += k;
        }
    }
    try writeRaw(io, dir, "depth_check.f32", std.mem.sliceAsBytes(out.items));
    std.debug.print("RESULT lfm3-omni depth-check OK rows={d}\n", .{n});
}

/// A reference's gen_codes.i32 ([12, frames]) as frames.
fn readFrames(ar: std.mem.Allocator, io: std.Io, dir: []const u8) ![]const [cfg.codebooks]i32 {
    const codes_b = try readRaw(ar, io, dir, "gen_codes.i32");
    const cm: []const i32 = @alignCast(std.mem.bytesAsSlice(i32, codes_b));
    const n = cm.len / cfg.codebooks;
    const frames = try ar.alloc([cfg.codebooks]i32, n);
    for (frames, 0..) |*fr, t| for (fr, 0..) |*c, q| {
        c.* = cm[q * n + t];
    };
    return frames;
}

fn deviceRows(ar: std.mem.Allocator, e: *Engine, src: u64, n: usize) ![]f32 {
    const hb = try ar.alloc(u16, n);
    try e.d2h(std.mem.sliceAsBytes(hb), src);
    const out = try ar.alloc(f32, n);
    for (out, hb) |*o, h| o.* = @bitCast(@as(u32, h) << 16);
    return out;
}

/// The first frame's depth logits from the prompt (every codebook masked) and the whole reply through the codec.
fn dumpDepthAndCodec(ar: std.mem.Allocator, io: std.Io, e: *Engine, d: []const u8, ids: []const u32, audio_at: usize, frames: []const [cfg.codebooks]i32, sm: om.engine.Streaming) !void {
    // the reference scores the prompt without the interleaved marker
    try e.prefill(ids[0 .. ids.len - 1], audio_at, null);
    try e.depthStart();
    const codes: [cfg.codebooks]i32 = @splat(0);
    const masked: [cfg.codebooks]bool = @splat(true);
    const lg = try e.depthPass(&codes, &masked);
    try writeRaw(io, d, "depth_logits0.f32", std.mem.sliceAsBytes(lg));
    try e.codecReset();
    var lin: std.ArrayList(f32) = .empty;
    var ola = om.engine.OverlapAdd.init(ar, &e.window);
    var wav: std.ArrayList(f32) = .empty;
    var at: usize = 0;
    // in the streaming chunks the server uses: the first `first` frames, then `chunk` at a time
    while (at < frames.len) {
        const n = @min(if (at == 0) sm.first else sm.chunk, frames.len - at);
        const l = try ar.alloc(f32, n * cfg.frame_stack * 2 * cfg.spec_bins);
        const fr = try e.codecChunk(frames[at .. at + n], l);
        try lin.appendSlice(ar, l);
        try wav.appendSlice(ar, try ola.add(fr, at + n == frames.len));
        at += n;
    }
    try writeRaw(io, d, "codec_lin.f32", std.mem.sliceAsBytes(lin.items));
    try writeRaw(io, d, "reply.f32", std.mem.sliceAsBytes(wav.items));
}
