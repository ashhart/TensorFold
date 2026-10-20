//! drama3-generate: Drama 3 text to WAV on one GPU (CUDA or Metal), plus the codec's encode and the oracle checks.
const std = @import("std");
const core = @import("core");
const d3 = @import("drama3");
const checks = @import("check.zig");

const W = d3.config.frame_width;
const H = d3.config.hidden;

pub const Args = struct {
    model: []const u8 = "",
    text: ?[]const u8 = null,
    ref_text: ?[]const u8 = null,
    ref_codes: ?[]const u8 = null,
    rows: ?[]const u8 = null,
    rows_out: ?[]const u8 = null,
    out: ?[]const u8 = null,
    check: ?[]const u8 = null,
    check_frames: usize = 40,
    smp: d3.Sampling = .{},
    max_frames: usize = 1500,
    graphs: bool = true,
    sync_every: usize = 8,
    opts: d3.Options = .{},
    replay: bool = false,
    speed_probe: bool = false,
    repeat: usize = 1,
    codec: ?[]const u8 = null,
    codec_check: ?[]const u8 = null,
    frames: ?[]const u8 = null,
    ref_audio: ?[]const u8 = null,
    first_chunk: usize = 4,
    chunk: usize = 16,
    wav: ?[]const u8 = null,
    pcm: ?[]const u8 = null,
    encode_out: ?[]const u8 = null,
    verbose_chunks: bool = false,
};

const usage =
    \\usage: drama3-generate --model DIR --codec CODEC --text "..." --wav OUT.wav [--ref-audio REF.wav --ref-text "..."]
    \\                       [--pcm-out PATH|-] [--first-chunk 4] [--chunk 16] [--chunks-log 1]
    \\       drama3-generate --codec CODEC --encode OUT.codes --ref-audio REF.wav
    \\       drama3-generate --model DIR --text "..." [--ref-text "..." --ref-codes FILE] [--rows FILE]
    \\                       [--temperature 1.0] [--top-k 30] [--top-p 0.9] [--seed 1] [--greedy]
    \\                       [--max-frames 1500] [--graphs 1] [--sync-every 8] [--out FILE] [--rows-out FILE]
    \\                       [--max-ctx 4096] [--max-rows 128] [--threads 16] [--repeat N] [--pdl 1] [--split 1]
    \\       drama3-generate --model DIR --check DUMPDIR [--check-frames 40]
    \\       drama3-generate --codec-check REF --codec CODEC --frames FRAMES [--ref-audio WAV]
    \\
    \\WAV out: 16-bit PCM mono 44.1 kHz; --pcm-out streams raw s16le ("-" is stdout). Reference WAVs: any PCM or float32.
    \\--replay 1 checks eager/graphs, scratch poisoning, codec chunks and interleaved independent state.
    \\--check honors --max-rows for prompt probes; compare the same oracle vectors at 1 and N rows.
    \\--speed-probe 1 compares the staged CUDA GEMV candidate; it never promotes a serving kernel.
    \\--rows: prompt rows int32 [n, 11]; --ref-codes and --encode: int32 [frames, 10]; --out: frames int32 [n, 11].
    \\The last stderr line is "RESULT drama3 OK ..." with the frame count, timings and frames a second.
    \\
;

pub fn main(init: std.process.Init) !u8 {
    const arena = init.arena.allocator();
    const gpa = std.heap.c_allocator;
    const io = init.io;
    const argv = try init.minimal.args.toSlice(arena);
    var a: Args = .{};
    var i: usize = 1;
    while (i < argv.len) : (i += 1) {
        const k = argv[i];
        if (std.mem.eql(u8, k, "--greedy")) {
            a.smp.temperature = 0;
            a.smp.top_k = 1;
            continue;
        }
        if (i + 1 >= argv.len) {
            std.debug.print("{s}", .{usage});
            return 2;
        }
        const v = argv[i + 1];
        i += 1;
        if (std.mem.eql(u8, k, "--replay")) a.replay = !std.mem.eql(u8, v, "0") else if (std.mem.eql(u8, k, "--speed-probe")) a.speed_probe = !std.mem.eql(u8, v, "0") else if (std.mem.eql(u8, k, "--model")) a.model = v else if (std.mem.eql(u8, k, "--text")) a.text = v else if (std.mem.eql(u8, k, "--ref-text")) a.ref_text = v else if (std.mem.eql(u8, k, "--ref-codes")) a.ref_codes = v else if (std.mem.eql(u8, k, "--rows")) a.rows = v else if (std.mem.eql(u8, k, "--rows-out")) a.rows_out = v else if (std.mem.eql(u8, k, "--out")) a.out = v else if (std.mem.eql(u8, k, "--check")) a.check = v else if (std.mem.eql(u8, k, "--check-frames")) a.check_frames = try std.fmt.parseInt(usize, v, 10) else if (std.mem.eql(u8, k, "--temperature")) a.smp.temperature = try std.fmt.parseFloat(f32, v) else if (std.mem.eql(u8, k, "--top-k")) a.smp.top_k = try std.fmt.parseInt(i32, v, 10) else if (std.mem.eql(u8, k, "--top-p")) a.smp.top_p = try std.fmt.parseFloat(f32, v) else if (std.mem.eql(u8, k, "--seed")) a.smp.seed = try std.fmt.parseInt(u64, v, 10) else if (std.mem.eql(u8, k, "--max-frames")) a.max_frames = try std.fmt.parseInt(usize, v, 10) else if (std.mem.eql(u8, k, "--graphs")) a.graphs = !std.mem.eql(u8, v, "0") else if (std.mem.eql(u8, k, "--sync-every")) a.sync_every = try std.fmt.parseInt(usize, v, 10) else if (std.mem.eql(u8, k, "--max-ctx")) a.opts.max_ctx = try std.fmt.parseInt(usize, v, 10) else if (std.mem.eql(u8, k, "--max-rows")) a.opts.max_rows = try std.fmt.parseInt(usize, v, 10) else if (std.mem.eql(u8, k, "--threads")) a.opts.threads = try std.fmt.parseInt(usize, v, 10) else if (std.mem.eql(u8, k, "--repeat")) a.repeat = try std.fmt.parseInt(usize, v, 10) else if (std.mem.eql(u8, k, "--wav")) a.wav = v else if (std.mem.eql(u8, k, "--pcm-out")) a.pcm = v else if (std.mem.eql(u8, k, "--encode")) a.encode_out = v else if (std.mem.eql(u8, k, "--chunk")) a.chunk = try std.fmt.parseInt(usize, v, 10) else if (std.mem.eql(u8, k, "--chunks-log")) a.verbose_chunks = !std.mem.eql(u8, v, "0") else if (std.mem.eql(u8, k, "--codec")) a.codec = v else if (std.mem.eql(u8, k, "--codec-check")) a.codec_check = v else if (std.mem.eql(u8, k, "--frames")) a.frames = v else if (std.mem.eql(u8, k, "--ref-audio")) a.ref_audio = v else if (std.mem.eql(u8, k, "--first-chunk")) a.first_chunk = try std.fmt.parseInt(usize, v, 10) else if (std.mem.eql(u8, k, "--pdl")) a.opts.pdl = !std.mem.eql(u8, v, "0") else if (std.mem.eql(u8, k, "--split")) a.opts.split = !std.mem.eql(u8, v, "0") else {
            std.debug.print("unknown flag {s}\n{s}", .{ k, usage });
            return 2;
        }
    }
    if (a.codec_check) |ref| return checks.codecCheck(gpa, io, a, ref);
    // --encode needs only the codec and the reference audio
    if (a.model.len == 0 and (a.encode_out == null or a.check != null)) {
        std.debug.print("{s}", .{usage});
        return 2;
    }
    a.opts.max_frames = @max(a.opts.max_frames, a.max_frames);
    if (a.check) |dir| return checks.check(gpa, io, a, dir);

    // audio file -> codes (the codec alone)
    if (a.encode_out) |out| {
        const wp = a.ref_audio orelse return error.RefAudioRequired;
        const c = try d3.tts.CodecOnly.open(gpa, io, a.codec orelse return error.CodecRequired);
        defer c.close();
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, wp, gpa, .limited(1 << 30));
        const au = try d3.wav.parse(gpa, bytes);
        try d3.model.reference(au.samples, au.rate);
        const x44 = try d3.wav.resample(gpa, au.samples, au.rate, d3.codec.sample_rate);
        const t0 = d3.engine.nowNs(io);
        const codes = try c.encode(x44);
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = out, .data = std.mem.sliceAsBytes(codes) });
        std.debug.print("RESULT drama3 encode OK frames={d} seconds={d:.2} ms={d:.1} out={s}\n", .{ codes.len, @as(f64, @floatFromInt(au.samples.len)) / @as(f64, @floatFromInt(au.rate)), @as(f64, @floatFromInt(d3.engine.nowNs(io) - t0)) / 1e6, out });
        return 0;
    }

    const tpath = try std.fs.path.join(arena, &.{ a.model, "tokenizer.json" });
    var tok = try core.tokenizer.loadTokenizer(io, gpa, tpath);
    defer tok.deinit();
    const t_load = d3.engine.nowNs(io);
    const e = try d3.Engine.create(gpa, io, a.model, a.opts);
    defer e.destroy();
    var tts: ?*d3.tts.Tts = null;
    if (a.codec) |cp| tts = try d3.tts.Tts.create(gpa, io, e, cp, @max(if (a.replay) @as(usize, 16) else 1, @max(a.chunk, a.first_chunk)));
    defer if (tts) |t| t.destroy();
    const load_s = @as(f64, @floatFromInt(d3.engine.nowNs(io) - t_load)) / 1e9;
    std.log.info("drama3: loaded in {d:.1} s (codec {s})", .{ load_s, if (tts != null) "native" else "none" });
    const req: Request = .{ .text = a.text, .rows = a.rows, .ref_text = a.ref_text, .ref_codes = a.ref_codes, .ref_audio = a.ref_audio, .wav = a.wav, .pcm = a.pcm, .frames_out = a.out, .rows_out = a.rows_out, .smp = a.smp, .max_frames = a.max_frames };
    var r: usize = 0;
    while (r < a.repeat) : (r += 1) try runRequest(gpa, io, a, e, tts, &tok, req);
    return 0;
}

const Request = struct {
    text: ?[]const u8 = null,
    rows: ?[]const u8 = null,
    ref_text: ?[]const u8 = null,
    ref_codes: ?[]const u8 = null,
    ref_audio: ?[]const u8 = null,
    wav: ?[]const u8 = null,
    pcm: ?[]const u8 = null, // raw s16le 44.1 kHz mono as it streams ("-" is stdout)
    frames_out: ?[]const u8 = null,
    rows_out: ?[]const u8 = null,
    smp: d3.Sampling = .{},
    max_frames: usize = 1500,
};

extern "c" fn fopen(path: [*:0]const u8, mode: [*:0]const u8) ?*anyopaque;
extern "c" fn fwrite(p: [*]const u8, size: usize, n: usize, f: *anyopaque) usize;
extern "c" fn fseek(f: *anyopaque, off: c_long, whence: c_int) c_int;
extern "c" fn fflush(f: ?*anyopaque) c_int;
extern "c" fn fclose(f: *anyopaque) c_int;
extern "c" fn fdopen(fd: c_int, mode: [*:0]const u8) ?*anyopaque;

/// One fwrite that fails loudly: a short write is an error, never a sample counted.
fn writeAll(f: *anyopaque, b: []const u8) !void {
    if (fwrite(b.ptr, 1, b.len, f) != b.len) return error.AudioWrite;
}

/// Where the samples go as they stream: a WAV file whose header is patched at the end, and / or raw PCM.
const AudioOut = struct {
    gpa: std.mem.Allocator,
    wav: ?*anyopaque = null,
    pcm: ?*anyopaque = null,
    samples: usize = 0,
    chunks: usize = 0,
    t0: i128 = 0,
    io: std.Io,
    verbose: bool,
    buf: []u8 = &.{},

    fn open(gpa: std.mem.Allocator, io: std.Io, wav_path: ?[]const u8, pcm_path: ?[]const u8, verbose: bool) !AudioOut {
        var o: AudioOut = .{ .gpa = gpa, .io = io, .verbose = verbose, .t0 = d3.engine.nowNs(io) };
        errdefer o.abandon();
        if (wav_path) |p| {
            const z = try gpa.dupeSentinel(u8, p, 0);
            defer gpa.free(z);
            o.wav = fopen(z.ptr, "wb") orelse return error.WavOpen;
            var h: [44]u8 = undefined;
            d3.wav.header(&h, 44100, 0x7FFFFFF0 / 2);
            try writeAll(o.wav.?, &h);
        }
        if (pcm_path) |p| {
            if (std.mem.eql(u8, p, "-")) o.pcm = fdopen(1, "wb") else {
                const z = try gpa.dupeSentinel(u8, p, 0);
                defer gpa.free(z);
                o.pcm = fopen(z.ptr, "wb") orelse return error.PcmOpen;
            }
        }
        return o;
    }

    fn put(ctx: *anyopaque, x: []const f32) anyerror!void {
        const o: *AudioOut = @ptrCast(@alignCast(ctx));
        if (o.buf.len < x.len * 2) {
            o.gpa.free(o.buf);
            o.buf = try o.gpa.alloc(u8, x.len * 2);
        }
        const b = o.buf[0 .. x.len * 2];
        d3.wav.pcm16(x, b);
        if (o.wav) |f| try writeAll(f, b);
        if (o.pcm) |f| {
            try writeAll(f, b);
            if (fflush(f) != 0) return error.AudioWrite;
        }
        o.samples += x.len;
        o.chunks += 1;
        if (o.verbose) std.debug.print("CHUNK {d} samples={d} total_s={d:.3} at_ms={d:.1}\n", .{ o.chunks, x.len, @as(f64, @floatFromInt(o.samples)) / 44100.0, @as(f64, @floatFromInt(d3.engine.nowNs(o.io) - o.t0)) / 1e6 });
    }

    /// Patch the WAV header and close; any failure is returned before the request reports success.
    fn close(o: *AudioOut) !void {
        defer o.abandon();
        if (o.wav) |f| {
            o.wav = null;
            var h: [44]u8 = undefined;
            d3.wav.header(&h, 44100, o.samples);
            const ok = fseek(f, 0, 0) == 0 and fwrite(&h, 1, 44, f) == 44;
            if (fclose(f) != 0 or !ok) return error.AudioWrite;
        }
        if (o.pcm) |f| if (fflush(f) != 0) return error.AudioWrite;
    }

    /// Release without reporting: the path taken when the request already failed.
    fn abandon(o: *AudioOut) void {
        if (o.wav) |f| _ = fclose(f);
        o.wav = null;
        o.gpa.free(o.buf);
        o.buf = &.{};
    }
};

fn runRequest(gpa: std.mem.Allocator, io: std.Io, a: Args, e: *d3.Engine, tts: ?*d3.tts.Tts, tok: *const core.tokenizer.Tokenizer, q: Request) !void {
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const ar = arena_state.allocator();
    const t_req = d3.engine.nowNs(io);
    // prompt rows, with a reference given as audio encoded by the native codec
    var rows: []const [W]i32 = undefined;
    var ref_frames: usize = 0;
    var encode_ms: f64 = 0;
    if (q.rows) |path| {
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, ar, .limited(1 << 30));
        if (bytes.len % (W * 4) != 0) return error.BadRows;
        rows = @alignCast(std.mem.bytesAsSlice([W]i32, bytes));
    } else {
        const text = q.text orelse return error.TextRequired;
        var refs: [1]d3.prompt.Reference = undefined;
        var nref: usize = 0;
        if (q.ref_codes) |p| {
            refs[0] = .{ .transcript = q.ref_text orelse return error.RefTextRequired, .codes = try readCodes(ar, io, p) };
            nref = 1;
        } else if (q.ref_audio) |p| {
            const t = tts orelse return error.CodecRequired;
            const e0 = d3.engine.nowNs(io);
            const codes = try t.encodeWav(try std.Io.Dir.cwd().readFileAlloc(io, p, ar, .limited(1 << 30)));
            encode_ms = @as(f64, @floatFromInt(d3.engine.nowNs(io) - e0)) / 1e6;
            refs[0] = .{ .transcript = q.ref_text orelse return error.RefTextRequired, .codes = codes };
            nref = 1;
        }
        if (nref > 0) ref_frames = refs[0].codes.len;
        rows = try d3.prompt.build(ar, tok, text, refs[0..nref]);
    }
    if (a.replay) try d3.replay.verify(gpa, io, e, tts orelse return error.CodecRequired, rows, q.smp);
    if (a.speed_probe) {
        if (@hasDecl(d3, "speed_probe")) try d3.speed_probe.verify(e) else return error.CudaOnlyProbe;
    }
    if (q.rows_out) |p| try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = p, .data = std.mem.sliceAsBytes(rows) });
    var frames: []const [W]i32 = undefined;
    var st: d3.tts.Stats = undefined;
    if (tts) |t| {
        var out = try AudioOut.open(ar, io, q.wav, q.pcm, a.verbose_chunks);
        out.t0 = t_req;
        errdefer out.abandon();
        const r = try t.speak(io, rows, q.smp, q.max_frames, .{ .first_chunk = a.first_chunk, .chunk = a.chunk }, .{ .ctx = &out, .audio = AudioOut.put });
        errdefer gpa.free(r.frames);
        try out.close();
        frames = r.frames;
        st = r.stats;
    } else {
        const res = try e.generate(io, rows, q.smp, q.max_frames, a.graphs, a.sync_every);
        frames = res.frames;
        st = .{ .frames = res.frames.len, .eos = res.eos, .prompt_rows = rows.len, .prefill_ms = res.prefill_ms, .first_audio_ms = 0, .total_ms = res.prefill_ms + res.decode_ms, .audio_s = @as(f64, @floatFromInt(res.frames.len)) * 2048.0 / 44100.0 };
    }
    defer gpa.free(frames.ptr[0..frames.len]);
    if (q.frames_out) |p| try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = p, .data = std.mem.sliceAsBytes(frames) });
    const total_ms = @as(f64, @floatFromInt(d3.engine.nowNs(io) - t_req)) / 1e6;
    std.debug.print("RESULT drama3 OK frames={d} eos={} audio_s={d:.2} prompt_rows={d} ref_frames={d} encode_ms={d:.1} prefill_ms={d:.1} first_audio_ms={d:.1} total_ms={d:.1} fps={d:.2} rtf={d:.2} device_peak_gb={d:.2}\n", .{ st.frames, st.eos, st.audio_s, rows.len, ref_frames, encode_ms, st.prefill_ms, st.first_audio_ms + encode_ms, total_ms, st.fps(), st.fps() / 21.533, @as(f64, @floatFromInt(d3.engine.peakBytes())) / 1e9 });
}

/// Reference codes: raw int32 [frames, 10].
pub fn readCodes(gpa: std.mem.Allocator, io: std.Io, path: []const u8) ![]const [10]i32 {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 28));
    if (!std.mem.endsWith(u8, path, ".npy")) {
        if (bytes.len % 40 != 0) return error.BadCodes;
        return @alignCast(std.mem.bytesAsSlice([10]i32, bytes));
    }
    return error.NpyCodesUnsupported; // --encode writes the raw form
}
