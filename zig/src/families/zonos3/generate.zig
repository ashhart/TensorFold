//! zonos3-generate: ZONOS3 Mini's backbone on one CUDA GPU against the PyTorch reference's dumps.
const std = @import("std");
const z3 = @import("zonos3");
const cuda = @import("cuda");
const core = @import("core");
const encoder_checks = @import("check_encoder.zig");

const cfg = z3.config;
const W = cfg.frame_width;
const D = cfg.dim;

const usage =
    \\usage: zonos3-generate --model LM_DIR --check DUMP_DIR [--max-rows 64] [--max-ctx 4096]
    \\       zonos3-generate --model LM_DIR --frames-check GEN_DIR [--max-frames 2000]
    \\       zonos3-generate --codec-check DUMP_DIR --generator GENERATOR.pt --head HEAD.pt
    \\       zonos3-generate --encoder-check DUMP_DIR --generator GENERATOR.pt
    \\       zonos3-generate --features-check DUMP_DIR
    \\       zonos3-generate --teacher-check DUMP_DIR --teacher TEACHER_DIR
    \\       zonos3-generate --checkpoint RELEASE_DIR --text "..." --wav OUT.wav [--voice-embedding E.npy | --voice-codes C.npy]
    \\                       [--seed 0] [--temperature 0.75] [--top-k 80] [--top-p 0.69] [--max-frames 2000] [--decoder ID]
    \\                       [--silence-frames 3] [--reference GEN_DIR]
    \\
    \\--check runs DUMP_DIR/input_ids.npy (prompt rows [n, 11]) with DUMP_DIR/input_ids.cond.json through the backbone
    \\and compares every sublayer with the reference's .npy captures (tools/dump_golden.py lm of Zyphra/ZONOS3-Mini).
    \\--frames-check generates greedily (temperature 0, the other sampling defaults) after GEN_DIR's prompt and compares
    \\the frames with GEN_DIR/rows.npy, the reference's generation of the same request.
    \\--codec-check decodes DUMP_DIR/codes.npy ([frames, 10]) with the codec decoder and compares its captures and
    \\audio with the reference's (dump_codec.py); --generator is autoencoder/generator.pt, --head a decoder checkpoint.
    \\--text speaks the text as typed (line breaks collapsed, no number or date expansion) in the voice of a 192-d
    \\speaker embedding (float32 .npy) or of reference codes ([frames, 10] .npy); RELEASE_DIR is a snapshot of
    \\Zyphra/ZONOS3-Mini. With --reference the prompt rows, frames and audio are compared with GEN_DIR's (dump_rows.py).
    \\--encoder-check encodes DUMP_DIR/audio.npy (44.1 kHz, one lane) and compares the acoustic branch's captures with
    \\the reference's (tools/dump_golden.py codec-encoder, a CUDA bf16 run, so the match is to bf16 rounding).
    \\--features-check computes the SSL teacher's input features of DUMP_DIR/lane.0.teacher_audio.npy on the host and
    \\compares them with DUMP_DIR/lane.0.ssl.features.npy. The last stderr line is "RESULT zonos3 <mode> ...".
    \\
;

pub const Check = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    ctx: *anyopaque,
    get: *const fn (ctx: *anyopaque, ptr: u64, out: []u8) anyerror!void,
    dir: []const u8,
    n: usize = 0,
    fails: usize = 0,
    missing: usize = 0,
    worst: f64 = 1.0,
    worst_rel: f64 = 0,
    audio_snr_db: f64 = 0,
    prefix: bool = false,
    floor: f64 = 0.9999,

    pub fn probe(ctx: *anyopaque, name: []const u8, ptr: u64, rows: usize, width: usize) anyerror!void {
        const c: *Check = @ptrCast(@alignCast(ctx));
        var nb: [96]u8 = undefined;
        const path = try std.fs.path.join(c.gpa, &.{ c.dir, try std.fmt.bufPrint(&nb, "{s}.npy", .{name}) });
        defer c.gpa.free(path);
        const bytes = std.Io.Dir.cwd().readFileAlloc(c.io, path, c.gpa, .limited(1 << 30)) catch {
            c.missing += 1;
            std.debug.print("  {s:<20} (no reference tensor)\n", .{name});
            return;
        };
        defer c.gpa.free(bytes);
        const arr = try z3.npy.parse(bytes);
        // a reference lane is padded to its full length: the rows this run has are its first
        if (arr.numel() < rows * width or (!c.prefix and arr.numel() != rows * width)) return error.ReferenceShape;
        const ref = try arr.floats(c.gpa);
        defer c.gpa.free(ref);
        const mine = try c.gpa.alloc(f32, rows * width);
        defer c.gpa.free(mine);
        try c.get(c.ctx, ptr, std.mem.sliceAsBytes(mine));
        var worst_row: f64 = 1;
        var at: usize = 0;
        var max_d: f64 = 0;
        var max_ref: f64 = 0;
        for (0..rows) |r| {
            var d: f64 = 0;
            var a2: f64 = 0;
            var b2: f64 = 0;
            for (mine[r * width ..][0..width], ref[r * width ..][0..width]) |x, y| {
                d += @as(f64, x) * y;
                a2 += @as(f64, x) * x;
                b2 += @as(f64, y) * y;
                max_d = @max(max_d, @abs(@as(f64, x) - y));
                max_ref = @max(max_ref, @abs(@as(f64, y)));
            }
            const cos: f64 = if (a2 > 0 and b2 > 0) d / @sqrt(a2 * b2) else if (a2 == b2) 1.0 else 0.0;
            if (cos < worst_row) {
                worst_row = cos;
                at = r;
            }
        }
        const rel = if (max_ref > 0) max_d / max_ref else max_d;
        const ok = worst_row >= c.floor;
        c.n += 1;
        c.fails += @intFromBool(!ok);
        c.worst = @min(c.worst, worst_row);
        c.worst_rel = @max(c.worst_rel, rel);
        if (std.mem.eql(u8, name, "audio")) {
            var noise: f64 = 0;
            var signal: f64 = 0;
            for (mine, ref) |x, y| {
                noise += (@as(f64, x) - y) * (@as(f64, x) - y);
                signal += @as(f64, y) * y;
            }
            c.audio_snr_db = 10.0 * @log10(signal / @max(noise, 1e-300));
        }
        std.debug.print("  {s:<20} rows {d:>4} worst-row cos {d:.7} (row {d}) max|d| {e:.3} of {e:.3} {s}\n", .{ name, rows, worst_row, at, max_d, max_ref, if (ok) "ok" else "LOW" });
    }
};

/// The dump's prompt: rows and the rows a speaker or condition vector replaces.
fn loadPrompt(gpa: std.mem.Allocator, io: std.Io, e: *z3.Engine, dir: []const u8, rows: *[][W]i32, replaced: *std.ArrayList(z3.Replace)) !void {
    const ids_path = try std.fs.path.join(gpa, &.{ dir, "input_ids.npy" });
    defer gpa.free(ids_path);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, ids_path, gpa, .limited(1 << 28));
    defer gpa.free(bytes);
    const arr = try z3.npy.parse(bytes);
    if (arr.rank != 2 or arr.shape[1] != W) return error.BadPromptRows;
    const flat = try arr.ints(gpa);
    defer gpa.free(flat);
    rows.* = try gpa.alloc([W]i32, arr.shape[0]);
    for (rows.*, 0..) |*r, i| r.* = flat[i * W ..][0..W].*;

    const cond_path = try std.fs.path.join(gpa, &.{ dir, "input_ids.cond.json" });
    defer gpa.free(cond_path);
    const text = try std.Io.Dir.cwd().readFileAlloc(io, cond_path, gpa, .limited(1 << 22));
    defer gpa.free(text);
    var tree = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
    defer tree.deinit();
    const o = tree.value.object;
    const speaker_row = o.get("speaker_row").?.integer;
    if (speaker_row >= 0) {
        const v = o.get("speaker_embedding").?.array.items;
        if (v.len != cfg.speaker_dim) return error.BadSpeakerEmbedding;
        var vec: [cfg.speaker_dim]f32 = undefined;
        for (&vec, v) |*x, j| x.* = switch (j) {
            .float => |f| @floatCast(f),
            .integer => |n| @floatFromInt(n),
            else => return error.BadSpeakerEmbedding,
        };
        var r: z3.Replace = .{ .row = @intCast(speaker_row), .vec = undefined };
        e.w.speakerRow(&vec, &r.vec);
        try replaced.append(gpa, r);
    }
    const first: usize = @intCast(o.get("condition_row").?.integer);
    for (o.get("condition_buckets").?.array.items, 0..) |b, i| {
        const bucket: usize = @intCast(b.integer);
        if (i >= cfg.conditions or bucket > e.c.buckets[i]) return error.BadConditionBucket;
        var r: z3.Replace = .{ .row = first + i, .vec = undefined };
        @memcpy(&r.vec, e.w.conditions[i][bucket * D ..][0..D]);
        try replaced.append(gpa, r);
    }
}

pub fn nowNs(io: std.Io) i128 {
    return @intCast(std.Io.Clock.awake.now(io).toNanoseconds());
}

fn framesCheck(gpa: std.mem.Allocator, io: std.Io, e: *z3.Engine, dir: []const u8, prompt: []const [W]i32, replaced: []const z3.Replace, max_frames: usize) !u8 {
    const path = try std.fs.path.join(gpa, &.{ dir, "rows.npy" });
    defer gpa.free(path);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 28));
    defer gpa.free(bytes);
    const arr = try z3.npy.parse(bytes);
    if (arr.rank != 2 or arr.shape[1] != W) return error.BadReferenceRows;
    const want = try arr.ints(gpa);
    defer gpa.free(want);
    const t0 = nowNs(io);
    const r = try z3.generation.generate(gpa, e, prompt, replaced, .{ .temperature = 0 }, max_frames);
    defer gpa.free(r.rows);
    const seconds = @as(f64, @floatFromInt(nowNs(io) - t0)) / 1e9;
    const n = @min(r.rows.len, arr.shape[0]);
    var same: usize = 0;
    while (same < n and std.mem.eql(i32, &r.rows[same], want[same * W ..][0..W])) same += 1;
    const pass = same == arr.shape[0] and r.rows.len == arr.shape[0];
    if (!pass and same < n) std.debug.print("first difference at frame {d}: mine {any} reference {any}\n", .{ same, r.rows[same], want[same * W ..][0..W] });
    std.debug.print("RESULT zonos3 frames {s} identical={d} mine={d} reference={d} eos_frame={?d} prompt_rows={d} seconds={d:.3} frames_per_s={d:.1}\n", .{ if (pass) "PASS" else "FAIL", same, r.rows.len, arr.shape[0], r.eos_frame, prompt.len, seconds, @as(f64, @floatFromInt(r.rows.len)) / seconds });
    return if (pass) 0 else 1;
}

fn codecCheck(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, generator: []const u8, head: []const u8) !u8 {
    const path = try std.fs.path.join(gpa, &.{ dir, "codes.npy" });
    defer gpa.free(path);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 28));
    defer gpa.free(bytes);
    const arr = try z3.npy.parse(bytes);
    if (arr.rank != 2 or arr.shape[1] != 10) return error.BadCodes;
    const flat = try arr.ints(gpa);
    defer gpa.free(flat);
    const codes = try gpa.alloc([10]i32, arr.shape[0]);
    defer gpa.free(codes);
    for (codes, 0..) |*r, j| r.* = flat[j * 10 ..][0..10].*;

    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var stream = try cuda.Stream.init(&driver, true);
    defer stream.deinit();
    const t_load = nowNs(io);
    const codec = try z3.Codec.create(gpa, io, &driver, stream, generator, head, .{ .max_frames = codes.len, .pass_frames = codes.len });
    defer codec.destroy();
    const Dec = struct {
        fn get(c: *anyopaque, ptr: u64, out: []u8) anyerror!void {
            try @as(*z3.Codec, @ptrCast(@alignCast(c))).download(ptr, out);
        }
    };
    var c: Check = .{ .gpa = gpa, .io = io, .ctx = codec, .get = Dec.get, .dir = dir };
    std.debug.print("zonos3 codec check: {d} frames, {d} MiB of weights, loaded in {d:.1} s\n", .{ codes.len, codec.w.bytes >> 20, @as(f64, @floatFromInt(nowNs(io) - t_load)) / 1e9 });
    const probe: z3.codec.Probe = .{ .ctx = &c, .f = Check.probe };
    try codec.decode(codes, probe);
    const pcm = try codec.finish(gpa, probe);
    gpa.free(pcm);
    // the same frames in uneven chunks, the settled samples taken after each, then timed whole
    try codec.reset();
    var settled: std.ArrayList(f32) = .empty;
    defer settled.deinit(gpa);
    var at: usize = 0;
    var step: usize = 1;
    while (at < codes.len) : (step = step * 2 + 1) {
        const n = @min(step, codes.len - at);
        try codec.decode(codes[at..][0..n], null);
        at += n;
        const part = try codec.emit(gpa, at == codes.len, null);
        defer gpa.free(part);
        try settled.appendSlice(gpa, part);
    }
    const chunked = settled.items;
    try codec.reset();
    const t0 = nowNs(io);
    try codec.decode(codes, null);
    const whole = try codec.finish(gpa, null);
    defer gpa.free(whole);
    const seconds = @as(f64, @floatFromInt(nowNs(io) - t0)) / 1e9;
    const same = std.mem.eql(u8, std.mem.sliceAsBytes(chunked), std.mem.sliceAsBytes(whole));
    const audio_s = @as(f64, @floatFromInt(whole.len)) / z3.codec.sample_rate;
    const pass = c.fails == 0 and c.n > 0 and c.missing == 0 and same;
    std.debug.print("RESULT zonos3 codec {s} probes={d} low={d} missing={d} worst_row_cos={d:.7} audio_snr_db={d:.1} chunked_equals_whole={} audio_s={d:.2} decode_s={d:.3} x_realtime={d:.1}\n", .{ if (pass) "PASS" else "FAIL", c.n, c.fails, c.missing, c.worst, c.audio_snr_db, same, audio_s, seconds, audio_s / seconds });
    return if (pass) 0 else 1;
}

const Speak = struct {
    checkpoint: []const u8 = "",
    text: []const u8 = "",
    wav: ?[]const u8 = null,
    embedding: ?[]const u8 = null,
    codes: ?[]const u8 = null,
    decoder: ?[]const u8 = null,
    reference: ?[]const u8 = null,
    sampling: z3.Sampling = .{},
    max_frames: usize = 2000,
    silence_frames: usize = 3,
};

pub fn readNpy(gpa: std.mem.Allocator, io: std.Io, path: []const u8) !struct { bytes: []u8, arr: z3.npy.Array } {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 28));
    return .{ .bytes = bytes, .arr = try z3.npy.parse(bytes) };
}

fn speak(gpa: std.mem.Allocator, io: std.Io, a: Speak) !u8 {
    var rel = try z3.release.open(gpa, io, a.checkpoint, a.decoder);
    defer rel.deinit();
    var tok = try core.tokenizer.loadTokenizer(io, gpa, rel.tokenizer);
    defer tok.deinit();
    const t_load = nowNs(io);
    const e = try z3.Engine.create(gpa, io, rel.lm, .{ .max_ctx = @min(z3.config.max_seqlen, a.max_frames + 2048) });
    defer e.destroy();
    const codec = try z3.Codec.create(gpa, io, &e.driver, e.stream, rel.generator, rel.head, .{ .max_frames = a.max_frames + 64 });
    defer codec.destroy();
    const load_s = @as(f64, @floatFromInt(nowNs(io) - t_load)) / 1e9;

    // the prompt
    const text = try z3.prompt.collapseLineBreaks(gpa, a.text);
    defer gpa.free(text);
    const ids = try tok.encode(gpa, text);
    defer gpa.free(ids);
    var codes: [][z3.config.codebooks]i32 = &.{};
    defer gpa.free(codes);
    var speaker: z3.prompt.Speaker = .none;
    var vec: [z3.config.speaker_dim]f32 = undefined;
    if (a.embedding) |path| {
        const f = try readNpy(gpa, io, path);
        defer gpa.free(f.bytes);
        if (f.arr.numel() != z3.config.speaker_dim or f.arr.kind != .f32) return error.BadSpeakerEmbedding;
        const v = try f.arr.floats(gpa);
        defer gpa.free(v);
        vec = v[0..z3.config.speaker_dim].*;
        speaker = .embedding;
    } else if (a.codes) |path| {
        const f = try readNpy(gpa, io, path);
        defer gpa.free(f.bytes);
        if (f.arr.rank != 2 or f.arr.shape[1] != z3.config.codebooks) return error.BadReferenceCodes;
        const flat = try f.arr.ints(gpa);
        defer gpa.free(flat);
        codes = try gpa.alloc([z3.config.codebooks]i32, f.arr.shape[0]);
        for (codes, 0..) |*r, j| r.* = flat[j * z3.config.codebooks ..][0..z3.config.codebooks].*;
        speaker = .{ .reference = codes };
    }
    var p = try z3.prompt.build(gpa, ids, speaker, rel.silence, a.silence_frames);
    defer p.deinit(gpa);
    var replaced: std.ArrayList(z3.Replace) = .empty;
    defer replaced.deinit(gpa);
    if (p.speaker_row) |row| {
        var r: z3.Replace = .{ .row = row, .vec = undefined };
        e.w.speakerRow(&vec, &r.vec);
        try replaced.append(gpa, r);
    }
    // every condition unknown: the last row of its table
    for (0..z3.config.conditions) |c| {
        var r: z3.Replace = .{ .row = p.condition_row + c, .vec = undefined };
        @memcpy(&r.vec, e.w.conditions[c][e.c.buckets[c] * D ..][0..D]);
        try replaced.append(gpa, r);
    }

    const t0 = nowNs(io);
    const r = try z3.generation.generate(gpa, e, p.rows, replaced.items, a.sampling, a.max_frames);
    defer gpa.free(r.rows);
    const t1 = nowNs(io);
    const frames = try z3.speech.frames(gpa, p.rows[p.rows.len - p.prefill ..], r.rows, r.eos_frame);
    defer gpa.free(frames);
    try codec.decode(frames, null);
    const pcm = try codec.finish(gpa, null);
    defer gpa.free(pcm);
    z3.speech.output(pcm, rel.headroom_db);
    const t2 = nowNs(io);
    if (a.wav) |path| {
        const bytes = try z3.speech.wav(gpa, pcm, z3.codec.sample_rate);
        defer gpa.free(bytes);
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = bytes });
    }
    const audio_s = @as(f64, @floatFromInt(pcm.len)) / z3.codec.sample_rate;
    const gen_s = @as(f64, @floatFromInt(t1 - t0)) / 1e9;
    const dec_s = @as(f64, @floatFromInt(t2 - t1)) / 1e9;
    var verdict: []const u8 = "OK";
    if (a.reference) |dir| {
        // the reference's prompt, frames and audio for the same request
        var want_rows: [][W]i32 = &.{};
        var want_replaced: std.ArrayList(z3.Replace) = .empty;
        try loadPrompt(gpa, io, e, dir, &want_rows, &want_replaced);
        const same_prompt = want_rows.len == p.rows.len and std.mem.eql(u8, std.mem.sliceAsBytes(want_rows), std.mem.sliceAsBytes(p.rows));
        const rows_path = try std.fs.path.join(gpa, &.{ dir, "rows.npy" });
        const f = try readNpy(gpa, io, rows_path);
        const want = try f.arr.ints(gpa);
        var same: usize = 0;
        while (same < @min(r.rows.len, f.arr.shape[0]) and std.mem.eql(i32, &r.rows[same], want[same * W ..][0..W])) same += 1;
        const audio_path = try std.fs.path.join(gpa, &.{ dir, "audio.f32" });
        const raw = try std.Io.Dir.cwd().readFileAlloc(io, audio_path, gpa, .limited(1 << 30));
        const ref = std.mem.bytesAsSlice(f32, @as([]align(1) const u8, raw));
        var noise: f64 = 0;
        var signal: f64 = 0;
        var max_d: f64 = 0;
        for (pcm[0..@min(pcm.len, ref.len)], ref[0..@min(pcm.len, ref.len)]) |x, y| {
            noise += (@as(f64, x) - y) * (@as(f64, x) - y);
            signal += @as(f64, y) * y;
            max_d = @max(max_d, @abs(@as(f64, x) - y));
        }
        const snr = 10.0 * @log10(signal / @max(noise, 1e-300));
        const pass = same_prompt and same == f.arr.shape[0] and r.rows.len == f.arr.shape[0] and pcm.len == ref.len and snr >= 60;
        verdict = if (pass) "PASS" else "FAIL";
        std.debug.print("reference: prompt_rows_equal={} frames_identical={d}/{d} samples={d}/{d} audio_snr_db={d:.1} max_abs_diff={e:.3}\n", .{ same_prompt, same, f.arr.shape[0], pcm.len, ref.len, snr, max_d });
    }
    std.debug.print("RESULT zonos3 speech {s} text_tokens={d} prompt_rows={d} frames={d} eos_frame={?d} audio_s={d:.2} load_s={d:.1} generate_s={d:.3} decode_s={d:.3} x_realtime={d:.1}\n", .{ verdict, ids.len, p.rows.len, r.rows.len, r.eos_frame, audio_s, load_s, gen_s, dec_s, audio_s / (gen_s + dec_s) });
    return if (std.mem.eql(u8, verdict, "FAIL")) 1 else 0;
}

pub fn main(init: std.process.Init) !u8 {
    const arena = init.arena.allocator();
    const gpa = std.heap.c_allocator;
    const io = init.io;
    const argv = try init.minimal.args.toSlice(arena);
    var model: []const u8 = "";
    var check: ?[]const u8 = null;
    var frames_check: ?[]const u8 = null;
    var codec_check: ?[]const u8 = null;
    var encoder_check: ?[]const u8 = null;
    var features_check: ?[]const u8 = null;
    var teacher_check: ?[]const u8 = null;
    var teacher_dir: []const u8 = "";
    var generator: []const u8 = "";
    var head: []const u8 = "";
    var sp: Speak = .{};
    var max_frames: usize = 2000;
    var opts: z3.Options = .{};
    var i: usize = 1;
    while (i + 1 < argv.len) : (i += 2) {
        const k = argv[i];
        const v = argv[i + 1];
        if (std.mem.eql(u8, k, "--model")) model = v else if (std.mem.eql(u8, k, "--check")) check = v else if (std.mem.eql(u8, k, "--frames-check")) frames_check = v else if (std.mem.eql(u8, k, "--codec-check")) codec_check = v else if (std.mem.eql(u8, k, "--encoder-check")) encoder_check = v else if (std.mem.eql(u8, k, "--features-check")) features_check = v else if (std.mem.eql(u8, k, "--teacher-check")) teacher_check = v else if (std.mem.eql(u8, k, "--teacher")) teacher_dir = v else if (std.mem.eql(u8, k, "--generator")) generator = v else if (std.mem.eql(u8, k, "--checkpoint")) sp.checkpoint = v else if (std.mem.eql(u8, k, "--text")) sp.text = v else if (std.mem.eql(u8, k, "--wav")) sp.wav = v else if (std.mem.eql(u8, k, "--voice-embedding")) sp.embedding = v else if (std.mem.eql(u8, k, "--voice-codes")) sp.codes = v else if (std.mem.eql(u8, k, "--decoder")) sp.decoder = v else if (std.mem.eql(u8, k, "--reference")) sp.reference = v else if (std.mem.eql(u8, k, "--seed")) sp.sampling.seed = try std.fmt.parseInt(u64, v, 10) else if (std.mem.eql(u8, k, "--temperature")) sp.sampling.temperature = try std.fmt.parseFloat(f32, v) else if (std.mem.eql(u8, k, "--top-k")) sp.sampling.top_k = try std.fmt.parseInt(u32, v, 10) else if (std.mem.eql(u8, k, "--top-p")) sp.sampling.top_p = try std.fmt.parseFloat(f32, v) else if (std.mem.eql(u8, k, "--silence-frames")) sp.silence_frames = try std.fmt.parseInt(usize, v, 10) else if (std.mem.eql(u8, k, "--head")) head = v else if (std.mem.eql(u8, k, "--max-frames")) max_frames = try std.fmt.parseInt(usize, v, 10) else if (std.mem.eql(u8, k, "--max-rows")) opts.max_rows = try std.fmt.parseInt(usize, v, 10) else if (std.mem.eql(u8, k, "--max-ctx")) opts.max_ctx = try std.fmt.parseInt(usize, v, 10) else {
            std.debug.print("unknown flag {s}\n{s}", .{ k, usage });
            return 2;
        }
    }
    if (teacher_check) |dir| return if (i == argv.len and teacher_dir.len > 0) encoder_checks.teacher(gpa, io, dir, teacher_dir) else 2;
    if (features_check) |dir| return if (i == argv.len) encoder_checks.features(gpa, io, dir) else 2;
    if (encoder_check) |dir| {
        if (i != argv.len or generator.len == 0) {
            std.debug.print("{s}", .{usage});
            return 2;
        }
        return encoder_checks.acoustic(gpa, io, dir, generator);
    }
    if (codec_check) |dir| {
        if (i != argv.len or generator.len == 0 or head.len == 0) {
            std.debug.print("{s}", .{usage});
            return 2;
        }
        return codecCheck(gpa, io, dir, generator, head);
    }
    if (sp.checkpoint.len > 0) {
        if (i != argv.len or sp.text.len == 0 or (sp.embedding != null and sp.codes != null)) {
            std.debug.print("{s}", .{usage});
            return 2;
        }
        sp.max_frames = max_frames;
        return speak(gpa, io, sp);
    }
    const dir = check orelse frames_check orelse {
        std.debug.print("{s}", .{usage});
        return 2;
    };
    if (model.len == 0 or i != argv.len or (check != null and frames_check != null)) {
        std.debug.print("{s}", .{usage});
        return 2;
    }
    const e = try z3.Engine.create(gpa, io, model, opts);
    defer e.destroy();
    var rows: [][W]i32 = &.{};
    var replaced: std.ArrayList(z3.Replace) = .empty;
    try loadPrompt(gpa, io, e, dir, &rows, &replaced);
    if (frames_check != null) return framesCheck(gpa, io, e, dir, rows, replaced.items, max_frames);
    const Lm = struct {
        fn get(ctx: *anyopaque, ptr: u64, out: []u8) anyerror!void {
            try @as(*z3.Engine, @ptrCast(@alignCast(ctx))).download(ptr, out);
        }
    };
    var c: Check = .{ .gpa = gpa, .io = io, .ctx = e, .get = Lm.get, .dir = dir };
    std.debug.print("zonos3 check: {d} prompt rows, {d} replaced, {d} MiB of weights\n", .{ rows.len, replaced.items.len, e.w.bytes >> 20 });
    try e.run(rows, replaced.items, .{ .ctx = &c, .f = Check.probe });
    const pass = c.fails == 0 and c.n > 0;
    std.debug.print("RESULT zonos3 check {s} probes={d} low={d} missing={d} worst_row_cos={d:.7} worst_rel_diff={e:.3}\n", .{ if (pass) "PASS" else "FAIL", c.n, c.fails, c.missing, c.worst, c.worst_rel });
    return if (pass) 0 else 1;
}
