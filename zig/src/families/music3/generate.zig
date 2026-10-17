//! music3-generate: MiniMax Music 3 to a song on one CUDA GPU, a --serve mode, parity dumps (MiniMax-Music3 License).
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const m3 = @import("music3");

const cfg = m3.config;

const usage =
    \\usage: music3-generate --caption-file C [--lyrics-file L] --out song.wav [--seconds 60] [--seed 0]
    \\                       [--ar-cfg 1.5] [--top-k 50] [--steps 30] [--dit-cfg 1.7] [--stats stats.json]
    \\                       [--model DIR] [--weights int8|bf16] [--int8-group 128] [--comfy-rope 1] [--flash 1] [--dav-tf32 0] [--graphs 1] [--sync-every 16]
    \\       music3-generate --serve 1 [--model DIR] [--weights int8|bf16]   (one JSON request a line on stdin)
    \\       music3-generate --ids --caption-file C --lyrics-file L          (prompt ids as JSON, no GPU)
    \\       music3-generate --parity-ar REF --out DIR --caption-file C --lyrics-file L [--frames N]
    \\       music3-generate --parity-dit REF --out DIR [--steps 30 --dit-cfg 1.7]
    \\       music3-generate --parity-probe REF --out DIR      music3-generate --parity-dav REF --out DIR
    \\
    \\--model (or $TF_MUSIC3_MODEL) is the folder holding Comfy-Org's three MiniMax-Music3 files; it is required.
    \\--caption / --lyrics take the text itself. WAV out: 16-bit PCM stereo 44.1 kHz.
    \\
;

const built_with = "[music3] MiniMax-Music3 runs locally under the MiniMax-Music3 Community License (show \"MiniMax-Music3\" in a commercial UI).";

const Args = struct {
    model: []const u8 = "",
    caption: ?[]const u8 = null,
    lyrics: ?[]const u8 = null,
    out: ?[]const u8 = null,
    stats: ?[]const u8 = null,
    s: cfg.Settings = .{},
    serve: bool = false,
    ids: bool = false,
    format: m3.weights.Format = .int8,
    comfy_rope: bool = true,
    flash: bool = true,
    dav_tf32: bool = false,
    depth_group: usize = 128,
    graphs: bool = true,
    sync_every: usize = 16,
    parity_ar: ?[]const u8 = null,
    parity_dit: ?[]const u8 = null,
    parity_probe: ?[]const u8 = null,
    parity_dav: ?[]const u8 = null,
    frames: usize = 0,
};

var home: []const u8 = "";

fn expand(a: std.mem.Allocator, p: []const u8) ![]const u8 {
    if (std.mem.startsWith(u8, p, "~/")) return std.mem.concat(a, u8, &.{ home, p[1..] });
    return p;
}

fn readText(a: std.mem.Allocator, io: std.Io, path: []const u8) ![]const u8 {
    return std.Io.Dir.cwd().readFileAlloc(io, try expand(a, path), a, .limited(16 << 20));
}

pub fn main(init: std.process.Init) !u8 {
    const arena = init.arena.allocator();
    const gpa = std.heap.c_allocator;
    const io = init.io;
    home = init.environ_map.get("HOME") orelse "";
    const argv = try init.minimal.args.toSlice(arena);
    var a: Args = .{};
    a.model = init.environ_map.get("TF_MUSIC3_MODEL") orelse "";
    var i: usize = 1;
    while (i < argv.len) : (i += 1) {
        const k = argv[i];
        if (std.mem.eql(u8, k, "--ids")) {
            a.ids = true;
            continue;
        }
        if (i + 1 >= argv.len) {
            std.debug.print("{s}", .{usage});
            return 2;
        }
        const v = argv[i + 1];
        i += 1;
        const eq = std.mem.eql;
        if (eq(u8, k, "--model")) a.model = try expand(arena, v) else if (eq(u8, k, "--caption")) a.caption = v else if (eq(u8, k, "--caption-file")) a.caption = try readText(arena, io, v) else if (eq(u8, k, "--lyrics")) a.lyrics = v else if (eq(u8, k, "--lyrics-file")) a.lyrics = try readText(arena, io, v) else if (eq(u8, k, "--out")) a.out = try expand(arena, v) else if (eq(u8, k, "--stats")) a.stats = try expand(arena, v) else if (eq(u8, k, "--seconds")) a.s.seconds = try std.fmt.parseFloat(f64, v) else if (eq(u8, k, "--seed")) a.s.seed = try std.fmt.parseInt(u64, v, 10) else if (eq(u8, k, "--ar-cfg")) a.s.ar_cfg = try std.fmt.parseFloat(f32, v) else if (eq(u8, k, "--top-k")) a.s.top_k = try std.fmt.parseInt(i32, v, 10) else if (eq(u8, k, "--steps")) a.s.steps = try std.fmt.parseInt(usize, v, 10) else if (eq(u8, k, "--dit-cfg")) a.s.dit_cfg = try std.fmt.parseFloat(f32, v) else if (eq(u8, k, "--serve")) a.serve = !(eq(u8, v, "0") or eq(u8, v, "false") or eq(u8, v, "no")) else if (eq(u8, k, "--weights")) a.format = std.meta.stringToEnum(m3.weights.Format, v) orelse return error.BadWeights else if (eq(u8, k, "--comfy-rope")) a.comfy_rope = !eq(u8, v, "0") else if (eq(u8, k, "--int8-group")) a.depth_group = try std.fmt.parseInt(usize, v, 10) else if (eq(u8, k, "--dav-tf32")) a.dav_tf32 = !eq(u8, v, "0") else if (eq(u8, k, "--flash")) a.flash = !eq(u8, v, "0") else if (eq(u8, k, "--graphs")) a.graphs = !eq(u8, v, "0") else if (eq(u8, k, "--sync-every")) a.sync_every = try std.fmt.parseInt(usize, v, 10) else if (eq(u8, k, "--parity-ar")) a.parity_ar = try expand(arena, v) else if (eq(u8, k, "--parity-dit")) a.parity_dit = try expand(arena, v) else if (eq(u8, k, "--parity-probe")) a.parity_probe = try expand(arena, v) else if (eq(u8, k, "--parity-dav")) a.parity_dav = try expand(arena, v) else if (eq(u8, k, "--frames")) a.frames = try std.fmt.parseInt(usize, v, 10) else {
            std.debug.print("unknown flag {s}\n{s}", .{ k, usage });
            return 2;
        }
    }
    if (a.model.len == 0) {
        std.debug.print("pass --model DIR or set TF_MUSIC3_MODEL\n{s}", .{usage});
        return 2;
    }
    if (a.ids) return idsOnly(gpa, io, a);
    const parity = a.parity_ar != null or a.parity_dit != null or a.parity_probe != null or a.parity_dav != null;
    if (!a.serve and !parity and (a.caption == null or a.out == null)) {
        std.debug.print("{s}", .{usage});
        return 2;
    }
    var opts: m3.Options = .{ .format = a.format, .graphs = a.graphs, .sync_every = a.sync_every, .dit = .{ .comfy_rope = a.comfy_rope, .flash = a.flash }, .dav_tf32 = a.dav_tf32, .depth_group = a.depth_group };
    if (parity) {
        opts.load_ar = a.parity_ar != null;
        opts.load_dit = a.parity_dit != null or a.parity_probe != null;
        opts.load_dav = a.parity_dav != null;
    }
    if (a.serve) return serve(gpa, io, a, opts);
    std.debug.print("{s}\n", .{built_with});
    const e = try m3.Engine.create(gpa, io, a.model, opts);
    defer e.destroy();
    if (parity) return m3.parity.run(gpa, io, e, .{ .ar = a.parity_ar, .dit = a.parity_dit, .probe = a.parity_probe, .dav = a.parity_dav, .out = a.out orelse return error.OutRequired, .caption = a.caption orelse "", .lyrics = a.lyrics orelse "", .frames = a.frames, .steps = a.s.steps, .dit_cfg = a.s.dit_cfg });
    var pc: PrintProgress = .{ .io = io };
    const song = try e.generate(io, a.caption.?, a.lyrics orelse "", a.s, .{ .ctx = &pc, .f = PrintProgress.f });
    defer gpa.free(song.audio);
    try writeWav(gpa, io, a.out.?, song.audio);
    const stats = try statsJson(gpa, e, song, a.s, null);
    defer gpa.free(stats);
    if (a.stats) |p| try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = p, .data = stats });
    std.debug.print("[music3] wrote {s}: {d:.1}s ({d} frames, {s}); AR {d:.1}s ({d:.2} frames/s), DiT {d:.1}s, DAV {d:.1}s, total {d:.1}s, RTF {d:.3}; device peak {d:.2} GB\n", .{ a.out.?, audioS(song), song.frames, if (song.stopped) "model ended it" else "hit the cap", song.ar_ms / 1000, song.frames_per_s, song.dit_ms / 1000, song.dav_ms / 1000, song.total_ms / 1000, song.total_ms / 1000 / @max(audioS(song), 1e-9), song.peak_device_gb });
    std.debug.print("RESULT music3 OK {s}\n", .{stats});
    return 0;
}

fn audioS(song: m3.Song) f64 {
    return @as(f64, @floatFromInt(song.samples)) / 44100.0;
}

const PrintProgress = struct {
    io: std.Io,
    last: i128 = 0,
    fn f(ctx: *anyopaque, stage: []const u8, done: usize, total: usize) void {
        const p: *PrintProgress = @ptrCast(@alignCast(ctx));
        const now = m3.ar.nowNs(p.io);
        if (done != total and now - p.last < 2_000_000_000) return;
        p.last = now;
        std.debug.print("[music3] {s} {d}/{d}\n", .{ stage, done, total });
    }
};

fn writeWav(gpa: std.mem.Allocator, io: std.Io, path: []const u8, audio: []const f32) !void {
    const bytes = try m3.wav.encode(gpa, audio, 2, 44100);
    defer gpa.free(bytes);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = bytes });
}

fn statsJson(gpa: std.mem.Allocator, e: *m3.Engine, song: m3.Song, s: cfg.Settings, extra: ?[]const u8) ![]u8 {
    const aus = audioS(song);
    const gbs = e.frameBytes() * song.frames_per_s / 1e9;
    return std.fmt.allocPrint(gpa, "{{\"engine\":\"tensorfold-music3-cuda\",\"weights\":\"{s}\",\"prompt_tokens\":{d},\"frames\":{d},\"stopped\":{},\"prefill_s\":{d:.3},\"ar_s\":{d:.3},\"frames_per_s\":{d:.2},\"ar_weight_gb_per_frame\":{d:.2},\"ar_achieved_gb_s\":{d:.1},\"dit_s\":{d:.3},\"dav_s\":{d:.3},\"total_s\":{d:.3},\"audio_s\":{d:.3},\"rtf\":{d:.3},\"latent_frames\":{d},\"steps\":{d},\"seed\":{d},\"load_s\":{d:.2},\"peak_memory_gb\":{d:.2}{s}}}", .{ @tagName(e.o.format), song.prompt_tokens, song.frames, song.stopped, song.prefill_ms / 1000, song.ar_ms / 1000, song.frames_per_s, e.frameBytes() / 1e9, gbs, song.dit_ms / 1000, song.dav_ms / 1000, song.total_ms / 1000, aus, song.total_ms / 1000 / @max(aus, 1e-9), song.latent_frames, s.steps, s.seed, e.load_ms / 1000, song.peak_device_gb, extra orelse "" });
}

fn idsOnly(gpa: std.mem.Allocator, io: std.Io, a: Args) !u8 {
    const path = try std.fs.path.join(gpa, &.{ a.model, m3.weights.text_encoder });
    defer gpa.free(path);
    var f = try core.safetensors.File.open(gpa, io, path);
    defer f.close(io);
    const tj = f.get("tokenizer_json") orelse return error.NoTokenizer;
    var tok = try core.tokenizer.parse(gpa, tj.bytes);
    defer tok.deinit();
    try m3.prompt.checkTokenizer(&tok);
    const ids = try m3.prompt.encode(gpa, &tok, a.caption orelse "", a.lyrics orelse "");
    defer gpa.free(ids);
    const js = try std.json.Stringify.valueAlloc(gpa, .{ .ids = ids }, .{});
    defer gpa.free(js);
    std.debug.print("{s}\n", .{js});
    return 0;
}

// ---------------------------------------------------------------------------------------------- --serve

extern "c" fn getline(line: *?[*]u8, n: *usize, f: *anyopaque) isize;
extern "c" fn fdopen(fd: c_int, mode: [*:0]const u8) ?*anyopaque;
extern "c" fn free(p: ?*anyopaque) void;

/// One protocol line on stderr (log lines never start with '{').
fn answer(text: []const u8) void {
    std.debug.print("{s}\n", .{text});
}

const ServeProgress = struct {
    io: std.Io,
    id: []const u8, // the request's id as JSON
    on: bool,
    last: [3]i128 = @splat(0),
    fn f(ctx: *anyopaque, stage: []const u8, done: usize, total: usize) void {
        const p: *ServeProgress = @ptrCast(@alignCast(ctx));
        if (!p.on) return;
        const si: usize = if (std.mem.eql(u8, stage, "ar")) 0 else if (std.mem.eql(u8, stage, "dit")) 1 else 2;
        const now = m3.ar.nowNs(p.io);
        if (done != total and now - p.last[si] < 1_000_000_000) return;
        p.last[si] = now;
        std.debug.print("{{\"id\":{s},\"progress\":{{\"stage\":\"{s}\",\"done\":{d},\"total\":{d}}}}}\n", .{ p.id, stage, done, total });
    }
};

fn serve(gpa: std.mem.Allocator, io: std.Io, a: Args, opts: m3.Options) !u8 {
    std.debug.print("{s}\n", .{built_with});
    const t0 = m3.ar.nowNs(io);
    const e = m3.Engine.create(gpa, io, a.model, opts) catch |er| {
        var buf: [256]u8 = undefined;
        answer(std.fmt.bufPrint(&buf, "{{\"ok\":false,\"error\":\"load failed: {s}\"}}", .{@errorName(er)}) catch "{\"ok\":false,\"error\":\"load failed\"}");
        return 1;
    };
    defer e.destroy();
    const load_s = m3.ar.ms(t0, m3.ar.nowNs(io)) / 1000;
    std.debug.print("[music3] serving {s}: loaded in {d:.1}s; weights on the device: {d:.1} GB ({s})\n", .{ a.model, load_s, @as(f64, @floatFromInt(e.store.bytes)) / 1e9, @tagName(opts.format) });
    {
        var buf: [256]u8 = undefined;
        answer(try std.fmt.bufPrint(&buf, "{{\"ready\":true,\"engine\":\"tensorfold-music3-cuda\",\"load_s\":{d:.2},\"sample_rate\":44100,\"channels\":2,\"license_display\":\"MiniMax-Music3\",\"weights\":\"{s}\"}}", .{ load_s, @tagName(opts.format) }));
    }
    const in = fdopen(0, "r") orelse return error.Stdin;
    var line: ?[*]u8 = null;
    var cap: usize = 0;
    defer free(line);
    while (true) {
        const n = getline(&line, &cap, in);
        if (n <= 0) break;
        const text = std.mem.trim(u8, line.?[0..@intCast(n)], " \t\r\n");
        if (text.len == 0) continue;
        var parsed = std.json.parseFromSlice(std.json.Value, gpa, text, .{}) catch {
            answer("{\"ok\":false,\"error\":\"bad json\"}");
            continue;
        };
        defer parsed.deinit();
        if (parsed.value != .object) {
            answer("{\"ok\":false,\"error\":\"bad json\"}");
            continue;
        }
        var ra = std.heap.ArenaAllocator.init(gpa);
        defer ra.deinit();
        const ar = ra.allocator();
        const o = parsed.value.object;
        const id_json = try std.json.Stringify.valueAlloc(ar, o.get("id") orelse std.json.Value.null, .{});
        const reply = serveOne(gpa, ar, io, e, a, o, id_json) catch |er| blk: {
            const msg = try std.json.Stringify.valueAlloc(ar, @errorName(er), .{});
            break :blk try std.fmt.allocPrint(ar, "{{\"ok\":false,\"error\":{s},\"id\":{s}}}", .{ msg, id_json });
        };
        answer(reply);
    }
    return 0;
}

fn str(o: std.json.ObjectMap, k: []const u8) ?[]const u8 {
    const v = o.get(k) orelse return null;
    return if (v == .string) v.string else null;
}

fn num(o: std.json.ObjectMap, k: []const u8) ?f64 {
    const v = o.get(k) orelse return null;
    return switch (v) {
        .float => |x| x,
        .integer => |x| @floatFromInt(x),
        .number_string => |x| std.fmt.parseFloat(f64, x) catch null,
        else => null,
    };
}

fn flag(o: std.json.ObjectMap, k: []const u8) bool {
    const v = o.get(k) orelse return false;
    return switch (v) {
        .bool => |b| b,
        .integer => |x| x != 0,
        else => false,
    };
}

fn serveOne(gpa: std.mem.Allocator, ar: std.mem.Allocator, io: std.Io, e: *m3.Engine, a: Args, o: std.json.ObjectMap, id_json: []const u8) ![]u8 {
    const caption = str(o, "caption") orelse return error.RequestNeedsCaption;
    if (std.mem.trim(u8, caption, " \t\r\n").len == 0) return error.RequestNeedsCaption;
    var lyrics: []const u8 = str(o, "lyrics") orelse "";
    if (o.get("lyrics") == null or o.get("lyrics").? == .null) {
        if (str(o, "lyrics_file")) |p| lyrics = try readText(ar, io, p);
    }
    const wav = str(o, "wav") orelse return error.RequestNeedsWav;
    var s = a.s;
    if (num(o, "seconds")) |x| s.seconds = x;
    if (num(o, "seed")) |x| s.seed = @intFromFloat(@max(0, x));
    if (num(o, "ar_cfg")) |x| s.ar_cfg = @floatCast(x);
    if (num(o, "top_k")) |x| s.top_k = @intFromFloat(x);
    if (num(o, "steps")) |x| s.steps = @intFromFloat(@max(1, x));
    if (num(o, "dit_cfg")) |x| s.dit_cfg = @floatCast(x);
    std.debug.print("[music3] request {s}: {d:.0}s cap, seed {d}\n", .{ id_json, s.seconds, s.seed });
    var pc: ServeProgress = .{ .io = io, .id = id_json, .on = flag(o, "progress") };
    const song = try e.generate(io, caption, lyrics, s, .{ .ctx = &pc, .f = ServeProgress.f });
    defer gpa.free(song.audio);
    const path = try expand(ar, wav);
    if (std.fs.path.dirname(path)) |dir| std.Io.Dir.cwd().createDirPath(io, dir) catch {};
    try writeWav(gpa, io, path, song.audio);
    const aus = audioS(song);
    const wav_json = try std.json.Stringify.valueAlloc(ar, wav, .{});
    const reply = try std.fmt.allocPrint(ar, "{{\"ok\":true,\"id\":{s},\"wav\":{s},\"audio_s\":{d:.3},\"frames\":{d},\"eos\":{},\"ar_ms\":{d:.1},\"dit_ms\":{d:.1},\"dav_ms\":{d:.1},\"total_ms\":{d:.1},\"rtf\":{d:.3},\"first_audio_ms\":{d:.1},\"stream\":false,\"seed\":{d},\"peak_memory_gb\":{d:.2}}}", .{ id_json, wav_json, aus, song.frames, song.stopped, song.ar_ms, song.dit_ms, song.dav_ms, song.total_ms, song.total_ms / 1000 / @max(aus, 1e-9), song.total_ms, s.seed, song.peak_device_gb });
    if (str(o, "stats")) |sp| {
        const st = try statsJson(gpa, e, song, s, try std.fmt.allocPrint(ar, ",\"reply\":{s}", .{reply}));
        defer gpa.free(st);
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try expand(ar, sp), .data = st });
    }
    return reply;
}
