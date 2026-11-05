//! tf-qwen3-asr: a clip through Qwen3-ASR on the first GPU, greedy, one stream; each stage against a reference case.
const std = @import("std");
const cuda = @import("cuda");
const qa = @import("qwen3_asr");
const tok = @import("tokenizer");
const host = qa.host;

const usage = "usage: tf-qwen3-asr MODEL_DIR CLIP.wav [--case DIR [--ids-only]] [--tokens N]\n";
const eos = [_]u32{ 151645, 151643 };

fn seconds() f64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(std.c.CLOCK.MONOTONIC, &ts);
    return @as(f64, @floatFromInt(ts.sec)) + @as(f64, @floatFromInt(ts.nsec)) * 1e-9;
}

fn widen(bits: u16) f64 {
    return @as(f32, @bitCast(@as(u32, bits) << 16));
}

/// Each dumped stage downloaded and compared with the case's; the worst cosine kept.
const Compare = struct {
    gpa: std.mem.Allocator,
    case: *host.case.Case,
    s: cuda.Stream,
    d: *const cuda.Driver,
    worst: f64 = 1,
    conv: std.StringHashMapUnmanaged(std.ArrayList(f64)) = .empty,

    fn put(ctx: *anyopaque, stage: []const u8, ptr: u64, rows: usize, cols: usize, conv: ?[3]usize) anyerror!void {
        const c: *Compare = @ptrCast(@alignCast(ctx));
        const raw = try c.gpa.alloc(u16, rows * cols);
        defer c.gpa.free(raw);
        try c.s.synchronize();
        try c.d.check(c.d.api.cuMemcpyDtoH_v2(raw.ptr, ptr, raw.len * 2), "cuMemcpyDtoH");
        const vals = try c.gpa.alloc(f64, raw.len);
        defer c.gpa.free(vals);
        if (conv) |sh| { // channels last [n][f][t][c] to the export's [n][c][f][t]
            const per = sh[1] * sh[2] * cols;
            for (0..sh[0]) |n| for (0..sh[1]) |f| for (0..sh[2]) |t| for (0..cols) |k| {
                vals[n * per + (k * sh[1] + f) * sh[2] + t] = widen(raw[((n * sh[1] + f) * sh[2] + t) * cols + k]);
            };
            const got = try c.conv.getOrPut(c.gpa, stage);
            if (!got.found_existing) got.value_ptr.* = .empty;
            try got.value_ptr.appendSlice(c.gpa, vals);
            return;
        }
        for (vals, raw) |*v, r| v.* = widen(r);
        try c.report(stage, vals);
    }

    fn report(c: *Compare, stage: []const u8, vals: []const f64) !void {
        const want = c.case.floats(c.gpa, stage) catch |e| return if (e == error.NoSuchStage) {} else e; // not every layer
        defer c.gpa.free(want);
        if (want.len != vals.len) return error.StageSize;
        const m = host.case.Match.of(vals, want);
        c.worst = @min(c.worst, m.cosine);
        std.debug.print("STAGE {s:<12} cos {d:.9} worst {e:.3} at {d} rms(diff)/rms(ref) {e:.3}\n", .{ stage, m.cosine, m.worst, m.at, m.rms_diff / @max(m.rms_want, 1e-300) });
    }

    /// The conv stages, gathered over every chunk group, compared once whole.
    fn finishConvs(c: *Compare) !void {
        for ([_][]const u8{ "conv1", "conv2", "conv3" }) |n| if (c.conv.getPtr(n)) |l| {
            try c.report(n, l.items);
            l.deinit(c.gpa);
        };
        c.conv.deinit(c.gpa);
    }
};

pub fn main(init: std.process.Init) !u8 {
    const gpa = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 3) {
        std.debug.print(usage, .{});
        return 2;
    }
    var case_dir: ?[]const u8 = null;
    var max_new: usize = 448;
    var ids_only = false; // a case of ids and margins only: no stage compared
    var i: usize = 3;
    while (i < args.len) : (i += 2) {
        if (std.mem.eql(u8, args[i], "--ids-only")) {
            ids_only = true;
            i -= 1;
        } else if (i + 1 >= args.len) {
            std.debug.print(usage, .{});
            return 2;
        } else if (std.mem.eql(u8, args[i], "--case")) case_dir = args[i + 1] else if (std.mem.eql(u8, args[i], "--tokens")) {
            max_new = try std.fmt.parseInt(usize, args[i + 1], 10);
        } else {
            std.debug.print(usage, .{});
            return 2;
        }
    }
    const cfg = try host.config.read(gpa, io, args[1]);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, args[2], gpa, .limited(1 << 31));
    defer gpa.free(bytes);
    const samples = try host.wav.decode(gpa, bytes);
    defer gpa.free(samples);
    const t_mel = seconds();
    var feats = try host.mel.features(gpa, samples, cfg.audio.mels);
    defer feats.deinit(gpa);
    const mel_s = seconds() - t_mel;
    const plan = try host.chunks.Plan.init(cfg.audio, feats.frames);
    var tk = try tok.loadTokenizer(io, gpa, args[1]);
    defer tk.deinit();
    const text = try host.prompt.text(gpa, plan.rows, "", null);
    defer gpa.free(text);
    const ids = try tk.encode(gpa, text);
    defer gpa.free(ids);
    const at = std.mem.indexOfScalar(u32, ids, cfg.audio_token) orelse return error.NoAudioRows;

    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var name_buf: [256]u8 = undefined;
    var s = try cuda.Stream.init(&driver, false);
    defer s.deinit();
    const t_load = seconds();
    const opts: qa.Options = .{ .audio_rows = plan.rows, .pass_rows = ids.len, .positions = ids.len + max_new + 1 };
    const m = try qa.Model.load(gpa, io, &driver, s, args[1], cfg, opts);
    defer m.deinit();
    var rows_buf = try cuda.DeviceBuffer.alloc(&driver, plan.rows * cfg.audio.out * 2);
    defer rows_buf.free();
    const gib = struct {
        fn f(x: u64) f64 {
            return @as(f64, @floatFromInt(x)) / (1 << 30);
        }
    }.f;
    std.debug.print("GPU {s}: weights {d:.2} GiB, scratch {d:.2} GiB, load {d:.2} s\n", .{ ctx.name(&name_buf) catch "?", gib(m.weight_bytes), gib(m.st.bytes - m.weight_bytes), seconds() - t_load });
    std.debug.print("CLIP {d:.2} s: {d} frames, {d} chunks, {d} rows, {d} windows; prompt {d} ids\n", .{ @as(f64, @floatFromInt(samples.len)) / 16000.0, feats.frames, plan.chunks, plan.rows, plan.windows(), ids.len });

    var case: ?host.case.Case = if (case_dir) |dir| try host.case.Case.open(gpa, io, dir) else null;
    defer if (case) |*c| c.deinit();
    var cmp: ?Compare = if (case) |*c| .{ .gpa = gpa, .case = c, .s = s, .d = &driver } else null;
    if (cmp) |*c| {
        const want = try c.case.ids(gpa, "prompt_ids");
        defer gpa.free(want);
        std.debug.print("PROMPT ids {s} the case's\n", .{if (std.mem.eql(u32, want, ids)) "equal to" else "DIFFERENT from"});
    }
    // a warm pass first: the timed pass after it sees no cuBLASLt plan creation or module load
    _ = try m.encode(feats, rows_buf.ptr, null);
    try s.synchronize();
    const t_tower = seconds();
    const rows = try m.encode(feats, rows_buf.ptr, null);
    try s.synchronize();
    const tower_s = seconds() - t_tower;
    if (cmp) |*c| if (!ids_only) {
        _ = try m.encode(feats, rows_buf.ptr, .{ .ctx = c, .put = Compare.put });
        try c.finishConvs();
    };
    const t_prompt = seconds();
    const splice: qa.Splice = .{ .ptr = rows_buf.ptr, .at = at, .n = rows };
    var next = try m.forward(ids, 0, splice);
    const prompt_s = seconds() - t_prompt;
    var out: std.ArrayList(u32) = .empty;
    defer out.deinit(gpa);
    var margins: std.ArrayList(f32) = .empty;
    defer margins.deinit(gpa);
    try margins.append(gpa, m.lr.last_margin);
    if (cmp) |*c| if (!ids_only) { // the prompt pass's stages, then the pass again: its caches written as before
        _ = try m.forwardDump(ids, 0, splice, .{ .ctx = c, .put = Compare.put });
    };
    const t_dec = seconds();
    while (out.items.len < max_new) {
        try out.append(gpa, next);
        if (std.mem.indexOfScalar(u32, &eos, next) != null) break;
        next = try m.forward(&.{next}, ids.len + out.items.len - 1, null);
        try margins.append(gpa, m.lr.last_margin);
    }
    const dec_s = seconds() - t_dec;
    const body = if (out.items.len > 0 and std.mem.indexOfScalar(u32, &eos, out.items[out.items.len - 1]) != null) out.items[0 .. out.items.len - 1] else out.items;
    const reply = try tk.decode(gpa, body, true);
    defer gpa.free(reply);
    std.debug.print("TIMES mel {d:.1} ms, tower {d:.1} ms ({d} rows), prompt pass {d:.1} ms ({d} rows), decode {d} tokens {d:.1} tok/s\n", .{ mel_s * 1e3, tower_s * 1e3, rows, prompt_s * 1e3, ids.len, out.items.len, @as(f64, @floatFromInt(out.items.len)) / @max(dec_s, 1e-9) });
    std.debug.print("TEXT {s}\n", .{reply});
    if (cmp) |*c| {
        const want = try c.case.ids(gpa, "greedy_ids");
        defer gpa.free(want);
        const n = @min(want.len, out.items.len);
        const first = std.mem.indexOfDiff(u32, want[0..n], out.items[0..n]) orelse n;
        const same = first == n and want.len == out.items.len;
        std.debug.print("GREEDY {d} tokens vs the case's {d}: {s}", .{ out.items.len, want.len, if (same) "byte-identical\n" else "differ" });
        if (!same) {
            const ref = c.case.floats(gpa, "greedy_margin") catch try gpa.alloc(f32, 0); // older cases have none
            defer gpa.free(ref);
            std.debug.print(" from token {d}: ours {d} (margin {d:.4}), the case's {d} (margin {d:.4})\n", .{
                first,                                                         if (first < out.items.len) out.items[first] else 0,
                if (first < margins.items.len) margins.items[first] else 0,    if (first < want.len) want[first] else 0,
                if (first < ref.len) ref[first] else 0,
            });
        }
        std.debug.print("SUMMARY qwen3-asr cuda: worst stage cosine {d:.9}, transcript {s}\n", .{ c.worst, if (same) "equal" else "different" });
    }
    return 0;
}
