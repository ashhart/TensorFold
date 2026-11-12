//! tf-qwen3-asr-forced: a clip on CUDA teacher-forced on a case's greedy ids, every step judged by the tie rule.
const std = @import("std");
const cuda = @import("cuda");
const qa = @import("qwen3_asr");
const host = qa.host;
const forced = host.forced;

const usage = "usage: tf-qwen3-asr-forced MODEL_DIR CLIP.wav --case DIR [--plant splice|norms] [--rows K]\n" ++
    "  every step on the case's greedy history: a differing top token passes only as a swap with both margins <= 0.25\n" ++
    "  (each listed); anything else fails, exit 1. With logits_fp32 and logits_bf16 in the case, the distances to fp32.\n" ++
    "  --rows K: the prompt pass over the first K reply ids against the decode steps, and the prompt in two passes\n" ++
    "  against one (caches and logits): exit 1 unless bit-equal.\n" ++
    "  --plant: the audio rows one position late (splice) or q_norm and k_norm swapped in every layer (norms).\n";

/// A planted fault the tool must catch: the audio rows one position late, or q_norm and k_norm swapped.
const Plant = enum { splice, norms };

fn widen(bits: u16) f64 {
    return @as(f32, @bitCast(@as(u32, bits) << 16));
}

/// The largest |a - b| over every column of every step, where it is, and the mean |a - b|.
const Dist = struct {
    worst: f64 = 0,
    step: usize = 0,
    col: usize = 0,
    sum: f64 = 0,
    n: usize = 0,

    fn add(d: *Dist, step: usize, a: []const f64, b: []const f32) void {
        for (a, b, 0..) |x, y, c| {
            const e = @abs(x - y);
            d.sum += e;
            d.n += 1;
            if (!(e <= d.worst)) {
                d.worst = e;
                d.step = step;
                d.col = c;
            }
        }
    }

    fn print(d: Dist, label: []const u8) void {
        std.debug.print("DIST {s}: worst column {d:.6} (step {d}, id {d}), mean {e:.4} over {d} logits\n", .{ label, d.worst, d.step, d.col, d.sum / @as(f64, @floatFromInt(@max(d.n, 1))), d.n });
    }
};

const Rows = struct { equal: usize, rows: usize, worst: f64 };

/// Rows of logits bit-equal between two runs, and the largest |difference|.
fn compareRows(a: []const u16, b: []const u16, vocab: usize) Rows {
    var r: Rows = .{ .equal = 0, .rows = a.len / vocab, .worst = 0 };
    for (0..r.rows) |i| {
        const x = a[i * vocab ..][0..vocab];
        const y = b[i * vocab ..][0..vocab];
        if (std.mem.eql(u16, x, y)) r.equal += 1;
        for (x, y) |p, q| r.worst = @max(r.worst, @abs(widen(p) - widen(q)));
    }
    return r;
}

pub fn main(init: std.process.Init) !u8 {
    const gpa = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 5) {
        std.debug.print(usage, .{});
        return 2;
    }
    var case_dir: ?[]const u8 = null;
    var plant: ?Plant = null;
    var check_rows: usize = 0;
    var i: usize = 3;
    while (i < args.len) : (i += 2) {
        if (i + 1 >= args.len) break;
        if (std.mem.eql(u8, args[i], "--case")) case_dir = args[i + 1] else if (std.mem.eql(u8, args[i], "--rows")) {
            check_rows = try std.fmt.parseInt(usize, args[i + 1], 10);
        } else if (std.mem.eql(u8, args[i], "--plant")) {
            plant = std.meta.stringToEnum(Plant, args[i + 1]) orelse {
                std.debug.print(usage, .{});
                return 2;
            };
        } else {
            std.debug.print(usage, .{});
            return 2;
        }
    }
    if (i < args.len or case_dir == null) {
        std.debug.print(usage, .{});
        return 2;
    }
    const cfg = try host.config.read(gpa, io, args[1]);
    var case = try host.case.Case.open(gpa, io, case_dir.?);
    defer case.deinit();
    if (case.attention != .windows) return error.GlobalCase;
    const ids = try case.ids(gpa, "prompt_ids");
    defer gpa.free(ids);
    const want = try case.ids(gpa, "greedy_ids");
    defer gpa.free(want);
    const top2 = try case.ids(gpa, "greedy_top2");
    defer gpa.free(top2);
    const ref_margin = try case.floats(gpa, "greedy_margin");
    defer gpa.free(ref_margin);
    if (want.len == 0 or top2.len != 2 * want.len or ref_margin.len != want.len) return error.BadCase;
    check_rows = @min(check_rows, want.len - 1);

    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, args[2], gpa, .limited(1 << 31));
    defer gpa.free(bytes);
    const samples = try host.wav.decode(gpa, bytes);
    defer gpa.free(samples);
    var feats = try host.mel.features(gpa, samples, cfg.audio.mels);
    defer feats.deinit(gpa);
    const plan = try host.chunks.Plan.init(cfg.audio, feats.frames);
    const at = std.mem.indexOfScalar(u32, ids, cfg.audio_token) orelse return error.NoAudioRows;
    if (std.mem.count(u32, ids, &.{cfg.audio_token}) != plan.rows) return error.AudioRowsDiffer;

    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var s = try cuda.Stream.init(&driver, false);
    defer s.deinit();
    const opts: qa.Options = .{ .audio_rows = plan.rows, .pass_rows = ids.len + check_rows, .positions = ids.len + want.len + 1, .head_rows = check_rows + 1 };
    const m = try qa.Model.load(gpa, io, &driver, s, args[1], cfg, opts);
    defer m.deinit();
    var rows_buf = try cuda.DeviceBuffer.alloc(&driver, plan.rows * cfg.audio.out * 2);
    defer rows_buf.free();
    const n_rows = try m.encode(feats, rows_buf.ptr, null);
    const splice: qa.Splice = .{ .ptr = rows_buf.ptr, .at = at + @intFromBool(plant == .splice), .n = n_rows };
    if (plant == .norms) m.swapQkNorms();
    const vocab = cfg.text.vocab;

    // the prompt pass, then one decode step a reference id: every step's logits row kept
    var steps: std.ArrayList(u16) = .empty;
    defer steps.deinit(gpa);
    m.keep = .{ .gpa = gpa, .rows = &steps };
    _ = try m.forward(ids, 0, splice);
    for (want[0 .. want.len - 1], 0..) |id, k| _ = try m.forward(&.{id}, ids.len + k, null);
    m.keep = null;

    const clip = std.fs.path.basename(case_dir.?);
    var same: usize = 0;
    var swaps: usize = 0;
    var fails: usize = 0;
    for (0..want.len) |k| {
        const ours = forced.top(steps.items[k * vocab ..][0..vocab]);
        const ref = try forced.reference(top2[2 * k], top2[2 * k + 1], ref_margin[k], want[k]);
        switch (forced.judge(ours, ref, forced.bar)) {
            .same => same += 1,
            inline else => |v| {
                if (v == .swap) swaps += 1 else fails += 1;
                std.debug.print("{s} {s} step {d}: ours {d} (margin {d:.4}, runner-up {d}), reference {d} (margin {d:.4}, runner-up {d})\n", .{ if (v == .swap) "SWAP" else "FAIL", clip, k, ours.first, ours.margin, ours.second, ref.first, ref.margin, ref.second });
            },
        }
    }
    var ok = fails == 0;
    std.debug.print("FORCED {s}: {d} steps, {d} same top token, {d} swaps within {d}, {d} fail{s}\n", .{ clip, want.len, same, swaps, forced.bar, fails, if (plant) |p| switch (p) {
        .splice => " (planted: the audio rows one position late)",
        .norms => " (planted: q_norm and k_norm swapped in every layer)",
    } else "" });

    if (case.floats(gpa, "logits_fp32")) |fp32| {
        defer gpa.free(fp32);
        const bf = try case.floats(gpa, "logits_bf16");
        defer gpa.free(bf);
        if (fp32.len != want.len * vocab or bf.len != fp32.len) return error.StageSize;
        var ours: Dist = .{};
        var pkg: Dist = .{};
        const row = try gpa.alloc(f64, vocab);
        defer gpa.free(row);
        for (0..want.len) |k| {
            for (row, steps.items[k * vocab ..][0..vocab]) |*r, b| r.* = widen(b);
            ours.add(k, row, fp32[k * vocab ..][0..vocab]);
            for (row, bf[k * vocab ..][0..vocab]) |*r, b| r.* = b;
            pkg.add(k, row, fp32[k * vocab ..][0..vocab]);
        }
        ours.print("ours to the package's fp32");
        pkg.print("the package's bf16 to its fp32");
    } else |e| if (e != error.NoSuchStage) return e;

    if (check_rows > 0) {
        // one prompt pass over the prompt and the first K reply ids, the head over its last K + 1 rows
        const long = try std.mem.concat(gpa, u32, &.{ ids, want[0..check_rows] });
        defer gpa.free(long);
        var pass: std.ArrayList(u16) = .empty;
        defer pass.deinit(gpa);
        m.keep = .{ .gpa = gpa, .rows = &pass };
        _ = try m.forwardHeads(long, 0, splice, check_rows + 1);
        m.keep = null;
        const dec = compareRows(pass.items, steps.items[0 .. (check_rows + 1) * vocab], vocab);
        std.debug.print("ROWS {s}: the prompt pass's last {d} rows against the decode steps: {d} bit-equal, worst |diff| {d:.6}\n", .{ clip, dec.rows, dec.equal, dec.worst });
        ok = ok and dec.equal == dec.rows;
        // the prompt in one pass, then in two cut inside the audio rows: caches and the last row's logits
        var one: std.ArrayList(u16) = .empty;
        defer one.deinit(gpa);
        m.keep = .{ .gpa = gpa, .rows = &one };
        _ = try m.forward(ids, 0, splice);
        const c1 = try m.cacheRows(gpa, ids.len);
        defer gpa.free(c1);
        const cut = at + n_rows / 2;
        _ = try m.forward(ids[0..cut], 0, splice);
        _ = try m.forward(ids[cut..], cut, splice);
        m.keep = null;
        const c2 = try m.cacheRows(gpa, ids.len);
        defer gpa.free(c2);
        var differ: usize = 0;
        for (c1, c2) |x, y| differ += @intFromBool(x != y);
        const lg = compareRows(one.items[0..vocab], one.items[2 * vocab ..][0..vocab], vocab);
        std.debug.print("SPLIT {s}: cut at {d} of {d} (inside the audio rows {d}..{d}): {d} of {d} cache values differ, last logits {s} (worst |diff| {d:.6})\n", .{ clip, cut, ids.len, at, at + n_rows, differ, c1.len, if (lg.equal == 1) "bit-equal" else "differ", lg.worst });
        ok = ok and differ == 0 and lg.equal == 1;
    }
    std.debug.print("SUMMARY qwen3-asr forced {s}: {s}\n", .{ clip, if (ok) "PASS" else "FAIL" });
    return if (ok) 0 else 1;
}
