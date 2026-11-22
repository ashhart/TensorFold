//! tf-qwen3-asr-xpu-forced: a clip on the Intel GPU teacher-forced on a case's greedy ids, each step by the tie rule.
const std = @import("std");
const xpu = @import("xpu");
const qa = @import("qwen3_asr");
const host = qa.host;
const forced = host.forced;

const usage = "usage: tf-qwen3-asr-xpu-forced MODEL_DIR CLIP.wav --case DIR [--plant splice|norms]\n" ++
    "  every step on the case's greedy history: a differing top token passes only as a swap with both margins <= 0.25\n" ++
    "  (each listed); anything else fails, exit 1. With logits_fp32 and logits_bf16 in the case, the distances to fp32.\n" ++
    "  --plant: the audio rows one position late (splice) or q_norm and k_norm swapped in every layer (norms).\n";

/// A planted fault the tool must catch: the audio rows one position late, or q_norm and k_norm swapped.
const Plant = enum { splice, norms };

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
    var i: usize = 3;
    while (i + 1 < args.len) : (i += 2) {
        if (std.mem.eql(u8, args[i], "--case")) case_dir = args[i + 1] else if (std.mem.eql(u8, args[i], "--plant")) {
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

    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, args[2], gpa, .limited(1 << 31));
    defer gpa.free(bytes);
    const samples = try host.wav.decode(gpa, bytes);
    defer gpa.free(samples);
    var feats = try host.mel.features(gpa, samples, cfg.audio.mels);
    defer feats.deinit(gpa);
    const plan = try host.chunks.Plan.init(cfg.audio, feats.frames);
    const at = std.mem.indexOfScalar(u32, ids, cfg.audio_token) orelse return error.NoAudioRows;
    if (std.mem.count(u32, ids, &.{cfg.audio_token}) != plan.rows) return error.AudioRowsDiffer;

    var driver = try xpu.Driver.open();
    var ctx = try xpu.Context.init(&driver, null);
    const stream = try xpu.Stream.init(&ctx);
    var r = try xpu.rt.Runtime.init(&ctx, stream);
    defer r.deinit();
    const opts: qa.Options = .{ .audio_rows = plan.rows, .pass_rows = ids.len, .positions = ids.len + want.len + 1 };
    const m = try qa.Model.load(gpa, io, &r, args[1], cfg, opts);
    defer m.deinit();
    var rows_buf = try r.alloc(plan.rows * cfg.audio.out * 2);
    defer rows_buf.free();
    const rows_ptr: u64 = @intFromPtr(rows_buf.ptr.?);
    const n_rows = try m.encode(feats, rows_ptr, null);
    const splice: qa.Splice = .{ .ptr = rows_ptr, .at = at + @intFromBool(plant == .splice), .n = n_rows };
    if (plant == .norms) m.swapQkNorms();
    const vocab = cfg.text.vocab;

    // the prompt pass, then one decode step a reference id: each pass's head row (the greedy pick's logits) kept
    var steps: std.ArrayList(u16) = .empty;
    defer steps.deinit(gpa);
    _ = try m.forward(ids, 0, splice);
    try steps.appendSlice(gpa, m.lr.logits);
    for (want[0 .. want.len - 1], 0..) |id, k| {
        _ = try m.forward(&.{id}, ids.len + k, null);
        try steps.appendSlice(gpa, m.lr.logits);
    }
    if (steps.items.len != want.len * vocab) return error.StepRows;

    const clip = std.fs.path.basename(case_dir.?);
    var tally: forced.Tally = .{};
    for (0..want.len) |k| try tally.step(clip, k, steps.items[k * vocab ..][0..vocab], top2[2 * k], top2[2 * k + 1], ref_margin[k], want[k]);
    tally.print(clip, want.len, if (plant) |p| switch (p) {
        .splice => " (planted: the audio rows one position late)",
        .norms => " (planted: q_norm and k_norm swapped in every layer)",
    } else "");
    try forced.distances(gpa, &case, steps.items, want.len, vocab);
    const ok = tally.fails == 0;
    std.debug.print("SUMMARY qwen3-asr xpu forced {s}: {s}\n", .{ clip, if (ok) "PASS" else "FAIL" });
    return if (ok) 0 else 1;
}
