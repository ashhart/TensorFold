//! tf-qwen3-asr-check: a Qwen3-ASR checkpoint's weights, tokenizer, mel rows and float64 tower against a case.
const std = @import("std");
const qa = @import("qwen3_asr");
const st = @import("safetensors");
const tok = @import("tokenizer");

const usage = "usage: tf-qwen3-asr-check MODEL_DIR CLIP.wav CASE_DIR [--points exact|torch] [--no-tower]\n";

const Mapped = struct {
    file: st.File,

    fn get(ctx: *anyopaque, name: []const u8) ?[]const u8 {
        const m: *Mapped = @ptrCast(@alignCast(ctx));
        const t = m.file.get(name) orelse return null;
        return t.bytes;
    }
};

/// Each stage gathered (the convs arrive a chunk at a time, interleaved), then compared with the case's.
const Compare = struct {
    gpa: std.mem.Allocator,
    case: *qa.case.Case,
    worst_cos: f64 = 1,
    stages: std.StringArrayHashMapUnmanaged(std.ArrayList(f64)) = .empty,

    fn put(ctx: *anyopaque, stage: []const u8, offset: usize, values: []const f64) anyerror!void {
        const c: *Compare = @ptrCast(@alignCast(ctx));
        const got = try c.stages.getOrPut(c.gpa, stage);
        if (!got.found_existing) {
            got.key_ptr.* = try c.gpa.dupe(u8, stage);
            got.value_ptr.* = .empty;
        }
        if (offset != got.value_ptr.items.len) return error.StageOrder;
        try got.value_ptr.appendSlice(c.gpa, values);
        // a row stage is whole when it arrives: compare it now and free it
        if (!std.mem.startsWith(u8, stage, "conv")) try c.report(got.key_ptr.*);
    }

    fn report(c: *Compare, name: []const u8) !void {
        const l = c.stages.getPtr(name).?;
        const want = try c.case.floats(c.gpa, name);
        defer c.gpa.free(want);
        if (l.items.len != want.len) return error.StageSize;
        const m = qa.case.Match.of(l.items, want);
        c.worst_cos = @min(c.worst_cos, m.cosine);
        std.debug.print("STAGE {s:<12} cos {d:.9} worst {e:.3} at {d} rms(diff)/rms(ref) {e:.3}\n", .{ name, m.cosine, m.worst, m.at, m.rms_diff / @max(m.rms_want, 1e-300) });
        l.clearAndFree(c.gpa);
    }

    fn finish(c: *Compare) !void {
        for (c.stages.keys()) |k| if (std.mem.startsWith(u8, k, "conv")) try c.report(k);
        for (c.stages.keys(), c.stages.values()) |k, *v| {
            v.deinit(c.gpa);
            c.gpa.free(k);
        }
        c.stages.deinit(c.gpa);
    }
};

fn checkWeights(gpa: std.mem.Allocator, cfg: qa.config.Config, file: *const st.File) !void {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var have: std.ArrayList(qa.names.Have) = .empty;
    for (file.names.keys(), file.names.values()) |name, *e| try have.append(a, .{ .name = name, .bf16 = e.dtype == .bf16, .shape = e.shape[0..e.rank] });
    var problem: []const u8 = "";
    qa.names.validate(a, cfg, have.items, &problem) catch |err| {
        std.debug.print("WEIGHTS refused: {s} ({t})\n", .{ problem, err });
        return err;
    };
    const head = file.get(qa.names.head).?.bytes;
    const embed = file.get(qa.names.lm ++ "embed_tokens.weight").?.bytes;
    if (!std.mem.eql(u8, head, embed)) return error.UntiedHead;
    std.debug.print("WEIGHTS ok: {d} tensors, all bf16 at the config's shapes; lm_head bit-equal to the embedding\n", .{have.items.len});
}

fn checkTokens(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, case: *qa.case.Case, rows: usize) !void {
    var t = try tok.loadTokenizer(io, gpa, dir); // a Qwen2 slow export when there is no tokenizer.json (#2225)
    defer t.deinit();
    const text = try qa.prompt.text(gpa, rows, "", null);
    defer gpa.free(text);
    const ids = try t.encode(gpa, text);
    defer gpa.free(ids);
    const want = try case.ids(gpa, "prompt_ids");
    defer gpa.free(want);
    const same_prompt = std.mem.eql(u32, ids, want);
    const greedy = try case.ids(gpa, "greedy_ids");
    defer gpa.free(greedy);
    const reply = try t.decode(gpa, greedy, true);
    defer gpa.free(reply);
    const same_text = std.mem.eql(u8, reply, case.text);
    std.debug.print("TOKENS prompt {d} ids {s}; reference reply decodes {s}: {s}\n", .{ ids.len, if (same_prompt) "equal" else "DIFFER", if (same_text) "equal" else "DIFFERENT", reply });
    if (!same_prompt or !same_text) return error.TokenizerMismatch;
}

pub fn main(init: std.process.Init) !u8 {
    const gpa = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 4) {
        std.debug.print(usage, .{});
        return 2;
    }
    var points: qa.ref_ops.Points = .torch;
    var tower = true;
    var i: usize = 4;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--no-tower")) tower = false else if (std.mem.eql(u8, args[i], "--points") and i + 1 < args.len) {
            i += 1;
            points = std.meta.stringToEnum(qa.ref_ops.Points, args[i]) orelse return 2;
        } else {
            std.debug.print(usage, .{});
            return 2;
        }
    }
    const cfg = try qa.config.read(gpa, io, args[1]);
    const path = try std.fs.path.join(gpa, &.{ args[1], "model.safetensors" });
    defer gpa.free(path);
    var mapped: Mapped = .{ .file = try st.File.open(gpa, io, path) };
    defer mapped.file.close(io);
    try checkWeights(gpa, cfg, &mapped.file);
    var case = try qa.case.Case.open(gpa, io, args[3]);
    defer case.deinit();
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, args[2], gpa, .limited(1 << 31));
    defer gpa.free(bytes);
    const samples = try qa.wav.decode(gpa, bytes);
    defer gpa.free(samples);
    const plan = try qa.chunks.Plan.init(cfg.audio, qa.chunks.frames(samples.len));
    std.debug.print("CLIP {d} samples, {d} frames ({d} in the case), {d} chunks, {d} rows, {d} windows of {d}\n", .{ samples.len, plan.frames, case.frames, plan.chunks, plan.rows, plan.windows(), plan.window_rows });
    if (plan.frames != case.frames) return error.FrameCount;
    try checkTokens(gpa, io, args[1], &case, plan.rows);
    var feats = try qa.mel.features(gpa, samples, cfg.audio.mels);
    defer feats.deinit(gpa);
    var cmp: Compare = .{ .gpa = gpa, .case = &case };
    const wide = try gpa.alloc(f64, feats.data.len);
    defer gpa.free(wide);
    for (wide, feats.data) |*w, f| w.* = f;
    try Compare.put(&cmp, "mel", 0, wide);
    if (tower) {
        const src: qa.tower_ref.Source = .{ .ctx = &mapped, .get = Mapped.get };
        const out = try qa.tower_ref.run(gpa, cfg.audio, points, src, feats, .{ .ctx = &cmp, .put = Compare.put });
        gpa.free(out);
    }
    try cmp.finish();
    std.debug.print("SUMMARY qwen3-asr host ({t} points): worst stage cosine {d:.9}\n", .{ points, cmp.worst_cos });
    return 0;
}
