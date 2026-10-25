//! Gemma 4 + DFlash: states kept by stepped passes carry the drafter context, resumed == fresh.
const std = @import("std");
const core = @import("core");
const lanes = @import("lanes");
const g4 = @import("gemma4_cuda");
const check = @import("check.zig");
const Gpu = check.Gpu;

const native = g4.native31;
const snap = g4.snap;

/// Marks kept: off the 512-row chunk grid, and on it; the prompt runs past both.
const marks = [_]u32{ 1000, 1536 };
const prompt_len = 2600;

const Keep = struct {
    gb: *g4.backend.Gemma,
    saved: [marks.len]?*snap.Saved = @splat(null),
    fn hook(ptr: *anyopaque, s: *lanes.Stream, at: u32) void {
        const k: *@This() = @ptrCast(@alignCast(ptr));
        const i = std.mem.indexOfScalar(u32, &marks, at) orelse return;
        k.saved[i] = k.gb.saveState(s, at) catch null;
    }
};

const Case = struct { prompt: []const u32, drafts: bool = true, sampling: ?lanes.Sampling = null, reuse: lanes.stream.Reuse = .{} };
const Out = struct { ids: []u32, logits: []u8, cached: u64 };

/// One stream, its prompt stepped as the lane host runs it: reply, last-row logits, rows resumed.
fn run(gpu: Gpu, engine: *lanes.Engine, gb: *g4.backend.Gemma, c: Case, count: u32) !Out {
    const gpa = gpu.gpa;
    var s = try lanes.Stream.init(gpa, .{ .id = "g", .prompt = c.prompt, .max_new = count, .sampling = c.sampling, .drafts = c.drafts, .reuse = c.reuse });
    defer s.deinit(gpa);
    var first = true;
    while (!try engine.fillStream(&s, first)) first = false;
    const logits = try gpa.alloc(u8, g4.config.vocab * 2);
    errdefer gpa.free(logits);
    try gb.s.synchronize();
    const l = gb.lanes.get(&s) orelse return error.UnknownStream;
    try gb.d.check(gb.d.api.cuMemcpyDtoH_v2(logits.ptr, l.last, logits.len), "cuMemcpyDtoH");
    while (engine.activeCount() > 0) try engine.step();
    return .{ .ids = try gpa.dupe(u32, s.emitted()), .logits = logits, .cached = if (s.reuse_failed) 0 else s.cached };
}

fn free(gpa: std.mem.Allocator, o: Out) void {
    gpa.free(o.ids);
    gpa.free(o.logits);
}

/// tf-cuda-test gemma4-states MODEL DRAFTER TEXT N: states kept at 1000 and 1536 resume a drafted pass == a fresh one.
pub fn states(gpu: Gpu, rest: []const []const u8) !void {
    if (rest.len < 4) return error.MissingArguments;
    const gpa = gpu.gpa;
    const count = try std.fmt.parseInt(u32, rest[3], 10);
    var loaded = try native.open(gpa, gpu.io, gpu.ctx, rest[0], null, .{ .context = 8192, .drafts = true, .lanes = 2, .drafter = rest[1] });
    defer loaded.deinit(loaded.ctx);
    const opened = try native.openDrafter(gpa, gpu.io, loaded.ctx, rest[1], 4);
    defer opened.close(opened.ctx);
    var drafted = try lanes.drafted.Drafted.init(gpa, loaded.backend, opened.drafter);
    defer drafted.deinit();
    const facts = drafted.facts(loaded.facts);
    var cfg = try lanes.Config.init(gpa, facts, loaded.rows, loaded.rows - 1);
    defer cfg.deinit(gpa);
    var clock: lanes.fake.FixedClock = .{};
    var engine = lanes.Engine.init(gpa, &cfg, drafted.backend(), clock.clock());
    defer engine.deinit();
    try check.expect(engine.fills(), "gemma4 states: the drafted backend steps its prompt passes", .{});
    const gb = loaded.target;

    var tok = try core.tokenizer.loadTokenizer(gpu.io, gpa, rest[0]);
    defer tok.deinit();
    const text = try std.Io.Dir.cwd().readFileAlloc(gpu.io, rest[2], gpa, .limited(1 << 26));
    defer gpa.free(text);
    const enc = try tok.encode(gpa, text);
    defer gpa.free(enc);
    if (enc.len == 0) return error.EmptyText;
    var ids: [prompt_len]u32 = undefined;
    var other: [prompt_len]u32 = undefined;
    ids[0] = 2;
    other[0] = 2;
    for (ids[1..], other[1..], 0..) |*t, *o, i| {
        t.* = enc[i % enc.len];
        o.* = enc[(enc.len - 1) - i % enc.len];
    }

    // the keep pass: a prompt past both marks, stepped through the drafter's wrapper as the lane host runs it
    var keep: Keep = .{ .gb = gb };
    defer for (keep.saved) |k| if (k) |v| gb.dropState(v);
    var ms = marks;
    {
        var s = try lanes.Stream.init(gpa, .{ .id = "keep", .prompt = ids[0 .. marks[marks.len - 1] + 100], .max_new = 1, .drafts = true, .reuse = .{ .marks = &ms, .hook = .{ .ptr = &keep, .at = Keep.hook } } });
        defer s.deinit(gpa);
        var first = true;
        while (!try engine.fillStream(&s, first)) first = false;
        while (engine.activeCount() > 0) try engine.step();
    }
    var bad: usize = 0;
    for (keep.saved, marks) |k, at| {
        const v = k orelse return error.NothingKept;
        const ok = v.draft != null;
        if (!ok) bad += 1;
        std.debug.print("RESULT gemma4 state at {d}: drafter context {s}\n", .{ at, if (ok) "kept" else "MISSING" });
    }

    const sp: lanes.Sampling = .{ .seed = 3, .temperature = 1.0, .top_k = 0, .top_p = 0.95 };
    for ([_]?lanes.Sampling{ null, sp }) |sampling| {
        const fresh = try run(gpu, &engine, gb, .{ .prompt = &ids, .sampling = sampling }, count);
        defer free(gpa, fresh);
        if (sampling == null) {
            const plain = try run(gpu, &engine, gb, .{ .prompt = &ids, .drafts = false }, count);
            defer free(gpa, plain);
            const eq = std.mem.eql(u32, plain.ids, fresh.ids);
            if (!eq) bad += 1;
            std.debug.print("RESULT gemma4 drafted vs plain (greedy): {s}\n", .{if (eq) "same" else "DIFFER"});
        }
        for (keep.saved, marks) |k, at| {
            // another prompt through the same slot first: only what the restore writes back holds this prompt's rows
            const scrub = try run(gpu, &engine, gb, .{ .prompt = &other }, 2); // two: a reply of one releases the stream with its first token
            free(gpa, scrub);
            const got = try run(gpu, &engine, gb, .{ .prompt = &ids, .sampling = sampling, .reuse = .{ .saved = k.?, .at = at } }, count);
            defer free(gpa, got);
            const eq = got.cached == at and std.mem.eql(u8, got.logits, fresh.logits) and std.mem.eql(u32, got.ids, fresh.ids);
            if (!eq) bad += 1;
            std.debug.print("RESULT gemma4 resumed at {d} ({s}): {d} cached, logits {s}, reply {s}\n", .{ at, if (sampling == null) "greedy" else "sampled", got.cached, if (std.mem.eql(u8, got.logits, fresh.logits)) "same" else "DIFFER", if (std.mem.eql(u32, got.ids, fresh.ids)) "same" else "DIFFER" });
        }
    }
    try check.expect(bad == 0, "gemma4 states: {d} checks failed", .{bad});
    check.pass("gemma4 states: stepped drafted passes keep the drafter's context at 1000 and 1536; resumed == fresh (logits, greedy and sampled, {d} tokens)", .{count});
}
