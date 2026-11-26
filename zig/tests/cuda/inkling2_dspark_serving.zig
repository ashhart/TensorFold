//! CUDA DSpark serving: greedy and sampled parity, chunked prompts and four shared target streams.
const std = @import("std");
const lanes = @import("lanes");
const ink = @import("inkling2");
const check = @import("check.zig");
const Backend = @FieldType(ink.native.Owned, "backend");
const Saved = @FieldType(ink.native.Owned, "model").Saved;
const sampled: lanes.Sampling = .{ .seed = 42, .temperature = 0.8, .top_k = 64, .top_p = 0.95 };

const Result = struct {
    ids: []u32,
    proposed: u64,
    cached: u32,
    reuse_failed: bool,
    accepted: u64,

    fn deinit(r: Result, gpa: std.mem.Allocator) void {
        gpa.free(r.ids);
    }
};

fn idsFrom(gpu: check.Gpu, path: []const u8, prefix: []const u8) ![]u32 {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(gpu.io, path, gpu.gpa, .limited(1 << 20));
    defer gpu.gpa.free(bytes);
    const parsed = try std.json.parseFromSlice(std.json.Value, gpu.gpa, bytes, .{});
    defer parsed.deinit();
    const cases = parsed.value.object.get("cases") orelse return error.DSparkTestFixture;
    for (cases.array.items) |case| {
        const text = case.object.get("text") orelse continue;
        if (!std.mem.startsWith(u8, text.string, prefix)) continue;
        const saved = case.object.get("ids") orelse return error.DSparkTestFixture;
        const ids = try gpu.gpa.alloc(u32, saved.array.items.len);
        for (saved.array.items, ids) |number, *id| id.* = @intCast(number.integer);
        return ids;
    }
    return error.DSparkTestFixture;
}

fn generate(gpu: check.Gpu, backend: lanes.backend.Backend, facts: lanes.Model, prompt: []const u32, drafts: bool, sampling: ?lanes.Sampling, count: u32, reuse: lanes.stream.Reuse) !Result {
    var stream = try lanes.Stream.init(gpu.gpa, .{ .id = "inkling-dspark", .prompt = prompt, .max_new = count, .drafts = drafts, .sampling = sampling, .reuse = reuse });
    defer stream.deinit(gpu.gpa);
    var config = try lanes.Config.init(gpu.gpa, facts, ink.native.prompt_rows, ink.native.window_rows);
    defer config.deinit(gpu.gpa);
    config.timed = false;
    config.lone_copies = false;
    var clock: lanes.fake.FixedClock = .{};
    var runner = lanes.Engine.init(gpu.gpa, &config, backend, clock.clock());
    defer runner.deinit();
    try runner.addStream(&stream);
    var steps: usize = 0;
    while (!stream.finished) {
        if (steps == 4096) return error.DSparkServingTimedOut;
        try runner.step();
        steps += 1;
    }
    if (stream.reason != .length) return error.DSparkServingDidNotFinish;
    return .{ .ids = try gpu.gpa.dupe(u32, stream.emitted()), .proposed = stream.drafted, .accepted = stream.accepted, .cached = stream.cached, .reuse_failed = stream.reuse_failed };
}

fn cohort(gpu: check.Gpu, backend: lanes.backend.Backend, facts: lanes.Model, prompts: [4][]const u32, drafts: bool, reuse: [4]lanes.stream.Reuse, count: u32) ![4]Result {
    var streams: [4]lanes.Stream = undefined;
    var made: usize = 0;
    defer for (streams[0..made]) |*s| s.deinit(gpu.gpa);
    for (prompts, 0..) |prompt, i| {
        streams[i] = try lanes.Stream.init(gpu.gpa, .{
            .id = "inkling-dspark-cohort",
            .prompt = prompt,
            .max_new = count,
            .drafts = drafts,
            .sampling = if (i % 2 == 0) null else sampled,
            .reuse = reuse[i],
        });
        made += 1;
    }
    var config = try lanes.Config.init(gpu.gpa, facts, ink.native.prompt_rows, ink.native.window_rows);
    defer config.deinit(gpu.gpa);
    config.timed = false;
    config.lone_copies = false;
    var clock: lanes.fake.FixedClock = .{};
    var runner = lanes.Engine.init(gpu.gpa, &config, backend, clock.clock());
    defer runner.deinit();
    for (&streams) |*s| try runner.addStream(s);
    var steps: usize = 0;
    while (true) {
        var live = false;
        for (&streams) |*s| live = live or !s.finished;
        if (!live) break;
        if (steps == 4096) return error.DSparkServingTimedOut;
        try runner.step();
        steps += 1;
    }
    var out: [4]Result = undefined;
    var done: usize = 0;
    errdefer for (out[0..done]) |r| r.deinit(gpu.gpa);
    for (&streams, 0..) |*s, i| {
        if (s.reason != .length) return error.DSparkServingDidNotFinish;
        out[i] = .{ .ids = try gpu.gpa.dupe(u32, s.emitted()), .proposed = s.drafted, .accepted = s.accepted, .cached = s.cached, .reuse_failed = s.reuse_failed };
        done += 1;
    }
    return out;
}

/// Capture the target's actual saved prompt state, not a synthetic cached-token count.
fn snapshot(gpu: check.Gpu, loaded: *ink.native.Loaded, prompt: []const u32, cut: u32) !*Saved {
    const Keeper = struct {
        target: *Backend,
        saved: ?*Saved = null,

        fn at(ptr: *anyopaque, s: *lanes.Stream, mark: u32) void {
            const k: *@This() = @ptrCast(@alignCast(ptr));
            k.saved = k.target.saveState(s, mark) catch null;
        }
    };
    var s = try lanes.Stream.init(gpu.gpa, .{ .id = "dspark-snapshot", .prompt = prompt, .max_new = 1, .reuse = .{ .marks = &.{cut} } });
    defer s.deinit(gpu.gpa);
    defer loaded.backend.release(&s);
    var keeper: Keeper = .{ .target = &loaded.target.backend };
    s.reuse.hook = .{ .ptr = &keeper, .at = Keeper.at };
    try loaded.backend.prefill(&s);
    return keeper.saved orelse return error.DSparkSnapshotNotSaved;
}

fn resumedAt(gpu: check.Gpu, loaded: *ink.native.Loaded, drafted: *lanes.drafted.Drafted, prompt: []const u32, cut: u32, sampling: ?lanes.Sampling) !void {
    const saved = try snapshot(gpu, loaded, prompt, cut);
    defer loaded.target.backend.dropState(saved);
    const plain = try generate(gpu, loaded.backend, loaded.facts, prompt, false, sampling, 32, .{});
    defer plain.deinit(gpu.gpa);
    const proposed = try generate(gpu, drafted.backend(), drafted.facts(loaded.facts), prompt, true, sampling, 32, .{ .saved = saved, .at = cut });
    defer proposed.deinit(gpu.gpa);
    if (proposed.reuse_failed or proposed.cached != cut) return error.DSparkSnapshotNotRestored;
    if (!std.mem.eql(u32, plain.ids, proposed.ids)) return error.DSparkChangedResumedTokens;
    if (proposed.proposed == 0) return error.DSparkResumedNotExercised;
    check.pass("Mini CUDA DSpark resumed target equality: cut {d}, {d} proposed", .{ cut, proposed.proposed });
}

pub fn run(gpu: check.Gpu, model: []const u8, ids_path: []const u8) !void {
    const short = try idsFrom(gpu, ids_path, "def f(");
    defer gpu.gpa.free(short);
    const chat = try idsFrom(gpu, ids_path, "<|message_user|>");
    defer gpu.gpa.free(chat);
    if (short.len < 16 or short.len > 512) return error.DSparkTestPromptLength;
    const long = try gpu.gpa.alloc(u32, 2053);
    defer gpu.gpa.free(long);
    for (long, 0..) |*id, i| id.* = short[i % short.len];
    var loaded = try ink.native.open(gpu.gpa, gpu.io, gpu.ctx, model, null, .{ .context = 4096, .drafts = true });
    defer loaded.deinit(loaded.ctx);
    const owned = try ink.native.openDrafter(gpu.gpa, gpu.io, loaded.ctx, model, 4);
    defer owned.close(owned.ctx);
    var drafted = try lanes.drafted.Drafted.init(gpu.gpa, loaded.backend, owned.drafter);
    defer drafted.deinit();
    try owned.reserve(owned.ctx, 1);
    for ([5][]const u32{ short, long, short, chat, chat }, 0..) |prompt, case| {
        const count: u32 = if (case == 1) 24 else 96;
        const sampling: ?lanes.Sampling = if (case == 2 or case == 3) sampled else null;
        const plain = try generate(gpu, loaded.backend, loaded.facts, prompt, false, sampling, count, .{});
        defer plain.deinit(gpu.gpa);
        const proposed = try generate(gpu, drafted.backend(), drafted.facts(loaded.facts), prompt, true, sampling, count, .{});
        defer proposed.deinit(gpu.gpa);
        if (!std.mem.eql(u32, plain.ids, proposed.ids)) return error.DSparkChangedTokens;
        if (proposed.proposed == 0 or (sampling == null and proposed.accepted == 0)) return error.DSparkNotExercised;
        check.pass("Mini CUDA DSpark target equality: case {d}, {d} proposed, {d} accepted", .{ case, proposed.proposed, proposed.accepted });
    }
    try resumedAt(gpu, &loaded, &drafted, short, @intCast(short.len / 2), null);
    try resumedAt(gpu, &loaded, &drafted, chat, @intCast(chat.len - 1), sampled);
    try resumedAt(gpu, &loaded, &drafted, long, 2048, sampled);
    try loaded.reserve(loaded.ctx, 4);
    try owned.reserve(owned.ctx, 4);
    const prompts = [4][]const u32{ short, chat, long, short[3..] };
    const plain = try cohort(gpu, loaded.backend, loaded.facts, prompts, false, .{ .{}, .{}, .{}, .{} }, 32);
    defer for (plain) |r| r.deinit(gpu.gpa);
    const proposed = try cohort(gpu, drafted.backend(), drafted.facts(loaded.facts), prompts, true, .{ .{}, .{}, .{}, .{} }, 32);
    defer for (proposed) |r| r.deinit(gpu.gpa);
    var accepted: u64 = 0;
    for (plain, proposed) |want, got| {
        if (!std.mem.eql(u32, want.ids, got.ids)) return error.DSparkChangedConcurrentTokens;
        accepted += got.accepted;
    }
    if (accepted == 0) return error.DSparkConcurrentNotExercised;
    check.pass("Mini CUDA DSpark four-stream greedy/sampled target equality, {d} accepted", .{accepted});
    // Identical prompts and sampling settings must give the same draft acceptance in separate batched slots.
    const identical = [4][]const u32{ short, short, short, short };
    const repeated = try cohort(gpu, drafted.backend(), drafted.facts(loaded.facts), identical, true, .{ .{}, .{}, .{}, .{} }, 64);
    defer for (repeated) |r| r.deinit(gpu.gpa);
    for ([_][2]usize{ .{ 0, 2 }, .{ 1, 3 } }) |pair| {
        const first = repeated[pair[0]];
        const second = repeated[pair[1]];
        if (!std.mem.eql(u32, first.ids, second.ids) or first.proposed != second.proposed or first.accepted != second.accepted)
            return error.DSparkBatchChangedProposal;
    }
    if (repeated[0].accepted == 0 or repeated[1].accepted == 0) return error.DSparkBatchNotExercised;
    check.pass("Mini CUDA DSpark identical four-stream proposals match per sampling mode", .{});
    const code_cut: u32 = @intCast(short.len / 2);
    const chat_cut: u32 = @intCast(chat.len - 1);
    const code_saved = try snapshot(gpu, &loaded, short, code_cut);
    defer loaded.target.backend.dropState(code_saved);
    const chat_saved = try snapshot(gpu, &loaded, chat, chat_cut);
    defer loaded.target.backend.dropState(chat_saved);
    const long_saved = try snapshot(gpu, &loaded, long, 2048);
    defer loaded.target.backend.dropState(long_saved);
    const reuse = [4]lanes.stream.Reuse{
        .{ .saved = code_saved, .at = code_cut },
        .{ .saved = chat_saved, .at = chat_cut },
        .{ .saved = long_saved, .at = 2048 },
        .{},
    };
    const resumed = try cohort(gpu, drafted.backend(), drafted.facts(loaded.facts), prompts, true, reuse, 32);
    defer for (resumed) |r| r.deinit(gpu.gpa);
    for (resumed, plain, 0..) |got, want, i| {
        if (!std.mem.eql(u32, got.ids, want.ids)) return error.DSparkChangedResumedConcurrentTokens;
        if (got.reuse_failed or (i < 3 and got.cached != reuse[i].at)) return error.DSparkConcurrentSnapshotNotRestored;
    }
    check.pass("Mini CUDA DSpark four-stream resumed greedy/sampled target equality", .{});
    // A closed drafter must release its CUDA module and permit a fresh head on the same target.
    var rebound = try ink.native.open(gpu.gpa, gpu.io, gpu.ctx, model, null, .{ .context = 512, .drafts = true });
    defer rebound.deinit(rebound.ctx);
    const first_head = try ink.native.openDrafter(gpu.gpa, gpu.io, rebound.ctx, model, 1);
    first_head.close(first_head.ctx);
    const next_head = try ink.native.openDrafter(gpu.gpa, gpu.io, rebound.ctx, model, 1);
    defer next_head.close(next_head.ctx);
    var next_drafted = try lanes.drafted.Drafted.init(gpu.gpa, rebound.backend, next_head.drafter);
    defer next_drafted.deinit();
    const next_plain = try generate(gpu, rebound.backend, rebound.facts, short, false, null, 32, .{});
    defer next_plain.deinit(gpu.gpa);
    const next_proposed = try generate(gpu, next_drafted.backend(), next_drafted.facts(rebound.facts), short, true, null, 32, .{});
    defer next_proposed.deinit(gpu.gpa);
    if (!std.mem.eql(u32, next_plain.ids, next_proposed.ids) or next_proposed.accepted == 0) return error.DSparkReattachChangedTokens;
    check.pass("Mini CUDA DSpark reattached head preserves target tokens", .{});
}
