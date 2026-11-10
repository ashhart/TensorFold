//! Readout requests through the real lane host: no token drawn, logits into the request, one release on every path.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const pc = @import("prompt_cache.zig");
const readout = lanes.readout;
const a = std.testing.allocator;
const io = std.testing.io;

/// A stepped fake whose readout is each option span's token sum plus 100 times its question, as BF16.
const Reader = struct {
    base: lanes.fake.Fake = .{ .gpa = a, .prefill_chunks = 3 },
    table: lanes.backend.Backend.VTable = undefined,
    batched: bool = false, // the fake's `prefill_many` with `many_rows`
    firsts: usize = 0,
    finishes: usize = 0,
    releases: usize = 0,
    mode: enum { normal, fail, infinite, partial, blocked, fail_piece } = .normal,
    hold: enum { none, inside, between } = .none, // where piece `hold_at` waits for `gate`
    hold_at: usize = 1,
    gate: std.atomic.Value(bool) = .init(true),
    held: std.atomic.Value(bool) = .init(false),
    steps: [256]u32 = undefined,
    step_count: usize = 0,
    first_release_step: usize = 0,
    fn backend(r: *Reader) lanes.backend.Backend {
        r.table = (if (r.batched) r.base.batchedSteps() else r.base.stepped()).vtable.*;
        r.table.first = first;
        r.table.prefill_step = step;
        r.table.readout_finish = finish;
        r.table.release = release;
        return .{ .ptr = &r.base, .vtable = &r.table };
    }
    fn cast(ptr: *anyopaque) *Reader {
        const base: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
        return @fieldParentPtr("base", base);
    }
    fn wait(r: *Reader) void {
        r.held.store(true, .release);
        while (!r.gate.load(.acquire)) std.Thread.yield() catch {};
    }
    fn first(ptr: *anyopaque, _: *lanes.Stream, _: u64) !u64 {
        cast(ptr).firsts += 1;
        return error.ReadoutDrewToken;
    }
    fn step(ptr: *anyopaque, s: *lanes.Stream, max_rows: u32) !bool {
        const r = cast(ptr);
        const at = r.step_count;
        std.debug.assert(at < r.steps.len);
        r.steps[at] = s.prompt()[0];
        r.step_count += 1;
        if (r.hold == .inside and at == r.hold_at) r.wait();
        if (r.mode == .fail_piece and at == r.hold_at) return error.InjectedPieceFailure;
        const done = try r.base.stepped().vtable.prefill_step.?(ptr, s, max_rows);
        if (r.hold == .between and at + 1 == r.hold_at) r.wait();
        return done;
    }
    fn finish(ptr: *anyopaque, s: *lanes.Stream, logits: []u16) !void {
        const r = cast(ptr);
        r.finishes += 1;
        if (r.mode == .fail) return error.InjectedReadoutFailure;
        if (r.mode == .blocked) r.wait();
        const spec = s.readout orelse return error.ReadoutMetadataMissing;
        const history = r.base.lanes.get(s).?.history.items;
        if (!std.mem.eql(u32, history, s.prompt())) return error.IncompleteReadoutPrompt;
        for (spec.option_spans, spec.option_question, logits, 0..) |span, k, *out, j| {
            if (r.mode == .partial and j + 1 == logits.len) continue; // left as the host preset it
            var sum: u32 = 100 * @as(u32, k);
            for (history[span[0]..span[1]]) |token| sum += token;
            out.* = bf16(@floatFromInt(sum));
        }
        if (r.mode == .infinite) logits[0] = 0x7f80;
    }
    fn release(ptr: *anyopaque, s: *lanes.Stream) void {
        const r = cast(ptr);
        if (r.releases == 0) r.first_release_step = r.step_count;
        r.releases += 1;
        r.base.backend().vtable.release(ptr, s);
    }
};

fn bf16(v: f32) u16 {
    return @intCast(@as(u32, @bitCast(v)) >> 16); // exact for the small integers these tests make
}

const Box = struct {
    mutex: std.Io.Mutex = .init,
    done: ?api.Reason = null,
    tokens: usize = 0,
    cached: ?u32 = null,
    fn event(ptr: *anyopaque, _: api.Id, value: *const api.Event) void {
        const b: *Box = @ptrCast(@alignCast(ptr));
        b.mutex.lockUncancelable(io);
        defer b.mutex.unlock(io);
        switch (value.*) {
            .tokens => |tokens| b.tokens += tokens.len,
            .prefilled => |c| b.cached = c,
            .finished => |f| b.done = f.reason,
            else => {},
        }
    }
    fn wait(b: *Box) !api.Reason {
        for (0..10000) |_| {
            b.mutex.lockUncancelable(io);
            const done = b.done;
            b.mutex.unlock(io);
            if (done) |r| return r;
            try std.Io.sleep(io, .fromMilliseconds(1), .awake);
        }
        return error.ReadoutWorkerTimeout;
    }
};

const cap: readout.Info = .{ .max_rows = 64, .max_questions = 4, .max_options = 8 };
const prompt = [_]u32{ 2, 7, 1, 8, 3, 5, 9, 4 };

/// Question 0 at [0, 2) with options [2, 4) and [4, 5); question 1 at [5, 6) with option [6, 8).
fn specFor(logits: []u16) readout.Spec {
    return .{ .question_spans = &.{ .{ 0, 2 }, .{ 5, 6 } }, .question_types = &.{ 1, 0 }, .option_spans = &.{ .{ 2, 4 }, .{ 4, 5 }, .{ 6, 8 } }, .option_question = &.{ 0, 0, 1 }, .logits = logits };
}

fn ask(s: *const readout.Spec, tokens: []const u32) api.Request {
    return .{ .prompt = tokens, .max_tokens = 0, .drafts = false, .readout = s };
}

const Rig = struct {
    cfg: lanes.Config,
    clock: lanes.fake.FixedClock = .{},
    core: lanes.Engine,
    host: api.LaneHost,
    /// `r` must stay at its address; the host is not started.
    fn init(rig: *Rig, r: *Reader, lane_count: u32) !void {
        rig.cfg = try lanes.Config.init(a, .{}, 1, 0);
        rig.clock = .{};
        rig.core = lanes.Engine.init(a, &rig.cfg, r.backend(), rig.clock.clock());
        rig.host = api.LaneHost.init(a, io, &rig.core, .{ .text_generation = false, .lanes = lane_count, .context_window = 64, .readouts = cap });
    }
    fn deinit(rig: *Rig) void {
        rig.core.deinit();
        rig.cfg.deinit(a);
    }
};

fn expectLogits(logits: []const u16) !void {
    try std.testing.expectEqualSlices(u16, &.{ bf16(9), bf16(3), bf16(113) }, logits);
}

test "readout lane: a filled prompt gives its logits with no first draw and exactly one release" {
    var r: Reader = .{};
    defer r.base.deinit();
    var rig: Rig = undefined;
    try rig.init(&r, 2);
    defer rig.deinit();
    try rig.host.start();
    defer rig.host.stop();
    var logits: [3]u16 = @splat(0);
    const s = specFor(&logits);
    const request = ask(&s, &prompt);
    var box: Box = .{};
    try rig.host.engine().submit(1, &request, .{ .ctx = &box, .event = Box.event });
    try std.testing.expectEqual(api.Reason.length, try box.wait());
    try expectLogits(&logits);
    try std.testing.expectEqual(@as(usize, 0), r.firsts);
    try std.testing.expectEqual(@as(usize, 1), r.finishes);
    try std.testing.expectEqual(@as(usize, 1), r.releases);
    try std.testing.expectEqual(@as(usize, 0), box.tokens);
    try std.testing.expectEqual(@as(usize, 0), r.base.lanes.count());
}

test "readout lane: two readouts' prompt pieces interleave before either finishes" {
    var r: Reader = .{ .gate = .init(false) };
    r.base.prefill_chunks = 32; // a row a piece: the waiting one ages into its turns
    defer r.base.deinit();
    var rig: Rig = undefined;
    try rig.init(&r, 2);
    defer rig.deinit();
    r.hold = .inside;
    r.hold_at = 0;
    try rig.host.start();
    defer rig.host.stop();
    defer r.gate.store(true, .release);
    const tail: [24]u32 = @splat(1); // long prompts: short ones fill back to back
    const one = prompt ++ tail;
    const two = [_]u32{6} ++ one[1..].*;
    var logits: [2][3]u16 = undefined;
    const specs = [_]readout.Spec{ specFor(&logits[0]), specFor(&logits[1]) };
    const requests = [_]api.Request{ ask(&specs[0], &one), ask(&specs[1], &two) };
    var boxes: [2]Box = .{ .{}, .{} };
    for (&requests, &boxes, 1..) |*q, *box, id| try rig.host.engine().submit(id, q, .{ .ctx = box, .event = Box.event });
    r.gate.store(true, .release);
    for (&boxes) |*box| try std.testing.expectEqual(api.Reason.length, try box.wait());
    const before = r.steps[0..r.first_release_step];
    try std.testing.expect(std.mem.indexOfScalar(u32, before, 2) != null and std.mem.indexOfScalar(u32, before, 6) != null);
    for (&logits) |*l| try expectLogits(l);
    try std.testing.expectEqual(@as(usize, 0), r.firsts);
    try std.testing.expectEqual(@as(usize, 2), r.releases);
}

test "readout lane: a failing, non-finite or unwritten finish ends failed with one release" {
    for ([_]@TypeOf(@as(Reader, .{}).mode){ .fail, .infinite, .partial }) |mode| {
        var r: Reader = .{ .mode = mode };
        defer r.base.deinit();
        var rig: Rig = undefined;
        try rig.init(&r, 1);
        defer rig.deinit();
        try rig.host.start();
        defer rig.host.stop();
        var logits: [3]u16 = @splat(bf16(1)); // finite: only the host's preset makes an unwritten logit fail
        const s = specFor(&logits);
        const request = ask(&s, &prompt);
        var box: Box = .{};
        try rig.host.engine().submit(1, &request, .{ .ctx = &box, .event = Box.event });
        try std.testing.expectEqual(api.Reason.failed, try box.wait());
        try std.testing.expectEqual(@as(usize, 1), r.finishes);
        try std.testing.expectEqual(@as(usize, 1), r.releases);
        try std.testing.expectEqual(@as(usize, 0), r.base.lanes.count());
    }
}

test "readout lane: a cancel inside the finish, between pieces or inside a piece ends cancelled with one release" {
    const Where = enum { finish, between, inside };
    for ([_]Where{ .finish, .between, .inside }) |where| {
        var r: Reader = .{ .gate = .init(false) };
        r.base.prefill_chunks = 4;
        defer r.base.deinit();
        var rig: Rig = undefined;
        try rig.init(&r, 1);
        defer rig.deinit();
        var tmp = std.testing.tmpDir(.{});
        defer tmp.cleanup();
        const path = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}/trace.jsonl", .{tmp.sub_path});
        defer a.free(path);
        rig.host.trace = try @import("lane_trace.zig").Trace.open(a, io, path);
        switch (where) {
            .finish => r.mode = .blocked,
            .between => r.hold = .between,
            .inside => r.hold = .inside,
        }
        try rig.host.start();
        defer rig.host.stop();
        defer r.gate.store(true, .release);
        var logits: [3]u16 = undefined;
        const s = specFor(&logits);
        const request = ask(&s, &prompt);
        var box: Box = .{};
        try rig.host.engine().submit(1, &request, .{ .ctx = &box, .event = Box.event });
        var waited: usize = 0;
        while (!r.held.load(.acquire)) : (waited += 1) {
            if (waited > 10000) return error.ReadoutWorkerTimeout;
            try std.Io.sleep(io, .fromMilliseconds(1), .awake);
        }
        rig.host.engine().cancel(1);
        r.gate.store(true, .release);
        try std.testing.expectEqual(api.Reason.cancelled, try box.wait());
        try std.testing.expectEqual(@as(usize, @intFromBool(where == .finish)), r.finishes);
        try std.testing.expectEqual(@as(usize, 1), r.releases);
        try std.testing.expectEqual(@as(usize, 0), r.base.lanes.count());
        try std.testing.expectEqual(@as(usize, 0), r.firsts);
        if (where != .between) continue;
        // no call fails between pieces: the trace's one cancel line ends the request, inside its prompt
        const text = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 20));
        defer a.free(text);
        const Line = struct { kind: []const u8, id: u64, rows_to: usize, ok: bool };
        var cancels: usize = 0;
        var lines = std.mem.tokenizeScalar(u8, text, '\n');
        while (lines.next()) |raw| {
            const l = try std.json.parseFromSliceLeaky(Line, a, raw, .{ .ignore_unknown_fields = true });
            if (!std.mem.eql(u8, l.kind, "cancel")) continue;
            cancels += 1;
            try std.testing.expect(l.id == 1 and l.rows_to > 0 and l.rows_to < prompt.len);
        }
        try std.testing.expectEqual(@as(usize, 1), cancels);
    }
}

test "readout lane: every spec rule refuses before admission" {
    var logits: [3]u16 = undefined;
    const good = specFor(&logits);
    try good.validate(cap, &prompt, 64);
    const R = readout.Spec;
    const Case = struct { err: anyerror, s: R, info: readout.Info = cap, context: u32 = 64, prompt: []const u32 = &prompt };
    var tokens = prompt;
    const cases = [_]Case{
        .{ .err = error.UnsupportedReadoutKind, .s = good, .info = .{ .kinds = &.{}, .max_rows = 64, .max_questions = 4, .max_options = 8 } },
        .{ .err = error.InvalidReadoutPrompt, .s = good, .info = .{ .max_rows = 7, .max_questions = 4, .max_options = 8 } },
        .{ .err = error.InvalidReadoutPrompt, .s = good, .context = 7 },
        .{ .err = error.InvalidReadoutPrompt, .s = good, .prompt = &.{} },
        .{ .err = error.InvalidReadoutQuestions, .s = good, .info = .{ .max_rows = 64, .max_questions = 1, .max_options = 8 } },
        .{ .err = error.InvalidReadoutQuestions, .s = with(good, .{ .question_types = &.{1} }) },
        .{ .err = error.InvalidReadoutQuestions, .s = with(good, .{ .question_spans = &.{}, .question_types = &.{} }) },
        .{ .err = error.InvalidReadoutOptions, .s = good, .info = .{ .max_rows = 64, .max_questions = 4, .max_options = 2 } },
        .{ .err = error.InvalidReadoutOptions, .s = with(good, .{ .option_question = &.{ 0, 0 } }) },
        .{ .err = error.InvalidReadoutLogits, .s = with(good, .{ .logits = logits[0..2] }) },
        .{ .err = error.InvalidReadoutSpan, .s = with(good, .{ .question_spans = &.{ .{ 0, 2 }, .{ 6, 6 } } }) },
        .{ .err = error.InvalidReadoutSpan, .s = with(good, .{ .option_spans = &.{ .{ 2, 4 }, .{ 4, 5 }, .{ 6, 9 } } }) },
        .{ .err = error.InvalidReadoutSpan, .s = with(good, .{ .option_spans = &.{ .{ 2, 4 }, .{ 5, 4 }, .{ 6, 8 } } }) },
        .{ .err = error.UnknownReadoutType, .s = with(good, .{ .question_types = &.{ 1, 3 } }) },
        .{ .err = error.InvalidReadoutGroups, .s = with(good, .{ .option_question = &.{ 0, 1, 0 } }) }, // goes back
        .{ .err = error.InvalidReadoutGroups, .s = with(good, .{ .option_question = &.{ 0, 0, 2 } }) }, // names no question
        .{ .err = error.InvalidReadoutGroups, .s = with(good, .{ .option_question = &.{ 1, 1, 1 } }) }, // question 0 owns none
        .{ .err = error.InvalidReadoutGroups, .s = with(good, .{ .option_question = &.{ 0, 0, 0 } }) }, // question 1 owns none
        .{ .err = error.ReadoutOutputAliasesInput, .s = with(good, .{ .logits = @as([*]u16, @ptrCast(&tokens))[0..3] }), .prompt = &tokens },
    };
    for (cases) |c| try std.testing.expectError(c.err, c.s.validate(c.info, c.prompt, c.context));
    var types: [6]u8 align(2) = .{ 1, 0, 0, 0, 0, 0 };
    const typed = with(good, .{ .question_types = types[0..2], .logits = @as([*]u16, @ptrCast(&types))[0..3] });
    try std.testing.expectError(error.ReadoutOutputAliasesInput, typed.validate(cap, &prompt, 64));
    var spans: [3][2]u32 = .{ .{ 2, 4 }, .{ 4, 5 }, .{ 6, 8 } };
    const aliased = with(good, .{ .option_spans = &spans, .logits = @as([*]u16, @ptrCast(&spans))[0..3] });
    try std.testing.expectError(error.ReadoutOutputAliasesInput, aliased.validate(cap, &prompt, 64));
}

fn with(base: readout.Spec, changes: anytype) readout.Spec {
    var s = base;
    inline for (@typeInfo(@TypeOf(changes)).@"struct".field_names) |name| @field(s, name) = @field(changes, name);
    return s;
}

test "readout lane: generation fields, a mismatched engine and missing hooks refuse at submit" {
    var r: Reader = .{};
    defer r.base.deinit();
    var rig: Rig = undefined;
    try rig.init(&r, 1);
    defer rig.deinit();
    var logits: [3]u16 = undefined;
    const s = specFor(&logits);
    var embedded: [2]f32 = undefined;
    var box: Box = .{};
    const sink: api.Sink = .{ .ctx = &box, .event = Box.event };
    const e = rig.host.engine();
    const base = ask(&s, &prompt);
    var refused = [_]api.Request{ base, base, base, base, base, base, base, base, base };
    refused[0].max_tokens = 1;
    refused[1].drafts = true;
    refused[2].sampling = .{ .seed = 1, .temperature = 0.5 };
    refused[3].logprobs = 1;
    refused[4].prompt_logprobs = 1;
    refused[5].decode_spans = &.{.{ 0, 2 }};
    refused[6].embedding = .{ .ends = &.{8}, .dimensions = 2, .output = &embedded };
    refused[7].structure = .{ .kind = .json_schema, .text = "{}" };
    refused[8].images = &.{.{ .pixels = &.{}, .gh = 1, .gw = 1, .at = 0, .tokens = 1 }};
    for (&refused, 1..) |*q, id| try std.testing.expectError(error.InvalidRequest, e.submit(id, q, sink));
    var bad_logits = s;
    bad_logits.logits = logits[0..1];
    const invalid = ask(&bad_logits, &prompt);
    try std.testing.expectError(error.InvalidRequest, e.submit(20, &invalid, sink));
    const generation: api.Request = .{ .prompt = &prompt, .max_tokens = 1 };
    try std.testing.expectError(error.InvalidRequest, e.submit(21, &generation, sink)); // a readout engine generates nothing
    r.table.readout_finish = null;
    try std.testing.expectError(error.InvalidRequest, e.submit(22, &base, sink));
    r.table.readout_finish = Reader.finish;
    r.table.prefill_step = null;
    try std.testing.expectError(error.InvalidRequest, e.submit(23, &base, sink));
    r.table.prefill_step = Reader.step;
    rig.host.info_.readouts = null;
    rig.host.info_.text_generation = true;
    try std.testing.expectError(error.InvalidRequest, e.submit(24, &base, sink)); // a text engine takes no readout
    try std.testing.expectEqual(@as(usize, 0), r.finishes);
    try std.testing.expectEqual(@as(usize, 0), r.releases);
    try std.testing.expectEqual(@as(usize, 0), rig.host.queued.items.len);
}

test "readout lane: a request carrying both an embedding and a readout refuses on either engine" {
    var r: Reader = .{};
    defer r.base.deinit();
    var rig: Rig = undefined;
    try rig.init(&r, 1);
    defer rig.deinit();
    var logits: [3]u16 = undefined;
    const s = specFor(&logits);
    var embedded: [2]f32 = undefined;
    var both = ask(&s, &prompt);
    both.embedding = .{ .ends = &.{8}, .dimensions = 2, .output = &embedded };
    var box: Box = .{};
    const sink: api.Sink = .{ .ctx = &box, .event = Box.event };
    try std.testing.expectError(error.InvalidRequest, rig.host.engine().submit(1, &both, sink));
    rig.host.info_.readouts = null;
    rig.host.info_.embeddings = .{ .dimensions = 2, .vocabulary = 256, .max_batch_tokens = 16 };
    try std.testing.expectError(error.InvalidRequest, rig.host.engine().submit(2, &both, sink));
    try std.testing.expectEqual(@as(usize, 0), rig.host.queued.items.len);
}

test "readout lane: short readouts on a backend that batches prompts each complete alone" {
    var r: Reader = .{ .batched = true };
    defer r.base.deinit();
    var rig: Rig = undefined;
    try rig.init(&r, 2);
    defer rig.deinit();
    var logits: [2][3]u16 = undefined;
    const specs = [_]readout.Spec{ specFor(&logits[0]), specFor(&logits[1]) };
    const requests = [_]api.Request{ ask(&specs[0], &prompt), ask(&specs[1], &prompt) };
    var boxes: [2]Box = .{ .{}, .{} };
    for (&requests, &boxes, 1..) |*q, *box, id| try rig.host.engine().submit(id, q, .{ .ctx = box, .event = Box.event });
    try rig.host.start(); // both queued first: a batching backend would take them in one pass
    defer rig.host.stop();
    for (&boxes) |*box| try std.testing.expectEqual(api.Reason.length, try box.wait());
    for (&logits) |*l| try expectLogits(l);
    try std.testing.expectEqual(@as(usize, 0), r.firsts);
    try std.testing.expectEqual(@as(usize, 2), r.releases);
    try std.testing.expectEqual(@as(usize, 0), r.base.many_passes);
}

test "readout lane: a repeated readout with the prompt cache on reuses and keeps nothing" {
    var r: Reader = .{};
    defer r.base.deinit();
    var rig: Rig = undefined;
    try rig.init(&r, 1);
    defer rig.deinit();
    const Snaps = struct {
        fn bytes(_: *anyopaque, at: u32) u64 {
            return 64 + 4 * @as(u64, at);
        }
        fn save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!pc.Saved {
            const f: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
            return f.save(@ptrCast(@alignCast(owner.?)), at);
        }
        fn restore(_: *anyopaque, _: ?*anyopaque, _: pc.Saved) anyerror!void {
            return error.BackendRestores;
        }
        fn drop(ptr: *anyopaque, saved: pc.Saved) void {
            const f: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
            f.drop(saved);
        }
    };
    var store = pc.Store.init(a, .{ .ptr = &r.base, .vtable = &.{ .bytes = Snaps.bytes, .save = Snaps.save, .restore = Snaps.restore, .drop = Snaps.drop } }, .{ .min_gap = 2, .min_prompt = 0 }, 1 << 20);
    defer store.deinit();
    rig.host.cache = &store;
    try rig.host.start();
    defer rig.host.stop();
    var logits: [2][3]u16 = undefined;
    for (&logits, 1..) |*l, id| {
        const s = specFor(l);
        var request = ask(&s, &prompt);
        request.history_len = 6;
        var box: Box = .{};
        try rig.host.engine().submit(id, &request, .{ .ctx = &box, .event = Box.event });
        try std.testing.expectEqual(api.Reason.length, try box.wait());
        try std.testing.expectEqual(@as(?u32, 0), box.cached);
        try expectLogits(l);
    }
    try std.testing.expectEqualSlices(u8, std.mem.asBytes(&logits[0]), std.mem.asBytes(&logits[1]));
    try std.testing.expectEqual(@as(u64, 0), store.counts.kept);
    try std.testing.expectEqual(@as(usize, 0), store.entries.items.len);
    try std.testing.expectEqual(@as(usize, 2), r.releases);
}

test "readout lane: TF_LANE_TRACE's lines give each request's submit, pieces in row order and completion" {
    var r: Reader = .{}; // three pieces a prompt
    defer r.base.deinit();
    var rig: Rig = undefined;
    try rig.init(&r, 2);
    defer rig.deinit();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}/trace.jsonl", .{tmp.sub_path});
    defer a.free(path);
    rig.host.trace = try @import("lane_trace.zig").Trace.open(a, io, path);
    try rig.host.start();
    var stopped = false;
    defer if (!stopped) rig.host.stop(); // an early error still joins the worker before the rig goes

    const tail: [24]u32 = @splat(1);
    const long = prompt ++ tail;
    var logits: [2][3]u16 = undefined;
    const specs = [_]readout.Spec{ specFor(&logits[0]), specFor(&logits[1]) };
    const requests = [_]api.Request{ ask(&specs[0], &long), ask(&specs[1], &prompt) };
    var boxes: [2]Box = .{ .{}, .{} };
    for (&requests, &boxes, 1..) |*q, *box, id| try rig.host.engine().submit(id, q, .{ .ctx = box, .event = Box.event });
    for (&boxes) |*box| try std.testing.expectEqual(api.Reason.length, try box.wait());
    rig.host.stop(); // closes the trace
    stopped = true;
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 20));
    defer a.free(text);
    const Line = struct { t0_ns: i64, t1_ns: i64, kind: []const u8, id: u64, stream: u64, rows_from: usize, rows_to: usize, ok: bool };
    var submits: [2]usize = @splat(0);
    var reached: [2]usize = @splat(0); // each request's pieces must run in order, each from where the last ended
    var pieces: [2]usize = @splat(0);
    var completes: [2]usize = @splat(0);
    var lines = std.mem.tokenizeScalar(u8, text, '\n');
    while (lines.next()) |raw| {
        const parsed = try std.json.parseFromSlice(Line, a, raw, .{});
        defer parsed.deinit();
        const l = parsed.value;
        try std.testing.expect(l.t1_ns >= l.t0_ns and l.ok);
        if (std.mem.eql(u8, l.kind, "round")) continue;
        const k = l.id - 1;
        const len = requests[k].prompt.len;
        if (std.mem.eql(u8, l.kind, "submit")) {
            submits[k] += 1;
            try std.testing.expectEqual(len, l.rows_to);
        } else if (std.mem.eql(u8, l.kind, "fill")) {
            try std.testing.expectEqual(reached[k], l.rows_from);
            try std.testing.expect(l.rows_to > l.rows_from and l.rows_to <= len);
            reached[k] = l.rows_to;
            pieces[k] += 1;
        } else if (std.mem.eql(u8, l.kind, "complete")) {
            completes[k] += 1; // inside the last piece's call, so its line, ending first, comes first
            try std.testing.expectEqual(len, l.rows_to);
        } else return error.UnknownTraceKind;
    }
    try std.testing.expectEqualSlices(usize, &.{ 1, 1 }, &submits);
    try std.testing.expectEqualSlices(usize, &.{ 3, 3 }, &pieces);
    try std.testing.expectEqualSlices(usize, &.{ long.len, prompt.len }, &reached);
    try std.testing.expectEqualSlices(usize, &.{ 1, 1 }, &completes);
}

test "readout lane: a piece that fails is traced with ok false and the rows filled before it" {
    var r: Reader = .{ .mode = .fail_piece, .hold_at = 1 }; // three pieces; the second fails
    defer r.base.deinit();
    var rig: Rig = undefined;
    try rig.init(&r, 2);
    defer rig.deinit();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}/trace.jsonl", .{tmp.sub_path});
    defer a.free(path);
    rig.host.trace = try @import("lane_trace.zig").Trace.open(a, io, path);
    try rig.host.start();
    var stopped = false;
    defer if (!stopped) rig.host.stop();
    const tail: [24]u32 = @splat(1);
    const long = prompt ++ tail;
    var logits: [3]u16 = undefined;
    const s = specFor(&logits);
    const request = ask(&s, &long);
    var box: Box = .{};
    try rig.host.engine().submit(1, &request, .{ .ctx = &box, .event = Box.event });
    try std.testing.expectEqual(api.Reason.failed, try box.wait());
    rig.host.stop();
    stopped = true;
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 20));
    defer a.free(text);
    const Line = struct { t0_ns: i64, t1_ns: i64, kind: []const u8, id: u64, stream: u64, rows_from: usize, rows_to: usize, ok: bool };
    var fills: [2]Line = undefined;
    var n: usize = 0;
    var lines = std.mem.tokenizeScalar(u8, text, '\n');
    while (lines.next()) |raw| {
        const parsed = try std.json.parseFromSliceLeaky(Line, a, raw, .{});
        if (!std.mem.eql(u8, parsed.kind, "fill")) continue;
        try std.testing.expect(n < fills.len);
        fills[n] = parsed;
        n += 1;
    }
    try std.testing.expectEqual(@as(usize, 2), n);
    try std.testing.expect(fills[0].ok and fills[0].rows_from == 0 and fills[0].rows_to > 0);
    // the failed call filled nothing past the first piece: its rows end where they began, not at the prompt's end
    try std.testing.expect(!fills[1].ok and fills[1].rows_from == fills[0].rows_to and fills[1].rows_to == fills[0].rows_to);
}

test "readout lane: a completion that fails marks its line and the call holding it ok false" {
    var r: Reader = .{ .mode = .fail }; // every piece fills; the head's finish fails
    defer r.base.deinit();
    var rig: Rig = undefined;
    try rig.init(&r, 2);
    defer rig.deinit();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}/trace.jsonl", .{tmp.sub_path});
    defer a.free(path);
    rig.host.trace = try @import("lane_trace.zig").Trace.open(a, io, path);
    try rig.host.start();
    var stopped = false;
    defer if (!stopped) rig.host.stop();
    const tail: [24]u32 = @splat(1);
    const long = prompt ++ tail;
    var logits: [3]u16 = undefined;
    const s = specFor(&logits);
    const request = ask(&s, &long);
    var box: Box = .{};
    try rig.host.engine().submit(1, &request, .{ .ctx = &box, .event = Box.event });
    try std.testing.expectEqual(api.Reason.failed, try box.wait());
    rig.host.stop();
    stopped = true;
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 20));
    defer a.free(text);
    const Line = struct { t0_ns: i64, t1_ns: i64, kind: []const u8, id: u64, stream: u64, rows_from: usize, rows_to: usize, ok: bool };
    var fills: [3]bool = undefined;
    var n: usize = 0;
    var complete: ?bool = null;
    var lines = std.mem.tokenizeScalar(u8, text, '\n');
    while (lines.next()) |raw| {
        const parsed = try std.json.parseFromSliceLeaky(Line, a, raw, .{});
        if (std.mem.eql(u8, parsed.kind, "complete")) complete = parsed.ok;
        if (!std.mem.eql(u8, parsed.kind, "fill")) continue;
        try std.testing.expect(n < fills.len);
        fills[n] = parsed.ok;
        n += 1;
    }
    try std.testing.expectEqual(@as(usize, 3), n);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false }, &fills);
    try std.testing.expectEqual(@as(?bool, false), complete);
}
