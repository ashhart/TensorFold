//! Real Inkling 2 Mini CUDA serving: device draws against host selection, snapshots, admission and copied drafts.
const std = @import("std");
const lanes = @import("lanes");
const mini = @import("inkling2");
const cuda = @import("cuda");
const check = @import("check.zig");

const native = mini.native;
const Backend = @TypeOf(@as(native.Owned, undefined).backend);
const Gpu = check.Gpu;
const sampled: lanes.Sampling = .{ .seed = 37, .temperature = 0.8, .top_k = 64, .top_p = 0.95 };

const Trace = struct {
    ids: []u32,
    logits: []f32,
    cached: u32,
    failed: bool,

    fn deinit(t: *Trace, gpa: std.mem.Allocator) void {
        gpa.free(t.ids);
        gpa.free(t.logits);
    }
};

fn target(loaded: *native.Loaded) *Backend {
    return &loaded.target.backend;
}

fn download(b: *Backend, src: u64, out: []f32) !void {
    try b.s.synchronize();
    try cuda.memory.download(b.d, src, std.mem.sliceAsBytes(out));
}

/// A real target pass with either CUDA's draws or host selection over its downloaded fp32 logits.
fn trace(gpu: Gpu, loaded: *native.Loaded, prompt: []const u32, sampling: ?lanes.Sampling, n: usize, host: bool, reuse: lanes.stream.Reuse) !Trace {
    const t = target(loaded);
    const v = t.m.w.c.text.unpadded_vocab;
    var s = try lanes.Stream.init(gpu.gpa, .{ .id = "mini-trace", .prompt = prompt, .max_new = @intCast(n), .sampling = sampling, .reuse = reuse });
    defer s.deinit(gpu.gpa);
    const b = loaded.backend;
    defer b.release(&s);
    const ids = try gpu.gpa.alloc(u32, n);
    errdefer gpu.gpa.free(ids);
    const logits = try gpu.gpa.alloc(f32, n * v);
    errdefer gpu.gpa.free(logits);
    try b.prefill(&s);
    try download(t, try t.lastOf(&s), logits[0..v]);
    var handle: ?u64 = try b.first(&s, prompt.len);
    for (0..n) |i| {
        ids[i] = if (host) try lanes.gpu_full.draw(gpu.gpa, logits[i * v ..][0..v], sampling, @intCast(prompt.len + i), null) else try b.read(handle.?);
        if (i + 1 < n) {
            handle = try b.queue(&s, .{ .value = ids[i] }, prompt.len + i + 1);
            try download(t, t.m.s.logits, logits[(i + 1) * v ..][0..v]);
        }
    }
    return .{ .ids = ids, .logits = logits, .cached = s.cached, .failed = s.reuse_failed };
}

fn sameTrace(a: *const Trace, b: *const Trace) !void {
    if (!std.mem.eql(u32, a.ids, b.ids)) return error.MiniServingTokenMismatch;
    if (!std.mem.eql(u8, std.mem.sliceAsBytes(a.logits), std.mem.sliceAsBytes(b.logits))) return error.MiniServingLogitMismatch;
}

fn hostReference(gpu: Gpu, loaded: *native.Loaded, prompt: []const u32) !void {
    for ([_]?lanes.Sampling{ null, sampled }, 0..) |sampling, case| {
        var device = try trace(gpu, loaded, prompt, sampling, 12, false, .{});
        defer device.deinit(gpu.gpa);
        var host = try trace(gpu, loaded, prompt, sampling, 12, true, .{});
        defer host.deinit(gpu.gpa);
        sameTrace(&device, &host) catch |err| {
            std.debug.print("RESULT mini-serving host-reference case={d} mismatch={s}\n", .{ case, @errorName(err) });
            return err;
        };
    }
}

/// Save one complete state at `cut`, restore it through the genuine prompt-reuse hook and rerun the suffix.
fn resumedAt(gpu: Gpu, loaded: *native.Loaded, prompt: []const u32, cut: u32, sampling: ?lanes.Sampling, fresh: *const Trace) !void {
    const Saved = @TypeOf(loaded.target.model).Saved;
    const Keeper = struct {
        b: *Backend,
        saved: ?*Saved = null,
        fn at(ptr: *anyopaque, s: *lanes.Stream, mark: u32) void {
            const k: *@This() = @ptrCast(@alignCast(ptr));
            k.saved = k.b.saveState(s, mark) catch null;
        }
    };
    const t = target(loaded);
    const saved = snapshot: {
        var s = try lanes.Stream.init(gpu.gpa, .{ .id = "mini-snapshot", .prompt = prompt, .max_new = 1, .reuse = .{ .marks = &.{cut} } });
        defer s.deinit(gpu.gpa);
        const b = loaded.backend;
        defer b.release(&s);
        var keeper: Keeper = .{ .b = t };
        s.reuse.hook = .{ .ptr = &keeper, .at = Keeper.at };
        try b.prefill(&s);
        break :snapshot keeper.saved orelse return error.MiniSnapshotNotSaved;
    };
    defer t.dropState(saved);
    var resumed = try trace(gpu, loaded, prompt, sampling, fresh.ids.len, false, .{ .saved = saved, .at = cut });
    defer resumed.deinit(gpu.gpa);
    if (resumed.cached != cut or resumed.failed) return error.MiniSnapshotNotRestored;
    try sameTrace(fresh, &resumed);
}

/// A keeps states at 14 and 39; B resumes A's 14, keeps its 39 and decodes; A resumed at 39 must equal fresh A.
fn crossPrompt(gpu: Gpu, loaded: *native.Loaded, ids: []const u32) !void {
    const Saved = @TypeOf(loaded.target.model).Saved;
    const Keeper = struct {
        b: *Backend,
        saved: [2]?*Saved = .{ null, null },
        fn at(ptr: *anyopaque, s: *lanes.Stream, mark: u32) void {
            const k: *@This() = @ptrCast(@alignCast(ptr));
            k.saved[@intFromBool(mark != 14)] = k.b.saveState(s, mark) catch null;
        }
    };
    const t = target(loaded);
    const b = loaded.backend;
    const a = ids[0..40];
    var other: [40]u32 = undefined;
    @memcpy(other[0..14], ids[0..14]);
    @memcpy(other[14..], ids[500..526]);
    var fresh = try trace(gpu, loaded, a, null, 8, false, .{});
    defer fresh.deinit(gpu.gpa);
    var ka: Keeper = .{ .b = t };
    defer for (ka.saved) |x| if (x) |y| t.dropState(y);
    {
        var s = try lanes.Stream.init(gpu.gpa, .{ .id = "mini-a", .prompt = a, .max_new = 1, .reuse = .{ .marks = &.{ 14, 39 } } });
        defer s.deinit(gpu.gpa);
        defer b.release(&s);
        s.reuse.hook = .{ .ptr = &ka, .at = Keeper.at };
        try b.prefill(&s);
    }
    var kb: Keeper = .{ .b = t };
    defer for (kb.saved) |x| if (x) |y| t.dropState(y);
    {
        var s = try lanes.Stream.init(gpu.gpa, .{ .id = "mini-b", .prompt = &other, .max_new = 8, .reuse = .{ .saved = ka.saved[0] orelse return error.MiniSnapshotNotSaved, .at = 14, .marks = &.{39} } });
        defer s.deinit(gpu.gpa);
        defer b.release(&s);
        s.reuse.hook = .{ .ptr = &kb, .at = Keeper.at };
        try b.prefill(&s);
        if (s.cached != 14 or kb.saved[1] == null) return error.MiniSnapshotNotRestored;
        var h = try b.first(&s, other.len);
        for (0..6) |i| h = try b.queue(&s, .{ .handle = h }, other.len + i + 1);
    }
    var resumed = try trace(gpu, loaded, a, null, 8, false, .{ .saved = ka.saved[1] orelse return error.MiniSnapshotNotSaved, .at = 39 });
    defer resumed.deinit(gpu.gpa);
    if (resumed.cached != 39 or resumed.failed) return error.MiniSnapshotNotRestored;
    sameTrace(&fresh, &resumed) catch |err| {
        std.debug.print("RESULT mini-serving cross-prompt resume mismatch={s} fresh={any} resumed={any}\n", .{ @errorName(err), fresh.ids, resumed.ids });
        return err;
    };
}

/// A history on the chunk grid is kept at its end and decoded on; after another prompt, its next turn resumed == fresh.
fn turnAfterHistory(gpu: Gpu, loaded: *native.Loaded, ids: []const u32) !void {
    const Saved = @TypeOf(loaded.target.model).Saved;
    const Keeper = struct {
        b: *Backend,
        saved: ?*Saved = null,
        fn at(ptr: *anyopaque, s: *lanes.Stream, mark: u32) void {
            const k: *@This() = @ptrCast(@alignCast(ptr));
            k.saved = k.b.saveState(s, mark) catch null;
        }
    };
    const t = target(loaded);
    const b = loaded.backend;
    const history = native.prompt_rows;
    var turn: [history + 16]u32 = undefined;
    @memcpy(turn[0..history], ids[0..history]);
    @memcpy(turn[history..], ids[6000..6016]);
    var keeper: Keeper = .{ .b = t };
    defer if (keeper.saved) |x| t.dropState(x);
    {
        var s = try lanes.Stream.init(gpu.gpa, .{ .id = "mini-history", .prompt = ids[0..history], .max_new = 24, .reuse = .{ .marks = &.{history} } });
        defer s.deinit(gpu.gpa);
        defer b.release(&s);
        s.reuse.hook = .{ .ptr = &keeper, .at = Keeper.at };
        try b.prefill(&s);
        if (keeper.saved == null) return error.MiniSnapshotNotSaved;
        var h = try b.first(&s, history);
        for (0..23) |i| h = try b.queue(&s, .{ .handle = h }, history + i + 1);
    }
    for ([_]?lanes.Sampling{ null, sampled }, 0..) |sampling, case| {
        var other = try trace(gpu, loaded, ids[3000 .. 3000 + history + 100], null, 2, false, .{});
        other.deinit(gpu.gpa);
        var resumed = try trace(gpu, loaded, &turn, sampling, 24, false, .{ .saved = keeper.saved.?, .at = history });
        defer resumed.deinit(gpu.gpa);
        if (resumed.cached != history or resumed.failed) return error.MiniSnapshotNotRestored;
        var fresh = try trace(gpu, loaded, &turn, sampling, 24, false, .{});
        defer fresh.deinit(gpu.gpa);
        if (fresh.cached != 0) return error.MiniFreshTurnCached;
        sameTrace(&fresh, &resumed) catch |err| {
            std.debug.print("RESULT mini-serving turn after history case={d} mismatch={s}\n", .{ case, @errorName(err) });
            return err;
        };
    }
}

/// A state saved from one capacity restores into a cache of another: each head's kept rows land at their positions.
fn otherCapacity(gpu: Gpu, loaded: *native.Loaded) !void {
    const Model = @TypeOf(loaded.target.model);
    const ks = mini.kept_state;
    const a = &loaded.target.model;
    const s = loaded.target.stream;
    a.slot = 0;
    const window = a.w.c.text.window;
    const from = try gpu.gpa.alloc(u8, a.layout.bytes);
    defer gpu.gpa.free(from);
    // At context 3500 B's sliding rings are A's and wrap by 3300, its global ones shorter; at 1500 all are shorter.
    for ([_][2]u32{ .{ 3500, 3300 }, .{ 1500, 1400 } }) |case| {
        var b = try Model.init(gpu.gpa, a.d, s, a.w, case[0]);
        defer b.deinit();
        const to = try gpu.gpa.alloc(u8, b.layout.bytes);
        defer gpu.gpa.free(to);
        var prng = std.Random.DefaultPrng.init(case[0]);
        prng.random().bytes(from);
        prng.random().bytes(to);
        try cuda.copy_source.enqueue(a.d, a.slots[0].cache.ptr, from, s);
        try cuda.copy_source.enqueue(a.d, b.slots[0].cache.ptr, to, s);
        const saved = try a.save(s, case[1]);
        defer a.drop(s, saved);
        try b.restore(s, saved);
        try s.synchronize();
        try cuda.memory.download(a.d, b.slots[0].cache.ptr, to);
        var va: [a.w.p.visits.len]ks.Visit = undefined;
        var vb: [a.w.p.visits.len]ks.Visit = undefined;
        const at = case[1];
        for (a.keptVisits(&va), b.keptVisits(&vb), 0..) |x, y, visit| for (0..2) |side| for (0..x.heads) |h| {
            for (at - ks.kept(x.global, window, at)..at) |p| {
                const ra = (if (side == 0) x.k else x.v) - a.slots[0].cache.ptr + (h * x.ring + p % x.ring) * x.row;
                const rb = (if (side == 0) y.k else y.v) - b.slots[0].cache.ptr + (h * y.ring + p % y.ring) * y.row;
                if (!std.mem.eql(u8, from[ra..][0..x.row], to[rb..][0..y.row])) {
                    std.debug.print("RESULT mini-serving other capacity {d}: visit {d} side {d} head {d} position {d} moved\n", .{ case[0], visit, side, h, p });
                    return error.MiniKeptRowMoved;
                }
            }
        };
        if (!std.mem.eql(u8, from[a.layout.at(0).conv_offset..], to[b.layout.at(0).conv_offset..])) return error.MiniKeptConvMoved;
    }
}

fn freshAndResumed(gpu: Gpu, loaded: *native.Loaded, prompt: []const u32) !void {
    for ([_]?lanes.Sampling{ null, sampled }, 0..) |sampling, case| {
        var fresh = try trace(gpu, loaded, prompt, sampling, 8, false, .{});
        defer fresh.deinit(gpu.gpa);
        const cuts: []const u32 = if (case == 0) &.{ 17, 2048 } else &.{ 2049, @intCast(prompt.len - 1) };
        for (cuts) |cut| try resumedAt(gpu, loaded, prompt, cut, sampling, &fresh);
    }
}

fn serve(gpu: Gpu, loaded: *native.Loaded, s: *lanes.Stream, proposer: ?lanes.proposer.Proposer) !void {
    var cfg = try lanes.Config.init(gpu.gpa, loaded.facts, native.prompt_rows, native.window_rows);
    defer cfg.deinit(gpu.gpa);
    cfg.timed = false;
    var clock: lanes.fake.FixedClock = .{};
    var e = lanes.Engine.init(gpu.gpa, &cfg, loaded.backend, clock.clock());
    defer e.deinit();
    if (proposer) |p| s.proposer = p;
    try e.addStream(s);
    while (!s.finished) try e.step();
    if (s.reason != .length) return error.MiniServingDidNotFinish;
}

fn admission(gpu: Gpu, loaded: *native.Loaded, a: []const u32, b: []const u32) !void {
    var plain_a = try trace(gpu, loaded, a, null, 10, false, .{});
    defer plain_a.deinit(gpu.gpa);
    var plain_b = try trace(gpu, loaded, b, sampled, 10, false, .{});
    defer plain_b.deinit(gpu.gpa);
    var first = try lanes.Stream.init(gpu.gpa, .{ .id = "mini-first", .prompt = a, .max_new = 10 });
    defer first.deinit(gpu.gpa);
    var second = try lanes.Stream.init(gpu.gpa, .{ .id = "mini-second", .prompt = b, .max_new = 10, .sampling = sampled });
    defer second.deinit(gpu.gpa);
    var cfg = try lanes.Config.init(gpu.gpa, loaded.facts, native.prompt_rows, native.window_rows);
    defer cfg.deinit(gpu.gpa);
    cfg.timed = false;
    var clock: lanes.fake.FixedClock = .{};
    var e = lanes.Engine.init(gpu.gpa, &cfg, loaded.backend, clock.clock());
    defer e.deinit();
    if (!native.fits(loaded.ctx, @intCast(a.len), first.max_new)) return error.MiniUnexpectedFreeSlot;
    try e.addStream(&first);
    if (native.fits(loaded.ctx, @intCast(b.len), second.max_new)) return error.MiniBusyAdmissionRefused;
    try std.testing.expectError(error.NoFreeSlot, loaded.backend.prefill(&second));
    while (!first.finished) try e.step();
    try std.testing.expectEqual(lanes.stream.Reason.length, first.reason);
    if (!std.mem.eql(u32, first.emitted(), plain_a.ids)) return error.MiniAdmissionChangedFirst;
    if (!native.fits(loaded.ctx, @intCast(b.len), second.max_new)) return error.MiniSlotNotReleased;
    try e.addStream(&second);
    while (!second.finished) try e.step();
    try std.testing.expectEqual(lanes.stream.Reason.length, second.reason);
    if (!std.mem.eql(u32, second.emitted(), plain_b.ids)) return error.MiniAdmissionChangedSecond;
}

/// Four slots: interleaved greedy and seeded streams equal solo; a fifth is refused before anything is queued.
fn concurrent(gpu: Gpu, loaded: *native.Loaded, ids: []const u32) !void {
    const n = 4;
    const cuts = [n][2]usize{ .{ 0, 257 }, .{ 17, 1042 }, .{ 300, 2400 }, .{ 5, 70 } };
    var solo: [n]Trace = undefined;
    var done: usize = 0;
    defer for (solo[0..done]) |*x| x.deinit(gpu.gpa);
    for (cuts, 0..) |c, i| {
        solo[i] = try trace(gpu, loaded, ids[c[0]..c[1]], if (i % 2 == 0) null else sampled, 10, false, .{});
        done += 1;
    }
    try loaded.reserve(loaded.ctx, n);
    {
        var busy: [n]lanes.Stream = undefined;
        var opened: usize = 0;
        defer for (busy[0..opened]) |*x| {
            loaded.backend.release(x);
            x.deinit(gpu.gpa);
        };
        for (&busy, 0..) |*x, i| {
            x.* = try lanes.Stream.init(gpu.gpa, .{ .id = "mini-busy", .prompt = ids[i .. i + 8], .max_new = 1 });
            opened += 1;
            if (!native.fits(loaded.ctx, 8, 1)) return error.MiniSlotMissing;
            try loaded.backend.prefill(x);
        }
        if (native.fits(loaded.ctx, 8, 1)) return error.MiniFifthAdmitted;
        var fifth = try lanes.Stream.init(gpu.gpa, .{ .id = "mini-fifth", .prompt = ids[0..8], .max_new = 1 });
        defer fifth.deinit(gpu.gpa);
        try std.testing.expectError(error.NoFreeSlot, loaded.backend.prefill(&fifth));
        if (fifth.fill_at.load(.monotonic) != 0) return error.MiniFifthQueued;
    }
    var streams: [n]lanes.Stream = undefined;
    var made: usize = 0;
    defer for (streams[0..made]) |*x| x.deinit(gpu.gpa);
    for (cuts, 0..) |c, i| {
        streams[i] = try lanes.Stream.init(gpu.gpa, .{ .id = "mini-concurrent", .prompt = ids[c[0]..c[1]], .max_new = 10, .sampling = if (i % 2 == 0) null else sampled });
        made += 1;
    }
    var cfg = try lanes.Config.init(gpu.gpa, loaded.facts, native.prompt_rows, native.window_rows);
    defer cfg.deinit(gpu.gpa);
    cfg.timed = false;
    var clock: lanes.fake.FixedClock = .{};
    var e = lanes.Engine.init(gpu.gpa, &cfg, loaded.backend, clock.clock());
    defer e.deinit();
    for (&streams) |*x| try e.addStream(x);
    var live = true;
    while (live) {
        try e.step();
        live = false;
        for (&streams) |*x| live = live or !x.finished;
    }
    for (&streams, solo) |*x, want| {
        if (x.reason != .length) return error.MiniServingDidNotFinish;
        if (!std.mem.eql(u32, x.emitted(), want.ids)) return error.MiniConcurrentChangedTokens;
    }
}

/// Priced copies deliberately alternate whole matches, first-row rejection and a prefix match.
const Copier = struct {
    prompt_len: usize,
    plain: []const u32,
    out: [4]u32 = undefined,
    calls: usize = 0,
    rejected: bool = false,
    partial: bool = false,
    continued: bool = false,
    full: bool = false,
    was_rejected: bool = false,

    fn propose(ptr: *anyopaque, context: []const u32, most: i64) anyerror![]const u32 {
        const c: *@This() = @ptrCast(@alignCast(ptr));
        const n = context.len - c.prompt_len;
        if (n >= c.plain.len or most < 2) return &.{};
        if (c.was_rejected) c.continued = true;
        const count = @min(c.out.len, @min(c.plain.len - n, @as(usize, @intCast(most))));
        for (c.out[0..count], 0..) |*token, i| token.* = c.plain[n + i];
        const mode = c.calls % 3;
        c.was_rejected = false;
        if (mode == 0 and count >= 2) {
            c.full = true;
        } else if (mode == 1 and count >= 2) {
            c.out[0] = if (c.out[0] == 0) 1 else 0;
            c.rejected = true;
            c.was_rejected = true;
        } else if (count >= 2) {
            c.out[1] = if (c.out[1] == 0) 1 else 0;
            c.partial = true;
        }
        c.calls += 1;
        return c.out[0..count];
    }

    fn proposer(c: *Copier) lanes.proposer.Proposer {
        return .{ .ptr = c, .vtable = &.{
            .propose = propose,
            .last_match = struct {
                fn f(_: *anyopaque) i64 {
                    return 1;
                }
            }.f,
            .priced = true,
        } };
    }
};

/// Rewind a rejected, partially accepted or fully accepted window, then compare continuation bits with serial decode.
fn rollback(gpu: Gpu, loaded: *native.Loaded, prompt: []const u32, sampling: ?lanes.Sampling, plain: []const u32, accepted: usize) !void {
    const t = target(loaded);
    const v = t.m.w.c.text.unpadded_vocab;
    const next = accepted + 1;
    const ref = try gpu.gpa.alloc(f32, v);
    defer gpu.gpa.free(ref);
    var serial = try lanes.Stream.init(gpu.gpa, .{ .id = "mini-serial", .prompt = prompt, .max_new = 12, .sampling = sampling });
    defer serial.deinit(gpu.gpa);
    const b = loaded.backend;
    try b.prefill(&serial);
    for (plain[0 .. next + 1], 0..) |token, i| _ = try b.queue(&serial, .{ .value = token }, prompt.len + i + 1);
    try download(t, t.m.s.logits, ref);
    b.release(&serial);

    var drafted = try lanes.Stream.init(gpu.gpa, .{ .id = "mini-drafted", .prompt = prompt, .max_new = 12, .sampling = sampling, .drafts = true });
    defer drafted.deinit(gpu.gpa);
    defer b.release(&drafted);
    try b.prefill(&drafted);
    const pending = plain[0];
    var tokens: [4]u32 = undefined;
    for (&tokens, 0..) |*token, i| token.* = plain[i + 1];
    if (accepted < tokens.len) tokens[accepted] = if (tokens[accepted] == 0) 1 else 0;
    var positions: [5]u64 = undefined;
    for (&positions, 0..) |*p, i| p.* = prompt.len + i + 1;
    const window: lanes.backend.Window = .{ .stream = &drafted, .tokens = &tokens, .parents = null, .positions = &positions, .pending = pending, .held = 0 };
    var picks: [5]u32 = undefined;
    var drafts: [4]u32 = undefined;
    var out = [1]lanes.backend.Verified{.{ .sampled = &picks, .drafts = &drafts }};
    try b.verify(&.{window}, &out);
    if (picks[accepted] != plain[next]) return error.MiniWrongCorrection;
    var path: [5]u32 = undefined;
    for (path[0..next], 0..) |*p, i| p.* = @intCast(i);
    try b.keep(&.{window}, &.{path[0..next]});
    const handle = try b.queue(&drafted, .{ .value = plain[next] }, prompt.len + next + 1);
    const after = try gpu.gpa.alloc(f32, v);
    defer gpu.gpa.free(after);
    try download(t, t.m.s.logits, after);
    if (!std.mem.eql(u8, std.mem.sliceAsBytes(ref), std.mem.sliceAsBytes(after))) return error.MiniRollbackChangedLogits;
    if (try b.read(handle) != plain[next + 1]) return error.MiniPostRejectionTokenChanged;
}

fn copiedDrafts(gpu: Gpu, loaded: *native.Loaded, prompt: []const u32) !void {
    for ([_]?lanes.Sampling{ null, sampled }) |sampling| {
        var plain = try trace(gpu, loaded, prompt, sampling, 12, false, .{});
        defer plain.deinit(gpu.gpa);
        var copier: Copier = .{ .prompt_len = prompt.len, .plain = plain.ids };
        var s = try lanes.Stream.init(gpu.gpa, .{ .id = "mini-copied", .prompt = prompt, .max_new = 12, .sampling = sampling, .drafts = true });
        defer s.deinit(gpu.gpa);
        try serve(gpu, loaded, &s, copier.proposer());
        if (!std.mem.eql(u32, s.emitted(), plain.ids)) return error.MiniCopiesChangedTarget;
        if (s.drafted == 0 or s.accepted == 0 or s.accepted == s.drafted) return error.MiniCopyCoverageMissing;
        if (!(copier.full and copier.rejected and copier.partial and copier.continued)) return error.MiniCopyCaseMissing;
        for ([_]usize{ 0, 1, 4 }) |accepted| try rollback(gpu, loaded, prompt, sampling, plain.ids, accepted);
    }
}

fn readIds(gpu: Gpu, path: []const u8) ![]u32 {
    const text = try std.Io.Dir.cwd().readFileAlloc(gpu.io, path, gpu.gpa, .limited(1 << 26));
    defer gpu.gpa.free(text);
    var out: std.ArrayList(u32) = .empty;
    errdefer out.deinit(gpu.gpa);
    var it = std.mem.tokenizeAny(u8, text, ", \n\r\t[]");
    while (it.next()) |t| try out.append(gpu.gpa, try std.fmt.parseInt(u32, t, 10));
    return out.toOwnedSlice(gpu.gpa);
}

pub fn run(gpu: Gpu, model: []const u8, ids_path: []const u8) !void {
    const ids = try readIds(gpu, ids_path);
    defer gpu.gpa.free(ids);
    if (ids.len < 4096) return error.MiniServingPromptTooShort;
    const context = ids.len + 64;
    var loaded = try native.open(gpu.gpa, gpu.io, gpu.ctx, model, null, .{ .context = context, .drafts = true });
    defer loaded.deinit(loaded.ctx);
    try hostReference(gpu, &loaded, ids[0..4096]);
    check.pass("Mini CUDA serving matches the host reference token for token", .{});
    try freshAndResumed(gpu, &loaded, ids[0..2309]);
    check.pass("Mini CUDA fresh and resumed prompts produce identical logits and tokens", .{});
    try crossPrompt(gpu, &loaded, ids);
    check.pass("Mini CUDA a state resumed after another prompt resumed its prefix equals the fresh prompt", .{});
    try turnAfterHistory(gpu, &loaded, ids);
    check.pass("Mini CUDA a turn after a history kept at the chunk grid resumes equal to fresh, after another prompt", .{});
    try admission(gpu, &loaded, ids[0..257], ids[17..1042]);
    check.pass("Mini CUDA one-slot admission serializes concurrent requests without changing tokens", .{});
    try concurrent(gpu, &loaded, ids);
    check.pass("Mini CUDA four slots: concurrent greedy and seeded replies equal solo, a fifth is refused", .{});
    try copiedDrafts(gpu, &loaded, ids[0..64]);
    check.pass("Mini CUDA opt-in copy drafting preserves target tokens and rolls back exactly", .{});
    try otherCapacity(gpu, &loaded);
    check.pass("Mini CUDA a kept state restores each head's rows into a cache of another capacity", .{});
}
