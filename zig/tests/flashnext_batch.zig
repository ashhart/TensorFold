//! Checkpoint gate: shared target rounds against fresh legacy greedy replies, including admission and cancellation.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const lanes = tf.lanes;
const fx = tf.flashnext_engine;
const Backend = tf.flashnext_backend.Backend;

const Mark = struct {
    back: *Backend,
    saved: ?*anyopaque = null,
    err: ?anyerror = null,
    fn at(ptr: *anyopaque, stream: *lanes.Stream, pos: u32) void {
        const m: *Mark = @ptrCast(@alignCast(ptr));
        m.saved = Backend.snapSave(m.back, stream, pos) catch |err| {
            m.err = err;
            return;
        };
    }
};

const Reply = struct {
    a: std.mem.Allocator,
    tokens: std.ArrayList(u32) = .empty,
    fn prefilled(_: *anyopaque) void {}
    fn emitted(ptr: *anyopaque, tokens: []const u32) bool {
        const r: *Reply = @ptrCast(@alignCast(ptr));
        r.tokens.appendSlice(r.a, tokens) catch @panic("reply allocation failed");
        return false;
    }
    fn cancelled(_: *anyopaque) bool {
        return false;
    }
    fn marked(_: *anyopaque, _: usize) void {}
    fn out(r: *Reply) fx.Out {
        return .{ .ctx = r, .prefilled = prefilled, .tokens = emitted, .cancelled = cancelled, .marked = marked };
    }
};

fn prompt(a: std.mem.Allocator, path: []const u8) ![]u32 {
    const f = try mtl.MappedFile.open(try std.fmt.allocPrintSentinel(a, "{s}", .{path}, 0));
    defer f.deinit();
    const json = try std.json.parseFromSliceLeaky(std.json.Value, a, f.bytes[0..f.size], .{});
    const items = json.object.get("prompt").?.array.items;
    const ids = try a.alloc(u32, items.len);
    for (items, ids) |item, *id| id.* = @intCast(item.integer);
    return ids;
}

pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    if (args.len < 4) return error.ExpectedModelAndTwoPromptFiles;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const n: u32 = if (std.c.getenv("FZ_N")) |v| try std.fmt.parseInt(u32, std.mem.span(v), 10) else 32;
    if (n < 8) return error.NeedEightReplyTokens;
    const count = args.len - 2;
    if (count > 8) return error.TooManyPrompts;
    const prompts = try a.alloc([]u32, count);
    const refs = try a.alloc(Reply, count);
    const eng = try fx.Engine.load(init.gpa, init.io, args[1], null);
    defer eng.deinit();
    try eng.warm();
    eng.copy = false;
    var capacity: usize = 1024;
    var baseline_seconds: f64 = 0;
    var baseline_mtp_seconds: f64 = 0;
    for (args[2..], prompts, refs) |path, *ids, *ref| {
        ids.* = try prompt(a, path);
        capacity = @max(capacity, ids.len + n + fx.MARGIN);
        ref.* = .{ .a = a };
        const start = mtl.clock.seconds();
        _ = try eng.generateFrom(ids.*, 0, &.{}, n, &.{}, 0, ref.out());
        baseline_seconds += mtl.clock.seconds() - start;
        if (ref.tokens.items.len != n) return error.ShortBaseline;
        var drafted: Reply = .{ .a = a };
        const mtp_start = mtl.clock.seconds();
        _ = try eng.generateFrom(ids.*, 0, &.{}, n, &.{}, null, drafted.out());
        baseline_mtp_seconds += mtl.clock.seconds() - mtp_start;
        if (!std.mem.eql(u32, drafted.tokens.items, ref.tokens.items)) return error.LegacyMtpTokenMismatch;
    }
    eng.hostMode();
    var longest: usize = 0;
    for (prompts, 0..) |ids, i| if (ids.len > prompts[longest].len) {
        longest = i;
    };
    var back = try Backend.init(init.gpa, eng, @intCast(@max(4, count)), capacity);
    defer back.deinit();
    for ([_]bool{ false, true }) |drafts| {
        var cfg = try lanes.Config.init(init.gpa, back.facts(), tf.flashnext_replay.MAXR, 15);
        defer cfg.deinit(init.gpa);
        var clock: lanes.backend.WallClock = .{ .io = init.io };
        var core = lanes.Engine.init(init.gpa, &cfg, back.backend(), clock.clock());
        defer core.deinit();
        const streams = try a.alloc(lanes.Stream, count);
        for (streams, prompts, 0..) |*s, ids, i| s.* = try lanes.Stream.init(init.gpa, .{
            .id = try std.fmt.allocPrint(a, "stream-{d}", .{i}),
            .prompt = ids,
            .max_new = n,
            .drafts = drafts,
        });
        defer for (streams) |*s| s.deinit(init.gpa);
        const start = mtl.clock.seconds();
        const before = back.shared_count;
        for (streams) |*s| try core.addStream(s);
        while (core.activeCount() > 0) try core.step();
        const seconds = mtl.clock.seconds() - start;
        for (streams, refs) |s, ref| {
            if (!std.mem.eql(u32, s.emitted(), ref.tokens.items)) {
                std.debug.print("FAIL {s} drafts {any}: got {any}, expected {any}\n", .{ s.id, drafts, s.emitted(), ref.tokens.items });
                return error.TokenMismatch;
            }
        }
        if (back.shared_count == before) return error.NoSharedForward;
        const baseline = if (drafts) baseline_mtp_seconds else baseline_seconds;
        std.debug.print("PASS drafts {any}: {d} streams, {d} exact tokens, {d} shared forwards, {d:.3}s, legacy serial {d:.3}s\n", .{ drafts, count, count * n, back.shared_count - before, seconds, baseline });
        // Admit a third prompt after a shared round, cancel its peer, and reuse the released slot.
        var first = try lanes.Stream.init(init.gpa, .{ .id = "survivor", .prompt = prompts[0], .max_new = n, .drafts = drafts });
        defer first.deinit(init.gpa);
        var cancelled = try lanes.Stream.init(init.gpa, .{ .id = "cancelled", .prompt = prompts[1], .max_new = n, .drafts = drafts });
        defer cancelled.deinit(init.gpa);
        var admitted = try lanes.Stream.init(init.gpa, .{ .id = "admitted", .prompt = prompts[1], .max_new = n, .drafts = drafts });
        defer admitted.deinit(init.gpa);
        try core.addStream(&first);
        try core.addStream(&cancelled);
        try core.step();
        core.discard(&cancelled);
        try core.addStream(&admitted);
        while (core.activeCount() > 0) try core.step();
        if (!std.mem.eql(u32, first.emitted(), refs[0].tokens.items) or
            !std.mem.eql(u32, admitted.emitted(), refs[1].tokens.items)) return error.AdmissionTokenMismatch;
        std.debug.print("PASS drafts {any}: cancellation, slot reuse and prompt admission\n", .{drafts});
        if (prompts[longest].len <= eng.pr.step) return error.NeedMultiChunkFixture;
        const marks = [_]u32{@intCast(eng.pr.step / 2)};
        var mark: Mark = .{ .back = &back };
        var primer = try lanes.Stream.init(init.gpa, .{
            .id = "cache-primer",
            .prompt = prompts[longest],
            .max_new = 1,
            .drafts = drafts,
            .reuse = .{ .marks = &marks, .hook = .{ .ptr = &mark, .at = Mark.at } },
        });
        defer primer.deinit(init.gpa);
        try core.addStream(&primer);
        if (mark.err) |err| return err;
        const saved = mark.saved orelse return error.NoSnapshot;
        defer Backend.snapDrop(&back, saved);
        var peer = try lanes.Stream.init(init.gpa, .{ .id = "cache-peer", .prompt = prompts[0], .max_new = n, .drafts = drafts });
        defer peer.deinit(init.gpa);
        var resumed = try lanes.Stream.init(init.gpa, .{
            .id = "cache-resumed",
            .prompt = prompts[longest],
            .max_new = n,
            .drafts = drafts,
            .reuse = .{ .saved = saved, .at = marks[0] },
        });
        defer resumed.deinit(init.gpa);
        try core.addStream(&peer);
        try core.step();
        try core.addStream(&resumed);
        if (resumed.cached != marks[0]) return error.PrefixWasNotRestored;
        while (core.activeCount() > 0) try core.step();
        if (!std.mem.eql(u32, peer.emitted(), refs[0].tokens.items) or
            !std.mem.eql(u32, resumed.emitted(), refs[longest].tokens.items)) return error.CacheTokenMismatch;
        std.debug.print("PASS drafts {any}: multi-chunk mark and restored prefix in a dynamic-capacity slot\n", .{drafts});
    }
}
