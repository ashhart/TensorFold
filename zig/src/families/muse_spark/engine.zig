//! Muse Spark's engine on one rank: the model, its DFlash2 drafter (opt-in) and the plan-link steps rank 0 sends.

const std = @import("std");
const cuda = @import("cuda");
const tp = @import("tp");
const load_ = @import("load.zig");
const fwd = @import("forward.zig");
const dflash2 = @import("dflash2.zig");
const prompt_ = @import("prompt.zig");
const kept = @import("kept.zig");
const exl3 = @import("exl3.zig");

/// The plan link's words: one GPU step each, executed by every rank in order.
pub const Op = enum(i64) { hello = 0x3A5E, forward = 1, draft = 2, stop = 3, prompt = 4, cache = 5 };

/// What rank 0 announces first so every rank builds the same model (the prompt path's chunking is prompt.zig's).
pub const Setup = struct { context: usize, slots: usize, drafts: bool = false };

const hello_version = 9;

/// The hello words: [hello, version, context, slots, drafts].
pub fn hello(su: Setup) [5]i64 {
    return .{ @backingInt(Op.hello), hello_version, @intCast(su.context), @intCast(su.slots), @intFromBool(su.drafts) };
}

/// The setup a follower reads from rank 0's hello; another build's hello is refused.
pub fn setupOf(words: []const i64) !Setup {
    if (words.len < 2 or words[0] != @backingInt(Op.hello)) return error.NoHello;
    if (words[1] != hello_version or words.len != 5) return error.OtherBuild;
    return .{ .context = @intCast(words[2]), .slots = @intCast(words[3]), .drafts = words[4] != 0 };
}

/// The drafter pack: `given` (the same path on every rank), else `<pack>/dflash2`.
pub fn drafterDir(gpa: std.mem.Allocator, pack: []const u8, given: ?[]const u8) ![]u8 {
    if (given) |g| return gpa.dupe(u8, g);
    return std.fmt.allocPrint(gpa, "{s}/dflash2", .{pack});
}

/// The model, its kernels and the TP session: what rank 0 serves and rank 1 follows.
pub const Engine = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    w: load_.Weights,
    k3: exl3.Kernels,
    model: fwd.Model,
    dr: ?*dflash2.Drafter = null,
    pr: ?*prompt_.Prompt = null,
    session: ?*tp.Session = null,
    s: cuda.Stream,
    logits: []f32, // [DECODE_ROWS, V] host
    ids: [fwd.DECODE_ROWS]i32 = undefined, // device argmax of each row (Want.ids)
    words: std.ArrayList(i64) = .empty,
    copies: kept.Copies = undefined, // kept prompt states' windows (kept.zig), on every rank

    /// Load everything on this rank; `session` null on one GPU, `drafter` null: `<pack>/dflash2` (with su.drafts).
    pub fn init(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, dir: []const u8, su: Setup, session: ?*tp.Session, drafter: ?[]const u8) !*Engine {
        const e = try gpa.create(Engine);
        errdefer gpa.destroy(e);
        e.* = .{ .gpa = gpa, .d = d, .w = undefined, .k3 = undefined, .model = undefined, .s = undefined, .logits = &.{}, .session = session };
        const rank: usize = if (session) |ss| ss.cfg.rank else 0;
        const world: usize = if (session) |ss| ss.cfg.world else 1;
        e.s = try cuda.Stream.init(d, false);
        errdefer e.s.deinit();
        e.w = try load_.load(gpa, io, d, dir, rank, world, .{ .log = rank == 0 });
        errdefer e.w.deinit();
        e.k3 = try exl3.Kernels.load(d);
        errdefer e.k3.deinit();
        const comm = if (session) |ss| ss.comm() else null;
        e.model = try fwd.Model.init(gpa, d, &e.w, &e.k3, comm, su.context, fwd.DECODE_ROWS, su.slots, e.s);
        errdefer e.model.deinit();
        if (su.drafts) {
            const ddir = try drafterDir(gpa, dir, drafter);
            defer gpa.free(ddir);
            e.dr = try dflash2.Drafter.init(gpa, io, &e.model, ddir, e.s, true, null);
            e.model.dr = e.dr;
            if (rank == 0) std.log.info("muse drafter: {s}, block {d}, {d:.2} GiB on this rank (split over {d})", .{ ddir, e.dr.?.block, @as(f64, @floatFromInt(e.dr.?.bytes())) / (1 << 30), world });
        }
        errdefer if (e.dr) |dr| dr.deinit();
        e.pr = try prompt_.Prompt.init(&e.model, gpa, prompt_.CAP, prompt_.EWIN, e.s);
        if (rank == 0) std.log.info("muse prompt path: chunks of {d} rows, experts in {d}-row windows, {d:.2} GiB scratch", .{ e.pr.?.cap, e.pr.?.ewin, @as(f64, @floatFromInt(e.pr.?.bytes())) / (1 << 30) });
        errdefer if (e.pr) |pr| pr.deinit();
        e.copies = try kept.Copies.init(gpa, &e.model, e.dr);
        errdefer e.copies.deinit();
        e.logits = try gpa.alloc(f32, fwd.DECODE_ROWS * e.w.full.vocab);
        return e;
    }

    pub fn deinit(e: *Engine) void {
        const gpa = e.gpa;
        e.words.deinit(gpa);
        gpa.free(e.logits);
        e.copies.deinit();
        if (e.pr) |pr| pr.deinit();
        if (e.dr) |dr| dr.deinit();
        e.model.deinit();
        e.k3.deinit();
        e.w.deinit();
        e.s.deinit();
        gpa.destroy(e);
    }

    fn send(e: *Engine, words: []const i64) !void {
        if (e.session) |ss| try ss.plan.send(words);
    }

    pub const Want = enum { none, ids, logits };

    /// A graphed forward window on every rank; on rank 0 `ids` leaves each row's argmax in `ids`, `logits` [n, V].
    pub fn forward(e: *Engine, rows: []const fwd.Row, want: Want) !void {
        e.words.clearRetainingCapacity();
        try e.words.appendSlice(e.gpa, &.{ @backingInt(Op.forward), @intCast(rows.len) });
        for (rows) |r| try e.words.appendSlice(e.gpa, &.{ r.slot, r.pos, r.token });
        try e.send(e.words.items);
        try e.model.graphed(e.s, rows, if (want == .logits) e.logits[0 .. rows.len * e.w.full.vocab] else null);
        if (want == .ids) try e.model.argmaxTo(e.s, rows.len, e.ids[0..rows.len]);
    }

    /// A prompt chunk on every rank: `tokens` at [p0, p0 + n) of `slot`; the last row's argmax or logits.
    pub fn prompt(e: *Engine, slot: u32, p0: usize, tokens: []const i64, want: Want) !void {
        const pr = e.pr orelse return error.NoPromptPath;
        e.words.clearRetainingCapacity();
        try e.words.appendSlice(e.gpa, &.{ @backingInt(Op.prompt), @intCast(tokens.len), slot, @intCast(p0) });
        try e.words.appendSlice(e.gpa, tokens);
        try e.send(e.words.items);
        try pr.chunk(e.s, slot, p0, tokens);
        switch (want) {
            .none => try e.s.synchronize(),
            .ids => try e.model.argmaxTo(e.s, 1, e.ids[0..1]),
            .logits => try e.model.logitsTo(e.s, 1, e.logits[0..e.w.full.vocab]),
        }
    }

    /// A draft pass on every rank: drafts [S, block - 1] into `out` (each stream's anchor token, position, slot).
    pub fn draft(e: *Engine, tokens: []const i64, pos0: []const u32, slots: []const u32, out: []i32) !void {
        e.words.clearRetainingCapacity();
        try e.words.appendSlice(e.gpa, &.{ @backingInt(Op.draft), @intCast(tokens.len) });
        for (tokens, pos0, slots) |t, p, s| try e.words.appendSlice(e.gpa, &.{ t, p, s });
        try e.send(e.words.items);
        try e.dr.?.draft(e.s, tokens, pos0, slots, out);
    }

    /// A kept state's step on every rank (kept.zig: save, restore, copy prefix, drop); rank 0 sends it first.
    pub fn cache(e: *Engine, kind: kept.Kind, id: u64, slot: u32, slot2: u32, at: usize) !void {
        try e.send(&kept.words(@backingInt(Op.cache), kind, id, slot, slot2, at));
        switch (kind) {
            .save => try e.copies.save(e.s, id, slot),
            .restore => try e.copies.restore(e.s, id, slot),
            .copy => try e.copies.copyPrefix(e.s, slot, slot2, at),
            .drop => e.copies.drop(id),
        }
    }

    /// One step from the plan link, on a follower; false at stop.
    pub fn follow(e: *Engine, words: []const i64) !bool {
        if (words.len < 2) return error.ShortPlan;
        const op: Op = @fromBackingInt(@intCast(words[0]));
        const n: usize = @intCast(words[1]);
        switch (op) {
            .forward => {
                if (words.len != 2 + 3 * n or n > fwd.DECODE_ROWS) return error.BadPlan;
                var rows: [fwd.DECODE_ROWS]fwd.Row = undefined;
                for (0..n) |i| rows[i] = .{ .slot = @intCast(words[2 + 3 * i]), .pos = @intCast(words[3 + 3 * i]), .token = words[4 + 3 * i] };
                try e.model.graphed(e.s, rows[0..n], null);
            },
            .draft => {
                if (words.len != 2 + 3 * n or n > dflash2.MAX_STREAMS) return error.BadPlan;
                const dr = e.dr orelse return error.NoDrafter;
                var t: [dflash2.MAX_STREAMS]i64 = undefined;
                var p: [dflash2.MAX_STREAMS]u32 = undefined;
                var sl: [dflash2.MAX_STREAMS]u32 = undefined;
                var out: [dflash2.MAX_STREAMS * 16]i32 = undefined;
                for (0..n) |i| {
                    t[i] = words[2 + 3 * i];
                    p[i] = @intCast(words[3 + 3 * i]);
                    sl[i] = @intCast(words[4 + 3 * i]);
                }
                try dr.draft(e.s, t[0..n], p[0..n], sl[0..n], &out);
            },
            .prompt => {
                const pr = e.pr orelse return error.NoPromptPath;
                if (words.len != 4 + n or n > pr.cap) return error.BadPlan;
                try pr.chunk(e.s, @intCast(words[2]), @intCast(words[3]), words[4..]);
                try e.s.synchronize();
            },
            .cache => try e.copies.follow(e.s, words),
            .stop => return false,
            .hello => return error.LateHello,
        }
        return true;
    }
};

test "the hello carries the setup, and another build's is refused" {
    const su: Setup = .{ .context = 131072, .slots = 4, .drafts = true };
    try std.testing.expectEqual(su, try setupOf(&hello(su)));
    const plain: Setup = .{ .context = 65536, .slots = 2 };
    try std.testing.expectEqual(plain, try setupOf(&hello(plain)));
    try std.testing.expectError(error.OtherBuild, setupOf(&.{ @backingInt(Op.hello), 7, 65536, 2, 0, 1024, 512, 1, 0 }));
    try std.testing.expectError(error.NoHello, setupOf(&.{ 3, 0 }));
}
