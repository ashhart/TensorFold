//! MiMo-V2's engine on one rank: the model, its drafter and prompt path, and the steps rank 0 sends rank 1.

const std = @import("std");
const cuda = @import("cuda");
const tp = @import("tp");
const load_ = @import("load.zig");
const ops_ = @import("ops.zig");
const exl3 = @import("exl3.zig");
const fwd = @import("forward.zig");
const dflash = @import("dflash.zig");
const prompt_mod = @import("prompt.zig");
const kept = @import("kept.zig");

/// The plan link's words: one GPU step each, executed by every rank in order.
pub const Op = enum(i64) { hello = 0x31A0, forward = 1, draft = 2, stop = 3, prompt = 4, cache = 5 };

/// What rank 0 announces first so every rank builds the same model.
pub const Setup = struct { context: usize, slots: usize, drafts: bool };

/// The hello words: [hello, 3, context, slots, drafts, 0, 0, 0].
pub fn hello(su: Setup) [8]i64 {
    return .{ @backingInt(Op.hello), 3, @intCast(su.context), @intCast(su.slots), @intFromBool(su.drafts), 0, 0, 0 };
}

/// The setup a follower reads from rank 0's hello.
pub fn setupOf(words: []const i64) !Setup {
    if (words.len < 5 or words[0] != @backingInt(Op.hello)) return error.NoHello;
    return .{ .context = @intCast(words[2]), .slots = @intCast(words[3]), .drafts = words[4] != 0 };
}

/// The model, its kernels and the TP session: what rank 0 serves and rank 1 follows.
pub const Engine = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    w: load_.Weights,
    k3: exl3.Kernels,
    ops: ops_.Ops,
    model: fwd.Model,
    dr: ?*dflash.Drafter = null,
    pr: *prompt_mod.Prompt = undefined, // prompt chunks of up to fwd.PROMPT_CHUNK rows
    copies: kept.Copies = undefined, // kept prompt states' windows (kept.zig), on every rank
    session: ?*tp.Session = null,
    s: cuda.Stream,
    logits: []f32, // [128, V] host
    ids: [fwd.DECODE_ROWS]i32 = undefined, // device argmax of each row (Want.ids)
    words: std.ArrayList(i64) = .empty,

    /// Load everything on this rank. `session`: the open TP session (world > 1), null on one GPU.
    pub fn init(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, dir: []const u8, su: Setup, session: ?*tp.Session) !*Engine {
        const e = try gpa.create(Engine);
        errdefer gpa.destroy(e);
        e.* = .{ .gpa = gpa, .d = d, .w = undefined, .k3 = undefined, .ops = undefined, .model = undefined, .s = undefined, .logits = &.{}, .session = session };
        const rank: usize = if (session) |ss| ss.cfg.rank else 0;
        const world: usize = if (session) |ss| ss.cfg.world else 1;
        e.s = try cuda.Stream.init(d, false);
        errdefer e.s.deinit();
        e.w = try load_.load(gpa, io, d, dir, rank, world, rank == 0);
        errdefer e.w.deinit();
        e.k3 = try exl3.Kernels.load(d);
        errdefer e.k3.deinit();
        e.ops = try ops_.Ops.load(d);
        errdefer e.ops.unload();
        const comm = if (session) |ss| ss.comm() else null;
        e.model = try fwd.Model.init(gpa, d, &e.w, &e.ops, &e.k3, comm, su.context, fwd.DECODE_ROWS, su.slots, e.s);
        errdefer e.model.deinit();
        if (su.drafts) {
            const ddir = try std.fmt.allocPrint(gpa, "{s}/dflash", .{dir});
            defer gpa.free(ddir);
            e.dr = try dflash.Drafter.init(gpa, io, &e.model, ddir, e.s);
            e.model.dr = e.dr;
        }
        errdefer if (e.dr) |dr| dr.deinit();
        e.pr = try prompt_mod.Prompt.init(&e.model, gpa, fwd.PROMPT_CHUNK, e.s);
        errdefer e.pr.deinit();
        e.copies = try kept.Copies.init(gpa, &e.model, e.dr);
        errdefer e.copies.deinit();
        if (rank == 0) std.log.info("mimo prompt path: {d}-row chunks, {d:.2} GiB scratch", .{ fwd.PROMPT_CHUNK, @as(f64, @floatFromInt(e.pr.bytes())) / (1 << 30) });
        e.logits = try gpa.alloc(f32, fwd.DECODE_ROWS * e.w.full.vocab);
        return e;
    }

    pub fn deinit(e: *Engine) void {
        const gpa = e.gpa;
        e.words.deinit(gpa);
        gpa.free(e.logits);
        e.copies.deinit();
        e.pr.deinit();
        if (e.dr) |dr| dr.deinit();
        e.model.deinit();
        e.ops.unload();
        e.k3.deinit();
        e.w.deinit();
        e.s.deinit();
        gpa.destroy(e);
    }

    fn send(e: *Engine, words: []const i64) !void {
        if (e.session) |ss| try ss.plan.send(words);
    }

    pub const Want = enum { none, ids, logits };

    /// A forward window on every rank; on rank 0 `ids` leaves each row's argmax in `ids`, `logits` the rows [n, V].
    pub fn forward(e: *Engine, rows: []const fwd.Row, want: Want) !void {
        e.words.clearRetainingCapacity();
        try e.words.appendSlice(e.gpa, &.{ @backingInt(Op.forward), @intCast(rows.len) });
        for (rows) |r| try e.words.appendSlice(e.gpa, &.{ r.slot, r.pos, r.token });
        try e.send(e.words.items);
        try e.model.graphed(e.s, rows, if (want == .logits) e.logits else null);
        if (want == .ids) try e.model.argmaxTo(e.s, rows.len, e.ids[0..rows.len]);
    }

    /// A prompt chunk on every rank: `want` reads its last row (ids[0] or logits row 0).
    pub fn promptChunk(e: *Engine, slot: u32, p0: usize, tokens: []const u32, want: Want) !void {
        e.words.clearRetainingCapacity();
        try e.words.appendSlice(e.gpa, &.{ @backingInt(Op.prompt), @intCast(tokens.len), slot, @intCast(p0) });
        for (tokens) |t| try e.words.append(e.gpa, t);
        try e.send(e.words.items);
        try e.pr.chunk(e.s, slot, p0, e.words.items[4..], null);
        switch (want) {
            .logits => try e.model.logitsTo(e.s, 1, e.logits[0..e.w.full.vocab]),
            .ids => try e.model.argmaxTo(e.s, 1, e.ids[0..1]),
            .none => try e.s.synchronize(),
        }
    }

    /// A draft pass on every rank: drafts [S, block - 1] into `out`.
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
            .save => try e.copies.save(e.s, id, slot, at),
            .restore => try e.copies.restore(e.s, id, slot, at),
            .copy => try e.copies.copyPrefix(e.s, slot, slot2, at),
            .drop => e.copies.drop(id),
        }
    }

    /// One step from the plan link, on a follower; false at stop.
    pub fn follow(e: *Engine, words: []const i64) !bool {
        if (words.len < 2) return error.ShortPlan;
        const op: Op = @fromBackingInt(@intCast(words[0]));
        const n: usize = @intCast(words[1]);
        if (op == .cache) {
            try e.copies.follow(e.s, words);
            return true;
        }
        if (op == .prompt) {
            if (words.len != 4 + n) return error.BadPlan;
            try e.pr.chunk(e.s, @intCast(words[2]), @intCast(words[3]), words[4..], null);
            try e.s.synchronize();
            return true;
        }
        if (words.len != 2 + 3 * n) return error.BadPlan;
        switch (op) {
            .forward => {
                var rows: [fwd.DECODE_ROWS]fwd.Row = undefined;
                if (n > rows.len) return error.BadPlan;
                for (0..n) |i| rows[i] = .{ .slot = @intCast(words[2 + 3 * i]), .pos = @intCast(words[3 + 3 * i]), .token = words[4 + 3 * i] };
                try e.model.graphed(e.s, rows[0..n], null);
            },
            .draft => {
                var t: [16]i64 = undefined;
                var p: [16]u32 = undefined;
                var sl: [16]u32 = undefined;
                var out: [16 * 16]i32 = undefined;
                if (n > t.len) return error.BadPlan;
                for (0..n) |i| {
                    t[i] = words[2 + 3 * i];
                    p[i] = @intCast(words[3 + 3 * i]);
                    sl[i] = @intCast(words[4 + 3 * i]);
                }
                try e.dr.?.draft(e.s, t[0..n], p[0..n], sl[0..n], &out);
            },
            .stop => return false,
            .hello => return error.LateHello,
            .prompt, .cache => unreachable,
        }
        return true;
    }
};

test "the hello carries the setup" {
    const su: Setup = .{ .context = 608256, .slots = 4, .drafts = true };
    try std.testing.expectEqual(su, try setupOf(&hello(su)));
    try std.testing.expectError(error.NoHello, setupOf(&.{ 3, 0 }));
}
