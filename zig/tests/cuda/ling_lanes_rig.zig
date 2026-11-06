//! What the Ling lane checks share: requests through one lanes.Engine, joined streams, kept states, byte comparisons.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const check = @import("check.zig");
const ling = @import("ling_cuda");

pub const Engine = ling.engine.Engine;

/// A mark hook that keeps the state as the lane host's prompt cache does (native.snap_save).
pub const Keeper = struct {
    e: *Engine,
    kept: [8]*anyopaque = undefined,
    at: [8]u32 = undefined,
    n: usize = 0,

    pub fn hook(k: *Keeper) lanes.stream.MarkHook {
        return .{ .ptr = k, .at = keep };
    }

    fn keep(ptr: *anyopaque, s: *lanes.Stream, pos: u32) void {
        _ = s;
        const k: *Keeper = @ptrCast(@alignCast(ptr));
        k.kept[k.n] = ling.native.snap_save(k.e, null, pos) catch |err| {
            std.debug.print("FAIL keeping the state at {d}: {s}\n", .{ pos, @errorName(err) });
            return;
        };
        k.at[k.n] = pos;
        k.n += 1;
    }

    pub fn drop(k: *Keeper) void {
        for (k.kept[0..k.n]) |x| ling.native.snap_drop(k.e, x);
        k.n = 0;
    }
};

/// A cancel hook that answers yes from its `at`-th ask on.
pub const Cut = struct {
    at: u32,
    asked: u32 = 0,

    fn ask(ptr: *anyopaque) bool {
        const cut: *Cut = @ptrCast(@alignCast(ptr));
        cut.asked += 1;
        return cut.asked > cut.at;
    }
};

/// Drafts from a known reply (the plain run's): 2 to 15 a round, most rounds one of them wrong, to reject anywhere.
pub const Oracle = struct {
    prompt_len: usize,
    truth: []const u32,
    out: [16]u32 = undefined,
    rounds: u32 = 0,

    pub fn proposer(o: *Oracle) lanes.proposer.Proposer {
        return .{ .ptr = o, .vtable = &.{ .propose = propose, .last_match = lastMatch } };
    }

    fn propose(ptr: *anyopaque, context: []const u32, max_draft: i64) anyerror![]const u32 {
        const o: *Oracle = @ptrCast(@alignCast(ptr));
        o.rounds += 1;
        const k = context.len - o.prompt_len; // the reply so far, its last token pending: drafts follow it
        if (max_draft < 2 or k >= o.truth.len) return &.{};
        const n = @min(@as(usize, @intCast(max_draft)), o.truth.len - k, 2 + o.rounds % 14);
        if (n < 2) return &.{};
        @memcpy(o.out[0..n], o.truth[k..][0..n]);
        if (o.rounds % 4 != 0) o.out[(o.rounds * 5) % n] ^= 1; // a wrong token at a moving row
        return o.out[0..n];
    }

    fn lastMatch(_: *anyopaque) i64 {
        return 64;
    }
};

/// A concurrent case's stream: prompt `p` joining at round `at`; `filled`: its prompt a chunk at a time between rounds.
pub const Join = struct { p: usize, at: u32 = 0, sampled: bool = false, discard_at: ?u32 = null, filled: bool = false, drafts: bool = false };

pub const Cover = struct {
    gpa: std.mem.Allocator,
    a: std.mem.Allocator,
    le: *lanes.Engine,
    failed: bool = false,
    checks: usize = 0,

    pub const Ask = struct {
        prompt: []const u32,
        max_new: u32,
        eos: []const u32 = &.{},
        drafts: bool = true,
        sampling: ?lanes.Sampling = null,
        truth: []const u32 = &.{}, // drafts from this reply (Oracle); empty: none
        cancel: ?*Cut = null,
        reuse: lanes.stream.Reuse = .{},
    };
    pub const Got = struct { ids: []const u32, reason: lanes.stream.Reason, rounds: u64, cached: u32, drafted: u64, accepted: u64 };

    /// One request through the shared lanes.Engine, as the server's lane host runs it.
    pub fn run(c: *Cover, ask: Ask) !Got {
        var o: Oracle = .{ .prompt_len = ask.prompt.len, .truth = ask.truth };
        var s = try lanes.Stream.init(c.gpa, .{ .id = "cover", .prompt = ask.prompt, .max_new = ask.max_new, .eos = ask.eos, .drafts = ask.drafts, .sampling = ask.sampling, .proposer = if (ask.truth.len > 0) o.proposer() else null, .cancel_check = if (ask.cancel) |cut| .{ .ptr = cut, .check = Cut.ask } else null, .reuse = ask.reuse });
        defer s.deinit(c.gpa);
        try c.le.addStream(&s);
        while (c.le.activeCount() > 0) try c.le.step();
        return .{ .ids = try c.a.dupe(u32, s.emitted()), .reason = s.reason, .rounds = s.rounds, .cached = s.cached, .drafted = s.drafted, .accepted = s.accepted };
    }

    pub const Together = struct { ids: []const []const u32, most: usize, shared: u32, between: u32, drafted: u64 };

    /// The streams of `joins` through the rounds together, each joining at its round; how many were live at once.
    pub fn together(c: *Cover, joins: []const Join, ps: []const []const u32, truths: []const []const u32, seeds: []const lanes.Sampling, max_new: u32) !Together {
        const ss = try c.a.alloc(lanes.Stream, joins.len);
        const looks = try c.a.alloc(Oracle, joins.len);
        const made = try c.a.alloc(bool, joins.len);
        @memset(made, false);
        defer for (ss, made) |*s, m| if (m) s.deinit(c.gpa);
        var most: usize = 0;
        var shared: u32 = 0;
        var between: u32 = 0;
        var round: u32 = 0;
        var filling: std.ArrayList(*lanes.Stream) = .empty; // the lane host's fillOne: the oldest's next chunk a round
        defer filling.deinit(c.gpa);
        var began = false;
        while (true) : (round += 1) {
            for (joins, ss, looks, made) |j, *s, *l, *m| if (!m.* and j.at <= round) {
                l.* = .{ .prompt_len = ps[j.p].len, .truth = truths[j.p] };
                s.* = try lanes.Stream.init(c.gpa, .{ .id = "joined", .prompt = ps[j.p], .max_new = max_new, .sampling = if (j.sampled) seeds[j.p % seeds.len] else null, .proposer = if (j.drafts) l.proposer() else null });
                m.* = true;
                if (j.filled) try filling.append(c.gpa, s) else try c.le.addStream(s);
            };
            if (filling.items.len > 0) {
                if (try c.le.fillStream(filling.items[0], !began)) {
                    _ = filling.orderedRemove(0);
                    began = false;
                } else began = true;
            }
            for (joins, ss, made) |j, *s, m| if (m and j.discard_at == round and !s.finished) c.le.discard(s);
            const live = c.le.activeCount();
            most = @max(most, live);
            shared += @intFromBool(live >= 2);
            if (live == 0) {
                if (std.mem.indexOfScalar(bool, made, false) == null and filling.items.len == 0) break;
                continue;
            }
            between += @intFromBool(began);
            try c.le.step();
        }
        const ids = try c.a.alloc([]const u32, joins.len);
        var drafted: u64 = 0;
        for (ss, ids) |*s, *i| {
            i.* = try c.a.dupe(u32, s.emitted());
            drafted += s.drafted;
        }
        return .{ .ids = ids, .most = most, .shared = shared, .between = between, .drafted = drafted };
    }

    /// A drafted stream over `prompt` (drafts from `truth`), discarded after `rounds`; false if it ended first.
    pub fn discardAfter(c: *Cover, prompt: []const u32, truth: []const u32, rounds: u32) !bool {
        var o: Oracle = .{ .prompt_len = prompt.len, .truth = truth };
        var s = try lanes.Stream.init(c.gpa, .{ .id = "discarded", .prompt = prompt, .max_new = 512, .proposer = o.proposer() });
        defer s.deinit(c.gpa);
        try c.le.addStream(&s);
        var n: u32 = 0;
        while (c.le.activeCount() > 0 and n < rounds) : (n += 1) try c.le.step();
        if (c.le.activeCount() == 0) return false;
        c.le.discard(&s);
        return true;
    }

    pub fn same(c: *Cover, what: []const u8, got: Got, want: []const u32, reason: lanes.stream.Reason) !void {
        c.checks += 1;
        const n = @min(got.ids.len, want.len);
        const first = for (got.ids[0..n], want[0..n], 0..) |x, y, i| {
            if (x != y) break i;
        } else if (got.ids.len != want.len) n else null;
        if (first == null and got.reason == reason) {
            check.pass("{s}: {d} ids equal, finished by {s} ({d} rounds, {d} of {d} drafts kept)", .{ what, want.len, reason.name(), got.rounds, got.accepted, got.drafted });
            return;
        }
        c.failed = true;
        if (first) |i| {
            std.debug.print("FAIL {s}: token {d} of {d} is {d}, want {d} (emitted {d}, {s})\n", .{ what, i, want.len, if (i < got.ids.len) got.ids[i] else 0, if (i < want.len) want[i] else 0, got.ids.len, got.reason.name() });
        } else std.debug.print("FAIL {s}: finished by {s}, want {s}\n", .{ what, got.reason.name(), reason.name() });
    }

    pub fn expect(c: *Cover, ok: bool, comptime fmt: []const u8, args: anytype) !void {
        c.checks += 1;
        if (ok) return check.pass(fmt, args);
        c.failed = true;
        std.debug.print("FAIL " ++ fmt ++ "\n", args);
    }
};

pub fn back(e: *Engine, a: std.mem.Allocator, ptr: u64, len: usize) ![]u8 {
    const out = try a.alloc(u8, len);
    try e.stream.synchronize();
    const b: cuda.DeviceBuffer = .{ .d = e.d, .ptr = ptr, .len = len };
    try b.download(0, out);
    return out;
}

/// Slots `x` and `y` hold the same committed sequence: position, KDA states and conv windows, MLA rows below it.
pub fn sameSeq(c: *Cover, e: *Engine, what: []const u8, x: u32, y: u32) !void {
    const p = &e.seqs.items[x].?.st;
    const q = &e.seqs.items[y].?.st;
    var bad: ?[]const u8 = null;
    if (p.pos != q.pos) bad = "positions";
    for (0..p.kda_layers) |li| {
        if (!std.mem.eql(u8, try back(e, c.a, p.recAt(p.cur[li], li), p.rec_layer), try back(e, c.a, q.recAt(q.cur[li], li), q.rec_layer))) bad = "a KDA state";
        if (!std.mem.eql(u8, try back(e, c.a, p.convAt(li), 3 * p.convWidth() * 2), try back(e, c.a, q.convAt(li), 3 * q.convWidth() * 2))) bad = "a conv window";
    }
    for (0..p.mla_layers) |mi| if (!std.mem.eql(u8, try back(e, c.a, p.cacheAt(mi), p.pos * ling.forward.cache_width * 2), try back(e, c.a, q.cacheAt(mi), p.pos * ling.forward.cache_width * 2))) {
        bad = "MLA cache rows";
    };
    if (e.seqs.items[x].?.hd) |*ph| {
        const qh = &e.seqs.items[y].?.hd.?;
        if (ph.pos != qh.pos) bad = "MTP positions";
        if (!std.mem.eql(u8, try back(e, c.a, ph.cacheAt(0), ph.pos * ling.forward.cache_width * 2), try back(e, c.a, qh.cacheAt(0), ph.pos * ling.forward.cache_width * 2))) bad = "MTP cache rows";
    }
    try c.expect(bad == null, "{s}: slots {d} and {d} at {d}, byte for byte{s}{s}", .{ what, x, y, p.pos, if (bad != null) ", but differ in " else "", bad orelse "" });
}
