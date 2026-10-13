//! decode.draft_decode: MTP chains (or copied chains) verified in one window a round; kept drafts equal serial tokens.

const std = @import("std");
const core = @import("core");
const lanes = @import("lanes");
const Engine = @import("cuda_engine.zig").Engine;
const mtp = @import("cuda_mtp.zig");
const Head = mtp.Head;
const Mask = @import("cuda_engine.zig").Mask;
const DepthRule = core.draft_depth.DepthRule;

pub const Stats = struct { rounds: usize = 0, drafted: usize = 0, accepted: usize = 0 };

/// Start a round's head work: an absorb for a copied chain, else level 1 (the rule reads the later levels).
fn queue(h: *Head, keep: usize, copied: bool) !void {
    try h.begin(keep);
    try h.launch(if (copied) 0 else 1);
}

/// The drafts this round verifies: the rule reads each level's confidence before drafting the next.
fn depth(h: *Head, rule: *const DepthRule) !usize {
    var run: f64 = 1.0;
    var n: usize = 0;
    var j: usize = 1;
    while (j <= mtp.max_chain) : (j += 1) {
        if (j > 1) try h.launch(j);
        run *= try h.confidence(j);
        if (!rule.keep(j, run)) break;
        n = j;
        if (!rule.more(j, run)) break;
    }
    return n;
}

/// A grammar's hold on a window: of `ids` (row 0 pending) the rows it keeps, then its own drafts, written in `ids`.
pub const Gate = struct {
    ptr: *anyopaque,
    window: *const fn (ptr: *anyopaque, ids: []u32, planned: usize) anyerror!Gated,
};

pub const Gated = struct { rows: usize, mask: Mask };

/// A drafted decode's rounds over a token context: each verifies a copied chain or the head's chain in one window.
pub const Rounds = struct {
    e: *Engine,
    h: *Head,
    rule: *DepthRule,
    index: core.CopyIndex,
    cbuf: [mtp.max_chain]u32 = undefined,
    ids: [mtp.max_chain + 1]u32 = undefined,
    copied: []const u32 = &.{},
    recall: ?*lanes.recall.Reader = null, // the stream's reads of Recall's record: its copy when the context has none
    recalled: bool = false, // the copy came from Recall
    proposal: []const u32 = &.{},
    copy_round: bool = false,
    st: Stats = .{},

    /// The head reads the prompt's last hidden row and final output token.
    pub fn init(gpa: std.mem.Allocator, e: *Engine, h: *Head, rule: *DepthRule, prompt: []const u32, out: []const u32, last_hidden: ?u64, recall: ?*lanes.recall.Reader) !Rounds {
        var r: Rounds = .{ .e = e, .h = h, .rule = rule, .index = try core.CopyIndex.init(gpa, prompt), .recall = recall };
        errdefer r.index.deinit();
        try r.index.extend(out);
        if (last_hidden) |x| try e.ops().copy(e.b.hidden, x, e.c.hidden * 2);
        try e.ops().fill32(e.b.sampled, out[out.len - 1], 1);
        r.copied = r.copies();
        try queue(h, 1, r.copied.len > 0);
        return r;
    }

    /// Rounds over a stream the lane core drafted last: the head's kept rows absorbed again, level 1 drafted again.
    pub fn resumed(gpa: std.mem.Allocator, e: *Engine, h: *Head, rule: *DepthRule, prompt: []const u32, out: []const u32, kept: usize, recall: ?*lanes.recall.Reader) !Rounds {
        var r: Rounds = .{ .e = e, .h = h, .rule = rule, .index = try core.CopyIndex.init(gpa, prompt), .recall = recall };
        errdefer r.index.deinit();
        try r.index.extend(out);
        r.copied = r.copies();
        try h.again(kept);
        try h.launch(if (r.copied.len > 0) 0 else 1);
        return r;
    }

    pub fn deinit(r: *Rounds) void {
        r.index.deinit();
    }

    /// The context's own copied chain, else Recall's continuation when it is expected to pay (in `cbuf`).
    fn copies(r: *Rounds) []const u32 {
        r.recalled = false;
        const own = r.index.chain(&r.cbuf);
        const x = r.recall orelse return own;
        if (own.len > 0) return own;
        const got = x.propose(r.index.ctx.items, r.cbuf.len, r.st.rounds);
        if (got.len < 2) return &.{};
        @memcpy(r.cbuf[0..got.len], got);
        r.recalled = true;
        return r.cbuf[0..got.len];
    }

    /// One window after `last`: the copied chain or the head's drafts, verified; its drawn tokens and drafts accepted.
    pub fn verify(r: *Rounds, last: u32, gate: ?Gate) !struct { sampled: []const u32, accepted: usize } {
        const e = r.e;
        r.copy_round = r.copied.len > 0;
        if (gate) |g| { // the planned chain as host tokens, cut and grown by the grammar, its rows masked
            const planned = if (r.copy_round) r.copied else r.h.drafts()[0..try depth(r.h, r.rule)];
            r.ids[0] = last;
            @memcpy(r.ids[1..][0..planned.len], planned);
            const w = try g.window(g.ptr, &r.ids, 1 + planned.len);
            try e.verifyMasked(r.ids[0..w.rows], w.rows, null, w.mask);
            r.proposal = r.ids[1..w.rows];
        } else if (r.copy_round) {
            r.ids[0] = last;
            @memcpy(r.ids[1..][0..r.copied.len], r.copied);
            try e.verify(r.ids[0 .. 1 + r.copied.len], 1 + r.copied.len, null);
            r.proposal = r.copied;
        } else {
            const n = try depth(r.h, r.rule);
            try e.verify(&.{last}, 1 + n, null);
            r.proposal = r.h.drafts()[0..n];
        }
        const sampled = try e.tokens();
        var accepted: usize = 0;
        while (accepted < r.proposal.len and r.proposal[accepted] == sampled[accepted]) accepted += 1;
        return .{ .sampled = sampled, .accepted = accepted };
    }

    /// Keep leading rows and their sampled tokens, then queue head work when more tokens are needed.
    pub fn keep(r: *Rounds, sampled: []const u32, n: usize, more: bool) !void {
        try r.e.commit(n);
        r.rule.done(n, 1 + r.proposal.len, if (r.copy_round) 0 else r.h.levels);
        if (r.recall) |x| { // Recall's copy is judged apart; the head's rounds set the bar its copies must beat
            if (r.copy_round and r.recalled) x.observe(@intCast(r.proposal.len), @intCast(n -| 1), r.st.rounds) else if (!r.copy_round) x.noteHead(n);
        }
        try r.index.extend(sampled[0..n]);
        r.st.rounds += 1;
        r.st.drafted += r.proposal.len;
        r.st.accepted += n - 1;
        if (!more) return;
        r.copied = r.copies();
        try queue(r.h, n, r.copied.len > 0);
    }
};

/// `count` tokens in `out` (its first the prompt's pending token), drafting from the prompt's last hidden row.
pub fn decode(gpa: std.mem.Allocator, e: *Engine, h: *Head, rule: *DepthRule, prompt: []const u32, out: *std.ArrayList(u32), count: usize, stop_eos: bool, last_hidden: u64) !Stats {
    var r = try Rounds.init(gpa, e, h, rule, prompt, out.items, last_hidden, null);
    defer r.deinit();
    while (out.items.len < count and !(stop_eos and e.c.isEos(out.items[out.items.len - 1]))) {
        const v = try r.verify(out.items[out.items.len - 1], null);
        var accepted = v.accepted;
        if (stop_eos) for (v.sampled[0..accepted], 0..) |t, j| if (e.c.isEos(t)) {
            accepted = @intCast(j);
            break;
        };
        accepted = @min(accepted, count - out.items.len - 1);
        try out.appendSlice(gpa, v.sampled[0 .. accepted + 1]);
        try r.keep(v.sampled, accepted + 1, out.items.len < count and !(stop_eos and e.c.isEos(out.items[out.items.len - 1])));
    }
    return r.st;
}

/// The MTP head and the measured-cost depth rule (app._rules: greedy calibration power 2, every draft depth).
pub const Drafter = struct {
    head: *Head,
    rule: DepthRule,

    /// `costs`: window ms at 1..16 rows then a level's ms, as another run measured them; null measures them here.
    pub fn init(gpa: std.mem.Allocator, io: std.Io, e: *Engine, model_dir: []const u8, graphs: bool, costs: ?[]const f64) !Drafter {
        const h = try Head.init(e);
        errdefer h.deinit();
        if (graphs) try h.capture();
        var c: core.draft_depth.Costs = .{};
        if (costs) |given| {
            if (given.len != 17) return error.CostsNeedSeventeenValues;
            c.rows = 16;
            @memcpy(c.verify[1..17], given[0..16]);
            c.level = given[16];
        } else c = try @import("cuda_costs.zig").measure(gpa, io, e, h, model_dir);
        return .{ .head = h, .rule = depthRule(c) };
    }

    pub fn deinit(d: *Drafter) void {
        d.head.deinit();
    }
};

/// The depth rule `run` and a lone served stream draft by: greedy calibration power 2, every draft depth.
pub fn depthRule(c: core.draft_depth.Costs) DepthRule {
    return DepthRule.init(c, mtp.max_chain, null, 2.0);
}
