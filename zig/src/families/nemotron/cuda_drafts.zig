//! decode.draft_decode: MTP chains (or copied chains) verified in one window a round; kept drafts equal serial tokens.

const std = @import("std");
const core = @import("core");
const Engine = @import("cuda_engine.zig").Engine;
const mtp = @import("cuda_mtp.zig");
const Head = mtp.Head;
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

/// `count` tokens in `out` (its first the prompt's pending token), drafting from the prompt's last hidden row.
pub fn decode(gpa: std.mem.Allocator, e: *Engine, h: *Head, rule: *DepthRule, prompt: []const u32, out: *std.ArrayList(u32), count: usize, stop_eos: bool, last_hidden: u64) !Stats {
    var st: Stats = .{};
    var index = try core.CopyIndex.init(gpa, prompt);
    defer index.deinit();
    try index.extend(out.items);
    try e.ops().copy(e.b.hidden, last_hidden, e.c.hidden * 2);
    try e.ops().fill32(e.b.sampled, out.items[0], 1);
    var cbuf: [mtp.max_chain]u32 = undefined;
    var copied = index.chain(&cbuf);
    try queue(h, 1, copied.len > 0);
    var ids: [mtp.max_chain + 1]u32 = undefined;
    while (out.items.len < count and !(stop_eos and e.c.isEos(out.items[out.items.len - 1]))) {
        const last = out.items[out.items.len - 1];
        var proposal: []const u32 = undefined;
        if (copied.len > 0) {
            ids[0] = last;
            @memcpy(ids[1..][0..copied.len], copied);
            try e.verify(ids[0 .. 1 + copied.len], @intCast(1 + copied.len), null);
            proposal = copied;
        } else {
            const n = try depth(h, rule);
            try e.verify(&.{last}, 1 + n, null);
            proposal = h.drafts()[0..n];
        }
        const sampled = try e.tokens();
        var accepted: usize = 0;
        while (accepted < proposal.len and proposal[accepted] == sampled[accepted]) accepted += 1;
        if (stop_eos) for (sampled[0..accepted], 0..) |t, j| if (e.c.isEos(t)) {
            accepted = @intCast(j);
            break;
        };
        accepted = @min(accepted, count - out.items.len - 1);
        const keep = accepted + 1;
        try e.commit(keep);
        rule.done(keep, 1 + proposal.len, if (copied.len > 0) 0 else h.levels);
        try out.appendSlice(gpa, sampled[0..keep]);
        try index.extend(sampled[0..keep]);
        st.rounds += 1;
        st.drafted += @intCast(proposal.len);
        st.accepted += accepted;
        if (out.items.len < count and !(stop_eos and e.c.isEos(out.items[out.items.len - 1]))) {
            copied = index.chain(&cbuf);
            try queue(h, keep, copied.len > 0);
        }
    }
    return st;
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
        return .{ .head = h, .rule = DepthRule.init(c, mtp.max_chain, null, 2.0) };
    }

    pub fn deinit(d: *Drafter) void {
        d.head.deinit();
    }
};
