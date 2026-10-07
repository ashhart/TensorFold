//! A GPU round's options, its result and the host's hooks between rounds (gpu_round.zig runs it).
const st = @import("state.zig");

pub const Options = struct {
    depth: usize = 4, // the head's levels a round at most (a verify takes depth + 1 rows at most)
    fixed: bool = false, // every round drafts `depth` levels (else the stream's records move it from 2 up)
    ahead: usize = 2, // rounds queued before the host reads the oldest
    power: f32 = 2, // the running top-1 product's power the stop weighs
    stop: bool = true, // false: every level verified
    split: bool = false, // the head's part in its own command buffer (GPU time by part)
    bar: ?f32 = null, // every level's bar fixed (probes: 2 stops at level 0), else from the stream's rate
    profile: usize = 0, // rounds after the reply with every dispatch alone: GPU ms by kernel
    head_bar: bool = true, // a level's bar also prices the head level passing it drafts
    trees: bool = true, // windows verify as trees (GPU-written tables): branches beside the head's drafts or a copy
    siblings: bool = false, // leaves: the head's likeliest second choice past a row's price, its guess beside a copy
    sib_bar: ?f64 = null, // a fixed bar for second choices (0: every round has one), else a row's price
    guess_match: u32 = 24, // a copy whose match is shorter also takes the head's guess beside its first row
    level_bench: bool = false, // after the reply: a live head level's GPU time (chains of 1 and 8 live levels, medians)
    copy: bool = true, // copy lanes: a window copied after the context's matched suffix when its rows are priced
    copy_max: usize = 30, // copy rows a window takes at most
    min_match: u32 = 8, // context tokens a copy's match needs (shorter ones are coincidental phrases)
    tail: bool = true, // the head's drafts continued by a copy when the context with them matches
};

pub const Result = struct {
    rounds: u64 = 0,
    accepted: u64 = 0,
    by_rows: [st.max_rows + 1]u32 = @splat(0), // rounds by the rows their verify took
    by_cap: [st.max_levels + 1]u32 = @splat(0), // rounds by the head depth the rule chose for them
    head_kept: [17][17]u32 = @splat(@splat(0)), // head windows by [verify rows][tokens kept] (16+ in the last)
    copied: u64 = 0, // rounds whose window was a copy
    siblings: u64 = 0, // rounds with the head's second choice beside its first draft, and those that kept it
    siblings_kept: u64 = 0,
    guesses: u64 = 0, // copies with the head's guess beside their first row, and those that kept it
    guesses_kept: u64 = 0,
    prefill_s: f64 = 0,
    decode_s: f64 = 0,
    paused: bool = false, // the stream went to the lane core, its cache as the last round read left it
};

/// A server's engine thread between rounds: told when tokens land, asked whether to hand the stream to the lane core.
pub const Hooks = struct {
    ctx: ?*anyopaque = null,
    committed: ?*const fn (ctx: *anyopaque) void = null,
    yield: ?*const fn (ctx: *anyopaque) bool = null,

    pub fn note(k: Hooks) void {
        if (k.committed) |f| f(k.ctx.?);
    }

    pub fn asked(k: Hooks) bool {
        return if (k.yield) |f| f(k.ctx.?) else false;
    }
};
