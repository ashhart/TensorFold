//! Session rules over several live slots: pool admission (`admit`), the RAM tier's `Measure`, `dsRows` / `stashRows`.

const std = @import("std");
const sessions = @import("sessions");

/// Positions past prompt + max_tokens a request may write (verify / draft windows)
pub const slack: u64 = 64;

pub const Error = error{RequestTooLarge};

/// A request's admission: `need` pages for its slot, `want` of them new; `wait`: no fit even with every RAM entry gone.
pub const Plan = struct { need: u32, want: u32, wait: bool = false };

/// Pool admission of `prompt` + `max_new` on a `capacity` slot: pages, and the leader-ordered `spills` to evict first.
pub fn admit(store: *sessions.Store, hit: ?sessions.store.Hit, prompt: u64, max_new: u64, capacity: u64, held: []const u32, extra: u32, spills: *std.ArrayList(u32)) !Plan {
    spills.clearRetainingCapacity();
    const pool = store.pool;
    const need = pool.needPages(prompt, max_new, capacity, slack);
    if (need > pool.npages) return error.RequestTooLarge;
    // a RAM hit's full pages are shared, not new
    const shared: u32 = if (hit) |h| (if (h.ram) @intCast(h.pos / pool.page) else 0) else 0;
    const want = need - @min(need, shared) + extra;
    if (pool.available() >= want) return .{ .need = need, .want = want };
    if (!try store.planSpill(want, held, spills)) {
        spills.clearRetainingCapacity();
        return .{ .need = need, .want = want, .wait = true };
    }
    return .{ .need = need, .want = want };
}

/// The snapshot point (last 16-grid point before n) if past `cached` and >= 64 tokens; same rule as ced.savePoint.
pub fn savePoint(n: u64, cached: u64) ?u64 {
    const s: u64 = if (n > 0) (n - 1) / 16 * 16 else 0;
    return if (s > cached and s >= 64) s else null;
}

/// An SWA / DSpark ring row as the snapshot holds it (Records raw: 576 values + 8 scales)
pub const row_bytes: u64 = 584;
/// Engram's lookback ids (int64)
pub const lookback: u64 = 3;
/// the stash rows a slot keeps (REPLAY - 1)
pub const keep: u64 = 127;

/// What a slot's snapshot holds, as sizes (the RAM tier's measure of an entry).
pub const Measure = struct {
    /// SWA rings in the snapshot: replay mode the layers below the decoder, full mode every one
    rings: u64,
    /// the SWA window (cfg.sliding_window: the rings' rows and the drafter's)
    window: u64,
    /// ratio-2 kv sources (their float32 [2 x head_dim] carries)
    carries: u64,
    head_dim: u64,
    hidden: u64,
    hc: u64,
    /// the drafter's blocks (0: no drafter, nothing of DSpark in the snapshot)
    ds_blocks: u64 = 0,
    /// CED replay: the stash's rows ride along
    replay: bool,

    /// A replay stash row: ced_x [hc x D], ced_out [D] bf16, ced_pre / ced_post [4], ced_comb [16] f32, its id
    fn stashRow(m: Measure) u64 {
        return 2 * m.hc * m.hidden + 2 * m.hidden + 4 * (4 + 4 + 16) + 8;
    }

    /// the entry's bytes at `pos`, with `stash_rows` (`stashRows`) and `ds_rows` (`dsRows`)
    pub fn bytes(m: Measure, pos: u64, stash_rows: u64, ds_rows: u64) u64 {
        var n: u64 = 0;
        // swa_values / swa_scales: [rings, rows, 576 | 8] uint8 (none below one row)
        n += m.rings * @min(pos, m.window) * row_bytes;
        // the stash (replay: ced_lo int64, then its rows)
        if (m.replay) n += 8 + stash_rows * m.stashRow();
        // carry: [carries, 2 x head_dim] float32
        if (m.carries > 0) n += m.carries * 2 * m.head_dim * 4;
        // lookback [3] and rings (the layer ids) int64
        n += 8 * lookback + 8 * m.rings;
        // dspark_values / dspark_scales: [blocks, rows, 576 | 8] uint8
        if (m.ds_blocks > 0) n += m.ds_blocks * ds_rows * row_bytes;
        return n;
    }
};

/// The drafter snapshot's rows: [max(0, pos - window, valid), pos); `valid` 0 after a reset, pos - n after a restore.
pub fn dsRows(pos: u64, valid: u64, window: u64) u64 {
    return pos -| @max(pos -| window, valid);
}

/// The CED stash's rows at `pos` for a stash started at `from`: the last KEEP of them.
pub fn stashRows(pos: u64, from: u64) u64 {
    return pos -| @max(from, pos -| keep);
}

test "the measure is the reference's Bounded.nbytes (golden.py: 96 and >= 128 positions)" {
    const m: Measure = .{ .rings = 20, .window = 128, .carries = 3, .head_dim = 512, .hidden = 5120, .hc = 4, .ds_blocks = 3, .replay = true };
    try std.testing.expectEqual(@as(u64, 6_227_136), m.bytes(96, stashRows(96, 0), dsRows(96, 0, 128)));
    try std.testing.expectEqual(@as(u64, 8_247_384), m.bytes(1488, stashRows(1488, 0), dsRows(1488, 0, 128)));
    // a restored slot: the drafter's context from the restored rows on, the stash from its restored first row
    try std.testing.expectEqual(@as(u64, 128), dsRows(1696, 1488 - 128, 128));
    try std.testing.expectEqual(@as(u64, 0), dsRows(1488, 1488, 128));
    try std.testing.expectEqual(@as(u64, 40), dsRows(1528, 1488, 128));
    try std.testing.expectEqual(@as(u64, 127), stashRows(1696, 1488 - 127));
    try std.testing.expectEqual(@as(u64, 10), stashRows(10, 0));
}
