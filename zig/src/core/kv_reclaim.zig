//! A pool's reclaim chain: the stores on one load's pool, asked in the order they joined until enough pages are free.
const std = @import("std");

/// One store's eviction: free its idle pages until `want` are available (true: they are).
pub const Reclaim = struct {
    ctx: *anyopaque,
    call: *const fn (ctx: *anyopaque, want: usize) bool,
};

/// Stores one pool asks at most: a family's prompt store today, room for a drafter's and a second family's (P8).
pub const most = 4;

/// The stores in the order they joined; a promise or a hold asks them in that order (kv_pool.zig).
pub const Chain = struct {
    at: [most]Reclaim = undefined,
    n: u8 = 0,

    pub fn hooks(c: *const Chain) []const Reclaim {
        return c.at[0..c.n];
    }

    /// `r` joins last; a store already in the chain, or one past `most`, is refused.
    pub fn join(c: *Chain, r: Reclaim) error{ Joined, ChainFull }!void {
        for (c.hooks()) |h| if (h.ctx == r.ctx) return error.Joined;
        if (c.n == most) return error.ChainFull;
        c.at[c.n] = r;
        c.n += 1;
    }

    /// The store at `ctx` leaves and the rest keep their order (nothing when it never joined).
    pub fn leave(c: *Chain, ctx: *anyopaque) void {
        for (c.hooks(), 0..) |h, i| if (h.ctx == ctx) {
            std.mem.copyForwards(Reclaim, c.at[i .. c.n - 1], c.at[i + 1 .. c.n]);
            c.n -= 1;
            return;
        };
    }
};
