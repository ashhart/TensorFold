//! How an engine admits a request beside its running streams: a fit check, or a byte budget it promises from.
const kv_ledger = @import("kv_ledger.zig");

pub const Admit = struct { ctx: *anyopaque, fits: *const fn (ctx: *anyopaque, prompt: u32, max_tokens: u32) bool };

/// The bytes the lane host may promise at once, and what one request's caches cost at most (core/kv_ledger.zig).
pub const Ledger = struct {
    ctx: *anyopaque,
    total: u64,
    cost: *const fn (ctx: *anyopaque, prompt: u32, max_tokens: u32) u64,
    standing: kv_ledger.Standing = .{}, // streams paid at load, in `total`: the engine's own sequence (#1943)
};
