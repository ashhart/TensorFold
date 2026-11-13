//! What a Step-5 load may hold: core's load budget over Step-5's caches.
const cfg = @import("config.zig");
const st = @import("state.zig");
const load = @import("../../core/load_budget.zig");

pub const caches = load.caches;

/// The longest context one stream's caches fit in `bytes`, at most `most`.
pub fn longest(c: *const cfg.Config, bytes: u64, most: usize) usize {
    return load.longest(c, st.cacheBytes, bytes, most);
}

test {
    _ = load;
}
