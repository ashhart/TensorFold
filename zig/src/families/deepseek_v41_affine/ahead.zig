//! The next prompt chunk's engram ids and rows, gathered on the host while this chunk's command buffer runs.
const std = @import("std");
const st = @import("state.zig");
const c = @import("config.zig");
const fwd = @import("forward.zig");
const Model = @import("model.zig").Model;

fn hashOf(tokens: []const u32) u64 {
    return std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(tokens));
}

/// Whether `s.prepared` holds this call's rows: the same cache at the same rows, its ids appended, the tokens the same.
pub fn preparedFor(s: *const st.Scratch, g: fwd.Seg) bool {
    const p = s.prepared orelse return false;
    return p.cache == g.cache and p.base == g.cache.len and p.n == g.tokens.len and g.cache.engram_ids.items.len == p.base + p.n and p.hash == hashOf(g.tokens);
}

/// The next chunk's engram ids and rows into the other set, recorded in `s.prepared`; on any failure, nothing is kept.
pub fn prepare(m: *Model, s: *st.Scratch, cache: *st.Cache, base: usize, nx: fwd.Next) void {
    const n = nx.tokens.len;
    if (n == 0 or n > s.rows or base + n > cache.capacity or cache.engram_ids.items.len != base) return;
    const set: u8 = 1 - s.engram_set;
    for (nx.tokens, 0..) |t, i| cache.engram_ids.appendAssumeCapacity(m.engram.compress(t, fwd.Vis.code(nx.vis, base + i) == -1));
    var dsts: [c.engram_layers.len][]u16 = undefined;
    const width = c.engram_cols * c.engram_dim;
    for (&dsts, s.engram_rows[set]) |*d, b| d.* = b.slice(u16, n * width);
    m.engram.gatherRows(cache.engram_ids.items, base, n, dsts) catch {
        cache.engram_ids.shrinkRetainingCapacity(base);
        return;
    };
    s.prepared = .{ .cache = cache, .base = base, .n = n, .hash = hashOf(nx.tokens), .set = set };
}
