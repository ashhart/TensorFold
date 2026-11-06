//! Kolibri-1's paged full-layer caches: page geometry, the host pool's layout, the window and tree page views.
const cuda = @import("cuda");
const kv = @import("core").kv_pool;
const fwd = @import("forward.zig");

const Model = fwd.Model;
const Pages = cuda.window_attention.Pages;
const page_tokens = fwd.page_tokens;
const round_rows = fwd.round_rows;

/// Table entries a slot: the pages its context takes.
pub fn perSlot(m: *const Model) usize {
    return (m.context + page_tokens - 1) / page_tokens;
}

/// bf16 elements a page id steps in one full layer's K (or V) region: page_tokens rows of its kv heads.
pub fn pageElems(m: *const Model) usize {
    return page_tokens * m.c.kv_heads * m.c.head_dim;
}

pub fn fullLayers(m: *const Model) usize {
    var n: usize = 0;
    for (m.c.full[0..m.c.layers]) |full| n += @intFromBool(full);
    return n;
}

/// Bytes a page holds over every full layer's K and V.
pub fn pageBytes(m: *const Model) u64 {
    return fullLayers(m) * 2 * m.pageElems() * 2;
}

/// What the host's pool plans: a stream's sliding rings and table row fixed, a token its full layers' rows.
pub fn layout(m: *const Model) kv.Layout {
    const c = m.c;
    var fixed: u64 = m.perSlot() * 4;
    for (c.full[0..c.layers]) |full| if (!full) {
        fixed += 2 * m.ring * c.kv_heads * c.head_dim * 2;
    };
    return .{ .fixed_bytes = fixed, .token_bytes = m.pageBytes() / page_tokens, .page_tokens = page_tokens, .snap_bytes = 0 };
}

/// Slots a backend holds: the model's, or (paged) as many as a round takes.
pub fn slotCap(m: *const Model) usize {
    return if (m.paged) round_rows else m.slots;
}

/// The full layers' pages as a window reads them (a prompt's slot: its own table row); null: linear caches.
pub fn pagesFor(m: *const Model, prompt_slot: ?usize) ?Pages {
    if (!m.paged) return null;
    return .{ .table = m.table + (prompt_slot orelse 0) * m.perSlot() * 4, .page_elems = m.pageElems(), .tokens = page_tokens, .stride = m.perSlot() };
}

/// Tree attention's pages over `table`: a row of page ids for each stream of the call, in its order.
pub fn treePages(m: *const Model, table: u64) ?cuda.tree_attention.Pages {
    if (!m.paged) return null;
    return .{ .table = table, .page = m.pageElems(), .tok = page_tokens, .stride = m.perSlot() };
}
