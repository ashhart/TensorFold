//! A paged slot's KV pages over its stream's life: promised as it opens, grown as rows come, given back as it goes.
const kvp = @import("core").metal_kv_pages;

/// The stream opening lane `slot` is promised the pages of `rows` rows past those it holds; LaneMemory: not now.
pub fn promise(p: *kvp.Pages, slot: usize, rows: u64) !void {
    const l = &p.leases[slot];
    p.pool.unpromise(l);
    const need = p.layout.pages(rows);
    if (need > l.ids.items.len) try p.admit(slot, need - l.ids.items.len);
}

/// Lane `slot` holds the pages of `rows` rows; a pass past its promise promises the rest first (LaneMemory: none).
pub fn grow(p: *kvp.Pages, slot: usize, rows: u64) !void {
    const l = &p.leases[slot];
    const short = p.layout.pages(rows) -| (l.ids.items.len + l.promised);
    if (short > 0) try p.admit(slot, short); // fails before cover: no page taken, no table entry written
    try p.cover(slot, rows);
}

/// Lane `slot`'s stream left: every page and its promise go back, or with `kept`, the pages of those rows stay.
pub fn leave(p: *kvp.Pages, slot: usize, kept: ?u64) void {
    const rows = kept orelse return p.finish(slot);
    p.trim(slot, rows);
    p.pool.unpromise(&p.leases[slot]);
}

/// Lane `slot` starts over at row 0: its pages go back to its promise (a kept holder's page stays held).
pub fn startOver(p: *kvp.Pages, slot: usize) void {
    p.trim(slot, 0);
}

/// Runner.cover: slot `slot` holds the pages of its first `rows` rows before a pass or a copy writes them.
pub fn cover(r: anytype, slot: u32, rows: u32) !void {
    if (r.kv) |k| try grow(k.pages, slot, @min(rows, r.capacity));
}

/// Runner.promise: a stream opens in slot `slot`, the pages of `rows` rows promised; LaneMemory: not now.
pub fn promiseRows(r: anytype, slot: u32, rows: u32) !void {
    if (r.kv) |k| try promise(k.pages, slot, @min(rows, r.capacity));
}

/// Runner.release: slot `slot`'s stream left; `standing`, the pages of its committed rows stay for a resume.
pub fn release(r: anytype, slot: u32, standing: bool) void {
    const k = r.kv orelse return;
    if (!standing) r.settleSlot(slot); // a keep's blit still reading the slot lands before its pages go back
    leave(k.pages, slot, if (standing) r.offsets[slot] else null);
}

/// Runner.reset: slot `slot` starts over at row 0, its DeltaNet state cleared and its pages given back.
pub fn reset(r: anytype, slot: u32) !void {
    try r.clearState(slot);
    if (r.kv) |k| startOver(k.pages, slot);
}
