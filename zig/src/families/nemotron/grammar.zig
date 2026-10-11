//! Structured output on Nemotron's Metal backend: bits staged by window row, a masked first draw, held drafts read.
const lanes = @import("lanes");
const st = @import("state.zig");
const Masks = @import("../../core/grammar_mask.zig").Masks;
const Metal = @import("backend.zig").Metal;

const be = lanes.backend;

/// The backend's bits, made at the first structured request (one row a forward row, and a first token's).
fn masks(b: *Metal) !*Masks {
    if (b.masks == null) b.masks = try Masks.init(b.m.device, b.scratch.rows, b.m.config.vocab);
    return &b.masks.?;
}

/// A window's bits into the rows it takes from `row0` of the forward; the segment's mask row (null: no grammar).
pub fn stage(b: *Metal, w: be.Window, row0: usize) !?usize {
    if (w.masks.len == 0) return null;
    const m = try masks(b);
    if (w.masks.len != w.rows() * m.words) return error.InvalidGrammarMasks;
    try m.put(row0, w.masks);
    return row0;
}

/// The first token drawn among `mask`'s allowed tokens from the prompt's last row (Backend.first_masked).
pub fn firstMasked(ptr: *anyopaque, s: *lanes.Stream, position: u64, mask: []const u32) anyerror!u64 {
    const b: *Metal = @ptrCast(@alignCast(ptr));
    const m = try masks(b);
    if (mask.len != m.words) return error.InvalidGrammarMasks;
    try b.drain(); // no command in flight reads the first token's bits
    try m.put(m.firstRow(), mask);
    return b.firstAt(s, position, m.firstRow());
}

/// The drafts the head wrote into the stream's next window, once they land (Backend.held).
pub fn held(ptr: *anyopaque, s: *lanes.Stream, out: []u32) anyerror!void {
    const b: *Metal = @ptrCast(@alignCast(ptr));
    const c = try b.cacheOf(s);
    if (out.len + 1 > st.max_rows) return error.WindowTooWide;
    try b.drain();
    @memcpy(out, c.windows[c.wcur].slice(u32, st.max_rows)[1..][0..out.len]);
}
