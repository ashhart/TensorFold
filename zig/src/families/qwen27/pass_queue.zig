//! Metal 3/4 pass selection for Qwen27 verify and decode.

/// Passes past this many rows are prompt chunks (128 rows and up), 3.5% slower on Metal 4 with qk: Metal 3 takes them.
const metal4_rows = 64;

/// The model's Metal 4 queue, unless the pass is a prompt chunk or needs Metal 3: a trace, the Neural Engine, seams.
pub fn queue(r: anytype, rows: usize) @TypeOf(r.model.queue) {
    const q = r.model.metal4 orelse return r.model.queue;
    if (!r.metal4 or rows > metal4_rows or r.trace != null or r.model.ane != null or r.reduce != null) return r.model.queue;
    return q;
}
