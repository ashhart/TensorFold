//! Where a prompt pass cuts its chunks, for the prompt cache's resumes and marks.
const modes = @import("cache_modes.zig");

/// The chunk start at or before `at` (0 when none).
pub fn floorStart(starts: []const u32, at: u32) u32 {
    var best: u32 = 0;
    for (starts) |p| if (p <= at and p > best) {
        best = p;
    };
    return best;
}

/// The last chunk end at or before `at` of a pass from `from`: whole chunks from its last start, span edge or `from`.
pub fn chunkEnd(step: u32, from: u32, at: u32, len: usize, starts: []const u32, spans: []const modes.Span) u32 {
    if (at <= from or at >= len) return at; // the prompt's end ends a chunk; the caller drops marks at or before `from`
    var cut = from;
    for (starts) |c| cut = if (c > cut and c <= at) c else cut;
    for (spans) |span| for (span) |c| {
        cut = if (c > cut and c <= at) c else cut;
    };
    return cut + (at - cut) / step * step;
}
