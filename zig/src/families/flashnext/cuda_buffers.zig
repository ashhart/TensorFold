//! Shared sequence-buffer allocation and rollback, with filling explicit at each call site.
const std = @import("std");
const cuda = @import("cuda");

/// `n` buffers from `src.make`, optionally filled; on failure every buffer made so far is freed.
pub fn makeBuffers(comptime B: type, gpa: std.mem.Allocator, n: usize, src: anytype, comptime fill: bool) ![]B {
    const out = try gpa.alloc(B, n);
    var held: usize = 0;
    errdefer {
        if (@hasDecl(@TypeOf(src), "drain")) src.drain();
        for (out[0..held]) |*b| b.free();
        gpa.free(out);
    }
    while (held < n) {
        out[held] = try src.make();
        held += 1; // counted before its fill, so a failed fill frees it with the rest
        if (fill) try src.fill(&out[held - 1]);
    }
    return out;
}
