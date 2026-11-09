//! Forward.stage: a pass's host part ahead of its launches, apart from run so a window's graph can replay the launches.
const std = @import("std");
const cuda = @import("cuda");
const fw = @import("cuda_forward.zig");
const shared = @import("cuda_shared.zig");

/// run's host part, ahead of its launches: each rank's state checked (its rings rebased if due) and `ids` uploaded.
pub fn stage(f: *const fw.Forward, ids: []const i32) !void {
    const R = ids.len;
    for (f.ranks) |r| {
        if (R == 0 or R > r.b.rows) return error.BadWindow;
        if (r.segs.len == 0) {
            try r.st.ready(f.stream, R);
            r.st.pending = R;
            r.st.prompt = r.b.prefill;
        } else try shared.begin(r, R, f.stream);
        try cuda.copy_source.enqueue(f.d, r.b.ids, std.mem.sliceAsBytes(ids), f.stream);
    }
}
