//! LFM2's tap sources: a stepped chunk's rows come from the lane's stage, not only from its last-row anchor.
const std = @import("std");
const st = @import("state.zig");
const tap = @import("backend_tap.zig");

test "lfm2 dspark: a stepped prompt's last chunk serves its rows from the lane's stage" {
    const chunk = st.TapRows{ .start = 8, .row = 0, .rows = 4, .gen = 7, .stage = 80 };
    const anchor = st.TapRows{ .start = 11, .row = 0, .rows = 1, .gen = 0, .stage = 104 };
    // the wrapper absorbs rows [8, 11) right after the final step: the anchor alone would refuse them
    const rows = (try tap.source(anchor, chunk, 7, 8, 3)) orelse return error.TestExpectedRows;
    try std.testing.expectEqual(tap.Place.stage, rows.place);
    try std.testing.expectEqual(@as(?usize, 80), rows.rows.stage);
    // the prompt's last row still comes from the kept gen-0 anchor, across other streams' forwards
    const last = (try tap.source(anchor, chunk, 9, 11, 1)) orelse return error.TestExpectedRows;
    try std.testing.expectEqual(tap.Place.capture, last.place);
    try std.testing.expectEqual(@as(?usize, 104), last.rows.stage);
    // a forward since the chunk rewrote the scratch: the chunk's rows are no longer served
    try std.testing.expectEqual(@as(?tap.Source, null), try tap.source(anchor, chunk, 8, 8, 3));
    // the scratch path still serves a chunk's rows while current, and a stale capture is an error
    const scratch = st.TapRows{ .start = 4, .row = 0, .rows = 3, .gen = 5 };
    const fresh = (try tap.source(null, scratch, 5, 4, 3)) orelse return error.TestExpectedRows;
    try std.testing.expectEqual(tap.Place.scratch, fresh.place);
    try std.testing.expectError(error.StaleLfmTaps, tap.source(.{ .start = 0, .row = 0, .rows = 4, .gen = 3 }, null, 7, 0, 2));
}
