//! Compact cache accounting and transient convolution staging bounds.
const std = @import("std");
const st = @import("state.zig");

test "compact cache budget excludes transient full convolution staging" {
    try std.testing.expectEqual(@as(usize, 491_091_432), st.cacheBytes(8192));
    try std.testing.expectEqual(st.kvBytes(8192) + st.conv_bytes + st.taps_bytes + st.vocabulary * 4, st.cacheBytes(8192));
}

test "forward staging covers disjoint K V and post-convolution ranges for every segment count" {
    try std.testing.expect(2 * st.kv_width <= st.hidden);
    for ([_]usize{ 1, 2, 7, 8, 15, 16, 32, 256 }) |rows| {
        for (1..@min(rows, st.max_segments) + 1) |segments| {
            var base: usize = 0;
            var end: usize = 0;
            for (0..segments) |si| {
                const n = rows / segments + @intFromBool(si < rows % segments);
                const start = st.convStageByteOffset(base, si);
                const k_end = start + (3 + n) * st.kv_width * 2;
                const v_end = k_end + (3 + n) * st.kv_width * 2;
                const post_end = start + (3 + n) * st.hidden * 2;
                try std.testing.expectEqual(end, start);
                try std.testing.expect(v_end <= post_end);
                try std.testing.expect(post_end <= st.convStageBytes(rows));
                base += n;
                end = post_end;
            }
            try std.testing.expectEqual(rows, base);
        }
    }
    try std.testing.expectEqual(@as(usize, 589_824), st.convStageBytes(st.batch_rows));
    try std.testing.expectEqual(@as(usize, 1_622_016), st.convStageBytes(st.prompt_rows));
}
