//! Owned-run capacity respects admitted lanes and the memory that backs their pages.
const std = @import("std");
const pool = @import("kv_pool.zig");
const capacity = @import("kv_pool_plan_cap.zig");

test "owned page capacity follows admitted windows and the backing budget" {
    var layout: pool.Layout = .{ .fixed_bytes = 10, .token_bytes = 3, .page_tokens = 4, .snap_bytes = 0, .run_bytes = 13, .engine_run = true };
    const Backing = struct { room: u64, engine: u64, engine_paged: bool, window: u64, pages_room: u64 };
    const Planned = struct { lanes: u32, snaps: usize, copied: u64, kept: u64, pages: usize, host: usize = 0 };
    var backing: Backing = .{ .room = 208, .engine = 10, .engine_paged = true, .window = 9, .pages_room = 240 };
    const admitted: Planned = .{ .lanes = 4, .snaps = 0, .copied = 0, .kept = 0, .pages = 0 };
    const host = capacity.bind(layout, backing, admitted);
    try std.testing.expectEqual(@as(usize, 20), host.pages);
    try std.testing.expectEqual(host.pages, host.host);
    backing.pages_room = 84;
    try std.testing.expectEqual(@as(usize, 7), capacity.bind(layout, backing, admitted).pages);
    backing.pages_room = 240;
    layout.engine_run = false;
    try std.testing.expectEqual(@as(usize, 18), capacity.bind(layout, backing, admitted).pages);
    layout.engine_run = true;
    backing.engine_paged = false;
    try std.testing.expectEqual(@as(usize, 15), capacity.bind(layout, backing, admitted).pages);
    backing.engine_paged = true;
    backing.pages_room = 0;
    const device = capacity.bind(layout, backing, admitted);
    try std.testing.expectEqual(@as(usize, 14), device.pages);
    try std.testing.expectEqual(@as(usize, 0), device.host);
    backing.window = 0;
    try std.testing.expectEqual(@as(usize, 14), capacity.bind(layout, backing, admitted).pages);
}
