//! Assistants read only the newest target row; a later useful absorb replaces a skipped row in full.
const std = @import("std");
const lanes = @import("lanes");
const dr = lanes.drafter;

pub fn useful(d: anytype, items: []const dr.Absorb) !bool {
    var any = false;
    for (items) |it| {
        const l = d.lanes.get(it.stream) orelse return error.DraftStreamUnknown;
        if (it.propose) {
            any = true;
        } else {
            l.held_len = 0;
            l.rows = false;
        }
    }
    return any;
}

pub fn Cases(comptime Drafter: type, comptime Lane: type) type {
    return struct {
        test "zero grant: Gemma assistant skips its actual absorbed-row norm without touching a device" {
            var d: Drafter = undefined;
            d.lanes = .empty;
            defer d.lanes.deinit(std.testing.allocator);
            var s: lanes.Stream = undefined;
            var l: Lane = undefined;
            l.held_len = 6;
            l.rows = true;
            try d.lanes.put(std.testing.allocator, &s, &l);
            var it: dr.Absorb = undefined;
            it.stream = &s;
            it.propose = false;
            for (0..180) |_| try d.drafter().absorb(&.{it});
            try std.testing.expectEqual(@as(usize, 0), l.held_len);
            try std.testing.expect(!l.rows);
            it.propose = true;
            try std.testing.expect(try useful(&d, &.{it}));
        }
    };
}
