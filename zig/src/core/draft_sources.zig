//! Descriptive draft provenance; lane availability comes from the same config the round loop reads.
const std = @import("std");
const lanes = @import("lanes");

pub const Sources = struct {
    copies: bool = false,
    mtp: bool = false,
    drafter: ?Named = null,

    pub const Named = struct { kind: []const u8, name: []const u8 = "" };

    pub fn ofLanes(s: Sources, cfg: *const lanes.Config, streams: u32) Sources {
        const head = cfg.family_mtp and cfg.most_drafts > 0;
        return .{
            .copies = cfg.max_copy > 0 and (cfg.lone_copies or streams > 1),
            .mtp = head and s.drafter == null,
            .drafter = if (head) s.drafter else null,
        };
    }
};

test "draft sources: lane availability follows verification width and head controls" {
    const a = std.testing.allocator;
    const cases = [_]struct { model: lanes.Model, rows: u32 = 16, drafts: u32 = 15, streams: u32 = 1, copies: bool, mtp: bool }{
        .{ .model = .{}, .copies = false, .mtp = false },
        .{ .model = .{ .exact_width = 16 }, .copies = true, .mtp = false },
        .{ .model = .{ .exact_width = 16, .mtp = true, .speculate = true }, .copies = true, .mtp = true },
        .{ .model = .{ .exact_width = 16, .mtp = true, .speculate = true }, .rows = 1, .copies = false, .mtp = false },
        .{ .model = .{ .exact_width = 16, .mtp = true, .speculate = true }, .drafts = 0, .copies = true, .mtp = false },
        .{ .model = .{ .exact_width = 16, .mtp = true }, .copies = true, .mtp = false },
        .{ .model = .{ .exact_width = 16, .copy_drafts = false }, .copies = false, .mtp = false },
        .{ .model = .{ .exact_width = 16, .lone_copies = false }, .copies = false, .mtp = false },
        .{ .model = .{ .exact_width = 16, .lone_copies = false }, .streams = 2, .copies = true, .mtp = false },
    };
    for (cases) |c| {
        var cfg = try lanes.Config.init(a, c.model, c.rows, c.drafts);
        defer cfg.deinit(a);
        const got = (Sources{}).ofLanes(&cfg, c.streams);
        try std.testing.expectEqual(c.copies, got.copies);
        try std.testing.expectEqual(c.mtp, got.mtp);
        try std.testing.expect(got.drafter == null);
        const named = (Sources{ .drafter = .{ .kind = "DFlash2", .name = "example/drafter" } }).ofLanes(&cfg, c.streams);
        try std.testing.expect(!named.mtp);
        try std.testing.expectEqual(c.mtp, named.drafter != null);
    }
}
