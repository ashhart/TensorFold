//! Host-only assertions through the real lane-host engine info, without starting its worker.
const std = @import("std");
const api = @import("engine_api");
const lanes = @import("lanes");

pub fn check(model: lanes.Model, named: ?api.DraftSources.Named, copies: bool, mtp: bool) !void {
    var cfg = try lanes.Config.init(std.testing.allocator, model, model.exact_width, model.exact_width - 1);
    defer cfg.deinit(std.testing.allocator);
    var core: lanes.Engine = undefined;
    core.cfg = &cfg;
    var host = api.LaneHost.init(std.testing.allocator, std.testing.io, &core, .{ .lanes = @max(1, model.max_streams), .draft_sources = .{ .drafter = named } });
    var backend: lanes.fake.Fake = .{ .gpa = std.testing.allocator };
    core.backend = backend.backend();
    const info = host.engine().info();
    try std.testing.expectEqual(copies, info.draft_sources.copies);
    try std.testing.expectEqual(mtp, info.draft_sources.mtp);
    if (named) |n| {
        const got = info.draft_sources.drafter orelse return error.TestUnexpectedResult;
        try std.testing.expectEqualStrings(n.kind, got.kind);
        try std.testing.expectEqualStrings(n.name, got.name);
    } else try std.testing.expect(info.draft_sources.drafter == null);
}

pub fn present(comptime T: type) T {
    return if (@typeInfo(T) == .pointer) @ptrFromInt(@alignOf(@typeInfo(T).pointer.child)) else @as(T, undefined);
}
