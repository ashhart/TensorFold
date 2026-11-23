//! A Metal pass that fails after it fitted the prompt cache's room short gives the room back, as a release would.
const std = @import("std");

/// Each family's open path: the charge's undo and the release's refit (the Metal backends build only on macOS).
const Case = struct { name: []const u8, src: []const u8, undo: []const u8, refit: []const u8 };
const cases = [_]Case{
    .{ .name = "llama", .src = @embedFile("llama/backend.zig"), .undo = "self.ledger.give(need);", .refit = "r.fit(r.ctx, self.ledger.left(), null, &.{});" },
    .{ .name = "ripple", .src = @embedFile("ricursive/backend.zig"), .undo = "self.ledger.give(need);", .refit = "r.fit(r.ctx, self.ledger.left(), null, &.{});" },
    .{ .name = "spark25", .src = @embedFile("spark25/metal_backend.zig"), .undo = "self.held -= need;", .refit = "r.fit(r.ctx, self.options.budget - self.held, null, &.{});" },
    .{ .name = "flint", .src = @embedFile("flint/backend.zig"), .undo = "self.held -= need;", .refit = "r.fit(r.ctx, self.options.budget - self.held, null, &.{});" },
    .{ .name = "step5", .src = @embedFile("step5/backend.zig"), .undo = "self.held -= need;", .refit = "r.fit(r.ctx, self.options.budget - self.held, null, &.{});" },
    .{ .name = "minimax_m3", .src = @embedFile("minimax_m3/prompt_pass.zig"), .undo = "self.held -= need;", .refit = "r.fit(r.ctx, self.options.budget - self.held, null, &.{});" },
};

test "Metal open room: a pass failing after its room fit gives the room back with the bytes, in every family (D1073)" {
    for (cases) |c| {
        // the errdefer that undoes the stream's charge, up to its block's end
        const at = std.mem.indexOf(u8, c.src, "errdefer {") orelse {
            std.debug.print("{s}: the charge's undo is a single statement: the room stays fitted short\n", .{c.name});
            return error.RoomNotGivenBack;
        };
        const block = c.src[at..][0 .. std.mem.indexOf(u8, c.src[at..], "}\n").? + 1];
        try std.testing.expect(std.mem.indexOf(u8, block, c.undo) != null);
        try std.testing.expect(std.mem.indexOf(u8, block, c.refit) != null);
        // and it comes before the cache is made, so a failing createCache or caches.put runs it
        try std.testing.expect(at < std.mem.indexOf(u8, c.src, "try self.createCache(capacity);").?);
    }
}
