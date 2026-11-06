//! Flash Next admission uses the registered family and production opener before any checkpoint or CUDA access.
const std = @import("std");
const api = @import("engine_api");
const F = @import("flashnext").native;
const refusals = @import("cuda_refusal.zig");

pub fn cache() !void {
    for ([_]?f64{ null, 0, 4 }) |gib| {
        for ([_]u32{ 1, 2 }) |tp| {
            for ([_]bool{ false, true }) |drafts| {
                for (0..tp) |rank| {
                    const o: api.Open = .{
                        .dir = "/missing-flashnext-test-checkpoint",
                        .model_type = F.model_type,
                        .tp = tp,
                        .rank = @intCast(rank),
                        .drafts = drafts,
                        .prompt_cache_gib = gib,
                    };
                    try std.testing.expect(refusals.family(F, o) == null);
                }
            }
        }
    }
}

pub fn vision() !void {
    const o: api.Open = .{ .dir = "/missing-flashnext-test-checkpoint", .model_type = F.model_type, .vision = .{} };
    try std.testing.expect(refusals.family(F, o) == null);
    const VisionFamily = struct {
        pub const Options = struct { vision: bool };
    };
    try std.testing.expect(refusals.family(VisionFamily, o) == null);
    try std.testing.expect(refusals.family(F, .{ .dir = "d", .model_type = F.model_type }) == null);
}
