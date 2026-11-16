//! The MTP confidence floor each HIP card, rank count and family resolves, from the cards' real caps rows.
const std = @import("std");
const hip = @import("hip");
const api = @import("engine_api");
const qwen35 = @import("qwen3_5");
const flashnext = @import("flashnext");
const fam = @import("hip_families.zig");
const native_hip = @import("hip.zig");

test "Flash Next's MTP floor is 0.8 on one RDNA 3.5 GPU and 0.3 on other cards, other families and groups; the flag wins" {
    const wide: hip.Policy = .{};
    try std.testing.expectEqual(@as(f64, 0.3), wide.mtp.confidence);
    const plain: api.Open = .{ .dir = "d", .model_type = "qwen4_exp" };
    const floor = struct {
        fn of(comptime F: type, o: api.Open, gfx: []const u8, ranks: u32) f64 {
            return fam.confidenceFloor(F, o, native_hip.rdna3_5Alone(hip.Caps.of(gfx).?, ranks), (hip.Policy{}).mtp.confidence);
        }
    }.of;
    for ([_][]const u8{ "gfx1150", "gfx1151", "gfx1152", "gfx1153" }) |gfx| try std.testing.expectEqual(@as(f64, 0.8), floor(flashnext.native, plain, gfx, 1));
    try std.testing.expectEqual(@as(f64, 0.3), floor(flashnext.native, plain, "gfx1151", 2));
    for ([_][]const u8{ "gfx1030", "gfx1100", "gfx1201" }) |gfx| try std.testing.expectEqual(@as(f64, 0.3), floor(flashnext.native, plain, gfx, 1));
    inline for (.{ qwen35.native, qwen35.native_moe }) |F| try std.testing.expectEqual(@as(f64, 0.3), floor(F, .{ .dir = "d", .model_type = F.model_type }, "gfx1151", 1));
    const asked: api.Open = .{ .dir = "d", .model_type = "qwen4_exp", .mtp_confidence = 0.3 };
    try std.testing.expectEqual(@as(f64, 0.3), floor(flashnext.native, asked, "gfx1151", 1));
    const high: api.Open = .{ .dir = "d", .model_type = "qwen4_exp", .mtp_confidence = 0.9 };
    try std.testing.expectEqual(@as(f64, 0.9), floor(flashnext.native, high, "gfx1201", 2));
}
