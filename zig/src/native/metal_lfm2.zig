//! LFM2.5-24B-A2B's Metal entry: DSpark drafts need M5 or newer and stand aside on older Macs.
const std = @import("std");
const mtl = @import("metal");
const api = @import("engine_api");
const metal_admit = @import("metal_admit.zig");
const metal = @import("metal.zig");

/// The engine for an `lfm2_moe` checkpoint, with a DSpark drafter only where the chip serves one.
pub fn open(a: std.mem.Allocator, gpa: std.mem.Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !?api.Opened {
    var serving = o;
    if (o.drafter != null) {
        const device = try mtl.Device.init();
        defer device.deinit();
        const name = std.mem.span(device.name());
        serving = metal_admit.lfm2DrafterOptions(o, name);
        if (serving.drafter == null and o.drafts) std.log.info(
            "lfm2: DSpark standing aside on {s}; serving without a neural drafter (requires Apple M5 or newer)",
            .{name},
        );
    }
    return @import("lfm2_moe.zig").open(a, gpa, io, serving, problem);
}

test "drafter admission: LFM2 DSpark and Llama survive centralized Metal refusal" {
    const a = std.testing.allocator;
    for ([_][]const u8{ "lfm2_moe", "llama", "gemma4", "qwen3_5_moe" }) |model| {
        try std.testing.expect((try metal.refusal(a, .{ .dir = "d", .model_type = model, .drafter = "draft" }, false)) == null);
    }
    try std.testing.expect((try metal.refusal(a, .{ .dir = "d", .model_type = "qwen3_5", .drafter = "draft" }, true)) == null);
    for ([_][]const u8{ "qwen3_5", "nemotron_h", "unknown" }) |model| {
        try std.testing.expect((try metal.refusal(a, .{ .dir = "d", .model_type = model, .drafter = "draft" }, false)) != null);
    }
}
