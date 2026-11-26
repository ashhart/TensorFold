//! Every registry name walks the actual launch resolver without compiling or loading a shader.
const std = @import("std");
const registry = @import("contract").registry;
const launches = @import("metal_limits.zig");

test "Metal launch map: every registered launch and control has a resolver branch" {
    var missing: usize = 0;
    for ([_][]const @import("contract").limits.MetalLaunch{ &registry.all_metal, &registry.metal_controls }) |list| {
        for (list) |d| launches.requireMapped(d) catch |err| {
            std.debug.print("UNMAPPED {s}: {t}\n", .{ d.name, err });
            missing += 1;
        };
    }
    std.debug.print("Metal launch map: {d} registered launches, {d} controls, {d} missing\n", .{ registry.all_metal.len, registry.metal_controls.len, missing });
    try std.testing.expectEqual(@as(usize, 0), missing);
}

test "Metal launch map: every DSpark and DFlash2 fast limit pipeline has a resolver branch" {
    var count: usize = 0;
    var fast_count: usize = 0;
    for (registry.all_metal) |d| {
        if (std.mem.startsWith(u8, d.name, "metal inkling2 dspark ")) {
            try launches.requireMapped(d);
            count += 1;
        } else if (std.mem.startsWith(u8, d.name, "metal inkling2 draft_head_q4 ") or std.mem.startsWith(u8, d.name, "metal inkling2 dflash2_fast ")) {
            try launches.requireMapped(d);
            fast_count += 1;
        }
    }
    try std.testing.expectEqual(@as(usize, 12), count);
    try std.testing.expectEqual(@as(usize, 3), fast_count);
}

test "Metal launch map: unknown registrations stay errors on every chip" {
    try std.testing.expectError(error.UnknownLaunch, launches.requireMapped(.{ .name = "unregistered family missing_kernel", .threads = 32, .tg_memory = 0 }));
}

test {
    _ = @import("metal_support.zig");
}

test "Metal launch map: LFM2 masked prompt follows its loader on each device class" {
    var masked: usize = 0;
    for (registry.all_metal) |d| {
        if (!std.mem.startsWith(u8, d.name, "lfm2 masked prompt ")) continue;
        masked += 1;
        try launches.requireMapped(d);
        try std.testing.expect(!launches.supported(.{ .units = false }, d));
        try std.testing.expect(launches.supported(.{ .units = true }, d));
    }
    try std.testing.expect(masked > 0);
    const base = @import("contract").limits.MetalLaunch{ .name = "metal lfm2 lfm2_embed_b4", .threads = 32, .tg_memory = 0 };
    try launches.requireMapped(base);
    try std.testing.expect(launches.supported(.{ .units = false }, base));
}

test "Metal launch map: DeepSeek-V4.1 tensor-unit expert launches follow their loader" {
    var units: usize = 0;
    var simd: usize = 0;
    for (registry.all_metal) |d| {
        if (std.mem.startsWith(u8, d.name, "dsv41 xmma units ")) {
            units += 1;
            try launches.requireMapped(d);
            try std.testing.expect(!launches.supported(.{ .units = false }, d));
            try std.testing.expect(launches.supported(.{ .units = true }, d));
        } else if (std.mem.startsWith(u8, d.name, "dsv41 xmma ")) {
            simd += 1;
            try launches.requireMapped(d);
            try std.testing.expect(launches.supported(.{ .units = false }, d));
        }
    }
    try std.testing.expectEqual(@as(usize, 4), units);
    try std.testing.expectEqual(@as(usize, 4), simd);
}

test "Metal launch map: paged prompt variants follow M5 and simdgroup eligibility" {
    var count: usize = 0;
    for (registry.all_metal) |d| {
        if (!std.mem.startsWith(u8, d.name, "qwen27 prompt ")) continue;
        count += 1;
        const units = std.mem.indexOf(u8, d.name, " m5 ") != null;
        try launches.requireMapped(d);
        try std.testing.expectEqual(!units, launches.supported(.{ .units = false }, d));
        try std.testing.expectEqual(units, launches.supported(.{ .units = true }, d));
    }
    try std.testing.expectEqual(@as(usize, 32), count);
}

test "Metal launch map: Inkling2 MMA launches resolve and support all Metal GPUs" {
    var count: usize = 0;
    for (registry.all_metal) |d| {
        if (!std.mem.startsWith(u8, d.name, "metal inkling2 mma ")) continue;
        count += 1;
        try launches.requireMapped(d);
        try std.testing.expect(launches.supported(.{ .units = false }, d));
        try std.testing.expect(launches.supported(.{ .units = true }, d));
    }
    try std.testing.expectEqual(@as(usize, 7), count);
}
