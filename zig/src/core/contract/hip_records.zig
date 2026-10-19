//! The HIP launches gate 5 checks: each one's threads and LDS, and what hipcc made of it on every RDNA chip we serve.
const std = @import("std");
const limits = @import("limits.zig");

const U = limits.HipUsage;

/// A HIP launch and what hipcc made of it for gfx1030, gfx1100, gfx1151 and gfx1201 (limits.rdna's order).
pub const Record = struct { launch: limits.HipLaunch, usage: [limits.rdna.len]U };

const rms_ns = "_ZN2tf4rocm10rms_kernelI";

/// HIP kernels as launched; tf-hip-kernel-contract limits DIR re-derives the usage records from the code objects.
const own_records = [_]Record{
    .{
        .launch = .{ .name = "RMS norm tf_rms_rows", .symbol = "tf_rms_rows", .image = "ops", .threads = 256 },
        .usage = .{
            .{ .vgprs = 58, .sgprs = 45, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 43, .sgprs = 45, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 42, .sgprs = 45, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 42, .sgprs = 45, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        },
    },
    .{
        .launch = .{
            .name = "RMS norm rms_kernel<__half>",
            .symbol = rms_ns ++ "6__halfEEvPKT_PKfPS3_if",
            .image = "act",
            .threads = 256,
        },
        .usage = .{
            .{ .vgprs = 11, .sgprs = 18, .lds = 32772, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 11, .sgprs = 18, .lds = 32772, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 10, .sgprs = 18, .lds = 32772, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 10, .sgprs = 18, .lds = 32772, .private = 0, .bound = 256, .wave = 32 },
        },
    },
    .{
        .launch = .{
            .name = "RMS norm rms_kernel<hip_bfloat16>",
            .symbol = rms_ns ++ "12hip_bfloat16EEvPKT_PKfPS3_if",
            .image = "act",
            .threads = 256,
        },
        .usage = .{
            .{ .vgprs = 11, .sgprs = 18, .lds = 32772, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 11, .sgprs = 18, .lds = 32772, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 10, .sgprs = 18, .lds = 32772, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 10, .sgprs = 18, .lds = 32772, .private = 0, .bound = 256, .wave = 32 },
        },
    },
};

pub const records = own_records ++ @import("hip_records_stream.zig").records;

/// Gate 5's planted HIP controls: a launch past its __launch_bounds__, and 72 KiB of LDS where a workgroup has 64.
pub const controls = [_]Record{
    .{
        .launch = .{
            .name = "control: past its bounds",
            .symbol = "tf_ctl_bounded",
            .image = "contract_controls",
            .threads = 512,
        },
        .usage = .{
            .{ .vgprs = 4, .sgprs = 10, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 4, .sgprs = 18, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 4, .sgprs = 10, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 4, .sgprs = 7, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        },
    },
    .{
        .launch = .{
            .name = "control: 72 KiB of LDS",
            .symbol = "tf_ctl_big_lds",
            .image = "contract_controls",
            .threads = 256,
            .dynamic = 32 * 1024,
        },
        .usage = .{
            .{ .vgprs = 6, .sgprs = 9, .lds = 40960, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 6, .sgprs = 8, .lds = 40960, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 6, .sgprs = 8, .lds = 40960, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 6, .sgprs = 8, .lds = 40960, .private = 0, .bound = 1024, .wave = 32 },
        },
    },
};

/// Every HIP record's problems on every chip we serve, printed; returns how many (a missing record is one).
pub fn check(a: std.mem.Allocator, list: []const Record, quiet: bool) !usize {
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(a);
    var n: usize = 0;
    for (list) |r| for (limits.rdna, r.usage) |t, u| {
        if (u.vgprs == 0) {
            try out.print(a, "{s} on {s}: no compiled usage recorded\n", .{ r.launch.name, t.name });
            n += 1;
        } else n += try limits.hip(r.launch, u, t, &out, a);
    };
    if (!quiet and out.items.len > 0) std.debug.print("{s}", .{out.items});
    return n;
}

/// A planted control needs compiled usage and a real limit violation on every chip; returns missed chips.
pub fn controlProblems(a: std.mem.Allocator, r: Record, quiet: bool) !usize {
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(a);
    var n: usize = 0;
    for (limits.rdna, r.usage) |chip, usage| {
        if (usage.vgprs == 0) {
            n += 1;
            try out.print(a, "{s} on {s}: no compiled usage for control\n", .{ r.launch.name, chip.name });
        } else if (try limits.hip(r.launch, usage, chip, &out, a) == 0) {
            n += 1;
            try out.print(a, "{s} on {s}: planted control fits\n", .{ r.launch.name, chip.name });
        }
    }
    if (!quiet and n != 0) std.debug.print("{s}", .{out.items});
    return n;
}

test "every registered HIP launch fits every RDNA chip we serve, and each planted control breaks one" {
    const a = std.testing.allocator;
    try std.testing.expectEqual(@as(usize, 0), try check(a, &records, false));
    for (controls) |r| try std.testing.expectEqual(@as(usize, 0), try controlProblems(a, r, true));
    for (controls) |r| {
        var good = r;
        for (&good.usage) |*u| u.* = .{ .vgprs = 16 };
        good.launch.threads = 256;
        good.launch.dynamic = 0;
        try std.testing.expectEqual(@as(usize, 0), try check(a, &.{good}, true)); // red only for what it plants
    }
}

test "a control cannot fit or lack compiled usage on any one chip" {
    const a = std.testing.allocator;
    for (controls) |r| {
        for (limits.rdna, 0..) |_, i| {
            var fits = r;
            fits.usage[i] = .{ .vgprs = 16, .bound = fits.launch.threads };
            try std.testing.expectEqual(@as(usize, 1), try controlProblems(a, fits, true));
            var missing = r;
            missing.usage[i].vgprs = 0;
            try std.testing.expectEqual(@as(usize, 1), try controlProblems(a, missing, true));
        }
    }
}
