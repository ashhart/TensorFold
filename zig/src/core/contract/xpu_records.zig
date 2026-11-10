//! The XPU launches gate 5 checks: Qwen3.8-27B glue and attention merge, what ocloc made of them, and the controls.
const std = @import("std");
const limits = @import("xpu_limits.zig");

const X = limits.XpuLaunch;
const U = limits.XpuUsage;

/// A launch and what ocloc made of it for each card class; `compiled` false: only the declaration is checked.
pub const Record = struct { launch: X, usage: [limits.xe.len]U, compiled: bool = true };

const glue = limits.Format.native;

/// Qwen3.8-27B glue and the attention merge: row arithmetic, a stated rule, a row copy and a replacement.
pub const records = [_]Record{
    .{
        .launch = .{ .name = "add_rmsnorm", .symbol = "add_rmsnorm", .image = "glue", .format = glue, .threads = .{ 128, 1, 1 }, .slm = 10272, .sub_group = 16, .grf = 256 },
        .usage = .{.{ .threads = .{ 128, 1, 1 }, .slm = 10272, .sub_group = 16, .grf = 256 }},
    },
    .{
        .launch = .{ .name = "add_rmsnorm_dec", .symbol = "add_rmsnorm_dec", .image = "glue", .format = glue, .threads = .{ 512, 1, 1 }, .slm = 10368, .sub_group = 16, .grf = 256 },
        .usage = .{.{ .threads = .{ 512, 1, 1 }, .slm = 10368, .sub_group = 16, .grf = 256 }},
    },
    .{
        .launch = .{ .name = "gated_norm", .symbol = "gated_norm", .image = "glue", .format = glue, .threads = .{ 256, 1, 1 }, .slm = 12288, .sub_group = 16, .grf = 256 },
        .usage = .{.{ .threads = .{ 256, 1, 1 }, .slm = 12288, .sub_group = 16, .grf = 256 }},
    },
    .{
        .launch = .{ .name = "swiglu", .symbol = "swiglu", .image = "glue", .format = glue, .threads = .{ 256, 1, 1 }, .slm = 2048, .sub_group = 16, .grf = 256 },
        .usage = .{.{ .threads = .{ 256, 1, 1 }, .slm = 2048, .sub_group = 16, .grf = 256 }},
    },
    .{
        .launch = .{ .name = "argmax", .symbol = "argmax", .image = "glue", .format = glue, .threads = .{ 256, 1, 1 }, .slm = 2048, .sub_group = 16, .grf = 256 },
        .usage = .{.{ .threads = .{ 256, 1, 1 }, .slm = 2048, .sub_group = 16, .grf = 256 }},
    },
    .{
        .launch = .{ .name = "embed", .symbol = "embed", .image = "glue", .format = glue, .threads = .{ 256, 1, 1 }, .sub_group = 16, .grf = 256 },
        .usage = .{.{ .threads = .{ 256, 1, 1 }, .sub_group = 16, .grf = 256 }},
    },
    .{
        .launch = .{ .name = "attn_merge", .symbol = "attn_merge", .image = "decode", .format = glue, .threads = .{ 16, 1, 1 }, .sub_group = 16, .grf = 256 },
        .usage = .{.{ .threads = .{ 16, 1, 1 }, .sub_group = 16, .grf = 256 }},
    },
    .{
        .launch = .{ .name = "upstream attn_merge", .symbol = "attn_merge", .image = "contract_merge_upstream", .format = glue, .threads = .{ 16, 1, 1 }, .sub_group = 16, .grf = 256 },
        .usage = .{.{ .threads = .{ 16, 1, 1 }, .sub_group = 16, .grf = 256 }},
    },
    .{
        .launch = .{ .name = "planted attn_merge weighing past its last slice", .symbol = "tf_ctl_attn_merge_unguarded", .image = "contract_gate_controls", .format = glue, .threads = .{ 16, 1, 1 }, .sub_group = 16, .grf = 256 },
        .usage = .{.{ .threads = .{ 16, 1, 1 }, .sub_group = 16, .grf = 256 }},
    },
};

/// Gate 5's planted XPU controls: one for each fit rule the audit must fire.
pub const controls = [_]Record{
    .{
        .launch = .{ .name = "control: 2048 work-items a work-group", .symbol = "tf_ctl_wide_group", .image = "contract_controls", .format = glue, .threads = .{ 2048, 1, 1 }, .sub_group = 16, .grf = 256 },
        .usage = .{.{ .threads = .{ 256, 1, 1 }, .sub_group = 16, .grf = 256 }},
    },
    .{
        .launch = .{ .name = "control: 200 work-items, not a multiple of sub-group 16", .symbol = "tf_ctl_wide_group", .image = "contract_controls", .format = glue, .threads = .{ 200, 1, 1 }, .sub_group = 16, .grf = 256 },
        .usage = .{.{ .threads = .{ 256, 1, 1 }, .sub_group = 16, .grf = 256 }},
    },
    .{
        .launch = .{ .name = "control: 160 KiB of SLM", .symbol = "tf_ctl_big_slm", .image = "contract_controls", .format = glue, .threads = .{ 256, 1, 1 }, .slm = 160 * 1024, .sub_group = 16, .grf = 256 },
        .usage = .{.{ .threads = .{ 256, 1, 1 }, .slm = 160 * 1024, .sub_group = 16, .grf = 256 }},
    },
    .{
        .launch = .{ .name = "control: 1024 work-items at 256 GRF", .symbol = "tf_ctl_large_grf", .image = "contract_controls", .format = glue, .threads = .{ 1024, 1, 1 }, .sub_group = 16, .grf = 256 },
        .usage = .{.{ .threads = .{ 1024, 1, 1 }, .sub_group = 16, .grf = 256 }},
    },
    .{
        .launch = .{ .name = "control: sub-group 32 at 256 GRF", .symbol = "tf_ctl_subgroup32", .image = "contract_controls", .format = glue, .threads = .{ 256, 1, 1 }, .sub_group = 32, .grf = 256 },
        .usage = .{.{ .threads = .{ 256, 1, 1 }, .sub_group = 32, .grf = 256 }},
    },
    .{
        .launch = .{ .name = "control: sub-group 8", .symbol = "tf_ctl_subgroup8", .image = "contract_controls_sg8", .format = glue, .threads = .{ 256, 1, 1 }, .sub_group = 8, .grf = 256 },
        .usage = .{.{ .threads = .{ 256, 1, 1 }, .sub_group = 8, .grf = 256 }},
        .compiled = false,
    },
};

fn empty(u: U) bool {
    return u.grf == 0 and u.sub_group == 0 and u.threads[0] == 0;
}

/// Every record's problems on every card class, printed; returns how many (a record with no compile is one).
pub fn check(a: std.mem.Allocator, list: []const Record, quiet: bool) !usize {
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(a);
    var n: usize = 0;
    for (list) |r| for (limits.xe, r.usage) |t, u| {
        if (!r.compiled) {
            n += try limits.fits(r.launch, t, &out, a);
            continue;
        }
        if (empty(u)) {
            try out.print(a, "{s} on {s}: no compiled usage recorded\n", .{ r.launch.name, t.name });
            n += 1;
        } else n += try limits.xpu(r.launch, u, t, &out, a);
    };
    if (!quiet and out.items.len > 0) std.debug.print("{s}", .{out.items});
    return n;
}

/// A planted control needs a compiled usage and a real limit violation on every card class; returns missed rows.
pub fn controlProblems(a: std.mem.Allocator, r: Record, quiet: bool) !usize {
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(a);
    var n: usize = 0;
    for (limits.xe, r.usage) |t, u| {
        if (!r.compiled) {
            if (try limits.fits(r.launch, t, &out, a) == 0) {
                n += 1;
                try out.print(a, "{s} on {s}: planted control fits\n", .{ r.launch.name, t.name });
            }
        } else if (empty(u)) {
            n += 1;
            try out.print(a, "{s} on {s}: no compiled usage for control\n", .{ r.launch.name, t.name });
        } else if (try limits.fits(r.launch, t, &out, a) == 0) {
            n += 1;
            try out.print(a, "{s} on {s}: planted control fits\n", .{ r.launch.name, t.name });
        }
    }
    if (!quiet and n != 0) std.debug.print("{s}", .{out.items});
    return n;
}

test "every registered XPU launch fits the BMG card, and each planted control breaks one" {
    const a = std.testing.allocator;
    try std.testing.expectEqual(@as(usize, 0), try check(a, &records, false));
    for (controls) |r| try std.testing.expectEqual(@as(usize, 0), try controlProblems(a, r, true));
    for (controls) |r| {
        var good = r;
        good.launch.threads = .{ 256, 1, 1 };
        good.launch.slm = 0;
        good.launch.sub_group = 16;
        for (&good.usage) |*u| u.* = .{ .threads = .{ 256, 1, 1 }, .sub_group = 16, .grf = 256 };
        try std.testing.expectEqual(@as(usize, 0), try check(a, &.{good}, true));
    }
}

test "a control cannot fit or lack a compile on any card class" {
    const a = std.testing.allocator;
    for (controls) |r| {
        for (limits.xe, 0..) |_, i| {
            var legal = r;
            legal.launch.threads = .{ 256, 1, 1 };
            legal.launch.slm = 0;
            legal.launch.sub_group = 16;
            legal.usage[i] = .{ .threads = .{ 256, 1, 1 }, .sub_group = 16, .grf = 256 };
            try std.testing.expectEqual(@as(usize, 1), try controlProblems(a, legal, true));
            if (!r.compiled) continue;
            var missing = r;
            missing.usage[i] = .{};
            try std.testing.expectEqual(@as(usize, 1), try controlProblems(a, missing, true));
        }
    }
}
