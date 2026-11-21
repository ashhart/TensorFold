//! The Qwen27 replacements join the normal Metal SUMMARY, with a red control for each variant.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const affine = @import("metal_affine.zig");
const columns = @import("metal_delta_slots.zig").Columns;
const sums = @import("metal_q27_sums.zig");
const glue = @import("metal_q27_glue.zig");
const g = contract.gates;
const shapes = [_]contract.spec.Shape{
    .{ .rows = 16, .n = 64, .k = 6144 }, .{ .rows = 1, .n = 248320, .k = 64 },
    .{ .rows = 16, .n = 64, .k = 6144 }, .{ .rows = 1, .n = 248320, .k = 64 },
    .{ .rows = 16, .n = 64, .k = 5120 }, .{ .rows = 49, .n = 32, .k = 17408 },
    .{ .rows = 16, .n = 64, .k = 6144 }, .{ .rows = 1, .n = 248320, .k = 64 },
    .{ .rows = 128, .n = 32, .k = 64 },  .{ .rows = 2048, .n = 32, .k = 64 },
};
pub const Suite = struct {
    projections: std.ArrayList(*affine.Proj) = .empty,
    scans: std.ArrayList(*columns.Probe) = .empty,
    glues: std.ArrayList(*glue.Probe) = .empty,
    precision: std.ArrayList(*sums.Probe) = .empty,
    names: std.ArrayList([]u8) = .empty,
    fn label(s: *Suite, a: std.mem.Allocator, comptime fmt: []const u8, args: anytype) ![]const u8 {
        const name = try std.fmt.allocPrint(a, fmt, args);
        try s.names.append(a, name);
        return name;
    }
    fn wanted(only: ?[]const u8, name: []const u8) bool {
        return if (only) |filter| std.mem.indexOf(u8, name, filter) != null else true;
    }
    pub fn collect(s: *Suite, a: std.mem.Allocator, d: mtl.Device, q: mtl.Queue, only: ?[]const u8, kernels: *std.ArrayList(g.Kernel)) !void {
        try s.scans.ensureUnusedCapacity(a, 9);
        try s.glues.ensureUnusedCapacity(a, 6);
        try s.projections.ensureUnusedCapacity(a, 16);
        for ([_]usize{ 1, 2, 4 }) |slots| for (0..3) |fault| {
            const bad = fault == 1;
            const precision_bad = fault == 2;
            const name = try s.label(a, "{s}core deltanet columns{d}{s}", .{ if (bad or precision_bad) "control: " else "", slots, if (precision_bad) " half sums" else "" });
            if (!wanted(only, name)) continue;
            const p = try a.create(columns.Probe);
            errdefer a.destroy(p);
            p.* = try columns.Probe.init(a, d, q, slots, bad, precision_bad, name);
            errdefer p.deinit();
            try kernels.append(a, p.kernel());
            s.scans.appendAssumeCapacity(p);
        };
        for (std.enums.values(glue.Kind)) |kind| for ([_]bool{ false, true }) |bad| {
            const name = try s.label(a, "{s}qwen27 flat {t}", .{ if (bad) "control: " else "", kind });
            if (!wanted(only, name)) continue;
            const p = try a.create(glue.Probe);
            errdefer a.destroy(p);
            p.* = glue.Probe.init(a, d, q, kind, bad, name);
            errdefer p.deinit();
            try kernels.append(a, p.kernel());
            s.glues.appendAssumeCapacity(p);
        };
        try s.precision.ensureUnusedCapacity(a, 6);
        for ([_]contract.q27_stack_sums.Kind{ .conv, .norm, .xs }) |kind| for ([_]bool{ false, true }) |bad| {
            const name = try s.label(a, "{s}qwen27 flat sums {t}", .{ if (bad) "control: " else "", kind });
            if (!wanted(only, name)) continue;
            const p = try a.create(sums.Probe);
            errdefer a.destroy(p);
            p.* = sums.Probe.init(a, d, q, kind, bad, name);
            errdefer p.deinit();
            try kernels.append(a, p.kernel());
            s.precision.appendAssumeCapacity(p);
        };
        if (d.tensorUnits()) for (1..9) |sk| for ([_]bool{ false, true }) |bad| {
            const name = try s.label(a, "{s}metal qk q4 g64 sk{d}", .{ if (bad) "control: " else "", sk });
            if (!wanted(only, name)) continue;
            const p = try a.create(affine.Proj);
            errdefer a.destroy(p);
            p.* = try affine.Proj.init(a, d, q, .{ .kind = .qk, .sk = sk, .control = bad, .served = &shapes }, name);
            errdefer p.deinit();
            try kernels.append(a, p.kernel());
            s.projections.appendAssumeCapacity(p);
        };
    }
    pub fn deinit(s: *Suite, a: std.mem.Allocator) void {
        for (s.precision.items) |p| {
            p.deinit();
            a.destroy(p);
        }
        s.precision.deinit(a);
        for (s.precision.items) |p| {
            p.deinit();
            a.destroy(p);
        }
        s.precision.deinit(a);
        for (s.projections.items) |p| {
            p.deinit();
            a.destroy(p);
        }
        s.projections.deinit(a);
        for (s.scans.items) |p| {
            p.deinit();
            a.destroy(p);
        }
        s.scans.deinit(a);
        for (s.glues.items) |p| {
            p.deinit();
            a.destroy(p);
        }
        s.glues.deinit(a);
        for (s.names.items) |name| a.free(name);
        s.names.deinit(a);
    }
};
