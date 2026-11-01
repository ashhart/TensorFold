//! Idol's background remover's CUDA kernels under the contract, with fp16 or NaN-dropping controls.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const cm = @import("contract_birefnet_common.zig");
const cv = @import("contract_birefnet_conv.zig");
const nm = @import("contract_birefnet_norm.zig");
const rw = @import("contract_birefnet_rows.zig");
const wn = @import("contract_birefnet_window.zig");

const g = cm.g;
const ks = cm.ks;

const controls_image = struct {
    pub const bytes align(16) = @embedFile("fatbin_contract_controls").*;
};

const Control = cm.Control;
const controls = cm.controls;

/// A test the set made, and how it goes.
const Made = struct { ptr: *anyopaque, free: *const fn (std.mem.Allocator, *anyopaque) void };

/// Every registration, on the heap: the gates hold pointers to them.
pub const Set = struct {
    gpa: std.mem.Allocator,
    k: ks.Kernels,
    ctl: cuda.Module,
    /// The kernels with one function swapped for a control's, one copy a control.
    swapped: [controls.len]ks.Kernels = undefined,
    made: std.ArrayList(Made) = .empty,
    names: std.ArrayList([]u8) = .empty,

    fn make(set: *Set, comptime T: type, value: T) !*T {
        const t = try set.gpa.create(T);
        errdefer set.gpa.destroy(t);
        t.* = value;
        const f = struct {
            fn free(a: std.mem.Allocator, p: *anyopaque) void {
                const x: *T = @ptrCast(@alignCast(p));
                x.deinit();
                a.destroy(x);
            }
        };
        try set.made.append(set.gpa, .{ .ptr = t, .free = f.free });
        return t;
    }

    fn label(set: *Set, comptime fmt: []const u8, args: anytype) ![]const u8 {
        const name = try std.fmt.allocPrint(set.gpa, fmt, args);
        errdefer set.gpa.free(name);
        try set.names.append(set.gpa, name);
        return name;
    }

    /// Each kernel on `k` (or, with `ctl`, the one it breaks), appended to `list`.
    fn add(set: *Set, gpu: check.Gpu, s: cuda.Stream, k: *const ks.Kernels, ctl: ?Control, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !void {
        const rig: cm.rig_.Rig = .{ .gpu = gpu, .s = s };
        const gate: ?g.Gate = if (ctl) |x| x.gate else null;
        const W = struct {
            set: *Set,
            ctl: ?Control,
            fn name(w: @This(), kernel: []const u8) !?[]const u8 {
                if (w.ctl) |x| {
                    if (!std.mem.eql(u8, x.label, kernel)) return null;
                    return try w.set.label("control: cuda idol birefnet {s} {s}", .{ kernel, x.what });
                }
                return try w.set.label("cuda idol birefnet {s}", .{kernel});
            }
        };
        const w = W{ .set = set, .ctl = ctl };
        const gp = set.gpa;
        for (cv.forms) |f| {
            const name = (try w.name(f.label)) orelse continue;
            const t = try set.make(cv.Conv, .{ .a = gp, .rig = rig, .k = k, .form = f, .name = name });
            try list.append(a, t.kernel(gate));
        }
        if (try w.name("tf_br_conv3")) |name| try list.append(a, (try set.make(cv.Conv3, .{ .a = gp, .rig = rig, .k = k, .name = name })).kernel(gate));
        for ([_]nm.NormForm{ .rows, .map, .merge }, [_][]const u8{ "tf_br_norm", "tf_br_norm map", "tf_br_norm merge" }) |f, l| {
            const name = (try w.name(l)) orelse continue;
            try list.append(a, (try set.make(nm.Norm, .{ .a = gp, .rig = rig, .k = k, .form = f, .name = name })).kernel(gate));
        }
        for ([_]usize{ 0, 6 }, [_][]const u8{ "tf_br_window", "tf_br_window shift 6" }) |shift, l| {
            const name = (try w.name(l)) orelse continue;
            try list.append(a, (try set.make(wn.Window, .{ .a = gp, .rig = rig, .k = k, .shift = shift, .name = name })).kernel(gate));
        }
        if (try w.name("tf_br_resize")) |name| try list.append(a, (try set.make(rw.Resize, .{ .a = gp, .rig = rig, .k = k, .name = name, .ends = .{ .a = gp } })).kernel(gate));
        for ([_]bool{ false, true }, [_][]const u8{ "tf_br_tiles", "tf_br_tiles grid" }) |grid, l| {
            const name = (try w.name(l)) orelse continue;
            try list.append(a, (try set.make(rw.Tiles, .{ .a = gp, .rig = rig, .k = k, .grid = grid, .name = name })).kernel(gate));
        }
        for ([_]usize{ 1, 3, 7 }, [_][]const u8{ "tf_br_deform 1x1", "tf_br_deform 3x3", "tf_br_deform 7x7" }) |side, l| {
            const name = (try w.name(l)) orelse continue;
            try list.append(a, (try set.make(nm.Deform, .{ .a = gp, .rig = rig, .k = k, .side = side, .name = name, .ends = .{ .a = gp } })).kernel(gate));
        }
        if (try w.name("tf_br_pool")) |name| try list.append(a, (try set.make(rw.Pool, .{ .a = gp, .rig = rig, .k = k, .name = name })).kernel(gate));
        if (try w.name("tf_br_gate")) |name| try list.append(a, (try set.make(rw.Gate, .{ .a = gp, .rig = rig, .k = k, .name = name })).kernel(gate));
        if (try w.name("tf_br_head")) |name| try list.append(a, (try set.make(rw.Head, .{ .a = gp, .rig = rig, .k = k, .name = name })).kernel(gate));
    }

    /// The set, its kernels appended to `list` (the controls last).
    pub fn init(gpu: check.Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !*Set {
        const set = try gpu.gpa.create(Set);
        errdefer gpu.gpa.destroy(set);
        var k = try ks.Kernels.load(gpu.d);
        errdefer k.unload();
        var ctl = try cuda.Module.load(gpu.d, &controls_image.bytes);
        errdefer ctl.unload();
        set.* = .{ .gpa = gpu.gpa, .k = k, .ctl = ctl };
        errdefer set.free();
        for (&set.swapped, controls) |*sw, x| {
            sw.* = k;
            inline for (ks.names) |n| if (std.mem.eql(u8, n, x.field)) {
                @field(sw.*, n) = try ctl.function(x.symbol);
            };
        }
        try set.add(gpu, s, &set.k, null, a, list);
        for (&set.swapped, controls) |*sw, x| try set.add(gpu, s, sw, x, a, list);
        return set;
    }

    fn free(set: *Set) void {
        for (set.made.items) |m| m.free(set.gpa, m.ptr);
        set.made.deinit(set.gpa);
        for (set.names.items) |n| set.gpa.free(n);
        set.names.deinit(set.gpa);
    }

    pub fn deinit(set: *Set) void {
        set.free();
        set.ctl.unload();
        set.k.unload();
        set.gpa.destroy(set);
    }
};

/// The background remover's kernels' gates 1-4 and their controls alone.
pub fn run(gpu: check.Gpu) !void {
    const a = gpu.gpa;
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var list: std.ArrayList(g.Kernel) = .empty;
    defer list.deinit(a);
    const set = try Set.init(gpu, s, a, &list);
    defer set.deinit();
    var t: g.Tally = .{};
    try g.runAll(a, list.items, 0xb12e, &t);
    std.debug.print("SUMMARY idol birefnet: {d} kernels, {d} pass gates 1-4, {d} fail; {d} controls: {d} red as planted, {d} missed\n", .{ t.kernels, t.passed, t.failed, t.controls, t.caught, t.missed });
    try check.expect(t.problems() == 0, "idol-birefnet-contract: {d} gate failures, {d} controls missed", .{ t.failed, t.missed });
}
