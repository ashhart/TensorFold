//! The E4B's per-layer kernels metal_main registers: each built when its name matches `only`, kept at a fixed address.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const ple = @import("metal_gemma4_ple.zig");
const g = contract.gates;

pub const limits = ple.limits;

pub const Set = struct {
    dense: ?ple.Dense = null,
    gate: ?ple.Dense = null,
    embed: ?ple.Embed = null,
    combine: ?ple.Combine = null,
    control: ?ple.Dense = null,

    pub fn register(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, only: ?[]const u8, kernels: *std.ArrayList(g.Kernel)) !*Set {
        const s = try a.create(Set);
        s.* = .{};
        errdefer s.deinit(a);
        if (wanted(only, ple.dense_name)) {
            s.dense = try ple.Dense.init(a, device, queue, false);
            try kernels.append(a, s.dense.?.kernel());
        }
        if (wanted(only, ple.gate_name)) {
            s.gate = try ple.Dense.init(a, device, queue, true);
            try kernels.append(a, s.gate.?.kernel());
        }
        if (wanted(only, ple.embed_name)) {
            s.embed = try ple.Embed.init(a, device, queue);
            try kernels.append(a, s.embed.?.kernel());
        }
        if (wanted(only, ple.combine_name)) {
            s.combine = try ple.Combine.init(a, device, queue);
            try kernels.append(a, s.combine.?.kernel());
        }
        if (wanted(only, ple.control_name)) {
            s.control = try ple.Dense.initControl(a, device, queue);
            try kernels.append(a, s.control.?.kernel());
        }
        return s;
    }

    pub fn deinit(s: *Set, a: std.mem.Allocator) void {
        if (s.dense) |*x| x.deinit();
        if (s.gate) |*x| x.deinit();
        if (s.embed) |*x| x.deinit();
        if (s.combine) |*x| x.deinit();
        if (s.control) |*x| x.deinit();
        a.destroy(s);
    }
};

fn wanted(only: ?[]const u8, name: []const u8) bool {
    const text = only orelse return true;
    return std.mem.indexOf(u8, name, text) != null;
}
