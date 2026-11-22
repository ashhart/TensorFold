//! Nemotron 3.5 Bolt's whole Metal contract behind one hook: its served kernels, then a control for each planted fault.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const bolt = @import("metal_bolt.zig");
const norm = @import("metal_bolt_norm.zig");
const subset = @import("metal_bolt_subset.zig");
const mamba = @import("metal_bolt_mamba.zig");
const scan = @import("metal_bolt_scan.zig");
const g = contract.gates;

fn wanted(only: ?[]const u8, name: []const u8) bool {
    const text = only orelse return true;
    return std.mem.indexOf(u8, name, text) != null;
}

/// Every Bolt probe, kept where it was made: a probe's address is its kernel's context.
pub const Suite = struct {
    a: std.mem.Allocator,
    gemvs: [4]bolt.GemvProbe = undefined,
    gemv_made: usize = 0,
    subset_probe: ?subset.Probe = null,
    routes: [3]bolt.RouteProbe = undefined,
    route_made: usize = 0,
    combines: [2]bolt.CombineProbe = undefined,
    combine_made: usize = 0,
    norms: [7]norm.NormProbe = undefined,
    norm_made: usize = 0,
    convs: [4]mamba.ConvProbe = undefined,
    conv_made: usize = 0,
    scans: [4]scan.ScanProbe = undefined,
    scan_made: usize = 0,

    const Kernels = std.ArrayList(g.Kernel);

    /// The served kernels in the order main held them, then each control after the kernels it mutates.
    pub fn create(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kernels: *Kernels, only: ?[]const u8) !*Suite {
        const s = try a.create(Suite);
        s.* = .{ .a = a };
        errdefer s.destroy();
        const bc = bolt.controls;
        const mc = mamba.controls;
        for (std.enums.values(bolt.Gemv)) |v| try s.gemv(device, queue, kernels, only, v, null);
        if (wanted(only, subset.spec().name)) {
            s.subset_probe = try subset.Probe.init(a, device, queue);
            try kernels.append(a, s.subset_probe.?.kernel());
        }
        try s.route(device, queue, kernels, only, null);
        try s.combine(device, queue, kernels, only, null);
        for (std.enums.values(norm.Norm)) |v| try s.normOf(device, queue, kernels, only, v, null);
        for ([_]bolt.Mut{ bc.gemv_half, bc.gemv_tail }) |mut| try s.gemv(device, queue, kernels, only, .bm4, mut);
        for ([_]bolt.Mut{ bc.route_slice, bc.route_nan }) |mut| try s.route(device, queue, kernels, only, mut);
        try s.combine(device, queue, kernels, only, bc.combine_ninth);
        try s.normOf(device, queue, kernels, only, .add, bc.add_residual);
        for ([_]bolt.Mut{ bc.group_sums, bc.group_mean }) |mut| try s.normOf(device, queue, kernels, only, .group16, mut);
        for ([_]?bolt.Mut{ null, mc.conv_tap, mc.conv_state, mc.conv_half }) |mut| {
            if (!wanted(only, if (mut) |x| x.name else mamba.convSpec().name)) continue;
            s.convs[s.conv_made] = try mamba.ConvProbe.init(a, device, queue, mut);
            s.conv_made += 1;
            try kernels.append(a, s.convs[s.conv_made - 1].kernel());
        }
        for ([_]?bolt.Mut{ null, mc.scan_skip, mc.scan_decay, mc.scan_half }) |mut| {
            if (!wanted(only, if (mut) |x| x.name else mamba.scanSpec().name)) continue;
            s.scans[s.scan_made] = try scan.ScanProbe.init(a, device, queue, mut);
            s.scan_made += 1;
            try kernels.append(a, s.scans[s.scan_made - 1].kernel());
        }
        return s;
    }

    fn gemv(s: *Suite, device: mtl.Device, queue: mtl.Queue, kernels: *Kernels, only: ?[]const u8, v: bolt.Gemv, mut: ?bolt.Mut) !void {
        if (!wanted(only, if (mut) |x| x.name else bolt.gemvSpec(v).name)) return;
        s.gemvs[s.gemv_made] = try bolt.GemvProbe.init(s.a, device, queue, v, mut);
        s.gemv_made += 1;
        try kernels.append(s.a, s.gemvs[s.gemv_made - 1].kernel());
    }

    fn route(s: *Suite, device: mtl.Device, queue: mtl.Queue, kernels: *Kernels, only: ?[]const u8, mut: ?bolt.Mut) !void {
        if (!wanted(only, if (mut) |x| x.name else bolt.routeSpec().name)) return;
        s.routes[s.route_made] = try bolt.RouteProbe.init(s.a, device, queue, mut);
        s.route_made += 1;
        try kernels.append(s.a, s.routes[s.route_made - 1].kernel());
    }

    fn combine(s: *Suite, device: mtl.Device, queue: mtl.Queue, kernels: *Kernels, only: ?[]const u8, mut: ?bolt.Mut) !void {
        if (!wanted(only, if (mut) |x| x.name else bolt.combineSpec().name)) return;
        s.combines[s.combine_made] = try bolt.CombineProbe.init(s.a, device, queue, mut);
        s.combine_made += 1;
        try kernels.append(s.a, s.combines[s.combine_made - 1].kernel());
    }

    fn normOf(s: *Suite, device: mtl.Device, queue: mtl.Queue, kernels: *Kernels, only: ?[]const u8, v: norm.Norm, mut: ?bolt.Mut) !void {
        if (!wanted(only, if (mut) |x| x.name else norm.normSpec(v).name)) return;
        s.norms[s.norm_made] = try norm.NormProbe.init(s.a, device, queue, v, mut);
        s.norm_made += 1;
        try kernels.append(s.a, s.norms[s.norm_made - 1].kernel());
    }

    pub fn destroy(s: *Suite) void {
        for (s.gemvs[0..s.gemv_made]) |*p| p.deinit();
        if (s.subset_probe) |*p| p.deinit();
        for (s.routes[0..s.route_made]) |*p| p.deinit();
        for (s.combines[0..s.combine_made]) |*p| p.deinit();
        for (s.norms[0..s.norm_made]) |*p| p.deinit();
        for (s.convs[0..s.conv_made]) |*p| p.deinit();
        for (s.scans[0..s.scan_made]) |*p| p.deinit();
        s.a.destroy(s);
    }
};
