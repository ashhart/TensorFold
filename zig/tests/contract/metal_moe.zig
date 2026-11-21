//! Ornith-1.5-122B's routed contract probes: experts' decode rows, routing and combining.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const q27_moe = @import("metal_qwen27_moe.zig");
const q27_rows = @import("metal_qwen27_moe_rows.zig");
const g = contract.gates;

pub const Moe = struct {
    extra: std.ArrayList(*q27_moe.Extra) = .empty,
    route: ?*q27_moe.Route = null,
    sort: ?*q27_rows.Sort = null,
    gather: ?*q27_rows.Gather = null,
    glu: ?*q27_rows.Glu = null,
    combine: std.ArrayList(*q27_rows.Combine) = .empty,

    pub fn collect(m: *Moe, a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, only: ?[]const u8, kernels: *std.ArrayList(g.Kernel)) !void {
        for ([_]u8{ 4, 8 }, [_][]const u8{ "metal qwen27 q27_rows_extra q4 g64", "metal qwen27 q27_rows_extra q8 g64" }) |bits, name| {
            if (!wanted(only, name)) continue;
            const x = try a.create(q27_moe.Extra);
            x.* = try q27_moe.Extra.init(a, device, queue, bits, name);
            try m.extra.append(a, x);
            try kernels.append(a, x.kernel());
        }
        if (wanted(only, q27_moe.Route.spec().name)) {
            m.route = try a.create(q27_moe.Route);
            m.route.?.* = try q27_moe.Route.init(a, device, queue);
            try kernels.append(a, m.route.?.kernel());
        }
        if (wanted(only, q27_rows.Sort.spec().name)) {
            m.sort = try a.create(q27_rows.Sort);
            m.sort.?.* = try q27_rows.Sort.init(a, device, queue);
            try kernels.append(a, m.sort.?.kernel());
        }
        if (wanted(only, q27_rows.Gather.spec().name)) {
            m.gather = try a.create(q27_rows.Gather);
            m.gather.?.* = try q27_rows.Gather.init(a, device, queue);
            try kernels.append(a, m.gather.?.kernel());
        }
        if (wanted(only, q27_rows.Glu.spec().name)) {
            m.glu = try a.create(q27_rows.Glu);
            m.glu.?.* = try q27_rows.Glu.init(a, device, queue);
            try kernels.append(a, m.glu.?.kernel());
        }
        for ([_]bool{ true, false }) |in_out| {
            if (!wanted(only, q27_rows.Combine.spec(in_out).name)) continue;
            const x = try a.create(q27_rows.Combine);
            x.* = try q27_rows.Combine.init(a, device, queue, in_out);
            try m.combine.append(a, x);
            try kernels.append(a, x.kernel());
        }
    }

    pub fn deinit(m: *Moe, a: std.mem.Allocator) void {
        for (m.extra.items) |x| x.deinit();
        m.extra.deinit(a);
        if (m.route) |x| x.deinit();
        if (m.sort) |x| x.deinit();
        if (m.gather) |x| x.deinit();
        if (m.glu) |x| x.deinit();
        for (m.combine.items) |x| x.deinit();
        m.combine.deinit(a);
    }
};

fn wanted(only: ?[]const u8, name: []const u8) bool {
    const prefix = only orelse return true;
    return std.mem.indexOf(u8, name, prefix) != null;
}
