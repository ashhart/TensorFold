//! One collector for K3.1's experimental index operators and supplemental exact addressing checks.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const scores = @import("metal_kimi_index.zig");
const select = @import("metal_kimi_select.zig");
const attention = @import("metal_kimi_attention.zig");
pub const Suite = struct {
    a: std.mem.Allocator,
    index: *scores.Suite,
    selected: [2]?*select.Probe = .{ null, null },
    attn: [5]?*attention.Probe = @splat(null),
    pub fn create(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kernels: *std.ArrayList(contract.gates.Kernel), only: ?[]const u8) !*Suite {
        const s = try a.create(Suite);
        const index = scores.Suite.create(a, device, queue, kernels, only) catch |err| {
            a.destroy(s);
            return err;
        };
        s.* = .{ .a = a, .index = index };
        errdefer s.destroy();
        for ([_]bool{ false, true }, 0..) |control, i| {
            if (only) |text| if (std.mem.indexOf(u8, select.Probe.name(control), text) == null) continue;
            const p = try a.create(select.Probe);
            p.* = select.Probe.init(a, device, queue, control) catch |err| {
                a.destroy(p);
                return err;
            };
            s.selected[i] = p;
            try kernels.append(a, p.kernel());
        }
        for (0..5) |ctl| {
            if (only) |text| if (std.mem.indexOf(u8, attention.Probe.name(@intCast(ctl)), text) == null) continue;
            const p = try a.create(attention.Probe);
            p.* = attention.Probe.init(a, device, queue, @intCast(ctl)) catch |err| {
                a.destroy(p);
                return err;
            };
            s.attn[ctl] = p;
            try kernels.append(a, p.kernel());
        }
        return s;
    }
    pub fn destroy(s: *Suite) void {
        s.index.destroy();
        for (s.selected) |maybe| if (maybe) |p| {
            p.deinit();
            s.a.destroy(p);
        };
        for (s.attn) |maybe| if (maybe) |p| {
            p.deinit();
            s.a.destroy(p);
        };
        s.a.destroy(s);
    }
    pub fn exact(s: *Suite) !void {
        try s.index.invariance();
        if (s.attn[0]) |p| {
            for ([_]usize{ 1, 15, 16, 17, 511, 512, 513, 2047, 2048, 2049, 4097 }) |keys| {
                if (!try p.prepare(.{ .rows = 2, .n = 3 * 512, .k = keys }, .random, 973)) return error.IndexFixtureRefused;
                try p.denseBytes();
            }
            std.debug.print("PASS kimi index supplemental: selected and compacted dense MLA bytes equal at eleven boundaries\n", .{});
        }
    }
};
