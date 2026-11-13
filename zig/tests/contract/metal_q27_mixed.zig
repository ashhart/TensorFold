//! The changed head norm and both cache append layouts share one scoped contract collector.
const std = @import("std");
const mtl = @import("metal");
const ct = @import("contract");
const norm = @import("metal_q27_mixed_norm.zig");
const append = @import("metal_q27_mixed_append.zig");
pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(ct.gates.Kernel),
    norms: std.ArrayList(*norm.Probe) = .empty,
    appends: std.ArrayList(*append.Probe) = .empty,
    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        for ([_]bool{ false, true }) |bad| {
            if (only == null or std.mem.indexOf(u8, norm.label(bad), only.?) != null) {
                const p = try s.a.create(norm.Probe);
                errdefer s.a.destroy(p);
                p.* = try norm.Probe.init(s.a, s.device, s.queue, bad);
                try s.norms.append(s.a, p);
                try s.kernels.append(s.a, p.kernel());
            }
            for ([_]bool{ false, true }) |pages| {
                if (only) |f| if (std.mem.indexOf(u8, append.label(pages, bad), f) == null) continue;
                const p = try s.a.create(append.Probe);
                errdefer s.a.destroy(p);
                p.* = try append.Probe.init(s.a, s.device, s.queue, pages, bad);
                try s.appends.append(s.a, p);
                try s.kernels.append(s.a, p.kernel());
            }
        }
    }
    pub fn deinit(s: *Suite) void {
        for (s.norms.items) |p| {
            p.deinit();
            s.a.destroy(p);
        }
        for (s.appends.items) |p| {
            p.deinit();
            s.a.destroy(p);
        }
        s.norms.deinit(s.a);
        s.appends.deinit(s.a);
    }
};
