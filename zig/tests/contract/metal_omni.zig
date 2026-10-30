//! LFM3-Omni's Metal kernels under the contract: every pipeline of lfm3_omni.metallib, each with a planted control.
const std = @import("std");
const mtl = @import("metal");
const k = @import("metal_omni_common.zig");
const rig_ = @import("rig.zig");

const g = k.g;

/// Every registration: norms and rotary, convs and gathers, elementwise, picks, attention, products.
pub const all = @import("metal_omni_norms.zig").entries ++ @import("metal_omni_conv.zig").entries ++ @import("metal_omni_elem.zig").entries ++ @import("metal_omni_pick.zig").entries ++ @import("metal_omni_attn.zig").entries ++ @import("metal_omni_gemm.zig").entries;

/// A gate 5 label's source ("lfm3 omni <file> <function>"), its includes inlined.
pub fn limitsSource(a: std.mem.Allocator, label: []const u8) ![]u8 {
    var it = std.mem.tokenizeScalar(u8, label, ' ');
    _ = it.next();
    _ = it.next();
    const word = it.next() orelse return error.UnknownLaunch;
    const file = std.meta.stringToEnum(k.File, word) orelse return error.UnknownLaunch;
    return k.text(a, file);
}

pub const Suite = struct {
    a: std.mem.Allocator,
    probes: std.ArrayList(*k.Probe) = .empty,
    libs: std.ArrayList(mtl.Library) = .empty,
    names: std.ArrayList([]u8) = .empty,

    fn keep(o: ?[]const u8, name: []const u8) bool {
        return if (o) |needle| std.mem.indexOf(u8, name, needle) != null else true;
    }

    fn add(s: *Suite, device: mtl.Device, queue: mtl.Queue, lib: mtl.Library, e: k.Entry, name: []const u8, control: ?g.Gate, out: *std.ArrayList(g.Kernel)) !void {
        const pipe = try mtl.Pipeline.init(device, lib, e.pipe, false);
        errdefer pipe.deinit();
        const p = try s.a.create(k.Probe);
        errdefer s.a.destroy(p);
        var spec = e.spec;
        spec.name = name;
        p.* = .{ .a = s.a, .rig = .{ .a = s.a, .device = device, .queue = queue }, .pipe = pipe, .def = e.def, .spec_ = spec, .variant = e.variant };
        try s.probes.append(s.a, p);
        try out.append(s.a, p.kernel(control));
    }

    pub fn create(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, out: *std.ArrayList(g.Kernel), only: ?[]const u8) !*Suite {
        const s = try a.create(Suite);
        s.* = .{ .a = a };
        errdefer s.destroy();
        const tensor = device.tensorUnits();
        var served: [4]?mtl.Library = @splat(null);
        defer for (served) |l| if (l) |x| x.deinit();
        for (all) |e| {
            if (e.file == .nax and !tensor) {
                if (keep(only, e.spec.name)) std.debug.print("UNSUPPORTED {s}: tensor units required; the simdgroup products serve this GPU\n", .{e.spec.name});
                continue;
            }
            if (keep(only, e.spec.name)) {
                const i = @intFromEnum(e.file);
                if (served[i] == null) served[i] = try k.compile(a, device, e.file, null);
                try s.add(device, queue, served[i].?, e, e.spec.name, null, out);
            }
            const ctl = e.control orelse continue;
            const name = try std.fmt.allocPrint(a, "{s} control: {s}", .{ e.spec.name, e.what });
            try s.names.append(a, name);
            if (!keep(only, name)) continue;
            const lib = try k.compile(a, device, e.file, ctl);
            try s.libs.append(a, lib);
            try s.add(device, queue, lib, e, name, .shapes, out);
        }
        return s;
    }

    pub fn destroy(s: *Suite) void {
        for (s.probes.items) |p| {
            p.pipe.deinit();
            p.deinit();
            s.a.destroy(p);
        }
        s.probes.deinit(s.a);
        for (s.libs.items) |l| l.deinit();
        s.libs.deinit(s.a);
        for (s.names.items) |n| s.a.free(n);
        s.names.deinit(s.a);
        s.a.destroy(s);
    }
};
