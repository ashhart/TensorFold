//! Native recurrence probes cover readouts and every kept F32 row through the production Metal launch owner.
const std = @import("std");
const metal = @import("metal");
const family = @import("qwen4").family;
const contract = @import("contract");
const common = family.gdn_contract;
const api = family.metal_gdn.Kernels(metal);
const Backend = struct {
    pub const name = "metal";
    pub const Rig = @import("rig.zig").Rig;
    pub const served: []const contract.spec.Shape = &.{ .{ .rows = 1, .n = 1, .k = 2 }, .{ .rows = 16, .n = 1, .k = 2 }, .{ .rows = 1, .n = 16, .k = 48 }, .{ .rows = 16, .n = 16, .k = 48 } };
    a: std.mem.Allocator,
    device: metal.Device,
    kernels: ?api = null,
    views: [10]api.View = undefined,
    pub fn deinit(b: *Backend) void {
        if (b.kernels) |*k| k.deinit();
    }
    pub fn bind(b: *Backend, r: *Rig, f: common.Fixture, control: common.Control) !void {
        if (b.kernels) |*k| if (k.keys != f.p.keys or k.values != f.p.values or k.gate != f.p.gate or k.control != control) {
            k.deinit();
            b.kernels = null;
        };
        if (b.kernels == null) {
            const text = try family.metal_gdn.source(b.a, f.p.keys, f.p.values, f.p.gate);
            defer b.a.free(text);
            b.kernels = try api.loadControl(b.a, b.device, text, f.p.keys, f.p.values, f.p.gate, control);
        }
        for ([_]usize{ 0, 1, 3, 2, 4, 5, 6 }, 0..) |from, i| {
            const bytes = f.inputs[from];
            const ref = try r.input(bytes);
            b.views[i] = .{ .buffer = ref.buf, .offset = ref.off, .len = bytes.len };
        }
        const sizes = try family.cuda_gdn.sizes(f.p);
        const lengths = [_]usize{ sizes.output, f.p.rows * sizes.conv_state, sizes.snapshots };
        for (lengths, 7..) |len, i| {
            const selected = (i == 7 and f.part == .readout) or (i == 9 and f.part == .state) or (i == 8 and f.part == .conv);
            const ref = if (selected) try r.output(len) else try r.scratch(len);
            b.views[i] = .{ .buffer = ref.buf, .offset = ref.off, .len = len };
        }
    }
    pub fn launch(b: *Backend, r: *Rig, f: common.Fixture, repeat: u32) !void {
        const pass = r.begin();
        b.kernels.?.testEncode(pass.e, f.p, b.views, repeat) catch |err| {
            try r.finish(pass);
            return err;
        };
        try r.finish(pass);
    }
};
pub const Case = common.Case(contract, Backend);
pub const Suite = struct {
    a: std.mem.Allocator,
    cases: [60]Case = undefined,
    made: usize = 0,
    pub fn create(a: std.mem.Allocator, device: metal.Device, queue: metal.Queue, out: *std.ArrayList(contract.gates.Kernel), only: ?[]const u8) !*Suite {
        const s = try a.create(Suite);
        s.* = .{ .a = a };
        errdefer s.destroy();
        for ([_]family.cuda_gdn.Gate{ .silu, .sigmoid }) |gate| for ([_]u4{ 3, 7, 11, 15 }) |flags| for (std.enums.values(common.Part)) |part| {
            if (part == .state and flags != 3 and flags != 11) continue;
            if (part == .conv and flags != 11) continue;
            for (std.enums.values(common.Control)) |control| {
                if (control == .sums and (flags != 11 or part == .conv)) continue;
                const case = &s.cases[s.made];
                case.* = .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .backend = .{ .a = a, .device = device }, .gate = gate, .flags = flags, .part = part, .control = control };
                s.made += 1;
                const kernel = case.kernel();
                if (only) |needle| if (std.mem.indexOf(u8, kernel.spec.name, needle) == null) continue;
                try out.append(a, kernel);
            }
        };
        std.debug.assert(s.made == s.cases.len);
        return s;
    }
    pub fn destroy(s: *Suite) void {
        for (s.cases[0..s.made]) |*case| case.deinit();
        const a = s.a;
        a.destroy(s);
    }
};
