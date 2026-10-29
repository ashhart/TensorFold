//! Native HC probes launch the production geometry adapter and compile real-source fault variants separately.
const std = @import("std");
const metal = @import("metal");
const family = @import("qwen4").family;
const contract = @import("contract");
const common = family.hc_native_contract;
const api = family.metal_hc.Kernels(metal);
const Backend = struct {
    pub const Rig = @import("rig.zig").Rig;
    kernels: *api,
    views: [4]api.View = undefined,
    count: usize = 0,
    pub fn bind(b: *Backend, r: *Rig, f: common.Fixture) !void {
        for (f.inputs[0..f.count], 0..) |bytes, i| {
            const ref = try r.input(bytes);
            b.views[i] = .{ .buffer = ref.buf, .offset = ref.off, .len = bytes.len };
        }
        const ref = try r.output(f.output_bytes);
        b.views[f.count] = .{ .buffer = ref.buf, .offset = ref.off, .len = f.output_bytes };
        b.count = f.count + 1;
    }
    pub fn launch(b: *Backend, r: *Rig, op: family.hc_layout.Op, p: family.hc_layout.Call, repeat: u32) !void {
        const pass = r.begin();
        b.kernels.testEncode(pass.e, op, p, b.views[0..b.count], repeat) catch |err| {
            try r.finish(pass);
            return err;
        };
        try r.finish(pass);
    }
};
pub const Case = common.Case(contract, Backend);
pub const Suite = struct {
    a: std.mem.Allocator,
    pipes: [5]api = undefined,
    created: usize = 0,
    cases: [17]Case = undefined,
    made: usize = 0,
    pub fn create(a: std.mem.Allocator, device: metal.Device, queue: metal.Queue, out: *std.ArrayList(contract.gates.Kernel), only: ?[]const u8) !*Suite {
        const s = try a.create(Suite);
        s.* = .{ .a = a };
        errdefer s.destroy();
        for (std.enums.values(common.Control)) |control| {
            s.pipes[s.created] = try api.loadControl(a, device, .{ .hidden = 5120, .streams = 4 }, control);
            const pipe = &s.pipes[s.created];
            s.created += 1;
            for (std.enums.values(family.hc_layout.Op)) |op| {
                if (control == .sums and op != .sum) continue;
                const case = &s.cases[s.made];
                case.* = .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .backend = .{ .kernels = pipe }, .op = op, .control = control };
                s.made += 1;
                const kernel = case.kernel();
                if (only) |needle| if (std.mem.indexOf(u8, kernel.spec.name, needle) == null) continue;
                try out.append(a, kernel);
            }
        }
        return s;
    }
    pub fn destroy(s: *Suite) void {
        for (s.cases[0..s.made]) |*case| case.deinit();
        for (s.pipes[0..s.created]) |*pipe| pipe.deinit();
        const a = s.a;
        a.destroy(s);
    }
};
