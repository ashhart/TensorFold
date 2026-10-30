//! Both HC activation roles reach the production Metal adapter through guarded contract buffers.
const std = @import("std");
const metal = @import("metal");
const tf = @import("tensorfold");
const sources = @import("kernel_sources");
const contract = @import("contract");
const common = tf.hc_activation_contract;
const api = tf.hc_activation_metal_adapter(tf.hc_activation, metal);
const Backend = struct {
    pub const name = "metal";
    pub const Rig = @import("rig.zig").Rig;
    kernels: *api.Kernels,
    views: [2]api.View = undefined,
    pub fn bind(b: *Backend, r: *Rig, f: common.Fixture) !void {
        const x = try r.input(std.mem.sliceAsBytes(f.x));
        const y = try r.output(f.x.len * 2);
        b.views = .{ .{ .buffer = x.buf, .offset = x.off, .len = f.x.len * 2 }, .{ .buffer = y.buf, .offset = y.off, .len = f.x.len * 2 } };
    }
    pub fn launch(b: *Backend, r: *Rig, p: tf.hc_activation.Params, control: common.Control, repeat: c_int) !void {
        const pass = r.begin();
        b.kernels.testEncode(pass.e, p, b.views[0], b.views[1], @fromBackingInt(@backingInt(control)), repeat) catch |err| {
            try r.finish(pass);
            return err;
        };
        try r.finish(pass);
    }
};
pub const Case = common.Case(contract, Backend);
pub const Kernels = api.Kernels;
pub const count = 7 * 2 * std.enums.values(common.Control).len;
pub fn text(a: std.mem.Allocator) ![]u8 {
    return api.source(a, sources.core_weight_only_body, sources.core_hc_activation_body, sources.core_hc_activation, true);
}
pub fn collect(a: std.mem.Allocator, device: metal.Device, queue: metal.Queue, k: *Kernels, cases: *[count]Case, out: *std.ArrayList(contract.gates.Kernel), only: ?[]const u8) !void {
    var i: usize = 0;
    for (2..9) |streams| for ([_]bool{ false, true }) |injection| for (std.enums.values(common.Control)) |control| {
        cases[i] = .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .backend = .{ .kernels = k }, .streams = @intCast(streams), .injection = injection, .control = control };
        i += 1;
    };
    for (cases) |*case| {
        const kernel = case.kernel();
        if (only) |needle| if (std.mem.indexOf(u8, kernel.spec.name, needle) == null) continue;
        try out.append(a, kernel);
    }
}
