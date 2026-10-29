//! Both pointwise policies reach production Metal pipelines with guarded device-owned views.
const std = @import("std");
const metal = @import("metal");
const tf = @import("tensorfold");
const contract = @import("contract");
const sources = @import("kernel_sources");
const common = tf.hc_pointwise_contract;
const api = tf.hc_pointwise_metal_adapter(tf.hc_pointwise, metal);
const Backend = struct {
    pub const name = "metal";
    pub const Rig = @import("rig.zig").Rig;
    kernels: *api.Kernels,
    views: [4]api.View = undefined,
    pub fn bind(b: *Backend, r: *Rig, f: common.Fixture) !void {
        for ([_][]const u8{ std.mem.sliceAsBytes(f.x), std.mem.sliceAsBytes(f.b), std.mem.sliceAsBytes(f.g) }, 0..) |bytes, i| {
            const ref = try r.input(bytes);
            b.views[i] = .{ .buffer = ref.buf, .offset = ref.off, .len = bytes.len };
        }
        const ref = try r.output(f.x.len * 2);
        b.views[3] = .{ .buffer = ref.buf, .offset = ref.off, .len = f.x.len * 2 };
    }
    pub fn launch(b: *Backend, r: *Rig, p: tf.hc_pointwise.Params, control: common.Control, repeat: c_int) !void {
        const pass = r.begin();
        b.kernels.testEncode(pass.e, p, b.views, @fromBackingInt(@backingInt(control)), repeat) catch |err| {
            try r.finish(pass);
            return err;
        };
        try r.finish(pass);
    }
};
pub const Case = common.Case(contract, Backend);
pub const Kernels = api.Kernels;
pub const count = common.variants.len * std.enums.values(common.Control).len;
pub fn text(a: std.mem.Allocator) ![]u8 {
    return api.source(a, sources.core_weight_only_body, sources.core_hc_activation_body, sources.core_hc_pointwise, true);
}
pub fn collect(a: std.mem.Allocator, device: metal.Device, queue: metal.Queue, k: *Kernels, cases: *[count]Case, out: *std.ArrayList(contract.gates.Kernel), only: ?[]const u8) !void {
    var i: usize = 0;
    for (common.variants) |variant| for (std.enums.values(common.Control)) |control| {
        cases[i] = .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .backend = .{ .kernels = k }, .variant = variant, .control = control };
        i += 1;
    };
    for (cases) |*case| {
        const kernel = case.kernel();
        if (only) |needle| if (std.mem.indexOf(u8, kernel.spec.name, needle) == null) continue;
        try out.append(a, kernel);
    }
}
