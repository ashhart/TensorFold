//! Every raw weight format reaches the production Metal projection owner through guarded contract buffers.
const std = @import("std");
const metal = @import("metal");
const tf = @import("tensorfold");
const sources = @import("kernel_sources");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const common = tf.weight_only_contract;
const api = tf.weight_only_metal_adapter(tf.weight_only, metal);
const Backend = struct {
    pub const name = "metal";
    pub const Rig = rig_.Rig;
    kernels: *api.Kernels,
    views: [5]api.View = undefined,
    pub fn bind(b: *Backend, r: *Rig, f: common.Fixture) !void {
        for ([_][]const u8{ std.mem.sliceAsBytes(f.x), f.weight, f.scales, f.biases }, 0..) |bytes, i| {
            const ref = try r.input(bytes);
            b.views[i] = .{ .buffer = ref.buf, .offset = ref.off, .len = bytes.len };
        }
        const size = (try tf.weight_only.validate(f.p)).out;
        const ref = try r.output(size);
        b.views[4] = .{ .buffer = ref.buf, .offset = ref.off, .len = size };
    }
    pub fn launch(b: *Backend, r: *Rig, p: tf.weight_only.Params, control: common.Control, repetition: c_int) !void {
        const pass = r.begin();
        b.kernels.testEncode(pass.e, p, b.views[0], b.views[1], b.views[2], b.views[3], b.views[4], @fromBackingInt(@backingInt(control)), repetition) catch |err| {
            // Even refused encodes finish the command-buffer lifetime before the rig can release buffers.
            try r.finish(pass);
            return err;
        };
        try r.finish(pass);
    }
};
pub const Case = common.Case(contract, Backend);
pub const count = common.variants.len * std.enums.values(common.Control).len;
pub const Kernels = api.Kernels;
pub fn text(a: std.mem.Allocator) ![]u8 {
    return api.source(a, sources.core_weight_only_body, sources.core_weight_only, true);
}
pub fn collect(a: std.mem.Allocator, device: metal.Device, queue: metal.Queue, kernels: *api.Kernels, cases: *[count]Case, out: *std.ArrayList(contract.gates.Kernel), only: ?[]const u8) !void {
    var i: usize = 0;
    for (common.variants) |variant| for (std.enums.values(common.Control)) |control| {
        cases[i] = .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .backend = .{ .kernels = kernels }, .format = variant.format, .group = variant.group, .scale_kind = variant.scale_kind, .control = control };
        i += 1;
    };
    for (cases) |*case| {
        const kernel = case.kernel();
        if (only) |needle| if (std.mem.indexOf(u8, kernel.spec.name, needle) == null) continue;
        try out.append(a, kernel);
    }
}
