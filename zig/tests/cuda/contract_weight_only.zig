//! Every raw weight format reaches the production CUDA projection owner through guarded contract buffers.
const std = @import("std");
const core = @import("core");
const cuda = @import("cuda");
const rig_ = @import("contract_rig.zig");
const common = core.weight_only_contract;
const api = core.weight_only_cuda_adapter(core.weight_only, cuda);
const Backend = struct {
    pub const name = "cuda";
    pub const Rig = rig_.Rig;
    kernels: *api.Kernels,
    views: [5]api.View = undefined,
    buffers: [5]cuda.DeviceBuffer = undefined,
    fn view(b: *Backend, i: usize, r: *Rig, ptr: u64, len: usize) void {
        // The rig retains the guarded allocation until this case drains and resets it.
        b.buffers[i] = .{ .d = r.gpu.d, .ptr = ptr, .len = len };
        b.views[i] = .{ .buffer = &b.buffers[i], .len = len };
    }
    pub fn bind(b: *Backend, r: *Rig, f: common.Fixture) !void {
        for ([_][]const u8{ std.mem.sliceAsBytes(f.x), f.weight, f.scales, f.biases }, 0..) |bytes, i|
            b.view(i, r, try r.input(bytes), bytes.len);
        const size = (try core.weight_only.validate(f.p)).out;
        b.view(4, r, try r.output(size), size);
    }
    pub fn launch(b: *Backend, r: *Rig, p: core.weight_only.Params, control: common.Control, repetition: c_int) !void {
        try b.kernels.testEncode(r.s, p, b.views[0], b.views[1], b.views[2], b.views[3], b.views[4], @fromBackingInt(@backingInt(control)), repetition);
        try r.s.synchronize();
    }
};
pub const Case = common.Case(core.contract, Backend);
pub const count = common.variants.len * std.enums.values(common.Control).len;
pub fn collect(a: std.mem.Allocator, gpu: @import("check.zig").Gpu, stream: cuda.Stream, kernels: *api.Kernels, cases: *[count]Case, out: *std.ArrayList(core.contract.gates.Kernel)) !void {
    var i: usize = 0;
    for (common.variants) |variant| for (std.enums.values(common.Control)) |control| {
        cases[i] = .{ .a = a, .rig = .{ .gpu = gpu, .s = stream }, .backend = .{ .kernels = kernels }, .format = variant.format, .group = variant.group, .scale_kind = variant.scale_kind, .control = control };
        i += 1;
    };
    for (cases) |*case| try out.append(a, case.kernel());
}
pub const Kernels = api.Kernels;
