//! Both pointwise policies reach production CUDA adapters with immutable guarded input buffers.
const std = @import("std");
const core = @import("core");
const cuda = @import("cuda");
const common = core.hc_pointwise_contract;
const api = core.hc_pointwise_cuda_adapter(core.hc_pointwise, cuda);
const Backend = struct {
    pub const name = "cuda";
    pub const Rig = @import("contract_rig.zig").Rig;
    kernels: *api.Kernels,
    buffers: [4]cuda.DeviceBuffer = undefined,
    views: [4]api.View = undefined,
    fn set(b: *Backend, r: *Rig, i: usize, ptr: u64, n: usize) void {
        b.buffers[i] = .{ .d = r.gpu.d, .ptr = ptr, .len = n };
        b.views[i] = .{ .buffer = &b.buffers[i], .len = n };
    }
    pub fn bind(b: *Backend, r: *Rig, f: common.Fixture) !void {
        for ([_][]const u8{ std.mem.sliceAsBytes(f.x), std.mem.sliceAsBytes(f.b), std.mem.sliceAsBytes(f.g) }, 0..) |bytes, i| b.set(r, i, try r.input(bytes), bytes.len);
        b.set(r, 3, try r.output(f.x.len * 2), f.x.len * 2);
    }
    pub fn launch(b: *Backend, r: *Rig, p: core.hc_pointwise.Params, control: common.Control, repeat: c_int) !void {
        try b.kernels.testEncode(r.s, p, b.views, @fromBackingInt(@backingInt(control)), repeat);
        try r.s.synchronize();
    }
};
pub const Case = common.Case(core.contract, Backend);
pub const Kernels = api.Kernels;
pub const count = common.variants.len * std.enums.values(common.Control).len;
pub fn collect(a: std.mem.Allocator, gpu: @import("check.zig").Gpu, s: cuda.Stream, k: *Kernels, cases: *[count]Case, out: *std.ArrayList(core.contract.gates.Kernel)) !void {
    var i: usize = 0;
    for (common.variants) |variant| for (std.enums.values(common.Control)) |control| {
        cases[i] = .{ .a = a, .rig = .{ .gpu = gpu, .s = s }, .backend = .{ .kernels = k }, .variant = variant, .control = control };
        i += 1;
    };
    for (cases) |*case| try out.append(a, case.kernel());
}
