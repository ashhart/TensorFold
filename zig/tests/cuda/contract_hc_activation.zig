//! Both HC activation roles reach the production CUDA launch adapter through guarded buffers.
const std = @import("std");
const core = @import("core");
const cuda = @import("cuda");
const common = core.hc_activation_contract;
const api = core.hc_activation_cuda_adapter(core.hc_activation, cuda);
const Backend = struct {
    pub const name = "cuda";
    pub const Rig = @import("contract_rig.zig").Rig;
    kernels: *api.Kernels,
    views: [2]api.View = undefined,
    buffers: [2]cuda.DeviceBuffer = undefined,
    pub fn bind(b: *Backend, r: *Rig, f: common.Fixture) !void {
        const n = f.x.len * 2;
        const pointers = [_]u64{ try r.input(std.mem.sliceAsBytes(f.x)), try r.output(n) };
        for (pointers, 0..) |ptr, i| {
            b.buffers[i] = .{ .d = r.gpu.d, .ptr = ptr, .len = n };
            b.views[i] = .{ .buffer = &b.buffers[i], .len = n };
        }
    }
    pub fn launch(b: *Backend, r: *Rig, p: core.hc_activation.Params, control: common.Control, repeat: c_int) !void {
        try b.kernels.testEncode(r.s, p, b.views[0], b.views[1], @fromBackingInt(@backingInt(control)), repeat);
        try r.s.synchronize();
    }
};
pub const Case = common.Case(core.contract, Backend);
pub const Kernels = api.Kernels;
pub const count = 7 * 2 * std.enums.values(common.Control).len;
pub fn collect(a: std.mem.Allocator, gpu: @import("check.zig").Gpu, s: cuda.Stream, k: *Kernels, cases: *[count]Case, out: *std.ArrayList(core.contract.gates.Kernel)) !void {
    var i: usize = 0;
    for (2..9) |streams| for ([_]bool{ false, true }) |injection| for (std.enums.values(common.Control)) |control| {
        cases[i] = .{ .a = a, .rig = .{ .gpu = gpu, .s = s }, .backend = .{ .kernels = k }, .streams = @intCast(streams), .injection = injection, .control = control };
        i += 1;
    };
    for (cases) |*case| try out.append(a, case.kernel());
}
