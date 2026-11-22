//! Own the draw/controller contract fixtures for the CUDA runner; every symbol comes from native production images.
const std = @import("std");
const cuda = @import("cuda");
const nemo = @import("nemotron");
const core = @import("core");
const check = @import("check.zig");
const Greedy = @import("contract_nemotron_mtp_greedy.zig").Greedy;
const Controller = @import("contract_nemotron_mtp_controller.zig").Controller;
const images = cuda.kernels;
pub const Set = struct {
    modules: [7]cuda.Module = undefined,
    made: usize = 0,
    torch: nemo.torch_ops.Functions = undefined,
    greedy: [3]Greedy = undefined,
    controller: [2]Controller = undefined,
    pub fn init(self: *Set, gpu: check.Gpu, stream: cuda.Stream) !void {
        self.* = .{};
        errdefer {
            for (self.modules[0..self.made]) |*m| m.unload();
        }
        for ([_][]const u8{ images.torch_argmax, images.torch_topk, images.torch_pointwise, images.torch_indexing, images.torch_movement, images.torch_nemotron_constants, images.nemotron_keyed }, 0..) |bytes, i| {
            self.modules[i] = try cuda.Module.load(gpu.d, bytes);
            self.made += 1;
        }
        self.torch = try nemo.torch_ops.Functions.resolve(self.modules[0..6]);
        const mod = self.modules[6];
        const fused = try mod.function("tf_nemo_mtp_greedy");
        const keyed = try mod.function("tf_nemo_keyed_greedy");
        for (&self.greedy, 0..) |*x, i| x.* = .{ .rig = .{ .gpu = gpu, .s = stream }, .fused = fused, .keyed = keyed, .torch = &self.torch, .mapped = i != 1, .corrupt = i == 2 };
        const start = try mod.function("tf_nemo_mtp_start");
        const step = try mod.function("tf_nemo_mtp_control");
        const feed = try mod.function("tf_ctl_nemo_mtp_feed");
        for (&self.controller, 0..) |*x, i| x.* = .{ .rig = .{ .gpu = gpu, .s = stream }, .start = start, .step = step, .feed = feed, .corrupt = i == 1 };
    }
    pub fn append(self: *Set, a: std.mem.Allocator, out: *std.ArrayList(core.contract.gates.Kernel)) !void {
        for (&self.greedy) |*x| try out.append(a, x.kernel());
        for (&self.controller) |*x| try out.append(a, x.kernel());
    }
    pub fn deinit(self: *Set) void {
        for (&self.controller) |*x| x.deinit();
        for (&self.greedy) |*x| x.deinit();
        for (self.modules[0..self.made]) |*m| m.unload();
    }
};
