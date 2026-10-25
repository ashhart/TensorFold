//! Readout and kept-state probes reach the serving CUDA recurrence and actual-source fault functions.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const family = @import("qwen4").family;
const common = family.gdn_contract;
const api = family.cuda_gdn.Kernels(cuda);
const check = @import("check.zig");
const Backend = struct {
    pub const name = "cuda";
    pub const Rig = @import("contract_rig.zig").Rig;
    pub const served: []const core.contract.spec.Shape = &.{ .{ .rows = 1, .n = 1, .k = 2 }, .{ .rows = 16, .n = 1, .k = 2 }, .{ .rows = 1, .n = 16, .k = 48 }, .{ .rows = 16, .n = 16, .k = 48 } };
    kernels: *api,
    ranges: [10]core.weight_only.Range = undefined,
    control: common.Control = .production,
    pub fn bind(b: *Backend, r: *Rig, f: common.Fixture, control: common.Control) !void {
        b.control = control;
        for (f.inputs, 0..) |bytes, i| b.ranges[i] = .{ .address = @intCast(try r.input(bytes)), .len = bytes.len };
        const sizes = try family.cuda_gdn.sizes(f.p);
        for ([_]usize{ sizes.output, sizes.sums, sizes.snapshots }, 7..) |len, i| {
            const selected = (i == 7 and f.part == .readout) or (i == 9 and f.part == .state);
            b.ranges[i] = .{ .address = @intCast(if (selected) try r.output(len) else try r.scratch(len)), .len = len };
        }
    }
    pub fn launch(b: *Backend, r: *Rig, f: common.Fixture, repeat: u32) !void {
        try b.kernels.testEncode(r.s, f.p, b.ranges, b.control, repeat);
        try r.s.synchronize();
    }
};
pub const Case = common.Case(core.contract, Backend);
pub const Suite = struct {
    kernels: api,
    stream: cuda.Stream,
    cases: [52]Case = undefined,
    made: usize = 0,
    pub fn create(a: std.mem.Allocator, gpu: check.Gpu, stream: cuda.Stream, out: *std.ArrayList(core.contract.gates.Kernel)) !*Suite {
        return createImage(a, gpu, stream, cuda.kernels.qwen4_gdn, out);
    }
    pub fn createImage(a: std.mem.Allocator, gpu: check.Gpu, stream: cuda.Stream, image: []const u8, out: *std.ArrayList(core.contract.gates.Kernel)) !*Suite {
        const s = try a.create(Suite);
        errdefer a.destroy(s);
        s.* = .{ .kernels = try api.loadControls(gpu.d, image, true), .stream = stream };
        errdefer {
            for (s.cases[0..s.made]) |*case| case.deinit();
            s.kernels.module.unload();
        }
        for ([_]family.cuda_gdn.Gate{ .silu, .sigmoid }) |gate| for ([_]u4{ 3, 7, 11, 15 }) |flags| for (std.enums.values(common.Part)) |part| {
            if (part == .state and flags != 3 and flags != 11) continue;
            if (part == .conv) continue;
            for (std.enums.values(common.Control)) |control| {
                if (control == .sums and flags != 11) continue;
                const case = &s.cases[s.made];
                case.* = .{ .a = a, .rig = .{ .gpu = gpu, .s = stream }, .backend = .{ .kernels = &s.kernels }, .gate = gate, .flags = flags, .part = part, .control = control };
                s.made += 1;
                try out.append(a, case.kernel());
            }
        };
        std.debug.assert(s.made == s.cases.len);
        return s;
    }
    pub fn destroy(s: *Suite, a: std.mem.Allocator) void {
        s.stream.synchronize() catch return;
        for (s.cases[0..s.made]) |*case| case.deinit();
        s.kernels.close(s.stream) catch return;
        a.destroy(s);
    }
};
