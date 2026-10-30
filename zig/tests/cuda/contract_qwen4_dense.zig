//! Dense activation probes use the serving HC adapter with its explicitly borrowed scalar-image function.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const family = @import("qwen4").family;
const common = family.hc_native_contract;
const Api = family.cuda_hc.Kernels(cuda);
const check = @import("check.zig");
const Backend = struct {
    pub const name = "cuda";
    pub const Rig = @import("contract_rig.zig").Rig;
    kernels: *Api,
    function: cuda.Function,
    buffers: [3]cuda.DeviceBuffer = undefined,
    views: [3]Api.View = undefined,
    pub fn bind(b: *@This(), r: *Rig, f: common.Fixture) !void {
        for (f.inputs[0..2], 0..) |bytes, i| b.buffers[i] = .{ .d = r.gpu.d, .ptr = try r.input(bytes), .len = bytes.len };
        b.buffers[2] = .{ .d = r.gpu.d, .ptr = try r.output(f.output_bytes), .len = f.output_bytes };
        for (&b.buffers, &b.views) |*buffer, *view| view.* = .{ .buffer = buffer, .len = buffer.len };
    }
    pub fn launch(b: *@This(), r: *Rig, op: family.hc_layout.Op, p: family.hc_layout.Call, repeat: u32) !void {
        if (op != .act) return error.NotDenseActivation;
        var local = b.kernels.*;
        try local.bindDense(b.function);
        local.dense_repeat = repeat;
        try local.encode(r.s, op, p, &b.views);
        try r.s.synchronize();
    }
};
pub const Case = common.Case(core.contract, Backend);
pub const Suite = struct {
    kernels: Api,
    cases: [4]Case = undefined,
    made: usize = 0,
    stream: cuda.Stream,
    pub fn create(a: std.mem.Allocator, gpu: check.Gpu, stream: cuda.Stream, out: *std.ArrayList(core.contract.gates.Kernel)) !*Suite {
        return createImage(a, gpu, stream, cuda.kernels.qwen4_hc, out);
    }
    pub fn createImage(a: std.mem.Allocator, gpu: check.Gpu, stream: cuda.Stream, image: []const u8, out: *std.ArrayList(core.contract.gates.Kernel)) !*Suite {
        const s = try a.create(Suite);
        errdefer a.destroy(s);
        s.* = .{ .kernels = .{ .module = try cuda.Module.load(gpu.d, image), .geometry = .{ .hidden = 5120, .streams = 4 } }, .stream = stream };
        errdefer {
            for (s.cases[0..s.made]) |*case| case.deinit();
            s.kernels.module.unload();
        }
        for ([_][:0]const u8{ "qwen4_dense_act", "qwen4_dense_act_bad_nan", "qwen4_dense_act_bad_repeat", "qwen4_dense_act_bad_tail" }, 0..) |name, i| {
            const function = try s.kernels.module.function(name);
            try s.kernels.bindDense(function);
            const case = &s.cases[s.made];
            case.* = .{ .a = a, .rig = .{ .gpu = gpu, .s = stream }, .backend = .{ .kernels = &s.kernels, .function = function }, .op = .act, .control = std.enums.values(common.Control)[i] };
            s.made += 1;
            try out.append(a, case.kernel());
        }
        return s;
    }
    pub fn destroy(s: *Suite, a: std.mem.Allocator) void {
        s.stream.synchronize() catch return;
        for (s.cases[0..s.made]) |*case| case.deinit();
        s.kernels.close(s.stream) catch return;
        a.destroy(s);
    }
};
