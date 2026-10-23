//! Exercise the actual MTP head quantizer and packer through an initialized fake CUDA driver, never device code.
const std = @import("std");
const core = @import("core");
const cuda = @import("cuda");
const load = @import("cuda_mtp_load.zig");
const weights = @import("cuda_weights.zig");
const quant = @import("cuda_quant.zig");
const q4 = @import("cuda_q4.zig");

const Fake = struct {
    var allocs: usize = 0;
    var live: usize = 0;
    var queued: bool = false;
    var freed_early: bool = false;
    var phase: usize = 0;
    var contexts: usize = 0;
    fn alloc(p: *u64, _: usize) callconv(.c) cuda.abi.Result {
        if (phase == 2 and allocs == 4) return cuda.abi.error_not_ready;
        allocs += 1;
        live += 1;
        p.* = allocs * 256;
        return cuda.abi.success;
    }
    fn free(p: u64) callconv(.c) cuda.abi.Result {
        if (queued and p <= 4 * 256) freed_early = true;
        live -= 1;
        return cuda.abi.success;
    }
    fn copy(_: u64, _: ?*const anyopaque, _: usize) callconv(.c) cuda.abi.Result {
        return cuda.abi.success;
    }
    fn wait(s: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        if (s == null) return cuda.abi.success;
        if (phase == 1) return cuda.abi.error_not_ready;
        queued = false;
        return cuda.abi.success;
    }
    fn context() callconv(.c) cuda.abi.Result {
        contexts += 1;
        queued = false;
        return cuda.abi.success;
    }
    fn launch(_: cuda.abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, _: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        queued = true;
        return if (phase == 0) cuda.abi.error_not_ready else cuda.abi.success;
    }
    fn driver() cuda.Driver {
        var d: cuda.Driver = undefined;
        d.api.cuMemAlloc_v2 = alloc;
        d.api.cuMemFree_v2 = free;
        d.api.cuMemcpyHtoD_v2 = copy;
        d.api.cuStreamSynchronize = wait;
        d.api.cuCtxSynchronize = context;
        d.api.cuLaunchKernel = launch;
        return d;
    }
};

const Loader = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    stream: cuda.Stream,
    tools: weights.Tools,
    w: *weights.Weights,
    pub fn scratch(l: *@This(), n: usize) !cuda.DeviceBuffer {
        return cuda.DeviceBuffer.alloc(l.d, n);
    }
};

test "MTP widened head and quantizer scratch drain before free on launch, wait, device or label allocation failure" {
    const gpa = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var dir: [64]u8 = undefined;
    const at = try std.fmt.bufPrint(&dir, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    var path: [96]u8 = undefined;
    const codes: [64 * 64]u8 = @splat(0x38);
    const scales: [64 * 2]u8 = @splat(127);
    const tensors = [_]core.shard_edit.New{
        .{ .name = "lm_head.weight", .dtype = .f8_e4m3, .shape = &.{ 64, 64 }, .bytes = &codes },
        .{ .name = "lm_head.weight_scale", .dtype = .u8, .shape = &.{ 64, 2 }, .bytes = &scales },
    };
    try core.shard_edit.create(gpa, try std.fmt.bufPrint(&path, "{s}/model.safetensors", .{at}), &.{}, &tensors);
    var ck = try core.Checkpoint.openModel(gpa, std.testing.io, at);
    defer ck.close();
    for (0..5) |phase| {
        Fake.allocs = 0;
        Fake.live = 0;
        Fake.queued = false;
        Fake.freed_early = false;
        Fake.phase = phase;
        Fake.contexts = 0;
        var d = Fake.driver();
        const f: cuda.Function = .{ .d = &d, .handle = null };
        const qk: quant.Kernels = .{ .amax = f, .nvfp4 = f, .q4 = f };
        const tk: q4.Kernels = .{ .group = f, .pack_dense = f, .transpose = f, .gb10 = false };
        var fail = std.testing.FailingAllocator.init(gpa, .{ .fail_index = 1 });
        const allocator = if (phase == 3) fail.allocator() else gpa;
        var share: @import("cuda_shard.zig").Share = undefined;
        share.vocab = .{ .lo = 0, .hi = 64 };
        var w: weights.Weights = .{ .gpa = allocator, .config = .{ .hidden = 64 }, .share = share };
        w.head = .{ .mxfp8 = cuda.mxfp8.weight(0, 0, 64, 64) };
        var l: Loader = .{ .gpa = allocator, .d = &d, .stream = .{ .d = &d, .handle = @ptrFromInt(64) }, .tools = .{ .pack_dense = f, .lane = undefined, .quant = &qk, .q4 = &tk }, .w = &w };
        const result = load.draft(&l, &ck);
        if (phase < 3) try std.testing.expectError(error.NotReady, result) else if (phase == 3) {
            try std.testing.expect(fail.has_induced_failure);
            try std.testing.expectError(error.OutOfMemory, result);
        } else _ = try result;
        w.deinit();
        try std.testing.expect(!Fake.queued);
        try std.testing.expect(!Fake.freed_early);
        try std.testing.expectEqual(@as(usize, 0), Fake.live);
        if (phase == 1) try std.testing.expectEqual(@as(usize, 1), Fake.contexts);
    }
}
