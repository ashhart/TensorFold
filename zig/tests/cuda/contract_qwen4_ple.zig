//! Lexical contracts call the serving CUDA owner, including gate sums and every queue-ordered kept tail.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const family = @import("qwen4").family;
const common = family.ple_contract;
const Api = family.cuda_ple.Kernels(cuda);
const check = @import("check.zig");
const Backend = struct {
    pub const name = "cuda";
    pub const Rig = @import("contract_rig.zig").Rig;
    pub const served: []const core.contract.spec.Shape = &.{ .{ .rows = 1, .n = 256, .k = 2 }, .{ .rows = 16, .n = 256, .k = 2 }, .{ .rows = 1, .n = 5120, .k = 4 }, .{ .rows = 128, .n = 5120, .k = 4 } };
    a: std.mem.Allocator,
    storage: family.cuda_storage.Backend(cuda),
    image: []const u8,
    c: ?family.Config = null,
    kernels: ?Api = null,
    buffers: [12]cuda.DeviceBuffer = undefined,
    views: [12]Api.View = undefined,
    original: cuda.DeviceBuffer = undefined,
    control: common.Control = .production,
    pub fn deinit(b: *Backend) void {
        if (b.kernels) |*k| k.close(b.storage.stream) catch return;
        if (b.c) |*c| c.deinit();
    }
    pub fn bind(b: *Backend, r: *Rig, f: common.Fixture, control: common.Control) !void {
        if (b.c) |*c| if (c.hidden != f.p.hidden or c.streams != f.p.streams) {
            try b.kernels.?.close(b.storage.stream);
            b.kernels = null;
            c.deinit();
            b.c = null;
        };
        if (b.c == null) {
            b.c = try common.configuration(b.a, f.p);
            errdefer {
                b.c.?.deinit();
                b.c = null;
            }
            b.kernels = try Api.loadControls(r.gpu.d, b.image, &b.c.?, true);
        }
        b.control = control;
        for (f.inputs, [_]usize{ 0, 1, 2, 3, 4, 8, 9 }, 0..) |bytes, i, from| {
            const ptr = try r.input(bytes);
            if (from == 1) {
                b.original = .{ .d = r.gpu.d, .ptr = ptr, .len = bytes.len };
                continue;
            }
            b.buffers[i] = .{ .d = r.gpu.d, .ptr = ptr, .len = bytes.len };
        }
        const w = f.p.hidden * f.p.streams;
        const tail = (f.p.taps - 1) * f.p.dilation * w * 2;
        const selected = ([_]usize{ 5, 6, 1, 11 })[@backingInt(f.part)];
        const lengths = [_]usize{ f.p.rows * w * 2, f.p.rows * w * 2, f.p.rows * w * 2, f.p.rows * f.p.streams * 4, tail + f.p.rows * w * 2, f.p.rows * tail };
        for ([_]usize{ 1, 5, 6, 7, 10, 11 }, lengths) |i, n| b.buffers[i] = .{ .d = r.gpu.d, .ptr = if (i == selected) try r.output(n) else try r.scratch(n), .len = n };
        for (&b.buffers, &b.views) |*buffer, *view| view.* = .{ .buffer = buffer, .len = buffer.len };
    }
    pub fn launch(b: *Backend, _: *Rig, f: common.Fixture, repeat: u32) !void {
        try b.storage.copy(b.buffers[1], 0, b.original, 0, b.original.len);
        const tail = (f.p.taps - 1) * f.p.dilation * f.p.hidden * f.p.streams * 2;
        try b.kernels.?.testEncode(&b.storage, f.p.rows, b.views, tail, b.control, repeat);
        try b.storage.drain();
    }
};
pub const Case = common.Case(core.contract, Backend);
pub const Suite = struct {
    cases: [20]Case = undefined,
    made: usize = 0,
    stream: cuda.Stream,
    pub fn create(a: std.mem.Allocator, gpu: check.Gpu, stream: cuda.Stream, out: *std.ArrayList(core.contract.gates.Kernel)) !*Suite {
        return createImage(a, gpu, stream, cuda.kernels.qwen4_ple, out);
    }
    pub fn createImage(a: std.mem.Allocator, gpu: check.Gpu, stream: cuda.Stream, image: []const u8, out: *std.ArrayList(core.contract.gates.Kernel)) !*Suite {
        const s = try a.create(Suite);
        s.* = .{ .stream = stream };
        errdefer s.destroy(a);
        for (std.enums.values(common.Part)) |part| for (std.enums.values(common.Control)) |control| {
            const case = &s.cases[s.made];
            case.* = .{ .a = a, .rig = .{ .gpu = gpu, .s = stream }, .backend = .{ .a = a, .image = image, .storage = .{ .driver = gpu.d, .stream = stream, .context = gpu.ctx } }, .part = part, .control = control };
            s.made += 1;
            try out.append(a, case.kernel());
        };
        return s;
    }
    pub fn destroy(s: *Suite, a: std.mem.Allocator) void {
        s.stream.synchronize() catch return;
        for (s.cases[0..s.made]) |*case| case.deinit();
        a.destroy(s);
    }
};
