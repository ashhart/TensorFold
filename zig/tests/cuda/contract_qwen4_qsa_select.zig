//! The unchanged CUDA selector checks Qwen4's top128 ratio8 lists, signed order, tails and dense transitions.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const family = @import("qwen4").family;
const common = family.qsa_select_contract;
const check = @import("check.zig");
const Backend = struct {
    pub const name = "cuda";
    pub const Rig = @import("contract_rig.zig").Rig;
    pub const served: []const core.contract.spec.Shape = &.{ .{ .rows = 8, .n = 129, .k = 1 }, .{ .rows = 1, .n = 0, .k = 2 }, .{ .rows = 128, .n = 128, .k = 1 }, .{ .rows = 1, .n = 127, .k = 8 }, .{ .rows = 8, .n = 513, .k = 8 }, .{ .rows = 3, .n = 4097, .k = 8 }, .{ .rows = 1, .n = 8192, .k = 8 } };
    function: cuda.Function,
    scores: u64 = 0,
    position: u64 = 0,
    output: u64 = 0,
    counts: u64 = 0,
    sparse: u64 = 0,
    pub fn bind(b: *@This(), r: *Rig, f: common.Fixture, control: common.Control) !void {
        if (control != .production) return error.Qwen4ExistingKernelControl;
        b.scores = try r.input(std.mem.sliceAsBytes(f.scores));
        const position = f.ends[0] - 1;
        b.position = try r.input(std.mem.asBytes(&position));
        b.output = try r.output(f.want.len * 4);
        b.counts = try r.scratch(f.rows * 4);
        b.sparse = try r.scratch(f.rows * 4);
    }
    pub fn launch(b: *@This(), r: *Rig, f: common.Fixture, _: u32) !void {
        var args: cuda.Args = .{};
        for ([_]u64{ b.scores, b.position, b.output, b.counts, b.sparse }) |ptr| args.add(ptr);
        for ([_]usize{ f.stride, common.idw, common.top, common.ratio }) |n| args.add(@as(c_int, @intCast(n)));
        try cuda.launch.launch(b.function, .{ .grid = .{ .x = @intCast(f.rows) }, .block = .{ .x = 1024 } }, r.s, &args);
        try r.s.synchronize();
    }
};
pub const Case = common.Case(core.contract, Backend);
pub const Suite = struct {
    module: cuda.Module,
    case: Case,
    stream: cuda.Stream,
    pub fn create(a: std.mem.Allocator, gpu: check.Gpu, stream: cuda.Stream, out: *std.ArrayList(core.contract.gates.Kernel)) !*Suite {
        return createImage(a, gpu, stream, cuda.kernels.flashnext_batch, out);
    }
    pub fn createImage(a: std.mem.Allocator, gpu: check.Gpu, stream: cuda.Stream, image: []const u8, out: *std.ArrayList(core.contract.gates.Kernel)) !*Suite {
        const s = try a.create(Suite);
        errdefer a.destroy(s);
        var module = try cuda.Module.load(gpu.d, image);
        errdefer module.unload();
        const function = try module.function("fn_qsa_select");
        if (try function.attribute(.max_threads_per_block) < 1024) return error.Qwen4AttentionThreads;
        s.* = .{ .module = module, .stream = stream, .case = .{ .a = a, .rig = .{ .gpu = gpu, .s = stream }, .backend = .{ .function = function } } };
        try out.append(a, s.case.kernel());
        return s;
    }
    pub fn destroy(s: *Suite, a: std.mem.Allocator) void {
        s.stream.synchronize() catch return;
        s.case.deinit();
        s.module.unload();
        a.destroy(s);
    }
};
