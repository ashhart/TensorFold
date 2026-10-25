//! The unchanged CUDA score image checks Qwen4's index geometry through the same independent Cases.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const family = @import("qwen4").family;
const common = family.qsa_scores_contract;
const check = @import("check.zig");
const Backend = struct {
    pub const name = "cuda";
    pub const Rig = @import("contract_rig.zig").Rig;
    pub const served: []const core.contract.spec.Shape = &.{ .{ .rows = 1, .n = 129, .k = 128 }, .{ .rows = 128, .n = 129, .k = 128 }, .{ .rows = 8, .n = 513, .k = 128 }, .{ .rows = 1, .n = 4097, .k = 128 } };
    function: cuda.Function,
    input: [3]u64 = undefined,
    output: u64 = undefined,
    pub fn bind(b: *@This(), r: *Rig, f: common.Fixture, control: common.Control) !void {
        if (control != .production) return error.Qwen4ExistingKernelControl;
        b.input[0] = try r.input(std.mem.sliceAsBytes(f.query));
        b.input[1] = try r.input(std.mem.sliceAsBytes(f.pooled));
        const positions = try r.gpu.gpa.alloc(i32, f.rows);
        defer r.gpu.gpa.free(positions);
        @memset(positions, @intCast(f.blocks * 8 - 1));
        b.input[2] = try r.input(std.mem.sliceAsBytes(positions));
        b.output = try r.output(f.rows * f.blocks * 4);
    }
    pub fn launch(b: *@This(), r: *Rig, f: common.Fixture, _: u32) !void {
        for (0..f.rows) |row| {
            var args: cuda.Args = .{};
            for ([_]u64{ b.input[0] + row * 4 * 128 * 2, b.input[1], b.input[2] + row * 4, b.output + row * f.blocks * 4 }) |ptr| args.add(ptr);
            for ([_]usize{ f.blocks, 4, 8, 128 }) |n| args.add(@as(c_int, @intCast(n)));
            try cuda.launch.launch(b.function, .{ .grid = .{ .x = 1, .y = @intCast((f.blocks + 63) / 64) }, .block = .{ .x = 256 } }, r.s, &args);
        }
        try r.s.synchronize();
    }
};
pub const Case = common.Case(core.contract, Backend);
pub const Suite = struct {
    module: cuda.Module,
    case: Case,
    stream: cuda.Stream,
    pub fn create(a: std.mem.Allocator, gpu: check.Gpu, stream: cuda.Stream, out: *std.ArrayList(core.contract.gates.Kernel)) !*Suite {
        return createImage(a, gpu, stream, cuda.kernels.flashnext_attn, out);
    }
    pub fn createImage(a: std.mem.Allocator, gpu: check.Gpu, stream: cuda.Stream, image: []const u8, out: *std.ArrayList(core.contract.gates.Kernel)) !*Suite {
        const s = try a.create(Suite);
        errdefer a.destroy(s);
        var module = try cuda.Module.load(gpu.d, image);
        errdefer module.unload();
        const function = try module.function("fn_qsa_scores");
        if (try function.attribute(.max_threads_per_block) < 256) return error.Qwen4AttentionThreads;
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
