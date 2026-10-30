//! Qwen4 geometry exercises the unchanged CUDA chunk and merge kernels against independent binary64.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const common = @import("qwen4").family.qsa_attn_contract;
const check = @import("check.zig");
const Backend = struct {
    pub const name = "cuda";
    pub const Rig = @import("contract_rig.zig").Rig;
    pub const served: []const core.contract.spec.Shape = &.{ .{ .rows = 1, .n = 1, .k = 1024 }, .{ .rows = 128, .n = 2, .k = 33 }, .{ .rows = 1, .n = 2, .k = 513 }, .{ .rows = 1, .n = 2, .k = 1025 } };
    functions: [2]cuda.Function,
    input: [7]u64 = undefined,
    partials: [3]u64 = undefined,
    output: u64 = undefined,
    pub fn bind(b: *@This(), r: *Rig, f: common.Fixture, control: common.Control) !void {
        if (control != .production) return error.Qwen4ExistingKernelControl;
        for ([_][]const u8{ std.mem.sliceAsBytes(f.query), std.mem.sliceAsBytes(f.key), std.mem.sliceAsBytes(f.value), std.mem.sliceAsBytes(f.ids), std.mem.sliceAsBytes(f.nk), std.mem.sliceAsBytes(f.spr) }, 0..) |bytes, i| b.input[i] = try r.input(bytes);
        const pos: i32 = @intCast(f.keys - 1);
        b.input[6] = try r.input(std.mem.asBytes(&pos));
        const nch = (f.keys + 511) / 512;
        for ([_]usize{ 256, 1, 1 }, 0..) |n, i| b.partials[i] = try r.scratch(f.rows * nch * 24 * n * 4);
        b.output = try r.output(f.rows * 24 * 256 * 2);
    }
    pub fn launch(b: *@This(), r: *Rig, f: common.Fixture, _: u32) !void {
        const nch = (f.keys + 511) / 512;
        for (0..f.rows) |row| {
            const po = b.partials[0] + row * nch * 24 * 256 * 4;
            const pm = b.partials[1] + row * nch * 24 * 4;
            const pl = b.partials[2] + row * nch * 24 * 4;
            var args: cuda.Args = .{};
            for ([_]u64{ b.input[0] + row * 24 * 256 * 2, b.input[1], b.input[2], b.input[6], po, pm, pl, b.input[3] + row * f.keys * 4, b.input[4] + row * 4, b.input[5] + row * 4 }) |ptr| args.add(ptr);
            for ([_]usize{ 24, 4, 6, 512, nch }) |n| args.add(@as(c_int, @intCast(n)));
            args.add(common.scale);
            args.add(@as(c_int, @intCast(f.keys)));
            args.add(@as(c_int, 1));
            try cuda.launch.launch(b.functions[0], .{ .grid = .{ .x = 1, .y = 4, .z = @intCast(nch) }, .block = .{ .x = 128 }, .shared = 26624 }, r.s, &args);
            args = .{};
            for ([_]u64{ po, pm, pl, b.input[6], b.output + row * 24 * 256 * 2, b.input[4] + row * 4, b.input[5] + row * 4 }) |ptr| args.add(ptr);
            for ([_]usize{ 24, 4, 6, 512, nch, 1 }) |n| args.add(@as(c_int, @intCast(n)));
            try cuda.launch.launch(b.functions[1], .{ .grid = .{ .x = 1, .y = 24 }, .block = .{ .x = 256 } }, r.s, &args);
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
        const parts = try module.function("fn_attn_chunks_one");
        const merge = try module.function("fn_attn_merge");
        if (try parts.attribute(.max_threads_per_block) < 128 or try merge.attribute(.max_threads_per_block) < 256) return error.Qwen4AttentionThreads;
        try parts.allowDynamicShared(26624);
        s.* = .{ .module = module, .stream = stream, .case = .{ .a = a, .rig = .{ .gpu = gpu, .s = stream }, .backend = .{ .functions = .{ parts, merge } } } };
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
