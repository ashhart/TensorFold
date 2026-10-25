//! Qwen4 geometry exercises the unchanged serving CUDA QSA frontend against independent node bounds.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const family = @import("qwen4").family;
const common = family.qsa_norm_contract;
const check = @import("check.zig");
const Backend = struct {
    pub const name = "cuda";
    pub const Rig = @import("contract_rig.zig").Rig;
    pub const served: []const core.contract.spec.Shape = &.{ .{ .rows = 1, .n = 24, .k = 4 }, .{ .rows = 128, .n = 24, .k = 4 } };
    functions: *const [2]cuda.Function,
    input: [6]u64 = undefined,
    output: [3]u64 = undefined,
    cache: [3]u64 = undefined,
    position: u64 = undefined,
    pub fn bind(b: *@This(), r: *Rig, f: common.Fixture, control: common.Control) !void {
        if (control != .production) return error.Qwen4ExistingKernelControl;
        for (f.inputs[0..f.count], 0..) |bytes, i| b.input[i] = try r.input(bytes);
        if (f.part == .pool) {
            const start: i32 = 24;
            b.position = try r.input(std.mem.asBytes(&start));
            b.output[0] = try r.output(f.want.len * 2);
            b.cache[0] = try r.scratch((f.rows + 3) * 128 * 2);
        } else {
            for ([_]usize{ f.rows * 24 * 256 * 2, f.rows * 4 * 256 * 2, f.rows * 4 * 128 * 2 }, 0..) |n, i| b.output[i] = if (i == @backingInt(f.part)) try r.output(n) else try r.scratch(n);
            for (&b.cache, [_]usize{ 4 * 256, 4 * 256, 128 }) |*ptr, n| ptr.* = try r.scratch((f.rows + 3) * n * 2);
        }
    }
    pub fn launch(b: *@This(), r: *Rig, f: common.Fixture, _: u32) !void {
        const pool = f.part == .pool;
        var args: cuda.Args = .{};
        const g = common.reference.geometry;
        if (pool) {
            for ([_]u64{ b.input[0], b.cache[0], b.position, b.input[1], b.input[3] + 12 }) |ptr| args.add(ptr);
            args.add(@as(f32, 1e-6));
            for ([_]usize{ f.rows * 8, 128, 32, 8 }) |v| args.add(@as(c_int, @intCast(v)));
        } else {
            for ([_]u64{ b.input[0], b.input[4], b.input[1], b.input[2], b.input[3], b.input[5] + 12, b.output[0], b.cache[0], b.cache[1], b.output[2], b.cache[2] }) |ptr| args.add(ptr);
            args.add(@as(f32, 1e-6));
            for ([_]usize{ g.projectionWidth(), 24, 4, 4, 128, 32 }) |v| args.add(@as(c_int, @intCast(v)));
        }
        try cuda.launch.launch(b.functions[if (pool) 1 else 0], .{ .grid = .{ .x = @intCast(if (pool) f.rows + 2 else f.rows), .y = if (pool) 1 else 33 }, .block = .{ .x = if (pool) 128 else 256 } }, r.s, &args);
        const src = b.cache[0] + (if (pool) @as(u64, 3 * 128 * 2) else 3 * 4 * 256 * 2);
        const n = f.rows * (if (pool) @as(usize, 128) else 4 * 256) * 2;
        const destination: cuda.DeviceBuffer = .{ .d = r.gpu.d, .ptr = b.output[if (pool) 0 else 1], .len = n };
        try destination.copyFrom(0, src, n, r.s.handle);
        try r.s.synchronize();
    }
};
pub const Case = common.Case(core.contract, Backend);
pub const Suite = struct {
    module: cuda.Module,
    functions: [2]cuda.Function,
    cases: [4]Case = undefined,
    made: usize = 0,
    stream: cuda.Stream,
    pub fn create(a: std.mem.Allocator, gpu: check.Gpu, stream: cuda.Stream, out: *std.ArrayList(core.contract.gates.Kernel)) !*Suite {
        return createImage(a, gpu, stream, cuda.kernels.flashnext_attn, out);
    }
    pub fn createImage(a: std.mem.Allocator, gpu: check.Gpu, stream: cuda.Stream, image: []const u8, out: *std.ArrayList(core.contract.gates.Kernel)) !*Suite {
        const s = try a.create(Suite);
        errdefer a.destroy(s);
        s.* = .{ .module = try cuda.Module.load(gpu.d, image), .functions = undefined, .stream = stream };
        errdefer {
            for (s.cases[0..s.made]) |*case| case.deinit();
            s.module.unload();
        }
        for ([_][:0]const u8{ "fn_attn_prep", "fn_qsa_pool" }, [_]usize{ 256, 128 }, 0..) |name, threads, i| {
            s.functions[i] = try s.module.function(name);
            if (try s.functions[i].attribute(.max_threads_per_block) < threads) return error.Qwen4AttentionThreads;
        }
        for (std.enums.values(common.Part)) |part| {
            const case = &s.cases[s.made];
            case.* = .{ .a = a, .rig = .{ .gpu = gpu, .s = stream }, .backend = .{ .functions = &s.functions }, .part = part };
            s.made += 1;
            try out.append(a, case.kernel());
        }
        return s;
    }
    pub fn destroy(s: *Suite, a: std.mem.Allocator) void {
        s.stream.synchronize() catch return;
        for (s.cases[0..s.made]) |*case| case.deinit();
        s.module.unload();
        a.destroy(s);
    }
};
