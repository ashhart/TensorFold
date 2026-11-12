//! Expert-parallel NVFP4 ranks: whole experts a rank, fn_ep_picks, FP64 rows, and their sum equal to one GPU's bytes.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const flash = @import("flashnext");
const check = @import("check.zig");
const rig = @import("flashnext_rig.zig");
const d = 64;
const width = 64;
const experts = 2;
const slots = 3;
const most = 17;
const fx = cuda.experts.Nvfp4Experts;

const Source = struct {
    codes: [experts][3][d * width / 2]u8 = undefined,
    scales: [experts][3][d * width / 16]u8 = undefined,
    globals: [experts][3][4]u8 = undefined,
    input: [4]u8 = undefined,
    releases: usize = 0,

    fn init() Source {
        var s: Source = .{};
        for (0..experts) |e| for (0..3) |p| {
            for (&s.codes[e][p], 0..) |*v, i| v.* = @truncate(i * 13 + e * 37 + p * 19);
            for (&s.scales[e][p], 0..) |*v, i| v.* = @intCast(0x20 + (i % 3) * 8);
            std.mem.writeInt(u32, &s.globals[e][p], @bitCast(@as(f32, if (p == 2) 0.25 else 0.125)), .little);
        };
        std.mem.writeInt(u32, &s.input, @bitCast(@as(f32, 1)), .little);
        return s;
    }

    pub fn has(_: *Source, name: []const u8) bool {
        return std.mem.endsWith(u8, name, ".weight") or std.mem.endsWith(u8, name, ".weight_scale") or
            std.mem.endsWith(u8, name, ".weight_scale_2") or std.mem.endsWith(u8, name, ".input_scale");
    }

    pub fn get(s: *Source, name: []const u8) !core.safetensors.Tensor {
        const marker = std.mem.indexOf(u8, name, ".experts.") orelse return error.MissingTensor;
        const tail = name[marker + ".experts.".len ..];
        const end = std.mem.indexOfScalar(u8, tail, '.') orelse return error.MissingTensor;
        const e = try std.fmt.parseInt(usize, tail[0..end], 10);
        if (e >= experts) return error.MissingTensor;
        const p: usize = if (std.mem.indexOf(u8, tail, ".gate_proj.") != null) 0 else if (std.mem.indexOf(u8, tail, ".up_proj.") != null) 1 else if (std.mem.indexOf(u8, tail, ".down_proj.") != null) 2 else return error.MissingTensor;
        if (std.mem.endsWith(u8, name, ".weight")) return tensor(.u8, &.{ width, d / 2 }, &s.codes[e][p]);
        if (std.mem.endsWith(u8, name, ".weight_scale")) return tensor(.f8_e4m3, &.{ width, d / 16 }, &s.scales[e][p]);
        if (std.mem.endsWith(u8, name, ".weight_scale_2")) return tensor(.f32, &.{}, &s.globals[e][p]);
        if (std.mem.endsWith(u8, name, ".input_scale")) return tensor(.f32, &.{}, &s.input);
        return error.MissingTensor;
    }

    pub fn release(s: *Source) void {
        s.releases += 1;
    }
};

fn tensor(dtype: core.safetensors.DType, shape: []const usize, bytes: []const u8) core.safetensors.Tensor {
    var t: core.safetensors.Tensor = .{ .dtype = dtype, .rank = @intCast(shape.len), .shape = @splat(1), .bytes = bytes };
    @memcpy(t.shape[0..shape.len], shape);
    return t;
}

fn weight(s: *const Source, e: usize, p: usize, n: usize, k: usize) f64 {
    const byte = s.codes[e][p][n * (d / 2) + k / 2];
    const code: u4 = if (k % 2 == 0) @truncate(byte) else @truncate(byte >> 4);
    const scale = s.scales[e][p][n * (d / 16) + k / 16];
    const global: f32 = @bitCast(std.mem.readInt(u32, &s.globals[e][p], .little));
    return core.quant.nvfp4.e2m1(code) * core.quant.fp8.e4m3(scale) * global;
}

pub fn run(gpu: check.Gpu) !void {
    var arena = std.heap.ArenaAllocator.init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var rng = std.Random.DefaultPrng.init(0x2f4);
    var r = try rig.Rig.init(gpu, a, rng.random());
    defer r.deinit();
    var source = Source.init();
    var whole = try flash.modelopt_experts.loadShape(a, gpu.d, &source, "model.mlp", 1, 0, d, width, experts);
    defer whole.free();
    const one_gpu_reads = source.releases;
    var ranks: [2]flash.modelopt_experts.Owned = undefined;
    for (&ranks, 0..) |*o, rank| o.* = try flash.modelopt_experts.loadShape(a, gpu.d, &source, "model.mlp", 2, rank, d, width, experts);
    defer for (&ranks) |*o| o.free();
    try check.expect(source.releases - one_gpu_reads == one_gpu_reads, "the two ranks read every expert once between them", .{});
    for (ranks, 0..) |o, rank| try check.expect(o.first == rank and o.count == 1 and o.width == width, "rank {d} owns one whole expert", .{rank});
    const x = r.bfs(most * d, 0.125);
    const dx = try r.dev(std.mem.sliceAsBytes(x));
    const picks = try a.alloc(i32, most * slots);
    for (0..most) |row| {
        picks[row * slots] = @intCast(row % experts);
        picks[row * slots + 1] = @intCast((row + 1) % experts);
        picks[row * slots + 2] = experts; // shared expert is a separate format, skipped here
    }
    const dp = try r.dev(std.mem.sliceAsBytes(picks));
    const local = try r.out(most * slots * 4);
    const plan: cuda.grouped.Plan = .{
        .members = try r.out(most * slots * 4),
        .items = try r.out((most * slots + experts + 1) * 12),
        .counts = try r.out(8),
        .rank = try r.out(most * slots * 4),
        .hist = try r.out((experts + 1) * 4),
    };
    var mod = try cuda.Module.load(gpu.d, cuda.kernels.experts);
    defer mod.unload();
    const router = try cuda.grouped.Router.resolve(mod);
    var batch = try cuda.Module.load(gpu.d, cuda.kernels.flashnext_batch);
    defer batch.unload();
    const ep_picks = try batch.function("fn_ep_picks");
    var kernels = try fx.loadWarps(gpu.d, @intCast(try gpu.ctx.attribute(.multiprocessor_count)), 4);
    defer kernels.unload();
    const act = try r.out(most * slots * width * 2);
    const out = try r.out(most * slots * d * 4);
    const pairs = most * slots;
    // one GPU: both experts, global picks, the shared slot skipped
    try gpu.d.check(gpu.d.api.cuMemsetD8Async(out, 0, pairs * d * 4, r.s.handle), "cuMemsetD8Async");
    try router.route(r.s, dp, pairs, experts + 1, 16, plan);
    try kernels.run(r.s, .up, .{ .x = dx, .stride = d, .slots = slots }, whole.view(), plan, act, cuda.grouped.maxItems(pairs, experts + 1, 16), experts);
    try kernels.run(r.s, .down_f32, .{ .x = act, .stride = width }, whole.view(), plan, out, cuda.grouped.maxItems(pairs, experts + 1, 16), experts);
    const one = try r.back(f32, out, pairs * d);
    const summed = try a.alloc(f32, pairs * d);
    @memset(summed, 0);
    for (ranks, 0..) |o, rank| {
        var args: cuda.Args = .{};
        args.add(dp);
        args.add(local);
        for ([_]usize{ pairs, o.first, o.count }) |v| args.add(@as(i32, @intCast(v)));
        try cuda.launch.launch(ep_picks, .{ .grid = .{ .x = 1 }, .block = .{ .x = 256 } }, r.s, &args);
        const got_picks = try r.back(i32, local, pairs);
        for (picks, got_picks) |g, l| {
            const want: i32 = if (g >= o.first and g < o.first + o.count) g - @as(i32, @intCast(o.first)) else @intCast(o.count);
            try check.expect(l == want, "fn_ep_picks maps global pick {d} to {d} on rank {d}, not {d}", .{ g, want, rank, l });
        }
        try gpu.d.check(gpu.d.api.cuMemsetD8Async(out, 0, pairs * d * 4, r.s.handle), "cuMemsetD8Async");
        const skip: i32 = @intCast(o.count);
        try router.route(r.s, local, pairs, o.count + 1, 16, plan);
        try kernels.run(r.s, .up, .{ .x = dx, .stride = d, .slots = slots }, o.view(), plan, act, cuda.grouped.maxItems(pairs, o.count + 1, 16), skip);
        try kernels.run(r.s, .down_f32, .{ .x = act, .stride = width }, o.view(), plan, out, cuda.grouped.maxItems(pairs, o.count + 1, 16), skip);
        const all = try r.back(f32, out, pairs * d);
        for (0..most) |row| for (0..slots) |slot| {
            const e: usize = @intCast(picks[row * slots + slot]);
            const at = (row * slots + slot) * d;
            if (e == experts or e < o.first or e >= o.first + o.count) {
                for (all[at..][0..d]) |v| try check.expect(@as(u32, @bitCast(v)) == 0, "a pair rank {d} does not own stays +0", .{rank});
                continue;
            }
            var activation: [width]f64 = undefined;
            for (&activation, 0..) |*v, j| {
                var gate: f64 = 0;
                var up: f64 = 0;
                for (0..d) |k| {
                    gate += rig.bf(x[row * d + k]) * weight(&source, e, 0, j, k);
                    up += rig.bf(x[row * d + k]) * weight(&source, e, 1, j, k);
                }
                gate = rig.rbf(gate);
                up = rig.rbf(up);
                v.* = rig.rbf(rig.rbf(gate / (1 + @exp(-gate))) * up);
            }
            for (0..d) |j| {
                var want: f64 = 0;
                var mag: f64 = 0;
                for (activation, 0..) |v, k| {
                    const term = v * weight(&source, e, 2, j, k);
                    want += term;
                    mag += @abs(term);
                }
                try r.within("NVFP4 whole-expert FP32 down", all[at + j], want, rig.sumTol(mag, width) + 1e-6);
            }
        };
        for (summed, all) |*s, v| s.* += v; // the all-reduce: one rank's row plus the other's +0
    }
    try check.sameBytes("two expert-parallel ranks summed equal one GPU's bytes", std.mem.sliceAsBytes(summed), std.mem.sliceAsBytes(one));
    r.report("NVFP4 expert parallel: whole experts a rank, fn_ep_picks' local ids, FP64 rows, untouched skipped pairs, and the ranks' sum byte-equal to one GPU");
}
