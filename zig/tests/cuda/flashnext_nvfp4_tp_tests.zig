//! Intermediate-channel NVFP4 partitions: exact source cuts, owned upload and FP32 expert partials against FP64.
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
const rank_width = width / 2;
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
    var left = try flash.modelopt_experts.loadShape(a, gpu.d, &source, "model.mlp", 2, 0, d, width, experts);
    defer left.free();
    var right = try flash.modelopt_experts.loadShape(a, gpu.d, &source, "model.mlp", 2, 1, d, width, experts);
    defer right.free();
    try check.expect(source.releases == 2 * experts * 3, "every source projection released after its owned rank cut", .{});
    const x = r.bfs(most * d, 0.125);
    const dx = try r.dev(std.mem.sliceAsBytes(x));
    const picks = try a.alloc(i32, most * slots);
    for (0..most) |row| {
        picks[row * slots] = @intCast(row % experts);
        picks[row * slots + 1] = @intCast((row + 1) % experts);
        picks[row * slots + 2] = experts; // shared expert is a separate format, skipped here
    }
    const dp = try r.dev(std.mem.sliceAsBytes(picks));
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
    var kernels = try fx.load(gpu.d, @intCast(try gpu.ctx.attribute(.multiprocessor_count)));
    defer kernels.unload();
    const act = try r.out(most * slots * rank_width * 2);
    const out = try r.out(most * slots * d * 4);
    const combined = try a.alloc(f32, most * slots * d);
    @memset(combined, 0);
    const reference_full = try a.alloc(f64, combined.len);
    @memset(reference_full, 0);
    const reference_abs = try a.alloc(f64, combined.len);
    @memset(reference_abs, 0);
    for ([_]*const flash.modelopt_experts.Owned{ &left, &right }, 0..) |owned, rank| {
        var all: []f32 = undefined;
        for ([_]usize{ most, 1, 7 }) |rows| {
            try gpu.d.check(gpu.d.api.cuMemsetD8Async(out, 0, most * slots * d * 4, r.s.handle), "cuMemsetD8Async");
            const items = cuda.grouped.maxItems(rows * slots, experts + 1, 16);
            try router.route(r.s, dp, rows * slots, experts + 1, 16, plan);
            try kernels.run(r.s, .up, .{ .x = dx, .stride = d, .slots = slots }, owned.view(), plan, act, items, experts);
            try kernels.run(r.s, .down_f32, .{ .x = act, .stride = rank_width }, owned.view(), plan, out, items, experts);
            if (rows == most) all = try r.back(f32, out, most * slots * d) else try r.sameRows("NVFP4 rank output rows", f32, out, all[0 .. rows * slots * d]);
        }
        for (0..most) |row| for (0..slots) |slot| {
            const e: usize = @intCast(picks[row * slots + slot]);
            if (e == experts) {
                for (all[(row * slots + slot) * d ..][0..d]) |v| try check.expect(v == 0, "shared slot untouched by NVFP4", .{});
                continue;
            }
            var activation: [rank_width]f64 = undefined;
            for (&activation, 0..) |*v, j| {
                const channel = rank * rank_width + j;
                var gate: f64 = 0;
                var up: f64 = 0;
                for (0..d) |k| {
                    gate += rig.bf(x[row * d + k]) * weight(&source, e, 0, channel, k);
                    up += rig.bf(x[row * d + k]) * weight(&source, e, 1, channel, k);
                }
                gate = rig.rbf(gate);
                up = rig.rbf(up);
                v.* = rig.rbf(rig.rbf(gate / (1 + @exp(-gate))) * up);
            }
            for (0..d) |j| {
                var want: f64 = 0;
                var mag: f64 = 0;
                for (activation, 0..) |v, k| {
                    const term = v * weight(&source, e, 2, j, rank * rank_width + k);
                    want += term;
                    mag += @abs(term);
                }
                const at = (row * slots + slot) * d + j;
                try r.within("NVFP4 rank FP32 down", all[at], want, rig.sumTol(mag, rank_width) + 1e-6);
                combined[at] += all[at];
                reference_full[at] += want;
                reference_abs[at] += mag;
            }
        };
    }
    for (combined, reference_full, reference_abs) |got, want, mag| try r.within("NVFP4 summed rank partials", got, want, rig.sumTol(mag, width) + 2e-6);
    r.report("NVFP4 TP cuts: both intermediate halves, exact source reader and packing, independent FP64 reference, FP32 partials and untouched shared slot; rows 1/7/17 equal");
}
