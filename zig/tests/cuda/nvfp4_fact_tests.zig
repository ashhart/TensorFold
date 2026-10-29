//! NVFP4 experts' prompt tiles with the decode form's sums (runFact): bytes equal to run at every row, FP64-bounded.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const flash = @import("flashnext");
const check = @import("check.zig");
const rig = @import("flashnext_rig.zig");
const d = 256; // eight k32 groups: four 64-input stages, so every pipeline depth is crossed
const width = 160; // five 32-column blocks: gate-up's 4-warp CTA holds a partial second block row
const experts = 6;
const slots = 3; // two routed picks and the skipped shared slot
const most = 300; // 600 routed pairs: items of 64 pairs and their remainders
const fx = cuda.experts.Nvfp4Experts;

const Source = struct {
    codes: [experts][3][d * width / 2]u8 = undefined,
    scales: [experts][3][d * width / 16]u8 = undefined,
    globals: [experts][3][4]u8 = undefined,
    input: [4]u8 = undefined,

    fn init(rng: std.Random) Source {
        var s: Source = .{};
        for (0..experts) |e| for (0..3) |p| {
            rng.bytes(&s.codes[e][p]);
            for (&s.scales[e][p]) |*v| v.* = 0x20 + rng.uintLessThan(u8, 0x28);
            std.mem.writeInt(u32, &s.globals[e][p], @bitCast(@as(f32, 0.01 + 0.02 * rng.float(f32))), .little);
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
        const n: usize = if (p == 2) d else width;
        const k: usize = if (p == 2) width else d;
        if (std.mem.endsWith(u8, name, ".weight")) return tensor(.u8, &.{ n, k / 2 }, &s.codes[e][p]);
        if (std.mem.endsWith(u8, name, ".weight_scale")) return tensor(.f8_e4m3, &.{ n, k / 16 }, &s.scales[e][p]);
        if (std.mem.endsWith(u8, name, ".weight_scale_2")) return tensor(.f32, &.{}, &s.globals[e][p]);
        if (std.mem.endsWith(u8, name, ".input_scale")) return tensor(.f32, &.{}, &s.input);
        return error.MissingTensor;
    }

    pub fn release(_: *Source) void {}
};

fn tensor(dtype: core.safetensors.DType, shape: []const usize, bytes: []const u8) core.safetensors.Tensor {
    var t: core.safetensors.Tensor = .{ .dtype = dtype, .rank = @intCast(shape.len), .shape = @splat(1), .bytes = bytes };
    @memcpy(t.shape[0..shape.len], shape);
    return t;
}

/// Projection p's weight [n][k] of expert e, dequantized exactly.
fn weight(s: *const Source, e: usize, p: usize, n: usize, k: usize) f64 {
    const kk: usize = if (p == 2) width else d;
    const byte = s.codes[e][p][n * (kk / 2) + k / 2];
    const code: u4 = if (k % 2 == 0) @truncate(byte) else @truncate(byte >> 4);
    const scale = s.scales[e][p][n * (kk / 16) + k / 16];
    const global: f32 = @bitCast(std.mem.readInt(u32, &s.globals[e][p], .little));
    return core.quant.nvfp4.e2m1(code) * core.quant.fp8.e4m3(scale) * global;
}

pub fn run(gpu: check.Gpu) !void {
    var arena = std.heap.ArenaAllocator.init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var rng = std.Random.DefaultPrng.init(0xfac7);
    var r = try rig.Rig.init(gpu, a, rng.random());
    defer r.deinit();
    var source = Source.init(rng.random());
    var owned = try flash.modelopt_experts.loadShape(a, gpu.d, &source, "model.mlp", 1, 0, d, width, experts);
    defer owned.free();
    const x = r.bfs(most * d, 0.5);
    const dx = try r.dev(std.mem.sliceAsBytes(x));
    const picks = try a.alloc(i32, most * slots);
    for (0..most) |row| {
        // expert 0 takes most pairs (several 64-pair items), the others a few each
        picks[row * slots] = if (row % 3 == 0) @intCast(1 + row % (experts - 1)) else 0;
        picks[row * slots + 1] = @intCast(1 + (row * 7) % (experts - 1));
        picks[row * slots + 2] = experts; // the shared expert, skipped
    }
    const dp = try r.dev(std.mem.sliceAsBytes(picks));
    const pairs_most = most * slots;
    const plan: cuda.grouped.Plan = .{
        .members = try r.out(pairs_most * 4),
        .items = try r.out(cuda.grouped.maxItems(pairs_most, experts + 1, 16) * 12),
        .counts = try r.out(8),
        .rank = try r.out(pairs_most * 4),
        .hist = try r.out(((pairs_most + 1023) / 1024) * (experts + 1) * 4),
    };
    var mod = try cuda.Module.load(gpu.d, cuda.kernels.experts);
    defer mod.unload();
    const router = try cuda.grouped.Router.resolve(mod);
    var kernels = try fx.load(gpu.d, @intCast(try gpu.ctx.attribute(.multiprocessor_count)));
    defer kernels.unload();
    var tile_k = try cuda.experts_tile.Tile.load(gpu.d, @intCast(try gpu.ctx.attribute(.multiprocessor_count)), .both);
    defer tile_k.unload();
    const act = [3]u64{ try r.out(pairs_most * width * 2), try r.out(pairs_most * width * 2), try r.out(pairs_most * width * 2) };
    const out = [3]u64{ try r.out(pairs_most * d * 4), try r.out(pairs_most * d * 4), try r.out(pairs_most * d * 4) };
    const tile = cuda.experts.promptTile(.nvfp4);
    var control_differs = false;
    for ([_]usize{ 1, 7, 17, 22, 64, 65, 130, most }) |rows| {
        const pairs = rows * slots;
        for (0..3) |form| {
            for ([_]u64{ act[form], out[form] }, [_]usize{ width * 2, d * 4 }) |p, bytes| try gpu.d.check(gpu.d.api.cuMemsetD8Async(p, 0xff, pairs * bytes, r.s.handle), "cuMemsetD8Async");
            const t: usize = if (form == 0) 16 else tile;
            const items = cuda.grouped.maxItems(pairs, experts + 1, t);
            try router.route(r.s, dp, pairs, experts + 1, t, plan);
            if (form == 2) {
                try tile_k.run(r.s, .up, .{ .x = dx, .stride = d, .slots = slots }, owned.view(), plan, act[form], items, experts);
                try tile_k.run(r.s, .down_f32, .{ .x = act[form], .stride = width }, owned.view(), plan, out[form], items, experts);
            } else {
                const f = if (form == 0) &fx.run else &fx.runFact;
                try f(&kernels, r.s, .up, .{ .x = dx, .stride = d, .slots = slots }, owned.view(), plan, act[form], items, experts);
                try f(&kernels, r.s, .down_f32, .{ .x = act[form], .stride = width }, owned.view(), plan, out[form], items, experts);
            }
        }
        const act0 = try r.back(u16, act[0], pairs * width);
        const act1 = try r.back(u16, act[1], pairs * width);
        const y0 = try r.back(f32, out[0], pairs * d);
        const y1 = try r.back(f32, out[1], pairs * d);
        // the shared slots are left alone by both forms (still the 0xff fill), so compare every byte
        try check.sameBytes("NVFP4 fact gate-up rows equal the decode form's bytes", std.mem.sliceAsBytes(act1), std.mem.sliceAsBytes(act0));
        try check.sameBytes("NVFP4 fact fp32 down rows equal the decode form's bytes", std.mem.sliceAsBytes(y1), std.mem.sliceAsBytes(y0));
        const k32_act = try r.back(u16, act[2], pairs * width);
        const k32_y = try r.back(f32, out[2], pairs * d);
        try check.sameBytes("K32 tile gate-up equals the decode form", std.mem.sliceAsBytes(k32_act), std.mem.sliceAsBytes(act0));
        try check.sameBytes("K32 tile fp32 down equals the decode form", std.mem.sliceAsBytes(k32_y), std.mem.sliceAsBytes(y0));
        // control: the existing prompt tile (scaled codes, one sum over K) on the same plan must part somewhere
        {
            const items = cuda.grouped.maxItems(pairs, experts + 1, tile);
            try router.route(r.s, dp, pairs, experts + 1, tile, plan);
            try kernels.runPrompt(r.s, .up, .{ .x = dx, .stride = d, .slots = slots }, owned.view(), plan, act[1], items, experts);
            const ctl = try r.back(u16, act[1], pairs * width);
            if (!std.mem.eql(u16, ctl, act0)) control_differs = true;
        }
        // FP64: every routed pair's down output within the decode form's own bound (bf16 activations, fp32 down)
        for (0..rows) |row| for (0..slots) |slot| {
            const e: usize = @intCast(picks[row * slots + slot]);
            if (e == experts) continue;
            var activation: [width]f64 = undefined;
            for (&activation, 0..) |*v, j| v.* = rig.bf(act0[(row * slots + slot) * width + j]);
            // gate-up against FP64 on the bf16 inputs, within the bf16 output's half ulp plus the sum's bound
            for (0..width) |j| {
                var gate: f64 = 0;
                var up: f64 = 0;
                var mg: f64 = 0;
                var mu: f64 = 0;
                for (0..d) |k| {
                    const xv = rig.bf(x[row * d + k]);
                    gate += xv * weight(&source, e, 0, j, k);
                    up += xv * weight(&source, e, 1, j, k);
                    mg += @abs(xv * weight(&source, e, 0, j, k));
                    mu += @abs(xv * weight(&source, e, 1, j, k));
                }
                const g = rig.rbf(gate);
                const want = rig.rbf(g / (1 + @exp(-g))) * rig.rbf(up);
                const tol = @abs(want) * (1.0 / 128.0) + rig.sumTol(mg, d) * 2 + rig.sumTol(mu, d) * 2 + 1e-6;
                try r.within("NVFP4 fact gate-up vs FP64", activation[j], want, tol);
            }
            for (0..d) |j| {
                var want: f64 = 0;
                var mag: f64 = 0;
                for (activation, 0..) |v, k| {
                    const term = v * weight(&source, e, 2, j, k);
                    want += term;
                    mag += @abs(term);
                }
                try r.within("NVFP4 fact fp32 down vs FP64", y1[(row * slots + slot) * d + j], want, rig.sumTol(mag, width) + 1e-6);
            }
        };
    }
    try check.expect(control_differs, "control: the scaled-code prompt tile parts from the decode form's bytes (the equality check is sensitive)", .{});
    r.report("NVFP4 fact tiles: gate-up and fp32 down bit-equal to the decode form at rows 1/7/17/22/64/65/130/300 (64-pair items vs 16), FP64-bounded; control red");
}
