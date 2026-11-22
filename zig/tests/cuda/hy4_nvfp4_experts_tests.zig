//! Hy4-flash's NVFP4 routed experts (moe_nvfp4_cuda.zig): ModelOpt per-expert tensors through core's grouped
//! kernels, every pair FP64-bounded with Hy4's SwiGLU clamp, the narrow and wide plans, and a wrong shape refused.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const hy4 = @import("hy4_nvfp4");
const check = @import("check.zig");
const rig = @import("flashnext_rig.zig");
const latent = 256; // eight k32 groups
const inner = 160; // five 32-column blocks
const experts = 8;
const slots = 4;
const most = 300; // 1,200 pairs: past the one-block plan's 1,024

const Source = struct {
    codes: [experts][3][latent * inner / 2]u8 = undefined,
    scales: [experts][3][latent * inner / 16]u8 = undefined,
    globals: [experts][3][4]u8 = undefined,
    input: [4]u8 = undefined,
    wide_gate: bool = false, // expert 3's gate stored with 32 more rows than the layer has

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
        const n: usize = if (p == 2) latent else inner;
        const k: usize = if (p == 2) inner else latent;
        if (std.mem.endsWith(u8, name, ".weight")) {
            const rows: usize = if (s.wide_gate and e == 3 and p == 0) n + 32 else n;
            return tensor(.u8, &.{ rows, k / 2 }, &s.codes[e][p]);
        }
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
    const kk: usize = if (p == 2) inner else latent;
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
    var rng = std.Random.DefaultPrng.init(0x4f4);
    var r = try rig.Rig.init(gpu, a, rng.random());
    defer r.deinit();
    var source = Source.init(rng.random());

    // a wider stored gate is refused by name of shape, never sliced down to the layer's rows
    source.wide_gate = true;
    if (hy4.Routed.init(a, gpu.d, &source, "model.layers.1.mlp", latent, inner, experts, most, slots, 1)) |loaded| {
        var b = loaded;
        b.free();
        try check.expect(false, "a gate stored with more rows than the layer loaded", .{});
    } else |err| try check.expect(err == error.BadHy4ExpertShape, "a wider stored gate: {t}", .{err});
    source.wide_gate = false;

    const x = r.bfs(most * latent, 0.5);
    // the clamp at a quarter of the largest gate the first row's first pick makes, so some pairs cross it
    var largest: f64 = 0;
    for (0..inner) |j| {
        var g: f64 = 0;
        for (0..latent) |k| g += rig.bf(x[k]) * weight(&source, 0, 0, j, k);
        largest = @max(largest, @abs(g));
    }
    const limit: f32 = @floatCast(largest / 4);
    var routed = try hy4.Routed.init(a, gpu.d, &source, "model.layers.1.mlp", latent, inner, experts, most, slots, limit);
    defer routed.free();
    var kernels = try hy4.Kernels.load(gpu.d, @intCast(try gpu.ctx.attribute(.multiprocessor_count)));
    defer kernels.unload();
    const dx = try r.dev(std.mem.sliceAsBytes(x));
    const picks = try a.alloc(i32, most * slots);
    for (0..most) |row| for (0..slots) |slot| {
        // expert 0 in most rows (many 16-pair items), the rest spread; a row's picks are distinct
        picks[row * slots + slot] = @intCast(if (slot == 0 and row % 4 != 0) 0 else (row + slot * 3) % (experts - 1) + 1);
    };
    const dp = try r.dev(std.mem.sliceAsBytes(picks));
    var clamped: usize = 0;
    for ([_]usize{ 1, 5, 64, 255, 256, most }) |rows| {
        const y = try routed.run(&kernels, r.s, dx, dp, rows);
        const act = try r.back(u16, routed.act.ptr, rows * slots * inner);
        const out = try r.back(f32, y, rows * slots * latent);
        for (0..rows) |row| for (0..slots) |slot| {
            const pair = row * slots + slot;
            const e: usize = @intCast(picks[pair]);
            for (0..inner) |j| {
                var gate: f64 = 0;
                var up: f64 = 0;
                var mg: f64 = 0;
                var mu: f64 = 0;
                for (0..latent) |k| {
                    const xv = rig.bf(x[row * latent + k]);
                    gate += xv * weight(&source, e, 0, j, k);
                    up += xv * weight(&source, e, 1, j, k);
                    mg += @abs(xv * weight(&source, e, 0, j, k));
                    mu += @abs(xv * weight(&source, e, 1, j, k));
                }
                // Hy4's clamp: the gate from above, the up both ways, before silu(gate) * up
                const lim: f64 = limit;
                const g = @min(rig.rbf(gate), lim);
                const u = @min(@max(rig.rbf(up), -lim), lim);
                if (rows == most and rig.rbf(gate) > lim) clamped += 1;
                const want = rig.rbf(g / (1 + @exp(-g))) * u;
                const tol = @abs(want) * (1.0 / 128.0) + rig.sumTol(mg, latent) * 2 + rig.sumTol(mu, latent) * 2 + 1e-6;
                try r.within("Hy4 NVFP4 gate-up vs FP64", rig.bf(act[pair * inner + j]), want, tol);
            }
            for (0..latent) |j| {
                var want: f64 = 0;
                var mag: f64 = 0;
                for (0..inner) |k| {
                    const term = rig.bf(act[pair * inner + k]) * weight(&source, e, 2, j, k);
                    want += term;
                    mag += @abs(term);
                }
                try r.within("Hy4 NVFP4 fp32 down vs FP64", out[pair * latent + j], want, rig.sumTol(mag, inner) + 1e-6);
            }
        };
    }
    try check.expect(clamped > 0, "the clamp is crossed by some gate ({d} of {d} at {d} rows)", .{ clamped, most * slots * inner, most });
    r.report("Hy4 NVFP4 routed experts: 8 experts, 4 picks a row, rows 1/5/64/255/256/300 (narrow and wide plans), gate-up and fp32 down FP64-bounded with the SwiGLU clamp crossed; a wider stored gate refused");
}
