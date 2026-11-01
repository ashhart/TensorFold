//! Flash Next's block-FP8 MTP experts at the head's shapes: loader and moe_native.fp8, decode and prompt, vs FP64.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const flash = @import("flashnext");
const check = @import("check.zig");
const rig = @import("flashnext_rig.zig");

const experts = 512; // routed FP8 experts
const entries = experts + 1; // routing entries: the shared slot last, which the FP8 call skips
const NI = 640; // width
const D = 2560; // dims
const slots = 11; // ten routed picks a row, then the shared slot
const blk = 128;
const most = 200;
const projs = [_][]const u8{ "gate_proj", "up_proj", "down_proj" };

/// 512 experts' E4M3 codes [expert][gate, up, down][n][k] and BF16 block scales [expert][proj][n/128][k/128].
const Source = struct {
    codes: []u8,
    scales: []u16,

    const per = NI * D; // codes a projection (gate / up [NI, D], down [D, NI])
    const blocks = (NI / blk) * (D / blk);

    fn init(a: std.mem.Allocator, r: std.Random) !Source {
        const s: Source = .{ .codes = try a.alloc(u8, experts * 3 * per), .scales = try a.alloc(u16, experts * 3 * blocks) };
        r.bytes(s.codes);
        for (s.codes) |*c| if (c.* & 0x7f == 0x7f) {
            c.* ^= 1; // E4M3's NaN codes: the checkpoint holds none
        };
        for (s.scales) |*v| v.* = rig.bf16(0.001 + r.float(f32) * 0.01);
        return s;
    }

    fn dims(p: usize) [2]usize {
        return if (p == 2) .{ D, NI } else .{ NI, D };
    }

    /// The f64 value of projection `p` of expert `e` at (row, col).
    fn value(s: *const Source, lut: *const [256]f64, e: usize, p: usize, row: usize, col: usize) f64 {
        const nk = dims(p);
        return lut[s.codes[(e * 3 + p) * per + row * nk[1] + col]] * rig.bf(s.scales[(e * 3 + p) * blocks + (row / blk) * (nk[1] / blk) + col / blk]);
    }

    pub fn has(_: *Source, name: []const u8) bool {
        return std.mem.indexOf(u8, name, ".experts.") != null;
    }

    pub fn get(s: *Source, name: []const u8) !core.safetensors.Tensor {
        const marker = std.mem.indexOf(u8, name, ".experts.") orelse return error.MissingTensor;
        const tail = name[marker + ".experts.".len ..];
        const dot = std.mem.indexOfScalar(u8, tail, '.') orelse return error.MissingTensor;
        const e = try std.fmt.parseInt(usize, tail[0..dot], 10);
        if (e >= experts) return error.MissingTensor;
        for (projs, 0..) |proj, p| {
            const rest = tail[dot + 1 ..];
            if (!std.mem.startsWith(u8, rest, proj)) continue;
            const nk = dims(p);
            if (std.mem.eql(u8, rest[proj.len..], ".weight")) return tensor(.f8_e4m3, nk, s.codes[(e * 3 + p) * per ..][0..per]);
            if (std.mem.eql(u8, rest[proj.len..], ".weight_scale_inv")) return tensor(.bf16, .{ nk[0] / blk, nk[1] / blk }, std.mem.sliceAsBytes(s.scales[(e * 3 + p) * blocks ..][0..blocks]));
        }
        return error.MissingTensor;
    }

    pub fn release(_: *Source) void {}
};

fn tensor(dtype: core.safetensors.DType, shape: [2]usize, bytes: []const u8) core.safetensors.Tensor {
    var t: core.safetensors.Tensor = .{ .dtype = dtype, .rank = 2, .shape = @splat(1), .bytes = bytes };
    t.shape[0] = shape[0];
    t.shape[1] = shape[1];
    return t;
}

/// SwiGLU as the kernels round it: gate and up to bf16, silu to bf16, the product (no clamp).
fn swiglu(g: f64, u: f64) f64 {
    const gv = rig.rbf(g);
    const uv = rig.rbf(u);
    return rig.rbf(gv / (1 + @exp(-gv))) * uv;
}

/// One thread's pairs vs FP64: gate-up from x, down from the kernel's bf16 activations (quant_ref_experts' bounds).
const Ref = struct {
    s: *const Source,
    lut: *const [256]f64,
    x: []const u16,
    picks: []const i32,
    act: []const u16,
    y: []const f32,
    lo: usize,
    hi: usize,
    worst: [2]f64 = .{ 0, 0 },
    bad: [2]usize = .{ 0, 0 },

    fn note(f: *Ref, which: usize, got: f64, want: f64, tol: f64) void {
        const q = @abs(got - want) / tol;
        if (!(q <= 1)) f.bad[which] += 1;
        if (q > f.worst[which] or std.math.isNan(q)) f.worst[which] = if (std.math.isNan(q)) std.math.inf(f64) else q;
    }

    fn run(f: *Ref) void {
        for (f.lo..f.hi) |p| {
            const e: usize = @intCast(f.picks[p]);
            if (e == experts) continue;
            const xr = f.x[(p / slots) * D ..][0..D];
            for (0..NI) |j| {
                var gs: f64 = 0;
                var us: f64 = 0;
                var abs: f64 = 0;
                for (0..D) |i| {
                    const xv = rig.bf(xr[i]);
                    const g = xv * f.s.value(f.lut, e, 0, j, i);
                    const u = xv * f.s.value(f.lut, e, 1, j, i);
                    gs += g;
                    us += u;
                    abs += @abs(g) + @abs(u);
                }
                const a = swiglu(gs, us);
                // a bf16 step either side of each of the three roundings, and fp32 sums
                f.note(0, rig.bf(f.act[p * NI + j]), a, @abs(a) * std.math.pow(f64, 2, -6) + abs * std.math.pow(f64, 2, -12) + 1e-30);
            }
            for (0..D) |j| {
                var sum: f64 = 0;
                var abs: f64 = 0;
                for (0..NI) |i| {
                    const t = rig.bf(f.act[p * NI + i]) * f.s.value(f.lut, e, 2, j, i);
                    sum += t;
                    abs += @abs(t);
                }
                f.note(1, f.y[p * D + j], sum, abs * std.math.pow(f64, 2, -12) + 1e-30);
            }
        }
    }
};

/// Every row's ten distinct routed experts, then the shared slot.
fn picksOf(a: std.mem.Allocator, r: std.Random, rows: usize) ![]i32 {
    const out = try a.alloc(i32, rows * slots);
    for (0..rows) |row| {
        for (0..slots - 1) |j| {
            var e: i32 = @intCast(r.uintLessThan(usize, experts));
            while (std.mem.indexOfScalar(i32, out[row * slots ..][0..j], e) != null) e = @intCast(r.uintLessThan(usize, experts));
            out[row * slots + j] = e;
        }
        out[row * slots + slots - 1] = experts;
    }
    return out;
}

pub fn run(gpu: check.Gpu) !void {
    // decode rows take tile 16 and prompt rows tile 64 at the default fact rows (64)
    if (std.c.getenv("TF_FLASHNEXT_FACT_ROWS") != null) return check.expect(false, "flashnext-fp8-experts runs with TF_FLASHNEXT_FACT_ROWS unset", .{});
    var arena = std.heap.ArenaAllocator.init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var rng = std.Random.DefaultPrng.init(0xf8e4);
    var r = try rig.Rig.init(gpu, a, rng.random());
    defer r.deinit();
    var source = try Source.init(a, rng.random());
    var lut: [256]f64 = undefined;
    for (&lut, 0..) |*v, c| v.* = core.quant.fp8.e4m3(@intCast(c));
    // the production loader (block-FP8 branch) and dispatch: the batch pass's kernels, the layer's view
    var owned = try flash.modelopt_experts.loadShape(gpu.gpa, gpu.d, &source, "language_model.mtp.layers.0.mlp", 1, 0, D, NI, experts);
    defer owned.free();
    const layer = owned.view();
    try check.expect(layer.format == .fp8g and layer.width == NI and layer.dims == D and layer.experts == experts, "the head's experts load as block FP8, {d} x {d}, {d} experts", .{ NI, D, experts });
    var k = try flash.batch_state.Kernels.load(gpu.d);
    defer k.deinit();
    const x = r.bfs(most * D, 1.0);
    const dx = try r.dev(std.mem.sliceAsBytes(x));
    var worst: [2]f64 = .{ 0, 0 };
    var checked: usize = 0;
    for ([_]usize{ 1, 3, 16, 64, most }) |rows| {
        const pairs = rows * slots;
        const picks = try picksOf(a, r.rng, rows);
        const plan: cuda.grouped.Plan = .{
            .members = try r.out(pairs * 4),
            .items = try r.out(((pairs + 15) / 16 + entries) * 12),
            .counts = try r.out(8),
            .rank = try r.out(pairs * 4),
            .hist = try r.out(((pairs + cuda.grouped.small - 1) / cuda.grouped.small) * entries * 4),
        };
        // every byte 0xff: a pair the call writes changes, the shared slot's must not
        const act = try r.out(pairs * NI * 2);
        const y = try r.out(pairs * D * 4);
        try flash.moe_native.fp8(&k, r.s, try r.dev(std.mem.sliceAsBytes(picks)), dx, layer, plan, pairs, rows, y, act);
        const got_act = try r.back(u16, act, pairs * NI);
        const got_y = try r.back(f32, y, pairs * D);
        for (picks, 0..) |e, p| if (e == experts) {
            for (got_act[p * NI ..][0..NI]) |v| try check.expect(v == 0xffff, "{d} rows: the shared slot's activations untouched", .{rows});
            for (std.mem.sliceAsBytes(got_y[p * D ..][0..D])) |v| try check.expect(v == 0xff, "{d} rows: the shared slot's down rows untouched", .{rows});
        };
        const threads = @min(32, @max(1, std.Thread.getCpuCount() catch 1));
        var refs: [32]Ref = undefined;
        var handles: [32]?std.Thread = @splat(null);
        const share = (pairs + threads - 1) / threads;
        for (0..threads) |t| {
            refs[t] = .{ .s = &source, .lut = &lut, .x = x, .picks = picks, .act = got_act, .y = got_y, .lo = @min(pairs, t * share), .hi = @min(pairs, (t + 1) * share) };
            handles[t] = std.Thread.spawn(.{}, Ref.run, .{&refs[t]}) catch blk: {
                refs[t].run();
                break :blk null;
            };
        }
        for (handles[0..threads]) |h| if (h) |th| th.join();
        var bad: [2]usize = .{ 0, 0 };
        for (refs[0..threads]) |f| for (0..2) |i| {
            bad[i] += f.bad[i];
            worst[i] = @max(worst[i], f.worst[i]);
        };
        const path = if (rows >= 64) "prompt (tile 64)" else "decode (tile 16)";
        try check.expect(bad[0] == 0, "fp8 head experts {s}, {d} rows: gate-up within FP64 tolerance ({d} outside, worst {d:.3})", .{ path, rows, bad[0], worst[0] });
        try check.expect(bad[1] == 0, "fp8 head experts {s}, {d} rows: fp32 down within FP64 tolerance ({d} outside, worst {d:.3})", .{ path, rows, bad[1], worst[1] });
        checked += rows * (slots - 1);
    }
    check.pass("flashnext-fp8-experts: 512 block-FP8 experts at {d} x {d} through moe_native.fp8, decode rows 1/3/16 and prompt rows 64/200; {d} routed pairs within FP64 (worst gate-up {d:.3}, down {d:.3} tolerances), the shared slot untouched", .{ NI, D, checked, worst[0], worst[1] });
}
