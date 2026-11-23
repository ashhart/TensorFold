//! Hy4-flash's route kernels under the contract (metal_hy4.zig's group), compiled as metal_routed.zig loads them.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const contract = @import("contract");
const sources = @import("kernel_sources");
const rig_ = @import("rig.zig");
const h = @import("metal_hy4.zig");
const sel = @import("metal_hy4_route_select.zig");
const sums = @import("metal_hy4_route_sum.zig");

const c = contract.spec;
const g = contract.gates;
const fl = h.fl;
const Range = fl.Range;
const route = tf.moe_route;

/// The router's and the routed sum's widths as metal_routed.zig reads them from the checkpoint's config.json.
pub const Dims = struct {
    hidden: u32,
    experts: u32,
    topk: u32,
    /// The routed experts' latent width (routed_expert_hidden_size): the weighted sum's row.
    width: u32,
    scale: f32,

    pub fn read(a: std.mem.Allocator) !Dims {
        const p = try h.text(a);
        defer p.deinit();
        const t = p.value.text_config;
        if (t.hidden_size != h.hidden or t.num_experts_per_tok > sel.max_topk or t.n_routed_experts > sel.max_experts) return error.NotHy4Router;
        return .{ .hidden = t.hidden_size, .experts = t.n_routed_experts, .topk = t.num_experts_per_tok, .width = t.routed_expert_hidden_size, .scale = t.routed_scaling_factor };
    }

    /// Kernels.loadImpl's route.Shape.
    pub fn shape(d: Dims) route.Shape {
        return .{ .hidden = d.hidden, .experts = d.experts, .topk = d.topk, .max_rows = h.hy.metal_routed.max_rows };
    }
};

/// The loader's router library: core moe_route at Hy4's shape with route_rows.metal after it, then `e` applied.
pub fn routeText(a: std.mem.Allocator, d: Dims, e: ?fl.Edit) ![]u8 {
    const core = try route.source(a, d.shape());
    defer a.free(core);
    const whole = try std.mem.concat(a, u8, &.{ core, sources.hy4_flash_route_rows });
    defer a.free(whole);
    return fl.edited(a, whole, e);
}

/// One function of `text` compiled with the loader's options (MLX's: safe math, fast fp32 functions).
pub fn compile(device: mtl.Device, text: []const u8, function: []const u8) !mtl.Pipeline {
    const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
    defer lib.deinit();
    return mtl.Pipeline.init(device, lib, function, false);
}

/// The planted controls: logits summed in half, a NaN key picked, rank ties to the higher id, the sum in half.
pub const half_logits: fl.Edit = .{ .kernel = "tf_route_logits", .from = "float acc[4][4];", .to = "half acc[4][4];" };
pub const nan_picked: fl.Edit = .{ .kernel = "tf_hy4_route_select_rows", .from = "(c[j] > best ||", .to = "(!(c[j] <= best) ||" };
pub const ties_high: fl.Edit = .{ .kernel = "tf_hy4_route_select_rank", .from = "(o == c && j < id)", .to = "(o == c && j > id)" };
pub const half_sum: fl.Edit = .{
    .kernel = "tf_hy4_routed_sum",
    .from = "sum = fma(weights[row + slot], parts[(row + slot) * width + at.x], sum);",
    .to = "sum = float(half(fma(weights[row + slot], parts[(row + slot) * width + at.x], sum)));",
};

// --- tf_route_logits: logits [rows, 512] = x W^T over the packed bf16 router, fp32 sums, fp32 out ------------------

/// A lane's chain (hidden / 1024 slices of 4 products each), the simd_sum tree, then the eight partials in order.
fn logitsDepth(s: c.Shape) usize {
    return 4 * (s.k / 1024) + 5 + 7;
}

pub const Logits = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    d: Dims,
    served: [4]c.Shape,
    control: ?g.Gate = null,
    base_buf: [128]u8 = undefined,
    label: [192]u8 = undefined,
    name: []const u8 = "",
    rows: usize = 0,
    fill: g.Fill = .random,
    /// The bf16 rows and router widened to fp32 (exact), and each output's float64 dot and its terms' magnitude.
    x: []f32 = &.{},
    w: []f32 = &.{},
    dots: [][2]f64 = &.{},
    ins: [2]rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    pub fn base(d: Dims, buf: []u8) []const u8 {
        return std.fmt.bufPrint(buf, "metal hy4 moe_route tf_route_logits hidden {d} experts {d}", .{ d.hidden, d.experts }) catch unreachable;
    }

    /// Rows: a window, a decode row, DFly2's window, a prompt chunk (tensor units); n: experts; k: the hidden width.
    pub fn spec(name: []const u8, served: []const c.Shape) c.Spec {
        return .{
            .name = name,
            .accum = .f32,
            .depth = logitsDepth,
            .out = .f32,
            .tiles = .{ .{ .size = 4, .step = 1, .min = 1 }, .{}, .{} },
            .served = served,
        };
    }
    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, d: Dims, control: bool) !Logits {
        const text = try routeText(a, d, if (control) half_logits else null);
        defer a.free(text);
        const s: c.Shape = .{ .rows = 16, .n = d.experts, .k = d.hidden };
        return .{
            .a = a,
            .rig = .{ .a = a, .device = device, .queue = queue },
            .pipe = try compile(device, text, route.names[0]),
            .d = d,
            .served = .{ s, .{ .rows = h.decode_rows, .n = s.n, .k = s.k }, .{ .rows = h.window_rows, .n = s.n, .k = s.k }, .{ .rows = h.prompt_rows, .n = s.n, .k = s.k } },
            .control = if (control) .sums else null,
        };
    }
    pub fn deinit(t: *Logits) void {
        t.free();
        t.rig.deinit();
        t.pipe.deinit();
    }
    fn free(t: *Logits) void {
        t.a.free(t.x);
        t.a.free(t.w);
        t.a.free(t.dots);
        t.x = &.{};
        t.w = &.{};
        t.dots = &.{};
    }
    pub fn kernel(t: *Logits) g.Kernel {
        t.name = fl.named(base(t.d, &t.base_buf), if (t.control != null) "its sums in half" else null, &t.label);
        return .{ .spec = spec(t.name, &t.served), .ctx = t, .vt = &vt, .control = t.control };
    }
    const vt = g.hooks(Logits);

    pub fn prepare(t: *Logits, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const H: usize = t.d.hidden;
        const E: usize = t.d.experts;
        if (shape.rows == 0 or shape.rows > h.prompt_rows or shape.n != E or shape.k != H) return false;
        t.rig.reset();
        t.free();
        t.rows = shape.rows;
        t.fill = fill;
        const x = try t.a.alloc(u16, shape.rows * H);
        defer t.a.free(x);
        const w = try t.a.alloc(u16, E * H);
        defer t.a.free(w);
        const packed_w = try t.a.alloc(u16, E * H);
        defer t.a.free(packed_w);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        if (fill == .stress) {
            // integers whose dots are exact in fp32 in any order: the second half mirrors the first, all but one nudged
            const half = H / 2;
            for (0..shape.rows) |r| for (0..half) |k| {
                const v: f32 = @floatFromInt(1 + rng.uintLessThan(u32, 63));
                x[r * H + k] = c.toBf16(v);
                x[r * H + half + k] = c.toBf16(v);
            };
            for (0..E) |e| for (0..half) |k| {
                const v: f32 = @floatFromInt(@as(i32, @intCast(1 + rng.uintLessThan(u32, 63))) * (if (rng.boolean()) @as(i32, 1) else -1));
                w[e * H + k] = c.toBf16(v);
                w[e * H + half + k] = c.toBf16(-v + @as(f32, if (k == e % 7) 1 else 0));
            };
        } else {
            for (x) |*v| v.* = c.toBf16((rng.float(f32) * 2 - 1) * 3);
            for (w) |*v| v.* = c.toBf16((rng.float(f32) * 2 - 1) * 0.05);
        }
        if (fill == .nonfinite) {
            // NaN in row 0, +Inf in row 1, -Inf in row 2, both infinities in row 3
            const at = [_][2]usize{ .{ 0, 3 }, .{ 1, 5 }, .{ 2, 9 }, .{ 3, 11 }, .{ 3, 700 } };
            const vals = [_]f32{ fl.planted[0], fl.planted[1], fl.planted[2], fl.planted[1], fl.planted[2] };
            for (at, vals) |p, v| if (p[0] < shape.rows) {
                x[p[0] * H + p[1]] = c.toBf16(v);
            };
        }
        route.pack(w, packed_w, E, H);
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(x));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(packed_w));
        t.out = try t.rig.output(shape.rows * E * 4);
        t.x = try t.a.alloc(f32, x.len);
        for (t.x, x) |*o, v| o.* = c.fromBf16(v);
        t.w = try t.a.alloc(f32, w.len);
        for (t.w, w) |*o, v| o.* = c.fromBf16(v);
        t.dots = try t.a.alloc([2]f64, shape.rows * E);
        for (t.dots, 0..) |*o, i| o.* = t.dot(i);
        return true;
    }

    /// Output i's float64 dot and the magnitude of its terms (each product of two bf16 values is exact).
    fn dot(t: *const Logits, i: usize) [2]f64 {
        const H: usize = t.d.hidden;
        const V = @Vector(8, f64);
        const xr = t.x[(i / t.d.experts) * H ..][0..H];
        const wr = t.w[(i % t.d.experts) * H ..][0..H];
        var s: V = @splat(0);
        var m: V = @splat(0);
        var k: usize = 0;
        while (k < H) : (k += 8) {
            const a: V = @floatCast(@as(@Vector(8, f32), xr[k..][0..8].*));
            const b: V = @floatCast(@as(@Vector(8, f32), wr[k..][0..8].*));
            s += a * b;
            m += @abs(a * b);
        }
        return .{ @reduce(.Add, s), @reduce(.Add, m) };
    }

    pub fn run(t: *Logits) !void {
        const p = t.rig.begin();
        // the loader's encoder: a 4-expert block for four rows a threadgroup of 256
        route.logits(p.e, t.pipe, t.d.shape(), t.ins[0], t.ins[1], t.out, @intCast(t.rows));
        try t.rig.finish(p);
    }

    pub fn count(t: *Logits) usize {
        return t.rows * t.d.experts;
    }

    pub fn reference(t: *Logits, want: []f64, mag: []f64) !void {
        for (want, mag, t.dots) |*w, *m, d| {
            w.* = d[0];
            m.* = if (std.math.isFinite(d[1])) d[1] else 0;
        }
    }

    /// The fp32 sums within gamma(depth) of their terms' size (none on gate 4's exact inputs).
    pub fn intervals(t: *Logits, low: []f64, high: []f64) !void {
        const gam = if (t.fill == .stress) 0 else c.gamma(@floatFromInt(logitsDepth(.{ .rows = t.rows, .n = t.d.experts, .k = t.d.hidden })), fl.u32_ulp);
        for (low, high, t.dots) |*l, *hi, d| {
            const r = Range.point(d[0]).widen(gam * d[1] + fl.tiny);
            l.* = r.lo;
            hi.* = r.hi;
        }
    }

    pub fn bounds(t: *Logits, want: []const f64, _: []const f64, allow: []f64) !void {
        try fl.allowFrom(t.a, t, want, allow);
    }

    pub fn simulate(t: *Logits, acc: c.Accum, order: g.Order, out: []f64) !void {
        const H: usize = t.d.hidden;
        const terms = try t.a.alloc(f64, H);
        defer t.a.free(terms);
        for (out, 0..) |*o, i| {
            const xr = t.x[(i / t.d.experts) * H ..][0..H];
            const wr = t.w[(i % t.d.experts) * H ..][0..H];
            for (terms, xr, wr) |*v, a, b| v.* = @as(f64, a) * b;
            o.* = contract.rows.sumIn(acc, order, terms);
        }
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    logits: fl.List(Logits) = .{},
    selects: fl.List(sel.Select) = .{},
    prompts: fl.List(sel.Prompt) = .{},
    sums: fl.List(sums.Sum) = .{},
    adds: fl.List(sums.Add) = .{},

    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        const d = try Dims.read(s.a);
        var buf: [160]u8 = undefined;
        for ([_]bool{ false, true }) |control| {
            if (fl.wanted(only, Logits.base(d, &buf))) try s.logits.add(s.a, s.kernels, try Logits.init(s.a, s.device, s.queue, d, control));
            if (fl.wanted(only, sums.Sum.base(d, &buf))) try s.sums.add(s.a, s.kernels, try sums.Sum.init(s.a, s.device, s.queue, d, control));
            for ([_]sel.Which{ .rows, .rank }) |which| {
                if (fl.wanted(only, sel.Prompt.base(d, which, &buf))) try s.prompts.add(s.a, s.kernels, try sel.Prompt.init(s.a, s.device, s.queue, d, which, control));
            }
        }
        if (fl.wanted(only, sel.Select.base(d, &buf))) try s.selects.add(s.a, s.kernels, try sel.Select.init(s.a, s.device, s.queue, d));
        if (fl.wanted(only, sums.Add.base(d, &buf))) try s.adds.add(s.a, s.kernels, try sums.Add.init(s.a, s.device, s.queue, d));
    }
    pub fn deinit(s: *Suite) void {
        s.logits.deinit(s.a);
        s.selects.deinit(s.a);
        s.prompts.deinit(s.a);
        s.sums.deinit(s.a);
        s.adds.deinit(s.a);
    }
};

/// Gate 5's launches (hy4_route_records.zig): each pipeline Kernels.loadImpl builds, and each control's own text.
const Source = enum { route, sum, add };
const Launch = struct { name: []const u8, source: Source, edit: ?fl.Edit = null };
pub const launches = [_]Launch{
    .{ .name = "hy4 moe_route tf_route_logits", .source = .route },
    .{ .name = "hy4 moe_route tf_route_select", .source = .route },
    .{ .name = "hy4 route_rows tf_hy4_route_select_rows", .source = .route },
    .{ .name = "hy4 route_rows tf_hy4_route_select_rank", .source = .route },
    .{ .name = "hy4 affine_experts tf_hy4_routed_sum", .source = .sum },
    .{ .name = "hy4 elementwise tf_add_bf16", .source = .add },
    .{ .name = "hy4 moe_route half-sums tf_route_logits", .source = .route, .edit = half_logits },
    .{ .name = "hy4 route_rows nan-picked tf_hy4_route_select_rows", .source = .route, .edit = nan_picked },
    .{ .name = "hy4 route_rows ties-high tf_hy4_route_select_rank", .source = .route, .edit = ties_high },
    .{ .name = "hy4 affine_experts half-sums tf_hy4_routed_sum", .source = .sum, .edit = half_sum },
};

/// Gate 5's texts for this group's launch `name` (null for any other): the library its loader compiles, as here.
pub fn limitsText(a: std.mem.Allocator, _: mtl.Device, name: []const u8) !?fl.Texts {
    for (launches) |l| if (std.mem.eql(u8, l.name, name)) {
        const text = switch (l.source) {
            .route => try routeText(a, try Dims.read(a), l.edit),
            .sum => try fl.edited(a, sources.hy4_flash_affine_experts, l.edit),
            .add => try a.dupe(u8, sources.elementwise),
        };
        return .{ .here = text, .pre_m5 = text };
    };
    return null;
}
