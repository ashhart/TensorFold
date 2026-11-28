//! Mini's packed NVFP4/MXFP8 expert pipeline on BF16 activations, with bit-exact and float64 references.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const ink = @import("inkling2");
const rig_ = @import("contract_rig.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const mo = ink.modelopt;
const experts = 8;
const slots = 8;

const CombineArgs = extern struct {
    experts: u64,
    pick: u64,
    weights: u64,
    out: u64,
    rows: i32,
    h: i32,
    slots: i32,
};

const TensorActArgs = extern struct {
    gateup: u64,
    out: u64,
    n: i32,
};

const Baseline = struct {
    mod: cuda.Module,
    tensor_mod: cuda.Module,
    pack: cuda.Function,
    project: cuda.Function,
    act: cuda.Function,
    combine: cuda.Function,
};

fn bf(x: f32) f32 {
    return c.fromBf16(c.toBf16(x));
}

fn e2m1(n: u8) f32 {
    const table = [8]f32{ 0, 0.5, 1, 1.5, 2, 3, 4, 6 };
    const v = table[n & 7];
    return if ((n & 8) != 0) -v else v;
}

fn e4m3(n: u8) f32 {
    const e = (n >> 3) & 15;
    const m: f32 = @floatFromInt(n & 7);
    var v: f32 = undefined;
    if (e == 0) {
        v = m * 0x1p-9;
    } else if (e == 15 and n & 7 == 7) {
        v = std.math.nan(f32);
    } else {
        v = std.math.pow(f32, 2, @as(f32, @floatFromInt(e)) - 7) * (1 + m / 8);
    }
    return if ((n & 128) != 0) -v else v;
}

fn route(slot: usize) u32 {
    const ids = [8]u32{ 7, 7, 2, 5, 0, 6, 3, 4 };
    return ids[slot];
}

fn routeWeight(token: usize, slot: usize) f32 {
    _ = token;
    // Route order loses the unit between opposite large terms; expert order keeps it.
    return switch (slot) {
        0 => 0x1p24,
        1 => 1,
        2 => -0x1p24,
        else => 0,
    };
}

const test_amax: f32 = 1.0039; // bf16: 1
const test_den: f32 = 2696.001; // bf16: 2704

fn nvfp4Weight(code: u8, scale: u8) u16 {
    const factor = bf(test_amax) / bf(test_den);
    return c.toBf16(e2m1(code) * e4m3(scale) * factor);
}

fn ue8m0(n: u8) f32 {
    return std.math.pow(f32, 2, @as(f32, @floatFromInt(@as(i32, n) - 127)));
}

fn mxfp8Weight(code: u8, scale: u8) u16 {
    return c.toBf16(e4m3(code) * ue8m0(scale));
}

const Phase = enum { decode, prompt };
const Fault = enum { none, wrong_combine, wrong_plan, wrong_up, wrong_prompt_up, wrong_act };
fn rawSpec(format: mo.Format, phase: Phase, fault: Fault, tail: bool) c.Spec {
    const served = &[_]c.Shape{
        .{ .rows = 1, .n = 2304, .k = 1152 },
        .{ .rows = 16, .n = 2304, .k = 1152 },
        .{ .rows = 31, .n = 2304, .k = 1152 },
        .{ .rows = 32, .n = 2304, .k = 1152 },
        .{ .rows = 2048, .n = 2304, .k = 1152 },
    };
    const name = if (tail) (if (format == .nvfp4) "inkling2 Mini NVFP4 decode experts at K = 32, scales ending at their allocation" else "inkling2 Mini MXFP8 decode experts at K = 32, scales ending at their allocation") else switch (fault) {
        .none => if (format == .nvfp4)
            (if (phase == .decode) "inkling2 Mini NVFP4 decode w13/w2 tensor-core experts" else "inkling2 Mini NVFP4 prompt w13/w2 tensor-core experts")
        else
            (if (phase == .decode) "inkling2 Mini MXFP8 decode w13/w2 tensor-core experts" else "inkling2 Mini MXFP8 prompt w13/w2 tensor-core experts"),
        .wrong_combine => if (format == .nvfp4) "control: inkling2 Mini NVFP4 int32 route IDs reinterpreted as FP32 weights" else "control: inkling2 Mini MXFP8 int32 route IDs reinterpreted as FP32 weights",
        .wrong_plan => "control: inkling2 Mini MXFP8 plan routes to the next expert IDs",
        .wrong_up => "control: inkling2 Mini MXFP8 decode up projection clears the planted up-row code",
        .wrong_prompt_up => if (format == .nvfp4) "control: inkling2 Mini NVFP4 prompt up projection clears the planted up-row code" else "control: inkling2 Mini MXFP8 prompt up projection clears the planted up-row code",
        .wrong_act => "control: inkling2 Mini MXFP8 activation uses gate cap eight",
    };
    const shapes = if (tail) &[_]c.Shape{ .{ .rows = 1, .n = 32, .k = 32 }, .{ .rows = 16, .n = 32, .k = 32 } } else if (fault == .wrong_prompt_up or (fault == .none and phase == .prompt))
        served[3..]
    else if (fault == .none)
        served[0..3]
    else
        served[0..2];
    return .{
        .name = name,
        .accum = .f32,
        .depth = depth,
        .ops = 8,
        .mids = 4,
        .out = .bf16,
        .replaces = "host-dequantized BF16 tensor projections, softcapped SwiGLU, ascending FP32 routing",
        .nan = .{ .stated = "a non-finite BF16 input row propagates through its selected experts" },
        .tiles = .{ .{ .size = 16 }, .{ .size = 32 }, .{ .size = 32 } },
        .served = shapes,
        .sums = if (tail) .{ .f16 = 0, .bf16 = 0, .why = "K = 32 sums are too short to tell precisions apart; the Mini specs prove them" } else .{ .f16 = 30, .bf16 = 30 },
    };
}
fn depth(s: c.Shape) usize {
    return s.n + s.k + 2 * core.attention_range.mma_step + slots;
}
pub const ModelOpt = struct {
    rig: rig_.Rig,
    k: *const mo.Kernels,
    format: mo.Format,
    phase: Phase,
    shape: c.Shape = undefined,
    fault: Fault = .none,
    tail: bool = false, // K = 32: each projection's scales in an allocation of their size, memcheck sees a read past
    tight: [2]?cuda.DeviceBuffer = .{ null, null },
    arena: ?std.heap.ArenaAllocator = null,
    x: []const u16 = &.{},
    w13: []const u16 = &.{},
    w2: []const u16 = &.{},
    l: mo.Layer = undefined,
    dev_x: u64 = 0,
    pick: u64 = 0,
    weights: u64 = 0,
    out: u64 = 0,
    stress: bool = false,
    scratch: u64 = 0,
    base_pick: u64 = 0,
    base_w13: u64 = 0,
    base_w2: u64 = 0,
    base_pack13: u64 = 0,
    base_pack2: u64 = 0,
    base_gateup: u64 = 0,
    base_act: u64 = 0,
    base_down: u64 = 0,
    base_out: u64 = 0,
    baseline: ?Baseline = null,

    pub fn kernel(self: *ModelOpt) g.Kernel {
        return .{ .spec = rawSpec(self.format, self.phase, self.fault, self.tail), .ctx = self, .vt = &vt, .control = if (self.fault == .none) null else .shapes };
    }

    const vt = g.hooks(ModelOpt);

    pub fn deinit(self: *ModelOpt) void {
        self.drop();
        if (self.baseline) |*b| {
            b.tensor_mod.unload();
            b.mod.unload();
        }
        self.baseline = null;
        self.rig.deinit();
    }

    fn drop(self: *ModelOpt) void {
        if (self.arena) |*a| a.deinit();
        self.arena = null;
        for (&self.tight) |*t| if (t.*) |*b| b.free();
        self.tight = .{ null, null };
        self.rig.reset();
    }

    fn projectionRef(a: std.mem.Allocator, format: mo.Format, nexperts: usize, n: usize, k: usize, gate_up: bool, fill: g.Fill) !struct { codes: []u8, scales: []u8, amax: []f32, den: [1]f32, ref: []u16 } {
        const values = nexperts * n * k;
        const raw = try a.alloc(u8, if (format == .nvfp4) values / 2 else values);
        const ref = try a.alloc(u16, values);
        const group: usize = if (format == .nvfp4) 16 else 32;
        const scales = try a.alloc(u8, values / group);
        const amax = try a.alloc(f32, if (format == .nvfp4) nexperts else 0);
        if (format == .nvfp4) @memset(amax, if (fill == .stress) bf(test_den) else test_amax);
        const codes = [16]u8{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
        const fp8 = [16]u8{ 0x00, 0x08, 0x10, 0x18, 0x20, 0x28, 0x30, 0x38, 0x80, 0x88, 0x90, 0x98, 0xa0, 0xa8, 0xb0, 0xb8 };
        const nv_scales = [4]u8{ 0x38, 0x30, 0x40, 0x34 };
        // General controls stay finite; the dedicated scale-255 test covers +inf without masking faults as shared NaNs.
        const mx_scales = [4]u8{ 127, 128, 126, 129 };
        if (fill == .stress) {
            for (0..nexperts) |e| for (0..n) |row| {
                const base = (e * n + row) * k;
                const scale_base = (e * n + row) * (k / group);
                @memset(scales[scale_base..][0 .. k / group], if (format == .nvfp4) 0x38 else 127);
                for (0..k) |col| {
                    // Mirrored +/-1 terms cancel to small integers; eight up rows retain a down-projection nudge.
                    const zero = gate_up and (row & 1) == 1 and row / 2 < 8 and col == k / 2 + 8;
                    const code: u8 = if (zero) 0 else if (format == .nvfp4) (if (col < k / 2) 2 else 10) else (if (col < k / 2) 0x38 else 0xb8);
                    if (format == .nvfp4) {
                        if ((col & 1) == 0) raw[base / 2 + col / 2] = code else raw[base / 2 + col / 2] |= code << 4;
                    } else raw[base + col] = code;
                    ref[base + col] = c.toBf16(if (zero) 0 else if (col < k / 2) 1 else -1);
                }
            };
            return .{ .codes = raw, .scales = scales, .amax = amax, .den = .{test_den}, .ref = ref };
        }
        for (0..nexperts) |e| for (0..n) |row| {
            const base = (e * n + row) * k;
            const scale_base = (e * n + row) * (k / group);
            if (format == .nvfp4) {
                for (0..k / 2) |cb| {
                    var n0 = codes[(2 * (base / 2 + cb) * 5 + 3) & 15];
                    const n1 = if (cb == 0) 0 else codes[(2 * (base / 2 + cb) * 5 + 7) & 15];
                    if (cb == 0) n0 = if (!gate_up or row < 2) (if (gate_up) 2 else 7) else 0;
                    raw[base / 2 + cb] = n0 | (n1 << 4);
                    const gi = cb / 8;
                    scales[scale_base + gi] = if (gi == 0) 0x38 else nv_scales[(scale_base + gi) & 3];
                    const scale = scales[scale_base + gi];
                    ref[base + 2 * cb] = nvfp4Weight(n0, scale);
                    ref[base + 2 * cb + 1] = nvfp4Weight(n1, scale);
                }
            } else {
                for (0..k) |col| {
                    var code = fp8[((base + col) * 5 + 3) & 15];
                    // Gate six makes caps four and eight differ deterministically on the one-token control.
                    if (col == 0) code = if (!gate_up) 0x40 else if (row == 0) 0x4c else if (row == 1) 0x38 else 0;
                    raw[base + col] = code;
                    const gi = col / 32;
                    scales[scale_base + gi] = if (gi == 0) 127 else mx_scales[(scale_base + gi) % mx_scales.len];
                    ref[base + col] = mxfp8Weight(code, scales[scale_base + gi]);
                }
            }
        };
        return .{ .codes = raw, .scales = scales, .amax = amax, .den = .{test_den}, .ref = ref };
    }

    pub fn prepare(self: *ModelOpt, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        _ = seed;
        if (self.phase == .prompt and shape.rows < 32 and fill != .stress and fill != .nonfinite) return false;
        if (shape.n != @as(usize, if (self.tail) 32 else 2304) or shape.k != @as(usize, if (self.tail) 32 else 1152) or shape.rows == 0 or shape.rows > 2048) return false;
        self.drop();
        self.shape = shape;
        self.stress = fill == .stress;
        self.arena = std.heap.ArenaAllocator.init(self.rig.gpu.gpa);
        const a = self.arena.?.allocator();
        const h = shape.n;
        const m = shape.k;
        const pairs = shape.rows * slots;
        const x = try a.alloc(u16, shape.rows * h);
        for (x, 0..) |*v, i| {
            const q: f32 = @as(f32, @floatFromInt(@as(i32, @intCast(i % 17)) - 8)) / 64;
            v.* = c.toBf16(q);
        }
        @memset(x[0..h], 0);
        x[0] = c.toBf16(1); // One gate/up activation and one down product make token zero a decode probe.
        if (fill == .nonfinite) x[0] = 0x7fc0;
        if (fill == .stress) for (0..shape.rows) |row| for (0..h) |col| {
            const v: f32 = 4 * @as(f32, @floatFromInt(1 + col % 64));
            x[row * h + col] = c.toBf16(v + @as(f32, if (col >= h / 2 and col < h / 2 + 8) 4 else 0));
        };
        self.x = x;
        const p13 = try projectionRef(a, self.format, experts, 2 * m, h, true, fill);
        const p2 = try projectionRef(a, self.format, experts, h, m, false, fill);
        self.w13 = p13.ref;
        self.w2 = p2.ref;
        self.dev_x = try self.rig.input(std.mem.sliceAsBytes(x));
        const dev13 = if (self.fault == .wrong_up or self.fault == .wrong_prompt_up) blk: {
            const wrong = try a.dupe(u8, p13.codes);
            for (0..experts) |e| {
                const code_stride: usize = if (self.format == .nvfp4) 2 else 1;
                const z = (e * 2 * m + 1) * h / code_stride;
                if (self.format == .nvfp4) wrong[z] &= 0xf0 else wrong[z] = 0;
            }
            break :blk try self.rig.input(wrong);
        } else try self.rig.input(p13.codes);
        const dev13s = try self.scaleInput(0, p13.scales);
        const dev2 = try self.rig.input(p2.codes);
        const dev2s = try self.scaleInput(1, p2.scales);
        var amax13: u64 = 0;
        var amax2: u64 = 0;
        var den13: u64 = 0;
        var den2: u64 = 0;
        if (self.format == .nvfp4) {
            amax13 = try self.rig.input(std.mem.sliceAsBytes(p13.amax));
            amax2 = try self.rig.input(std.mem.sliceAsBytes(p2.amax));
            den13 = try self.rig.input(std.mem.asBytes(&p13.den));
            den2 = try self.rig.input(std.mem.asBytes(&p2.den));
        }
        self.l = .{
            .format = self.format,
            .experts = experts,
            .w13 = .{ .codes = dev13, .scales = dev13s, .amax = amax13, .denominator = den13, .n = @intCast(2 * m), .k = @intCast(h) },
            .w2 = .{ .codes = dev2, .scales = dev2s, .amax = amax2, .denominator = den2, .n = @intCast(h), .k = @intCast(m) },
        };
        const picks = try a.alloc(i32, pairs);
        const weights = try a.alloc(f32, pairs);
        for (picks, weights, 0..) |*p, *w, i| {
            const slot = i % slots;
            const id = if (self.fault == .wrong_plan) (route(slot) + 1) % experts else route(slot);
            p.* = @intCast(id);
            w.* = routeWeight(i / slots, slot);
        }
        const base_picks = try a.alloc(i32, pairs);
        for (base_picks, 0..) |*p, i| p.* = @intCast(route(i % slots));
        self.pick = try self.rig.input(std.mem.sliceAsBytes(picks));
        self.weights = try self.rig.input(std.mem.sliceAsBytes(weights));
        self.out = try self.rig.output(shape.rows * h * 2);
        self.scratch = try self.rig.scratch(try mo.scratchBytes(shape.rows, slots, h, m));
        self.base_pick = try self.rig.input(std.mem.sliceAsBytes(base_picks));
        self.base_w13 = try self.rig.input(std.mem.sliceAsBytes(p13.ref));
        self.base_w2 = try self.rig.input(std.mem.sliceAsBytes(p2.ref));
        self.base_pack13 = try self.rig.scratch(experts * 2 * m * h * 2);
        self.base_pack2 = try self.rig.scratch(experts * h * m * 2);
        self.base_gateup = try self.rig.scratch(pairs * 2 * m * 2);
        self.base_act = try self.rig.scratch(pairs * m * 2);
        self.base_down = try self.rig.scratch(pairs * h * 2);
        self.base_out = try self.rig.scratch(shape.rows * h * 2);
        return true;
    }

    fn scaleInput(self: *ModelOpt, i: usize, bytes: []const u8) !u64 {
        if (!self.tail) return self.rig.input(bytes);
        self.tight[i] = try cuda.DeviceBuffer.fromHost(self.rig.gpu.d, bytes);
        return self.tight[i].?.ptr;
    }

    pub fn run(self: *ModelOpt) !void {
        const route_weights = if (self.fault == .wrong_combine) self.pick else self.weights;
        const caps = if (self.fault == .wrong_act) ink.experts.Caps{ .gate = 8, .up = 25 } else ink.experts.Caps{ .gate = 4, .up = 25 };
        if (self.phase == .prompt and self.shape.rows < 32) return self.runPrompt(caps);
        try mo.run(self.k, self.rig.s, self.l, caps, self.dev_x, self.shape.n, self.pick, route_weights, self.shape.rows, slots, self.out, self.scratch);
    }

    fn pack(self: *ModelOpt, b: *const Baseline, src: u64, n: usize, k: usize, dst: u64) !void {
        var a: cuda.Args = .{};
        a.add(src);
        a.add(@as(i64, @intCast(n * k)));
        a.add(@as(i32, @intCast(k)));
        for ([_]i32{ 0, 0, 0 }) |v| a.add(v);
        a.add(@as(i32, @intCast(n)));
        a.add(@as(i32, @intCast(k)));
        a.add(dst);
        const q = (n / 32) * (k / 32) * 128;
        try cuda.launch.launch(b.pack, .{ .grid = .{ .x = @intCast((q + 255) / 256), .y = experts, .z = 1 }, .block = .{ .x = 256 } }, self.rig.s, &a);
    }

    fn tensorProjection(self: *ModelOpt, b: *const Baseline, x: u64, x_stride: usize, slot_count: usize, w: u64, n: usize, k: usize, plan: mo.Layout, out: u64, item_bound: usize) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(@as(i32, @intCast(x_stride)));
        a.add(@as(i32, @intCast(slot_count)));
        a.add(w);
        a.add(@as(i32, @intCast(k / 32)));
        a.add(@as(i32, @intCast(n / 32)));
        for ([_]u64{ plan.items, plan.item_count, plan.members, out }) |v| a.add(v);
        a.add(@as(i32, @intCast(n)));
        a.add(@as(i32, -1));
        const nb = n / 32;
        const grid = (item_bound * nb + 3) / 4;
        try cuda.launch.launch(b.project, .{ .grid = .{ .x = @intCast(grid), .y = 1, .z = 1 }, .block = .{ .x = 128 } }, self.rig.s, &a);
    }

    /// Tensor projections over host-dequantized weights, the baseline epilogue and the original ascending combine.
    pub fn replaced(self: *ModelOpt) ![]const u8 {
        if (self.baseline == null) {
            var mod = try cuda.Module.load(self.rig.gpu.d, cuda.kernels.inkling2_experts);
            errdefer mod.unload();
            var tensor_mod = try cuda.Module.load(self.rig.gpu.d, cuda.kernels.experts_bf16);
            errdefer tensor_mod.unload();
            self.baseline = .{
                .mod = mod,
                .tensor_mod = tensor_mod,
                .pack = try tensor_mod.function("tf_bf16_experts_pack"),
                .project = try tensor_mod.function("tf_bf16_experts_down_bf16"),
                .act = try mod.function("tf_ink2moe_tensor_act"),
                .combine = try mod.function("tf_ink2moe_combine"),
            };
        }
        const b = &self.baseline.?;
        const pairs = self.shape.rows * slots;
        const h = self.shape.n;
        const m = self.shape.k;
        const plan = mo.layout(self.scratch, self.shape.rows, slots, h, m);
        const tile: usize = if (self.shape.rows < 32) 16 else 64;
        const item_bound = (pairs + tile - 1) / tile + @min(pairs, experts);
        var pa: cuda.Args = .{};
        for ([_]u64{ self.base_pick, plan.items, plan.members, plan.item_count }) |v| pa.add(v);
        for ([_]i32{ @intCast(self.shape.rows), slots, experts, @intCast(tile) }) |v| pa.add(v);
        try cuda.launch.launch(self.k.plan, .{ .grid = .{ .x = 1, .y = 1, .z = 1 }, .block = .{ .x = 32 } }, self.rig.s, &pa);
        try self.pack(b, self.base_w13, 2 * m, h, self.base_pack13);
        try self.pack(b, self.base_w2, h, m, self.base_pack2);
        try self.tensorProjection(b, self.dev_x, h, slots, self.base_pack13, 2 * m, h, plan, self.base_gateup, item_bound);
        var aa: cuda.Args = .{};
        aa.add(TensorActArgs{ .gateup = self.base_gateup, .out = self.base_act, .n = @intCast(pairs * m) });
        try cuda.launch.launch(b.act, .{ .grid = .{ .x = @intCast((pairs * m + 255) / 256), .y = 1, .z = 1 }, .block = .{ .x = 256 } }, self.rig.s, &aa);
        try self.tensorProjection(b, self.base_act, m, 0, self.base_pack2, h, m, plan, self.base_down, item_bound);
        var ca: cuda.Args = .{};
        ca.add(CombineArgs{ .experts = self.base_down, .pick = self.base_pick, .weights = self.weights, .out = self.base_out, .rows = @intCast(self.shape.rows), .h = @intCast(h), .slots = slots });
        const outputs = self.shape.rows * h;
        try cuda.launch.launch(b.combine, .{ .grid = .{ .x = @intCast((outputs + 255) / 256), .y = 1, .z = 1 }, .block = .{ .x = 256 } }, self.rig.s, &ca);
        return self.rig.bytes(self.base_out, 2 * self.count());
    }

    fn promptProjection(self: *ModelOpt, x: u64, stride: usize, pair_slots: usize, p: mo.Projection, row0: i32, step: i32, plan: mo.Layout, out: u64, bound: usize) !void {
        const outputs: usize = @intCast(if (step == 2) p.n / 2 else p.n);
        var a: cuda.Args = .{};
        a.add(x);
        a.add(@as(i32, @intCast(stride)));
        a.add(@as(i32, @intCast(pair_slots)));
        a.add(mo.weight(p, self.l, row0, step));
        a.add(@as(i32, @intCast(p.k / 32)));
        a.add(@as(i32, @intCast(outputs / 32)));
        for ([_]u64{ plan.items, plan.item_count, plan.members, out }) |v| a.add(v);
        a.add(@as(i32, @intCast(outputs)));
        a.add(@as(i32, -1));
        const grid = (bound * outputs / 32 + 3) / 4;
        try cuda.launch.launch(self.k.prompt, .{ .grid = .{ .x = @intCast(grid), .y = 1, .z = 1 }, .block = .{ .x = 128 } }, self.rig.s, &a);
    }

    fn runPrompt(self: *ModelOpt, caps: ink.experts.Caps) !void {
        const pairs = self.shape.rows * slots;
        const b = mo.layout(self.scratch, self.shape.rows, slots, self.shape.n, self.shape.k);
        const bound = (pairs + 63) / 64 + @min(pairs, experts);
        var plan_args: cuda.Args = .{};
        for ([_]u64{ self.pick, b.items, b.members, b.item_count }) |v| plan_args.add(v);
        for ([_]i32{ @intCast(self.shape.rows), slots, experts, 64 }) |v| plan_args.add(v);
        try cuda.launch.launch(self.k.plan, .{ .grid = .{ .x = 1, .y = 1, .z = 1 }, .block = .{ .x = 32 } }, self.rig.s, &plan_args);
        try self.promptProjection(self.dev_x, self.shape.n, slots, self.l.w13, 0, 2, b, b.gate, bound);
        try self.promptProjection(self.dev_x, self.shape.n, slots, self.l.w13, 1, 2, b, b.up, bound);
        var aa: cuda.Args = .{};
        for ([_]u64{ b.gate, b.up, b.act }) |v| aa.add(v);
        aa.add(@as(i32, @intCast(pairs * self.shape.k)));
        aa.add(caps.gate);
        aa.add(caps.up);
        try cuda.launch.launch(self.k.act, .{ .grid = .{ .x = @intCast((pairs * self.shape.k + 255) / 256), .y = 1, .z = 1 }, .block = .{ .x = 256 } }, self.rig.s, &aa);
        try self.promptProjection(b.act, self.shape.k, 0, self.l.w2, 0, 1, b, b.down, bound);
        var ca: cuda.Args = .{};
        for ([_]u64{ self.pick, self.weights, b.down, self.out }) |v| ca.add(v);
        for ([_]i32{ @intCast(self.shape.rows), @intCast(self.shape.n), slots, experts }) |v| ca.add(v);
        const output_count = self.shape.rows * self.shape.n;
        try cuda.launch.launch(self.k.combine, .{ .grid = .{ .x = @intCast((output_count + 255) / 256), .y = 1, .z = 1 }, .block = .{ .x = 256 } }, self.rig.s, &ca);
    }

    pub fn count(self: *const ModelOpt) usize {
        return self.shape.rows * self.shape.n;
    }

    fn roundTo(acc: c.Accum, v: f64) f64 {
        return switch (acc) {
            .f32 => @as(f32, @floatCast(v)),
            .f16 => @as(f16, @floatCast(v)),
            .bf16 => c.fromBf16(c.toBf16(@floatCast(v))),
            .none => v,
        };
    }
    fn rounded(v: f64) f64 {
        return c.fromBf16(c.toBf16(@floatCast(v)));
    }
    fn activate(gate: f64, up: f64) f64 {
        const gate_f32: f32 = @floatCast(gate);
        const up_f32: f32 = @floatCast(up);
        const gc = 4 * @as(f32, @floatCast(std.math.tanh(gate_f32 / 4)));
        const uc = 25 * @as(f32, @floatCast(std.math.tanh(up_f32 / 25)));
        const sigmoid: f32 = 1 / (1 + @exp(-gate_f32));
        return rounded(gc * sigmoid * uc);
    }

    fn evaluate(self: *const ModelOpt, acc: c.Accum, order: g.Order, out: []f64, mag: ?[]f64) !void {
        const a = self.rig.gpu.gpa;
        const pairs = self.shape.rows * slots;
        const h = self.shape.n;
        const m = self.shape.k;
        const t = try a.alloc(f64, h);
        defer a.free(t);
        const act = try a.alloc(f64, pairs * m);
        defer a.free(act);
        const upstream = try a.alloc(f64, pairs * m);
        defer a.free(upstream);
        const down = try a.alloc(f64, pairs * h);
        defer a.free(down);
        const down_mag = try a.alloc(f64, pairs * h);
        defer a.free(down_mag);
        for (0..if (self.stress) 1 else pairs) |p| {
            const token = p / slots;
            const e = route(p % slots);
            const x = self.x[token * h ..][0..h];
            const w = self.w13[e * 2 * m * h ..];
            for (0..m) |j| {
                var gabs: f64 = 0;
                var uabs: f64 = 0;
                for (0..h) |i| {
                    const xv = c.fromBf16(x[i]);
                    t[i] = xv * c.fromBf16(w[(2 * j) * h + i]);
                    gabs += @abs(t[i]);
                }
                const gate = rounded(core.contract.rows.sumIn(acc, order, t));
                for (0..h) |i| {
                    t[i] = c.fromBf16(x[i]) * c.fromBf16(w[(2 * j + 1) * h + i]);
                    uabs += @abs(t[i]);
                }
                const up = rounded(core.contract.rows.sumIn(acc, order, t));
                act[p * m + j] = activate(gate, up);
                upstream[p * m + j] = 32 * gabs + 8 * uabs;
            }
        }
        if (self.stress) for (1..pairs) |p| {
            @memcpy(act[p * m ..][0..m], act[0..m]);
            @memcpy(upstream[p * m ..][0..m], upstream[0..m]);
        };
        for (0..if (self.stress) 1 else pairs) |p| for (0..if (self.stress) 1 else h) |col| {
            const w = self.w2[route(p % slots) * h * m + col * m ..][0..m];
            var abs: f64 = 0;
            for (0..m) |j| {
                t[j] = act[p * m + j] * c.fromBf16(w[j]);
                abs += @abs(t[j]) + @abs(c.fromBf16(w[j])) * upstream[p * m + j];
            }
            down[p * h + col] = rounded(core.contract.rows.sumIn(acc, order, t[0..m]));
            down_mag[p * h + col] = abs;
        };
        if (self.stress) {
            @memset(down, down[0]);
            @memset(down_mag, down_mag[0]);
        }
        for (0..self.shape.rows) |token| for (0..h) |col| {
            var total: f64 = 0;
            var abs: f64 = 0;
            for (0..experts) |e| for (0..slots) |slot| {
                if (route(slot) != e) continue;
                const p = token * slots + slot;
                const weight = routeWeight(token, slot);
                total = roundTo(acc, total + roundTo(acc, weight * down[p * h + col]));
                abs += @abs(weight) * down_mag[p * h + col];
            };
            out[token * h + col] = rounded(total);
            if (mag) |v| v[token * h + col] = abs;
        };
    }

    pub fn reference(self: *ModelOpt, want: []f64, mag: []f64) !void {
        if (self.x[0] == 0x7fc0) {
            @memset(want, 0);
            @memset(mag, 0);
            @memset(want[0..self.shape.n], std.math.nan(f64));
            return;
        }
        return self.evaluate(.none, .sequential, want, mag);
    }

    pub fn simulate(self: *ModelOpt, acc: c.Accum, order: g.Order, out: []f64) !void {
        return self.evaluate(acc, order, out, null);
    }
};

test "A2: amax and denominator are BF16 before the FP32 quotient" {
    // code=6, scale=1: rounded sources give 0x3b11; f32 sources give 0x3b12.
    try std.testing.expectEqual(@as(u16, 0x3b11), nvfp4Weight(7, 0x38));
    const factor = bf(test_amax) / bf(test_den);
    try std.testing.expectEqual(@as(f32, 1) / 2704, factor);
    const unrounded = c.toBf16(e2m1(7) * e4m3(0x38) * (test_amax / test_den));
    try std.testing.expectEqual(@as(u16, 0x3b12), unrounded);
}

test "MXFP8 UE8M0 scale 255 decodes to positive infinity" {
    try std.testing.expectEqual(std.math.inf(f32), ue8m0(255));
    try std.testing.expectEqual(@as(u16, 0x7f80), mxfp8Weight(0x38, 255));
}

test "A12: ascending experts retain the unit cancelled in route order" {
    var route_sum: f32 = 0;
    for (0..slots) |slot| route_sum += routeWeight(0, slot);
    try std.testing.expectEqual(@as(f32, 0), route_sum);
    var expert_sum: f32 = 0;
    for (0..experts) |e| for (0..slots) |slot| {
        if (route(slot) == e) expert_sum += routeWeight(0, slot);
    };
    try std.testing.expectEqual(@as(f32, 1), expert_sum);
}

test "the independent baseline's CUDA structs match experts.cu" {
    try std.testing.expectEqual(@as(usize, 48), @sizeOf(CombineArgs));
    try std.testing.expectEqual(@as(usize, 24), @sizeOf(TensorActArgs));
    try std.testing.expectEqual(@as(usize, 16), @offsetOf(TensorActArgs, "n"));
}
