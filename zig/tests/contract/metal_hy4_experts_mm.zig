//! Hy4-flash's routed expert projections under the contract: the portable rows and the tensor units' gathers.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const contract = @import("contract");
const sources = @import("kernel_sources");
const rig_ = @import("rig.zig");
const h = @import("metal_hy4.zig");
const bank = @import("metal_hy4_experts_bank.zig");

const c = contract.spec;
const g = contract.gates;
const fl = h.fl;
const mm = tf.affine_mm;
const portable = h.hy.affine_experts_metal;

/// The routed block's widths and route (h.text's config.json).
pub const Widths = struct {
    width: usize,
    inner: usize,
    experts: usize,
    topk: usize,
    limit: f32,

    pub fn of(t: h.hy.config.Text) Widths {
        return .{ .width = t.routed_expert_hidden_size, .inner = t.moe_intermediate_size, .experts = t.n_routed_experts, .topk = t.num_experts_per_tok, .limit = t.swiglu_limit };
    }
};

/// The portable gate/up and down rows (affine_experts.metal), or the tensor units' GLU and scatter gathers (affine_mm).
pub const Kind = enum { up, down, glu, scatter };

/// A planted fault: an edit of the loader's text, or (glu) MLX's bf16 sigmoid steps in place of the rounded divide.
pub const Fault = struct { label: []const u8, what: []const u8, gate: g.Gate, edit: ?fl.Edit = null, mlx_silu: bool = false };

pub const Variant = struct { kind: Kind, bits: u8, fault: ?Fault = null };

/// metal_tensor_experts.Kernels.load's library: Hy4's two defines on, fp32 out for the down's scatter.
pub fn format(bits: u8, mlx_silu: bool) mm.Format {
    return .{ .bits = bits, .group = bank.group, .out_f32 = true, .glu = if (mlx_silu) .clipped else .clipped_silu, .mapped_experts = true };
}

/// The function a variant launches: the portable pipelines by bits, the 32-row gathers Hy4's ratios always pick.
pub fn function(v: Variant) []const u8 {
    return switch (v.kind) {
        .up => if (v.bits == 4) "tf_affine_experts_up_4" else "tf_affine_experts_up_8",
        .down => if (v.bits == 4) "tf_affine_experts_down_4" else "tf_affine_experts_down_8",
        .glu => "tf_affine_gather_glu_32",
        .scatter => "tf_affine_gather_scatter_32",
    };
}

/// A variant's source as its loader compiles it, the fault applied (owned).
pub fn text(a: std.mem.Allocator, device: mtl.Device, v: Variant) ![]u8 {
    const edit = if (v.fault) |f| f.edit else null;
    if (v.kind == .up or v.kind == .down) return fl.edited(a, sources.hy4_flash_affine_experts, edit);
    const body = try mm.source(device, a, format(v.bits, if (v.fault) |f| f.mlx_silu else false));
    defer a.free(body);
    return fl.edited(a, body, edit);
}

pub fn baseName(v: Variant, buf: []u8) []const u8 {
    const what = switch (v.kind) {
        .up => "(gate/up rows, the checkpoint's SwiGLU at its bf16 points)",
        .down => "(down rows, fp32 out)",
        .glu => "(tensor units after tf_affine_row_sums, mapped experts, rounded SiLU)",
        .scatter => "fp32 out (tensor units after tf_affine_row_sums, mapped experts)",
    };
    const stem = if (v.kind == .up or v.kind == .down) "affine_experts" else "affine_mm";
    const bits = if (v.kind == .glu or v.kind == .scatter) (if (v.bits == 4) " q4" else " q8") else "";
    return std.fmt.bufPrint(buf, "metal hy4 {s} {s}{s} {s}", .{ stem, function(v), bits, what }) catch unreachable;
}

/// A lane's fma chain (two inputs a group) and the simd_shuffle_xor tree.
fn portableDepth(s: c.Shape) usize {
    return s.k / 32 + 5;
}

/// affine_mm's: a group's products in the tile op, its scale and bias, the groups in turn.
fn unitsDepth(s: c.Shape) usize {
    return 64 + s.k / 32 + 8;
}

/// affine_mm's 4-bit codes enter as q + 128 and its bias as b - 128 s, so an infinite input meets opposite signs.
const offset_inf = c.Known{ .gate = .nonfinite, .finding = "an infinite input gives NaN where MLX's s*sum(q x) + b*sum(x) keeps the infinity (the +128 code offset moves the bias to b - 128 s)" };
/// The SwiGLU's clip: Metal's min and max return the other operand for a NaN, so a NaN sum leaves as the limit.
const clip_nan = "a NaN gate or up sum gives a finite output where float64 keeps the NaN: min(gate, 10) and the up's clamp to [-10, 10] return the limit for a NaN, so a row with a NaN input gives -100 = bf16(bf16(silu(10)) * -10) in every output";
const clip_known = c.Known{ .gate = .nonfinite, .finding = clip_nan };
const clip_offset_known = c.Known{ .gate = .nonfinite, .finding = clip_nan ++ "; and an infinite input gives NaN where MLX's s*sum(q x) + b*sum(x) keeps the infinity (the +128 code offset moves the bias to b - 128 s)" };

pub const Proj = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    v: Variant,
    w: Widths,
    served: [4]c.Shape,
    base_buf: [200]u8 = undefined,
    label: [320]u8 = undefined,
    name: []const u8 = "",
    ops: ?portable.Kernels = null,
    pipes: ?mm.Pipes = null,
    /// The bank (gate and up, or down) cached across shapes of the same widths and fill; freed after the gates.
    banks: [2]?bank.Bank = .{ null, null },
    key: [3]usize = .{ 0, 0, 0 },
    route: ?bank.Route = null,
    x: ?bank.Rows = null,
    uses: ?bank.Uses = null,
    fill: g.Fill = .random,
    shape: c.Shape = undefined,
    /// gate/up and GLU: float64's points and their ranges, bf16, made once a prepare (the gates ask for both).
    want16: []u16 = &.{},
    lo16: []u16 = &.{},
    hi16: []u16 = &.{},
    made: bool = false,
    refs: [10]rig_.Ref = undefined,
    out: rig_.Ref = undefined,
    sums: rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, v: Variant, w: Widths) !Proj {
        var p: Proj = .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .v = v, .w = w, .served = undefined };
        const n = if (v.kind == .up or v.kind == .glu) w.inner else w.width;
        const k = if (v.kind == .up or v.kind == .glu) w.width else w.inner;
        const last: usize = if (v.kind == .up or v.kind == .down) h.hy.metal_routed.max_rows else h.prompt_rows;
        const narrow: usize = if (v.kind == .glu) 64 else 128;
        // a narrow base for the edges, then one row, DFly2's window and the largest call, at the routed widths
        p.served = .{ .{ .rows = h.window_rows, .n = narrow, .k = k }, .{ .rows = h.decode_rows, .n = n, .k = k }, .{ .rows = h.window_rows, .n = n, .k = k }, .{ .rows = last, .n = n, .k = k } };
        const body = try text(a, device, v);
        defer a.free(body);
        const lib = try mtl.Library.fromSource(device, body, mtl.CompileOptions.mlx());
        defer lib.deinit();
        if (v.kind == .up or v.kind == .down) {
            // portable.Kernels.load's pipelines from this text, in its order
            const names = [_][]const u8{ "tf_affine_experts_up_4", "tf_affine_experts_down_4", "tf_affine_experts_up_8", "tf_affine_experts_down_8" };
            var ops: portable.Kernels = undefined;
            var made: usize = 0;
            errdefer for (ops.ops[0..made]) |x| x.deinit();
            for (&ops.ops, names) |*pipe, nm| {
                pipe.* = try mtl.Pipeline.init(device, lib, nm, false);
                made += 1;
            }
            p.ops = ops;
        } else {
            var pipes: mm.Pipes = undefined;
            var made: usize = 0;
            errdefer for (pipes[0..made]) |x| x.deinit();
            for (&pipes, mm.names) |*pipe, nm| {
                pipe.* = try mtl.Pipeline.init(device, lib, nm, false);
                made += 1;
            }
            p.pipes = pipes;
        }
        return p;
    }

    pub fn deinit(p: *Proj) void {
        p.finish();
        if (p.ops) |*o| o.deinit();
        if (p.pipes) |pp| for (pp) |x| x.deinit();
        p.rig.deinit();
    }

    fn drop(p: *Proj) void {
        if (p.route) |r| r.deinit();
        if (p.x) |x| x.deinit();
        if (p.uses) |u| u.deinit();
        p.route = null;
        p.x = null;
        p.uses = null;
        inline for (.{ "want16", "lo16", "hi16" }) |f| {
            p.a.free(@field(p, f));
            @field(p, f) = &.{};
        }
        p.made = false;
        p.rig.reset();
    }

    /// After a kernel's gates: its device buffers and its banks.
    fn finish(p: *Proj) void {
        p.drop();
        for (&p.banks) |*b| {
            if (b.*) |x| x.deinit();
            b.* = null;
        }
        p.key = .{ 0, 0, 0 };
    }

    fn release(ctx: *anyopaque) void {
        const p: *Proj = @ptrCast(@alignCast(ctx));
        p.finish();
    }

    const vt_ranged = blk: {
        var v = g.hooks(Proj);
        v.release = release;
        break :blk v;
    };
    const vt_plain = blk: {
        var v = vt_ranged;
        v.bounds = null;
        v.intervals = null;
        break :blk v;
    };

    fn ranged(p: *const Proj) bool {
        return p.v.kind == .up or p.v.kind == .glu;
    }

    fn units(p: *const Proj) bool {
        return p.v.kind == .glu or p.v.kind == .scatter;
    }

    /// What the kernel adds to each code before its sums: affine_mm's 4-bit codes enter its MMAs as q + 128.
    fn off(p: *const Proj) f64 {
        return if (p.units() and p.v.bits == 4) 128 else 0;
    }

    fn form(p: *const Proj) bank.Form {
        return if (p.units()) .factored else .rounded;
    }

    pub fn spec(p: *const Proj) c.Spec {
        const known: ?c.Known = if (p.v.fault != null) null else switch (p.v.kind) {
            .up => clip_known,
            .glu => if (p.v.bits == 4) clip_offset_known else clip_known,
            .scatter => if (p.v.bits == 4) offset_inf else null,
            .down => null,
        };
        const rows: c.Tile = if (p.units()) .{ .size = 32, .step = 1, .min = 1 } else .{};
        const cols: usize = if (p.v.kind == .glu) 32 else 64;
        return .{
            .name = p.name,
            .accum = .f32,
            .depth = if (p.units()) &unitsDepth else &portableDepth,
            .out = if (p.ranged()) .bf16 else .f32,
            .known = known,
            // rows: token rows (16 pairs each); n: outputs a pair; k: the reduction
            .tiles = .{ rows, .{ .size = cols, .step = cols, .min = cols }, .{ .size = bank.group, .step = bank.group, .min = bank.group } },
            .served = &p.served,
        };
    }

    pub fn kernel(p: *Proj) g.Kernel {
        p.name = fl.named(baseName(p.v, &p.base_buf), if (p.v.fault) |f| f.what else null, &p.label);
        return .{ .spec = p.spec(), .ctx = p, .vt = if (p.ranged()) &vt_ranged else &vt_plain, .control = if (p.v.fault) |f| f.gate else null };
    }

    fn pairs(p: *const Proj) usize {
        return p.shape.rows * p.w.topk;
    }

    /// Entry i's expert, input row and output row, in the order the kernel walks them.
    fn useOf(p: *Proj, i: usize) [3]usize {
        const r = p.route.?;
        return switch (p.v.kind) {
            .up => .{ @intCast(r.pick[i]), i / r.topk, i },
            .down => .{ @intCast(r.pick[i]), i, i },
            .glu => .{ r.expertAt(i), @as(usize, @intCast(r.order[i])) / r.topk, i },
            .scatter => .{ r.expertAt(i), i, @intCast(r.order[i]) },
        };
    }

    pub fn prepare(p: *Proj, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        p.drop();
        const n = shape.n;
        const k = shape.k;
        const step: usize = if (p.v.kind == .glu) 32 else 64;
        const most: usize = if (p.units()) h.prompt_rows else h.hy.metal_routed.max_rows;
        if (shape.rows == 0 or shape.rows > most or n == 0 or n % step != 0 or k == 0 or k % bank.group != 0 or k > bank.max_k) return false;
        p.shape = shape;
        p.fill = fill;
        const a = p.a;
        const stress = fill == .stress;
        const key = [3]usize{ n, k, @intFromBool(stress) + 1 };
        if (!std.mem.eql(usize, &key, &p.key)) {
            for (&p.banks) |*b| {
                if (b.*) |x| x.deinit();
                b.* = null;
            }
            p.key = .{ 0, 0, 0 };
            const wseed = std.hash.Wyhash.hash(n * 4099 + k * 7 + key[2], baseName(p.v, &p.base_buf));
            for (0..if (p.ranged()) @as(usize, 2) else 1) |i| p.banks[i] = try bank.Bank.init(a, bank.bank_experts, n, k, p.v.bits, stress, wseed +% @as(u64, i));
            p.key = key;
        }
        p.route = try bank.Route.init(a, shape.rows, p.w.topk, bank.bank_experts, p.w.experts, bank.modeOf(shape.rows), seed);
        const r = p.route.?;
        const xrows = if (p.ranged()) shape.rows else r.pairs();
        p.x = try bank.Rows.init(a, xrows, k, p.v.bits, fill, p.off(), seed ^ 0x9e37);
        p.uses = try bank.Uses.init(a, bank.bank_experts, r.pairs(), p, useOf);
        const x = p.x.?;
        const b0 = &p.banks[0].?;
        p.refs[0] = try p.rig.input(std.mem.sliceAsBytes(x.x));
        switch (p.v.kind) {
            .up, .down => {
                p.refs[1] = try p.rig.input(std.mem.sliceAsBytes(r.pick));
                const rows = try a.alloc(i32, r.pairs());
                defer a.free(rows);
                for (rows, 0..) |*v, i| v.* = @intCast(i / r.topk); // metal_routed's expert_rows
                p.refs[2] = try p.rig.input(std.mem.sliceAsBytes(rows));
                if (p.v.kind == .down) {
                    p.refs[3] = try p.rig.input(std.mem.sliceAsBytes(b0.words));
                    p.refs[4] = try p.rig.input(std.mem.sliceAsBytes(b0.scales));
                    p.refs[5] = try p.rig.input(std.mem.sliceAsBytes(b0.biases));
                } else try p.uploadGateUp();
            },
            .glu, .scatter => {
                for (p.banks[0..if (p.ranged()) @as(usize, 2) else 1], 0..) |b, i| {
                    p.refs[1 + 3 * i] = try p.rig.input(std.mem.sliceAsBytes(b.?.words));
                    p.refs[2 + 3 * i] = try p.rig.input(std.mem.sliceAsBytes(b.?.scales));
                    p.refs[3 + 3 * i] = try p.rig.input(std.mem.sliceAsBytes(b.?.biases));
                }
                p.refs[7] = try p.rig.input(std.mem.sliceAsBytes(r.offsets));
                p.refs[8] = try p.rig.input(std.mem.sliceAsBytes(r.order));
                p.refs[9] = try p.rig.input(std.mem.sliceAsBytes(r.ids));
                p.sums = try p.rig.scratch(xrows * (k / bank.group) * 4);
            },
        }
        p.out = try p.rig.output(r.pairs() * n * @as(usize, if (p.ranged()) 2 else 4));
        return true;
    }

    /// The checkpoint's gate_up layout [experts, 2 inner, width]: each expert's gate rows, then its up rows.
    fn uploadGateUp(p: *Proj) !void {
        const gate = &p.banks[0].?;
        const up = &p.banks[1].?;
        const n = gate.n;
        const each = gate.k * gate.bits / 32;
        const groups = gate.k / bank.group;
        const words = try p.a.alloc(u32, 2 * gate.words.len);
        defer p.a.free(words);
        const scales = try p.a.alloc(u16, 2 * gate.scales.len);
        defer p.a.free(scales);
        const biases = try p.a.alloc(u16, 2 * gate.biases.len);
        defer p.a.free(biases);
        for (0..gate.experts) |e| for ([_]*const bank.Bank{ gate, up }, 0..) |b, half| {
            const at = (2 * e + half) * n;
            @memcpy(words[at * each ..][0 .. n * each], b.words[e * n * each ..][0 .. n * each]);
            @memcpy(scales[at * groups ..][0 .. n * groups], b.scales[e * n * groups ..][0 .. n * groups]);
            @memcpy(biases[at * groups ..][0 .. n * groups], b.biases[e * n * groups ..][0 .. n * groups]);
        };
        p.refs[3] = try p.rig.input(std.mem.sliceAsBytes(words));
        p.refs[4] = try p.rig.input(std.mem.sliceAsBytes(scales));
        p.refs[5] = try p.rig.input(std.mem.sliceAsBytes(biases));
    }

    fn ref(r: rig_.Ref) portable.Ref {
        return .{ .buffer = r.buf, .offset = r.off };
    }

    fn matrix(p: *const Proj, at: usize) struct { w: rig_.Ref, s: rig_.Ref, b: rig_.Ref, n: u32, k: u32 } {
        return .{ .w = p.refs[at], .s = p.refs[at + 1], .b = p.refs[at + 2], .n = @intCast(p.shape.n), .k = @intCast(p.shape.k) };
    }

    pub fn run(p: *Proj) !void {
        const pass = p.rig.begin();
        const e = pass.e;
        const s = p.shape;
        const r = p.route.?;
        const np: u32 = @intCast(r.pairs());
        switch (p.v.kind) {
            .up, .down => {
                const up = p.v.kind == .up;
                // metal_routed's shape: the latent width, the expert width, every expert, this call's pairs and rows
                const shape: portable.Shape = .{ .width = @intCast(if (up) s.k else s.n), .inner = @intCast(if (up) s.n else s.k), .first = 0, .experts = @intCast(p.w.experts), .pairs = np, .rows = @intCast(s.rows), .swiglu_limit = p.w.limit };
                try p.ops.?.run(e, if (p.v.bits == 4) .q4 else .q8, if (up) .up else .down, .{
                    .input = ref(p.refs[0]),
                    .ids = ref(p.refs[1]),
                    .src_rows = if (up) ref(p.refs[2]) else null,
                    .weights = .{ .words = ref(p.refs[3]), .scales = ref(p.refs[4]), .biases = ref(p.refs[5]) },
                    .output = ref(p.out),
                }, shape);
            },
            // metal_tensor_experts.applyJobs's steps for this kind, on host-built jobs
            .glu => {
                const pipes = p.pipes.?;
                mm.rowSums(e, pipes, bank.group, p.refs[0], p.sums, @intCast(s.rows), @intCast(s.k));
                e.barrier();
                mm.gatherGluMapped(e, pipes, p.refs[0], p.sums, p.matrix(1), p.matrix(4), p.refs[7], p.refs[8], p.out, p.refs[9], np, @intCast(r.offsets.len), @intCast(r.topk), p.w.limit);
            },
            .scatter => {
                const pipes = p.pipes.?;
                mm.rowSums(e, pipes, bank.group, p.refs[0], p.sums, np, @intCast(s.k));
                e.barrier();
                mm.gatherToMapped(e, pipes, p.refs[0], p.sums, p.matrix(1), p.refs[7], p.out, p.refs[8], p.refs[9], np, @intCast(r.offsets.len));
            },
        }
        try p.rig.finish(pass);
    }

    pub fn count(p: *Proj) usize {
        return p.pairs() * p.shape.n;
    }

    /// One (output, expert) item of the float64 sweeps: every reader of expert e's output j.
    const Sweep = struct {
        p: *Proj,
        want: []f64 = &.{},
        mag: []f64 = &.{},
        acc: ?c.Accum = null,
        order: g.Order = .sequential,

        pub fn item(s: Sweep, i: usize) void {
            const p = s.p;
            const ne = bank.bank_experts;
            const e = i % ne;
            const j = i / ne;
            const u = p.uses.?;
            if (u.start[e] == u.start[e + 1]) return;
            const n = p.shape.n;
            const x = &p.x.?;
            const row = e * n + j;
            var ws: [2]bank.Weights = undefined;
            const two: usize = if (p.ranged()) 2 else 1;
            for (ws[0..two], p.banks[0..two]) |*w, *b| w.load(&b.*.?, row, p.form());
            const lim = fl.bf(p.w.limit);
            const gam = if (p.fill == .stress) 0 else c.gamma(@floatFromInt(p.spec().depth(p.shape)), fl.u32_ulp);
            for (u.start[e]..u.start[e + 1]) |at| {
                const xr = u.xrow[at];
                const o = @as(usize, u.orow[at]) * n + j;
                var d: [2][2]f64 = undefined;
                for (d[0..two], ws[0..two], p.banks[0..two]) |*dd, *w, *b| dd.* = w.dot(&b.*.?, row, x, xr, p.form(), p.off());
                if (s.acc) |acc| {
                    var sim: [2]f64 = undefined;
                    for (sim[0..two], d[0..two], p.banks[0..two]) |*v, dd, *b| v.* = bank.simDot(acc, s.order, &b.*.?, row, x, xr, p.off(), dd[0]);
                    s.want[o] = if (p.ranged()) bank.swiglu(sim[0], sim[1], lim) else sim[0];
                } else if (p.ranged()) {
                    p.want16[o] = c.toBf16(@floatCast(bank.swiglu(d[0][0], d[1][0], lim)));
                    const r = bank.swigluRange(d[0][0], gam * d[0][1], d[1][0], gam * d[1][1], lim);
                    p.lo16[o] = c.toBf16(@floatCast(r.lo));
                    p.hi16[o] = c.toBf16(@floatCast(r.hi));
                } else {
                    s.want[o] = d[0][0];
                    s.mag[o] = if (std.math.isFinite(d[0][1])) d[0][1] else 0;
                }
            }
        }
    };

    fn items(p: *const Proj) usize {
        return p.shape.n * bank.bank_experts;
    }

    /// gate/up and GLU: float64's bf16 points and their ranges, once a prepare.
    fn make(p: *Proj) !void {
        if (p.made) return;
        const n = p.count();
        p.want16 = try p.a.alloc(u16, n);
        p.lo16 = try p.a.alloc(u16, n);
        p.hi16 = try p.a.alloc(u16, n);
        bank.sweep(Sweep{ .p = p }, p.items());
        p.made = true;
    }

    pub fn reference(p: *Proj, want: []f64, mag: []f64) !void {
        if (!p.ranged()) return bank.sweep(Sweep{ .p = p, .want = want, .mag = mag }, p.items());
        try p.make();
        for (want, mag, p.want16) |*w, *m, v| {
            w.* = c.fromBf16(v);
            m.* = 0;
        }
    }

    /// Each output within its range through the fp32 sums' spread, the bf16 points, exp's ulps and the clips.
    pub fn intervals(p: *Proj, low: []f64, high: []f64) !void {
        try p.make();
        for (low, high, p.lo16, p.hi16) |*l, *hh, a, b| {
            l.* = c.fromBf16(a);
            hh.* = c.fromBf16(b);
        }
    }

    pub fn bounds(p: *Proj, want: []const f64, _: []const f64, allow: []f64) !void {
        try fl.allowFrom(p.a, p, want, allow);
    }

    pub fn simulate(p: *Proj, acc: c.Accum, order: g.Order, out: []f64) !void {
        bank.sweep(Sweep{ .p = p, .want = out, .acc = acc, .order = order }, p.items());
    }
};
