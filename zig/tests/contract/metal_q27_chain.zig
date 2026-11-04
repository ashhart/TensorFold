//! Long canonical prompt chains replace main's one-row tail on identical heads, cache rows and gates.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const common = @import("metal_q27_attn.zig");
const oracle = @import("q27_chain_ref.zig");
const c = contract.spec;
const g = contract.gates;
const q = tf.qwen27;
const attn = q.attention;
const planning = q.attention_plan;
const D = 256;
pub const Reg = struct { heads: common.Heads, gated: bool = false, fp64: bool = false, control: bool = false };

fn depth(shape: c.Shape) usize {
    return shape.k + shape.rows + D;
}
fn shapes(comptime h: common.Heads, comptime fp64: bool) []const c.Shape {
    const n = comptime common.geometry(h, false).heads;
    if (fp64) return &.{ .{ .rows = 1, .n = n, .k = 127 }, .{ .rows = 129, .n = n, .k = 0 }, .{ .rows = 3, .n = n, .k = 511 }, .{ .rows = 3, .n = n, .k = 8193 } };
    return &.{ .{ .rows = 1, .n = n, .k = 0 }, .{ .rows = 16, .n = n, .k = 63 }, .{ .rows = 127, .n = n, .k = 64 }, .{ .rows = 128, .n = n, .k = 127 }, .{ .rows = 129, .n = n, .k = 128 }, .{ .rows = 255, .n = n, .k = 511 }, .{ .rows = 256, .n = n, .k = 512 }, .{ .rows = 511, .n = n, .k = 8063 }, .{ .rows = 512, .n = n, .k = 8191 }, .{ .rows = 1023, .n = n, .k = 8192 }, .{ .rows = 1024, .n = n, .k = 8193 }, .{ .rows = 2048, .n = n, .k = 0 } };
}
pub fn spec(r: Reg) c.Spec {
    return switch (r.heads) {
        inline else => |h| switch (r.gated) {
            inline else => |gated| switch (r.fp64) {
                inline else => |fp| switch (r.control) {
                    inline else => |control| .{
                        .name = std.fmt.comptimePrint("{s}metal qwen27 chain {s} {s} {s}", .{ if (control) "control: " else "", comptime common.label(h), if (gated) "gated" else "plain", if (fp) "float64" else "main row bytes" }),
                        .accum = .f32,
                        .depth = depth,
                        .ops = 12,
                        .mids = if (gated) 1 else 0,
                        .out = .bf16,
                        .replaces = if (fp) null else "frozen main one-row tail and merge",
                        .nan = .{ .stated = "same output classes as main's one-row tail on identical nonfinite inputs" },
                        .tiles = .{ .{ .size = 16 }, .{}, .{ .size = 128 } },
                        .served = shapes(h, fp),
                    },
                },
            },
        },
    };
}

pub fn source(a: std.mem.Allocator, tensor: bool, baseline: bool, control: bool, heads: common.Heads) ![]u8 {
    const src = q.runtime_sources;
    const body = if (baseline) try std.mem.concat(a, u8, &.{ src.glue, @embedFile("q27_tail_baseline.metal"), src.attention_io, src.attention_prompt }) else try std.mem.concat(a, u8, &.{ src.glue, src.attention_mpp, src.attention_io, src.attention_prompt });
    defer a.free(body);
    const rewritten = if (tensor) try a.dupe(u8, body) else try q.simd_attention.rewrite(a, body);
    defer a.free(rewritten);
    const from = "? q : P + paths[node * MAXD + (q - P)]";
    if (control and std.mem.count(u8, rewritten, from) != 2) return error.ControlEdit;
    const edited = if (control) try std.mem.replaceOwned(u8, a, rewritten, from, "? P : P + paths[node * MAXD + (q - P)]") else try a.dupe(u8, rewritten);
    defer a.free(edited);
    const gm = common.geometry(heads, false);
    return std.fmt.allocPrint(a, "#define Q27_G 6\n#define Q27_KVH {d}\n{s}", .{ gm.kv_heads, edited });
}

pub const Case = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    reg: Reg,
    op: attn.Attention,
    old: attn.Attention,
    state: ?State = null,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    fill: g.Fill = .random,
    host: [5][]u16 = .{ &.{}, &.{}, &.{}, &.{}, &.{} },
    refs: [6]rig_.Ref = undefined,
    own_refs: [2]rig_.Ref = undefined,
    caches: [1]attn.Cache = undefined,
    baseline: []u8 = &.{},
    const State = struct {
        plan: attn.Plan,
        scratch: attn.Scratch,
        binding: attn.Binding,
        fn deinit(s: *State) void {
            s.binding.deinit();
            s.scratch.deinit();
            s.plan.deinit();
        }
    };
    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, r: Reg) !Case {
        var ops: [2]attn.Attention = undefined;
        var made: usize = 0;
        errdefer for (ops[0..made]) |*op| op.deinit();
        for (&ops, 0..) |*op, i| {
            const text = try source(a, device.tensorUnits(), i == 1, r.control and i == 0, r.heads);
            defer a.free(text);
            const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
            defer lib.deinit();
            op.* = try attn.Attention.init(device, lib);
            made += 1;
        }
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .reg = r, .op = ops[0], .old = ops[1] };
    }
    pub fn deinit(t: *Case) void {
        t.release();
        t.op.deinit();
        t.old.deinit();
        t.rig.deinit();
    }
    fn release(t: *Case) void {
        if (t.state) |*s| s.deinit();
        t.state = null;
        for (&t.host) |*h| {
            t.a.free(h.*);
            h.* = &.{};
        }
        t.a.free(t.baseline);
        t.baseline = &.{};
    }
    pub fn kernel(t: *Case) g.Kernel {
        return .{ .spec = spec(t.reg), .ctx = t, .vt = &g.hooks(Case), .control = if (t.reg.control) .shapes else null };
    }
    fn gm(t: *Case) planning.Geometry {
        return common.geometry(t.reg.heads, false);
    }
    fn stateFor(t: *Case, parents: []const i32, start: usize, frame: usize) !State {
        var plan = try attn.Plan.init(t.a, &.{.{ .parents = parents, .start = @intCast(start), .capacity = @intCast(t.shape.k + t.shape.rows + 64) }}, t.gm());
        errdefer plan.deinit();
        const limits = attn.Limits{ .rows = @intCast(frame), .streams = 1, .keys = @intCast(t.shape.k + t.shape.rows + 64), .shared_prefix = false, .geometry = t.gm() };
        var scratch = try attn.Scratch.init(t.rig.device, limits);
        errdefer scratch.deinit();
        var binding = try attn.Binding.init(t.rig.device, limits);
        errdefer binding.deinit();
        try scratch.upload(plan);
        try binding.upload(plan, &t.caches);
        return .{ .plan = plan, .scratch = scratch, .binding = binding };
    }
    pub fn prepare(t: *Case, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const gm_ = t.gm();
        if (shape.rows == 0 or shape.rows > 2048 or shape.n != gm_.heads or shape.k > 8193) return false;
        t.release();
        t.rig.reset();
        t.shape = shape;
        t.fill = fill;
        const cap = shape.k + shape.rows + 64;
        const hk = gm_.kv_heads;
        const heads = gm_.heads;
        const sizes = [_]usize{ shape.rows * heads * D, hk * cap * D, hk * cap * D, shape.rows * heads * 2 * D, shape.rows * hk * D };
        var rng_ = std.Random.DefaultPrng.init(seed);
        const rng = rng_.random();
        for (&t.host, sizes) |*h, n| {
            h.* = try t.a.alloc(u16, n);
            for (h.*) |*x| x.* = c.toBf16((rng.float(f32) * 2 - 1) * 0.5);
        }
        if (fill == .stress) {
            @memset(t.host[0], 0);
            @memset(t.host[1], 0);
            @memset(t.host[3], 0);
            for (0..hk) |h| for (0..cap) |key| @memset(t.host[2][(h * cap + key) * D ..][0..D], c.toBf16(oracle.stress(key)));
        }
        if (fill == .nonfinite) {
            for ([_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) }, 0..) |v, i| t.host[0][(i * heads + i) * D + 3] = c.toBf16(v);
        }
        for (t.host[0..4], 0..) |h, i| t.refs[i] = try t.rig.input(std.mem.sliceAsBytes(h));
        t.refs[4] = try t.rig.output(shape.rows * heads * D * 2);
        t.refs[5] = try t.rig.scratch(shape.rows * heads * D * 2);
        t.caches[0] = .{ .keys = common.ref(t.refs[1]), .values = common.ref(t.refs[2]), .capacity = @intCast(cap) };
        var parents: [2048]i32 = undefined;
        for (parents[0..shape.rows], 0..) |*p, r| p.* = @as(i32, @intCast(r)) - 1;
        t.own_refs = .{ try t.own(1), try t.own(2) };
        t.state = try t.stateFor(parents[0..shape.rows], shape.k, shape.rows);
        return true;
    }
    fn own(t: *Case, side: usize) !rig_.Ref {
        const gm_ = t.gm();
        const cap = t.shape.k + t.shape.rows + 64;
        for (0..t.shape.rows) |r| for (0..gm_.kv_heads) |h| @memcpy(t.host[4][(r * gm_.kv_heads + h) * D ..][0..D], t.host[side][(h * cap + t.shape.k + r) * D ..][0..D]);
        return t.rig.input(std.mem.sliceAsBytes(t.host[4]));
    }
    fn execute(t: *Case, op: *const attn.Attention, state: *State, query: rig_.Ref, key: rig_.Ref, value: rig_.Ref, gate: rig_.Ref, output: rig_.Ref, prompt: bool) !void {
        const pass = t.rig.begin();
        op.encode(pass.e, &state.scratch, &state.binding, state.plan, common.ref(query), common.ref(key), common.ref(value), common.ref(output), &t.caches, if (t.reg.gated) common.ref(gate) else null, prompt, false) catch |err| {
            try t.rig.finish(pass);
            return err;
        };
        try t.rig.finish(pass);
    }
    pub fn run(t: *Case) !void {
        const key = t.own_refs[0];
        const value = t.own_refs[1];
        try t.execute(&t.op, &t.state.?, t.refs[0], key, value, t.refs[3], t.refs[4], true);
    }
    pub fn replaced(t: *Case) ![]const u8 {
        t.a.free(t.baseline);
        t.baseline = try t.a.alloc(u8, t.shape.rows * t.gm().heads * D * 2);
        const key = t.own_refs[0];
        const value = t.own_refs[1];
        const qw = t.gm().heads * D;
        const kw = t.gm().kv_heads * D;
        for (0..t.shape.rows) |r| {
            var state = try t.stateFor(&.{-1}, t.shape.k + r, 1);
            defer state.deinit();
            var query = t.refs[0];
            query.off += r * qw * 2;
            var k = key;
            k.off += r * kw * 2;
            var v = value;
            v.off += r * kw * 2;
            var gate = t.refs[3];
            gate.off += r * qw * 4;
            var output = t.refs[5];
            output.off += r * qw * 2;
            try t.execute(&t.old, &state, query, k, v, gate, output, false);
        }
        @memcpy(t.baseline, t.rig.bytes(t.refs[5], t.baseline.len));
        return t.baseline;
    }
    pub fn count(t: *Case) usize {
        return t.shape.rows * t.gm().heads * D;
    }
    pub fn reference(t: *Case, want: []f64, mag: []f64) !void {
        if (t.fill == .stress) return oracle.stressReference(t.shape.k, t.shape.rows, t.gm().heads * D, t.reg.gated, want, mag);
        try oracle.reference(t.a, t.host[0], t.host[1], t.host[2], if (t.reg.gated) t.host[3] else null, t.shape.rows, t.gm().heads, t.gm().kv_heads, t.shape.k, t.shape.k + t.shape.rows + 64, want, mag);
    }
    pub fn bounds(t: *Case, want: []const f64, mag: []const f64, allow: []f64) !void {
        for (want, mag, allow) |w, m, *a| a.* = if (t.fill == .stress) @abs(w) * 0x1p-8 + 0x1p-133 else c.bound(spec(t.reg), t.shape, w, m) + 0x1p-11 * m;
    }
    pub fn simulate(t: *Case, acc: c.Accum, order: g.Order, output: []f64) !void {
        try oracle.simulate(t.a, t.shape.k, t.shape.rows, t.gm().heads * D, t.reg.gated, acc, order, output);
    }
    pub fn check(t: *Case, at: c.Gate) !?[]const u8 {
        if (at == .shapes and t.reg.fp64) {
            const want = try t.a.alloc(f64, 3 * t.count());
            defer t.a.free(want);
            const mag = want[t.count() .. 2 * t.count()];
            const allow = want[2 * t.count() ..];
            try t.reference(want[0..t.count()], mag);
            try t.bounds(want[0..t.count()], mag, allow);
            const now = try t.rig.outputBytes();
            for (want[0..t.count()], allow, 0..) |w, bound, i| {
                if (!(@abs(c.read(.bf16, now, i) - w) <= bound)) return "output leaves independent FP64 attention bound";
            }
        }
        if (at != .nonfinite) return null;
        const old = try t.replaced();
        const now = try t.rig.outputBytes();
        var bad: usize = 0;
        for (0..t.count()) |i| {
            const cls = c.classOf(c.read(.bf16, old, i));
            bad += @intFromBool(cls != .finite);
            if (cls != c.classOf(c.read(.bf16, now, i))) return "output class differs from main's row";
        }
        if (bad == 0 or bad == t.count()) return "nonfinite fixture must produce both finite and nonfinite output classes";
        return null;
    }
};

pub const Suite = struct {
    cases: std.ArrayList(*Case) = .empty,
    pub fn deinit(s: *Suite, a: std.mem.Allocator) void {
        for (s.cases.items) |x| {
            x.deinit();
            a.destroy(x);
        }
        s.cases.deinit(a);
    }
    pub fn collect(s: *Suite, a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kernels: *std.ArrayList(g.Kernel), only: ?[]const u8) !void {
        for (std.enums.values(common.Heads)) |heads| for ([_]Reg{
            .{ .heads = heads },                  .{ .heads = heads, .gated = true },
            .{ .heads = heads, .fp64 = true },    .{ .heads = heads, .fp64 = true, .gated = true },
            .{ .heads = heads, .control = true },
        }) |r| {
            if (only) |needle| if (std.mem.indexOf(u8, spec(r).name, needle) == null) continue;
            const x = try a.create(Case);
            errdefer a.destroy(x);
            x.* = try Case.init(a, device, queue, r);
            s.cases.append(a, x) catch |err| {
                x.deinit();
                return err;
            };
            try kernels.append(a, x.kernel());
        };
    }
};
