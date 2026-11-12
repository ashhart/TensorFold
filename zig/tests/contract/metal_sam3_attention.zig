const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const sources = @import("kernel_sources");
const rig_ = @import("rig.zig");
const c = contract.spec;
const g = contract.gates;
const ar = contract.attention_range;
const dim = 256;
const period = 67;
const vt = g.hooks(Probe);

const single_shapes = [_]c.Shape{
    .{ .rows = 65, .n = dim, .k = 97 },
    .{ .rows = 5184, .n = dim, .k = 5184 },
    .{ .rows = 5184, .n = dim, .k = 36544 },
    .{ .rows = 5185, .n = dim, .k = 33 },
};
const batch_shapes = [_]c.Shape{
    .{ .rows = 65, .n = 2 * dim, .k = 97 },
    .{ .rows = 5185, .n = 2 * dim, .k = 33 },
};

fn depth(s: c.Shape) usize {
    return dim + s.k;
}

fn label(batched: bool, control: bool) []const u8 {
    return if (control) "control: sam3 attention D256 value partition" else if (batched) "metal sam3 attention D256 batches and heads" else "metal sam3 attention D256 full";
}

fn stressValue(j: usize, n: usize) f32 {
    const half = n / 2;
    const at = if (j < half) j else j - half;
    if (j >= half * 2 or at >= 256) return 0;
    const big: f32 = if (at % 2 == 0) 127 else 16256;
    return if (j < half) big + @as(f32, if (at < 8 and at % 2 == 0) 1 else 0) else -big;
}

pub fn text(a: std.mem.Allocator, control: bool) ![]u8 {
    const raw = try sources.tensor_library.source(a, sources.sam3_source);
    if (!control) return raw;
    defer a.free(raw);
    const from = "P[19] + value_part * VDIM;";
    if (std.mem.count(u8, raw, from) != 1) return error.ControlTextNotFound;
    return std.mem.replaceOwned(u8, a, raw, from, "P[19];");
}

const Probe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    batched: bool,
    control: bool,
    shape: c.Shape = undefined,
    stress: bool = false,
    q: []f32 = &.{},
    k: []f32 = &.{},
    v: []f32 = &.{},
    half: []f64 = &.{},
    ins: [3]rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    fn spec(p: *const Probe) c.Spec {
        return .{
            .name = label(p.batched, p.control),
            .accum = .f32,
            .depth = depth,
            .ops = 8,
            .out = .f32,
            .served = if (p.batched) &batch_shapes else &single_shapes,
            .tiles = .{ .{ .size = 64 }, .{}, .{ .size = 32 } },
        };
    }

    fn heads(p: *const Probe) usize {
        return if (p.batched) 2 else 1;
    }

    fn batches(p: *const Probe) usize {
        return if (p.batched) 2 else 1;
    }

    fn reset(p: *Probe) void {
        for ([_][]f32{ p.q, p.k, p.v }) |x| p.a.free(x);
        p.a.free(p.half);
        p.q = &.{};
        p.k = &.{};
        p.v = &.{};
        p.half = &.{};
        p.rig.reset();
    }

    pub fn prepare(p: *Probe, s: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (s.rows == 0 or s.k == 0 or s.n != p.heads() * dim) return false;
        p.reset();
        p.shape = s;
        p.stress = fill == .stress;
        p.q = try p.a.alloc(f32, p.count());
        p.k = try p.a.alloc(f32, p.batches() * s.k * s.n);
        p.v = try p.a.alloc(f32, p.k.len);
        p.half = try p.a.alloc(f64, p.count());
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (0..p.batches()) |b| for (0..s.rows) |r| {
            const row = p.q[(b * s.rows + r) * s.n ..][0..s.n];
            if (r < period) {
                for (row) |*x| x.* = if (p.stress) 0 else rng.float(f32) - 0.5;
            } else @memcpy(row, p.q[(b * s.rows + r % period) * s.n ..][0..s.n]);
        };
        for (0..p.batches()) |b| for (0..s.k) |j| for (0..s.n) |d| {
            const i = (b * s.k + j) * s.n + d;
            p.k[i] = if (p.stress) 0 else rng.float(f32) - 0.5;
            p.v[i] = if (p.stress) stressValue(j, s.k) else rng.float(f32) * 2 - 1;
        };
        if (fill == .nonfinite) {
            p.q[0] = std.math.nan(f32);
            p.q[s.n] = std.math.inf(f32);
            p.q[2 * s.n] = -std.math.inf(f32);
            p.q[3 * s.n] = std.math.inf(f32);
            p.q[3 * s.n + 1] = -std.math.inf(f32);
            p.v[0] = std.math.inf(f32);
            p.v[1] = -std.math.inf(f32);
            p.v[2] = std.math.inf(f32);
            p.v[s.n + 2] = -std.math.inf(f32);
            p.v[3] = std.math.nan(f32);
            if (p.batched) p.k[s.k * s.n + dim] = std.math.nan(f32);
        }
        for ([_][]const f32{ p.q, p.k, p.v }, &p.ins) |values, *input| input.* = try p.rig.input(std.mem.sliceAsBytes(values));
        p.out = try p.rig.output(p.count() * 4);
        return true;
    }

    fn launch(p: *Probe) !void {
        const s = p.shape;
        const nheads = p.heads();
        const nq = (s.rows + 63) / 64;
        const vals = [_]usize{
            p.batches(),  nheads, s.rows, s.k,       1,   nq,  (s.k + 31) / 32, s.rows / 64, s.k / 32, s.rows % 64,  s.k % 32, 0,
            s.rows * s.n, dim,    s.n,    s.k * s.n, dim, s.n, s.k * s.n,       dim,         s.n,      s.rows * s.n, dim,      s.n,
        };
        var params: [24]i32 = undefined;
        for (vals, &params) |x, *v| v.* = @intCast(x);
        const pass = p.rig.begin();
        pass.e.setPipeline(p.pipe);
        for (p.ins, 0..) |r, i| pass.e.setBuffer(r.buf, r.off, i);
        pass.e.setValue(params, 3);
        pass.e.setValue([4]f32{ 0.0625, 0, 0, 0 }, 4);
        pass.e.setBuffer(p.out.buf, p.out.off, 5);
        pass.e.dispatchThreads(mtl.Size.of(nq * 32, nheads * 4, p.batches() * 2), mtl.Size.of(32, 4, 1));
        try p.rig.finish(pass);
    }

    pub fn run(p: *Probe) !void {
        const before = try p.a.dupe(u8, try p.rig.inputBytes());
        defer p.a.free(before);
        try p.launch();
        if (p.shape.rows > 64) {
            const first = try p.a.dupe(u8, p.rig.bytes(p.out, p.count() * 4));
            defer p.a.free(first);
            for (1..5) |_| {
                try p.launch();
                if (!std.mem.eql(u8, first, p.rig.bytes(p.out, first.len))) return error.AttentionNotRepeatable;
            }
        }
        if (!std.mem.eql(u8, before, try p.rig.inputBytes())) return error.AttentionInputsChanged;
        if (!try p.rig.intact()) return error.AttentionGuardsChanged;
    }

    pub fn count(p: *const Probe) usize {
        return p.batches() * p.shape.rows * p.shape.n;
    }

    pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
        const s = p.shape;
        const kv = try p.a.alloc(f64, 2 * s.k * dim + 2 * s.k);
        defer p.a.free(kv);
        const keys = kv[0 .. s.k * dim];
        const values = kv[s.k * dim ..][0 .. s.k * dim];
        const weights = kv[2 * s.k * dim ..][0..s.k];
        const tau = kv[2 * s.k * dim + s.k ..];
        var query: [dim]f64 = undefined;
        const blocks: f64 = @floatFromInt((s.k + 31) / 32);
        const form = ar.Form{
            .score = dim / 16 * 6 * ar.mma_step,
            .score_round = 4,
            .fast_exp = false,
            .chunk_exp = false,
            .p_rel = 0,
            .num = (12 * ar.mma_step + 1) * blocks,
            .den = 3 * blocks + 3,
            .div = 4,
            .exp_ulps = .{ 4 * blocks, 2 * std.math.log2e + 0.5 },
        };
        for (0..p.batches()) |b| for (0..p.heads()) |h| {
            for (0..s.k) |j| for (0..dim) |d| {
                const src = (b * s.k + j) * s.n + h * dim + d;
                keys[j * dim + d] = p.k[src];
                values[j * dim + d] = p.v[src];
            };
            for (0..s.rows) |r| {
                const at = (b * s.rows + r) * s.n + h * dim;
                if (r >= period) {
                    const from = (b * s.rows + r % period) * s.n + h * dim;
                    @memcpy(want[at..][0..dim], want[from..][0..dim]);
                    @memcpy(p.half[at..][0..dim], p.half[from..][0..dim]);
                    continue;
                }
                for (&query, p.q[at..][0..dim]) |*x, q| x.* = q;
                ar.head(&query, keys, values, s.k, dim, 0.0625, form, weights, tau, want[at..][0..dim], p.half[at..][0..dim]);
            }
        };
        @memcpy(mag, p.half);
    }

    pub fn bounds(p: *Probe, want: []const f64, _: []const f64, allow: []f64) !void {
        for (want, p.half, allow) |w, half, *v| v.* = if (p.stress) c.boundExact(p.spec(), w) else half;
    }

    pub fn simulate(p: *Probe, acc: c.Accum, order: g.Order, out: []f64) !void {
        const terms = try p.a.alloc(f64, p.shape.k);
        defer p.a.free(terms);
        for (terms, 0..) |*x, j| x.* = stressValue(j, terms.len);
        const result = contract.rows.sumIn(acc, order, terms) / @as(f64, @floatFromInt(terms.len));
        @memset(out, result);
    }

    fn deinit(p: *Probe) void {
        p.reset();
        p.rig.deinit();
        p.pipe.deinit();
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    probes: [3]?*Probe = @splat(null),

    pub fn collect(s: *Suite, device: mtl.Device, queue: mtl.Queue, kernels: *std.ArrayList(g.Kernel), only: ?[]const u8) !void {
        if (!device.tensorUnits()) return;
        for (0..3) |i| {
            const batched = i == 1;
            const control = i == 2;
            if (only) |needle| if (std.mem.indexOf(u8, label(batched, control), needle) == null) continue;
            const body = try text(s.a, control);
            defer s.a.free(body);
            const pipe = try rig_.pipeline(device, body, "im_attention_f32_full256");
            errdefer pipe.deinit();
            const p = try s.a.create(Probe);
            errdefer s.a.destroy(p);
            try kernels.ensureUnusedCapacity(s.a, 1);
            p.* = .{ .a = s.a, .rig = .{ .a = s.a, .device = device, .queue = queue }, .pipe = pipe, .batched = batched, .control = control };
            s.probes[i] = p;
            kernels.appendAssumeCapacity(.{ .spec = p.spec(), .ctx = p, .vt = &vt, .control = if (control) .shapes else null });
        }
    }

    pub fn deinit(s: *Suite) void {
        for (s.probes) |maybe| if (maybe) |p| {
            p.deinit();
            s.a.destroy(p);
        };
    }
};
