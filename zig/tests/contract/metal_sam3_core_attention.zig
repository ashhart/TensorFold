//! The tensor library's FP32 attention at head dims 32 and 64 (tensor_attention.metal), as the SAM3 family serves it.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const sources = @import("kernel_sources");
const rig_ = @import("rig.zig");
const c = contract.spec;
const g = contract.gates;
const ar = contract.attention_range;
const vt = g.hooks(Probe);
const period = 67;

const Kind = enum { full32, full64, tiles64, causal64 };

const Edit = struct { from: []const u8, to: []const u8, why: []const u8 };
const value_edit = Edit{ .from = "else attention_pv(acc[d], acc[d + 1], s[k], v0, v1, ushort(lane));", .to = "else attention_pv(acc[d], acc[d + 1], s[k], v0, v0, ushort(lane));", .why = "P x V reads one value fragment twice" };
const causal_edit = Edit{ .from = "((r0 + (e >> 2) * 8) < (c0 + 16 * f + TF_COL(e)) ||", .to = "((r0 + (e >> 2) * 8) <= (c0 + 16 * f + TF_COL(e)) ||", .why = "a causal row misses its own key" };

fn editOf(kind: Kind) Edit {
    return if (kind == .causal64) causal_edit else value_edit;
}

pub fn text(a: std.mem.Allocator, control: ?Edit) ![]u8 {
    const raw = try sources.tensor_library.source(a, sources.sam3_source);
    const e = control orelse return raw;
    defer a.free(raw);
    if (std.mem.count(u8, raw, e.from) != 1) return error.ControlTextNotFoundOnce;
    return std.mem.replaceOwned(u8, a, raw, e.from, e.to);
}

pub fn limitText(a: std.mem.Allocator, name: []const u8, control: bool) ![]u8 {
    if (!control) return text(a, null);
    return text(a, if (std.mem.endsWith(u8, name, "_causal64")) causal_edit else value_edit);
}

fn function(kind: Kind) []const u8 {
    return switch (kind) {
        .full32 => "im_attention_f32_full32",
        .full64, .tiles64 => "im_attention_f32_full64",
        .causal64 => "im_attention_f32_causal64",
    };
}

fn label(kind: Kind, control: bool) []const u8 {
    if (control) return switch (kind) {
        .full32 => "control: sam3 core attention D32 " ++ value_edit.why,
        .full64 => "control: sam3 core attention D64 " ++ value_edit.why,
        .tiles64 => "control: sam3 core attention D64 tiles " ++ value_edit.why,
        .causal64 => "control: sam3 core attention D64 causal " ++ causal_edit.why,
    };
    return switch (kind) {
        .full32 => "metal sam3 core attention D32 cross",
        .full64 => "metal sam3 core attention D64 full grid",
        .tiles64 => "metal sam3 core attention D64 tiles",
        .causal64 => "metal sam3 core attention D64 causal text",
    };
}

fn width(kind: Kind) usize {
    return if (kind == .full32) 32 else 64;
}

fn heads(kind: Kind) usize {
    return if (kind == .full32) 8 else 16;
}

fn batches(kind: Kind) usize {
    return if (kind == .tiles64) 9 else 1;
}

// rows = query rows, n = heads x width, k = keys (causal: the rows themselves).
fn served(kind: Kind) []const c.Shape {
    return switch (kind) {
        // Grounding over the grid and the cues, the class token, the queries, the mask decoder's padded 16-wide heads.
        .full32 => &.{
            .{ .rows = 5184, .n = 256, .k = 5184 }, .{ .rows = 5184, .n = 256, .k = 33 }, .{ .rows = 1, .n = 256, .k = 5184 },
            .{ .rows = 201, .n = 256, .k = 201 },   .{ .rows = 201, .n = 256, .k = 33 },  .{ .rows = 7, .n = 256, .k = 5184 },
            .{ .rows = 5184, .n = 256, .k = 7 },
        },
        .full64 => &.{ .{ .rows = 5184, .n = 1024, .k = 5184 }, .{ .rows = 65, .n = 1024, .k = 97 } },
        .tiles64 => &.{ .{ .rows = 576, .n = 1024, .k = 576 }, .{ .rows = 65, .n = 1024, .k = 97 } },
        .causal64 => &.{ .{ .rows = 32, .n = 1024, .k = 32 }, .{ .rows = 7, .n = 1024, .k = 7 } },
    };
}

fn depth(s: c.Shape) usize {
    return s.n + s.k;
}

fn stressValue(j: usize, n: usize) f32 {
    const half = n / 2;
    const at = if (j < half) j else j - half;
    if (j >= half * 2 or at >= 256) return 0;
    const big: f32 = if (at % 2 == 0) 127 else 16256;
    return if (j < half) big + @as(f32, if (at < 8 and at % 2 == 0) 1 else 0) else -big;
}

const Probe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    kind: Kind,
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
            .name = label(p.kind, p.control),
            .accum = .f32,
            .depth = depth,
            .ops = 8,
            .out = .f32,
            .served = served(p.kind),
            .tiles = .{ .{ .size = 64 }, .{}, .{ .size = if (p.kind == .causal64) 0 else 32 } },
            // A causal row sums at most its own position's keys: gate 4's four rows cannot make fp16 or bf16 miss.
            .sums = if (p.kind == .causal64) .{ .f16 = 0, .bf16 = 0, .why = "causal rows of gate 4 see at most four keys" } else .{},
        };
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

    fn keys(p: *const Probe) usize {
        return if (p.kind == .causal64) p.shape.rows else p.shape.k;
    }

    pub fn prepare(p: *Probe, s: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (s.rows == 0 or s.k == 0 or s.n != heads(p.kind) * width(p.kind)) return false;
        p.reset();
        p.shape = s;
        p.stress = fill == .stress;
        const nb = batches(p.kind);
        const nk = p.keys();
        p.q = try p.a.alloc(f32, p.count());
        p.k = try p.a.alloc(f32, nb * nk * s.n);
        p.v = try p.a.alloc(f32, p.k.len);
        p.half = try p.a.alloc(f64, p.count());
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (0..nb) |b| for (0..s.rows) |r| {
            const row = p.q[(b * s.rows + r) * s.n ..][0..s.n];
            if (r < period or p.kind == .causal64) {
                for (row) |*x| x.* = if (p.stress) 0 else rng.float(f32) - 0.5;
            } else @memcpy(row, p.q[(b * s.rows + r % period) * s.n ..][0..s.n]);
        };
        for (0..nb) |b| for (0..nk) |j| for (0..s.n) |d| {
            const i = (b * nk + j) * s.n + d;
            p.k[i] = if (p.stress) 0 else rng.float(f32) - 0.5;
            p.v[i] = if (p.stress) stressValue(j, nk) else rng.float(f32) * 2 - 1;
        };
        if (fill == .nonfinite) {
            p.q[0] = std.math.nan(f32);
            p.q[s.n] = std.math.inf(f32);
            p.q[2 * s.n] = -std.math.inf(f32);
            p.v[1] = std.math.inf(f32);
            p.v[2] = -std.math.inf(f32);
            p.v[3] = std.math.nan(f32);
        }
        for ([_][]const f32{ p.q, p.k, p.v }, &p.ins) |values, *input| input.* = try p.rig.input(std.mem.sliceAsBytes(values));
        p.out = try p.rig.output(p.count() * 4);
        return true;
    }

    pub fn run(p: *Probe) !void {
        const s = p.shape;
        const w = width(p.kind);
        const nh = heads(p.kind);
        const nk = p.keys();
        const nq = (s.rows + 63) / 64;
        const vals = [_]usize{
            batches(p.kind), nh, s.rows, nk,       1, nq,  (nk + 31) / 32, s.rows / 64, nk / 32, s.rows % 64,  nk % 32, 0,
            s.rows * s.n,    w,  s.n,    nk * s.n, w, s.n, nk * s.n,       w,           s.n,     s.rows * s.n, w,       s.n,
        };
        var params: [24]i32 = undefined;
        for (vals, &params) |x, *v| v.* = @intCast(x);
        const pass = p.rig.begin();
        pass.e.setPipeline(p.pipe);
        for (p.ins, 0..) |r, i| pass.e.setBuffer(r.buf, r.off, i);
        pass.e.setValue(params, 3);
        pass.e.setValue([4]f32{ 1 / @sqrt(@as(f32, @floatFromInt(w))), 0, 0, 0 }, 4);
        pass.e.setBuffer(p.out.buf, p.out.off, 5);
        pass.e.dispatchThreads(mtl.Size.of(nq * 32, nh * 4, batches(p.kind)), mtl.Size.of(32, 4, 1));
        try p.rig.finish(pass);
    }

    pub fn count(p: *const Probe) usize {
        return batches(p.kind) * p.shape.rows * p.shape.n;
    }

    pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
        const s = p.shape;
        const w = width(p.kind);
        const nk = p.keys();
        const kv = try p.a.alloc(f64, 2 * nk * w + 2 * nk);
        defer p.a.free(kv);
        const keys_ = kv[0 .. nk * w];
        const values = kv[nk * w ..][0 .. nk * w];
        const weights = kv[2 * nk * w ..][0..nk];
        const tau = kv[2 * nk * w + nk ..];
        var query: [64]f64 = undefined;
        for (0..batches(p.kind)) |b| for (0..heads(p.kind)) |h| {
            for (0..nk) |j| for (0..w) |d| {
                const src = (b * nk + j) * s.n + h * w + d;
                keys_[j * w + d] = p.k[src];
                values[j * w + d] = p.v[src];
            };
            for (0..s.rows) |r| {
                const at = (b * s.rows + r) * s.n + h * w;
                if (r >= period and p.kind != .causal64) {
                    const from = (b * s.rows + r % period) * s.n + h * w;
                    @memcpy(want[at..][0..w], want[from..][0..w]);
                    @memcpy(p.half[at..][0..w], p.half[from..][0..w]);
                    continue;
                }
                // Row r of a causal head sees keys 0 to r.
                const seen = if (p.kind == .causal64) r + 1 else nk;
                const blocks: f64 = @floatFromInt((seen + 31) / 32);
                const form = ar.Form{
                    .score = @floatFromInt(w / 16 * 6 * ar.mma_step),
                    .score_round = 4,
                    .fast_exp = false,
                    .chunk_exp = false,
                    .p_rel = 0,
                    .num = (12 * ar.mma_step + 1) * blocks,
                    .den = 3 * blocks + 3,
                    .div = 4,
                    .exp_ulps = .{ 4 * blocks, 2 * std.math.log2e + 0.5 },
                };
                for (query[0..w], p.q[at..][0..w]) |*x, q| x.* = q;
                ar.head(query[0..w], keys_[0 .. seen * w], values[0 .. seen * w], seen, w, 1 / @sqrt(@as(f64, @floatFromInt(w))), form, weights, tau, want[at..][0..w], p.half[at..][0..w]);
            }
        };
        @memcpy(mag, p.half);
    }

    pub fn bounds(p: *Probe, want: []const f64, _: []const f64, allow: []f64) !void {
        for (want, p.half, allow) |w, half, *v| v.* = if (p.stress) c.boundExact(p.spec(), w) else half;
    }

    pub fn simulate(p: *Probe, acc: c.Accum, order: g.Order, out: []f64) !void {
        const s = p.shape;
        const terms = try p.a.alloc(f64, p.keys());
        defer p.a.free(terms);
        for (0..batches(p.kind)) |b| for (0..s.rows) |r| {
            const seen = if (p.kind == .causal64) r + 1 else p.keys();
            for (terms[0..seen], 0..) |*x, j| x.* = stressValue(j, p.keys());
            const result = contract.rows.sumIn(acc, order, terms[0..seen]) / @as(f64, @floatFromInt(seen));
            @memset(out[(b * s.rows + r) * s.n ..][0..s.n], result);
        };
    }

    fn deinit(p: *Probe) void {
        p.reset();
        p.rig.deinit();
        p.pipe.deinit();
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    probes: std.ArrayList(*Probe) = .empty,

    pub fn collect(s: *Suite, device: mtl.Device, queue: mtl.Queue, kernels: *std.ArrayList(g.Kernel), only: ?[]const u8) !void {
        if (!device.tensorUnits()) return;
        // One library a source: the engine's, and each planted edit's.
        for ([_]?Edit{ null, value_edit, causal_edit }) |edit| {
            var lib: ?mtl.Library = null;
            defer if (lib) |l| l.deinit();
            for (std.enums.values(Kind)) |kind| {
                const control = edit != null;
                if (control and !std.mem.eql(u8, editOf(kind).from, edit.?.from)) continue;
                if (only) |needle| if (std.mem.indexOf(u8, label(kind, control), needle) == null) continue;
                if (lib == null) {
                    const body = try text(s.a, edit);
                    defer s.a.free(body);
                    lib = try mtl.Library.fromSource(device, body, mtl.CompileOptions.mlx());
                }
                const p = try s.a.create(Probe);
                errdefer s.a.destroy(p);
                p.* = .{ .a = s.a, .rig = .{ .a = s.a, .device = device, .queue = queue }, .pipe = try mtl.Pipeline.init(device, lib.?, function(kind), false), .kind = kind, .control = control };
                try s.probes.append(s.a, p);
                try kernels.append(s.a, .{ .spec = p.spec(), .ctx = p, .vt = &vt, .control = if (control) .shapes else null });
            }
        }
    }

    pub fn deinit(s: *Suite) void {
        for (s.probes.items) |p| {
            p.deinit();
            s.a.destroy(p);
        }
        s.probes.deinit(s.a);
    }
};
