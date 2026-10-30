//! A random Nemotron-H MoE layer as the checkpoint stores it, on the device, and its float64 references.

const std = @import("std");
const core = @import("core");
const rig = @import("../rig.zig");
const nm = @import("../../moe_nemotron.zig");
const ops = @import("../../ops/ops.zig");
const DeviceBuffer = @import("../../memory.zig").DeviceBuffer;

const quant = core.quant;
const Rig = rig.Rig;
const gpa = rig.gpa;
const at = rig.at;

pub const Shape = struct {
    experts: usize,
    top_k: usize,
    hidden: usize,
    width: usize,
    scaling: f32 = 2.5,
    norm: bool = true,
    /// Router rows e and e + E / 2 (and their biases) equal: every logit pair ties.
    ties: bool = false,
};

const bits = 4;
const group = 64;

/// One MLX affine matrix stack on the host: `rows` rows of K.
const Mat = struct {
    words: []u32,
    scales: []u16,
    biases: []u16,
    k: usize,

    fn init(a: std.mem.Allocator, rng: *rig.Rng, rows: usize, k: usize) !Mat {
        const m: Mat = .{
            .words = try a.alloc(u32, rows * k / 8),
            .scales = try a.alloc(u16, rows * k / group),
            .biases = try a.alloc(u16, rows * k / group),
            .k = k,
        };
        for (m.words) |*w| w.* = @truncate(rng.next() >> 16);
        // codes centred on 8 and scaled by 1 / sqrt(K): a product of unit inputs stays near unit size
        const s0 = 0.375 / @sqrt(@as(f32, @floatFromInt(k)));
        for (m.scales, m.biases) |*s, *b| {
            const sv = s0 * (1.0 + rng.unit() / 4.0);
            s.* = rig.bf16Bits(sv);
            b.* = rig.bf16Bits(-8.0 * @as(f32, @floatCast(rig.bf16Value(s.*))) * (1.0 + rng.unit() / 16.0));
        }
        return m;
    }

    /// Row `r` dequantized into `out` (K values).
    fn row(m: Mat, r: usize, out: []f64) void {
        const wr = m.words[r * m.k / 8 ..][0 .. m.k / 8];
        for (wr, 0..) |w, i| for (0..8) |j| {
            const g = (r * m.k + i * 8 + j) / group;
            const q: f64 = @floatFromInt((w >> @intCast(4 * j)) & 0xF);
            out[i * 8 + j] = q * rig.bf16Value(m.scales[g]) + rig.bf16Value(m.biases[g]);
        };
    }

    /// The matrix on the device: words, scales, biases.
    fn upload(m: Mat, t: *Rig) ![3]DeviceBuffer {
        var out: [3]DeviceBuffer = undefined;
        out[0] = try t.upload(m.words);
        errdefer out[0].free();
        out[1] = try t.upload(m.scales);
        errdefer out[1].free();
        out[2] = try t.upload(m.biases);
        return out;
    }
};

pub const Layer = struct {
    shape: Shape,
    arena: std.heap.ArenaAllocator,
    gate: []f32,
    bias: []f32,
    fc1: Mat, // (E, NI, D)
    fc2: Mat, // (E, D, NI)
    up: Mat, // (2 NI, D)
    down: Mat, // (D, 2 NI)
    bufs: [14]DeviceBuffer,
    /// The whole layer on one rank.
    view: nm.Layer,

    pub fn init(t: *Rig, rng: *rig.Rng, s: Shape) !*Layer {
        const l = try gpa.create(Layer);
        errdefer gpa.destroy(l);
        l.shape = s;
        l.arena = .init(gpa);
        errdefer l.arena.deinit();
        const a = l.arena.allocator();
        const e = s.experts;
        const d = s.hidden;
        const ni = s.width;
        l.gate = try a.alloc(f32, e * d);
        l.bias = try a.alloc(f32, e);
        const c = 3.0 / @sqrt(@as(f32, @floatFromInt(d)));
        for (l.gate) |*g| g.* = @floatCast(rig.bf16Value(rig.bf16Bits(rng.unit() * c)));
        for (l.bias) |*b| b.* = rng.unit() / 8.0;
        if (s.ties) for (0..e / 2) |i| {
            @memcpy(l.gate[(i + e / 2) * d ..][0..d], l.gate[i * d ..][0..d]);
            l.bias[i + e / 2] = l.bias[i];
        };
        l.fc1 = try Mat.init(a, rng, e * ni, d);
        l.fc2 = try Mat.init(a, rng, e * d, ni);
        l.up = try Mat.init(a, rng, 2 * ni, d);
        l.down = try Mat.init(a, rng, d, 2 * ni);
        var loaded: usize = 0;
        errdefer for (l.bufs[0..loaded]) |*b| b.free();
        l.bufs[0] = try t.upload(l.gate);
        loaded += 1;
        l.bufs[1] = try t.upload(l.bias);
        loaded += 1;
        for ([_]Mat{ l.fc1, l.fc2, l.up, l.down }, 0..) |m, i| {
            const b = try m.upload(t);
            @memcpy(l.bufs[2 + 3 * i ..][0..3], &b);
            loaded += 3;
        }
        l.view = l.rank(0, e, true, false);
        return l;
    }

    /// Experts from `first` of the stack in bufs[at_buf..] (each n rows of k), as a projection of one expert's shape.
    fn part(l: *const Layer, at_buf: usize, first: usize, n: usize, k: usize) quant.Projection {
        const b = l.bufs[at_buf..][0..3];
        const rows = first * n;
        return .{ .n = @intCast(n), .k = @intCast(k), .handle = .{ .mlx = .{
            .words = at(b[0]) + rows * k / 2,
            .scale = at(b[1]) + rows * (k / group) * 2,
            .bias = at(b[2]) + rows * (k / group) * 2,
            .tables = .bf16,
            .bits = bits,
            .group = group,
        } } };
    }

    /// A rank's view: experts [first, first + own), the shared expert when `shared`, an fp32 share when `share`.
    pub fn rank(l: *const Layer, first: usize, own: usize, shared: bool, share: bool) nm.Layer {
        const s = l.shape;
        return .{
            .router = at(l.bufs[0]),
            .bias = at(l.bufs[1]),
            .fc1 = l.part(2, first, s.width, s.hidden),
            .fc2 = l.part(5, first, s.hidden, s.width),
            .shared_up = if (shared) l.part(8, 0, 2 * s.width, s.hidden) else null,
            .shared_down = if (shared) l.part(11, 0, s.hidden, 2 * s.width) else null,
            .experts = s.experts,
            .first = first,
            .own = own,
            .top_k = s.top_k,
            .scaling = s.scaling,
            .norm = s.norm,
            .share = share,
        };
    }

    pub fn deinit(l: *Layer) void {
        for (&l.bufs) |*b| b.free();
        l.arena.deinit();
        gpa.destroy(l);
    }

    /// Row `r`'s scores sigmoid(x . gate) and selection keys (score + bias) in float64.
    pub fn scores(l: *const Layer, x: []const f64, score: []f64, sel: []f64) void {
        const d = l.shape.hidden;
        for (0..l.shape.experts) |e| {
            var logit: f64 = 0;
            for (0..d) |i| logit += x[i] * l.gate[e * d + i];
            score[e] = 1.0 / (1.0 + @exp(-logit));
            sel[e] = score[e] + l.bias[e];
        }
    }

    /// The layer's output row for input row `x` given the picked experts and their weights, in float64.
    pub fn output(l: *const Layer, x: []const f64, picks: []const i32, wts: []const f64, out: []f64) !void {
        const d = l.shape.hidden;
        const ni = l.shape.width;
        @memset(out, 0);
        const wrow = try gpa.alloc(f64, @max(d, 2 * ni));
        defer gpa.free(wrow);
        const act = try gpa.alloc(f64, 2 * ni);
        defer gpa.free(act);
        // the routed picks, then the shared expert at weight 1
        for (0..picks.len + 1) |k| {
            const shared = k == picks.len;
            const n = if (shared) 2 * ni else ni;
            for (0..n) |j| {
                if (shared) l.up.row(j, wrow[0..d]) else l.fc1.row(@as(usize, @intCast(picks[k])) * ni + j, wrow[0..d]);
                var u: f64 = 0;
                for (0..d) |i| u += x[i] * wrow[i];
                act[j] = if (u > 0) u * u else 0;
            }
            const w = if (shared) 1.0 else wts[k];
            for (0..d) |c| {
                if (shared) l.down.row(c, wrow[0..n]) else l.fc2.row(
                    @as(usize, @intCast(picks[k])) * d + c,
                    wrow[0..n],
                );
                var y: f64 = 0;
                for (0..n) |j| y += act[j] * wrow[j];
                out[c] += w * y;
            }
        }
    }
};

/// Random activation rows (rows, D): their bits and float64 values.
pub const Input = struct {
    bits: []u16,
    values: []f64,
    buf: DeviceBuffer,

    pub fn init(t: *Rig, rng: *rig.Rng, rows: usize, d: usize) !Input {
        const b = try gpa.alloc(u16, rows * d);
        errdefer gpa.free(b);
        const v = try gpa.alloc(f64, rows * d);
        errdefer gpa.free(v);
        for (b, v) |*h, *f| {
            h.* = t.bits(rng.unit());
            f.* = t.value(h.*);
        }
        return .{ .bits = b, .values = v, .buf = try t.upload(b) };
    }

    pub fn deinit(i: *Input) void {
        i.buf.free();
        gpa.free(i.bits);
        gpa.free(i.values);
    }

    pub fn tensor(i: Input, t: *const Rig, row: usize, d: usize) ops.Tensor {
        return .{ .ptr = at(i.buf) + row * d * 2, .kind = kind(t) };
    }
};

pub fn kind(t: *const Rig) ops.Kind {
    return if (t.bf16) .bf16 else .f16;
}

/// A device range's bytes on the host (the caller frees them).
pub fn bytesAt(t: *Rig, ptr: u64, len: usize) ![]u8 {
    const out = try gpa.alloc(u8, len);
    errdefer gpa.free(out);
    try t.stream.synchronize();
    if (len == 0) return out;
    const runtime = @import("../../runtime.zig");
    try runtime.check(t.r.api.hipMemcpyDtoH(out.ptr, @ptrFromInt(ptr), len));
    return out;
}

test "the fixture's matrices stay unit sized" {
    var rng: rig.Rng = .{ .state = 3 };
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const m = try Mat.init(arena.allocator(), &rng, 4, 512);
    var row: [512]f64 = undefined;
    m.row(1, &row);
    var sum: f64 = 0;
    for (row) |v| sum += v * v;
    try std.testing.expect(sum > 0.01 and sum < 10);
}
