//! A layer's launch read back stage by stage: each stage's output against float64 on the GPU's own input to it.
const std = @import("std");
const pkg = @import("dsv41_metal");
const ref = @import("ref.zig");
const fixture = @import("fixture.zig");

pub const Ref = pkg.ctx.Ref;
const Fp8 = pkg.weights.Fp8;
const Stage = ref.Stage;

pub const Probe = struct {
    a: std.mem.Allocator,
    r: *fixture.Rig,
    w: *const pkg.weights.Layer,
    rows: usize,
    p0: u32,
    /// the layer number printed beside each stage (null: the layer's own; the head prints the layer count)
    at: ?usize = null,
    /// the MoE's (experts, picks) when the layer is not the backbone's (a DSpark stage)
    moe: ?[2]u32 = null,
    stages: std.ArrayList(Stage) = .empty,

    /// The routed experts of the layer under test and the picks a token takes.
    pub fn experts(p: *const Probe) u32 {
        return if (p.moe) |m| m[0] else p.r.c.experts;
    }

    pub fn topk(p: *const Probe) u32 {
        return if (p.moe) |m| m[1] else p.r.c.topk;
    }

    pub fn deinit(p: *Probe) void {
        p.stages.deinit(p.a);
    }

    /// A new stage; its pointer stays valid because the list never grows past its first allocation.
    pub fn stage(p: *Probe, name: []const u8) !*Stage {
        if (p.stages.capacity == 0) try p.stages.ensureTotalCapacity(p.a, 128);
        if (p.stages.items.len == p.stages.capacity) return error.TooManyStages;
        p.stages.appendAssumeCapacity(.{ .name = name });
        return &p.stages.items[p.stages.items.len - 1];
    }

    /// Prints every stage; returns the failures.
    pub fn report(p: *const Probe) usize {
        var bad: usize = 0;
        for (p.stages.items) |*s| {
            s.print(p.at orelse p.w.layer);
            bad += s.bad;
        }
        return bad;
    }

    pub fn bytes(_: *const Probe, r: Ref, n: usize) []const u8 {
        return (r.buf.contents() + r.off)[0..n];
    }

    pub fn bfs(p: *const Probe, r: Ref, n: usize) []const u16 {
        return @alignCast(std.mem.bytesAsSlice(u16, p.bytes(r, 2 * n)));
    }

    pub fn f32s(p: *const Probe, r: Ref, n: usize) []const f32 {
        return @alignCast(std.mem.bytesAsSlice(f32, p.bytes(r, 4 * n)));
    }

    pub fn i32s(p: *const Probe, r: Ref, n: usize) []const i32 {
        return @alignCast(std.mem.bytesAsSlice(i32, p.bytes(r, 4 * n)));
    }

    pub fn meta(p: *const Probe, at: usize) []const i32 {
        return p.i32s(p.r.s.metaAt(at), pkg.pipelines.max_rows);
    }

    pub fn pos(p: *const Probe, row: usize) u32 {
        return p.p0 + @as(u32, @intCast(row));
    }

    /// Dequantised FP8 weight [j][i] of a projection.
    pub fn fp8W(p: *const Probe, f: Fp8, j: usize, i: usize) f64 {
        const code = p.bytes(f.w, f.n * f.k)[j * f.k + i];
        const sc = p.bytes(f.s, (f.n / 32) * (f.k / 32))[(j / 32) * (f.k / 32) + i / 32];
        return @as(f64, ref.e4m3(code)) * ref.e8m0(sc);
    }

    /// y [rows][n] = x [rows][k] (bf16) by an FP8 projection: fp32 chains of 32 and a tree, then bf16 or fp32 out.
    pub fn proj(p: *Probe, name: []const u8, x: Ref, f: Fp8, y: Ref, f32_out: bool, rows: []const bool) !void {
        const s = try p.stage(name);
        const xs = p.bfs(x, p.rows * f.k);
        const depth: f64 = 32 + @as(f64, @floatFromInt(std.math.log2_int_ceil(usize, f.k / 32 + 1))) + 8;
        for (0..p.rows) |r| {
            if (!rows[r]) continue;
            for (0..f.n) |j| {
                var want: f64 = 0;
                var mag: f64 = 0;
                for (0..f.k) |i| {
                    const t = @as(f64, ref.bf(xs[r * f.k + i])) * p.fp8W(f, j, i);
                    want += t;
                    mag += @abs(t);
                }
                if (f32_out) {
                    s.add(p.f32s(y, p.rows * f.n)[r * f.n + j], want, ref.fBound(want, mag, depth), r, j);
                } else {
                    s.add(ref.bf(p.bfs(y, p.rows * f.n)[r * f.n + j]), want, ref.bfBound(want, mag, depth), r, j);
                }
            }
        }
    }

    /// y [rows][n] = x [rows][k] by a bf16 weight [n][k] (lane chains of k / 32, then the butterfly).
    pub fn mv(p: *Probe, name: []const u8, x: Ref, w: Ref, y: Ref, k: usize, n: usize, f32_out: bool, rows: []const bool) !void {
        const s = try p.stage(name);
        const xs = p.bfs(x, p.rows * k);
        const ws = p.bfs(w, n * k);
        const depth: f64 = @as(f64, @floatFromInt(k / 32)) + 6;
        for (0..p.rows) |r| for (0..n) |j| {
            if (!rows[r]) continue;
            var want: f64 = 0;
            var mag: f64 = 0;
            for (0..k) |i| {
                const t = @as(f64, ref.bf(xs[r * k + i])) * ref.bf(ws[j * k + i]);
                want += t;
                mag += @abs(t);
            }
            if (f32_out) s.add(p.f32s(y, p.rows * n)[r * n + j], want, ref.fBound(want, mag, depth), r, j) else s.add(ref.bf(p.bfs(y, p.rows * n)[r * n + j]), want, ref.bfBound(want, mag, depth), r, j);
        };
    }

    /// RMSNorm rows: w * x / sqrt(mean(x^2) + eps), bf16 out.
    pub fn norm(p: *Probe, name: []const u8, x: Ref, w: Ref, y: Ref, dim: usize, rows: []const bool) !void {
        const s = try p.stage(name);
        const xs = p.bfs(x, p.rows * dim);
        const ws = p.f32s(w, dim);
        const ys = p.bfs(y, p.rows * dim);
        const depth = @as(f64, @floatFromInt(dim / 256 + 16)) / 2 + 6;
        for (0..p.rows) |r| {
            if (!rows[r]) continue;
            var ss: f64 = 0;
            for (0..dim) |i| ss += @as(f64, ref.bf(xs[r * dim + i])) * ref.bf(xs[r * dim + i]);
            const inv = 1 / @sqrt(ss / @as(f64, @floatFromInt(dim)) + p.r.c.eps);
            for (0..dim) |i| {
                const want = @as(f64, ws[i]) * ref.bf(xs[r * dim + i]) * inv;
                s.add(ref.bf(ys[r * dim + i]), want, ref.bfBound(want, @abs(want), depth), r, i);
            }
        }
    }

    /// act_quant's round trip, value (a, b, k) read at a*sa + b*sb + k and written at a*ta + b*tb + k: bit-equal.
    pub fn actq(p: *Probe, name: []const u8, x: Ref, y: Ref, st: pkg.glue.Strides) !void {
        const s = try p.stage(name);
        const n = (st.na - 1) * @max(st.sa, st.ta) + st.nb * @max(st.sb, st.tb) + st.k;
        const xs = p.bfs(x, n);
        const ys = p.bfs(y, n);
        var out: [32]u16 = undefined;
        for (0..st.na) |a| for (0..st.nb) |b| for (0..st.k / 32) |blk| {
            const i = a * st.sa + b * st.sb + 32 * blk;
            const o = a * st.ta + b * st.tb + 32 * blk;
            ref.actq32(xs[i..][0..32], &out);
            s.same(std.mem.eql(u16, &out, ys[o..][0..32]), b, 32 * blk);
        };
    }

    pub fn actqRows(p: *Probe, name: []const u8, x: Ref, y: Ref, k: usize) !void {
        const kk: u32 = @intCast(k);
        try p.actq(name, x, y, .{ .k = kk, .nb = @intCast(p.rows), .sb = kk, .tb = kk });
    }

    /// RoPE: each head's dims before the last 64 copied, the last 64 rotated by the f32 table at the row's position.
    pub fn rope(p: *Probe, name: []const u8, x: Ref, y: Ref, at: []const i32, layer: u32, heads: usize, head: usize) !void {
        const s = try p.stage(name);
        const n = p.rows * heads * head;
        const xs = p.bfs(x, n);
        const ys = p.bfs(y, n);
        const kind: pkg.rope.Kind = if (p.r.c.compressRatio(layer) > 0) .compressed else .window;
        const tab = p.r.tables.of(kind).slice(f32, p.r.tables.positions * 64);
        for (0..p.rows) |r| {
            if (at[r] < 0) continue;
            const t = tab[@as(usize, @intCast(at[r])) * 64 ..][0..64];
            for (0..heads) |h| for (0..head / 2) |pr| {
                const i = (r * heads + h) * head + 2 * pr;
                const x0: f64 = ref.bf(xs[i]);
                const x1: f64 = ref.bf(xs[i + 1]);
                if (2 * pr < head - 64) {
                    s.same(xs[i] == ys[i] and xs[i + 1] == ys[i + 1], r, i);
                    continue;
                }
                const q = pr - (head - 64) / 2;
                const c: f64 = t[q];
                const sn: f64 = t[32 + q];
                const mag = @abs(x0 * c) + @abs(x1 * sn);
                const w0 = x0 * c - x1 * sn;
                const w1 = x0 * sn + x1 * c;
                s.add(ref.bf(ys[i]), w0, ref.bfBound(w0, mag, 2), r, i);
                s.add(ref.bf(ys[i + 1]), w1, ref.bfBound(w1, mag, 2), r, i + 1);
            };
        }
    }
};

/// Every row of a launch.
pub fn all(rows: usize) []const bool {
    return every[0..rows];
}

const every: [16]bool = @splat(true);

/// The rows whose record at `at` is set (>= 0).
pub fn setRows(at: []const i32, rows: usize, out: *[16]bool) []const bool {
    for (0..rows) |r| out[r] = at[r] >= 0;
    return out[0..rows];
}
