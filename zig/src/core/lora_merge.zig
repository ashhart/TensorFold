//! A PEFT LoRA folded into a projection, onto its shipped grid or into dense weights quantized fresh; FP64 error.
const std = @import("std");
const host4 = @import("affine4_host.zig");
const lq = @import("lora_quant.zig");

pub const Quant = lq.Quant;
pub const View = lq.View;
pub const viewOf = lq.viewOf;
pub const allocQuant = lq.allocQuant;
pub const freeQuant = lq.freeQuant;
pub const quantize = lq.quantize;
pub const Agreement = lq.Agreement;
pub const agreement = lq.agreement;
pub const differingGroups = lq.differingGroups;
pub const toF32 = lq.toF32;

pub const Error = error{ BadShape, BadGroup, NonFinite };

pub const Side = enum { a, b };

/// A PEFT tensor name split into the checkpoint's module key and the LoRA side it holds.
pub const AdapterKey = struct { module: []const u8, side: Side };

/// `base_model.model.<module>.lora_{A,B}[.default].weight`, or null when the name is anything else.
pub fn parseAdapterKey(name: []const u8) ?AdapterKey {
    const head = "base_model.model.";
    if (!std.mem.startsWith(u8, name, head)) return null;
    const rest = name[head.len..];
    const ends = [_]struct { []const u8, Side }{
        .{ ".lora_A.weight", .a },         .{ ".lora_B.weight", .b },
        .{ ".lora_A.default.weight", .a }, .{ ".lora_B.default.weight", .b },
    };
    for (ends) |e| if (std.mem.endsWith(u8, rest, e[0])) return .{ .module = rest[0 .. rest.len - e[0].len], .side = e[1] };
    return null;
}

/// The factors: `a` is r x k, `b` is n x r, and `scale` is alpha / r.
pub const Lora = struct { a: []const f32, b: []const f32, r: usize, scale: f64 };

fn checkShape(n: usize, k: usize, group: usize, l: Lora) Error!void {
    if (group != 32 and group != 64 and group != 128) return error.BadGroup;
    const ak = std.math.mul(usize, l.r, k) catch return error.BadShape;
    const bn = std.math.mul(usize, n, l.r) catch return error.BadShape;
    _ = std.math.mul(usize, n, k) catch return error.BadShape;
    if (n == 0 or k == 0 or k % group != 0 or l.a.len != ak or l.b.len != bn) return error.BadShape;
}

/// Row `o` of `scale * (B * A)`: FP64 products accumulated in order, then scaled once; a zero sum stays zero.
pub fn deltaRow(l: Lora, k: usize, o: usize, out: []f64) void {
    @memset(out[0..k], 0);
    for (0..l.r) |j| {
        const c: f64 = l.b[o * l.r + j];
        if (c == 0) continue;
        for (out[0..k], l.a[j * k ..][0..k]) |*d, a| d.* += c * a;
    }
    for (out[0..k]) |*d| d.* *= l.scale;
}

fn dequant64(g: usize, v: View, i: usize) f64 {
    const q: u32 = (v.words[i / 8] >> @intCast(4 * (i % 8))) & 0xf;
    return @as(f64, host4.f32of(v.scales[i / g])) * @as(f64, @floatFromInt(q)) + @as(f64, host4.f32of(v.biases[i / g]));
}

/// `dense + scale * B * A` in FP64, rounded to f32, quantized fresh into `out`; `bad` is the group NonFinite names.
pub fn mergeDense(gpa: std.mem.Allocator, group: usize, n: usize, k: usize, dense: []const f32, l: Lora, out: Quant, bad: *usize) !void {
    try checkShape(n, k, group, l);
    if (dense.len != n * k) return error.BadShape;
    const delta = try gpa.alloc(f64, k);
    defer gpa.free(delta);
    const row = try gpa.alloc(f32, k);
    defer gpa.free(row);
    for (0..n) |o| {
        deltaRow(l, k, o, delta);
        for (row, dense[o * k ..][0..k], delta) |*r, d, x| r.* = @floatCast(@as(f64, d) + x);
        const q: Quant = .{ .words = out.words[o * k / 8 ..][0 .. k / 8], .scales = out.scales[o * k / group ..][0 .. k / group], .biases = out.biases[o * k / group ..][0 .. k / group] };
        if (try quantize(group, row, q)) |g| {
            bad.* = o * (k / group) + g;
            return error.NonFinite;
        }
    }
}

/// The groups of `merged` that differ from `dense + scale * B * A` computed again here, by the merge's own arithmetic.
pub fn recompute(gpa: std.mem.Allocator, group: usize, n: usize, k: usize, dense: []const f32, l: Lora, merged: View, bad: *usize) !usize {
    const fresh = try allocQuant(gpa, n, k, group);
    defer freeQuant(gpa, fresh);
    try mergeDense(gpa, group, n, k, dense, l, fresh, bad);
    return differingGroups(group, viewOf(fresh), merged);
}

fn checkShipped(n: usize, k: usize, group: usize, v: View) Error!void {
    if (v.words.len != n * k / 8 or v.scales.len != n * k / group or v.biases.len != v.scales.len) return error.BadShape;
}

/// The nearest code to `t` on the grid `s * q + b`, q from 0 to 15, whichever sign the scale has.
fn nearest(t: f64, s: f64, b: f64) u32 {
    return @intFromFloat(std.math.clamp(@round((t - b) / s), 0, 15));
}

fn codeOf(words: []const u32, i: usize) u32 {
    return (words[i / 8] >> @intCast(4 * (i % 8))) & 0xf;
}

/// `shipped + scale * B * A` on each group's own grid: nearest codes, the shipped tables; `bad` as in mergeDense.
pub fn foldShipped(gpa: std.mem.Allocator, group: usize, n: usize, k: usize, shipped: View, l: Lora, out: Quant, bad: *usize) !void {
    try checkShape(n, k, group, l);
    try checkShipped(n, k, group, shipped);
    try checkShipped(n, k, group, viewOf(out));
    @memcpy(out.words, shipped.words);
    @memcpy(out.scales, shipped.scales);
    @memcpy(out.biases, shipped.biases);
    const delta = try gpa.alloc(f64, k);
    defer gpa.free(delta);
    for (0..n) |o| {
        deltaRow(l, k, o, delta);
        for (delta, 0..) |d, c| {
            if (d == 0) continue; // an unchanged value keeps its shipped code
            const i = o * k + c;
            const s: f64 = host4.f32of(shipped.scales[i / group]);
            const b: f64 = host4.f32of(shipped.biases[i / group]);
            const t = dequant64(group, shipped, i) + d;
            if (!std.math.isFinite(t) or !std.math.isFinite(s) or !std.math.isFinite(b)) {
                bad.* = i / group;
                return error.NonFinite;
            }
            if (s == 0) continue; // every code of a zero-scale group decodes to its bias
            const shift: u5 = @intCast(4 * (i % 8));
            out.words[i / 8] = (out.words[i / 8] & ~(@as(u32, 0xf) << shift)) | (nearest(t, s, b) << shift);
        }
    }
}

/// The groups of `merged` that differ from `shipped + scale * B * A` folded again here, by the fold's own arithmetic.
pub fn refold(gpa: std.mem.Allocator, group: usize, n: usize, k: usize, shipped: View, l: Lora, merged: View, bad: *usize) !usize {
    const again = try allocQuant(gpa, n, k, group);
    defer freeQuant(gpa, again);
    try foldShipped(gpa, group, n, k, shipped, l, again, bad);
    return differingGroups(group, viewOf(again), merged);
}

/// Codes of `merged` off the nearest step to `shipped + scale * B * A`, by distance, or in a group whose table moved.
pub fn offGrid(gpa: std.mem.Allocator, group: usize, n: usize, k: usize, shipped: View, l: Lora, merged: View) !usize {
    try checkShape(n, k, group, l);
    try checkShipped(n, k, group, shipped);
    try checkShipped(n, k, group, merged);
    var far: usize = 0;
    for (shipped.scales, merged.scales, shipped.biases, merged.biases) |s0, s1, b0, b1| {
        if (s0 != s1 or b0 != b1) far += group;
    }
    const delta = try gpa.alloc(f64, k);
    defer gpa.free(delta);
    for (0..n) |o| {
        deltaRow(l, k, o, delta);
        for (delta, 0..) |d, c| {
            const i = o * k + c;
            const s: f64 = host4.f32of(shipped.scales[i / group]);
            const q: f64 = @floatFromInt(codeOf(merged.words, i));
            if (d == 0 or s == 0) {
                far += @intFromBool(codeOf(merged.words, i) != codeOf(shipped.words, i));
                continue;
            }
            // the code's place on the grid: within half a step of the target, or at the end the target lies past
            const b: f64 = host4.f32of(shipped.biases[i / group]);
            const u = (dequant64(group, shipped, i) + d - b) / s;
            far += @intFromBool(!(@abs(q - u) <= 0.5 + 1e-9 or (q == 0 and u < 0) or (q == 15 and u > 15)));
        }
    }
    return far;
}

/// The error of a quantization against an FP64 reference, per column (worst over rows) and in aggregate.
pub const Error64 = struct {
    worst_col: usize,
    worst_col_err: f64, // the largest |quantized - reference| in that column
    worst_col_rms: f64,
    rms_err: f64,
    max_err: f64,

    fn finite(e: Error64) bool {
        return std.math.isFinite(e.worst_col_err) and std.math.isFinite(e.worst_col_rms) and std.math.isFinite(e.rms_err) and std.math.isFinite(e.max_err);
    }
};

/// `num / den` as a ratio of errors: two exact zeros are equal (1), a positive error over zero is infinite.
fn ratio(num: f64, den: f64) f64 {
    return if (num == 0 and den == 0) 1 else num / den;
}

/// A sanity net, not a quality gate: only a merged RMS error over twice the shipped one (a wrong scale, a sign) fails.
pub const rms_sanity_bound = 2.0;

/// A merged tensor's error against `dense + scale * B * A`, beside the shipped quantization's against `dense`.
pub const Report = struct {
    merged: Error64,
    shipped: Error64,

    /// The merged aggregate RMS error over the shipped one: above 1 where the delta widens the groups' ranges.
    pub fn rmsRatio(r: Report) f64 {
        return ratio(r.merged.rms_err, r.shipped.rms_err);
    }

    /// The merged worst-column error over the shipped one: printed, not gated: a chance extreme.
    pub fn worstRatio(r: Report) f64 {
        return ratio(r.merged.worst_col_err, r.shipped.worst_col_err);
    }

    /// Whether the aggregate RMS errors are finite and the merged one is within `rms_sanity_bound` of the shipped one.
    pub fn withinSanity(r: Report) bool {
        if (!r.merged.finite() or !r.shipped.finite()) return false;
        return r.rmsRatio() <= rms_sanity_bound;
    }
};

/// `q` against `dense + scale * B * A` (a null `l` is `dense` alone).
fn errorOf(gpa: std.mem.Allocator, group: usize, n: usize, k: usize, dense: []const f32, l: ?Lora, q: View) !Error64 {
    const delta = try gpa.alloc(f64, k);
    defer gpa.free(delta);
    const cols = try gpa.alloc(f64, try std.math.mul(usize, 2, k)); // per column: worst error, sum of squares
    defer gpa.free(cols);
    @memset(cols, 0);
    @memset(delta, 0);
    var all_sq: f64 = 0;
    var all_max: f64 = 0;
    for (0..n) |o| {
        if (l) |x| deltaRow(x, k, o, delta);
        for (0..k) |c| {
            const i = o * k + c;
            const e = @abs(dequant64(group, q, i) - (@as(f64, dense[i]) + delta[c]));
            const col = cols[2 * c ..][0..2];
            col[0] = @max(col[0], e);
            col[1] += e * e;
            all_sq += e * e;
            all_max = @max(all_max, e);
        }
    }
    var r: Error64 = .{ .worst_col = 0, .worst_col_err = 0, .worst_col_rms = 0, .rms_err = @sqrt(all_sq / @as(f64, @floatFromInt(n * k))), .max_err = all_max };
    for (0..k) |c| {
        const col = cols[2 * c ..][0..2];
        if (col[0] >= r.worst_col_err) {
            r.worst_col = c;
            r.worst_col_err = col[0];
            r.worst_col_rms = @sqrt(col[1] / @as(f64, @floatFromInt(n)));
        }
    }
    return r;
}

pub fn measure(gpa: std.mem.Allocator, group: usize, n: usize, k: usize, dense: []const f32, l: Lora, shipped: View, merged: View) !Report {
    try checkShape(n, k, group, l);
    return .{ .merged = try errorOf(gpa, group, n, k, dense, l, merged), .shipped = try errorOf(gpa, group, n, k, dense, null, shipped) };
}

test {
    _ = @import("lora_merge_test.zig");
}
