//! Nemotron's routed-expert pipelines under the contract: which ones, their specs, served windows and source controls.
const std = @import("std");
const c = @import("contract").spec;

pub const Proj = enum { up, down };

/// Outputs a pair and the inputs each output sums: up 2688 -> 1856 (relu2), down 1856 -> 2688.
pub fn dims(p: Proj) [2]usize {
    return if (p == .up) .{ 1856, 2688 } else .{ 2688, 1856 };
}

/// The engine's (rows a simdgroup, simdgroups) geometries, as kernels.geometries lists them.
pub const geometries = [_][2]usize{ .{ 2, 2 }, .{ 8, 2 }, .{ 4, 4 }, .{ 8, 4 }, .{ 4, 1 }, .{ 2, 4 } };

/// The one control fault a source carries: member rows read one row, or tf_rowdot_w's x sums without bf16 steps.
pub const Fault = enum { none, coupled, fp32_sums };

/// One registration: a pipeline as the engine names it, its rows a simdgroup, simdgroups and member rows a pass.
pub const Entry = struct {
    proj: Proj,
    symbol: [:0]const u8,
    rps: usize,
    sg: usize,
    mb: usize,
    geo: bool = false,
    fault: Fault = .none,

    /// tf_xup_w3 and tf_xdown_w3: the window path every other pipeline's bytes must equal.
    pub fn base(v: Entry) bool {
        return std.mem.endsWith(u8, v.symbol, "_w3");
    }

    pub fn baseSymbol(v: Entry) [:0]const u8 {
        return if (v.proj == .up) "tf_xup_w3" else "tf_xdown_w3";
    }

    pub fn threads(v: Entry) usize {
        return 32 * v.sg;
    }
};

fn rowsEntry(p: Proj, comptime mb: usize, comptime rps: usize, fault: Fault) Entry {
    const which = if (p == .up) "up" else "down";
    const r8 = if (rps == 8) "r8" else "";
    const symbol = if (mb == 3) "tf_x" ++ which ++ "_w3" else std.fmt.comptimePrint("tf_x{s}_rows{d}{s}", .{ which, mb, r8 });
    return .{ .proj = p, .symbol = symbol, .rps = rps, .sg = if (mb == 3) 4 else 2, .mb = mb, .fault = fault };
}

/// The twenty pipelines, then the two controls: rows2 up with coupled members, w3 down summing x in fp32.
pub const entries = blk: {
    var out: [22]Entry = undefined;
    var n: usize = 0;
    for ([_]Proj{ .up, .down }) |p| {
        for ([_][2]usize{ .{ 3, 4 }, .{ 2, 4 }, .{ 4, 4 }, .{ 2, 8 } }) |mr| {
            out[n] = rowsEntry(p, mr[0], mr[1], .none);
            n += 1;
        }
        for (geometries) |geo| {
            const which = if (p == .up) "up" else "down";
            out[n] = .{ .proj = p, .symbol = std.fmt.comptimePrint("tf_x{s}_{d}_{d}", .{ which, geo[0], geo[1] }), .rps = geo[0], .sg = geo[1], .mb = 1, .geo = true };
            n += 1;
        }
    }
    out[n] = rowsEntry(.up, 2, 4, .coupled);
    out[n + 1] = rowsEntry(.down, 3, 4, .fp32_sums);
    break :blk out;
};

pub fn name(a: std.mem.Allocator, v: Entry) ![]const u8 {
    const kind = if (v.fault == .coupled) "control coupled members" else if (v.fault == .fp32_sums) "control fp32 x sums" else if (v.base()) "oracle" else "replacement";
    return std.fmt.allocPrint(a, "metal nemotron experts {t} {s} {s}", .{ v.proj, kind, v.symbol });
}

pub fn visit(visitor: anytype, only: ?[]const u8) !void {
    for (entries) |v| {
        const text = try name(visitor.a, v);
        defer visitor.a.free(text);
        if (only) |s| if (std.mem.indexOf(u8, text, s) == null) continue;
        try visitor.add(v, text);
    }
}

/// Window rows the engine runs: one row (route's tables; the geometries' use), then windows up to state.max_rows.
pub fn served(v: Entry) [7]c.Shape {
    const nk = dims(v.proj);
    const rows = if (v.geo) [_]usize{ 1, 2, 3, 4, 8, 16, 128 } else [_]usize{ 4, 2, 3, 5, 8, 16, 128 };
    var out: [7]c.Shape = undefined;
    for (&out, rows) |*s, r| s.* = .{ .rows = r, .n = nk[0], .k = nk[1] };
    return out;
}

/// A lane's 16-input pieces (two roundings a group of four, four groups), its 512-input chunks, then simd_sum.
pub fn depth(s: c.Shape) usize {
    return 16 + 3 * ((s.k + 511) / 512) + 8;
}

/// relu2 is metal::max(bfloat(acc), 0.0f), which returns 0 for a NaN dot; every up pipeline gives w3's bytes there.
pub const relu_nan = c.Known{ .gate = .nonfinite, .finding = "relu2's metal::max(bfloat(acc), 0) gives 0 for a NaN dot where float64 keeps the NaN" };

pub fn spec(v: Entry, text: []const u8, shapes: []const c.Shape) c.Spec {
    const replacement = !v.base();
    return .{
        .name = text,
        .accum = .f32,
        .depth = depth,
        .out = .bf16,
        .replaces = if (replacement) v.baseSymbol() else null,
        .nan = if (replacement and v.proj == .up) .{ .stated = "a nonfinite input gives tf_xup_w3's bytes" } else .propagate,
        .tiles = .{ .{ .size = v.mb }, .{}, .{} },
        .served = shapes,
        .known = if (v.proj == .up and v.base()) relu_nan else null,
    };
}

const coupled_from = "sum[b] = tf_load16(x0 + xoff[b] + xl + k0, xt[b]);";
const coupled_to = "sum[b] = tf_load16(x0 + xoff[0] + xl + k0, xt[b]);";
const sums_from = "sum[b] += float(bfloat(float(bfloat(float(bfloat(float(a) + float(bb))) + float(c))) + float(d)));";
const sums_to = "sum[b] += float(a) + float(bb) + float(c) + float(d);";

/// The source an entry's pipeline compiles from: the engine's own, or a control's one planted change.
pub fn source(a: std.mem.Allocator, original: []const u8, fault: Fault) ![]u8 {
    const from = switch (fault) {
        .none => return a.dupe(u8, original),
        .coupled => coupled_from,
        .fp32_sums => sums_from,
    };
    if (std.mem.count(u8, original, from) != 1) return error.SourceWitnessMissing;
    return std.mem.replaceOwned(u8, a, original, from, if (fault == .coupled) coupled_to else sums_to);
}
