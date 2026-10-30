//! Complete checkpoint storage owners. Dense matrices use [N,K]; trellises use [K/16,N/16,words].
const std = @import("std");
const pk = @import("pack.zig");
const exl3 = @import("core").exl3_layout;
pub const Scale = struct { format: enum { f16, packed_signs }, input: pk.Info, output: pk.Info };
pub const Projection = union(enum) {
    dense: struct { weight: pk.Info, scales: Scale, k: usize, n: usize },
    exl3: struct { trellis: pk.Info, scales: Scale, geometry: exl3.Group, marker: ?pk.Info },
    /// The release's dense FP8: E4M3 [N,K] with one E8M0 scale per 32x32 block.
    fp8_block: struct { weight: pk.Info, scale: pk.Info, k: usize, n: usize },
    /// The release's routed experts: I8 [N,K/2] holding two E2M1 codes a byte, one E8M0 scale per 32 of K.
    mxfp4: struct { weight: pk.Info, scale: pk.Info, k: usize, n: usize },
    /// The release's Engram table: E4M3 [rows,256] with one E8M0 scale per row and 32 columns.
    fp8_rows: struct { weight: pk.Info, scale: pk.Info, k: usize, n: usize },
    /// CUDA's current rank image executes only trellises with F16 scales. Refuse after shared admission.
    pub fn cudaGroup(p: Projection) !exl3.Group {
        return switch (p) {
            .dense => error.UnsupportedDenseProjection,
            .exl3 => |g| if (g.scales.format == .f16) g.geometry else error.UnsupportedPackedScales,
            .fp8_block, .mxfp4, .fp8_rows => error.UnsupportedProjection,
        };
    }
};
fn scales(p: *const pk.Pack, prefix: []const u8, k: usize, n: usize, allow_signs: bool) !Scale {
    var buf: [256]u8 = undefined;
    const sh = p.get(try std.fmt.bufPrint(&buf, "{s}.suh", .{prefix}));
    const vh = p.get(try std.fmt.bufPrint(&buf, "{s}.svh", .{prefix}));
    const su = p.get(try std.fmt.bufPrint(&buf, "{s}.su", .{prefix}));
    const sv = p.get(try std.fmt.bufPrint(&buf, "{s}.sv", .{prefix}));
    if ((sh != null and su != null) or (vh != null and sv != null)) return error.AmbiguousScales;
    if (sh != null and vh != null) {
        if (!sh.?.is(.f16, &.{k}) or !vh.?.is(.f16, &.{n})) return error.BadScales;
        return .{ .format = .f16, .input = sh.?, .output = vh.? };
    }
    if (allow_signs and su != null and sv != null) {
        if (k % 16 != 0 or n % 16 != 0 or !su.?.is(.i16, &.{k / 16}) or !sv.?.is(.i16, &.{n / 16})) return error.BadScales;
        return .{ .format = .packed_signs, .input = su.?, .output = sv.? };
    }
    return error.MissingTensor;
}
/// The release's E8M0 scales of `{prefix}` when they have exactly the wanted shape; null when they do not.
fn releaseScale(p: *const pk.Pack, prefix: []const u8, want: []const usize) !?pk.Info {
    var buf: [256]u8 = undefined;
    const s = (try p.scale(prefix, &buf)) orelse return error.MissingTensor;
    if (!s.is(.f8_e8m0, want)) return null;
    return s;
}
/// The release's stored weights: E4M3 with a 32x32 block scale, or I8 holding two E2M1 codes a byte.
fn release(p: *const pk.Pack, prefix: []const u8, w: pk.Info) !Projection {
    if (w.rank != 2 or w.shape[0] == 0 or w.shape[1] == 0) return error.UnsupportedProjection;
    const n = w.shape[0];
    const k = w.shape[1];
    switch (w.dtype) {
        .f8_e4m3 => {
            const blocked = &.{ (n + 31) / 32, (k + 31) / 32 };
            const rows = &.{ n, (k + 31) / 32 };
            if (try releaseScale(p, prefix, blocked)) |s| return .{ .fp8_block = .{ .weight = w, .scale = s, .k = k, .n = n } };
            if (try releaseScale(p, prefix, rows)) |s| return .{ .fp8_rows = .{ .weight = w, .scale = s, .k = k, .n = n } };
            return error.BadScales;
        },
        .i8 => {
            // two E2M1 codes a byte: the stored width is half the logical K and the scales run one per 32 of it
            if (k % 2 != 0) return error.BadScales;
            const s = (try releaseScale(p, prefix, &.{ n, k / 16 })) orelse return error.BadScales;
            return .{ .mxfp4 = .{ .weight = w, .scale = s, .k = k * 2, .n = n } };
        },
        else => return error.UnsupportedProjection,
    }
}
/// Kept dense weights take precedence, matching the Metal matrixAny contract. They still require rotation scales.
pub fn read(p: *const pk.Pack, prefix: []const u8) !Projection {
    var buf: [256]u8 = undefined;
    if (p.get(try std.fmt.bufPrint(&buf, "{s}.weight", .{prefix}))) |w| {
        if (w.dtype == .f8_e4m3 or w.dtype == .i8) return release(p, prefix, w);
        if (w.dtype != .bf16 or w.rank != 2 or w.shape[0] == 0 or w.shape[1] == 0) return error.UnsupportedProjection;
        return .{ .dense = .{ .weight = w, .k = w.shape[1], .n = w.shape[0], .scales = try scales(p, prefix, w.shape[1], w.shape[0], false) } };
    }
    const trellis = try p.need(try std.fmt.bufPrint(&buf, "{s}.trellis", .{prefix}));
    if (trellis.dtype != .i16) return error.BadTrellis;
    const g = try exl3.Group.fromShape(trellis.dims());
    const scale = try scales(p, prefix, g.k(), g.n(), true);
    const marker_name = try p.marker(prefix, &buf);
    const marker = if (marker_name) |name| p.get(name) else null;
    if (g.k2() % 2 != 0 and (marker_name == null or !std.mem.endsWith(u8, marker_name.?, ".mul1"))) return error.BadMarker;
    return .{ .exl3 = .{ .trellis = trellis, .geometry = g, .scales = scale, .marker = marker } };
}
