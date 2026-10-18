//! TF_DSV41_LIN2 (lin2.cu): the dense linears' one-pass 17-32-row variants, their symbols, mode readers and functions.

const std = @import("std");
const cuda = @import("root.zig");
const dense = @import("exl3_block_dense.zig");

const LinTable = dense.LinTable;
const dense3_k2s = dense.dense3_k2s;
const linear_k2s = dense.linear_k2s;

// TF_DSV41_LIN2 (ours, lin2.cu): the 17-32-row programs of the dense linears in one pass over the weights
fn lanes2Symbol(comptime k2: u32, comptime wk: u32, comptime mb: u32) [:0]const u8 {
    return std.fmt.comptimePrint("_ZN10dsv41_lin213lanes2_kernelILi{d}ELi2ELi{d}ELi{d}EEEvN11dsv41_x3seg8LinTableEii", .{ k2, wk, mb });
}
fn lanescSymbol(comptime k2: u32, comptime wk: u32, comptime mb: u32, comptime t: u32) [:0]const u8 {
    return std.fmt.comptimePrint("_ZN10dsv41_lin213lanesc_kernelILi{d}ELi2ELi{d}ELi{d}ELi{d}EEEvN11dsv41_x3seg8LinTableEii", .{ k2, wk, mb, t });
}
/// lanesc's instances in lin2.cu's order (WK, MINB, T): a 4-warp and an 8-warp CTA at T 1 and at T 2
pub const lanesc_cfgs = [_][3]u32{ .{ 4, 3, 1 }, .{ 8, 1, 1 }, .{ 4, 2, 2 }, .{ 8, 1, 2 } };
fn linear2Symbol(comptime k2: u32, comptime wk: u32, comptime pf: bool) [:0]const u8 {
    return std.fmt.comptimePrint("_ZN10dsv41_lin214linear2_kernelILi{d}ELi2ELi{d}ELb{d}EEEvPK6__halfPKjxxS3_S3_PviPfPiiiiiPKi", .{ k2, wk, @intFromBool(pf) });
}

/// TF_DSV41_LIN2[_PF|_WK|_CLUSTER]: lin2.cu's one-pass 17-32-row variants of the dense linears; bad values refused.
pub const Lin2Mode = struct { lanes: bool = false, linear: bool = false, pf: bool = true, wk4: bool = true, wk8: bool = true, cluster: bool = false };

/// TF_DSV41_LIN2_CLUSTER's value: unset / empty / 0 off, 1 on, anything else refused.
pub fn lin2Cluster(v: ?[]const u8) !bool {
    const t = std.mem.trim(u8, v orelse "", " \t\r\n");
    if (t.len == 0 or std.mem.eql(u8, t, "0")) return false;
    if (std.mem.eql(u8, t, "1")) return true;
    return error.BadLin2;
}

/// The split count (2-8) shared by every segment of a dense3 launch of at most 32 rows, else null.
pub fn clusterSplits(tab: LinTable, m: usize) ?u32 {
    if (m < 1 or m > 32 or tab.n < 1) return null;
    const sk = tab.s[0].SK;
    if (sk < 2 or sk > 8) return null;
    for (tab.s[0..@intCast(tab.n)]) |x| if (x.SK != sk or @mod(x.p0, sk) != 0) return null;
    return @intCast(sk);
}

pub fn lin2Mode(v: ?[]const u8, pf: ?[]const u8) !Lin2Mode {
    return lin2ModeWk(v, pf, null);
}

pub fn lin2ModeWk(v: ?[]const u8, pf: ?[]const u8, wk: ?[]const u8) !Lin2Mode {
    var m: Lin2Mode = .{};
    const t = std.mem.trim(u8, v orelse "", " \t\r\n");
    if (t.len == 0 or std.mem.eql(u8, t, "0") or std.ascii.eqlIgnoreCase(t, "off")) {
        m = .{};
    } else if (std.mem.eql(u8, t, "1") or std.ascii.eqlIgnoreCase(t, "on")) {
        m = .{ .lanes = true, .linear = true };
    } else if (std.ascii.eqlIgnoreCase(t, "lanes")) {
        m = .{ .lanes = true };
    } else if (std.ascii.eqlIgnoreCase(t, "linear")) {
        m = .{ .linear = true };
    } else return error.BadLin2;
    const p = std.mem.trim(u8, pf orelse "", " \t\r\n");
    if (p.len == 0 or std.mem.eql(u8, p, "1")) {
        m.pf = true;
    } else if (std.mem.eql(u8, p, "0")) {
        m.pf = false;
    } else return error.BadLin2;
    const w = std.mem.trim(u8, wk orelse "", " \t\r\n");
    if (w.len == 0) {
        m.wk4 = true;
        m.wk8 = true;
    } else if (std.mem.eql(u8, w, "4")) {
        m.wk8 = false;
    } else if (std.mem.eql(u8, w, "8")) {
        m.wk4 = false;
    } else return error.BadLin2;
    return m;
}

/// A launch of `m` rows the two-tile variant takes: the original's two passes (rows 17-32).
pub fn twoTile(m: usize) bool {
    return m > 16 and m <= 32;
}

/// lin2.cu's instances: lanes2 at dense3's widths (4- and 8-warp CTAs); linear2 with or without k-step prefetch.
pub const Lin2 = struct {
    module: cuda.Module,
    mode: Lin2Mode,
    lanes: [dense3_k2s.len][2]cuda.Function, // WK 4, 8
    linear: [linear_k2s.len][3]cuda.Function, // WK 2, 4, 8
    /// TF_DSV41_LIN2_CLUSTER: lanesc at lanesc_cfgs (resolved only with the knob: an older lin2 fatbin still loads)
    lanesc: [dense3_k2s.len][lanesc_cfgs.len]cuda.Function = undefined,

    pub fn resolve(module: cuda.Module, mode: Lin2Mode) !Lin2 {
        var l: Lin2 = .{ .module = module, .mode = mode, .lanes = undefined, .linear = undefined };
        inline for (dense3_k2s, 0..) |k2, i| {
            l.lanes[i][0] = try module.function(lanes2Symbol(k2, 4, 2));
            l.lanes[i][1] = try module.function(lanes2Symbol(k2, 8, 1));
        }
        inline for (linear_k2s, 0..) |k2, i| inline for (.{ 2, 4, 8 }, 0..) |wk, j| {
            l.linear[i][j] = try module.function(if (mode.pf) linear2Symbol(k2, wk, true) else linear2Symbol(k2, wk, false));
        };
        if (mode.cluster) inline for (dense3_k2s, 0..) |k2, i| inline for (lanesc_cfgs, 0..) |c, j| {
            l.lanesc[i][j] = try module.function(lanescSymbol(k2, c[0], c[1], c[2]));
        };
        return l;
    }
};
