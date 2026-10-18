//! Replays the EXL3 linear and routed-expert extension calls (linear, x3seg, dense3, experts, x3ld/x3pf, x3gm).

const std = @import("std");
const exl3 = @import("cuda").qlinear.block.experts;
const dense = @import("cuda").qlinear.block.dense;
const replay = @import("kernels_replay.zig");

const Call = replay.Call;
const Arg = replay.Arg;
const Kw = replay.Kw;
const numel = replay.numel;
const dim = replay.dim;
const ioType = replay.ioType;
const input = replay.input;
const elem = replay.elem;
const is = replay.is;

// tensorfold_exl3_linear_v4 (linear.cpp)
pub fn linearExt(o: dense.Ops, func: []const u8, args: []const Arg, kwargs: []const Kw) !bool {
    if (is(func, "rot_in")) {
        const c: Call = .{ .args = args, .kwargs = kwargs, .names = &.{ "x", "suh", "xh" } };
        const x = try c.tensor(0);
        try o.rotIn(x.ptr, try ioType(x.dtype), try c.ptr(1), try c.ptr(2), try dim(x, 0), try dim(x, 1));
        return true;
    }
    if (is(func, "linear") or is(func, "linear_skip")) {
        const c: Call = .{ .args = args, .kwargs = kwargs, .names = &.{ "xh", "T", "stride_k", "stride_nb", "svh", "bias", "y", "Z", "counters", "K2", "cb", "SK", "WK", "skip" } };
        if (try c.int(10) != 2) return false; // mul1 only is built (every DeepSeek-V4.1 matrix)
        const xh = try c.tensor(0);
        const y = try c.tensor(6);
        const skip: u64 = if (is(func, "linear_skip")) try c.ptrOrNull(13) else 0;
        try o.linear(xh.ptr, try c.ptr(1), try c.int(2), try c.int(3), try c.ptr(4), try c.ptrOrNull(5), y.ptr, try ioType(y.dtype), try c.ptrOrNull(7), try c.ptr(8), try dim(xh, 0), try dim(xh, 1), try dim(y, 1), try c.u(9), try c.usz(11), try c.usz(12), skip);
        return true;
    }
    if (is(func, "unpack")) {
        const c: Call = .{ .args = args, .kwargs = kwargs, .names = &.{ "T", "W", "stride_k", "stride_nb", "K2", "cb" } };
        if (try c.int(5) != 2) return false;
        const w = try c.tensor(1);
        try o.unpack(try c.ptr(0), w.ptr, try dim(w, 0), try dim(w, 1), try c.int(2), try c.int(3), try c.u(4));
        return true;
    }
    return false;
}

// tf_dsv41_x3seg_v3 / tf_dsv41_dense3_v1 (x3seg.cpp, dense3.cpp): the tables built as dsv41_*_linear_cuda builds them
pub fn segExt(o: dense.Ops, func: []const u8, args: []const Arg, kwargs: []const Kw, lanes: bool) !bool {
    if (is(func, "rot_in") and !lanes) {
        const c: Call = .{ .args = args, .kwargs = kwargs, .names = &.{ "xs", "suh", "xh", "xh_off", "pdl", "nw", "eps" } };
        const nws: []const Arg = if (try c.opt(5)) |v| switch (v) {
            .list => |l| l,
            else => return error.WrongArgument,
        } else &.{};
        const eps: f32 = if ((try c.opt(6)) != null) @floatCast(try c.float(6)) else 0;
        const xs = try c.list(0);
        const suh = try c.list(1);
        const offs = try c.list(3);
        const xh = try c.ptr(2);
        if (xs.len < 1 or xs.len > dense.max_seg or suh.len != xs.len or offs.len != xs.len) return error.WrongArgument;
        var tab = std.mem.zeroes(dense.RotTable);
        tab.n = @intCast(xs.len);
        var M: usize = 0;
        for (xs, suh, offs, 0..) |xa, sa, oa, i| {
            if (xa != .tensor or sa != .tensor or oa != .int) return error.WrongArgument;
            const x = xa.tensor;
            if (i == 0) M = try dim(x, 0);
            tab.s[i] = .{ .x = x.ptr, .suh = sa.tensor.ptr, .xh = xh + 2 * @as(u64, @intCast(oa.int)), .ld = x.stride[0], .K = @intCast(try dim(x, 1)), .x_dtype = @backingInt(try ioType(x.dtype)) };
            if (i < nws.len) { // R1c: the folded norm (an empty weight: none)
                const w = switch (nws[i]) {
                    .tensor => |t| t,
                    else => return error.WrongArgument,
                };
                if (numel(w) != 0) {
                    tab.s[i].nw = w.ptr;
                    tab.s[i].nw_dtype = @backingInt(try ioType(w.dtype));
                    tab.s[i].inv_k = dense.invK(try dim(x, 1));
                    tab.s[i].eps = eps;
                }
            }
        }
        try o.segRotIn(tab, M, try c.boolean(4, false));
        return true;
    }
    if (is(func, "linear")) {
        const c: Call = .{ .args = args, .kwargs = kwargs, .names = &.{ "xh", "T", "svh", "ys", "Z", "counters", "meta", "programs", "K2", "WK", "minb", "pdl" } };
        const T = try c.list(1);
        const svh = try c.list(2);
        const ys = try c.list(3);
        const meta = try c.list(6);
        const n = T.len;
        if (n < 1 or n > dense.max_seg or svh.len != n or ys.len != n or meta.len != 11 * n) return error.WrongArgument;
        const xh = try c.ptr(0);
        const z = try c.ptrOrNull(4);
        const counters = try c.ptr(5);
        if (ys[0] != .tensor) return error.WrongArgument;
        const y0 = ys[0].tensor;
        const M = try dim(y0, 0);
        const y_dtype = try ioType(y0.dtype);
        var tab = std.mem.zeroes(dense.LinTable);
        tab.n = @intCast(n);
        for (0..n) |i| {
            var m: [11]i64 = undefined;
            for (&m, meta[11 * i ..][0..11]) |*v, a| v.* = switch (a) {
                .int => |x| x,
                else => return error.WrongArgument,
            };
            if (T[i] != .tensor or svh[i] != .tensor or ys[i] != .tensor) return error.WrongArgument;
            const y = ys[i].tensor;
            tab.s[i] = .{
                .xh = xh + 2 * @as(u64, @intCast(m[6])),
                .T = T[i].tensor.ptr,
                .stride_k = m[4],
                .stride_nb = m[5],
                .svh = svh[i].tensor.ptr,
                .y = y.ptr + @as(u64, @intCast(m[9])) * elem(y.dtype),
                .Z = if (m[2] > 1) z + 4 * @as(u64, @intCast(m[7])) else 0,
                .counters = counters + 4 * @as(u64, @intCast(m[8])),
                .K = @intCast(m[0]),
                .N = @intCast(m[1]),
                .SK = @intCast(m[2]),
                .wk = @intCast(m[3]),
                .p0 = @intCast(m[10]),
                .ld = @intCast(y.stride[0]),
            };
        }
        const programs = try c.usz(7);
        const k2 = try c.u(8);
        const wk = try c.usz(9);
        const minb = try c.u(10);
        const pdl = try c.boolean(11, false);
        if (lanes) try o.lanesLinearRaw(tab, y_dtype, M, programs, k2, wk, minb, pdl) else try o.segLinearRaw(tab, y_dtype, M, programs, k2, wk, minb, pdl);
        return true;
    }
    if (is(func, "relayout") and lanes) {
        const c: Call = .{ .args = args, .kwargs = kwargs, .names = &.{ "src", "dst", "K2", "K", "lanes" } };
        const src = try c.tensor(0);
        try o.relayout(src.ptr, try c.ptr(1), @intCast(numel(src)), try c.u(2), try c.usz(3), try c.boolean(4, true));
        return true;
    }
    if (is(func, "unpack") and lanes) {
        const c: Call = .{ .args = args, .kwargs = kwargs, .names = &.{ "T", "W", "stride_k", "stride_nb", "K2" } };
        const w = try c.tensor(1);
        try o.lanesUnpack(try c.ptr(0), w.ptr, try dim(w, 0), try dim(w, 1), try c.int(2), try c.int(3), try c.u(4));
        return true;
    }
    return false;
}

const grouped_names = [_][]const u8{ "X0", "X1", "TP0", "TP1", "B0", "B1", "uids", "ucount", "members", "Z", "mats", "K", "N", "P", "SK", "slots", "cb" };

fn groupedArgs(c: Call, lo_i: usize) !exl3.Grouped {
    const uids = try c.tensor(6);
    const members = try c.tensor(8);
    return .{
        .x0 = try c.ptr(0),
        .x1 = try c.ptr(1),
        .tp0 = try c.ptr(2),
        .tp1 = try c.ptr(3),
        .k2_0 = try c.ptr(4),
        .k2_1 = try c.ptr(5),
        .uids = uids.ptr,
        .ucount = try c.ptr(7),
        .members = members.ptr,
        .z = try c.ptr(9),
        .mats = try c.usz(10),
        .K = try c.usz(11),
        .N = try c.usz(12),
        .P = try c.usz(13),
        .SK = try c.usz(14),
        .slots = try c.usz(15),
        .maxm = try dim(members, 1),
        .nexp = try dim(uids, 0),
        .lo = try c.u(lo_i),
        .hi = try c.u(lo_i + 1),
    };
}

// tensorfold_exl3_experts_v1 (experts.cpp)
pub fn expertsExt(o: exl3.Ops, func: []const u8, args: []const Arg, kwargs: []const Kw) !bool {
    if (is(func, "group_rot_in")) {
        const c: Call = .{ .args = args, .kwargs = kwargs, .names = &.{ "x", "x_stride", "pick", "suh0", "suh1", "out0", "out1", "rows", "K", "slots", "E", "uids", "ucount", "members" } };
        const x = try c.tensor(0);
        if (x.dtype != .bf16) return error.WrongArgument;
        const members = try c.tensor(13);
        try o.groupRotIn(x.ptr, try c.usz(1), try c.ptr(2), try c.ptr(3), try c.ptr(4), try c.ptr(5), try c.ptr(6), try c.ptr(11), try c.ptr(12), members.ptr, try c.usz(7), try c.usz(8), try c.usz(9), try c.usz(10), try dim(members, 1));
        return true;
    }
    if (is(func, "grouped")) {
        const c: Call = .{ .args = args, .kwargs = kwargs, .names = &(grouped_names ++ [_][]const u8{ "nt", "warps", "pf", "lo", "hi" }) };
        if (try c.int(16) != 2) return false; // cb 2 (mul1) instances only
        try o.grouped(try groupedArgs(c, 20), try c.u(17), try c.u(18), try c.u(19));
        return true;
    }
    if (is(func, "group")) {
        const c: Call = .{ .args = args, .kwargs = kwargs, .names = &.{ "pick", "uids", "ucount", "members", "R", "slots", "E" } };
        const members = try c.tensor(3);
        try o.group(try c.ptr(0), try c.ptr(1), try c.ptr(2), members.ptr, try c.usz(4), try c.usz(5), try c.usz(6), try dim(members, 1));
        return true;
    }
    if (is(func, "rot_in")) {
        const c: Call = .{ .args = args, .kwargs = kwargs, .names = &.{ "x", "x_stride", "pick", "suh0", "suh1", "out0", "out1", "rows", "K", "slots", "E" } };
        const x = try c.tensor(0);
        try o.rotIn(try input(x.dtype), x.ptr, try c.usz(1), try c.ptr(2), try c.ptr(3), try c.ptr(4), try c.ptr(5), try c.ptr(6), try c.usz(7), try c.usz(8), try c.usz(9), try c.usz(10));
        return true;
    }
    if (is(func, "gateup_epilogue")) {
        const c: Call = .{ .args = args, .kwargs = kwargs, .names = &.{ "Z", "pick", "svh_g", "svh_u", "suh_d", "xd", "rows", "P", "N", "SK", "slots", "E", "limit", "act_mode" } };
        try o.gateupEpilogue(try c.ptr(0), try c.ptr(1), try c.ptr(2), try c.ptr(3), try c.ptr(4), try c.ptr(5), try c.usz(6), try c.usz(7), try c.usz(8), try c.usz(9), try c.usz(10), try c.usz(11), @floatCast(try c.float(12)), @intCast(try c.int(13)));
        return true;
    }
    if (is(func, "down_epilogue")) {
        const c: Call = .{ .args = args, .kwargs = kwargs, .names = &.{ "Z", "pick", "svh_d", "y", "rows", "P", "D", "SK", "slots", "E" } };
        try o.downEpilogue(try c.ptr(0), try c.ptr(1), try c.ptr(2), try c.ptr(3), try c.usz(4), try c.usz(5), try c.usz(6), try c.usz(7), try c.usz(8), try c.usz(9));
        return true;
    }
    if (is(func, "combine")) {
        const c: Call = .{ .args = args, .kwargs = kwargs, .names = &.{ "y", "wts", "out", "rows", "D", "slots" } };
        try o.combine(try c.ptr(0), try c.ptr(1), try c.ptr(2), try c.usz(3), try c.usz(4), try c.usz(5));
        return true;
    }
    if (is(func, "down_combine")) {
        const c: Call = .{ .args = args, .kwargs = kwargs, .names = &.{ "Z", "pick", "svh_d", "y", "wts", "out", "rows", "P", "D", "SK", "slots", "E" } };
        const out = try c.tensor(5);
        if (out.dtype != .f32 and out.dtype != .bf16) return error.WrongArgument;
        try o.downCombine(try c.ptr(0), try c.ptr(1), try c.ptr(2), try c.ptr(3), try c.ptr(4), out.ptr, try c.usz(6), try c.usz(7), try c.usz(8), try c.usz(9), try c.usz(10), try c.usz(11), out.dtype == .bf16);
        return true;
    }
    if (is(func, "dequant")) {
        const c: Call = .{ .args = args, .kwargs = kwargs, .names = &.{ "T", "out", "k2", "cb" } };
        if (try c.int(3) != 2) return false;
        const t = try c.tensor(0);
        try o.dequant(t.ptr, try c.ptr(1), 16 * try dim(t, 0), 16 * try dim(t, 1), try c.u(2));
        return true;
    }
    return false;
}

// tf_dsv41_x3ld_v1.grouped (..., cb, nt, pd, probe, lo, hi, pdl) / tf_dsv41_x3pf_v1.grouped (..., cb, nt, mtl, lo, hi)
pub fn groupedExt(o: exl3.Ops, ext: []const u8, func: []const u8, args: []const Arg, kwargs: []const Kw) !bool {
    if (!is(func, "grouped")) return false;
    if (is(ext, "tf_dsv41_x3ld_v1")) {
        const c: Call = .{ .args = args, .kwargs = kwargs, .names = &(grouped_names ++ [_][]const u8{ "nt", "pd", "probe", "lo", "hi", "pdl" }) };
        if (try c.int(16) != 2) return error.Unsupported; // the binding refuses it too
        try o.x3ld(try groupedArgs(c, 20), try c.u(17), try c.u(18), try c.u(19), try c.boolean(22, false));
        return true;
    }
    const c: Call = .{ .args = args, .kwargs = kwargs, .names = &(grouped_names ++ [_][]const u8{ "nt", "mtl", "lo", "hi" }) };
    if (try c.int(16) != 2) return error.Unsupported;
    try o.x3pf(try groupedArgs(c, 19), try c.u(17), try c.u(18));
    return true;
}

// tf_dsv41_x3ld_epi_v1 (TF_DSV41_X3LD_EPI): x3ld.grouped's arguments to hi, then the epilogue's, then the ticket words
pub fn x3ldEpiExt(o: exl3.Ops, func: []const u8, args: []const Arg, kwargs: []const Kw) !bool {
    const base = grouped_names ++ [_][]const u8{ "nt", "pd", "lo", "hi", "pick" };
    if (is(func, "gateup")) {
        const c: Call = .{ .args = args, .kwargs = kwargs, .names = &(base ++ [_][]const u8{ "svh_g", "svh_u", "suh_d", "xd", "E", "limit", "act_mode", "ticket" }) };
        if (try c.int(16) != 2 or try c.int(17) != 8) return error.Unsupported;
        try o.x3ldEpi(try groupedArgs(c, 19), .gu, try c.u(18), .{
            .pick = try c.ptr(21),
            .sv0 = try c.ptr(22),
            .sv1 = try c.ptr(23),
            .sd = try c.ptr(24),
            .xd = try c.ptr(25),
            .E = @intCast(try c.int(26)),
            .limit = @floatCast(try c.float(27)),
            .act_mode = @intCast(try c.int(28)),
            .ticket = try c.ptr(29),
        });
        return true;
    }
    if (is(func, "down")) {
        const c: Call = .{ .args = args, .kwargs = kwargs, .names = &(base ++ [_][]const u8{ "svh_d", "y", "wts", "out", "E", "ticket" }) };
        if (try c.int(16) != 2 or try c.int(17) != 8) return error.Unsupported;
        const out = try c.tensor(25);
        if (out.dtype != .f32 and out.dtype != .bf16) return error.WrongArgument;
        try o.x3ldEpi(try groupedArgs(c, 19), if (out.dtype == .bf16) .dnb else .dn, try c.u(18), .{
            .pick = try c.ptr(21),
            .sv0 = try c.ptr(22),
            .y = try c.ptr(23),
            .wts = try c.ptr(24),
            .out = out.ptr,
            .E = @intCast(try c.int(26)),
            .ticket = try c.ptr(27),
        });
        return true;
    }
    return false;
}

// tf_dsv41_x3gm_v1 (x3gm.cpp)
pub fn x3gmExt(o: exl3.Ops, func: []const u8, args: []const Arg, kwargs: []const Kw) !bool {
    if (is(func, "gateup")) {
        const c: Call = .{ .args = args, .kwargs = kwargs, .names = &.{ "xg", "xu", "tg", "tu", "order", "pe", "poff", "pcnt", "npass", "ticket", "svh_g", "svh_u", "suh_d", "xd", "K", "N", "k2", "shx", "cfg", "use_ticket", "limit", "tables" } };
        try o.gmGateup(try c.ptr(0), try c.ptr(1), try c.ptr(2), try c.ptr(3), try c.ptr(4), try c.ptr(5), try c.ptr(6), try c.ptr(7), try c.ptr(8), try c.ptr(9), try c.ptr(10), try c.ptr(11), try c.ptr(12), try c.ptr(13), try c.usz(14), try c.usz(15), try c.u(16), try c.boolean(17, false), try c.usz(18), try c.boolean(19, true), @floatCast(try c.float(20)), try c.boolean(21, false));
        return true;
    }
    if (is(func, "down")) {
        const c: Call = .{ .args = args, .kwargs = kwargs, .names = &.{ "xd", "td", "order", "pe", "poff", "pcnt", "npass", "ticket", "svh_d", "y", "K", "N", "k2", "cfg", "use_ticket", "tables" } };
        try o.gmDown(try c.ptr(0), try c.ptr(1), try c.ptr(2), try c.ptr(3), try c.ptr(4), try c.ptr(5), try c.ptr(6), try c.ptr(7), try c.ptr(8), try c.ptr(9), try c.usz(10), try c.usz(11), try c.u(12), try c.usz(13), try c.boolean(14, true), try c.boolean(15, false));
        return true;
    }
    if (is(func, "gateup2")) {
        const c: Call = .{ .args = args, .kwargs = kwargs, .names = &.{ "xg", "xu", "tg", "tu", "k2e", "order", "pe", "poff", "pcnt", "npass", "ticket", "svh_g", "svh_u", "suh_d", "xd", "K", "N", "shx", "cfg", "use_ticket", "limit", "tables" } };
        try o.gm2Gateup(try c.ptr(0), try c.ptr(1), try c.ptr(2), try c.ptr(3), try c.ptr(4), try c.ptr(5), try c.ptr(6), try c.ptr(7), try c.ptr(8), try c.ptr(9), try c.ptr(10), try c.ptr(11), try c.ptr(12), try c.ptr(13), try c.ptr(14), try c.usz(15), try c.usz(16), try c.boolean(17, false), try c.usz(18), try c.boolean(19, true), @floatCast(try c.float(20)), try c.boolean(21, false));
        return true;
    }
    if (is(func, "down2")) {
        const c: Call = .{ .args = args, .kwargs = kwargs, .names = &.{ "xd", "td", "k2e", "order", "pe", "poff", "pcnt", "npass", "ticket", "svh_d", "y", "K", "N", "cfg", "use_ticket", "tables" } };
        try o.gm2Down(try c.ptr(0), try c.ptr(1), try c.ptr(2), try c.ptr(3), try c.ptr(4), try c.ptr(5), try c.ptr(6), try c.ptr(7), try c.ptr(8), try c.ptr(9), try c.ptr(10), try c.usz(11), try c.usz(12), try c.usz(13), try c.boolean(14, true), try c.boolean(15, false));
        return true;
    }
    if (is(func, "rot")) {
        const c: Call = .{ .args = args, .kwargs = kwargs, .names = &.{ "x", "x_stride", "pick", "suh0", "suh1", "out0", "out1", "K", "slots", "pairs", "mats" } };
        const x = try c.tensor(0);
        try o.gmRot(try input(x.dtype), x.ptr, try c.usz(1), try c.ptr(2), try c.ptr(3), try c.ptr(4), try c.ptr(5), try c.ptr(6), try c.usz(7), try c.usz(8), try c.usz(9), try c.usz(10));
        return true;
    }
    if (is(func, "dequant")) {
        const c: Call = .{ .args = args, .kwargs = kwargs, .names = &.{ "t", "out", "k2" } };
        const out = try c.tensor(1);
        try o.gmDequant(try c.ptr(0), out.ptr, try dim(out, 0), try dim(out, 1), try c.u(2));
        return true;
    }
    return false;
}
