//! Delta-rule learner tape extents; each value head owns its reverse-time state and its partial Q/K sums.
const std = @import("std");
const rows = @import("living_rows.zig");
const wo = @import("weight_only.zig");
pub const Params = rows.Params;
pub const Op = enum { front, norm, readout, scan, qk, gates, conv };
pub const names = [_][:0]const u8{ "tf_lw_gdn_front", "tf_lw_gdn_norm", "tf_lw_gdn_readout_back", "tf_lw_gdn_scan_back", "tf_lw_gdn_qk_back", "tf_lw_gdn_gates_back", "tf_lw_gdn_conv_back" };
pub const threads = 128;
fn mul(values: []const usize) !usize {
    var n: usize = 1;
    for (values) |v| n = try std.math.mul(usize, n, v);
    return n;
}
pub fn needs(op: Op, p: Params) !rows.Needs {
    const r: usize = p.u[0];
    const nk: usize = p.u[1];
    const nv: usize = p.u[2];
    const pw: usize = p.u[3];
    if (r == 0 or r > 512 or nk == 0 or nv == 0 or nv % nk != 0 or nv > 128 or
        p.u[4] > 1 or p.u[5] > 1 or p.u[6] > 1 or !std.math.isFinite(p.f[0]) or p.f[0] <= 0)
        return error.LivingGdnShape;
    const ch = (2 * nk + nv) * 128;
    if (pw != ch + nv * 130) return error.LivingGdnShape;
    const raw = try mul(&.{ r, pw, 2 });
    const act = try mul(&.{ r, ch, 2 });
    const qk = try mul(&.{ r, 2, nk, 128, 4 });
    const state = try mul(&.{ r, nv, 128, 128, 4 });
    const head = try mul(&.{ r, nv, 128, 4 });
    const proj = try mul(&.{ r, pw, 4 });
    const parts = try mul(&.{ r, nv, 386, 4 });
    var n: rows.Needs = .{ .cells = try mul(&.{ r, ch }) };
    switch (op) {
        .front => {
            n.bytes[0] = raw;
            n.bytes[1] = ch * 4 * 4;
            n.bytes[8] = act;
        },
        .norm => {
            n.cells = try mul(&.{ r, 2, nk, 128 });
            n.bytes[0] = act;
            n.bytes[8] = qk;
        },
        .readout => {
            n.cells = try mul(&.{ r, nv, 128 });
            n.bytes[0] = raw;
            n.bytes[1] = qk;
            n.bytes[2] = state;
            n.bytes[3] = 128 * 4;
            n.bytes[4] = head;
            n.bytes[8] = head;
            n.bytes[9] = proj;
        },
        .scan => {
            n.cells = nv;
            n.bytes[0] = act;
            n.bytes[1] = qk;
            n.bytes[2] = state;
            n.bytes[3] = head;
            n.bytes[4] = raw;
            n.bytes[5] = nv * 4;
            n.bytes[6] = nv * 4;
            n.bytes[8] = parts;
            n.bytes[9] = try mul(&.{ nv, 128, 128, 4 });
        },
        .qk => {
            n.bytes[0] = act;
            n.bytes[2] = parts;
            n.bytes[8] = act * 2;
        },
        .gates => {
            n.cells = r * nv;
            n.bytes[0] = parts;
            n.bytes[1] = raw;
            n.bytes[2] = nv * 4;
            n.bytes[3] = nv * 4;
            n.bytes[8] = proj;
        },
        .conv => {
            n.bytes[0] = raw;
            n.bytes[1] = ch * 4 * 4;
            n.bytes[2] = act * 2;
            n.bytes[8] = proj;
        },
    }
    for (n.bytes) |bytes| if (bytes > std.math.maxInt(u32)) return error.LivingGdnShape;
    if (n.cells > std.math.maxInt(u32)) return error.LivingGdnShape;
    return n;
}
pub fn finish(op: Op, p: Params) !Params {
    var out = p;
    out.u[15] = @intCast((try needs(op, p)).cells);
    return out;
}
pub fn validate(op: Op, p: Params, ranges: [11]wo.Range) !void {
    const n = try needs(op, p);
    if (p.u[15] != n.cells) return error.LivingGdnShape;
    for (ranges, n.bytes) |v, size| {
        if (size == 0) continue;
        if (v.address == 0 or v.address % 4 != 0 or v.len < size) return error.LivingGdnBuffer;
        _ = try std.math.add(usize, v.address, size);
    }
    for (8..11) |i| {
        if (n.bytes[i] == 0) continue;
        for (0..i) |j| {
            if (n.bytes[j] == 0) continue;
            const x = ranges[i].address;
            const y = ranges[j].address;
            if (if (x <= y) y - x < n.bytes[i] else x - y < n.bytes[j]) return error.LivingGdnAlias;
        }
    }
}
