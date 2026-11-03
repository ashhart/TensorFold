//! Fixed selected-attention pullbacks consume the recorded IDs; no index ranking or pooling is rerun.
const std = @import("std");
const rows = @import("living_rows.zig");
const wo = @import("weight_only.zig");
pub const Params = rows.Params;
pub const Op = enum { scores, softmax, query, key_value };
pub const names = [_][:0]const u8{ "tf_lw_qsa_scores", "tf_lw_qsa_softmax_back", "tf_lw_qsa_query_back", "tf_lw_qsa_kv_back" };
pub const threads = 128;
fn mul(xs: []const usize) !usize {
    var n: usize = 1;
    for (xs) |v| n = try std.math.mul(usize, n, v);
    return n;
}
pub fn needs(op: Op, p: Params) !rows.Needs {
    const r: usize = p.u[0];
    const h: usize = p.u[1];
    const kv: usize = p.u[2];
    const dim: usize = p.u[3];
    const ids: usize = p.u[4];
    if (r == 0 or r > 512 or h == 0 or kv == 0 or h % kv != 0 or dim == 0 or dim > 512 or ids == 0 or
        p.u[5] > 2 or (p.u[5] == 2 and (dim < 128 or dim % 128 != 0 or p.u[6] != dim / 128)) or !std.math.isFinite(p.f[0]) or p.f[0] <= 0) return error.LivingQsaShape;
    const query = try mul(&.{ r, h, dim, 2 });
    const key = try mul(&.{ r, kv, dim, 2 });
    const selected = try mul(&.{ r, ids, 4 });
    const weights = try mul(&.{ r, h, r, 4 });
    var n: rows.Needs = .{ .cells = try mul(&.{ r, h }) };
    switch (op) {
        .scores => {
            n.bytes = .{ query, key, key, query * 2, selected, r * 4, r * 4, 0, weights, weights, 0 };
        },
        .softmax => {
            n.bytes = .{ weights, weights, selected, r * 4, r * 4, 0, 0, 0, weights, weights, 0 };
        },
        .query => {
            n.cells = query / 2;
            n.bytes = .{ weights, key, selected, r * 4, r * 4, 0, 0, 0, query * 2, 0, 0 };
        },
        .key_value => {
            n.cells = key / 2;
            n.bytes = .{ weights, weights, query, query * 2, selected, r * 4, r * 4, 0, key * 2, key * 2, 0 };
        },
    }
    if (n.cells > std.math.maxInt(u32)) return error.LivingQsaShape;
    for (n.bytes) |bytes| if (bytes > std.math.maxInt(u32)) return error.LivingQsaShape;
    return n;
}
pub fn finish(op: Op, p: Params) !Params {
    var out = p;
    out.u[15] = @intCast((try needs(op, p)).cells);
    return out;
}
pub fn validate(op: Op, p: Params, ranges: [11]wo.Range) !void {
    const n = try needs(op, p);
    if (p.u[15] != n.cells) return error.LivingQsaShape;
    for (ranges, n.bytes) |v, len| {
        if (len == 0) continue;
        if (v.address == 0 or v.address % 4 != 0 or v.len < len) return error.LivingQsaBuffer;
        _ = try std.math.add(usize, v.address, len);
    }
    for (8..11) |i| {
        if (n.bytes[i] == 0) continue;
        for (0..i) |j| {
            if (n.bytes[j] == 0) continue;
            const x = ranges[i].address;
            const y = ranges[j].address;
            if (if (x <= y) y - x < n.bytes[i] else x - y < n.bytes[j]) return error.LivingQsaAlias;
        }
    }
}
