//! Native attention source expansion and faults in the real decode, query, score and value accumulator paths.
const std = @import("std");
const mtl = @import("metal");
const ks = @import("kernel_sources");
pub const Kind = enum { dense, scan };
pub const Fault = enum { clip, input_write, position, half_value, half_score, split_tail };
pub fn entry(kind: Kind) [:0]const u8 {
    return if (kind == .dense) "dsv41_dense_kv_rows" else "dsv41_sparse_nax_rows";
}

fn replace(a: std.mem.Allocator, input: *[]const u8, old: []const u8, new: []const u8) !void {
    if (std.mem.count(u8, input.*, old) != 1) return error.NativeAttentionFaultAnchorChanged;
    const next = try std.mem.replaceOwned(u8, a, input.*, old, new);
    a.free(input.*);
    input.* = next;
}

pub fn text(comptime frags: type, device: mtl.Device, a: std.mem.Allocator, kind: Kind, fault: ?Fault) ![]u8 {
    var rows: []const u8 = try a.dupe(u8, ks.deepseek_v41_csa2_rows);
    defer a.free(rows);
    var nax: []const u8 = try a.dupe(u8, ks.deepseek_v41_csa2_nax);
    defer a.free(nax);
    if (fault) |f| switch (f) {
        .clip => switch (kind) {
            .dense => try replace(a, &rows, "  return v;", "  for (short i=0;i<4;i++) if (!isfinite(float(v[i]))) v[i]=bfloat(0); return v;"),
            .scan => try replace(a, &rows, "for (short t = 0; t < 8; t++) frag_get(qf[t], qb, RANK, 0, 16 * t, home);", "for (short t=0;t<8;t++) { frag_get(qf[t],qb,RANK,0,16*t,home); for(short e=0;e<8;e++) if(!isfinite(float(qf[t][e]))) qf[t][e]=bfloat(0); }"),
        },
        .input_write => switch (kind) {
            .dense => try replace(a, &rows, "const uint position = pos[slot];", "const uint position = pos[slot]; const_cast<device uchar*>(chunk)[0]=0;"),
            .scan => try replace(a, &rows, "const uint row = tg.y, query = pos[row];", "const uint row = tg.y, query = pos[row]; const_cast<device float*>(sink)[0]=as_type<float>(0x7fc00000u);"),
        },
        .position => switch (kind) {
            .dense => try replace(a, &rows, "const uint position = pos[slot];", "const uint position = pos[0];"),
            .scan => try replace(a, &rows, "const uint row = tg.y, query = pos[row];", "const uint row = tg.y, query = pos[0];"),
        },
        .split_tail => {
            if (kind != .scan) return error.NoDenseAttentionSum;
            try replace(a, &rows, "if (j < j_end)", "if (j < count)");
        },
        .half_score => {
            if (kind != .scan) return error.NoDenseAttentionSum;
            try replace(a, &rows, "mma_16x32<false, true>(lo_s, hi_s, qf[t], frag<bfloat>(a, b), frag<bfloat>(c, e));", "mma_16x32<false, true>(lo_s,hi_s,qf[t],frag<bfloat>(a,b),frag<bfloat>(c,e)); for(short z=0;z<8;z++){lo_s[z]=float(half(lo_s[z]));hi_s[z]=float(half(hi_s[z]));}");
        },
        .half_value => {
            if (kind != .scan) return error.NoDenseAttentionSum;
            try replace(a, &rows, "mma_16x32<false, false>(acc[c][0], acc[c][1], p1[kb], v0, v1);", "mma_16x32<false,false>(acc[c][0],acc[c][1],p1[kb],v0,v1); for(short z=0;z<8;z++){acc[c][0][z]=float(half(acc[c][0][z]));acc[c][1][z]=float(half(acc[c][1][z]));}");
            try replace(a, &nax, "acc[j][e] += part[4 + c * 16 + j * 8 + e] * f[e >> 2];", "acc[j][e] = float(half(acc[j][e] + part[4+c*16+j*8+e]*f[e>>2]));");
            try replace(a, &nax, "acc[c][j][e] += part[4 + c * 16 + j * 8 + e] * f[e >> 2];", "acc[c][j][e] = float(half(acc[c][j][e] + part[4+c*16+j*8+e]*f[e>>2]));");
        },
    };
    const expanded = try frags.sourceMode(device, a, nax, false);
    defer a.free(expanded);
    return std.mem.concat(a, u8, &.{ ks.deepseek_v41_kv, ks.deepseek_v41_csa2, expanded, rows });
}
