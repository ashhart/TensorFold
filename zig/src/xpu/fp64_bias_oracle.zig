//! Independent host BiasCorr oracle: logical sidecar keys, never the device's assembled bias buffer.
const std = @import("std");
const q = @import("qwen27.zig");
const alloc = q.alloc;

fn canonical(name: []const u8) []const u8 {
    if (std.mem.startsWith(u8, name, "language_model.")) return name["language_model.".len..];
    if (std.mem.startsWith(u8, name, "model.language_model.")) return name["model.language_model.".len..];
    return name;
}

fn strictMode() bool {
    const v = std.c.getenv("XPU_BIASCORR_FILE") orelse return false;
    if (std.mem.span(v).len == 0) return false;
    const partial = std.c.getenv("XPU_BIASCORR_ALLOW_PARTIAL") orelse return true;
    return !std.mem.eql(u8, std.mem.span(partial), "1");
}

fn vector(c: *const q.Checkpoint, sidecar_start: usize, comptime fmt: []const u8, args: anytype) !?[]const u8 {
    var buf: [256]u8 = undefined;
    const want = std.fmt.bufPrint(&buf, fmt, args) catch return error.BadBiasCorrOracle;
    if (sidecar_start == std.math.maxInt(usize) or sidecar_start > c.tensors.items.len) return null;
    for (c.tensors.items[sidecar_start..]) |t| {
        if (!std.mem.eql(u8, canonical(t.name), want)) continue;
        if (!std.mem.eql(u8, t.dtype, "F32") or t.rank != 1 or t.bytes.len != @as(usize, @intCast(t.shape[0])) * 4) return error.BadBiasCorrVector;
        for (0..@as(usize, @intCast(t.shape[0]))) |i| {
            const v: f32 = @bitCast(std.mem.readInt(u32, t.bytes[4 * i ..][0..4], .little));
            if (!std.math.isFinite(v)) return error.NonFiniteBiasCorr;
        }
        return t.bytes;
    }
    return null;
}

fn f32at(bytes: []const u8, i: usize) f32 {
    return @bitCast(std.mem.readInt(u32, bytes[4 * i ..][0..4], .little));
}

fn copyMember(dst: []f32, dst0: usize, src: ?[]const u8, src0: usize, len: usize) void {
    const b = src orelse return;
    for (0..len) |i| dst[dst0 + i] = f32at(b, src0 + i);
}

fn member(c: *const q.Checkpoint, sidecar_start: usize, layer: usize, comptime leaf: []const u8) !?[]const u8 {
    return vector(c, sidecar_start, "model.layers.{d}." ++ leaf ++ ".bias_corr", .{layer});
}

fn need(found: ?[]const u8, comptime leaf: []const u8, layer: usize) !?[]const u8 {
    if (found == null and strictMode()) {
        std.debug.print("missing expected biascorr member: model.layers.{d}." ++ leaf ++ ".bias_corr\n", .{layer});
        return error.MissingBiasCorrMember;
    }
    return found;
}

/// Logical bias in this rank's device column order; probes run on rank 0.
pub fn assemble(c: *const q.Checkpoint, layer: usize, site: u32, n: usize, sidecar_start: usize) !?[]f32 {
    if (site == 4 or std.c.getenv("XPU_BIASCORR_FILE") == null) return null;
    const out = try alloc.alloc(f32, n);
    @memset(out, 0);
    var used = false;
    const tp = q.TP;
    if (site == 2) { // mlp gate|up, device-interleaved in blocks of 16
        const full = q.FF;
        const local = full / tp;
        const gate = try need(try member(c, sidecar_start, layer, "mlp.gate_proj"), "mlp.gate_proj", layer);
        const up = try need(try member(c, sidecar_start, layer, "mlp.up_proj"), "mlp.up_proj", layer);
        used = gate != null or up != null;
        for (0..local) |i| {
            const col0 = (i / 16) * 32 + i % 16;
            if (gate) |b| out[col0] = f32at(b, i); // rank 0 logical slice
            if (up) |b| out[col0 + 16] = f32at(b, i);
        }
    } else if (site == 3) {
        used = try copyDense(c, sidecar_start, layer, "mlp.down_proj", out, 0, n);
    } else if (site == 1) {
        used = if (q.isAttention(layer)) try copyDense(c, sidecar_start, layer, "self_attn.o_proj", out, 0, n) else try copyDense(c, sidecar_start, layer, "linear_attn.out_proj", out, 0, n);
    } else if (q.isAttention(layer)) { // q,k,v concatenated column-parallel members
        const qfull: usize = 24 * 512;
        const kvfull: usize = 4 * 256;
        const ql = qfull / tp;
        const kl = kvfull / tp;
        const qq = try need(try member(c, sidecar_start, layer, "self_attn.q_proj"), "self_attn.q_proj", layer);
        const kk = try need(try member(c, sidecar_start, layer, "self_attn.k_proj"), "self_attn.k_proj", layer);
        const vv = try need(try member(c, sidecar_start, layer, "self_attn.v_proj"), "self_attn.v_proj", layer);
        used = qq != null or kk != null or vv != null;
        copyMember(out, 0, qq, 0, ql);
        copyMember(out, ql, kk, 0, kl);
        copyMember(out, ql + kl, vv, 0, kl);
    } else { // GDN logical q|k|v inside in_proj_qkv, then z,a,b concatenated
        const qfull: usize = 16 * 128;
        const kfull: usize = 16 * 128;
        const vfull: usize = 48 * 128;
        const ql = qfull / tp;
        const kl = kfull / tp;
        const vl = vfull / tp;
        const zl = vfull / tp;
        const al = 48 / tp;
        const qkv = try need(try member(c, sidecar_start, layer, "linear_attn.in_proj_qkv"), "linear_attn.in_proj_qkv", layer);
        const zz = try need(try member(c, sidecar_start, layer, "linear_attn.in_proj_z"), "linear_attn.in_proj_z", layer);
        const aa = try need(try member(c, sidecar_start, layer, "linear_attn.in_proj_a"), "linear_attn.in_proj_a", layer);
        const bb = try need(try member(c, sidecar_start, layer, "linear_attn.in_proj_b"), "linear_attn.in_proj_b", layer);
        used = qkv != null or zz != null or aa != null or bb != null;
        var at: usize = 0;
        copyMember(out, at, qkv, 0, ql);
        at += ql;
        copyMember(out, at, qkv, qfull, kl);
        at += kl;
        copyMember(out, at, qkv, qfull + kfull, vl);
        at += vl;
        copyMember(out, at, zz, 0, zl);
        at += zl;
        copyMember(out, at, aa, 0, al);
        at += al;
        copyMember(out, at, bb, 0, al);
    }
    if (!used) {
        alloc.free(out);
        return null;
    }
    return out;
}

fn copyDense(c: *const q.Checkpoint, sidecar_start: usize, layer: usize, comptime leaf: []const u8, out: []f32, at: usize, n: usize) !bool {
    const b = try need(try member(c, sidecar_start, layer, leaf), leaf, layer) orelse return false;
    copyMember(out, at, b, 0, @min(n, b.len / 4));
    return true;
}
