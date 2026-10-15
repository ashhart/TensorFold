//! GLM-5.3-Flash's routed experts for one rank, read from the checkpoint and packed (weights.zig's Builder).

const std = @import("std");
const kern = @import("kern.zig");
const ckpt = @import("ckpt.zig");
const Builder = @import("weights.zig").Builder;
const WORLD = ckpt.WORLD;
const fullName = ckpt.fullName;
const numelOf = ckpt.numelOf;
const f16ToBf16 = ckpt.f16ToBf16;
const bf16Value = ckpt.bf16Value;

// -- routed experts: per-expert MLX triples, rank rows (gate, up) or columns (down), packed as experts_pack.cu ----

const Triple = struct {
    e: usize,
    n: usize,
    k8: usize,
    kg: usize,
    words: []u32,
    scales: []u16,
    biases: []u16,

    fn free(t: Triple, gpa: std.mem.Allocator) void {
        gpa.free(t.words);
        gpa.free(t.scales);
        gpa.free(t.biases);
    }
};

/// Every expert's ``proj`` of layer ``li``: rank rows (gate, up) or columns (down), fp16 scales and biases to bf16.
fn readExperts(b: *Builder, li: usize, proj: []const u8, count: usize) !Triple {
    var nb: [256]u8 = undefined;
    const w0 = try b.ck.get(try fullName(&nb, "layers.{d}.mlp.experts.0.{s}.weight", .{ li, proj }));
    const n_full: usize = @intCast(w0.shape[0]);
    const k8_full: usize = @intCast(w0.shape[1]);
    var nb2: [256]u8 = undefined;
    const s0 = try b.ck.get(try fullName(&nb2, "layers.{d}.mlp.experts.0.{s}.scales", .{ li, proj }));
    const kg_full: usize = @intCast(s0.shape[1]);
    const down = std.mem.eql(u8, proj, "down_proj");
    const nr = if (down) n_full else n_full / WORLD;
    const r0 = if (down) 0 else b.rank * nr;
    const ok8 = if (down) k8_full / WORLD else k8_full;
    const okg = if (down) kg_full / WORLD else kg_full;
    const c8 = if (down) b.rank * ok8 else 0;
    const cg = if (down) b.rank * okg else 0;
    var t: Triple = .{ .e = count, .n = nr, .k8 = ok8, .kg = okg, .words = try b.gpa.alloc(u32, count * nr * ok8), .scales = try b.gpa.alloc(u16, count * nr * okg), .biases = try b.gpa.alloc(u16, count * nr * okg) };
    errdefer t.free(b.gpa);
    const rowbuf = try b.gpa.alloc(u8, nr * k8_full * 4);
    defer b.gpa.free(rowbuf);
    const sbuf = try b.gpa.alloc(u8, nr * kg_full * 2);
    defer b.gpa.free(sbuf);
    for (0..count) |x| {
        var n1: [256]u8 = undefined;
        const we = try b.ck.get(try fullName(&n1, "layers.{d}.mlp.experts.{d}.{s}.weight", .{ li, x, proj }));
        try b.ck.readAt(we, r0 * k8_full * 4, rowbuf);
        const w32 = std.mem.bytesAsSlice(u32, rowbuf);
        for (0..nr) |r| @memcpy(t.words[(x * nr + r) * ok8 ..][0..ok8], w32[r * k8_full + c8 ..][0..ok8]);
        for ([_][]const u8{ "scales", "biases" }, 0..) |what, which| {
            var n2: [256]u8 = undefined;
            const ent = try b.ck.get(try fullName(&n2, "layers.{d}.mlp.experts.{d}.{s}.{s}", .{ li, x, proj, what }));
            try b.ck.readAt(ent, r0 * kg_full * 2, sbuf);
            const s16 = std.mem.bytesAsSlice(u16, sbuf);
            const dst = if (which == 0) t.scales else t.biases;
            for (0..nr) |r| for (0..okg) |g| {
                dst[(x * nr + r) * okg + g] = f16ToBf16(s16[r * kg_full + cg + g]);
            };
        }
    }
    return t;
}

/// experts_pack.cu's index map: per block 32 * 4 * H nibble-shuffled words, then 32 (16 symmetric) scale words.
fn checkSymmetric(t: Triple) !void {
    for (t.scales, t.biases) |sc, bi| {
        if (bf16Value(bi) != -8 * bf16Value(sc)) return error.ExpertsNotSymmetric;
    }
}

fn packExperts(t: Triple, gs: usize, out: []u32, mcount: usize, m: usize, sym: bool, three: bool) void {
    const h = gs / 32;
    const Work = struct {
        fn run(tt: Triple, hh: usize, o: []u32, mc: usize, mm: usize, e0: usize, e1: usize, sy: bool, th: bool) void {
            const wp = 32 * 4 * hh;
            const wsec = if (th) wp * 3 / 4 else wp;
            const blk = wsec + @as(usize, if (sy) 16 else 32);
            var tmp: [512]u32 = undefined; // a 3-bit block's 4-bit words before their planes
            const nbb = tt.n / 32;
            for (e0..e1) |e| for (0..nbb) |bb| for (0..tt.kg) |g| {
                const dst = o[(((e * nbb + bb) * tt.kg + g) * mc + mm) * blk ..][0..blk];
                const wdst = if (th) tmp[0..wp] else dst[0..wp];
                for (0..wp) |i| {
                    const row = i / (4 * hh);
                    const kk = i % (4 * hh);
                    const tq = row / 8;
                    const r = row % 8;
                    const q = kk / hh;
                    const j = kk % hh;
                    const w = tt.words[(e * tt.n + bb * 32 + row) * tt.k8 + g * 4 * hh + kk];
                    const f = ((tq * hh + j) * 8 + r) * 4 + q;
                    wdst[(f / 128) * 128 + (f % 32) * 4 + (f / 32) % 4] = shuffle(w);
                }
                if (th) for (0..hh) |c| for (0..32) |lane| { // the lane's uint4 c as three planes (experts_sym.cuh)
                    var lo: [4]u32 = undefined;
                    var hi: [4]u32 = undefined;
                    for (0..4) |k| {
                        const word = tmp[(c * 32 + lane) * 4 + k];
                        lo[k] = 0;
                        hi[k] = 0;
                        for (0..8) |i| {
                            const code = ((word >> @intCast(4 * i)) & 15) -% 4; // 3-bit codes are stored + 4
                            lo[k] |= (code & 3) << @intCast(2 * i);
                            hi[k] |= ((code >> 2) & 1) << @intCast(i);
                        }
                    }
                    dst[c * 96 + lane] = lo[0] | (lo[1] << 16);
                    dst[c * 96 + 32 + lane] = lo[2] | (lo[3] << 16);
                    dst[c * 96 + 64 + lane] = hi[0] | (hi[1] << 8) | (hi[2] << 16) | (hi[3] << 24);
                };
                if (sy) { // quad p: rows j * 8 + 2p and + 1's scales, j = 0..3 (experts_sym.cuh)
                    for (0..16) |s| {
                        const p = s / 4;
                        const j = s % 4;
                        const at = (e * tt.n + bb * 32 + j * 8 + p * 2) * tt.kg + g;
                        dst[wsec + s] = @as(u32, tt.scales[at]) | (@as(u32, tt.scales[at + tt.kg]) << 16);
                    }
                } else for (0..32) |s| {
                    const p = s / 8;
                    const tt4 = s % 4;
                    const src = if ((s / 4) % 2 == 1) tt.biases else tt.scales;
                    const at = (e * tt.n + bb * 32 + tt4 * 8 + p * 2) * tt.kg + g;
                    dst[wp + s] = @as(u32, src[at]) | (@as(u32, src[at + tt.kg]) << 16);
                }
            };
        }
    };
    const threads = 16;
    var pool: [threads]?std.Thread = @splat(null);
    const per = (t.e + threads - 1) / threads;
    for (0..threads) |i| {
        const e0 = i * per;
        const e1 = @min(t.e, e0 + per);
        if (e0 >= e1) break;
        pool[i] = std.Thread.spawn(.{}, Work.run, .{ t, h, out, mcount, m, e0, e1, sym, three }) catch blk: {
            Work.run(t, h, out, mcount, m, e0, e1, sym, three);
            break :blk null;
        };
    }
    for (pool) |th| if (th) |x| x.join();
}

fn shuffle(w: u32) u32 {
    var even = w & 0x0F0F0F0F;
    var odd = (w >> 4) & 0x0F0F0F0F;
    even = (even | (even >> 4)) & 0x00FF00FF;
    odd = (odd | (odd >> 4)) & 0x00FF00FF;
    even = (even | (even >> 8)) & 0x0000FFFF;
    odd = (odd | (odd >> 8)) & 0x0000FFFF;
    return even | (odd << 16);
}

/// grouped.make([gate, up], down, 64): up [E, N/32, K/64, 2, block], down [E, D/32, NI/64, 1, block] int32.
pub fn experts(b: *Builder, out: []const u8, li: usize) !void {
    const E: usize = @intCast(kern.EXPERTS);
    const gs: usize = 64;
    if (b.experts3 and !b.experts_sym) return error.Experts3NeedsSymmetric;
    const wsec: usize = 32 * 4 * (gs / 32) * @as(usize, if (b.experts3) 3 else 4) / 4;
    const blk: usize = wsec + @as(usize, if (b.experts_sym) 16 else 32);
    const width: usize = @intCast(@divExact(2048, WORLD)); // moe_intermediate_size / ranks
    const dims: usize = @intCast(kern.D);
    const up_shape = [_]i64{ @intCast(E), @intCast(width / 32), @intCast(dims / gs), 2, @intCast(blk) };
    const down_shape = [_]i64{ @intCast(E), @intCast(dims / 32), @intCast(width / gs), 1, @intCast(blk) };
    var nb: [256]u8 = undefined;
    var nb2: [256]u8 = undefined;
    const up_name = try std.fmt.bufPrint(&nb, "{s}.up", .{out});
    const down_name = try std.fmt.bufPrint(&nb2, "{s}.down", .{out});
    if (b.dry) {
        try b.put(up_name, .i32, &up_shape, null);
        try b.put(down_name, .i32, &down_shape, null);
        return;
    }
    {
        const up = try b.gpa.alloc(u32, numelOf(&up_shape));
        defer b.gpa.free(up);
        for ([_][]const u8{ "gate_proj", "up_proj" }, 0..) |proj, m| {
            const t = try readExperts(b, li, proj, E);
            defer t.free(b.gpa);
            if (t.n != width or t.k8 * 8 != dims) return error.ExpertShape;
            if (b.experts_sym) try checkSymmetric(t);
            if (b.sim_proj != 1) {
                if (b.experts3) simulateBits(t, 3) else if (b.expert_sim_bits != 0) simulateBits(t, b.expert_sim_bits);
            }
            packExperts(t, gs, up, 2, m, b.experts_sym, b.experts3);
        }
        try b.put(up_name, .i32, &up_shape, std.mem.sliceAsBytes(up));
    }
    const dn = try b.gpa.alloc(u32, numelOf(&down_shape));
    defer b.gpa.free(dn);
    const t = try readExperts(b, li, "down_proj", E);
    defer t.free(b.gpa);
    if (t.n != dims or t.k8 * 8 != width) return error.ExpertShape;
    if (b.experts_sym) try checkSymmetric(t);
    if (b.sim_proj != 2) {
        if (b.experts3) simulateBits(t, 3) else if (b.expert_sim_bits != 0) simulateBits(t, b.expert_sim_bits);
    }
    packExperts(t, gs, dn, 1, 0, b.experts_sym, b.experts3);
    try b.put(down_name, .i32, &down_shape, std.mem.sliceAsBytes(dn));
}

/// Layer ``li``'s EXL3 experts, rank tiles: trellis words [E, K/16, N/16, 32], fp16 suh [E, K] and svh [E, N].
pub fn expertsExl3(b: *Builder, out: []const u8, li: usize) !void {
    const E: usize = @intCast(kern.EXPERTS);
    const D: usize = @intCast(kern.D);
    const width: usize = @intCast(@divExact(2048, WORLD));
    const tile: usize = 128; // a 16x16 tile's bytes at 4 bits
    var nb: [256]u8 = undefined;
    const Mat = struct { proj: []const u8, kt: usize, nt: usize, by_cols: bool };
    const mats = [_]Mat{ .{ .proj = "gate_proj", .kt = D / 16, .nt = width / 16, .by_cols = true }, .{ .proj = "up_proj", .kt = D / 16, .nt = width / 16, .by_cols = true }, .{ .proj = "down_proj", .kt = width / 16, .nt = D / 16, .by_cols = false } };
    const short = [_][]const u8{ "g", "u", "d" };
    for (mats, short) |m, sh| {
        const tshape = [_]i64{ @intCast(E), @intCast(m.kt), @intCast(m.nt), 32 };
        const k_all: usize = if (m.by_cols) D else 2048;
        const n_all: usize = if (m.by_cols) 2048 else D;
        const sus = [_]i64{ @intCast(E), @intCast(m.kt * 16) };
        const svs = [_]i64{ @intCast(E), @intCast(m.nt * 16) };
        const tn = try std.fmt.bufPrint(&nb, "{s}.{s}t", .{ out, sh });
        if (b.dry) {
            try b.put(tn, .i32, &tshape, null);
            try b.putf("{s}.suh_{s}", .{ out, sh }, .f16, &sus, null);
            try b.putf("{s}.svh_{s}", .{ out, sh }, .f16, &svs, null);
            continue;
        }
        const per = m.kt * m.nt * tile;
        const words = try b.gpa.alignedAlloc(u8, .@"16", E * per);
        defer b.gpa.free(words);
        const suh = try b.gpa.alloc(u16, E * m.kt * 16);
        defer b.gpa.free(suh);
        const svh = try b.gpa.alloc(u16, E * m.nt * 16);
        defer b.gpa.free(svh);
        var f: [256]u8 = undefined;
        for (0..E) |x| {
            const tr = try b.ck.get(try fullName(&f, "layers.{d}.mlp.experts.{d}.{s}.trellis", .{ li, x, m.proj }));
            if (!std.mem.eql(u8, tr.dtype, "I16") or tr.shape.len != 3 or tr.shape[2] != 64 or tr.shape[0] != k_all / 16 or tr.shape[1] != n_all / 16) return error.NotExl3Mcg4;
            if (!b.ck.has(try fullName(&f, "layers.{d}.mlp.experts.{d}.{s}.mcg", .{ li, x, m.proj }))) return error.NotExl3Mcg4;
            const dst = words[x * per ..][0..per];
            if (m.by_cols) { // every tile row, this rank's tile columns
                const all = try b.ck.readAll(b.gpa, try fullName(&f, "layers.{d}.mlp.experts.{d}.{s}.trellis", .{ li, x, m.proj }));
                defer b.gpa.free(all);
                const row = (n_all / 16) * tile;
                for (0..m.kt) |r| @memcpy(dst[r * m.nt * tile ..][0 .. m.nt * tile], all[r * row + b.rank * m.nt * tile ..][0 .. m.nt * tile]);
            } else try b.ck.readAt(tr, b.rank * per, dst); // this rank's tile rows: one contiguous part
            const se = try b.ck.get(try fullName(&f, "layers.{d}.mlp.experts.{d}.{s}.suh", .{ li, x, m.proj }));
            const ve = try b.ck.get(try fullName(&f, "layers.{d}.mlp.experts.{d}.{s}.svh", .{ li, x, m.proj }));
            if (!std.mem.eql(u8, se.dtype, "F16") or !std.mem.eql(u8, ve.dtype, "F16")) return error.NotExl3Mcg4;
            const su = std.mem.sliceAsBytes(suh[x * m.kt * 16 ..][0 .. m.kt * 16]);
            const sv = std.mem.sliceAsBytes(svh[x * m.nt * 16 ..][0 .. m.nt * 16]);
            try b.ck.readAt(se, if (m.by_cols) 0 else b.rank * su.len, su);
            try b.ck.readAt(ve, if (m.by_cols) b.rank * sv.len else 0, sv);
            if (m.by_cols) { // gate and up: one input rotation shared by every expert of the layer? (exl3_shx)
                if (std.mem.eql(u8, sh, "g") and x == 0) @memcpy(b.suh_first[0..su.len], su) else if (!std.mem.eql(u8, su, b.suh_first[0..su.len])) b.exl3_shx = false;
            }
        }
        try b.put(tn, .i32, &tshape, words);
        try b.putf("{s}.suh_{s}", .{ out, sh }, .f16, &sus, std.mem.sliceAsBytes(suh));
        try b.putf("{s}.svh_{s}", .{ out, sh }, .f16, &svs, std.mem.sliceAsBytes(svh));
    }
}

/// Each group of 64 re-quantized to ``bits`` symmetric levels (least-squares clip search), stored as 4-bit codes + 8.
fn simulateBits(t: Triple, bits: u8) void {
    const Work = struct {
        fn run(tt: Triple, nb: u8, e0: usize, e1: usize) void {
            const lo: f32 = -@as(f32, @floatFromInt(@as(u32, 1) << @intCast(nb - 1)));
            const hi: f32 = -lo - 1;
            const wpg = 8; // 4-bit words a group of 64
            for (e0..e1) |e| for (0..tt.n) |r| for (0..tt.kg) |g| {
                const at = (e * tt.n + r) * tt.kg + g;
                const words = tt.words[(e * tt.n + r) * tt.k8 + g * wpg ..][0..wpg];
                const s0 = bf16Value(tt.scales[at]);
                var v: [64]f32 = undefined;
                var amax: f32 = 0;
                for (0..64) |i| {
                    const q: i32 = @intCast((words[i / 8] >> @intCast(4 * (i % 8))) & 15);
                    v[i] = @as(f32, @floatFromInt(q - 8)) * s0;
                    amax = @max(amax, @abs(v[i]));
                }
                if (amax == 0) continue;
                var best_s: f32 = amax / -lo;
                var best_e: f32 = std.math.inf(f32);
                for ([_]f32{ 1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6 }) |c| {
                    const sc: f32 = bf16Value(toBf16(c * amax / -lo));
                    if (sc == 0) continue;
                    var err: f32 = 0;
                    for (v) |x| {
                        const q = @max(lo, @min(hi, @round(x / sc)));
                        err += (q * sc - x) * (q * sc - x);
                    }
                    if (err < best_e) {
                        best_e = err;
                        best_s = sc;
                    }
                }
                for (words) |*w| w.* = 0;
                for (v, 0..) |x, i| {
                    const q: i32 = @intFromFloat(@max(lo, @min(hi, @round(x / best_s))));
                    words[i / 8] |= @as(u32, @intCast(q + 8)) << @intCast(4 * (i % 8));
                }
                tt.scales[at] = toBf16(best_s);
                tt.biases[at] = toBf16(-8 * best_s);
            };
        }
    };
    const threads = 16;
    var pool: [threads]?std.Thread = @splat(null);
    const per = (t.e + threads - 1) / threads;
    for (0..threads) |i| {
        const e0 = i * per;
        const e1 = @min(t.e, e0 + per);
        if (e0 >= e1) break;
        pool[i] = std.Thread.spawn(.{}, Work.run, .{ t, bits, e0, e1 }) catch blk: {
            Work.run(t, bits, e0, e1);
            break :blk null;
        };
    }
    for (pool) |th| if (th) |x| x.join();
}

fn toBf16(x: f32) u16 {
    const u: u32 = @bitCast(x);
    return @truncate((u + 0x7fff + ((u >> 16) & 1)) >> 16); // round to nearest even
}
