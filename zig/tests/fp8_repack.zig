// ! tf-fp8-repack: one EXL3 projection's trellis decoded to W_q [K, N] (or, with TF_REPACK_ROT=both, to the ! fully
// pre-rotated W_target = diag(suh) H_K W_q H_N diag(svh)), requantized per output row, and packed in the ! fp8
// dense body's 16-k x 64-n block order. A bench tool: no engine path reads this format yet. ! Usage: tf-fp8-repack
// PACK PREFIX OUT_BASE -> OUT_BASE.fp8, OUT_BASE.scale, OUT_BASE.meta (+ .trellis control). ! TF_REPACK_INT8=1
// per-row int8 instead of E4M3. ! TF_REPACK_ROT=both bake both Hadamards and suh/svh into the weight, so the
// runtime needs no transform pass. !
const std = @import("std");
const tf = @import("tensorfold");
const pk = tf.deepseek_v41_pack;
const exl3 = tf.exl3_format;
const codec = tf.exlr8.codec;

fn layerCheck(gpa: std.mem.Allocator, k: usize, n: usize, int8: bool, ref: []const f64, bytes: []const u8, scales: []const f32, rot: bool) !void {
    const what = try gpa.alloc(f64, k * n);
    defer gpa.free(what);
    const nb64 = n / 64;
    const nb = scales.len / n;
    const bsize = k / nb;
    for (0..nb) |b| {
        for (0..n) |nn| {
            const s: f64 = @floatCast(scales[b * n + nn]);
            for (b * bsize..(b + 1) * bsize) |kk| {
                const at = ((kk / 16) * nb64 + nn / 64) * 1024 + (kk % 16) * 64 + (nn % 64);
                const code = bytes[at];
                const q: f64 = if (int8) @floatFromInt(@as(i8, @bitCast(code))) else @floatCast(codec.e4m3Value(code));
                what[kk * n + nn] = q * s;
            }
        }
    }
    const m: usize = 64;
    const x = try gpa.alloc(f64, m * k);
    defer gpa.free(x);
    for (x, 0..) |*v, i| v.* = @as(f64, @floatFromInt(@as(i32, @intCast((i *% 2654435761) % 2001)) - 1000)) / 1000.0;
    const yref = try gpa.alloc(f64, m * n);
    defer gpa.free(yref);
    const yhat = try gpa.alloc(f64, m * n);
    defer gpa.free(yhat);
    for (0..m) |r| {
        @memset(yref[r * n ..][0..n], 0);
        @memset(yhat[r * n ..][0..n], 0);
        for (0..k) |kk| {
            const xv = x[r * k + kk];
            if (xv == 0) continue;
            const wrow = ref[kk * n ..][0..n];
            const hrow = what[kk * n ..][0..n];
            for (0..n) |nn| {
                yref[r * n + nn] += xv * wrow[nn];
                yhat[r * n + nn] += xv * hrow[nn];
            }
        }
    }
    var se: f64 = 0;
    var sb: f64 = 0;
    for (yref, yhat) |a, b| {
        const d = b - a;
        se += d * d;
        sb += a * a;
    }
    var agree: usize = 0;
    for (0..m) |r| {
        var br: usize = 0;
        var bh: usize = 0;
        for (1..n) |c| {
            if (yref[r * n + c] > yref[r * n + br]) br = c;
            if (yhat[r * n + c] > yhat[r * n + bh]) bh = c;
        }
        if (br == bh) agree += 1;
    }
    std.debug.print("  check rot={s}: NRMS {e:.4}, per-row argmax {d}/{d} = {d:.2}%\n", .{
        if (rot) "yes" else "none",                                          std.math.sqrt(se / sb), agree, m,
        100.0 * @as(f64, @floatFromInt(agree)) / @as(f64, @floatFromInt(m)),
    });
}

/// NRMS of a reconstructed weight's GEMM against the FP64 target on the fixed 64-row activation.
fn gemmNrms(gpa: std.mem.Allocator, ref: []const f64, what: []const f64, k: usize, n: usize) !f64 {
    const m: usize = 64;
    const x = try gpa.alloc(f64, m * k);
    defer gpa.free(x);
    for (x, 0..) |*v, i| v.* = @as(f64, @floatFromInt(@as(i32, @intCast((i *% 2654435761) % 2001)) - 1000)) / 1000.0;
    const yref = try gpa.alloc(f64, m * n);
    defer gpa.free(yref);
    const yhat = try gpa.alloc(f64, m * n);
    defer gpa.free(yhat);
    for (0..m) |r| {
        @memset(yref[r * n ..][0..n], 0);
        @memset(yhat[r * n ..][0..n], 0);
        for (0..k) |kk| {
            const xv = x[r * k + kk];
            if (xv == 0) continue;
            const wrow = ref[kk * n ..][0..n];
            const hrow = what[kk * n ..][0..n];
            for (0..n) |nn| {
                yref[r * n + nn] += xv * wrow[nn];
                yhat[r * n + nn] += xv * hrow[nn];
            }
        }
    }
    var se: f64 = 0;
    var sb: f64 = 0;
    for (yref, yhat) |a, b| {
        const d = b - a;
        se += d * d;
        sb += a * a;
    }
    return std.math.sqrt(se / sb);
}

/// Price a finer quantization grid without changing the pack: one scale per output row per `block` k-rows.
fn blockCheck(gpa: std.mem.Allocator, k: usize, n: usize, int8: bool, ref: []const f64, block: usize, half: bool) !void {
    const nb = (k + block - 1) / block;
    const scales = try gpa.alloc(f32, n * nb);
    defer gpa.free(scales);
    const what = try gpa.alloc(f64, k * n);
    defer gpa.free(what);
    const denom: f64 = if (int8) 127.0 else 448.0;
    for (0..n) |nn| {
        for (0..nb) |b| {
            var amax: f64 = 0;
            const lo = b * block;
            const hi = @min(k, lo + block);
            for (lo..hi) |kk| amax = @max(amax, @abs(ref[kk * n + nn]));
            const s: f64 = if (amax > 0) amax / denom else 1.0;
            scales[nn * nb + b] = @floatCast(s);
            for (lo..hi) |kk| {
                const x: f64 = ref[kk * n + nn] / s;
                const code: u8 = if (int8) blk: {
                    const q: f64 = @round(@max(-127.0, @min(127.0, x)));
                    const ci: i8 = @intFromFloat(q);
                    break :blk @bitCast(ci);
                } else codec.e4m3FromF32(@floatCast(x));
                const q: f64 = if (int8) @floatFromInt(@as(i8, @bitCast(code))) else @floatCast(codec.e4m3Value(code));
                if (half) {
                    const qh: f16 = @floatFromInt(@as(i8, @bitCast(code)));
                    const sh: f16 = @floatCast(s);
                    const ph: f16 = @floatCast(@as(f32, @floatCast(qh)) * @as(f32, @floatCast(sh)));
                    what[kk * n + nn] = @floatCast(ph);
                } else what[kk * n + nn] = q * s;
            }
        }
    }
    std.debug.print("  block {d}{s}: NRMS {e:.4}\n", .{ block, if (half) " half" else "", try gemmNrms(gpa, ref, what, k, n) });
}

/// The 3-bit affine requantization's layer price on a real routed-expert projection. Per output column, per 64-k
/// group, eight uniform levels spanning the group's [min, max] (scale + bias, the quality- viable form the fair
/// `tf_affine_tile3s_nax` body dispatches), against the decoded EXL3 weight as the FP64 target: the NRMS the model
/// band would see from this projection alone.
fn affine3Check(gpa: std.mem.Allocator, k: usize, n: usize, ref: []const f64) !void {
    const G: usize = 64;
    const nb = k / G;
    const what = try gpa.alloc(f64, k * n);
    defer gpa.free(what);
    for (0..n) |nn| {
        for (0..nb) |b| {
            var lo: f64 = 1e30;
            var hi: f64 = -1e30;
            for (b * G..(b + 1) * G) |kk| {
                const v = ref[kk * n + nn];
                lo = @min(lo, v);
                hi = @max(hi, v);
            }
            const range = hi - lo;
            const sc: f64 = if (range > 1e-12) range / 7.0 else 1.0;
            for (b * G..(b + 1) * G) |kk| {
                const q = @round((ref[kk * n + nn] - lo) / sc);
                const qi: i64 = @intFromFloat(@max(0.0, @min(7.0, q)));
                what[kk * n + nn] = @as(f64, @floatFromInt(qi)) * sc + lo;
            }
        }
    }
    const nr = try gemmNrms(gpa, ref, what, k, n);
    std.debug.print("  affine3 group-64 scale+bias: NRMS {e:.4} (GEMM vs the decoded EXL3 target)\n", .{nr});
}

pub fn main(init: std.process.Init) !u8 {
    const gpa = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 4) {
        std.debug.print("usage: tf-fp8-repack PACK PREFIX OUT_BASE\n", .{});
        return 2;
    }
    const dir = args[1];
    const prefix = args[2];
    const out_base = args[3];
    const int8 = std.c.getenv("TF_REPACK_INT8") != null;
    // TF_REPACK_CLIP scales the per-row max grid (1.0 = max/127).
    const clip: f64 = if (std.c.getenv("TF_REPACK_CLIP")) |v| (std.fmt.parseFloat(f64, std.mem.span(v)) catch 1.0) else 1.0;
    const block: usize = if (std.c.getenv("TF_REPACK_BLOCK")) |v| (std.fmt.parseInt(usize, std.mem.span(v), 10) catch 0) else 0;
    const rot_env = std.c.getenv("TF_REPACK_ROT");
    const rot_both = rot_env != null and std.mem.eql(u8, std.mem.span(rot_env.?), "both");
    const rot_n = rot_env != null and std.mem.eql(u8, std.mem.span(rot_env.?), "n");
    const rot_k = rot_env != null and std.mem.eql(u8, std.mem.span(rot_env.?), "k");
    const rot_any = rot_both or rot_n or rot_k;
    const check = std.c.getenv("TF_REPACK_CHECK") != null;
    var pack = try pk.Pack.open(gpa, io, dir);
    defer pack.deinit();
    const proj = try pack.projection(prefix);
    const g = switch (proj) {
        .exl3 => |e| e,
        .dense => return error.UnsupportedDenseProjection,
        else => return error.UnsupportedProjection,
    };
    if (g.scales.format != .f16) return error.UnsupportedScales;
    const geom = g.geometry;
    const k: usize = geom.k();
    const n: usize = geom.n();
    const halves: u16 = @intCast(geom.k2());
    const pcb = try pack.codebook(io, prefix);
    const cb: exl3.Codebook = switch (pcb) {
        .inst3 => .inst3,
        .mcg => .mcg,
        .mul1 => .mul1,
    };

    // The trellis bytes and the codebook's values, exactly as the engine's body reads them.
    const tbytes: usize = @intCast(g.trellis.nbytes);
    const raw = try gpa.alloc(u8, tbytes);
    defer gpa.free(raw);
    try pack.read(io, g.trellis, .{ .offset = 0, .len = tbytes }, raw);
    const words: []const i16 = @alignCast(std.mem.bytesAsSlice(i16, raw));
    std.debug.print("repack {s}: K {d} N {d} halves {d} tiles {d}x{d} codebook {s} rot={s} block={d}\n", .{ prefix, k, n, halves, geom.k_tiles, geom.n_tiles, @tagName(cb), if (rot_both) "both" else if (rot_n) "n" else if (rot_k) "k" else "none", block });

    // The rotation scales the engine's rotate_in/rotate_out apply.
    const suh = try gpa.alloc(f16, k);
    defer gpa.free(suh);
    const svh = try gpa.alloc(f16, n);
    defer gpa.free(svh);
    try pack.read(io, g.scales.input, .{ .offset = 0, .len = k * 2 }, std.mem.sliceAsBytes(suh));
    try pack.read(io, g.scales.output, .{ .offset = 0, .len = n * 2 }, std.mem.sliceAsBytes(svh));

    // W_q [K, N], the unrotated domain the engine's trellis bodies decode.
    const wq = try gpa.alloc(f16, k * n);
    defer gpa.free(wq);
    exl3.unpack(.{ .halves = halves }, cb, words, geom.k_tiles, geom.n_tiles, wq);

    // The domain the pack is quantized in. Unrotated: W_q itself. Rotated: the full EXL3 weight the engine's
    // three passes compute, decoded by the format's own FP64 reference (dequantize = rotate K, *suh, rotate N, *svh).
    const ref = try gpa.alloc(f64, k * n);
    defer gpa.free(ref);
    if (rot_both) {
        const scratch = try gpa.alloc(f64, 2 * k * n);
        defer gpa.free(scratch);
        exl3.dequantize(.{ .halves = halves }, cb, words, geom.k_tiles, geom.n_tiles, suh, svh, ref, scratch);
    } else if (rot_any) {
        for (ref, wq) |*d, s| d.* = @floatCast(s);
        if (rot_k) {
            exl3.rotate(ref, k, n, 0);
            for (ref, 0..) |*v, i| v.* *= @floatCast(suh[i / n]);
        }
        if (rot_n) {
            exl3.rotate(ref, k, n, 1);
            for (ref, 0..) |*v, i| v.* *= @floatCast(svh[i % n]);
        }
    } else {
        for (ref, wq) |*d, s| d.* = @floatCast(s);
    }

    if (std.c.getenv("TF_REPACK_DUMP_REF32")) |path| {
        const dpath = std.mem.span(path);
        const out = try gpa.alloc(u8, 16 + ref.len * 4);
        defer gpa.free(out);
        std.mem.writeInt(u64, out[0..8], @intCast(k), .little);
        std.mem.writeInt(u64, out[8..16], @intCast(n), .little);
        const dst: [*]f32 = @ptrCast(@alignCast(out[16..].ptr));
        for (ref, 0..) |v, i| dst[i] = @floatCast(v);
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = dpath, .data = out });
        std.debug.print("  dumped ref32 {d}x{d} to {s}\n", .{ k, n, dpath });
    }

    if (std.c.getenv("TF_REPACK_AFFINE3") != null) {
        try affine3Check(gpa, k, n, ref);
        return 0;
    }
    // Per output row n: scale = max_k |W[k, n]| / 448; code = e4m3(W / scale); the tile order the kernel reads.
    // With the rotations baked in, this scale is also the svh scale (it multiplies the finished output row), so
    // the runtime epilogue has no transform and no scale pass.
    const bytes = try gpa.alloc(u8, k * n);
    defer gpa.free(bytes);
    const nb: usize = if (block != 0) k / block else 1;
    const bsize: usize = if (block != 0) block else k;
    if (block != 0 and (block % 128 != 0 or k % block != 0)) return error.BadBlock;
    const scales = try gpa.alloc(f32, n * nb);
    defer gpa.free(scales);
    const nb64 = n / 64;
    if (n % 64 != 0 or k % 16 != 0) return error.BadShape;
    var gmax: f64 = 0;
    for (0..nb) |b| {
        for (0..n) |nn| {
            var amax: f64 = 0;
            for (b * bsize..(b + 1) * bsize) |kk| amax = @max(amax, @abs(ref[kk * n + nn]));
            const denom: f64 = if (int8) 127.0 else 448.0;
            const s: f64 = if (amax > 0) (amax * clip) / denom else 1.0;
            scales[b * n + nn] = @floatCast(s);
            gmax = @max(gmax, amax);
            for (b * bsize..(b + 1) * bsize) |kk| {
                const x: f64 = ref[kk * n + nn] / s;
                const code: u8 = if (int8) blk: {
                    const q: f64 = @round(@max(-127.0, @min(127.0, x)));
                    const ci: i8 = @intFromFloat(q);
                    break :blk @bitCast(ci);
                } else codec.e4m3FromF32(@floatCast(x));
                bytes[((kk / 16) * nb64 + nn / 64) * 1024 + (kk % 16) * 64 + (nn % 64)] = code;
            }
        }
    }
    std.debug.print("  scale range: min {e:.3} max {e:.3}; global |W| max {e:.3}\n", .{ blk: {
        var mn: f32 = scales[0];
        for (scales) |s| mn = @min(mn, s);
        break :blk mn;
    }, blk: {
        var mx: f32 = 0;
        for (scales) |s| mx = @max(mx, s);
        break :blk mx;
    }, gmax });
    if (check) try layerCheck(gpa, k, n, int8, ref, bytes, scales, rot_both);
    if (std.c.getenv("TF_REPACK_BLOCKCHECK") != null) {
        try blockCheck(gpa, k, n, int8, ref, 128, false);
        try blockCheck(gpa, k, n, int8, ref, 16, false);
    }
    if (std.c.getenv("TF_REPACK_HALFCHECK") != null) try blockCheck(gpa, k, n, int8, ref, 128, true);

    const fp8_path = try std.fmt.allocPrint(gpa, "{s}.fp8", .{out_base});
    defer gpa.free(fp8_path);
    const scale_path = try std.fmt.allocPrint(gpa, "{s}.scale", .{out_base});
    defer gpa.free(scale_path);
    const meta_path = try std.fmt.allocPrint(gpa, "{s}.meta", .{out_base});
    defer gpa.free(meta_path);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = fp8_path, .data = bytes });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = scale_path, .data = std.mem.sliceAsBytes(scales) });
    const fmt_name = if (block != 0 and int8) "int8b" else if (int8) "int8" else "fp8";
    const meta = try std.fmt.allocPrint(gpa, "{d} {d} {d} {s} rot={s} block={d}\n", .{ k, n, halves, fmt_name, if (rot_both) "both" else if (rot_n) "n" else if (rot_k) "k" else "none", block });
    defer gpa.free(meta);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = meta_path, .data = meta });
    if (std.c.getenv("TF_REPACK_NOTRELLIS") == null) {
        const tr_path = try std.fmt.allocPrint(gpa, "{s}.trellis", .{out_base});
        defer gpa.free(tr_path);
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = tr_path, .data = raw });
        std.debug.print("  wrote {s} ({d} B)\n", .{ tr_path, raw.len });
    }
    std.debug.print("  wrote {s} ({d} B), {s} ({d} B), {s}\n", .{ fp8_path, bytes.len, scale_path, scales.len * 4, meta_path });
    return 0;
}
