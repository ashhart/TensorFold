//! exl3-draft: the MTP draft head rows of an EXL3 pack (cuda_exl3_mtp.draftRows) against the full EXL3 lm_head.

const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const flash = @import("flashnext");
const check = @import("../../check.zig");
const st = core.safetensors;
const exl3 = cuda.exl3;
const fmt = core.exl3_format;
const mtp3 = flash.exl3_mtp;
const Gpu = check.Gpu;

fn bfVal(b: u16) f64 {
    return @floatCast(@as(f32, @bitCast(@as(u32, b) << 16)));
}

fn bf16Bits(v: f32) u16 {
    return flash.ple.toBf16(v);
}

pub fn run(gpu: Gpu, rest: []const [:0]const u8) !void {
    if (rest.len < 1) return error.MissingArgument;
    const dir = rest[0];
    const threads: usize = if (rest.len > 1) try std.fmt.parseInt(usize, rest[1], 10) else mtp3.loadThreads();
    var arena_state = std.heap.ArenaAllocator.init(gpu.gpa);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const io = gpu.io;
    const path = try std.fs.path.join(a, &.{ dir, "model-00007-of-00007.safetensors" });
    var file = try st.File.open(gpu.gpa, io, path);
    defer file.close(io);
    const tr = file.get("lm_head.trellis") orelse return error.MissingTensor;
    const su = file.get("lm_head.suh") orelse return error.MissingTensor;
    const sv = file.get("lm_head.svh") orelse return error.MissingTensor;
    const head = try mtp3.Head.of(tr, su, sv, file.get("lm_head.mul1"));
    const k = head.k;
    const n = head.n;
    std.debug.print("lm_head: K {d}, N {d}, halves {d} ({d}-bit), {d} MB trellis\n", .{ k, n, head.halves, head.halves / 2, tr.bytes.len >> 20 });

    const ids = try flash.mtp.draftIds(a);
    const rows = try a.alloc(u16, ids.len * k);
    const t0 = std.Io.Clock.awake.now(io).toNanoseconds();
    try mtp3.draftRows(gpu.gpa, head, ids, rows, threads);
    const ms = @as(f64, @floatFromInt(std.Io.Clock.awake.now(io).toNanoseconds() - t0)) / 1e6;
    std.debug.print("draft rows: {d} ids in {d} column blocks, {d} threads, {d:.0} ms (host, ReleaseFast)\n", .{ ids.len, blk: {
        var c: usize = 0;
        var prev: usize = std.math.maxInt(usize);
        for (ids) |id| {
            if (id / 128 != prev) c += 1;
            prev = id / 128;
        }
        break :blk c;
    }, threads, ms });

    // x: a real embedding row (bf16) and a random bf16 row
    const embed_path = try std.fs.path.join(a, &.{ dir, "model-00001-of-00007.safetensors" });
    var ef = try st.File.open(gpu.gpa, io, embed_path);
    defer ef.close(io);
    const emb = ef.get("model.language_model.embed_tokens.weight") orelse return error.MissingTensor;
    const xs_bits = try a.alloc(u16, 2 * k);
    for (0..k) |i| xs_bits[i] = std.mem.readInt(u16, emb.bytes[(9338 * k + i) * 2 ..][0..2], .little);
    var prng = std.Random.DefaultPrng.init(0xd1a);
    for (xs_bits[k..]) |*b| b.* = bf16Bits((prng.random().float(f32) - 0.5) * 2);

    // full head on the GPU, fp32 out
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var de = try exl3.Dense.load(gpu.d);
    defer de.unload();
    const strides = exl3.strides(k, n, head.halves, false);
    const trellis_al = try a.alloc(i16, tr.bytes.len / 2);
    @memcpy(std.mem.sliceAsBytes(trellis_al), tr.bytes);
    var d_tr = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(trellis_al));
    defer d_tr.free();
    var d_su = try cuda.DeviceBuffer.fromHost(gpu.d, su.bytes);
    defer d_su.free();
    var d_sv = try cuda.DeviceBuffer.fromHost(gpu.d, sv.bytes);
    defer d_sv.free();
    const wgt: exl3.Weight = .{ .trellis = d_tr.ptr, .suh = d_su.ptr, .svh = d_sv.ptr, .k = @intCast(k), .n = @intCast(n), .halves = head.halves, .codebook = .mul1, .stride_k = strides.k, .stride_nb = strides.nb };
    var d_x = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(xs_bits));
    defer d_x.free();
    var d_y = try cuda.DeviceBuffer.alloc(gpu.d, 2 * n * 4);
    defer d_y.free();
    try gpu.ctx.synchronize();
    try de.decode(s, wgt, d_x.ptr, .bf16, 2, d_y.ptr, .f32);
    try s.synchronize();
    const full = try a.alloc(f32, 2 * n);
    try d_y.download(0, std.mem.sliceAsBytes(full));

    // exact float64 for the first x (the exl3-pack test's `exact` product, one row)
    const exact = try a.alloc(f64, n);
    {
        const tiles_k = k / 16;
        const tiles_n = n / 16;
        const tw: []const i16 = trellis_al;
        const wq = try a.alloc(f16, k * n);
        fmt.unpack(.{ .halves = head.halves }, .mul1, tw, tiles_k, tiles_n, wq);
        const xr = try a.alloc(f64, k);
        for (xr, 0..) |*v, i| v.* = bfVal(xs_bits[i]) * @as(f64, @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, su.bytes[2 * i ..][0..2], .little)))));
        fmt.rotate(xr, 1, k, 1);
        @memset(exact, 0);
        for (0..k) |ki| {
            const xv = xr[ki];
            for (exact, wq[ki * n ..][0..n]) |*o, wv| o.* += xv * @as(f64, @floatCast(wv));
        }
        fmt.rotate(exact, 1, n, 1);
        for (exact, 0..) |*v, c| v.* *= @as(f64, @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, sv.bytes[2 * c ..][0..2], .little)))));
    }

    for (0..2) |xi| {
        const x = xs_bits[xi * k ..][0..k];
        var worst_gpu: f64 = 0; // in units of the bf16 bound
        var worst_exact: f64 = 0;
        var max_abs_gpu: f64 = 0;
        var peak: f64 = 0;
        var arg_draft: usize = 0;
        var arg_full: usize = 0;
        var best_draft = -std.math.inf(f64);
        var best_full = -std.math.inf(f64);
        var gpu_exact_rel: f64 = 0;
        var peak_full: f64 = 0;
        for (full[xi * n ..][0..n]) |v| peak_full = @max(peak_full, @abs(@as(f64, v)));
        for (ids, 0..) |id, i| {
            const row = rows[i * k ..][0..k];
            var got: f64 = 0;
            var mag: f64 = 0;
            for (row, x) |wb, xb| {
                const wv = bfVal(wb);
                got += wv * bfVal(xb);
                mag += @abs(wv * bfVal(xb));
            }
            const bound = mag / 512.0 * (1.0 + 1.0 / 128.0) + 1e-9; // |W - bf16(W)| <= 2^-9 |bf16(W)| (1 + 2^-7)
            const g: f64 = full[xi * n + id];
            // plus the kernel's fp16-input error, measured at 1.75e-4 - 2.4e-4 of the row's peak (exl3-pack)
            const tol_gpu = bound + 5e-4 * peak_full;
            worst_gpu = @max(worst_gpu, @abs(got - g) / tol_gpu);
            max_abs_gpu = @max(max_abs_gpu, @abs(got - g));
            peak = @max(peak, @abs(g));
            if (got > best_draft) {
                best_draft = got;
                arg_draft = id;
            }
            if (g > best_full) {
                best_full = g;
                arg_full = id;
            }
            if (xi == 0) {
                worst_exact = @max(worst_exact, @abs(got - exact[id]) / bound);
                gpu_exact_rel = @max(gpu_exact_rel, @abs(g - exact[id]));
            }
        }
        std.debug.print("x {s}: draft vs full head (GPU Dense fp32): max |diff| {e:.3} at draft peak {d:.3} (row peak {d:.3}), worst diff / tolerance {d:.3}; argmax over draft ids: draft {d}, full {d}\n", .{ if (xi == 0) "embed row 9338" else "random", max_abs_gpu, peak, peak_full, worst_gpu, arg_draft, arg_full });
        if (xi == 0) std.debug.print("  draft vs exact float64: worst diff / bf16 bound {d:.3}; GPU fp32 vs exact max |diff| {e:.3}\n", .{ worst_exact, gpu_exact_rel });
        try check.expect(worst_gpu <= 1.0, "draft logits off the full EXL3 head beyond bf16 weight rounding", .{});
        if (xi == 0) try check.expect(worst_exact <= 1.0, "draft logits off the exact product beyond the bf16 bound", .{});
    }
    check.pass("EXL3 draft head: {d} rows dequantised from lm_head; every draft logit equals the full head's (GPU Dense fp32 and exact f64) within bf16 weight rounding", .{ids.len});
}
