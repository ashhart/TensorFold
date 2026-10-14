//! The KDA attention block on real tensors: qkv and the bf16 pair, the chunked chain and its cut, o_proj's partials.

const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const quant_ref_tests = @import("quant_ref_tests.zig");
const exl3_tests = @import("exl3_tests.zig");
const ref = @import("glm_kda_ref.zig");
const base = @import("glm_kda_tests.zig");
const exl3 = cuda.exl3;
const exl3_load = @import("core").exl3_load;
const exl3_upload = cuda.exl3_upload;
const bmm_mod = cuda.glm_bmm;
const chunked_mod = cuda.glm_kda_chunk;

const Run = exl3_tests.Run;
const Rig = quant_ref_tests.RigT;

/// Three [start, end) runs: the fused q|k|v group's per-rank trellis columns and svh rows.
fn runs(v: std.json.Value) ![3][2]usize {
    if (v.array.items.len != 3) return error.BadRuns;
    var out: [3][2]usize = undefined;
    for (v.array.items, 0..) |item, i| {
        out[i][0] = @intCast(item.array.items[0].integer);
        out[i][1] = @intCast(item.array.items[1].integer);
    }
    return out;
}

fn groupBySuffix(scan: *exl3_load.Scan, suffix: []const u8) !usize {
    for (scan.groups, 0..) |grp, i| if (std.mem.endsWith(u8, grp.prefix, suffix)) return i;
    return error.GroupMissing;
}

/// The qkv group's trellis: the three runs' columns stacked back into one [K/16, n/16, words] tensor.
fn slicedTrellis(a: std.mem.Allocator, t: exl3_load.Parts, cols: [3][2]usize) ![]u8 {
    const wcol = t.trellis.dim(2) * 2;
    const full_w = t.trellis.dim(1);
    var width: usize = 0;
    for (cols) |rn| width += rn[1] - rn[0];
    const out = try a.alloc(u8, t.trellis.dim(0) * width * wcol);
    for (0..t.trellis.dim(0)) |row| {
        var dst: usize = 0;
        for (cols) |rn| {
            const len = (rn[1] - rn[0]) * wcol;
            const src = (row * full_w + rn[0]) * wcol;
            @memcpy(out[row * width * wcol + dst ..][0..len], t.trellis.bytes[src..][0..len]);
            dst += len;
        }
    }
    return out;
}

/// The qkv group's svh: the three runs' rows stacked back into one [n] fp16 scale row.
fn slicedRows(a: std.mem.Allocator, t: exl3_load.Parts, rows: [3][2]usize) ![]u8 {
    var total: usize = 0;
    for (rows) |rn| total += rn[1] - rn[0];
    const out = try a.alloc(u8, total * 2);
    var dst: usize = 0;
    for (rows) |rn| {
        const len = (rn[1] - rn[0]) * 2;
        @memcpy(out[dst..][0..len], t.svh.bytes[rn[0] * 2 ..][0..len]);
        dst += len;
    }
    return out;
}

pub fn block(r: *Rig, state: *Run, dir: []const u8) !void {
    if (std.mem.eql(u8, dir, "none")) {
        base.skip("glm.kda.block");
        return;
    }
    var man = try ref.Manifest.open(r.a, r.gpu.io, dir);
    defer man.deinit();
    _ = state;
    const o = man.kda() orelse {
        base.skip("glm.kda.block");
        return;
    };
    const q = try base.Params.read(o);
    const model_dir = ref.str(man.root(), "model");
    const e = o.get("block").?.object;
    const rows: u32 = @intCast(ref.int(e, "rows"));
    const cut: u32 = @intCast(ref.int(e, "cut"));
    const qkv_o = e.get("qkv").?.object;
    const qkv_n: usize = @intCast(ref.int(qkv_o, "n"));

    var scan = try exl3_load.scan(r.gpu.gpa, r.gpu.io, model_dir);
    defer scan.deinit();
    var up = exl3_upload.Uploader{ .d = r.gpu.d, .gpa = r.gpu.gpa };
    defer up.deinit();
    var dense = try exl3.Dense.load(r.gpu.d);
    defer dense.unload();

    // the fused q|k|v group: the rank's three runs stacked back into one [6912, 4096] linear
    const gi = try groupBySuffix(&scan, ref.str(qkv_o, "prefix"));
    const g = scan.groups[gi];
    const parts = try scan.parts(r.a, gi);
    const t_cols = try runs(qkv_o.get("t_runs").?);
    const s_rows = try runs(qkv_o.get("s_runs").?);
    var t2 = parts.trellis;
    t2.bytes = try slicedTrellis(r.a, parts, t_cols);
    var s2 = parts.svh;
    s2.bytes = try slicedRows(r.a, parts, s_rows);
    const w = try up.dense(.{ .trellis = t2, .suh = parts.suh, .svh = s2, .bias = parts.bias, .suh_packed = parts.suh_packed }, .{ .k = @as(usize, @intCast(ref.int(qkv_o, "k"))), .n = qkv_n, .halves = @as(u32, @intCast(g.halves)), .codebook = g.codebook });
    try check.expect(w.k == @as(u32, @intCast(ref.int(qkv_o, "k"))) and w.n == @as(u32, @intCast(qkv_n)), "glm.kda.block: qkv weight shape", .{});

    const normed = try r.dev(try ref.readFile(r.a, r.gpu.io, dir, ref.str(e, "normed")));
    const qkv_tmp = try r.zeros(rows * qkv_n * 2);
    try dense.prompt(r.s, w, normed, .bf16, rows, qkv_tmp, .bf16);

    // the plain pair and b: bf16 [n, k] exactly as the engine holds them
    var bmm = try bmm_mod.Bmm.load(r.gpu.d);
    defer bmm.unload();
    const mms = e.get("mms").?.object;
    const fa_w = try mm16(r, dir, mms, "fa");
    const ga_w = try mm16(r, dir, mms, "ga");
    const fb_w = try mm16(r, dir, mms, "fb");
    const gb_w = try mm16(r, dir, mms, "gb");
    const b_w = try mm16(r, dir, mms, "b");

    const fa_tmp = try r.zeros(rows * @as(usize, fa_w.n) * 2);
    const ga_tmp = try r.zeros(rows * @as(usize, ga_w.n) * 2);
    const b_tmp = try r.zeros(rows * @as(usize, b_w.n) * 2);
    const fa_part = try r.zeros(bmm_mod.Bmm.partBytes(fa_w, rows));
    const ga_part = try r.zeros(bmm_mod.Bmm.partBytes(ga_w, rows));
    const b_part = try r.zeros(bmm_mod.Bmm.partBytes(b_w, rows));
    try bmm.matmul(r.s, fa_w, normed, @intCast(fa_w.k), rows, fa_tmp, false, fa_part);
    try bmm.matmul(r.s, ga_w, normed, @intCast(ga_w.k), rows, ga_tmp, false, ga_part);
    try bmm.matmul(r.s, b_w, normed, @intCast(b_w.k), rows, b_tmp, false, b_part);
    const a_rows: usize = @as(usize, fb_w.n);
    const a_buf = try r.zeros(rows * a_rows * 2);
    const fa_part2 = try r.zeros(bmm_mod.Bmm.partBytes(fb_w, rows));
    try bmm.matmul(r.s, fb_w, fa_tmp, @intCast(fa_w.n), rows, a_buf, false, fa_part2);
    const g_buf = try r.zeros(rows * a_rows * 2);
    const ga_part2 = try r.zeros(bmm_mod.Bmm.partBytes(gb_w, rows));
    try bmm.matmul(r.s, gb_w, ga_tmp, @intCast(ga_w.n), rows, g_buf, false, ga_part2);

    // p's rows: q|k|v | f_a | g_a | b; the fixture's p holds the assembled bits
    const stride: usize = @intCast(q.p_stride);
    const p_bytes = rows * stride * 2;
    const p_buf = try r.zeros(p_bytes);
    const pbuf: cuda.DeviceBuffer = .{ .d = r.gpu.d, .ptr = p_buf, .len = p_bytes };
    for (0..rows) |ri| {
        try pbuf.copyFrom(ri * stride * 2, qkv_tmp + ri * qkv_n * 2, qkv_n * 2, r.s.handle);
        try pbuf.copyFrom(ri * stride * 2 + qkv_n * 2, fa_tmp + ri * @as(usize, fa_w.n) * 2, @as(usize, fa_w.n) * 2, r.s.handle);
        try pbuf.copyFrom(ri * stride * 2 + (qkv_n + 128) * 2, ga_tmp + ri * @as(usize, ga_w.n) * 2, @as(usize, ga_w.n) * 2, r.s.handle);
        try pbuf.copyFrom(ri * stride * 2 + @as(usize, @intCast(q.b_off)) * 2, b_tmp + ri * @as(usize, b_w.n) * 2, @as(usize, b_w.n) * 2, r.s.handle);
    }
    const want_p = try ref.readFile(r.a, r.gpu.io, dir, ref.str(e, "p"));
    // the chunked chain over the assembled p, and its cut
    try check.sameBytes("glm.kda.block p", try r.back(p_buf, p_bytes), want_p);
    var chunk = try chunked_mod.Chunk.load(r.gpu.d);
    defer chunk.unload();
    const a_log = try r.dev(try ref.readFile(r.a, r.gpu.io, dir, "kda_a_log.bin"));
    const dt_bias = try r.dev(try ref.readFile(r.a, r.gpu.io, dir, "kda_dt_bias.bin"));
    const norm_w = try r.dev(try ref.readFile(r.a, r.gpu.io, dir, "kda_norm.bin"));
    const conv_w = try r.dev(try ref.readFile(r.a, r.gpu.io, dir, "kda_conv.bin"));
    const conv0 = try r.dev(try ref.readFile(r.a, r.gpu.io, dir, ref.str(e, "conv0")));
    const state0 = try r.dev(try ref.readFile(r.a, r.gpu.io, dir, ref.str(e, "state0")));
    const want_out = try ref.readFile(r.a, r.gpu.io, dir, ref.str(e, "out"));
    const want_state = try ref.readFile(r.a, r.gpu.io, dir, ref.str(e, "state_out"));
    const out = try r.zeros(want_out.len);
    const st = try r.zeros(want_state.len);
    try chunk.chain(&r.s, q.h, p_buf, q.p_stride, @as(i32, @intCast(q.b_off)), a_buf, @as(i64, @intCast(q.per)), g_buf, @as(i64, @intCast(q.per)), conv0, conv_w, state0, a_log, dt_bias, norm_w, q.eps, q.lower, rows, 0, out, st);
    try check.sameBytes("glm.kda.block out", try r.back(out, want_out.len), want_out);
    try check.sameBytes("glm.kda.block state", try r.back(st, want_state.len), want_state);

    const cut_out1 = try r.zeros(@as(usize, cut) * q.per * 2);
    const cut_st1 = try r.zeros(want_state.len);
    try chunk.chain(&r.s, q.h, p_buf, q.p_stride, @as(i32, @intCast(q.b_off)), a_buf, @as(i64, @intCast(q.per)), g_buf, @as(i64, @intCast(q.per)), conv0, conv_w, state0, a_log, dt_bias, norm_w, q.eps, q.lower, cut, 0, cut_out1, cut_st1);
    const ch: usize = q.per * 3;
    const conv_cut = try r.a.alloc(u8, 3 * ch * 2);
    for (0..3) |ri| @memcpy(conv_cut[ri * ch * 2 ..][0 .. ch * 2], want_p[(cut - 3 + ri) * stride * 2 ..][0 .. ch * 2]);
    const cut_out2 = try r.zeros((@as(usize, rows) - cut) * q.per * 2);
    const cut_st2 = try r.zeros(want_state.len);
    try chunk.chain(&r.s, q.h, p_buf + @as(u64, cut) * @as(u64, stride) * 2, q.p_stride, @as(i32, @intCast(q.b_off)), a_buf + @as(u64, cut) * @as(u64, q.per) * 2, @as(i64, @intCast(q.per)), g_buf + @as(u64, cut) * @as(u64, q.per) * 2, @as(i64, @intCast(q.per)), try r.dev(conv_cut), conv_w, cut_st1, a_log, dt_bias, norm_w, q.eps, q.lower, rows - cut, 0, cut_out2, cut_st2);
    const got_c1 = try r.back(cut_out1, @as(usize, cut) * q.per * 2);
    const got_c2 = try r.back(cut_out2, (@as(usize, rows) - cut) * q.per * 2);
    const cat = try r.a.alloc(u8, got_c1.len + got_c2.len);
    @memcpy(cat[0..got_c1.len], got_c1);
    @memcpy(cat[got_c1.len..], got_c2);
    try check.sameBytes("glm.kda.block cut rows", cat, want_out);
    try check.sameBytes("glm.kda.block cut state", try r.back(cut_st2, want_state.len), want_state);

    // o_proj: the engine's fp32 partials (no rounding), through the group or the plain weight
    const o_o = e.get("o_proj").?.object;
    const want_part = try ref.readFile(r.a, r.gpu.io, dir, ref.str(e, "part"));
    const part = try r.zeros(want_part.len);
    if (o_o.get("w") == null) {
        const gi_o = try groupBySuffix(&scan, ref.str(o_o, "prefix"));
        const go = scan.groups[gi_o];
        const parts_o = try scan.parts(r.a, gi_o);
        const half = parts_o.trellis.dim(0) / 2;
        const suh_half = parts_o.suh.dim(0) / 2;
        var t3 = parts_o.trellis;
        t3.bytes = parts_o.trellis.bytes[0 .. half * t3.dim(1) * t3.dim(2) * 2];
        var suh3 = parts_o.suh;
        suh3.bytes = parts_o.suh.bytes[0 .. suh_half * 2];
        const w_o = try up.dense(.{ .trellis = t3, .suh = suh3, .svh = parts_o.svh, .bias = parts_o.bias, .suh_packed = parts_o.suh_packed }, .{ .k = half * 16, .n = @as(usize, @intCast(ref.int(o_o, "n"))), .halves = @as(u32, @intCast(go.halves)), .codebook = go.codebook });
        try check.expect(w_o.k == @as(u32, @intCast(ref.int(o_o, "k"))), "glm.kda.block: o_proj k is the rank's rows", .{});
        try dense.prompt(r.s, w_o, out, .bf16, rows, part, .f32);
    } else {
        const w_o = bmm_mod.B16{ .w = try r.dev(try ref.readFile(r.a, r.gpu.io, dir, ref.str(o_o, "w"))), .n = @intCast(ref.int(o_o, "n")), .k = @intCast(ref.int(o_o, "k")) };
        const part_scr = try r.zeros(bmm_mod.Bmm.partBytes(w_o, rows));
        try bmm.matmul(r.s, w_o, out, @intCast(w_o.k), rows, part, true, part_scr);
    }
    try check.sameBytes("glm.kda.block part", try r.back(part, want_part.len), want_part);

    // the block's own float64 reference over its prep outputs
    const out64 = try r.a.alloc(f64, @as(usize, rows) * q.per);
    const st64 = try r.a.alloc(f64, q.per * ref.DK);
    try ref.chunkedRef(r.a, ref.f32s(try ref.readFile(r.a, r.gpu.io, dir, ref.str(e, "prep_q"))), ref.f32s(try ref.readFile(r.a, r.gpu.io, dir, ref.str(e, "prep_k"))), ref.f32s(try ref.readFile(r.a, r.gpu.io, dir, ref.str(e, "prep_v"))), ref.f32s(try ref.readFile(r.a, r.gpu.io, dir, ref.str(e, "prep_lg"))), ref.f32s(try ref.readFile(r.a, r.gpu.io, dir, ref.str(e, "prep_beta"))), ref.f32s(try ref.readFile(r.a, r.gpu.io, dir, ref.str(e, "state0"))), ref.bf16s(try ref.readFile(r.a, r.gpu.io, dir, ref.str(e, "g"))), ref.bf16s(try ref.readFile(r.a, r.gpu.io, dir, "kda_norm.bin")), q.eps, q.h, rows, 0, out64, st64);
    var t = base.T{};
    try t.close("glm.kda.block out f64", try r.back(out, want_out.len), out64, false, 1.0 / 128.0, 2e-3);
    try t.close("glm.kda.block state f64", try r.back(st, want_state.len), st64, true, 2e-2, 2e-3);
    std.debug.print("RESULT glm.kda.block: f64 worst {d:.4} of tolerance\n", .{t.worst});
    check.pass("glm.kda.block", .{});
}

fn mm16(r: *Rig, dir: []const u8, mms: std.json.ObjectMap, key: []const u8) !bmm_mod.B16 {
    const mo = mms.get(key).?.object;
    return .{
        .w = try r.dev(try ref.readFile(r.a, r.gpu.io, dir, ref.str(mo, "file"))),
        .n = @intCast(ref.int(mo, "n")),
        .k = @intCast(ref.int(mo, "k")),
    };
}
