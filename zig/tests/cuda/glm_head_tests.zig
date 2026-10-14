//! glm.head: the model's tail on both ranks' head halves vs the Python engine (compute's end, sample_rows greedy).
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const glm_tests = @import("glm_tests.zig");
const glue_tests = @import("glm_glue_tests.zig");
const ref = @import("glm_kda_ref.zig");
const head = cuda.glm_head;
const Tol = glue_tests.Tol;

const id = "glm.head";

pub fn run(r: *glm_tests.Rig, state: *glm_tests.Run, dir: []const u8) !void {
    check_(r, dir) catch |e| state.fail(e, id);
}

fn bfv(a: std.mem.Allocator, bytes: []const u8) ![]f64 {
    const u = std.mem.bytesAsSlice(u16, bytes);
    const out = try a.alloc(f64, u.len);
    for (u, out) |v, *o| o.* = ref.bf(v);
    return out;
}

fn check_(r: *glm_tests.Rig, dir: []const u8) !void {
    if (std.mem.eql(u8, dir, "none")) return std.debug.print("SKIP {s}\n", .{id});
    var man = try ref.Manifest.open(r.gpu.gpa, r.gpu.io, dir);
    defer man.deinit();
    const o = (man.root().get("head") orelse return std.debug.print("SKIP {s}\n", .{id})).object;
    const p = o.get("params").?.object;
    const eps: f32 = @floatCast(ref.float(p, "eps"));
    const d: usize = @intCast(ref.int(p, "hidden"));
    const streams: usize = @intCast(ref.int(p, "streams"));
    var g = try cuda.glm_glue.Glue.load(r.gpu.d);
    defer g.unload();
    var b = try cuda.glm_bmm.Bmm.load(r.gpu.d);
    defer b.unload();
    const norm_bytes = try ref.readFile(r.a, r.gpu.io, dir, ref.str(o, "norm"));
    const norm = try r.dev(norm_bytes);
    const normv = try bfv(r.a, norm_bytes);
    const hs = o.get("heads").?.array.items;
    if (hs.len != 2) return error.GlmHeadShape;
    var ws: [2]head.HeadW = undefined;
    var hostw: [2][]const u8 = undefined;
    for (hs, &ws, &hostw) |hv, *w, *hw| {
        const h = hv.object;
        hw.* = try ref.readFile(r.a, r.gpu.io, dir, ref.str(h, "w"));
        w.* = .{ .norm = norm, .head = .{ .w = try r.dev(hw.*), .n = @intCast(ref.int(h, "n")), .k = @intCast(ref.int(h, "k")) }, .offset = @intCast(ref.int(h, "offset")) };
    }
    var t = Tol{};
    var it = o.get("cases").?.object.iterator();
    var count: usize = 0;
    while (it.next()) |kv| : (count += 1) {
        const tag = kv.key_ptr.*;
        const e = kv.value_ptr.object;
        const rows: usize = @intCast(ref.int(e, "rows"));
        const last = e.get("prefill").?.bool;
        const m: usize = if (last) 1 else rows;
        const xb = try ref.readFile(r.a, r.gpu.io, dir, ref.str(e, "x"));
        const x = try r.dev(xb);
        const lgs = e.get("logits").?.array.items;
        const gathered = try r.a.alloc(f32, 2 * m * 2);
        for (0..2) |rank| {
            var sc = try head.Scratch.init(r.gpu.d, r.s, ws[rank], d, rows);
            defer sc.deinit();
            try head.forward(&g, &b, r.s, eps, d, streams, ws[rank], &sc, x, rows, last);
            const nm = try std.fmt.allocPrint(r.a, "{s} {s}", .{ id, tag });
            if (rank == 0) for ([_][]const u8{ "hidden", "fnormed", "fxs" }, [_]u64{ sc.hidden, sc.fnormed, sc.fxs }) |key, ptr| {
                const want = try ref.readFile(r.a, r.gpu.io, dir, ref.str(e, key));
                try check.sameBytes(try std.fmt.allocPrint(r.a, "{s} {s}", .{ nm, key }), try r.back(ptr, want.len), want);
            };
            const want_lg = try ref.readFile(r.a, r.gpu.io, dir, lgs[rank].string);
            const got_lg = try r.back(sc.logits, want_lg.len);
            try check.sameBytes(try std.fmt.allocPrint(r.a, "{s} logits rank {d}", .{ nm, rank }), got_lg, want_lg);
            const cand = std.mem.bytesAsSlice(f32, try r.back(sc.cand, m * 8));
            @memcpy(gathered[rank * m * 2 ..][0 .. m * 2], cand);
            if (rank == 0) try f64Stages(r.a, &t, xb, try r.back(sc.hidden, rows * d * 2), try r.back(sc.fnormed, rows * d * 2), normv, hostw[0], got_lg, eps, rows, d, streams, m, ws[0].head.n);
        }
        const toks = try r.a.alloc(u32, m);
        try head.choose(gathered, 2, m, toks);
        const want_tok = try ref.readFile(r.a, r.gpu.io, dir, ref.str(e, "tok"));
        try check.sameBytes(try std.fmt.allocPrint(r.a, "{s} {s} tokens", .{ id, tag }), std.mem.sliceAsBytes(toks), want_tok);
        std.debug.print("RESULT {s} {s}: {d} rows, {d} tokens bit-exact across both ranks\n", .{ id, tag, rows, m });
    }
    try t.done(id);
    std.debug.print("RESULT {s}: {d} cases bit-exact\n", .{ id, count });
    check.pass("{s}", .{id});
}

/// Rank 0's stages in f64 from each stage's own inputs: the streams' mean, the final norm, sampled logit columns.
fn f64Stages(a: std.mem.Allocator, t: *Tol, xb: []const u8, hb: []const u8, fb: []const u8, normv: []const f64, wb: []const u8, lb: []const u8, eps: f32, rows: usize, d: usize, streams: usize, m: usize, n: usize) !void {
    const xv = try bfv(a, xb);
    const hv = try bfv(a, hb);
    const fv = try bfv(a, fb);
    const lv = try bfv(a, lb);
    const wv = std.mem.bytesAsSlice(u16, wb);
    for (0..rows) |i| {
        var ss: f64 = 0;
        for (0..d) |j| {
            var acc: f64 = 0;
            for (0..streams) |s| acc += xv[(i * streams + s) * d + j];
            t.f64v(hv[i * d + j], acc / @as(f64, @floatFromInt(streams)), 1.0 / 64.0, 3e-3);
            ss += hv[i * d + j] * hv[i * d + j];
        }
        const rinv = 1.0 / @sqrt(ss / @as(f64, @floatFromInt(d)) + eps);
        for (0..d) |j| t.f64v(fv[i * d + j], normv[j] * hv[i * d + j] * rinv, 1.0 / 64.0, 3e-3);
    }
    const first = rows - m;
    for (0..m) |i| {
        var col: usize = i * 7;
        while (col < n) : (col += 997) {
            var acc: f64 = 0;
            for (0..d) |j| acc += fv[(first + i) * d + j] * ref.bf(wv[col * d + j]);
            t.f64v(lv[i * n + col], acc, 1.0 / 64.0, 3e-3);
        }
    }
}
