//! GLM-5.3-Flash latent attention on CUDA: cache write, absorb/expand, dense and sparse chunks vs float64 and Python.

const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const quant_ref_tests = @import("quant_ref_tests.zig");
const exl3_tests = @import("exl3_tests.zig");
const ref = @import("glm_kda_ref.zig");
const sref = @import("glm_sparse_ref.zig");
const lref = @import("glm_latent_ref.zig");
const ktests = @import("glm_kda_tests.zig");
const latent = cuda.glm_latent;

const Run = exl3_tests.Run;
const Rig = quant_ref_tests.RigT;
const T = ktests.T;

pub fn run(r: *Rig, state: *Run, dir: []const u8) !void {
    write(r, state, dir) catch |e| state.fail(e, "glm.dsa.latent");
    absorb(r, state, dir) catch |e| state.fail(e, "glm.dsa.absorb");
    dense(r, state, dir) catch |e| state.fail(e, "glm.dsa.dense");
    sparseAttn(r, state, dir) catch |e| state.fail(e, "glm.dsa.sparse");
}

fn skip(id: []const u8) void {
    std.debug.print("SKIP {s}\n", .{id});
}

fn u16s(bytes: []const u8) []const u16 {
    return @alignCast(std.mem.bytesAsSlice(u16, bytes));
}
fn f32s(bytes: []const u8) []const f32 {
    return @alignCast(std.mem.bytesAsSlice(f32, bytes));
}
fn i32s(bytes: []const u8) []const i32 {
    return @alignCast(std.mem.bytesAsSlice(i32, bytes));
}

/// A device-side copy of the scalar window position the kernels read.
fn posDev(r: *Rig, pos: i64) !u64 {
    const bytes = try r.a.alloc(u8, 4);
    std.mem.writeInt(i32, bytes[0..4], @intCast(pos), .little);
    return r.dev(bytes);
}

/// The cache write: fp8 codes with a per-token amax/448 scale, and a plain bf16 copy.
fn write(r: *Rig, state: *Run, dir: []const u8) !void {
    if (std.mem.eql(u8, dir, "none")) {
        _ = state;
        skip("glm.dsa.latent");
        return;
    }
    var man = try sref.Manifest.open(r.a, r.gpu.io, dir);
    defer man.deinit();
    const fam = man.root().get("latent") orelse {
        skip("glm.dsa.latent");
        return;
    };
    var k = try latent.Latent.load(r.gpu.d);
    defer k.unload();
    var t = T{};
    var it = fam.object.get("write").?.object.iterator();
    while (it.next()) |kv| {
        const e = kv.value_ptr.object;
        const tag = kv.key_ptr.*;
        const rows: u32 = @intCast(ref.int(e, "R"));
        const pos: i64 = ref.int(e, "pos");
        const fp8 = e.get("fp8").?.bool;
        const lat = try sref.readFile(r.a, r.gpu.io, dir, ref.str(e, "lat"));
        const want_cache = try sref.readFile(r.a, r.gpu.io, dir, ref.str(e, "cache"));
        const cache = try r.dev(try sref.readFile(r.a, r.gpu.io, dir, ref.str(e, "cache_pre")));
        const pos_d = try posDev(r, pos);
        var want_s: []f64 = &.{};
        if (fp8) {
            const scales = try r.dev(try sref.readFile(r.a, r.gpu.io, dir, ref.str(e, "scales_pre")));
            try k.latWrite(&r.s, try r.dev(lat), 512, cache, pos_d, scales, rows, true);
            const want_scales = try sref.readFile(r.a, r.gpu.io, dir, ref.str(e, "scales"));
            const got_scales = try r.back(scales, want_scales.len);
            try check.sameBytes(try std.fmt.allocPrint(r.a, "glm.dsa.latent {s} scales", .{tag}), got_scales, want_scales);
            // float64: the scale is the row's amax / 448 (the engine's div.full rounding aside)
            want_s = try r.a.alloc(f64, rows);
            for (0..rows) |row| {
                var amax: f64 = 0;
                for (0..512) |i| amax = @max(amax, @abs(lref.bf(u16s(lat)[row * 512 + i])));
                want_s[row] = if (amax > 0) amax / 448.0 else 1.0;
            }
            try t.close(try std.fmt.allocPrint(r.a, "glm.dsa.latent {s} scales f64", .{tag}), got_scales[@as(usize, @intCast(pos)) * 4 ..], want_s, true, 1.0e-6, 1.0e-33);
        } else {
            try k.latWrite(&r.s, try r.dev(lat), 512, cache, pos_d, 0, rows, false);
        }
        const got = try r.back(cache, want_cache.len);
        try check.sameBytes(try std.fmt.allocPrint(r.a, "glm.dsa.latent {s} cache", .{tag}), got, want_cache);
        // float64: the fp8 codes decode to lat / s; the bf16 copy widens exactly
        for (0..rows) |row| {
            const want_row = try r.a.alloc(f64, 512);
            for (0..512) |i| {
                const x = lref.bf(u16s(lat)[row * 512 + i]);
                want_row[i] = if (fp8) x / want_s[row] else x;
            }
            const tol: f64 = if (fp8) 1.0 / 8.0 else 1.0 / 128.0;
            const off = (@as(usize, @intCast(pos)) + row) * (if (fp8) @as(usize, 512) else 1024);
            var gbuf: []u8 = got[off..];
            if (fp8) {
                // close reads bf16: widen each stored code's exact fp8 value to bf16 bytes
                gbuf = try r.a.alloc(u8, 1024);
                for (0..512) |i| std.mem.writeInt(u16, gbuf[2 * i ..][0..2], quant_ref_tests.helpers.bf16Of(@floatCast(lref.fp8c(got[off + i]))), .little);
            } else gbuf = gbuf[0..1024];
            try t.close(try std.fmt.allocPrint(r.a, "glm.dsa.latent {s} r{d} f64", .{ tag, row }), gbuf, want_row, false, tol, 0.01);
        }
    }
    std.debug.print("RESULT glm.dsa.latent: f64 worst {d:.4} of tolerance\n", .{t.worst});
    check.pass("glm.dsa.latent", .{});
}

/// The kv_b absorb and expand projections over the layer's bf16 weights.
fn absorb(r: *Rig, state: *Run, dir: []const u8) !void {
    if (std.mem.eql(u8, dir, "none")) {
        _ = state;
        skip("glm.dsa.absorb");
        return;
    }
    var man = try sref.Manifest.open(r.a, r.gpu.io, dir);
    defer man.deinit();
    const fam = man.root().get("latent") orelse {
        skip("glm.dsa.absorb");
        return;
    };
    var k = try latent.Latent.load(r.gpu.d);
    defer k.unload();
    const wk = try r.dev(try sref.readFile(r.a, r.gpu.io, dir, "wk.bin"));
    const wv = try r.dev(try sref.readFile(r.a, r.gpu.io, dir, "wv.bin"));
    var t = T{};
    var it = fam.object.get("absorb").?.object.iterator();
    while (it.next()) |kv| {
        const e = kv.value_ptr.object;
        const tag = kv.key_ptr.*;
        const rows: u32 = @intCast(ref.int(e, "R"));
        const q = try sref.readFile(r.a, r.gpu.io, dir, ref.str(e, "q"));
        const want_qa = try sref.readFile(r.a, r.gpu.io, dir, ref.str(e, "qa"));
        const ol = try sref.readFile(r.a, r.gpu.io, dir, ref.str(e, "ol"));
        const want_out = try sref.readFile(r.a, r.gpu.io, dir, ref.str(e, "out"));
        const qa = try r.zeros(want_qa.len);
        const out = try r.zeros(want_out.len);
        try k.absorbQ(&r.s, try r.dev(q), wk, qa, rows);
        try k.expandV(&r.s, try r.dev(ol), wv, out, rows);
        const got_qa = try r.back(qa, want_qa.len);
        const got_out = try r.back(out, want_out.len);
        try check.sameBytes(try std.fmt.allocPrint(r.a, "glm.dsa.absorb {s} qa", .{tag}), got_qa, want_qa);
        try check.sameBytes(try std.fmt.allocPrint(r.a, "glm.dsa.absorb {s} out", .{tag}), got_out, want_out);
        if (rows <= 64) {
            const want64 = try lref.absorbRef(r.a, u16s(q), u16s(try sref.readFile(r.a, r.gpu.io, dir, "wk.bin")), rows, 32, 256, 512);
            try t.close(try std.fmt.allocPrint(r.a, "glm.dsa.absorb {s} qa f64", .{tag}), got_qa, want64, false, 1.0 / 128.0, 2.0e-3);
            const want_o = try lref.expandRef(r.a, u16s(ol), u16s(try sref.readFile(r.a, r.gpu.io, dir, "wv.bin")), rows, 32, 512, 256);
            try t.close(try std.fmt.allocPrint(r.a, "glm.dsa.absorb {s} out f64", .{tag}), got_out, want_o, false, 1.0 / 128.0, 2.0e-3);
        }
    }
    std.debug.print("RESULT glm.dsa.absorb: f64 worst {d:.4} of tolerance\n", .{t.worst});
    check.pass("glm.dsa.absorb", .{});
}

/// Dense attention: 512-key chunks with the flash update, then the cross-chunk merge.
fn dense(r: *Rig, state: *Run, dir: []const u8) !void {
    if (std.mem.eql(u8, dir, "none")) {
        _ = state;
        skip("glm.dsa.dense");
        return;
    }
    var man = try sref.Manifest.open(r.a, r.gpu.io, dir);
    defer man.deinit();
    const fam = man.root().get("latent") orelse {
        skip("glm.dsa.dense");
        return;
    };
    const scale = ref.float(fam.object.get("params").?.object, "scale");
    var k = try latent.Latent.load(r.gpu.d);
    defer k.unload();
    var t = T{};
    var it = fam.object.get("dense").?.object.iterator();
    while (it.next()) |kv| {
        const e = kv.value_ptr.object;
        const tag = kv.key_ptr.*;
        const rows: u32 = @intCast(ref.int(e, "R"));
        const pos: i64 = ref.int(e, "pos");
        const nch: u32 = @intCast(ref.int(e, "nch"));
        const fp8 = e.get("fp8").?.bool;
        const qa = try sref.readFile(r.a, r.gpu.io, dir, ref.str(e, "qa"));
        const cache_pre = try sref.readFile(r.a, r.gpu.io, dir, ref.str(e, "cache_pre"));
        const want_out = try sref.readFile(r.a, r.gpu.io, dir, ref.str(e, "out"));
        const qa_dev = try r.dev(qa);
        const cache_dev = try r.dev(cache_pre);
        const pos_d = try posDev(r, pos);
        var scales_dev: u64 = 0;
        if (fp8) scales_dev = try r.dev(try sref.readFile(r.a, r.gpu.io, dir, ref.str(e, "scales_pre")));
        const po = try r.zeros(@as(usize, nch) * rows * 32 * 512 * 4);
        const pm = try r.zeros(@as(usize, nch) * rows * 32 * 4);
        const pl = try r.zeros(@as(usize, nch) * rows * 32 * 4);
        const out = try r.zeros(want_out.len);
        try k.attnChunks(&r.s, qa_dev, cache_dev, scales_dev, pos_d, 0, 0, po, pm, pl, rows, 0, nch, fp8, false, @floatCast(scale));
        try k.merge(&r.s, po, pm, pl, out, 0, rows, nch, false);
        const got = try r.back(out, want_out.len);
        if (rows <= 8) {
            const want64 = try lref.attnRef(r.a, u16s(qa), cache_pre, if (fp8) f32s(try sref.readFile(r.a, r.gpu.io, dir, ref.str(e, "scales_pre"))) else null, pos, rows, 32, 512, scale, fp8, 8);
            try t.close(try std.fmt.allocPrint(r.a, "glm.dsa.dense {s} out f64", .{tag}), got, want64, false, 1.0 / 64.0, 2.0e-3);
        }
        try check.sameBytes(try std.fmt.allocPrint(r.a, "glm.dsa.dense {s} out", .{tag}), got, want_out);
    }
    std.debug.print("RESULT glm.dsa.dense: f64 worst {d:.4} of tolerance\n", .{t.worst});
    check.pass("glm.dsa.dense", .{});
}

/// The sparse path: each row attends its selected tokens (count 0 rows keep their input).
fn sparseAttn(r: *Rig, state: *Run, dir: []const u8) !void {
    if (std.mem.eql(u8, dir, "none")) {
        _ = state;
        skip("glm.dsa.sparse");
        return;
    }
    var man = try sref.Manifest.open(r.a, r.gpu.io, dir);
    defer man.deinit();
    const sp = man.root().get("sparse") orelse {
        skip("glm.dsa.sparse");
        return;
    };
    const scale = ref.float(sp.object.get("params").?.object, "scale");
    var k = try latent.Latent.load(r.gpu.d);
    defer k.unload();
    var t = T{};
    var it = sp.object.get("cases").?.object.iterator();
    while (it.next()) |kv| {
        const e = kv.value_ptr.object;
        const tag = kv.key_ptr.*;
        const rows: u32 = @intCast(ref.int(e, "R"));
        const tokens = try sref.readFile(r.a, r.gpu.io, dir, ref.str(e, "tokens"));
        const counts = try sref.readFile(r.a, r.gpu.io, dir, ref.str(e, "counts"));
        const w = i32s(tokens).len / rows;
        const nch: u32 = @intCast((w + 511) / 512);
        const qa = try sref.readFile(r.a, r.gpu.io, dir, ref.str(e, "qa"));
        const cache = try sref.readFile(r.a, r.gpu.io, dir, ref.str(e, "cache"));
        const scales = try sref.readFile(r.a, r.gpu.io, dir, ref.str(e, "scales"));
        const want_out = try sref.readFile(r.a, r.gpu.io, dir, ref.str(e, "out"));
        const qa_dev = try r.dev(qa);
        const out = try r.dev(try sref.readFile(r.a, r.gpu.io, dir, ref.str(e, "out_in")));
        const tok_dev = try r.dev(tokens);
        const cnt_dev = try r.dev(counts);
        const scales_dev = try r.dev(scales);
        const po = try r.zeros(@as(usize, nch) * rows * 32 * 512 * 4);
        const pm = try r.zeros(@as(usize, nch) * rows * 32 * 4);
        const pl = try r.zeros(@as(usize, nch) * rows * 32 * 4);
        try k.attnChunks(&r.s, qa_dev, try r.dev(cache), scales_dev, 0, tok_dev, cnt_dev, po, pm, pl, rows, @intCast(w), nch, true, true, @floatCast(scale));
        try k.merge(&r.s, po, pm, pl, out, cnt_dev, rows, nch, true);
        const got = try r.back(out, want_out.len);
        if (rows <= 8) {
            const cnts = i32s(counts);
            const want64 = try sref.sparseRef(r.a, u16s(qa), cache, f32s(scales), i32s(tokens), cnts, 32, 512, scale);
            for (0..rows) |row| {
                if (cnts[row] == 0) continue;
                try t.close(try std.fmt.allocPrint(r.a, "glm.dsa.sparse {s} r{d} f64", .{ tag, row }), got[row * 32 * 512 * 2 ..][0 .. 32 * 512 * 2], want64[row * 32 * 512 ..][0 .. 32 * 512], false, 1.0 / 64.0, 2.0e-3);
            }
        }
        try check.sameBytes(try std.fmt.allocPrint(r.a, "glm.dsa.sparse {s} out", .{tag}), got, want_out);
    }
    std.debug.print("RESULT glm.dsa.sparse: f64 worst {d:.4} of tolerance\n", .{t.worst});
    check.pass("glm.dsa.sparse", .{});
}
