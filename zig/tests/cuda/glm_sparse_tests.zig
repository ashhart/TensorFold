//! GLM-5.3-Flash's DSA sparse path on CUDA: indexer and pool scores against float64 refs and Python engine bytes.

const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const quant_ref_tests = @import("quant_ref_tests.zig");
const exl3_tests = @import("exl3_tests.zig");
const ref = @import("glm_kda_ref.zig");
const sref = @import("glm_sparse_ref.zig");
const ktests = @import("glm_kda_tests.zig");
const sparse = cuda.glm_sparse;

const Run = exl3_tests.Run;
const Rig = quant_ref_tests.RigT;
const T = ktests.T;

pub fn run(r: *Rig, state: *Run, dir: []const u8) !void {
    index(r, state, dir) catch |e| state.fail(e, "glm.dsa.indexer");
    scores(r, state, dir) catch |e| state.fail(e, "glm.dsa.scores");
}

fn skip(id: []const u8) void {
    std.debug.print("SKIP {s}\n", .{id});
}

/// The fixture's sparse params (shared by every check).
const Params = struct {
    scale: f64,
    wscale: f64,

    fn read(o: std.json.ObjectMap) !Params {
        const p = o.get("params").?.object;
        return .{
            .scale = ref.float(p, "scale"),
            .wscale = ref.float(p, "wscale"),
        };
    }
};

/// The index update: LayerNorm'd keys and gates into the window's cache rows, then the completed pools' keys.
fn index(r: *Rig, state: *Run, dir: []const u8) !void {
    if (std.mem.eql(u8, dir, "none")) {
        _ = state;
        skip("glm.dsa.indexer");
        return;
    }
    var man = try sref.Manifest.open(r.a, r.gpu.io, dir);
    defer man.deinit();
    const fam = man.root().get("sparse") orelse {
        skip("glm.dsa.indexer");
        return;
    };
    _ = try Params.read(fam.object);
    var k = try sparse.Sparse.load(r.gpu.d);
    defer k.unload();
    var t = T{};
    var it = fam.object.get("cases").?.object.iterator();
    while (it.next()) |kv| {
        const e = kv.value_ptr.object;
        const tag = kv.key_ptr.*;
        const pos: i32 = @intCast(ref.int(e, "pos"));
        const rows: usize = @intCast(ref.int(e, "R"));
        const ikr = try sref.readFile(r.a, r.gpu.io, dir, ref.str(e, "ikr"));
        const igr = try sref.readFile(r.a, r.gpu.io, dir, ref.str(e, "igr"));
        const lnw = try sref.readFile(r.a, r.gpu.io, dir, "ln_w.bin");
        const lnb = try sref.readFile(r.a, r.gpu.io, dir, "ln_b.bin");
        const ape = try sref.readFile(r.a, r.gpu.io, dir, "ape.bin");
        const want_ik = try sref.readFile(r.a, r.gpu.io, dir, ref.str(e, "ik"));
        const want_ig = try sref.readFile(r.a, r.gpu.io, dir, ref.str(e, "ig"));
        const want_pk = try sref.readFile(r.a, r.gpu.io, dir, ref.str(e, "pk"));
        // the kernels run from the pre-state: rows before the window hold earlier windows' values
        const ik = try r.dev(try sref.readFile(r.a, r.gpu.io, dir, ref.str(e, "ik_pre")));
        const ig = try r.dev(try sref.readFile(r.a, r.gpu.io, dir, ref.str(e, "ig_pre")));
        const pk = try r.dev(try sref.readFile(r.a, r.gpu.io, dir, ref.str(e, "pk_pre")));
        try k.indexWrite(&r.s, try r.dev(ikr), 160, try r.dev(igr), try r.dev(lnw), try r.dev(lnb), ik, ig, pos, 1.0e-6, @intCast(rows));
        try k.poolKeys(&r.s, ik, ig, try r.dev(ape), pk, pos, @intCast(rows));
        const got_ik = try r.back(ik, want_ik.len);
        try check.sameBytes(try std.fmt.allocPrint(r.a, "glm.dsa.indexer {s} ik", .{tag}), got_ik, want_ik);
        try check.sameBytes(try std.fmt.allocPrint(r.a, "glm.dsa.indexer {s} ig", .{tag}), try r.back(ig, want_ig.len), want_ig);
        try check.sameBytes(try std.fmt.allocPrint(r.a, "glm.dsa.indexer {s} pk", .{tag}), try r.back(pk, want_pk.len), want_pk);
        // float64: the LayerNorm rows and the window's completed pools
        const want64 = try sref.indexRef(r.a, u16s(ikr), 160, u16s(lnw), u16s(lnb), 1.0e-6, rows);
        try t.close(try std.fmt.allocPrint(r.a, "glm.dsa.indexer {s} ik f64", .{tag}), got_ik[@as(usize, @intCast(pos)) * sref.D * 2 ..], want64, false, 1.0 / 128.0, 2.0e-3);
        const npool_end = @divTrunc(@as(i64, pos) + @as(i64, @intCast(rows)), 4);
        const want_pool = try sref.poolRef(r.a, u16s(want_ig), u16s(ape), u16s(want_ik), @intCast(@divTrunc(pos, 4)), @intCast(npool_end));
        const got_pk = try r.back(pk, want_pk.len);
        try t.close(try std.fmt.allocPrint(r.a, "glm.dsa.indexer {s} pk f64", .{tag}), got_pk[@as(usize, @intCast(@divTrunc(pos, 4))) * sref.D * 2 ..], want_pool, false, 1.0 / 64.0, 2.0e-3);
    }
    std.debug.print("RESULT glm.dsa.indexer: f64 worst {d:.4} of tolerance\n", .{t.worst});
    check.pass("glm.dsa.indexer", .{});
}

/// The pool scores: every visible pool per row, one block launch per 512 rows, then the float64 reference.
fn scores(r: *Rig, state: *Run, dir: []const u8) !void {
    if (std.mem.eql(u8, dir, "none")) {
        _ = state;
        skip("glm.dsa.scores");
        return;
    }
    var man = try sref.Manifest.open(r.a, r.gpu.io, dir);
    defer man.deinit();
    const fam = man.root().get("sparse") orelse {
        skip("glm.dsa.scores");
        return;
    };
    const q = try Params.read(fam.object);
    var k = try sparse.Sparse.load(r.gpu.d);
    defer k.unload();
    var t = T{};
    var it = fam.object.get("cases").?.object.iterator();
    while (it.next()) |kv| {
        const e = kv.value_ptr.object;
        const tag = kv.key_ptr.*;
        const pos: i32 = @intCast(ref.int(e, "pos"));
        const rows: usize = @intCast(ref.int(e, "R"));
        const np_max: usize = @intCast(ref.int(e, "np_max"));
        const qi = try sref.readFile(r.a, r.gpu.io, dir, ref.str(e, "qi"));
        const ikr = try sref.readFile(r.a, r.gpu.io, dir, ref.str(e, "ikr"));
        const pk = try sref.readFile(r.a, r.gpu.io, dir, ref.str(e, "pk"));
        const qi_dev = try r.dev(qi);
        const w_base = try r.dev(ikr); // the weights are ikr's last 32 columns, per block row
        const pk_dev = try r.dev(pk);
        const blocks = e.get("blocks").?.array;
        for (blocks.items, 0..) |bname, bi| {
            const b0 = bi * 512;
            const n = @min(@as(usize, 512), rows - b0);
            const want = try sref.readFile(r.a, r.gpu.io, dir, bname.string);
            const out = try r.zeros(want.len);
            try k.scores(&r.s, qi_dev + b0 * 4096 * 2, 4096, w_base + (b0 * 160 + sref.D) * 2, 160, pk_dev, out, pos + @as(i32, @intCast(b0)), @intCast(n), @intCast(np_max), @floatCast(q.scale), @floatCast(q.wscale));
            const got = try r.back(out, want.len);
            try check.sameBytes(try std.fmt.allocPrint(r.a, "glm.dsa.scores {s} b{d}", .{ tag, b0 }), got, want);
            if (rows <= 128) {
                const want64 = try sref.scoresRef(r.a, u16s(qi)[b0 * 4096 ..], 4096, u16s(ikr)[b0 * 160 + sref.D ..], 160, u16s(pk), n, np_max, q.scale, q.wscale, pos + @as(i32, @intCast(b0)));
                try t.close(try std.fmt.allocPrint(r.a, "glm.dsa.scores {s} f64", .{tag}), got, want64, true, 1.0 / 64.0, 2.0e-3);
            }
        }
    }
    std.debug.print("RESULT glm.dsa.scores: score f64 worst {d:.4} of tolerance\n", .{t.worst});
    try selectPhase(r, fam, q, dir);
    check.pass("glm.dsa.scores", .{});
}

/// The selection phase: scores over the row's visible bucket, the radix select, then tokens and counts.
fn selectPhase(r: *Rig, fam: std.json.Value, q: Params, dir: []const u8) !void {
    var k = try sparse.Sparse.load(r.gpu.d);
    defer k.unload();
    var it = fam.object.get("cases").?.object.iterator();
    while (it.next()) |kv| {
        const e = kv.value_ptr.object;
        const tag = kv.key_ptr.*;
        const pos: i32 = @intCast(ref.int(e, "pos"));
        const rows: usize = @intCast(ref.int(e, "R"));
        const np_max: usize = @intCast(ref.int(e, "np_max"));
        const visible = @divTrunc(pos + @as(i32, @intCast(rows)), 4) + 1;
        var bucket: usize = 1;
        var v: u32 = @intCast(visible - 1);
        while (v != 0) : (v >>= 1) bucket <<= 1;
        bucket = @max(bucket, 1024);
        bucket = @min(bucket, np_max);
        const qi = try sref.readFile(r.a, r.gpu.io, dir, ref.str(e, "qi"));
        const ikr = try sref.readFile(r.a, r.gpu.io, dir, ref.str(e, "ikr"));
        const pk = try sref.readFile(r.a, r.gpu.io, dir, ref.str(e, "pk"));
        const qi_dev = try r.dev(qi);
        const w_base = try r.dev(ikr);
        const pk_dev = try r.dev(pk);
        const sc = try r.zeros(rows * bucket * 4);
        var b0: usize = 0;
        while (b0 < rows) : (b0 += 512) {
            const n = @min(@as(usize, 512), rows - b0);
            try k.scores(&r.s, qi_dev + b0 * 4096 * 2, 4096, w_base + (b0 * 160 + sref.D) * 2, 160, pk_dev, sc + b0 * bucket * 4, pos + @as(i32, @intCast(b0)), @intCast(n), @intCast(bucket), @floatCast(q.scale), @floatCast(q.wscale));
        }
        const pools = try r.zeros(rows * 512 * 8);
        try k.selectRows(&r.s, sc, @intCast(bucket), pools, pos, 512, @intCast(rows), true);
        const tokens = try r.zeros(rows * 2051 * 4);
        const counts = try r.zeros(rows * 4);
        try k.tokensCounts(&r.s, pools, tokens, counts, pos, @intCast(rows));
        const want_tokens = try sref.readFile(r.a, r.gpu.io, dir, ref.str(e, "tokens"));
        const want_counts = try sref.readFile(r.a, r.gpu.io, dir, ref.str(e, "counts"));
        try check.sameBytes(try std.fmt.allocPrint(r.a, "glm.dsa.scores {s} tokens", .{tag}), try r.back(tokens, want_tokens.len), want_tokens);
        try check.sameBytes(try std.fmt.allocPrint(r.a, "glm.dsa.scores {s} counts", .{tag}), try r.back(counts, want_counts.len), want_counts);
        // float64-exact reference: the order-key sort of the fp32 scores
        if (rows <= 128) {
            const got_tokens = try r.back(tokens, want_tokens.len);
            const got_counts = try r.back(counts, want_counts.len);
            const got_scores_raw = try r.back(sc, rows * bucket * 4);
            const got_scores = try r.a.alloc(f32, rows * bucket);
            @memcpy(std.mem.sliceAsBytes(got_scores), got_scores_raw);
            const ref_tokens = try r.a.alloc(i32, rows * 2051);
            const ref_counts = try r.a.alloc(i32, rows);
            try sref.selectRef(r.a, got_scores, bucket, 512, pos, rows, ref_tokens, ref_counts);
            for (ref_tokens, 0..) |want_v, i| {
                const got_v = std.mem.readInt(i32, got_tokens[4 * i ..][0..4], .little);
                if (got_v != want_v) return error.TestFailed;
            }
            for (ref_counts, 0..) |want_v, i| {
                const got_v = std.mem.readInt(i32, got_counts[4 * i ..][0..4], .little);
                if (got_v != want_v) return error.TestFailed;
            }
        }
    }
    std.debug.print("RESULT glm.dsa.scores: tokens exact against the order-key sort\n", .{});
}

fn u16s(bytes: []const u8) []const u16 {
    return @alignCast(std.mem.bytesAsSlice(u16, bytes));
}
