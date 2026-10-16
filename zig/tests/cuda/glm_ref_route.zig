//! glue.select's group-limited routing (Ling's 512 / 8 groups / keep 4 / top 8) vs float64; GLM's bytes vs 6380c8da.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const qr = @import("quant_ref_tests.zig");
const glm = @import("glm_cuda");
const Gpu = check.Gpu;
const Rig = qr.RigT;
const bf = qr.helpers.bfOf;

const glue = glm.glue;

/// One routing problem: fp32 logits [rows][ne], fp32 bias, top_k of the `keep` best of `ng` groups, times `scale`.
const Route = struct { rows: usize, ne: usize, top_k: usize, ng: usize, keep: usize, scale: f32 };

/// The kernels' near ties: two choices (or group sums) this close in float64 may rank either way after fp32 rounding.
const tie_expert = std.math.pow(f64, 2, -20);
const tie_group = std.math.pow(f64, 2, -19);

fn sigmoid(x: f32) f64 {
    return 1 / (1 + @exp(-@as(f64, x)));
}

/// float64 rule: `keep` best top-2 group sums (or `forced`), then top_k in them, lower ids on ties; returns kept bits.
fn refPick(choice: []const f64, q: Route, forced: ?u32, order: []usize, gs_out: []f64) u32 {
    const gs = q.ne / q.ng;
    var kept: u32 = if (q.ng == 1) 1 else 0;
    if (q.ng > 1) {
        for (0..q.ng) |g| {
            var a: f64 = -std.math.inf(f64);
            var b: f64 = -std.math.inf(f64);
            for (choice[g * gs ..][0..gs]) |c| {
                if (c > a) {
                    b = a;
                    a = c;
                } else if (c > b) b = c;
            }
            gs_out[g] = a + b;
        }
        if (forced) |f| kept = f else for (0..q.keep) |_| {
            var best: usize = q.ng;
            for (0..q.ng) |g| if (kept >> @intCast(g) & 1 == 0 and (best == q.ng or gs_out[g] > gs_out[best])) {
                best = g;
            };
            kept |= @as(u32, 1) << @intCast(best);
        }
    }
    var taken: [512]bool = @splat(false);
    for (order[0..q.top_k]) |*o| {
        var best: usize = q.ne;
        for (0..q.ne) |e| {
            if (taken[e] or kept >> @intCast(e / gs) & 1 == 0) continue;
            if (best == q.ne or choice[e] > choice[best]) best = e;
        }
        o.* = best;
        taken[best] = true;
    }
    return kept;
}

/// The k-th and (k+1)-th best group sums' gap: below tie_group the kernel's kept set may legitimately differ.
fn boundaryGap(gsc: []const f64, keep: usize) f64 {
    var s: [32]f64 = undefined;
    @memcpy(s[0..gsc.len], gsc);
    std.mem.sort(f64, s[0..gsc.len], {}, std.sort.desc(f64));
    return s[keep - 1] - s[keep];
}

const Tally = struct { rows: usize = 0, group_ties: usize = 0, expert_ties: usize = 0, widest: f64 = 0, worst: f64 = 0 };

/// Kernel picks and weights for every row against the float64 rule; near ties counted, anything else fails.
fn against(r: *Rig, q: Route, logits: []const f32, bias: []const f32, got_pick: []const i32, got_wts: []const f32) !Tally {
    var t: Tally = .{ .rows = q.rows };
    const choice = try r.a.alloc(f64, q.ne);
    const score = try r.a.alloc(f64, q.ne);
    const order = try r.a.alloc(usize, q.top_k);
    var gsc: [32]f64 = undefined;
    const slots = q.top_k + 1;
    for (0..q.rows) |row| {
        for (score, choice, logits[row * q.ne ..][0..q.ne], bias) |*s, *c, l, b| {
            s.* = sigmoid(l);
            c.* = s.* + b;
        }
        const kept = refPick(choice, q, null, order, gsc[0..q.ng]);
        const gs = q.ne / q.ng;
        var mine: u32 = 0;
        for (got_pick[row * slots ..][0..q.top_k]) |id| mine |= @as(u32, 1) << @intCast(@as(usize, @intCast(id)) / gs);
        if (mine & ~kept != 0) { // a pick outside the float64 groups: only a near tie at the group boundary may do that
            const gap = boundaryGap(gsc[0..q.ng], q.keep);
            try check.expect(gap <= tie_group, "row {d}: groups {b} picked where float64 keeps {b} (boundary gap {e})", .{ row, mine, kept, gap });
            t.group_ties += 1;
            t.widest = @max(t.widest, gap);
            var f = mine;
            var g: usize = 0;
            while (@popCount(f) < q.keep) : (g += 1) if (kept >> @intCast(g) & 1 != 0) {
                f |= @as(u32, 1) << @intCast(g);
            };
            _ = refPick(choice, q, f, order, gsc[0..q.ng]);
        }
        var total: f64 = 0;
        for (got_pick[row * slots ..][0..q.top_k]) |id| total += score[@intCast(id)];
        for (0..slots) |i| {
            const id: usize = @intCast(got_pick[row * slots + i]);
            const wt: f64 = got_wts[row * slots + i];
            if (i == q.top_k) {
                try check.expect(id == q.ne and wt == 1, "row {d}: the shared expert last at weight 1", .{row});
                continue;
            }
            if (id != order[i]) {
                const gap = @abs(choice[id] - choice[order[i]]);
                try check.expect(gap <= tie_expert, "row {d} pick {d}: {d} where float64 ranks {d} (gap {e})", .{ row, i, id, order[i], gap });
                t.expert_ties += 1;
                t.widest = @max(t.widest, gap);
            }
            const want = score[id] / total * q.scale;
            t.worst = @max(t.worst, @abs(wt - want) / (@abs(want) * std.math.pow(f64, 2, -16)));
        }
    }
    return t;
}

fn f32s(r: *Rig, n: usize, lo: f32, hi: f32) ![]f32 {
    const out = try r.a.alloc(f32, n);
    for (out) |*v| v.* = lo + r.rng.float(f32) * (hi - lo);
    return out;
}

const Out = struct { pick: []const i32, wts: []const f32 };

fn runSelect(r: *Rig, set: glue.Set, q: Route, logits: []const f32, bias: []const f32, norm: bool) !Out {
    const n = q.rows * (q.top_k + 1);
    const pick = try r.zeros(n * 4);
    const wts = try r.zeros(n * 4);
    try glue.select(set, r.s, try r.dev(std.mem.sliceAsBytes(logits)), try r.dev(std.mem.sliceAsBytes(bias)), pick, wts, q.rows, q.top_k, q.ne, q.scale, norm, .{ .n = q.ng, .keep = q.keep });
    return .{ .pick = std.mem.bytesAsSlice(i32, @as([]align(4) u8, @alignCast(try r.back(pick, n * 4)))), .wts = std.mem.bytesAsSlice(f32, @as([]align(4) u8, @alignCast(try r.back(wts, n * 4)))) };
}

/// Rows built for exact ties: every group a copy of group 0 (all group sums tie), and a group whose top two are equal.
fn tieRows(r: *Rig, q: Route, logits: []f32, bias: []f32) void {
    const gs = q.ne / q.ng;
    @memset(bias, 0);
    for (0..q.rows) |row| {
        const l = logits[row * q.ne ..][0..q.ne];
        switch (row % 3) {
            0 => for (0..q.ne) |e| {
                l[e] = l[e % gs];
            },
            1 => { // equal pairs at 2.0 (one lane's in 6, two lanes' in 7) beat 2.2 + 0.0 only if the duplicate counts
                for (l) |*v| v.* = -6 + r.rng.float(f32);
                for (0..q.ng) |g| {
                    l[g * gs + 3] = if (g >= 6) 2.0 else 2.2;
                    l[g * gs + 35] = if (g == 6) 2.0 else 0.0;
                    if (g == 7) l[g * gs + 7] = 2.0;
                }
            },
            else => for (0..q.ne) |e| {
                l[e] = if (e % 7 == 0) 1.0 else l[e];
            },
        }
    }
}

/// Synthetic Ling-shaped routing (and a few odd shapes) against float64, plus exact-tie rows that must match exactly.
pub fn groups(r: *Rig, set: glue.Set) !void {
    const shapes = [_]Route{
        .{ .rows = 300, .ne = 512, .top_k = 8, .ng = 8, .keep = 4, .scale = 2.5 },
        .{ .rows = 300, .ne = 256, .top_k = 8, .ng = 8, .keep = 4, .scale = 2.5 }, // DeepSeek-V3's
        .{ .rows = 100, .ne = 480, .top_k = 6, .ng = 12, .keep = 3, .scale = 1.0 }, // groups of 40 straddle the lanes
        .{ .rows = 100, .ne = 512, .top_k = 8, .ng = 32, .keep = 1, .scale = 2.5 },
        .{ .rows = 100, .ne = 512, .top_k = 8, .ng = 1, .keep = 1, .scale = 2.5 }, // one group: no group pass
        .{ .rows = 100, .ne = 512, .top_k = 8, .ng = 8, .keep = 8, .scale = 2.5 }, // every group kept
        .{ .rows = 100, .ne = 64, .top_k = 8, .ng = 32, .keep = 4, .scale = 2.5 }, // groups of two
        .{ .rows = 100, .ne = 16, .top_k = 4, .ng = 8, .keep = 3, .scale = 1.0 }, // groups of two, the tiny config's experts
        .{ .rows = 100, .ne = 16, .top_k = 4, .ng = 4, .keep = 2, .scale = 2.5 }, // the tiny config's routing
    };
    for (shapes) |q| for ([_]f32{ 0.001, 0.05 }) |bspread| {
        const logits = try f32s(r, q.rows * q.ne, -6, 6);
        const bias = try f32s(r, q.ne, -bspread, bspread);
        const o = try runSelect(r, set, q, logits, bias, true);
        const t = try against(r, q, logits, bias, o.pick, o.wts);
        r.worst = @max(r.worst, t.worst);
        std.debug.print("RESULT group select {d} rows, top {d} of {d} in {d} of {d} groups (bias +-{d}): weights at {e:.3} of 2^-16 relative, {d} group and {d} expert near ties (widest gap {e:.2})\n", .{ q.rows, q.top_k, q.ne, q.keep, q.ng, bspread, t.worst, t.group_ties, t.expert_ties, t.widest });
        try check.expect(t.worst <= 1, "group select weights within 2^-16 of float64", .{});
    };
    const q: Route = .{ .rows = 30, .ne = 512, .top_k = 8, .ng = 8, .keep = 4, .scale = 2.5 };
    const logits = try f32s(r, q.rows * q.ne, -6, 6);
    const bias = try r.a.alloc(f32, q.ne);
    tieRows(r, q, logits, bias);
    const o = try runSelect(r, set, q, logits, bias, true);
    const t = try against(r, q, logits, bias, o.pick, o.wts);
    try check.expect(t.group_ties == 0 and t.expert_ties == 0, "exact-tie rows: {d} group and {d} expert swaps, want none", .{ t.group_ties, t.expert_ties });
    check.pass("group select: picks follow the float64 rule (exact ties to the lower id, near ties counted), weights within 2^-16", .{});
}

/// Ling's swiglu at limit 0 (its unclamped layers) against float64 unclamped, and limit 10 differing where it clamps.
pub fn swigluLimit(r: *Rig, fs: [2]cuda.Function) !void {
    for ([_]usize{ 1024, 704 }) |w| {
        const rows = 40;
        const gu = try r.bfs(rows * 2 * w, 16.0);
        const dgu = try r.dev(std.mem.sliceAsBytes(gu));
        const out0 = try r.zeros(rows * w * 2);
        const out10 = try r.zeros(rows * w * 2);
        const xs = try r.zeros(rows * w / 64 * 4);
        try glue.swigluWith(fs, r.s, dgu, out0, xs, 0, rows, w);
        try glue.swigluWith(fs, r.s, dgu, out10, xs, 10, rows, w);
        const raw0 = try r.back(out0, rows * w * 2);
        const raw10 = try r.back(out10, rows * w * 2);
        var most: f64 = 0;
        var clamped: usize = 0;
        for (0..rows) |m| for (0..w) |d| {
            const g = bf(gu[m * 2 * w + d]);
            const u = bf(gu[m * 2 * w + w + d]);
            const want = g / (1 + @exp(-g)) * u;
            const got = bf(std.mem.bytesToValue(u16, raw0[2 * (m * w + d) ..][0..2]));
            most = @max(most, @abs(got - want) / (@abs(want) * std.math.pow(f64, 2, -7) + 1e-30));
            if (!std.mem.eql(u8, raw0[2 * (m * w + d) ..][0..2], raw10[2 * (m * w + d) ..][0..2])) clamped += 1;
        };
        r.worst = @max(r.worst, most);
        std.debug.print("RESULT swiglu width {d}, limit 0: unclamped float64 at {e:.3} of 2^-7 relative; limit 10 changes {d} of {d}\n", .{ w, most, clamped, rows * w });
        try check.expect(most <= 1 and clamped > 0, "swiglu limit 0 is unclamped, limit 10 clamps", .{});
    }
    check.pass("Ling swiglu: the limit is per call, 0 runs unclamped (as the experts' 0), 10 clamps", .{});
}

const Old = struct { topk: cuda.Function, swiglu_512: cuda.Function, swiglu_64: cuda.Function };

/// GLM's calls through the new glue against 6380c8da's kernels (its glm_glue fatbin) on the same inputs: every byte.
fn glmBytes(r: *Rig, set: glue.Set, prev: Old) !void {
    for ([_]usize{ 288, 256, 160 }) |ne| for ([_]bool{ true, false }) |norm| {
        const q: Route = .{ .rows = 300, .ne = ne, .top_k = 8, .ng = 1, .keep = 1, .scale = 2.5 };
        const logits = try f32s(r, q.rows * ne, -8, 8);
        for (logits[0..ne], 0..) |*v, e| v.* = if (e % 5 == 0) 1.0 else v.*; // exact ties in row 0
        const bias = try f32s(r, ne, -0.05, 0.05);
        const n = q.rows * 9;
        const dl = try r.dev(std.mem.sliceAsBytes(logits));
        const db = try r.dev(std.mem.sliceAsBytes(bias));
        const pick = try r.zeros(n * 4);
        const wts = try r.zeros(n * 4);
        var a: cuda.Args = .{};
        inline for (.{ dl, db, pick, wts, q.scale, @as(c_int, @intCast(ne)), @as(c_int, 8), @as(c_int, @intFromBool(norm)) }) |v| a.add(v);
        try cuda.launch.launch(prev.topk, .{ .grid = .{ .x = @intCast(q.rows) }, .block = .{ .x = 32 } }, r.s, &a);
        const pick_new = try r.zeros(n * 4);
        const wts_new = try r.zeros(n * 4);
        try glue.select(set, r.s, dl, db, pick_new, wts_new, q.rows, 8, ne, q.scale, norm, .{});
        try check.sameBytes("GLM select picks vs 6380c8da", try r.back(pick_new, n * 4), try r.back(pick, n * 4));
        try check.sameBytes("GLM select weights vs 6380c8da", try r.back(wts_new, n * 4), try r.back(wts, n * 4));
    };
    for ([_]f32{ 10, 0 }) |lim| for ([_]usize{ 2048, 704 }) |w| {
        const rows = 40;
        const dgu = try r.dev(std.mem.sliceAsBytes(try r.bfs(rows * 2 * w, 16.0)));
        const outs = [2]u64{ try r.zeros(rows * w * 2), try r.zeros(rows * w * 2) };
        const xss = [2]u64{ try r.zeros(rows * w / 64 * 4), try r.zeros(rows * w / 64 * 4) };
        const block: usize = if (w % 512 == 0) 512 else 64;
        var a: cuda.Args = .{};
        inline for (.{ dgu, outs[0], xss[0], lim, @as(i64, @intCast(w)) }) |v| a.add(v);
        try cuda.launch.launch(if (block == 512) prev.swiglu_512 else prev.swiglu_64, .{ .grid = .{ .x = rows, .y = @intCast(w / block) }, .block = .{ .x = if (block == 512) 128 else 64 } }, r.s, &a);
        try glue.swiglu(set, r.s, dgu, outs[1], xss[1], lim, rows, w);
        try check.sameBytes("GLM swiglu vs 6380c8da", try r.back(outs[1], rows * w * 2), try r.back(outs[0], rows * w * 2));
        try check.sameBytes("GLM swiglu sums vs 6380c8da", try r.back(xss[1], rows * w / 64 * 4), try r.back(xss[0], rows * w / 64 * 4));
    };
    check.pass("GLM select (288, 256, 160 experts, normed and not) and swiglu (limits 10 and 0): every byte equals 6380c8da's kernels", .{});
}

const Env = struct {
    arena: std.heap.ArenaAllocator,
    s: cuda.Stream,
    mod: cuda.Module,
    ks: glue.Kernels,
    prng: std.Random.DefaultPrng,
    r: Rig,

    fn deinit(e: *Env) void {
        for (e.r.bufs.items) |*b| b.free();
        e.r.bufs.deinit(e.r.gpu.gpa);
        e.mod.unload();
        e.s.deinit();
        e.arena.deinit();
    }
};

fn env(gpu: Gpu, e: *Env) !void {
    e.arena = .init(gpu.gpa);
    e.s = try cuda.Stream.init(gpu.d, false);
    e.mod = try cuda.Module.load(gpu.d, cuda.kernels.glm_glue);
    e.ks = try glue.Kernels.load(e.mod);
    e.prng = .init(804);
    e.r = .{ .gpu = gpu, .a = e.arena.allocator(), .s = e.s, .rng = e.prng.random() };
}

/// `glm-route-old <fatbin>`: GLM's select and swiglu against 6380c8da's glm_glue fatbin, byte for byte.
pub fn old(gpu: Gpu, path: []const u8) !void {
    var e: Env = undefined;
    try env(gpu, &e);
    defer e.deinit();
    const bytes = try std.Io.Dir.cwd().readFileAllocOptions(gpu.io, path, e.r.a, .limited(1 << 26), .@"64", null);
    var m = try cuda.Module.load(gpu.d, bytes);
    defer m.unload();
    try glmBytes(&e.r, .{ .glue = &e.ks }, .{ .topk = try m.function("tf_glm_topk"), .swiglu_512 = try m.function("tf_glm_swiglu_512"), .swiglu_64 = try m.function("tf_glm_swiglu_64") });
}

/// `glm-route-ling <fixture>`: Ling layer 4's rows (HF's fp32 logits) vs float64, HF's routing and exllamav3's.
pub fn ling(gpu: Gpu, path: []const u8) !void {
    var e: Env = undefined;
    try env(gpu, &e);
    defer e.deinit();
    const r = &e.r;
    const raw = try std.Io.Dir.cwd().readFileAllocOptions(gpu.io, path, r.a, .limited(1 << 26), .@"4", null);
    const hdr = std.mem.bytesAsSlice(i32, raw[0..20]);
    const q: Route = .{ .rows = @intCast(hdr[0]), .ne = @intCast(hdr[1]), .top_k = @intCast(hdr[2]), .ng = @intCast(hdr[3]), .keep = @intCast(hdr[4]), .scale = std.mem.bytesToValue(f32, raw[20..24]) };
    var at: usize = 24;
    const R = q.rows;
    const K = q.top_k;
    const parts = .{ q.rows * q.ne, q.ne, R * K, R * K, R * K, R * K };
    var sl: [6][]align(4) u8 = undefined;
    inline for (parts, 0..) |n, i| {
        sl[i] = @alignCast(raw[at..][0 .. n * 4]);
        at += n * 4;
    }
    try check.expect(at == raw.len, "fixture: {d} bytes, read {d}", .{ raw.len, at });
    const logits = std.mem.bytesAsSlice(f32, sl[0]);
    const bias = std.mem.bytesAsSlice(f32, sl[1]);
    const hf_pick = std.mem.bytesAsSlice(i32, sl[2]);
    const hf_w = std.mem.bytesAsSlice(f32, sl[3]);
    const ex_pick = std.mem.bytesAsSlice(i32, sl[4]);
    const ex_w = std.mem.bytesAsSlice(f32, sl[5]);
    const o = try runSelect(r, .{ .glue = &e.ks }, q, logits, bias, true);
    const t = try against(r, q, logits, bias, o.pick, o.wts);
    std.debug.print("RESULT Ling L4 vs float64: {d} rows, weights at {e:.3} of 2^-16 relative, {d} group and {d} expert near ties (widest gap {e:.2})\n", .{ R, t.worst, t.group_ties, t.expert_ties, t.widest });
    try check.expect(t.worst <= 1, "Ling weights within 2^-16 of float64", .{});
    var hf_same: usize = 0;
    var ex_same: usize = 0;
    var d2: f64 = 0;
    var n2: f64 = 0;
    var hf_most: f64 = 0;
    var ex_most: f64 = 0;
    for (0..R) |row| {
        const mine = o.pick[row * (K + 1) ..][0..K];
        const mw = o.wts[row * (K + 1) ..][0..K];
        if (std.mem.eql(i32, mine, hf_pick[row * K ..][0..K])) {
            hf_same += 1;
            for (mw, hf_w[row * K ..][0..K]) |a, b| {
                d2 += (@as(f64, a) - b) * (@as(f64, a) - b);
                n2 += @as(f64, b) * b;
                hf_most = @max(hf_most, @abs(@as(f64, a) - b) / @abs(@as(f64, b)));
            }
        } else std.debug.print("  row {d}: ours {any} HF {any}\n", .{ row, mine, hf_pick[row * K ..][0..K] });
        var all = true;
        for (ex_pick[row * K ..][0..K], ex_w[row * K ..][0..K]) |id, w| {
            const j = std.mem.indexOfScalar(i32, mine, id) orelse {
                all = false;
                continue;
            };
            ex_most = @max(ex_most, @abs(@as(f64, mw[j]) - w) / @abs(@as(f64, mw[j])));
        }
        if (all) ex_same += 1 else std.debug.print("  row {d}: ours {any} exllamav3 {any}\n", .{ row, mine, ex_pick[row * K ..][0..K] });
    }
    std.debug.print("RESULT Ling L4 vs HF BailingMoeV3Gate: {d}/{d} rows pick the same experts in the same order; weights rel L2 {e:.3}, worst rel {e:.3}\n", .{ hf_same, R, @sqrt(d2 / n2), hf_most });
    std.debug.print("RESULT Ling L4 vs exllamav3 routing_ds3: {d}/{d} rows pick the same set; weights (half) worst rel {e:.3}\n", .{ ex_same, R, ex_most });
    try check.expect(hf_same == R and ex_same == R and hf_most <= std.math.pow(f64, 2, -16) and ex_most <= std.math.pow(f64, 2, -10), "Ling routing equals HF's and exllamav3's", .{});
    check.pass("Ling layer 4 routing: picks equal HF's and exllamav3's on all {d} rows, weights within float64 tolerance", .{R});
}

/// `ling-route-old <fatbin>`: group-limited select (Ling's shapes, odd groups, ties) vs an older glm_glue, bytes.
pub fn lingOld(gpu: Gpu, path: []const u8) !void {
    var e: Env = undefined;
    try env(gpu, &e);
    defer e.deinit();
    const r = &e.r;
    const bytes = try std.Io.Dir.cwd().readFileAllocOptions(gpu.io, path, r.a, .limited(1 << 26), .@"64", null);
    var m = try cuda.Module.load(gpu.d, bytes);
    defer m.unload();
    const prev = try m.function("tf_glm_topk");
    const shapes = [_]Route{
        .{ .rows = 300, .ne = 512, .top_k = 8, .ng = 8, .keep = 4, .scale = 2.5 },
        .{ .rows = 300, .ne = 256, .top_k = 8, .ng = 8, .keep = 4, .scale = 2.5 },
        .{ .rows = 100, .ne = 480, .top_k = 6, .ng = 12, .keep = 3, .scale = 1.0 },
        .{ .rows = 100, .ne = 512, .top_k = 8, .ng = 32, .keep = 1, .scale = 2.5 },
        .{ .rows = 100, .ne = 384, .top_k = 8, .ng = 4, .keep = 2, .scale = 2.5 },
    };
    var rows: usize = 0;
    for (shapes) |q| for ([_]bool{ true, false }) |norm| for ([_]bool{ false, true }) |ties| {
        const logits = try f32s(r, q.rows * q.ne, -6, 6);
        const bias = try f32s(r, q.ne, -0.05, 0.05);
        if (ties) tieRows(r, q, logits, bias);
        const n = q.rows * (q.top_k + 1);
        const dl = try r.dev(std.mem.sliceAsBytes(logits));
        const db = try r.dev(std.mem.sliceAsBytes(bias));
        const outs = [4]u64{ try r.zeros(n * 4), try r.zeros(n * 4), try r.zeros(n * 4), try r.zeros(n * 4) };
        var a: cuda.Args = .{};
        inline for (.{ dl, db, outs[0], outs[1], q.scale, @as(c_int, @intCast(q.ne)), @as(c_int, @intCast(q.top_k)), @as(c_int, @intFromBool(norm)), @as(c_int, @intCast(q.ng)), @as(c_int, @intCast(q.keep)) }) |v| a.add(v);
        try cuda.launch.launch(prev, .{ .grid = .{ .x = @intCast(q.rows) }, .block = .{ .x = 32 } }, r.s, &a);
        try glue.select(.{ .glue = &e.ks }, r.s, dl, db, outs[2], outs[3], q.rows, q.top_k, q.ne, q.scale, norm, .{ .n = q.ng, .keep = q.keep });
        try check.sameBytes("group-limited select picks vs the older build", try r.back(outs[2], n * 4), try r.back(outs[0], n * 4));
        try check.sameBytes("group-limited select weights vs the older build", try r.back(outs[3], n * 4), try r.back(outs[1], n * 4));
        rows += q.rows;
    };
    check.pass("ling-route-old: {d} rows of group-limited select (5 shapes, normed and not, tie rows) equal the older kernel's bytes", .{rows});
}
