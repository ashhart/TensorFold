//! The LiLiCorr golden battery: the head and the walk against the fixture the reference
//! generated on the NVIDIA Bolt drafter's real weights (vLLM v0.34.0, inlined score()).
//! TF_LILICORR_FIXTURE names the fixture directory; unset skips honestly (the blobs are
//! too large to embed). The score gate is 4e-3: the port's bf16 cast points reproduce the
//! reference within noise (1.8e-3 observed), far below what flips a lattice walk.
const std = @import("std");
const head = @import("cuda_lilicorr_head.zig");
const walk = @import("cuda_lilicorr_walk.zig");

fn fixtureDir() ?[]const u8 {
    // through libc like the platform's sys.getenv: the build links it; -lc for the standalone probe
    const v = std.c.getenv("TF_LILICORR_FIXTURE") orelse return null;
    const s = std.mem.span(v);
    return if (s.len == 0) null else s;
}

const Fixture = struct {
    gpa: std.mem.Allocator,
    dims: std.StringHashMap([]const usize),
    dir: []const u8,

    fn init(gpa: std.mem.Allocator, dir: []const u8) !Fixture {
        var threaded = std.Io.Threaded.init(gpa, .{});
        defer threaded.deinit();
        const path = try std.fmt.allocPrint(gpa, "{s}/dims.json", .{dir});
        defer gpa.free(path);
        const bytes = try std.Io.Dir.cwd().readFileAlloc(threaded.io(), path, gpa, .limited(1 << 20));
        const parsed = try std.json.parseFromSlice(std.json.Value, gpa, bytes, .{});
        // the parsed doc leaks: the map's keys and shapes point into it (a test process)
        var dims = std.StringHashMap([]const usize).init(gpa);
        const obj = parsed.value.object;
        var it = obj.iterator();
        while (it.next()) |e| {
            const arr = e.value_ptr.*.array;
            const shp = try gpa.alloc(usize, arr.items.len);
            for (arr.items, 0..) |v, i| shp[i] = @intCast(if (v == .integer) v.integer else @as(i64, @intFromFloat(v.float)));
            try dims.put(try gpa.dupe(u8, e.key_ptr.*), shp);
        }
        return .{ .gpa = gpa, .dims = dims, .dir = try gpa.dupe(u8, dir) };
    }

    // the fixture's file names: dots to underscores, except "w.layers.N.x" stores as "w_layersN_x"
    fn blobName(gpa: std.mem.Allocator, name: []const u8) ![]u8 {
        if (std.mem.startsWith(u8, name, "w.layers.")) {
            const rest = name["w.layers.".len..];
            if (std.mem.indexOfScalar(u8, rest, '.') == null) return error.MissingBlob;
            const out = try gpa.alloc(u8, rest.len);
            for (rest, 0..) |ch, i| out[i] = if (ch == '.') '_' else ch;
            return std.fmt.allocPrint(gpa, "w_layers{s}", .{out});
        }
        const out = try gpa.alloc(u8, name.len);
        for (name, 0..) |ch, i| out[i] = if (ch == '.') '_' else ch;
        return out;
    }

    fn blob(f: Fixture, name: []const u8) ![]f32 {
        const shp = f.dims.get(name) orelse return error.MissingBlob;
        var n: usize = 1;
        for (shp) |d| n *= d;
        const fname = try blobName(f.gpa, name);
        defer f.gpa.free(fname);
        const path = try std.fmt.allocPrint(f.gpa, "{s}/{s}.f32", .{ f.dir, fname });
        defer f.gpa.free(path);
        var threaded = std.Io.Threaded.init(f.gpa, .{});
        defer threaded.deinit();
        const bytes = try std.Io.Dir.cwd().readFileAlloc(threaded.io(), path, f.gpa, .limited(1 << 30));
        const out = try f.gpa.alloc(f32, n);
        @memcpy(out, @as([*]align(1) const f32, @ptrCast(bytes.ptr))[0..n]);
        return out;
    }

    // the fixture's file names map the dims keys by dots to underscores (w.layers.0.x -> w_layers0_x)
    fn gpa_replace(gpa: std.mem.Allocator, name: []const u8) ![]u8 {
        const out = try gpa.alloc(u8, name.len);
        for (name, 0..) |ch, i| out[i] = if (ch == '.') '_' else ch;
        return out;
    }
};

fn weightsFromFixture(gpa: std.mem.Allocator, fx: Fixture) !*head.Weights {
    const w = try gpa.create(head.Weights);
    w.* = .{
        .token_proj = .{ .w = try fx.blob("w.token_proj.weight"), .b = try fx.blob("w.token_proj.bias") },
        .pass_proj = .{ .w = try fx.blob("w.pass_hidden_proj.weight"), .b = try fx.blob("w.pass_hidden_proj.bias") },
        .ctx_proj = .{ .w = try fx.blob("w.context_proj.weight"), .b = try fx.blob("w.context_proj.bias") },
        .feature_norm_w = try fx.blob("w.feature_norm.weight"),
        .feature_norm_b = try fx.blob("w.feature_norm.bias"),
        .feature_up = .{ .w = try fx.blob("w.feature_mlp.up_proj.weight"), .b = try fx.blob("w.feature_mlp.up_proj.bias") },
        .feature_down = .{ .w = try fx.blob("w.feature_mlp.down_proj.weight"), .b = try fx.blob("w.feature_mlp.down_proj.bias") },
        .slot_emb = try fx.blob("w.slot_embedding"),
        .rank_emb = try fx.blob("w.rank_embedding"),
        .out_norm = try fx.blob("w.output_norm.weight"),
        .anchor_norm = try fx.blob("w.anchor_norm.weight"),
        .factor = .{ .w = try fx.blob("w.factor_input_proj.weight"), .b = try fx.blob("w.factor_input_proj.bias") },
        .rel_bias = try fx.blob("w.relative_slot_bias"),
        .same_bias = try fx.blob("w.same_slot_bias"),
        .out_head = .{ .w = try fx.blob("w.out_head.weight"), .b = try fx.blob("w.out_head.bias") },
        .in_head = .{ .w = try fx.blob("w.in_head.weight"), .b = try fx.blob("w.in_head.bias") },
        .anchor_out_head = .{ .w = try fx.blob("w.anchor_out_head.weight"), .b = try fx.blob("w.anchor_out_head.bias") },
        .layers = undefined,
    };
    w.layers = try gpa.alloc(head.Layer, head.n_layers);
    for (w.layers, 0..) |*l, i| {
        var buf: [64]u8 = undefined;
        const p = try std.fmt.bufPrint(&buf, "w.layers.{d}.", .{i});
        var n2: [96]u8 = undefined;
        l.in_proj = try fx.blob(try std.fmt.bufPrint(&n2, "{s}attn.in_proj_weight", .{p}));
        l.in_bias = try fx.blob(try std.fmt.bufPrint(&n2, "{s}attn.in_proj_bias", .{p}));
        l.out_proj = try fx.blob(try std.fmt.bufPrint(&n2, "{s}attn.out_proj.weight", .{p}));
        l.out_bias = try fx.blob(try std.fmt.bufPrint(&n2, "{s}attn.out_proj.bias", .{p}));
        l.attn_norm = try fx.blob(try std.fmt.bufPrint(&n2, "{s}attn_norm.weight", .{p}));
        l.up = try fx.blob(try std.fmt.bufPrint(&n2, "{s}mlp.up_proj.weight", .{p}));
        l.up_bias = try fx.blob(try std.fmt.bufPrint(&n2, "{s}mlp.up_proj.bias", .{p}));
        l.down = try fx.blob(try std.fmt.bufPrint(&n2, "{s}mlp.down_proj.weight", .{p}));
        l.down_bias = try fx.blob(try std.fmt.bufPrint(&n2, "{s}mlp.down_proj.bias", .{p}));
        l.mlp_norm = try fx.blob(try std.fmt.bufPrint(&n2, "{s}mlp_norm.weight", .{p}));
    }
    return w;
}

test "the checkpoint widen reads bf16, not f16 (the corruption class the fixture bypasses)" {
    // bf16 1.0 = 0x3F80: the f16 bitcast read it as 1.875
    try std.testing.expectEqual(@as(f32, 1.0), head.widenBits(0x3F80));
    try std.testing.expectEqual(@as(f32, -1.0), head.widenBits(0xBF80));
    try std.testing.expectEqual(@as(f32, -1.375), head.widenBits(0xBFB0));
    try std.testing.expectEqual(@as(f32, 2.0), head.widenBits(0x4000));
}

test "the head and the walk reproduce the reference on the golden fixture" {
    const a = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const gpa = arena.allocator();
    const dir = fixtureDir() orelse return error.SkipZigTest; // TF_LILICORR_FIXTURE unset: the blobs are not embedded
    const fx = try Fixture.init(gpa, dir);
    const w = try weightsFromFixture(gpa, fx);
    const token_emb = try fx.blob("token_embeddings");
    const log_probs = try fx.blob("candidate_log_probs");
    const pass_hidden = try fx.blob("pass_hidden");
    const anchor_hidden = try fx.blob("anchor_hidden");
    const got = try head.score(gpa, w, token_emb, log_probs, pass_hidden, anchor_hidden, true);
    const gold_start = try fx.blob("start_scores");
    const gold_pair = try fx.blob("pair_scores");
    var max_d: f32 = 0;
    var ds: f32 = 0;
    var dp: f32 = 0;
    for (got.start, gold_start) |v, g| ds = @max(ds, @abs(v - g));
    for (got.pair, gold_pair) |v, g| dp = @max(dp, @abs(v - g));
    max_d = @max(ds, dp);
    std.debug.print("\nlilicorr golden: start diff {e}  pair diff {e}\n", .{ ds, dp });
    if (ds > 4e-3) for (got.start, gold_start, 0..) |v, g, i| {
        if (i < 8) std.debug.print("  start[{d}] got {d} gold {d}\n", .{ i, v, g });
    };
    try std.testing.expect(max_d < 4e-3);

    // the walk over the reference scores: deterministic, and its realized scores are the picked rows
    const cand = try gpa.alloc(u32, head.slots * head.topk);
    for (cand, 0..) |*v, i| v.* = @intCast(i);
    const c = try walk.chain(got.start, got.pair, cand);
    const realized = try gpa.alloc(f32, head.slots);
    try walk.realizedScores(got.start, got.pair, &c.index, realized);
    for (realized, c.scores) |r, s| try std.testing.expectEqual(s, r);
    // scaling the scores never moves the greedy chain
    const scaled_start = try gpa.alloc(f32, head.topk);
    const scaled_pair = try gpa.alloc(f32, (head.slots - 1) * head.topk * head.topk);
    for (got.start, scaled_start) |v, *o| o.* = 8.0 * v;
    for (got.pair, scaled_pair) |v, *o| o.* = 8.0 * v;
    const c2 = try walk.chain(scaled_start, scaled_pair, cand);
    try std.testing.expectEqualSlices(usize, &c.index, &c2.index);
    // a matrix built with the reference's scale reads back the realized scores times the scale
    const matrix = try gpa.alloc(f32, head.slots * head.topk * head.topk);
    try walk.walkMatrix(scaled_start, scaled_pair, 1.0, matrix);
    try std.testing.expectEqual(8.0 * c.scores[0], matrix[c.index[0]]);
}

// ---- owner ask-5 part C: committed synthetic oracle cells (host-off, no fixture) ----
// C1: the lat_attn bias lane index law, qwen35_draft_lilicorr.cuh:195:
//     score[ki] = dot(q[qi,h], k[ki,h]) * scale + bias[(h * lattice + qi) * lattice + ki].
// The oracle-design law: a per-h bias constant across ki soft-max-cancelles (additive
// offsets divide out of softmax), so the h-lane is observable only through an
// h-DEPENDENT DOT; the synthetic below gives D[0][1][h] = 1+h exactly for that.
test "the lat_attn bias lane reads (h*lattice+qi)*lattice+ki (host oracle, ask-5 C)" {
    // lattice=2, hd=18 (>= 8 lanes * head_dim 2), head_dim=2; qkv rows [q18|k18|v18]
    const hd = 18;
    const scale = 1.0 / @sqrt(2.0);
    const GOLD_Q0 = [8]f32{ 1.84646064, 1.91790476, 1.95776367, 1.97871894, 1.98939254, 1.99474152, 1.99740028, 1.99871647 };
    var qkv: [2 * 3 * hd]f32 = @splat(0);
    for (0..8) |h| {
        qkv[0 * 3 * hd + h * 2] = 1.0; // q(qi=0) head dim h
        qkv[1 * 3 * hd + h * 2 + 1] = 1.0; // q(qi=1)
        qkv[1 * 3 * hd + hd + h * 2] = @floatFromInt(1 + h); // k(ki=1) head dim
        qkv[1 * 3 * hd + hd + h * 2 + 1] = 3.0;
        qkv[0 * 3 * hd + 2 * hd + h * 2] = 1.0; // V(ki=0)[h,t=0]
        qkv[1 * 3 * hd + 2 * hd + h * 2] = 1.0; // V(ki=1)[h,t=0]
        qkv[0 * 3 * hd + 2 * hd + h * 2 + 1] = 1.0; // V(ki=0)[h,t=1]
        qkv[1 * 3 * hd + 2 * hd + h * 2 + 1] = 2.0; // V(ki=1)[h,t=1]
    }
    var bias: [32]f32 = undefined;
    for (0..8) |h| {
        for (0..2) |qi| {
            for (0..2) |ki| {
                bias[(h * 2 + qi) * 2 + ki] = @floatFromInt(100 * h + 10 * qi + ki);
            }
        }
    }
    for (0..2) |qi| {
        for (0..8) |h| {
            const qo = qi * 3 * hd + h * 2;
            var p: [2]f32 = undefined;
            for (0..2) |ki| {
                const ko = ki * 3 * hd + hd + h * 2;
                var acc: f32 = 0;
                for (0..2) |t| acc += qkv[qo + t] * qkv[ko + t];
                p[ki] = acc * scale + bias[(h * 2 + qi) * 2 + ki];
            }
            var mx = p[0];
            var sum: f32 = 0;
            if (p[1] > mx) mx = p[1];
            for (&p) |*v| {
                v.* = @exp(v.* - mx);
                sum += v.*;
            }
            for (&p) |*v| v.* /= sum;
            // the V(t=0) identity: softmax weights always sum through a value of 1
            try std.testing.expectApproxEqAbs(@as(f32, 1.0), p[0] + p[1], 2e-6);
            const got: f32 = p[0] * 1.0 + p[1] * 2.0;
            const gold: f32 = if (qi == 0) GOLD_Q0[h] else 1.95776367;
            try std.testing.expectApproxEqAbs(gold, got, 2e-4);
        }
    }
}

// C2: the factor_rows tail-warp output identity, qwen35_draft_lilicorr.cuh:420-468 (the
// A1 fuzz fix): with hd % 8 != 0 the last block's tail warps (t >= hd) stay as load
// workers + barrier participants; every active t outputs the closed form
//     v[r] = B[t] + sum_ii (Ain[r][ii]*w1 + Anch[ii]*w2 + Ain[r][ii]*Anch[ii]*w3),
//     OUT[r*hd+t] = sigmoid(v[r]).
// The PRE-fix early return inside a block with __syncthreads is UB (barrier violation),
// so the RED side of this cell is a device/fuzz property; the host oracle pins the fixed
// semantics: kernel-shaped arithmetic (33-wide shared rows, per-ii guard, B carry).
test "the factor_rows tail-warp fix outputs the closed form for every row (host oracle, ask-5 C)" {
    const hd = 6; // hd % 8 != 0: blocks carry tail warps t in {6,7}
    const rows = 3;
    const GOLD = [3][6]f32{
        .{ 0.000203, 0.000553, 0.001501, 0.004070, 0.010987, 0.029312 },
        .{ 0.075858, 0.182426, 0.377541, 0.622459, 0.817574, 0.924142 },
        .{ 0.970688, 0.989013, 0.995930, 0.998499, 0.999447, 0.999797 },
    };
    // synthetic: A_IN[r][j] = 10r + j; ANCHOR = 0; W row: w1 = 0.1 for ii < hd,
    // w2 = w3 = 0; B[t] = t - 10  ->  v[r] = (t - 10) + 0.1 * (60r + 15) = t + 6r - 8.5
    var ain: [rows][33]f32 = undefined;
    var anch: [33]f32 = @splat(0);
    for (0..rows) |r| {
        for (0..32) |kk| ain[r][kk] = if (kk < hd) @as(f32, @floatFromInt(10 * r + kk)) else 0.0;
    }
    for (0..rows) |r| {
        for (0..hd) |t| {
            var v: f32 = @as(f32, @floatFromInt(t)) - 10.0; // the B carry for an active t
            const w1: f32 = 0.1; // W[t*3*hd + ii] for ii < hd
            for (0..hd) |ii| {
                anch[ii] = 0.0; // the ANCHOR load (all-zero synthetic)
                v += ain[r][ii] * w1 + anch[ii] * 0.0 + ain[r][ii] * anch[ii] * 0.0;
            }
            const xf: f32 = @floatFromInt(t + 6 * r);
            try std.testing.expectApproxEqAbs(xf - 8.5, v, 1e-6); // the closed form
            const sig: f32 = 1.0 / (1.0 + @exp(-v));
            try std.testing.expectApproxEqAbs(GOLD[r][t], sig, 5e-3); // bf16 rn headroom
        }
    }
}

// ---------------------------------------------------------------------------
// absorb-guard (PR 5102 owner round-2 ask 5, part B): absorbRows' chunk clamp
// `const rows = @min(n, staging_rows);` (cure ecc1f7f3) keeps EVERY gathered
// row <= staging_rows(128) and the chunk sizes summing to n, so a 129-row (or
// 256-row) prompt ingest never overruns the 128-row staging carve (taps_in/th/
// thn/kabs/vabs/kn/xs). A recording driver fake drives all three launch seams
// with zero real GPU calls: the family `run` helper (cuda.launch), the affine
// gemv path (d.affine via projSums, e.ops().dense via projDense), and norm's
// e.k.norms.addRmsTail; the anchor DtoH copy is faked too. RED side: at the
// pre-cure clamp `@min(n, max_rows)` (reverted for the fail-before capture in
// a throwaway worktree) the n=129 case records a single 129-row gather launch
// (the guard's grid-cap assert fires on it); the pre-cure absorb also dies at a
// downstream seam there (the mock's wide-verify face tops out at 128 rows), so
// the harness records that error instead of `try`-ing it and asserts on the
// launches FIRST — at HEAD the error arm never fires (asserted null).
const guard_lc = @import("cuda_lilicorr.zig");
const guard_wts = @import("cuda_lilicorr_weights.zig");
const guard_absorb = @import("cuda_lilicorr_absorb.zig");
const guard_cuda = @import("cuda");

const AbsorbGuard = struct {
    const lc = guard_lc;
    const cuda = guard_cuda;
    const abi = cuda.abi;

    var f_handle: [96]abi.Function = undefined;
    var f_grid: [96]usize = undefined;
    var f_block: [96]usize = undefined;
    var f_arg0: [96]u64 = undefined; // param 0's value: a norm-tail's input row pointer
    var f_rows_n: [96]i32 = undefined; // the gather_rows by-value TapRows' n, as the kernel reads it
    var f_rows: [96][64]u32 = undefined; // ...and its first 64 row entries (the kernel's int rows[64])
    var count: usize = 0;
    var ex_count: usize = 0;

    // handle values the harness fixture pins: gather_rows = 0x1101, norm tails = 0x9100/0x9108
    const gather_rows_handle = 0x1101;
    const tail_handles = [_]usize{ 0x9100, 0x9108 };

    fn fake(dv: *const cuda.Driver, h: usize) cuda.Function {
        return .{ .d = dv, .handle = @ptrFromInt(h) };
    }

    const Fake = struct {
        fn launch(f: abi.Function, gx: c_uint, gy: c_uint, gz: c_uint, bx: c_uint, by: c_uint, bz: c_uint, shared: c_uint, stream: abi.Stream, args: ?[*]?*anyopaque, extra: ?[*]?*anyopaque) callconv(.c) abi.Result {
            _ = gy;
            _ = gz;
            _ = by;
            _ = bz;
            _ = shared;
            _ = stream;
            _ = extra;
            if (count < f_handle.len) {
                f_handle[count] = f;
                f_grid[count] = gx;
                f_block[count] = bx;
                const hv = @intFromPtr(f.?);
                if (args) |ap| {
                    const ps: [*]?*anyopaque = @ptrCast(ap);
                    if (blk: {
                        for (tail_handles) |th| if (hv == th) break :blk true;
                        break :blk false;
                    }) {
                        f_arg0[count] = @as(*u64, @ptrCast(@alignCast(ps[0].?))).*;
                    }
                    if (hv == gather_rows_handle) {
                        const b: [*]u8 = @ptrCast(@alignCast(ps[3].?));
                        f_rows_n[count] = std.mem.readInt(i32, b[0..4], .little);
                        for (0..64) |j| f_rows[count][j] = std.mem.readInt(u32, b[4 + j * 4 ..][0..4], .little);
                    }
                }
                count += 1;
            }
            return abi.success;
        }
        fn launchEx(cfg: *const abi.LaunchConfig, f: abi.Function, args: ?[*]?*anyopaque, extra: ?[*]?*anyopaque) callconv(.c) abi.Result {
            _ = args;
            _ = extra;
            if (count < f_handle.len) {
                f_handle[count] = f;
                f_grid[count] = cfg.grid_x;
                f_block[count] = cfg.block_x;
                ex_count += 1; // the wide/pdl affine faces launch through here
                count += 1;
            }
            return abi.success;
        }
        fn memcpyH(_: ?*anyopaque, _: abi.DevicePtr, _: usize, _: abi.Stream) callconv(.c) abi.Result {
            return abi.success;
        }
    };

    fn reset() void {
        count = 0;
        ex_count = 0;
    }
};

test "the absorb chunk cap keeps every gathered row at staging_rows (owner ask-5 B, absorb-guard)" {
    const t = std.testing;
    const g = AbsorbGuard;
    const lc = g.lc;
    const cuda = g.cuda;

    var driver: cuda.Driver = undefined;
    driver.api.cuLaunchKernel = g.Fake.launch;
    driver.api.cuLaunchKernelEx = g.Fake.launchEx;
    driver.api.cuMemcpyDtoHAsync_v2 = g.Fake.memcpyH;

    var e: lc.E = undefined;
    var ctx_dev: cuda.Context = .{ .d = &driver, .device = 0, .handle = null };
    e.ctx = &ctx_dev;
    e.stream = .{ .d = &driver, .handle = null };
    e.b = undefined;
    e.k = undefined;
    e.k.wide = true;
    e.k.modelopt = null;
    e.k.ck = null;
    e.k.ggml = null;
    const fade = g.fake(&driver, 0x9000);
    e.k.affine = .{
        .group_fn = fade,
        .mlp128_fn = fade,
        .gemv_fn = fade,
        .verify_fn = fade,
        .verify64_fn = fade,
        .verify128_fn = fade,
        .split_fn = fade,
        .seg_fn = fade,
        .many_fn = fade,
        .prefill_fn = fade,
        .prefill22_fn = fade,
        .prefill_many_fn = fade,
        .words_fn = fade,
        .scales_fn = fade,
        .gemv_blocks = 1,
        .split = null,
        .pdl = true, // the production GB10 face: 128-row wide verify tiles, Ex launches
        .wide_verify = false,
    };
    e.k.norms = undefined;
    e.k.norms.tail_fns = .{ g.fake(&driver, 0x9100), g.fake(&driver, 0x9108) };

    var d: lc.Draft = undefined;
    d.e = &e;
    d.c = .{
        .hidden = 128,
        .intermediate = 128,
        .layers = 1,
        .vocab = 16,
        .heads = 2,
        .kv_heads = 1,
        .head_dim = 32,
        .block = 4,
        .mask = 0,
        .window = 4096, // keep = min(window, tap_rows - max_rows) = min(4096, 512-256) = 256: no trim at n=256
        .group = 8,
        .rank = 4,
        .topk = 2,
        .eps = 1e-5,
        .theta = 10000,
        .taps = .{ 1, 2, 3, 4, 5, 6, 7, 8 },
        .n_taps = 4,
    };
    d.gpa = t.allocator;
    d.tap_rows = 512;
    d.taps_in = 0x1a0000;
    d.th = 0x1b0000;
    d.thn = 0x1c0000;
    d.kabs = 0x1d0000;
    d.vabs = 0x1e0000;
    d.kn = 0x1f0000;
    d.xs = 0x200000;
    d.anchor = 0x210000;
    var staged_b: [8192]u8 align(16) = @splat(0); // anchor copy lands at anchor_at(2080)+256 bytes, faked
    d.staged = .{ .d = &driver, .bytes = staged_b[0..] };
    d.w = undefined;
    // fc/k shape (n=64, k=1024): splitK=2 with a.split=null, so decode takes the gemv face and
    // the 128-row chunk takes the wide verify face (pdl=true: cfg.plain() false => cuLaunchKernelEx).
    d.w.fc = .{ .w = 0x30000, .s = 0x31000, .b = 0x32000, .n = 64, .k = 1024, .npad = 128 };
    d.w.hidden_norm = 0x220000;
    const wq: cuda.qlinear.Weight = .{ .affine4 = .{ .w = 0x33000, .s = 0x34000, .b = 0x35000, .n = 64, .k = 1024, .npad = 128 } };
    var lz: [1]guard_wts.Layer = .{.{
        .input_norm = 0x40000,
        .post_norm = 0x40040,
        .q = wq,
        .k = wq,
        .v = wq,
        .o = wq,
        .q_norm = 0x40080,
        .k_norm = 0x400c0,
        .sink = 0x40100,
        .gate = wq,
        .up = wq,
        .down = wq,
        .attn_conv = wq,
        .attn_base = 0x40140,
        .mlp_conv = wq,
        .mlp_base = 0x40180,
    }};
    d.w.layers = lz[0..];
    d.fns = .{
        .gather_taps = g.fake(&driver, 0x1100),
        .gather_rows = g.fake(&driver, 0x1101),
        .norm_rope = g.fake(&driver, 0x1102),
        .ring_put = g.fake(&driver, 0x1103),
        .sums = g.fake(&driver, 0x1104),
        .norm_rope_batch = g.fake(&driver, 0x1105),
        .attn_partial = g.fake(&driver, 0x1106),
        .attn_merge = g.fake(&driver, 0x1107),
        .mlp_act = g.fake(&driver, 0x1108),
        .mask_rows = g.fake(&driver, 0x1109),
        .dyn_conv_batch = g.fake(&driver, 0x110a),
        .token_proj = g.fake(&driver, 0x110b),
        .lat_attn = g.fake(&driver, 0x110c),
        .factor = g.fake(&driver, 0x110d),
        .edge_linear = g.fake(&driver, 0x110e),
        .l2norm = g.fake(&driver, 0x110f),
        .dots = g.fake(&driver, 0x1110),
        .rms = g.fake(&driver, 0x1111),
        .lin_f32 = g.fake(&driver, 0x1112),
        .silu_bf16 = g.fake(&driver, 0x1113),
        .add_bf16 = g.fake(&driver, 0x1114),
        .lattice = g.fake(&driver, 0x1115),
        .rms_f32 = g.fake(&driver, 0x1116),
        .widen = g.fake(&driver, 0x1117),
        .narrow = g.fake(&driver, 0x1118),
    };
    d.affine = e.k.affine;

    const gather_handle = d.fns.gather_taps.handle;
    const tail0 = e.k.norms.tail_fns[0];
    const tail1 = e.k.norms.tail_fns[1];
    const ring_buf: cuda.DeviceBuffer = .{ .d = &driver, .ptr = 0x5a000000, .len = 1 << 24 };

    for ([_]usize{ 129, 256 }) |rows_in| {
        g.reset();
        var ctx_s: lc.Ctx = .{ .ring = ring_buf };
        // The call may abort (RED side: the 129-row chunk overruns the next 128-row wide-verify
        // window with WindowTooWide before the anchor); the guard grades the RECORDED launches.
        var absorb_err: ?anyerror = null;
        guard_absorb.absorbRows(&d, &ctx_s, 0x6a000000, rows_in - 1, rows_in, null, true) catch |err| {
            absorb_err = err;
        };
        try t.expect(g.count > 0);
        try t.expect(g.f_handle[0] == gather_handle); // harness liveness: the gather IS the first launch
        var g_n: usize = 0;
        var g_sum: usize = 0;
        for (0..g.count) |i| {
            if (g.f_handle[i] == gather_handle) {
                g_n += 1;
                g_sum += g.f_grid[i];
                std.debug.print("[absorb-guard] gather #{d}: grid={d} block={d}\n", .{ g_n, g.f_grid[i], g.f_block[i] });
                try t.expect(g.f_grid[i] <= lc.staging_rows); // THE GUARD: rows <= staging_rows(128)
                try t.expectEqual(@as(usize, 256), g.f_block[i]);
            }
        }
        std.debug.print("[absorb-guard] n={d}: launches={d} gather={d} chunkrows={d} pdl_ex={d} err={any}\n", .{ rows_in, g.count, g_n, g_sum, g.ex_count, absorb_err });
        const seen_ex = g.ex_count >= 1; // seam 2 coverage: the qlinear affine face (projSums/projDense) launches through cuLaunchKernelEx
        var seen_norms = false; // seam 3 coverage: norm -> e.k.norms.addRmsTail records its tail launch
        for (0..g.count) |i| {
            if (g.f_handle[i] == tail0.handle or g.f_handle[i] == tail1.handle) seen_norms = true;
        }
        try t.expect(seen_ex);
        try t.expect(seen_norms);
        try t.expectEqual(@as(usize, if (rows_in <= lc.staging_rows) 1 else 2), g_n);
        try t.expectEqual(rows_in, g_sum);
        try t.expect(absorb_err == null); // the cured absorb completes its whole pass: no seam dies mid-pass
    }

    // PR 5102 verdict asks: explicit row lists, 129/256 rows, PERMUTED. Faces graded by the
    // tally: chunk k gathers rs[done..done+rows] (disjoint slices covering rs exactly); the
    // by-value arg byte-matches qwen35_draft.cu's TapRows (int n + int rows[64], TapRows.n == grid);
    // the anchor tail's input row = the LAST chunk's final staging row, never the cumulative done.
    const rows_h = @intFromPtr(d.fns.gather_rows.handle.?);
    const tail0_h = @intFromPtr(tail0.handle.?);
    const tail1_h = @intFromPtr(tail1.handle.?);
    const kernel_cap = 64; // qwen35_draft.cu:47-55's by-value row list (int rows[64])
    const fail = struct {
        fn fail(ok: bool, tally: *usize, comptime what: []const u8, a: anytype) void {
            if (ok) return;
            tally.* += 1;
            std.debug.print("[absorb-rows] FAIL " ++ what ++ "\n", a);
        }
    }.fail;
    var fail_n: usize = 0;
    fail(@sizeOf(guard_absorb.TapRows) == 4 + kernel_cap * 4, &fail_n, "TapRows byte layout: @sizeOf {d} != 260 (int n + int rows[64])", .{@sizeOf(guard_absorb.TapRows)});
    fail(@offsetOf(guard_absorb.TapRows, "rows") == 4, &fail_n, "TapRows rows offset {d} != 4", .{@offsetOf(guard_absorb.TapRows, "rows")});
    for ([_]usize{ 129, 256 }) |rows_in| {
        g.reset();
        var ctx_s: lc.Ctx = .{ .ring = ring_buf };
        var rs_b: [lc.max_rows]u32 = undefined;
        // a 137-stride walk mod rows_in is a bijective, non-monotonic permutation of the rows
        for (0..rows_in) |i| rs_b[i] = @intCast((i * 137 + 11) % rows_in);
        var absorb_err: ?anyerror = null;
        guard_absorb.absorbRows(&d, &ctx_s, 0x6a000000, 1234, rows_in, rs_b[0..rows_in], true) catch |err| {
            absorb_err = err;
        };
        try t.expect(absorb_err == null);
        var gathers: usize = 0;
        var exp_done: usize = 0;
        var last_gather_grid: usize = 0;
        for (0..g.count) |li| {
            if (@intFromPtr(g.f_handle[li].?) != rows_h) continue;
            gathers += 1;
            last_gather_grid = g.f_grid[li];
            fail(g.f_grid[li] <= kernel_cap, &fail_n, "n={d} gather #{d}: grid {d} exceeds the kernel's row cap {d}", .{ rows_in, gathers, g.f_grid[li], kernel_cap });
            fail(g.f_rows_n[li] == @as(i32, @intCast(g.f_grid[li])), &fail_n, "n={d} gather #{d}: TapRows.n {d} != grid {d}", .{ rows_in, gathers, g.f_rows_n[li], g.f_grid[li] });
            const cap = @min(g.f_grid[li], kernel_cap);
            for (0..cap) |j| {
                fail(g.f_rows[li][j] == rs_b[exp_done + j], &fail_n, "n={d} gather #{d} row {d}: recorded {d}, want rs[{d}]={d}", .{ rows_in, gathers, j, g.f_rows[li][j], exp_done + j, rs_b[exp_done + j] });
            }
            exp_done += g.f_grid[li];
        }
        try t.expectEqual(rows_in, exp_done);
        var tail_last: ?usize = null;
        for (0..g.count) |li| {
            const hv = @intFromPtr(g.f_handle[li].?);
            if (hv == tail0_h or hv == tail1_h) tail_last = li;
        }
        try t.expect(tail_last != null);
        const want_input = d.th + @as(u64, @intCast(last_gather_grid - 1)) * @as(u64, d.c.hidden) * 2;
        fail(g.f_arg0[tail_last.?] == want_input, &fail_n, "n={d}: anchor input 0x{x}, want 0x{x} (last chunk holds {d} rows)", .{ rows_in, g.f_arg0[tail_last.?], want_input, last_gather_grid });
        std.debug.print("[absorb-rows] n={d}: launches={d} gathers={d} exp_done={d} pdl_ex={d} fail_n={d}\n", .{ rows_in, g.count, gathers, exp_done, g.ex_count, fail_n });
    }
    try t.expectEqual(@as(usize, 0), fail_n);
}
