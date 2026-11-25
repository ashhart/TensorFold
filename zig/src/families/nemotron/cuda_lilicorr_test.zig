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
