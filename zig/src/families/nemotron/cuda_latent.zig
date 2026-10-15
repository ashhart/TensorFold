//! A latent MoE on CUDA (Nemotron 3.5 Super): expert widths in shares summed in share order, so TP=1 bits == TP=2's.

const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const Config = @import("config.zig").Config;
const kern = @import("cuda_kernels.zig");
const W = @import("cuda_weights.zig");
const modelopt = @import("cuda_modelopt.zig");
const fwd = @import("cuda_forward.zig");

const Nvfp4Experts = cuda.experts.Nvfp4Experts;
const Tensor = core.checkpoint.Tensor;

/// A layer's weights past the router: hidden to latent and back (bf16), routed and shared experts by share.
pub const Latent = struct { into: kern.QLinear, out: kern.QLinear, routed: kern.Experts, shared: kern.Experts };

/// nemotron_latent.cu's kernels.
pub const Fns = struct {
    mod: cuda.Module,
    topk: cuda.Function,
    combine: cuda.Function,
    moe_out: cuda.Function,
    share_ids: cuda.Function,

    pub fn load(d: *const cuda.Driver) !Fns {
        var mod = try cuda.Module.load(d, cuda.kernels.nemotron_latent);
        errdefer mod.unload();
        return .{
            .mod = mod,
            .topk = try mod.function("tf_super_topk"),
            .combine = try mod.function("tf_super_combine"),
            .moe_out = try mod.function("tf_super_moe_out"),
            .share_ids = try mod.function("tf_super_share_ids"),
        };
    }

    pub fn unload(x: *Fns) void {
        x.mod.unload();
    }
};

/// Picks a row the wide top-k keeps (one a lane), and experts it ranks (32 a lane).
pub const max_picks = 32;
pub const max_experts = 1024;

/// The router's K slices for a latent model: whole 64-input groups, at most 8 (tf_super_topk sums them in order).
pub fn routerSlices(d: usize) usize {
    return @import("cuda_triton.zig").divisor(d / 64, 8);
}

/// Floats one call's exchange carries from a rank: its routed pairs' latent rows, then its shared rows.
pub fn sendFloats(c: Config, rows: usize) usize {
    const own = c.ownShares();
    return rows * c.top_k * own * c.latent + rows * own * c.hidden;
}

/// TF_NEMOTRON_RANK of TF_NEMOTRON_WORLD into `c`: both or neither; a world must hold whole shares.
pub fn place(c: *Config) !void {
    const r = std.c.getenv("TF_NEMOTRON_RANK");
    const w = std.c.getenv("TF_NEMOTRON_WORLD");
    if (r == null and w == null) return;
    if (r == null or w == null) return error.NoRank;
    c.rank = std.fmt.parseInt(usize, std.mem.span(r.?), 10) catch return error.NoRank;
    c.world = std.fmt.parseInt(usize, std.mem.span(w.?), 10) catch return error.NoRank;
    if (c.world < 1 or c.rank >= c.world) return error.NoRank;
    if (c.world == 1) return;
    if (c.latent == 0) {
        std.log.err("tensor parallel is built for latent MoE models (Nemotron 3.5 Super); this checkpoint runs on one GPU", .{});
        return error.TensorParallelNotBuilt;
    }
    if (c.shares % c.world != 0) {
        std.log.err("{d} ranks cannot hold this model's {d} expert width shares evenly", .{ c.world, c.shares });
        return error.TensorParallelNotBuilt;
    }
}

/// Whether the checkpoint holds the shares this rank needs: a whole one holds all, a split one only its own.
pub fn checkHeld(c: Config) !void {
    const held = c.held orelse return;
    const first = c.rank * c.ownShares();
    if (c.ownShares() == 1 and held == first) return;
    std.log.err("this checkpoint is one rank's split (share {d} of {d}); rank {d} of {d} needs share(s) {d}..{d}: load the whole checkpoint", .{ held, c.shares, c.rank, c.world, first, first + c.ownShares() - 1 });
    return error.CheckpointShareMissing;
}

/// What every rank must agree on before the weights load: the model, its cut, the window, drafts and the build.
const Fingerprint = extern struct {
    layout: u32 = 1, // this struct's version
    rank: u32,
    world: u32,
    mtp: u32,
    context: u64,
    dims: [12]u64, // the config's sizes, rank and world left out
    kinds: [32]u8, // sha256 of the block pattern
    program: [32]u8, // sha256 of this executable

    fn of(c: Config, mtp: bool, context: usize, program: [32]u8) Fingerprint {
        var f: Fingerprint = .{ .rank = @intCast(c.rank), .world = @intCast(c.world), .mtp = @intFromBool(mtp), .context = context, .dims = .{ c.hidden, c.vocab, c.layers, c.experts, c.top_k, c.latent, c.expert_width, c.shared_width, c.shares, c.mamba_heads, c.heads, c.kv_heads }, .kinds = undefined, .program = program };
        var kinds: [@import("config.zig").max_layers]u8 = undefined;
        for (c.kinds[0..c.layers], kinds[0..c.layers]) |k, *b| b.* = @intFromEnum(k);
        std.crypto.hash.sha2.Sha256.hash(kinds[0..c.layers], &f.kinds, .{});
        return f;
    }
};

/// Every rank's fingerprint gathered: anything but the rank differing from rank 0's stops every rank.
pub fn agree(gpa: std.mem.Allocator, io: std.Io, c: Config, g: *const cuda.tp_group.Group, d: *const cuda.Driver, s: cuda.Stream, mtp: bool, context: usize) !void {
    var program: [32]u8 = undefined;
    const exe = try std.Io.Dir.cwd().readFileAlloc(io, "/proc/self/exe", gpa, .limited(1 << 31));
    defer gpa.free(exe);
    std.crypto.hash.sha2.Sha256.hash(exe, &program, .{});
    var all: [16]Fingerprint = undefined;
    if (c.world > all.len) return error.TooManyRanks;
    try g.gather(Fingerprint, d, s, Fingerprint.of(c, mtp, context, program), all[0..c.world]);
    for (all[0..c.world], 0..) |f, r| {
        var x = f;
        x.rank = all[0].rank;
        if (f.rank != r or !std.mem.eql(u8, std.mem.asBytes(&x), std.mem.asBytes(&all[0]))) {
            std.log.err("rank {d}'s settings differ from rank 0's (model, cut, context, drafts or build): every rank stops", .{r});
            return error.RankMismatch;
        }
    }
    std.log.info("rank {d} of {d}: every rank agrees on the model, its {d} expert shares, the window and the build", .{ c.rank, c.world, c.shares });
}

/// A latent MoE layer: router, bias, the two latent projections and both expert tables.
pub fn load(L: *W.Loader, ck: *core.Checkpoint, label: []const u8, pre: []const u8, c: Config) !W.MoE {
    var a: [128]u8 = undefined;
    var b: [192]u8 = undefined;
    const router = try modelopt.router(L, try std.fmt.bufPrint(&a, "{s}.router", .{label}), try ck.get(try std.fmt.bufPrint(&b, "{s}gate.weight", .{pre})), c);
    const bias = try ck.expect(try std.fmt.bufPrint(&b, "{s}gate.e_score_correction_bias", .{pre}), .f32, &.{c.experts});
    var m: W.MoE = .{ .router = router, .bias = try L.raw(try std.fmt.bufPrint(&a, "{s}.bias", .{label}), bias), .experts = undefined };
    const into = try modelopt.dense(L, ck, try std.fmt.bufPrint(&a, "{s}.into", .{label}), &.{try std.fmt.bufPrint(&b, "{s}fc1_latent_proj", .{pre})});
    const out = try modelopt.dense(L, ck, try std.fmt.bufPrint(&a, "{s}.out", .{label}), &.{try std.fmt.bufPrint(&b, "{s}fc2_latent_proj", .{pre})});
    if (into.outputs() != c.latent or into.inputs() != c.hidden or out.outputs() != c.hidden or out.inputs() != c.latent) return error.UnexpectedTensor;
    const routed = try table(L, ck, try std.fmt.bufPrint(&a, "{s}.routed", .{label}), pre, "experts.{d}.", c.experts, c.latent, c.expert_width, c);
    const shared = try table(L, ck, try std.fmt.bufPrint(&a, "{s}.shared", .{label}), pre, "shared_experts.", 1, c.hidden, c.shared_width, c);
    m.experts = routed;
    m.latent = .{ .into = into, .out = out, .routed = routed, .shared = shared };
    return m;
}

/// Expert `e`'s projection name: `pre` then `mid` (with its index when `mid` takes one) then the leaf.
fn expertName(buf: []u8, pre: []const u8, comptime mid: []const u8, e: usize, leaf: []const u8) ![]const u8 {
    if (comptime std.mem.indexOf(u8, mid, "{d}") != null) return std.fmt.bufPrint(buf, "{s}" ++ mid ++ "{s}", .{ pre, e, leaf });
    return std.fmt.bufPrint(buf, "{s}" ++ mid ++ "{s}", .{ pre, leaf });
}

/// `count` relu^2 experts of `width` in c.shares shares; this rank's shares stacked share-major (share * count + e).
fn table(L: *W.Loader, ck: *core.Checkpoint, label: []const u8, pre: []const u8, comptime mid: []const u8, count: usize, dims: usize, width: usize, c: Config) !kern.Experts {
    const own = c.ownShares();
    const sw = width / c.shares;
    const first = c.rank * own;
    const m = if (L.ops.k.modelopt) |*x| x else return error.ModeloptNotLoaded;
    var layer: kern.Experts = .{ .format = .nvfp4, .up = 0, .down = 0, .width = sw, .dims = dims, .experts = count * own, .relu2 = true };
    const gs = try L.gpa.alloc(f32, 2 * count * own);
    defer L.gpa.free(gs);
    var a: [128]u8 = undefined;
    var b: [192]u8 = undefined;
    for ([2]bool{ true, false }) |up| {
        const n = if (up) sw else dims;
        const k = if (up) dims else sw;
        const out = try L.alloc(try std.fmt.bufPrint(&a, "{s}.{s}", .{ label, if (up) "up" else "down" }), Nvfp4Experts.bytes(count * own, n, k, 1));
        const one = Nvfp4Experts.bytes(count, n, k, 1);
        // down keeps whole file rows (its columns are the shares); up keeps one share's rows a pass
        const passes: usize = if (up) own else 1;
        for (0..passes) |pass| {
            var ks: usize = 0;
            var base: u64 = 0;
            for (0..count) |e| {
                const p = try modelopt.part(ck, try expertName(&b, pre, mid, e, if (up) "up_proj" else "down_proj"));
                const file_w = if (up) p.n else p.k;
                const whole = file_w == width;
                if (p.kind != .nvfp4 or (if (up) p.k else p.n) != dims or !(whole or (file_w == sw and c.held != null))) return error.UnexpectedTensor;
                if (up) {
                    const share = first + pass;
                    const r0 = if (whole) share * sw else 0;
                    const rows_codes = p.w.bytes[r0 * (k / 2) ..][0 .. sw * (k / 2)];
                    const rows_blocks = p.blocks.?.bytes[r0 * (k / 16) ..][0 .. sw * (k / 16)];
                    if (e == 0) base = try L.tmp(count * n * k / 2 + count * n * k / 16);
                    try L.src.upload(base + e * n * k / 2, rows_codes);
                    try L.src.upload(base + count * n * k / 2 + e * n * k / 16, rows_blocks);
                    ks = k;
                } else {
                    ks = file_w;
                    if (e == 0) base = try L.tmp(count * n * ks / 2 + count * n * ks / 16);
                    try L.src.upload(base + e * n * ks / 2, p.w.bytes);
                    try L.src.upload(base + count * n * ks / 2 + e * n * ks / 16, p.blocks.?.bytes);
                }
                for (0..own) |s| gs[@intFromBool(!up) * count * own + s * count + e] = p.scale;
            }
            try L.src.flush();
            if (up) {
                try m.pack.fp4Experts(L.ops.s, base, base + count * n * k / 2, out + pass * one, count, n, k, 0, k, 1, 0);
            } else for (0..own) |s| {
                const k0 = if (ks == width) (first + s) * sw else 0;
                try m.pack.fp4Experts(L.ops.s, base, base + count * n * ks / 2, out + s * one, count, n, ks, k0, sw, 1, 0);
            }
        }
        if (up) layer.up = out else layer.down = out;
    }
    layer.up_scale = try modelopt.scales(L, try std.fmt.bufPrint(&a, "{s}.up_scale", .{label}), gs[0 .. count * own]);
    layer.down_scale = try modelopt.scales(L, try std.fmt.bufPrint(&a, "{s}.down_scale", .{label}), gs[count * own ..]);
    return layer;
}

fn int(x: usize) c_int {
    return @intCast(x);
}

fn go(f: cuda.Function, s: cuda.Stream, grid: [3]usize, block: u32, shared: u32, a: *cuda.Args) !void {
    const cfg: cuda.Config = .{ .grid = .{ .x = @intCast(grid[0]), .y = @intCast(grid[1]), .z = @intCast(grid[2]) }, .block = .{ .x = block }, .shared = shared };
    try cuda.launch.launch(f, cfg, s, a);
}

/// The router's slices (Nemotron's router kernels), then the wide top-k: picks as share pair ids, weights by pick.
fn route(f: *const fwd.Forward, m: W.MoE, x: u64, rows: usize) !void {
    const c = f.c;
    const b = f.b;
    const g = f.glue;
    const sk = routerSlices(c.hidden);
    const per = c.hidden / sk;
    var a: cuda.Args = .{};
    for ([_]u64{ x, m.router, b.part }) |p| a.add(p);
    for ([_]usize{ rows, c.hidden, c.experts, sk }) |v| a.add(int(v));
    if (rows <= 16 and per % 8 == 0 and (x | m.router) % 16 == 0) {
        const staged: u32 = @intCast((rows + 16) * (per / 8 + 1) * 16);
        try go(g.f.router_rows, g.s, .{ (c.experts + 15) / 16, sk, 1 }, 256, staged, &a);
    } else {
        try go(g.f.router, g.s, .{ (rows + 15) / 16, (c.experts + 15) / 16, sk }, 256, @intCast(2 * 16 * (per / 2 + 1) * 4), &a);
    }
    const fns = f.ops.k.latent.?;
    var t: cuda.Args = .{};
    for ([_]u64{ b.part, m.bias, b.pick, b.wts }) |p| t.add(p);
    t.add(int(rows));
    t.add(c.routed_scaling);
    for ([_]usize{ c.experts, sk, c.top_k, c.ownShares() }) |v| t.add(int(v));
    t.add(@as(c_int, @intFromBool(c.norm_topk)));
    try go(fns.topk, g.s, .{ rows, 1, 1 }, 32, 0, &t);
}

/// The MoE block's delta into b.delta: routed experts in the latent width, the shared expert, the ranks met.
pub fn run(f: *const fwd.Forward, m: W.MoE, rows: usize) !void {
    const c = f.c;
    const b = f.b;
    const o = f.ops;
    const lt = m.latent.?;
    const fns = o.k.latent.?;
    const own = c.ownShares();
    const K = c.top_k;
    const pairs = rows * K * own;
    const D: u64 = c.hidden;
    const Lw: u64 = c.latent;
    const g = o.grouped();
    try route(f, m, b.y, rows);
    try f.mark(.route);
    try o.linear().decode(o.s, lt.into, .{ .x = b.y }, b.lat, rows);
    try o.plan(b.pick, pairs, lt.routed.experts, 16, b.plan);
    try f.mark(.plan);
    const items = kern.maxItems(pairs, lt.routed.experts, 16);
    try g.decode(o.s, .up, .{ .x = b.lat, .stride = c.latent, .slots = K * own }, lt.routed, b.plan, b.act, items, -1);
    try f.mark(.experts_up);
    try g.decode(o.s, .down_f32, .{ .x = b.act, .stride = lt.routed.width }, lt.routed, b.plan, b.ymoe, items, -1);
    try f.mark(.experts_down);
    // the shared expert's pairs are (row, share): share i % own, whatever the row
    const sh_out = b.ymoe + @as(u64, pairs) * Lw * 4;
    try o.plan(b.sids, rows * own, own, 16, b.plan);
    const sh_items = kern.maxItems(rows * own, own, 16);
    try g.decode(o.s, .up, .{ .x = b.y, .stride = c.hidden, .slots = own }, lt.shared, b.plan, b.sact, sh_items, -1);
    try g.decode(o.s, .down_f32, .{ .x = b.sact, .stride = lt.shared.width }, lt.shared, b.plan, sh_out, sh_items, -1);
    // the ranks' rows meet in rank order, else this rank's own shares sum in share order
    const count: u64 = sendFloats(c, rows);
    var parts: u64 = b.ymoe;
    var share_stride: u64 = Lw;
    var pair_stride: u64 = own * Lw;
    var shared_at: u64 = sh_out;
    var shared_share: u64 = D;
    var shared_row: u64 = own * D;
    var n: usize = own;
    if (f.tp) |tp| {
        try tp.allGather(o.s, b.ymoe, b.recv, count * 4);
        parts = b.recv;
        share_stride = count;
        pair_stride = Lw;
        shared_at = b.recv + @as(u64, pairs) * Lw * 4;
        shared_share = count;
        shared_row = D;
        n = c.world;
    }
    var a: cuda.Args = .{};
    a.add(parts);
    a.add(@as(i64, @intCast(share_stride)));
    a.add(@as(i64, @intCast(pair_stride)));
    a.add(int(n));
    a.add(b.wts);
    a.add(int(K));
    a.add(b.comb);
    a.add(int(c.latent));
    try go(fns.combine, o.s, .{ rows, (c.latent + 255) / 256, 1 }, 256, 0, &a);
    try o.linear().decode(o.s, lt.out, .{ .x = b.comb }, b.delta, rows);
    var z: cuda.Args = .{};
    z.add(b.delta);
    z.add(shared_at);
    z.add(@as(i64, @intCast(shared_share)));
    z.add(@as(i64, @intCast(shared_row)));
    z.add(int(n));
    z.add(int(c.hidden));
    try go(fns.moe_out, o.s, .{ rows, (c.hidden + 255) / 256, 1 }, 256, 0, &z);
}

/// The shared expert's pair ids for up to `rows` rows (share i % own), written once at load.
pub fn initShareIds(fns: Fns, s: cuda.Stream, ids: u64, rows: usize, own: usize) !void {
    var a: cuda.Args = .{};
    a.add(ids);
    a.add(int(rows * own));
    a.add(int(own));
    try go(fns.share_ids, s, .{ (rows * own + 255) / 256, 1, 1 }, 256, 0, &a);
}

test "router slices and a call's exchange floats" {
    try std.testing.expectEqual(@as(usize, 8), routerSlices(4096));
    var c: Config = undefined;
    c.top_k = 22;
    c.latent = 1024;
    c.hidden = 4096;
    c.shares = 2;
    c.world = 1;
    try std.testing.expectEqual(@as(usize, 3 * 22 * 2 * 1024 + 3 * 2 * 4096), sendFloats(c, 3));
    c.world = 2;
    try std.testing.expectEqual(@as(usize, 3 * 22 * 1024 + 3 * 4096), sendFloats(c, 3));
}
