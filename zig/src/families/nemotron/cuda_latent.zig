//! A latent MoE on CUDA (Nemotron 3.5 Super): expert widths in shares summed in share order, so TP=1 bits == TP=2's.

const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const Config = @import("config.zig").Config;
const kern = @import("cuda_kernels.zig");
const W = @import("cuda_weights.zig");
const modelopt = @import("cuda_modelopt.zig");
const fwd = @import("cuda_forward.zig");
const worker = @import("cuda_worker.zig");
const state = @import("cuda_state.zig");
const four = @import("cuda_latent_ck.zig");

const Nvfp4Experts = cuda.experts.Nvfp4Experts;
const Tensor = core.checkpoint.Tensor;

/// A layer's experts in the shares a rank holds: routed (latent width) and shared (hidden width); `ck` in its own math.
pub const Tables = struct { routed: kern.Experts, shared: kern.Experts, ck: ?four.Tables = null };

/// A layer's weights past the router: hidden to latent and back (bf16), its expert tables, its id in rank jobs.
pub const Latent = struct { into: kern.QLinear, out: kern.QLinear, t: Tables, id: u32 };

/// The MTP head's MoE in rank jobs (block indices name the model's own layers).
pub const mtp_id: u32 = 1 << 16;

/// The scratch an expert pass uses: the plan, routed and shared activations, the shared expert's pair ids, NVFP4 rows.
pub const Scratch = struct { plan: kern.Plan, act: u64, sact: u64, sids: u64, ck4: u64 = 0 };

/// nemotron_latent.cu's kernels.
pub const Fns = struct {
    mod: cuda.Module,
    topk: cuda.Function,
    topk_few: cuda.Function,
    combine: cuda.Function,
    moe_out: cuda.Function,
    share_ids: cuda.Function,

    pub fn load(d: *const cuda.Driver) !Fns {
        var mod = try cuda.Module.load(d, cuda.kernels.nemotron_latent);
        errdefer mod.unload();
        return .{
            .mod = mod,
            .topk = try mod.function("tf_super_topk"),
            .topk_few = try mod.function("tf_super_topk_few"),
            .combine = try mod.function("tf_super_combine"),
            .moe_out = try mod.function("tf_super_moe_out"),
            .share_ids = try mod.function("tf_super_share_ids"),
        };
    }

    pub fn unload(x: *Fns) void {
        x.mod.unload();
    }

    /// Each row's top-k of `sk` slices (ids by pick and share, weights by pick): `few` an expert a thread, else a warp.
    pub fn picks(x: Fns, s: cuda.Stream, few: bool, part: u64, bias: u64, ids: u64, wts: u64, rows: usize, scaling: f32, experts: usize, sk: usize, top_k: usize, shares: usize, norm_topk: bool) !void {
        var t: cuda.Args = .{};
        for ([_]u64{ part, bias, ids, wts }) |p| t.add(p);
        t.add(int(rows));
        t.add(scaling);
        for ([_]usize{ experts, sk, top_k, shares }) |v| t.add(int(v));
        t.add(@as(c_int, @intFromBool(norm_topk)));
        if (few) return go(x.topk_few, s, .{ rows, 1, 1 }, @intCast((experts + 31) / 32 * 32), 0, &t);
        try go(x.topk, s, .{ rows, 1, 1 }, 32, 0, &t);
    }
};

/// Picks a row the wide top-k keeps (one a lane), and experts it ranks (32 a lane); config.parse refuses past them.
pub const max_picks = @import("config.zig").latent_max_picks;
pub const max_experts = @import("config.zig").latent_max_experts;

/// The router's K slices for a latent model: whole 64-input groups, at most 8 (tf_super_topk sums them in order).
pub fn routerSlices(d: usize) usize {
    return @import("cuda_triton.zig").divisor(d / 64, 8);
}

/// Floats of a rank's expert rows for `rows` rows: its routed pairs' latent rows, then its shared rows.
pub fn partFloats(c: Config, rows: usize) usize {
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

/// What every rank must agree on before the weights load: the model, its cut, the checkpoint's layout and the build.
const Fingerprint = extern struct {
    layout: u32 = 5, // this struct's version
    rank: u32,
    world: u32,
    ck: u32, // the checkpoint's own math (--precision checkpoint) or full precision
    dims: [12]u64, // the config's sizes, rank and world left out
    kinds: [32]u8, // sha256 of the block pattern
    program: [32]u8, // sha256 of this executable
    weights: [32]u8, // the checkpoint's identity (cuda_identity.zig): config, index, shard names, sizes, headers

    fn of(c: Config, program: [32]u8, weights: [32]u8) Fingerprint {
        var f: Fingerprint = .{ .rank = @intCast(c.rank), .world = @intCast(c.world), .ck = @intFromBool(c.ck), .dims = .{ c.hidden, c.vocab, c.layers, c.experts, c.top_k, c.latent, c.expert_width, c.shared_width, c.shares, c.mamba_heads, c.heads, c.kv_heads }, .kinds = undefined, .program = program, .weights = weights };
        var kinds: [@import("config.zig").max_layers]u8 = undefined;
        for (c.kinds[0..c.layers], kinds[0..c.layers]) |k, *b| b.* = @backingInt(k);
        std.crypto.hash.sha2.Sha256.hash(kinds[0..c.layers], &f.kinds, .{});
        return f;
    }
};

/// Every rank's fingerprint gathered: anything but the rank differing from rank 0's stops every rank.
pub fn agree(gpa: std.mem.Allocator, io: std.Io, c: Config, g: *const cuda.tp_group.Group, d: *const cuda.Driver, s: cuda.Stream, dir: []const u8) !void {
    var program: [32]u8 = undefined;
    const exe = try std.Io.Dir.cwd().readFileAlloc(io, "/proc/self/exe", gpa, .limited(1 << 31));
    defer gpa.free(exe);
    std.crypto.hash.sha2.Sha256.hash(exe, &program, .{});
    const weights = try @import("cuda_identity.zig").identity(gpa, io, dir); // every byte is compared at ready
    var all: [16]Fingerprint = undefined;
    if (c.world > all.len) return error.TooManyRanks;
    try g.gather(Fingerprint, d, s, Fingerprint.of(c, program, weights), all[0..c.world]);
    for (all[0..c.world], 0..) |f, r| {
        var x = f;
        x.rank = all[0].rank;
        if (f.rank == r and std.mem.eql(u8, std.mem.asBytes(&x), std.mem.asBytes(&all[0]))) continue;
        const what = if (!std.mem.eql(u8, &f.weights, &all[0].weights)) "checkpoint (config, index or shard layout)" else if (!std.mem.eql(u8, &f.program, &all[0].program)) "build" else "model, cut or precision";
        std.log.err("rank {d}'s {s} differs from rank 0's: every rank stops", .{ r, what });
        return error.RankMismatch;
    }
    // p2p links set up now, while both ranks run eagerly: rank 0's captured graphs would wait on the setup forever
    var link = try cuda.DeviceBuffer.alloc(d, 128);
    defer link.free();
    if (c.rank == 0) {
        for (1..c.world) |peer| try g.connect(s, link.ptr, peer);
    } else try g.connect(s, link.ptr, 0);
    std.log.info("rank {d} of {d}: every rank agrees on the model, its {d} expert shares and the build", .{ c.rank, c.world, c.shares });
}

/// A layer's routed and shared expert tables in the shares this rank holds.
pub fn tables(L: *W.Loader, ck: *core.Checkpoint, label: []const u8, pre: []const u8, c: Config) !Tables {
    var a: [128]u8 = undefined;
    const r = try table(L, ck, try std.fmt.bufPrint(&a, "{s}.routed", .{label}), pre, "experts.{d}.", c.experts, c.latent, c.expert_width, c);
    const s = try table(L, ck, try std.fmt.bufPrint(&a, "{s}.shared", .{label}), pre, "shared_experts.", 1, c.hidden, c.shared_width, c);
    return .{ .routed = r.experts, .shared = s.experts, .ck = if (r.ck) |x| .{ .routed = x, .shared = s.ck.? } else null };
}

/// A latent MoE layer: router, bias, the two latent projections and both expert tables; `id` names it in rank jobs.
pub fn load(L: *W.Loader, ck: *core.Checkpoint, label: []const u8, pre: []const u8, c: Config, id: u32) !W.MoE {
    var a: [128]u8 = undefined;
    var b: [192]u8 = undefined;
    const router = try modelopt.router(L, try std.fmt.bufPrint(&a, "{s}.router", .{label}), try ck.get(try std.fmt.bufPrint(&b, "{s}gate.weight", .{pre})), c);
    const bias = try ck.expect(try std.fmt.bufPrint(&b, "{s}gate.e_score_correction_bias", .{pre}), .f32, &.{c.experts});
    var m: W.MoE = .{ .router = router, .bias = try L.raw(try std.fmt.bufPrint(&a, "{s}.bias", .{label}), bias), .experts = undefined };
    const into = try modelopt.dense(L, ck, try std.fmt.bufPrint(&a, "{s}.into", .{label}), &.{try std.fmt.bufPrint(&b, "{s}fc1_latent_proj", .{pre})});
    const out = try modelopt.dense(L, ck, try std.fmt.bufPrint(&a, "{s}.out", .{label}), &.{try std.fmt.bufPrint(&b, "{s}fc2_latent_proj", .{pre})});
    if (into.outputs() != c.latent or into.inputs() != c.hidden or out.outputs() != c.hidden or out.inputs() != c.latent) return error.UnexpectedTensor;
    const t = try tables(L, ck, label, pre, c);
    m.experts = t.routed;
    m.latent = .{ .into = into, .out = out, .t = t, .id = id };
    return m;
}

/// A bf16 norm weight [hidden] by its name under `pre`.
fn norm(L: *W.Loader, ck: *core.Checkpoint, label: []const u8, pre: []const u8, leaf: []const u8, c: Config) !u64 {
    var b: [192]u8 = undefined;
    return L.raw(label, try ck.expect(try std.fmt.bufPrint(&b, "{s}{s}", .{ pre, leaf }), .bf16, &.{c.hidden}));
}

/// The MTP head as stored (bf16 attention and input projection, a latent MoE); it drafts over the whole head.
pub fn mtp(L: *W.Loader, ck: *core.Checkpoint, c: Config) !void {
    var buf: [6][192]u8 = undefined;
    var p: [128]u8 = undefined;
    const pre = try std.fmt.bufPrint(&p, "{s}mtp.layers.", .{c.prefix()});
    const at = struct {
        fn name(b: *[192]u8, a: []const u8, z: []const u8) ![]const u8 {
            return std.fmt.bufPrint(b, "{s}{s}", .{ a, z });
        }
    }.name;
    const qkv = [3][]const u8{ try at(&buf[0], pre, "0.mixer.q_proj"), try at(&buf[1], pre, "0.mixer.k_proj"), try at(&buf[2], pre, "0.mixer.v_proj") };
    L.w.mtp = .{
        .enorm = try norm(L, ck, "mtp.enorm", pre, "0.enorm.weight", c),
        .hnorm = try norm(L, ck, "mtp.hnorm", pre, "0.hnorm.weight", c),
        .eh_proj = try modelopt.dense(L, ck, "mtp.eh_proj", &.{try at(&buf[3], pre, "0.eh_proj")}),
        .attn_norm = try norm(L, ck, "mtp.attn_norm", pre, "0.norm.weight", c),
        .attn = .{ .qkv = try modelopt.dense(L, ck, "mtp.attn.qkv", &qkv), .o = try modelopt.dense(L, ck, "mtp.attn.o", &.{try at(&buf[4], pre, "0.mixer.o_proj")}) },
        .moe_norm = try norm(L, ck, "mtp.moe_norm", pre, "1.norm.weight", c),
        .moe = try load(L, ck, "mtp.moe", try at(&buf[5], pre, "1.mixer."), c, mtp_id),
        .final_norm = try norm(L, ck, "mtp.final_norm", pre, "1.final_layernorm.weight", c),
    };
    // the draft head is the target's: every id a candidate, so no vocabulary table is needed for this tokenizer
    L.w.draft_head = L.w.head;
    const ids = try L.gpa.alloc(u32, c.vocab);
    defer L.gpa.free(ids);
    for (ids, 0..) |*x, i| x.* = @intCast(i);
    try W.draftIds(L, ids);
}

/// Expert `e`'s projection name: `pre` then `mid` (with its index when `mid` takes one) then the leaf.
fn expertName(buf: []u8, pre: []const u8, comptime mid: []const u8, e: usize, leaf: []const u8) ![]const u8 {
    if (comptime std.mem.indexOf(u8, mid, "{d}") != null) return std.fmt.bufPrint(buf, "{s}" ++ mid ++ "{s}", .{ pre, e, leaf });
    return std.fmt.bufPrint(buf, "{s}" ++ mid ++ "{s}", .{ pre, leaf });
}

/// A table as the math in force reads it: the grouped kernels' NVFP4 blocks, or the FP4 mma's tiles (checkpoint math).
const Table = struct { experts: kern.Experts, ck: ?cuda.ck.Layer = null };

/// `count` relu^2 experts of `width` in c.shares shares; this rank's shares stacked share-major (share * count + e).
fn table(L: *W.Loader, ck: *core.Checkpoint, label: []const u8, pre: []const u8, comptime mid: []const u8, count: usize, dims: usize, width: usize, c: Config) !Table {
    const own = c.ownShares();
    const sw = width / c.shares;
    const first = c.rank * own;
    const m = if (L.ops.k.modelopt) |*x| x else return error.ModeloptNotLoaded;
    const c4: ?*const cuda.ck.Ck = if (L.ops.k.ck) |*x| x else null;
    var layer: kern.Experts = .{ .format = .nvfp4, .up = 0, .down = 0, .width = sw, .dims = dims, .experts = count * own, .relu2 = true };
    var tiles: cuda.ck.Layer = .{ .up = 0, .up_s = 0, .down = 0, .down_s = 0, .up_alpha = 0, .down_alpha = 0, .up_act = 0, .down_act = 0, .experts = count * own, .width = sw, .dims = dims };
    const gs = try L.gpa.alloc(f32, 2 * count * own); // weight scales, or the checkpoint math's alphas
    defer L.gpa.free(gs);
    var a: [128]u8 = undefined;
    var b: [192]u8 = undefined;
    for ([2]bool{ true, false }) |up| {
        const n = if (up) sw else dims;
        const k = if (up) dims else sw;
        const pk = cuda.ck.Ck.packedBytes(count, n, k); // a share's words and scale tiles (checkpoint math)
        const dir: []const u8 = if (up) "up" else "down";
        const out = try L.alloc(try std.fmt.bufPrint(&a, "{s}.{s}", .{ label, dir }), if (c4 != null) pk[0] * own else Nvfp4Experts.bytes(count * own, n, k, 1));
        const out_s = if (c4 != null) try L.alloc(try std.fmt.bufPrint(&a, "{s}.{s}_s", .{ label, dir }), pk[1] * own) else 0;
        const one = Nvfp4Experts.bytes(count, n, k, 1);
        var act: f32 = 0; // the direction's input scale: every expert's must be the same
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
                if (e == 0) act = p.act;
                if (c4 != null and (p.act == 0 or p.act != act)) return error.MixedQuantization;
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
                for (0..own) |s| gs[@intFromBool(!up) * count * own + s * count + e] = if (c4 != null) cuda.ck.alpha(p.act, p.scale) else p.scale;
            }
            try L.src.flush();
            if (up) {
                if (c4) |q| try q.pack(L.ops.s, base, base + count * n * k / 2, count, n, k, 0, k, out + pass * pk[0], out_s + pass * pk[1]) else try m.pack.fp4Experts(L.ops.s, base, base + count * n * k / 2, out + pass * one, count, n, k, 0, k, 1, 0);
            } else for (0..own) |s| {
                const k0 = if (ks == width) (first + s) * sw else 0;
                if (c4) |q| try q.pack(L.ops.s, base, base + count * n * ks / 2, count, n, ks, k0, sw, out + s * pk[0], out_s + s * pk[1]) else try m.pack.fp4Experts(L.ops.s, base, base + count * n * ks / 2, out + s * one, count, n, ks, k0, sw, 1, 0);
            }
        }
        if (up) {
            layer.up = out;
            tiles.up = out;
            tiles.up_s = out_s;
            tiles.up_act = act;
        } else {
            layer.down = out;
            tiles.down = out;
            tiles.down_s = out_s;
            tiles.down_act = act;
        }
    }
    const g_up = try modelopt.scales(L, try std.fmt.bufPrint(&a, "{s}.up_scale", .{label}), gs[0 .. count * own]);
    const g_down = try modelopt.scales(L, try std.fmt.bufPrint(&a, "{s}.down_scale", .{label}), gs[count * own ..]);
    if (c4 != null) {
        tiles.up_alpha = g_up;
        tiles.down_alpha = g_down;
        return .{ .experts = layer, .ck = tiles };
    }
    layer.up_scale = g_up;
    layer.down_scale = g_down;
    return .{ .experts = layer };
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
    try f.ops.k.latent.?.picks(g.s, rows <= state.max_rows, b.part, m.bias, b.pick, b.wts, rows, c.routed_scaling, c.experts, sk, c.top_k, c.ownShares(), c.norm_topk);
}

/// One share set's expert rows into `out` (fp32): routed pairs by `picks`, then the shared expert's (row, share).
pub fn partials(o: kern.Ops, t: Tables, s: Scratch, picks: u64, lat: u64, hidden: u64, rows: usize, own: usize, c: Config, out: u64) !void {
    if (t.ck) |x| return four.partials(o, x, s.plan, s.act, s.sact, s.sids, s.ck4, picks, lat, hidden, rows, own, c, out);
    const pairs = rows * c.top_k * own;
    const g = o.grouped();
    try o.plan(picks, pairs, t.routed.experts, 16, s.plan);
    const items = kern.maxItems(pairs, t.routed.experts, 16);
    try g.decode(o.s, .up, .{ .x = lat, .stride = c.latent, .slots = c.top_k * own }, t.routed, s.plan, s.act, items, -1);
    try g.decode(o.s, .down_f32, .{ .x = s.act, .stride = t.routed.width }, t.routed, s.plan, out, items, -1);
    try o.plan(s.sids, rows * own, own, 16, s.plan);
    const sh_items = kern.maxItems(rows * own, own, 16);
    try g.decode(o.s, .up, .{ .x = hidden, .stride = c.hidden, .slots = own }, t.shared, s.plan, s.sact, sh_items, -1);
    try g.decode(o.s, .down_f32, .{ .x = s.sact, .stride = t.shared.width }, t.shared, s.plan, out + @as(u64, pairs) * c.latent * 4, sh_items, -1);
}

/// The MoE block's delta into b.delta: the other ranks' shares asked for, ours run, all summed in share order.
pub fn run(f: *const fwd.Forward, m: W.MoE, rows: usize) !void {
    const c = f.c;
    const b = f.b;
    const o = f.ops;
    const lt = m.latent.?;
    const fns = o.k.latent.?;
    const own = c.ownShares();
    const pairs = rows * c.top_k * own;
    const D: u64 = c.hidden;
    const Lw: u64 = c.latent;
    try route(f, m, b.y, rows);
    try f.mark(.route);
    try o.linear().decode(o.s, lt.into, .{ .x = b.y }, b.lat, rows);
    if (f.tp != null) try worker.send(f, lt.id, rows); // the other ranks' shares start while ours run
    try partials(o, lt.t, .{ .plan = b.plan, .act = b.act, .sact = b.sact, .sids = b.sids, .ck4 = b.ck4 }, b.pick, b.lat, b.y, rows, own, c, b.ymoe);
    try f.mark(.experts_down);
    // one GPU: shares interleaved by pair; ranks: each rank's rows in its own region of ymoe, in rank order
    var share_stride: u64 = Lw;
    var pair_stride: u64 = own * Lw;
    var shared_share: u64 = D;
    var shared_row: u64 = own * D;
    var n: usize = own;
    if (f.tp) |tp| {
        try worker.receive(f, rows);
        share_stride = partFloats(c, state.pass_rows);
        pair_stride = Lw;
        shared_share = share_stride;
        shared_row = D;
        n = tp.world;
    }
    var a: cuda.Args = .{};
    a.add(b.ymoe);
    a.add(@as(i64, @intCast(share_stride)));
    a.add(@as(i64, @intCast(pair_stride)));
    a.add(int(n));
    a.add(b.wts);
    a.add(int(c.top_k));
    a.add(b.comb);
    a.add(int(c.latent));
    try go(fns.combine, o.s, .{ rows, (c.latent + 255) / 256, 1 }, 256, 0, &a);
    try o.linear().decode(o.s, lt.out, .{ .x = b.comb }, b.delta, rows);
    var z: cuda.Args = .{};
    z.add(b.delta);
    z.add(b.ymoe + @as(u64, pairs) * Lw * 4);
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

test "router slices and a rank's expert rows" {
    try std.testing.expectEqual(@as(usize, 8), routerSlices(4096));
    var c: Config = undefined;
    c.top_k = 22;
    c.latent = 1024;
    c.hidden = 4096;
    c.shares = 2;
    c.world = 1;
    try std.testing.expectEqual(@as(usize, 3 * 22 * 2 * 1024 + 3 * 2 * 4096), partFloats(c, 3));
    c.world = 2;
    try std.testing.expectEqual(@as(usize, 3 * 22 * 1024 + 3 * 4096), partFloats(c, 3));
}
