//! DFlash2 on CUDA as dflash2's fast path runs it: a context from the target's taps, a masked block, the host tree.

const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const aot = cuda.aot;
const c = @import("shape.zig");
const kern = @import("kernels.zig");
const tri = @import("triton.zig");
const st = @import("state.zig");
const wts = @import("weights.zig");
const dl = @import("draft_load.zig");
const policy = @import("draft_policy.zig");

const bf = 2;
const H = dl.hidden;
const D = dl.head_dim;
pub const block = 16; // masked rows a round drafts at most (dflash2.block)
/// Block rows a batch takes: every stream's block in one forward (launch_blocks).
pub const most_rows = block * st.max_streams;
/// Copy items a batch takes: every layer's keys and values of every stream, two pieces a head (or the head's rows).
const most_copies = @max(dl.layers * st.max_streams * 2 * dl.kv_heads * 2, 2 * most_rows);

/// A stream's drafting context: each layer's keys and values ([kv heads, len, head dim], two buffers to swap).
pub const Context = struct {
    arena: st.Arena,
    keys: [dl.layers][2]u64,
    values: [dl.layers][2]u64,
    cur: u1 = 0,
    len: usize = 0, // rows held (at most dl.window)
    end: usize = 0, // the target position after the last absorbed row

    pub fn bytes() usize {
        return dl.layers * 4 * std.mem.alignForward(usize, dl.kv_heads * dl.window * D * bf, 256);
    }

    pub fn init(d: *const cuda.Driver) !Context {
        var x: Context = .{ .arena = try st.Arena.init(d, bytes()), .keys = undefined, .values = undefined };
        for (0..dl.layers) |l| for (0..2) |j| {
            x.keys[l][j] = try x.arena.take(dl.kv_heads * dl.window * D * bf);
            x.values[l][j] = try x.arena.take(dl.kv_heads * dl.window * D * bf);
        };
        return x;
    }

    pub fn deinit(x: *Context) void {
        x.arena.deinit();
    }

    /// dflash2.skip: `n` rows go untapped, so the context starts again after them.
    pub fn skip(x: *Context, n: usize) void {
        x.len = 0;
        x.end += n;
    }
};

/// One stream's share of a batch absorb: its context and how many of the gathered tap rows are its own (in turn).
pub const Absorb = struct { ctx: *Context, rows: usize };
/// One stream's block: its context and the pending token heading it.
pub const Block = struct { ctx: *const Context, pending: u32 };

pub const DFlash2 = struct {
    gpa: std.mem.Allocator,
    w: dl.Weights,
    target: *const wts.Weights,
    ops: kern.Ops,
    t: tri.Tri,
    arena: st.Arena,
    inv_freq: u64, // (D / 2) fp32 on the device, as dflash2 makes them
    positions: []i32, // a call's row positions on the host
    // a block's buffers (block rows) and absorbing's (up to the window's rows)
    x: u64,
    normed: u64,
    conv: u64,
    dyn: u64,
    xs: u64,
    qkv: u64,
    q: u64,
    k: u64,
    v: u64,
    attn: u64,
    o: u64,
    gate: u64,
    up: u64,
    act: u64,
    act_xs: u64,
    head_h: u64,
    selected: u64,
    logits: u64,
    logits_tail: u64,
    logits32: u64,
    cand_values: u64,
    cand_ids: u64,
    topk_scratch: u64,
    pos: u64,
    cos: u64,
    sin: u64,
    table: u64,
    copies: u64,
    ids: u64,
    // host copies of a round's candidates
    host_ids: []i64,
    host_values: []f32,
    host_selected: []u16,
    cands: []u32,
    unary: []f64,
    projected: []f64,
    tree: policy.Tree = .{},
    launched: usize = 0, // the last batch's block rows a stream (0: none)

    pub fn init(gpa: std.mem.Allocator, io: std.Io, ops: kern.Ops, t: tri.Tri, target: *const wts.Weights, dir: []const u8, target_dir: []const u8) !*DFlash2 {
        const x = try gpa.create(DFlash2);
        errdefer gpa.destroy(x);
        x.* = undefined;
        x.gpa = gpa;
        x.ops = ops;
        x.t = t;
        x.target = target;
        x.tree = .{};
        x.w = try dl.load(gpa, io, ops, dir, target_dir);
        errdefer x.w.deinit();
        const R = dl.window + 1; // rows absorbed at once at most (the prompt's tail)
        const W = most_rows;
        x.arena = .{ .buf = null };
        for (0..2) |pass| {
            if (pass == 1) x.arena = try st.Arena.init(ops.k.d, x.arena.used);
            x.arena.used = 0;
            const A = &x.arena;
            x.x = try A.take(R * H * bf);
            x.normed = try A.take(R * H * bf);
            x.conv = try A.take(W * H * bf);
            x.dyn = try A.take(W * 4 * (H / dl.conv_group) * bf);
            x.xs = try A.take(R * (H * 5) / 64 * 4);
            x.qkv = try A.take(R * (dl.heads + 2 * dl.kv_heads) * D * bf);
            x.q = try A.take(W * dl.heads * D * bf);
            x.k = try A.take(R * dl.kv_heads * D * bf);
            x.v = try A.take(R * dl.kv_heads * D * bf);
            x.attn = try A.take(W * dl.heads * D * bf);
            x.o = try A.take(W * H * bf);
            x.gate = try A.take(W * 17408 * bf);
            x.up = try A.take(W * 17408 * bf);
            x.act = try A.take(W * 17408 * bf);
            x.act_xs = try A.take(W * 17408 / 64 * 4);
            x.head_h = try A.take(W * H * bf);
            x.selected = try A.take(W * dl.selector_rank * bf);
            x.logits = try A.take(W * dl.head_rows * bf);
            x.logits_tail = try A.take(W * (dl.spans[1][1] - dl.spans[1][0]) * bf);
            x.logits32 = try A.take(W * dl.head_rows * 4);
            x.cand_values = try A.take(W * policy.top * 4);
            x.cand_ids = try A.take(W * policy.top * 8);
            x.topk_scratch = try A.take(kern.torch_ops.topkScratchBytes(W, dl.head_rows));
            x.inv_freq = try A.take(D / 2 * 4);
            x.pos = try A.take(R * 4);
            x.cos = try A.take(R * D / 2 * 4);
            x.sin = try A.take(R * D / 2 * 4);
            x.table = try A.take(dl.layers * st.max_streams * 24);
            x.copies = try A.take(most_copies * @sizeOf(kern.Copy));
            x.ids = try A.take(W * 4);
        }
        errdefer x.arena.deinit();
        {
            var a: cuda.Args = .{};
            a.add(x.inv_freq);
            a.add(@as(f32, @floatCast(dl.theta)));
            a.add(@as(c_int, D / 2));
            a.add(@as(c_int, D));
            try cuda.launch.launch(ops.k.inv_freq, .{ .grid = .{ .x = 1 }, .block = .{ .x = D / 2 } }, ops.s, &a);
        }
        x.positions = try gpa.alloc(i32, dl.window + 1);
        errdefer gpa.free(x.positions);
        x.host_ids = try gpa.alloc(i64, W * policy.top);
        x.host_values = try gpa.alloc(f32, W * policy.top);
        x.host_selected = try gpa.alloc(u16, W * dl.selector_rank);
        x.cands = try gpa.alloc(u32, W * policy.top);
        x.unary = try gpa.alloc(f64, W * policy.top);
        x.projected = try gpa.alloc(f64, W * dl.selector_rank);
        return x;
    }

    pub fn deinit(x: *DFlash2) void {
        x.gpa.free(x.positions);
        x.gpa.free(x.host_ids);
        x.gpa.free(x.host_values);
        x.gpa.free(x.host_selected);
        x.gpa.free(x.cands);
        x.gpa.free(x.unary);
        x.gpa.free(x.projected);
        x.tree.tokens.deinit(x.gpa);
        x.tree.parents.deinit(x.gpa);
        x.tree.scores.deinit(x.gpa);
        x.arena.deinit();
        x.w.deinit();
        const gpa = x.gpa;
        gpa.destroy(x);
    }

    /// _lin: a packed projection of `rows` rows (each row's bits its own at any count), group sums first.
    fn lin(x: *DFlash2, in: u64, q: kern.QLinear, out: u64, rows: usize) !void {
        try x.t.groupSums(in, x.xs, q.k, rows, q.k);
        try x.ops.dense(in, x.xs, q, out, rows);
    }

    /// F.rms_norm over `rows` contiguous rows of the hidden width (the layers' and the head's norms).
    fn torchNorm(x: *DFlash2, in: u64, weight: u64, out: u64, rows: usize) !void {
        var a: cuda.Args = .{};
        a.add(in);
        a.add(weight);
        a.add(out);
        a.add(@as(u64, 0));
        a.add(@as(f32, 1e-6));
        try cuda.launch.launch(x.ops.k.torch_rms, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 128 } }, x.ops.s, &a);
    }

    fn rmsNorm(x: *DFlash2, in: u64, stride: usize, weight: u64, out: u64, rows: usize) !void {
        var a: cuda.Args = .{};
        a.add(in);
        a.add(@as(c_int, @intCast(stride)));
        a.add(weight);
        a.add(out);
        a.add(@as(c_int, H));
        a.add(@as(f32, 1e-6));
        try cuda.launch.launch(x.ops.k.rms_norm, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 256 } }, x.ops.s, &a);
    }

    /// dflash2._rotary: cos and sin of each run's positions [start, start + rows) times the inverse frequencies, fp32.
    fn rotary(x: *DFlash2, runs: []const [2]usize) !void {
        var rows: usize = 0;
        for (runs) |run| for (0..run[1]) |j| {
            x.positions[rows] = @intCast(run[0] + j);
            rows += 1;
        };
        try x.ops.upload(x.pos, std.mem.sliceAsBytes(x.positions[0..rows]));
        var a: cuda.Args = .{};
        for ([_]u64{ x.pos, x.inv_freq, x.cos, x.sin }) |v| a.add(v);
        a.add(@as(c_int, D / 2));
        try cuda.launch.launch(x.ops.k.rotary, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = D / 2 } }, x.ops.s, &a);
    }

    /// dflash2._prep: q (heads, rows, D), k and v (kv heads, rows, D) from [q | k | v] rows, normed and rotated.
    fn prep(x: *DFlash2, qkv: u64, layer: usize, rows: usize, with_q: bool, q: u64, k: u64, v: u64) !void {
        const l = x.w.layers[layer];
        const h: usize = if (with_q) dl.heads else 0;
        const stride = (h + 2 * dl.kv_heads) * D;
        const p = aot.ptr;
        try x.t.set.run(x.t.s, "_prep_kernel", .{ @intCast(rows), @intCast(h + 2 * dl.kv_heads), 1 }, &.{
            p("QKV", "*bf16", qkv),                   p("QN", "*bf16", l.q_norm), p("KN", "*bf16", l.k_norm), p("COS", "*fp32", x.cos),     p("SIN", "*fp32", x.sin),
            p("QO", "*bf16", if (with_q) q else qkv), p("KO", "*bf16", k),        p("VO", "*bf16", v),        aot.int("L", @intCast(rows)), aot.int("stride", @intCast(stride)),
            aot.float("eps", 1e-6),
        }, &.{ aot.ci("H", @intCast(h)), aot.ci("HKV", dl.kv_heads), aot.ci("HALF", D / 2) });
    }

    /// dflash2._dconv: the two-tap dynamic conv of each `seg`-row block, plus a residual.
    fn dconv(x: *DFlash2, in: u64, dyn: u64, base: u64, branch: usize, res: ?u64, out: u64, rows: usize, seg: usize) !void {
        const p = aot.ptr;
        try x.t.set.run(x.t.s, "_dconv_kernel", .{ @intCast(rows), (H + 1023) / 1024, 1 }, &.{
            p("X", "*bf16", in), p("DYN", "*bf16", dyn), p("BASE", "*bf16", base), p("RES", "*bf16", res orelse in), p("OUT", "*bf16", out), aot.int("SEG", @intCast(seg)),
        }, &.{ aot.ci("D", H), aot.ci("G", H / dl.conv_group), aot.ci("GS", dl.conv_group), aot.ci("BRANCH", @intCast(branch)), aot.ci("HAS_RES", @intFromBool(res != null)), aot.ci("BLOCK", 1024) });
    }

    /// add_taps: the target's taps (rows of 5 x hidden bf16) projected into the context's layers.
    pub fn absorb(x: *DFlash2, ctx: *Context, taps: u64, rows: usize) !void {
        return x.absorbMany(&.{.{ .ctx = ctx, .rows = rows }}, taps);
    }

    /// add_taps_streams: every stream's rows (in turn) through each projection once, each context its last window.
    pub fn absorbMany(x: *DFlash2, items: []const Absorb, taps: u64) !void {
        var total: usize = 0;
        for (items) |it| total += it.rows;
        if (total == 0) return;
        if (total > dl.window + 1 or items.len > st.max_streams) return error.TooManyDraftTaps;
        try x.lin(taps, x.w.fc, x.qkv, total); // fc's (rows, hidden) output, in a buffer of the window's rows
        try x.rmsNorm(x.qkv, H, x.w.hidden_norm, x.normed, total);
        var runs: [st.max_streams][2]usize = undefined;
        for (items, 0..) |it, j| runs[j] = .{ it.ctx.end, it.rows };
        try x.rotary(runs[0..items.len]);
        // every layer's copies: the last `keep` rows of [context | new rows], a head at a time, both halves once
        var list: std.ArrayList(kern.Copy) = .empty;
        defer list.deinit(x.gpa);
        var starts: [dl.layers + 1]usize = undefined;
        for (0..dl.layers) |l| {
            starts[l] = list.items.len;
            var first: usize = 0;
            for (items) |it| {
                const ctx = it.ctx;
                const keep = @min(dl.window, ctx.len + it.rows);
                const old_n = ctx.len;
                const drop = old_n + it.rows - keep; // rows of [old | new] that fall out of the window
                const from_old = old_n -| drop;
                const new_from = drop -| old_n;
                const nxt: u1 = ctx.cur ^ 1;
                for ([_][2]u64{ x.ctxPair(ctx.keys[l], ctx.cur, nxt), x.ctxPair(ctx.values[l], ctx.cur, nxt) }, [_]u64{ x.k, x.v }) |pair, fresh| {
                    for (0..dl.kv_heads) |h| {
                        if (from_old > 0) try list.append(x.gpa, .{ .dst = pair[1] + h * keep * D * bf, .src = pair[0] + (h * old_n + drop) * D * bf, .bytes = from_old * D * bf });
                        try list.append(x.gpa, .{ .dst = pair[1] + (h * keep + from_old) * D * bf, .src = fresh + (h * total + first + new_from) * D * bf, .bytes = (it.rows - new_from) * D * bf });
                    }
                }
                first += it.rows;
            }
        }
        starts[dl.layers] = list.items.len;
        if (list.items.len > most_copies) return error.TooManyDraftTaps;
        try x.ops.upload(x.copies, std.mem.sliceAsBytes(list.items));
        for (0..dl.layers) |l| {
            try x.lin(x.normed, x.w.layers[l].kv, x.qkv, total);
            try x.prep(x.qkv, l, total, false, 0, x.k, x.v);
            try x.ops.copies(x.copies + starts[l] * @sizeOf(kern.Copy), starts[l + 1] - starts[l]);
        }
        try x.ops.s.synchronize();
        for (items) |it| {
            it.ctx.cur ^= 1;
            it.ctx.len = @min(dl.window, it.ctx.len + it.rows);
            it.ctx.end += it.rows;
        }
    }

    fn ctxPair(_: *DFlash2, bufs: [2]u64, cur: u1, nxt: u1) [2]u64 {
        return .{ bufs[cur], bufs[nxt] };
    }

    /// launch_blocks: each stream's pending token then masks through the layers in one forward; candidates on the host.
    pub fn launchMany(x: *DFlash2, blocks: []const Block, L: usize) !void {
        const S = blocks.len;
        const rows = S * L;
        if (S == 0 or S > st.max_streams or L < 2 or L > block) return error.NoDraftContext;
        for (blocks) |b| if (b.ctx.len == 0) return error.NoDraftContext;
        var ids: [most_rows]i32 = @splat(dl.mask_id);
        var runs: [st.max_streams][2]usize = undefined;
        // each layer's table: the streams' (keys, values) pointers, then their context lengths
        const per = std.mem.alignForward(usize, S * 20, 16); // 16-byte aligned tables, as fresh torch tensors are
        var table: [dl.layers * (st.max_streams * 20 + 16)]u8 align(8) = undefined;
        for (blocks, 0..) |b, j| {
            ids[j * L] = @intCast(b.pending);
            runs[j] = .{ b.ctx.end, L };
            for (0..dl.layers) |l| {
                const at = table[l * per ..];
                std.mem.writeInt(u64, at[j * 16 ..][0..8], b.ctx.keys[l][b.ctx.cur], .little);
                std.mem.writeInt(u64, at[j * 16 + 8 ..][0..8], b.ctx.values[l][b.ctx.cur], .little);
                std.mem.writeInt(i32, at[S * 16 + j * 4 ..][0..4], @intCast(b.ctx.len), .little);
            }
        }
        try x.ops.upload(x.ids, std.mem.sliceAsBytes(ids[0..rows]));
        try x.ops.upload(x.table, table[0 .. dl.layers * per]);
        const tw = x.target;
        try x.t.embed(x.ids, tw.embed.w, tw.embed.s, tw.embed.b, x.x, rows, H);
        try x.rotary(runs[0..S]);
        const scale: f32 = @floatCast(std.math.pow(f64, D, -0.5));
        for (x.w.layers, 0..) |l, i| {
            try x.torchNorm(x.x, l.in_norm, x.normed, rows);
            try x.lin(x.normed, l.attn_proj, x.dyn, rows);
            try x.dconv(x.normed, x.dyn, l.attn_base, 0, null, x.conv, rows, L);
            try x.lin(x.conv, l.qkv, x.qkv, rows);
            try x.prep(x.qkv, i, rows, true, x.q, x.k, x.v);
            const tb = x.table + i * per;
            const p = aot.ptr;
            try x.t.set.run(x.t.s, "_block_attention", .{ @intCast(S), dl.kv_heads, 1 }, &.{
                p("Q", "*bf16", x.q),      p("KB", "*bf16", x.k),        p("VB", "*bf16", x.v),        p("TABLE", "*i64", tb), p("LENS", "*i32", tb + S * 16), p("O", "*bf16", x.attn),
                aot.float("scale", scale), aot.int("window", dl.window), aot.int("R", @intCast(rows)),
            }, &.{ aot.ci("G", dl.heads / dl.kv_heads), aot.ci("HKV", dl.kv_heads), aot.ci("L", @intCast(L)), aot.ci("LP", @intCast(@max(16, std.math.ceilPowerOfTwo(usize, L) catch unreachable))), aot.ci("D", D), aot.ci("BN", 64), aot.ci("CAUSAL", 0) });
            try x.lin(x.attn, l.o, x.o, rows);
            try x.dconv(x.o, x.dyn, l.attn_base, 1, x.x, x.x, rows, L);
            try x.torchNorm(x.x, l.post_norm, x.normed, rows);
            try x.lin(x.normed, l.mlp_proj, x.dyn, rows);
            try x.dconv(x.normed, x.dyn, l.mlp_base, 0, null, x.conv, rows, L);
            try x.t.groupSums(x.conv, x.xs, H, rows, H);
            try x.ops.dense(x.conv, x.xs, l.gate, x.gate, rows);
            try x.ops.dense(x.conv, x.xs, l.up, x.up, rows);
            try x.t.swiglu(x.gate, x.up, x.act, x.act_xs, rows, l.gate.n);
            try x.ops.dense(x.act, x.act_xs, l.down, x.o, rows);
            try x.dconv(x.o, x.dyn, l.mlp_base, 1, x.x, x.x, rows, L);
        }
        // every block but its pending row, normed into stacked depths
        const depths = L - 1;
        const total = S * depths;
        for (0..S) |j| try x.torchNorm(x.x + (j * L + 1) * H * bf, x.w.norm, x.head_h + j * depths * H * bf, depths);
        try x.lin(x.head_h, x.w.select, x.selected, total);
        // matmul_rows: the head's spans at the stacked rows' K split (1), then joined column-wise
        try x.t.groupSums(x.head_h, x.xs, H, total, H);
        const head = tw.head;
        const first: kern.QLinear = .{ .w = head.w, .s = head.s, .b = head.b, .n = dl.spans[0][1], .k = head.k, .npad = head.npad };
        try x.ops.denseSplit(x.head_h, x.xs, first, x.logits32, total, 1); // a scratch of the right size
        try x.ops.denseSplit(x.head_h, x.xs, x.w.head_tail, x.logits_tail, total, 1);
        var items: std.ArrayList(kern.Copy) = .empty;
        defer items.deinit(x.gpa);
        const na = dl.spans[0][1];
        const nb = dl.spans[1][1] - dl.spans[1][0];
        for (0..total) |r| {
            try items.append(x.gpa, .{ .dst = x.logits + r * dl.head_rows * bf, .src = x.logits32 + r * na * bf, .bytes = na * bf });
            try items.append(x.gpa, .{ .dst = x.logits + (r * dl.head_rows + na) * bf, .src = x.logits_tail + r * nb * bf, .bytes = nb * bf });
        }
        try x.ops.upload(x.copies, std.mem.sliceAsBytes(items.items));
        try x.ops.copies(x.copies, items.items.len);
        const torch = x.ops.torch();
        try torch.toF32(x.logits, x.logits32, total * dl.head_rows);
        try torch.topk(x.logits32, dl.head_rows, total, policy.top, x.cand_values, x.cand_ids, x.topk_scratch);
        try x.ops.download(std.mem.sliceAsBytes(x.host_values[0 .. total * policy.top]), x.cand_values);
        try x.ops.download(std.mem.sliceAsBytes(x.host_ids[0 .. total * policy.top]), x.cand_ids);
        try x.ops.download(std.mem.sliceAsBytes(x.host_selected[0 .. total * dl.selector_rank]), x.selected);
        try x.ops.s.synchronize();
        for (0..total * policy.top) |j| {
            const local: usize = @intCast(x.host_ids[j]);
            x.cands[j] = @intCast(if (local < na) local else dl.spans[1][0] + local - na);
            x.unary[j] = x.host_values[j];
        }
        for (0..total * dl.selector_rank) |j| x.projected[j] = @as(f32, @bitCast(@as(u32, x.host_selected[j]) << 16));
        x.launched = L;
    }

    /// finish_tree for the batch's `j`th block: best_first into `out` (parents -1: under the pending row).
    pub fn finish(x: *DFlash2, j: usize, pending: u32, context_length: usize, max_nodes: usize, sampling: ?lanes.Sampling, out: *policy.Tree) !void {
        const depths = x.launched - 1;
        const at = j * depths;
        try policy.bestFirst(x.gpa, .{ .ids = x.cands[at * policy.top ..][0 .. depths * policy.top], .unary = x.unary[at * policy.top ..][0 .. depths * policy.top], .projected = x.projected[at * dl.selector_rank ..][0 .. depths * dl.selector_rank], .depths = depths }, x.w.pred, x.w.succ, pending, @min(127, max_nodes), sampling, context_length, out);
    }

    /// finish_tree: best_first over the launched block's candidates; nodes and parents (-1: under the pending row).
    pub fn propose(x: *DFlash2, ctx: *const Context, pending: u32, context_length: usize, max_nodes: usize, sampling: ?lanes.Sampling) !struct { tokens: []const u32, parents: []const i32 } {
        if (ctx.len == 0 or max_nodes == 0) return .{ .tokens = &.{}, .parents = &.{} };
        const rows: usize = @min(block, max_nodes + 1);
        try x.launchMany(&.{.{ .ctx = ctx, .pending = pending }}, rows);
        try x.finish(0, pending, context_length, max_nodes, sampling, &x.tree);
        return .{ .tokens = x.tree.tokens.items, .parents = x.tree.parents.items };
    }
};
