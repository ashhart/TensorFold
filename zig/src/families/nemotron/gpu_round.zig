//! A lone greedy stream's rounds with no host wait: the GPU accepts, keeps rows and drafts; the host reads behind it.
const std = @import("std");
const mtl = @import("metal");
const lanes = @import("lanes");
const st = @import("state.zig");
const fwd = @import("forward.zig");
const layers = @import("layers.zig");
const backend = @import("backend.zig");
const copy_lanes = @import("copy_lanes.zig");
const Depth = @import("round_depth.zig").Depth;

const Buffer = mtl.Buffer;
const Metal = backend.Metal;
const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;

// the args buffer's blocks, in i32 (nemotron_round.metal's A_*)
const a_kv = 0;
const a_dims = 64;
const a_abs = 128;
const a_rows = 192;
const a_md = 256;
const a_segi = 320;
const a_va = 384;
const a_vb = 640;
const a_vc = 896;
const a_step = 1024;
const a_stride = 128;
const a_tp = 2048; // a tree window's shared attention pass dims
const a_tt = 2112; // its tail and merge dims
const a_cmp = 2176; // the kept path's compaction dims
const a_par = 2240; // the window's rows' parents
const a_path = 2304; // the kept path's rows
const a_fol = 2368; // the token after each kept row
const a_end = 2432;
const ring = 64; // rounds in the records ring
const rec = 128; // u32 a record (nemotron_round.metal)
const per_level = 64; // a head level's dispatches at most (each a 3-u32 threadgroup count)
const tmpls = 4; // rounds whose level templates the host may be writing or the GPU reading

pub const Options = struct {
    depth: usize = 4, // the head's levels a round at most (a verify takes depth + 1 rows at most)
    fixed: bool = false, // every round drafts `depth` levels (else the stream's records move it from 2 up)
    ahead: usize = 2, // rounds queued before the host reads the oldest
    power: f32 = 2, // the running top-1 product's power the stop weighs
    stop: bool = true, // false: every level verified
    split: bool = false, // the head's part in its own command buffer (GPU time by part)
    bar: ?f32 = null, // every level's bar fixed (probes: 2 stops at level 0), else from the stream's rate
    profile: usize = 0, // rounds after the reply with every dispatch alone: GPU ms by kernel
    head_bar: bool = true, // a level's bar also prices the head level passing it drafts
    trees: bool = true, // windows verify as trees (GPU-written tables): branches beside the head's drafts or a copy
    siblings: bool = false, // leaves: the head's likeliest second choice past a row's price, its guess beside a copy
    sib_bar: ?f64 = null, // a fixed bar for second choices (0: every round has one), else a row's price
    guess_match: u32 = 24, // a copy whose match is shorter also takes the head's guess beside its first row
    level_bench: bool = false, // after the reply: a live head level's GPU time (chains of 1 and 8 live levels, medians)
    copy: bool = true, // copy lanes: a window copied after the context's matched suffix when its rows are priced
    copy_max: usize = 30, // copy rows a window takes at most
    min_match: u32 = 8, // context tokens a copy's match needs (shorter ones are coincidental phrases)
    tail: bool = true, // the head's drafts continued by a copy when the context with them matches
};

pub const Result = struct {
    rounds: u64 = 0,
    accepted: u64 = 0,
    by_rows: [st.max_rows + 1]u32 = @splat(0), // rounds by the rows their verify took
    head_kept: [17][17]u32 = @splat(@splat(0)), // head windows by [verify rows][tokens kept] (16+ in the last)
    copied: u64 = 0, // rounds whose window was a copy
    siblings: u64 = 0, // rounds with the head's second choice beside its first draft, and those that kept it
    siblings_kept: u64 = 0,
    guesses: u64 = 0, // copies with the head's guess beside their first row, and those that kept it
    guesses_kept: u64 = 0,
    prefill_s: f64 = 0,
    decode_s: f64 = 0,
    paused: bool = false, // the stream went to the lane core, its cache as the last round read left it
};

/// A server's engine thread between rounds: told when tokens land, asked whether to hand the stream to the lane core.
pub const Hooks = struct {
    ctx: ?*anyopaque = null,
    committed: ?*const fn (ctx: *anyopaque) void = null,
    yield: ?*const fn (ctx: *anyopaque) bool = null,

    fn note(k: Hooks) void {
        if (k.committed) |f| f(k.ctx.?);
    }

    fn asked(k: Hooks) bool {
        return if (k.yield) |f| f(k.ctx.?) else false;
    }
};

const Round = struct {
    b: *Metal,
    c: *st.Cache,
    o: Options,
    state: Buffer, // u32 [8] (nemotron_round.metal)
    args: Buffer,
    wins: [2]Buffer, // u32 [rows]: the pending token, then the drafts
    draws: Buffer,
    recs: Buffer,
    h0: Buffer, // the last kept row's hidden state, the head's first input
    tmpl: Buffer, // u32 [tmpls, per_level, 3]: a round's level-1 threadgroup counts (every later level dispatches the same)
    live: Buffer, // u32 [levels - 1, per_level, 3]: levels 1.. read theirs here (zeros: past the stop)
    tmpl0: Buffer, // u32 [tmpls, per_level, 3]: a round's level-0 threadgroup counts (the root absorbed)
    live0: Buffer, // u32 [per_level, 3]: level 0 reads its counts here (zeros: a copied window)
    context: Buffer, // u32 [capacity]: the prompt and every token since, the pending one last
    second: Buffer, // u32 [2]: the head's second choice at the first draft, and its chance
    tmpl_att: Buffer, // u32 [tmpls, per_level, 3]: a verify's attention counts (both paths, every layer)
    live_att: Buffer, // u32 [per_level, 3]: the path the window takes keeps its counts, the other zeros
    sib_scale: f64 = 1, // second choices' landed over offered chances (the stream's, with a prior)
    widest: usize, // a window's rows at most (the head's or a copy's)
    slots: [3]u32, // the prompt's state slot, then the two the rounds store to in turn
    parity: u1,

    fn deinit(r: *Round) void {
        for ([_]Buffer{ r.state, r.args, r.wins[0], r.wins[1], r.draws, r.recs, r.h0, r.tmpl, r.live, r.tmpl0, r.live0, r.context, r.second, r.tmpl_att, r.live_att }) |x| x.deinit();
    }

    fn u32s(x: Buffer, n: usize) []u32 {
        return x.slice(u32, n);
    }

    /// The cache and stream as a lane-core round leaves them, after `read` rounds (the last kept `kept` rows).
    fn handOver(r: *Round, s: *lanes.Stream, read: u64, kept: u32, len: usize, mtp: usize) void {
        const c = r.c;
        c.len = len;
        c.mtp_len = mtp;
        c.levels = 0;
        if (read > 0) {
            // the next verify replays the last round's kept path from the state its verify stored
            const k = read - 1;
            c.slot = if (k == 0) r.slots[0] else r.slots[1 + (k + 1) % 2];
            c.replay = kept;
            c.parity = r.parity ^ @as(u1, @intCast((k + 1) % 2));
            for (r.args.slice(i32, a_end)[a_path..][0..kept], 0..) |row, i| c.replay_map[i] = @intCast(row);
        }
        s.cache_len = len;
        s.pending = s.context.items[s.context.items.len - 1];
    }

    /// The bars each level's running top-1 product clears: tokens a ms times its verify row and the next head level.
    fn bars(r: *const Round, rate: f64, levels: usize) [st.max_levels]f32 {
        var out: [st.max_levels]f32 = @splat(0);
        if (!r.o.stop) return out;
        if (r.o.bar) |x| return @splat(x);
        const w = r.b.costs.window;
        for (0..levels) |j| {
            const hi = @min(j + 2, r.b.costs.windows);
            const lo = @min(j + 1, r.b.costs.windows);
            const added = if (hi > lo) w[hi - 1].ms - w[lo - 1].ms else w[hi - 1].ms - w[hi - 2].ms;
            const next_level = if (r.o.head_bar and j + 1 < levels) r.b.costs.head_ms else 0;
            out[j] = @floatCast(rate * (added + next_level));
        }
        return out;
    }

    /// The head's chain, a level at a time while levels clear their bars (level 1 records the later levels' counts).
    fn chain(r: *Round, e: *fwd.Enc, win: Buffer, slot: usize, levels: usize, rate: f64, chunks: usize, absorbed: bool, gated: bool) void {
        const b = r.b;
        const head = &b.head.?;
        const cfg = b.m.config;
        // an absorbed root's level 0 reads its counts from live0 (zeros when the window is a copy)
        var gate0 = fwd.Enc.Gate{ .live = r.live0, .base = 0, .tmpl = r.tmpl0.slice(u32, tmpls * per_level * 3)[slot * per_level * 3 ..][0 .. per_level * 3] };
        if (absorbed) {
            // the root's row was absorbed with the window's: its residual and q/k/v to row 0
            if (gated) e.gate = &gate0;
            e.pipe(b.m.kernels.get("tf_round_root"));
            e.buf(head.scratch.h, 0, 0);
            e.buf(head.scratch.qkv, 0, 1);
            e.buf(r.state, 0, 2);
            e.bytes([2]u32{ @intCast(cfg.hidden), @intCast(cfg.qkvDim()) }, 3);
            e.run(.{ @max(cfg.hidden, cfg.qkvDim()), 1, 1 }, .{ 256, 1, 1 });
            head.absorbed = true;
        } else {
            e.pipe(b.m.kernels.get("tf_round_gather"));
            e.buf(b.scratch.x, 0, 0);
            e.buf(r.h0, 0, 1);
            e.buf(r.state, 0, 2);
            e.bytes(@as(u32, @intCast(cfg.hidden)), 3);
            e.run(.{ cfg.hidden, 1, 1 }, .{ 128, 1, 1 });
        }
        head.keep_hid = false;
        head.pick = false;
        defer {
            head.gpu = null;
            head.keep_hid = true;
            head.pick = true;
        }
        const bars_at = r.bars(rate, levels);
        const tmpl = r.tmpl.slice(u32, tmpls * per_level * 3)[slot * per_level * 3 ..][0 .. per_level * 3];
        for (0..levels) |j| {
            const at = (a_step + a_stride * j) * 4;
            head.gpu = .{ .buffer = r.args, .kv = at, .attn = at + 64 * 4, .chunks = chunks };
            // level 0 always runs; level j > 0 reads its threadgroup counts from live (zeros past the stop)
            var gate = fwd.Enc.Gate{ .live = r.live, .base = (j -| 1) * per_level, .tmpl = if (j == 1) tmpl else null };
            e.gate = if (j > 0) &gate else if (absorbed and gated) &gate0 else null;
            head.stepAt(e, r.c, if (j == 0) r.h0 else head.scratch.x, 0, win, 4 * j, .greedy, win, 4 * (j + 1), j);
            e.gate = null;
            std.debug.assert(gate.at <= per_level and gate0.at <= per_level);
            // the draft, its chance (tf_topk_probs's first id and chance) and the stop in one kernel
            e.pipe(b.m.kernels.get("tf_round_top1"));
            e.buf(head.scratch.logits, 0, 0);
            e.buf(head.w.ids, 0, 1);
            e.buf(r.c.topk, j * 16, 2);
            e.buf(r.c.topk, st.max_levels * 16 + j * 16, 3);
            e.bytes(@as(u32, @intCast(head.w.vocab)), 4);
            e.buf(win, 4 * (j + 1), 5);
            e.e.setBytes(std.mem.sliceAsBytes(bars_at[0..@max(levels, 4)]), 6);
            e.buf(r.state, 0, 7);
            e.buf(r.tmpl, slot * per_level * 3 * 4, 8);
            e.buf(r.live, 0, 9);
            e.bytes([4]u32{ @intCast(j), if (j + 1 < levels) per_level else 0, @intCast(j * per_level), @bitCast(r.o.power) }, 10);
            e.buf(r.second, 0, 11);
            e.bytes(@as(u32, @intFromBool(r.o.trees and r.o.siblings)), 12);
            e.run(.{ 1024, 1, 1 }, .{ 1024, 1, 1 });
        }
    }

    /// Round n: the verify's arguments, the verify of window n % 2 (`rows` rows), the accept, the head's `levels`.
    fn round(r: *Round, e: *fwd.Enc, n: u64, rows: usize, levels: usize, copy: u32, tail: bool, rate: f64, chunks: usize, head_chunks: usize) void {
        const b = r.b;
        const c = b.m.config;
        const win = r.wins[n % 2];
        const next = r.wins[(n + 1) % 2];
        const virtual = rows + r.widest; // replayed rows (the last window's kept, at most the widest window), then these
        // verify n loads the state its predecessor stored (the prompt's before it), and stores to the other slot
        const load = if (n < 2) r.slots[0] else r.slots[1 + (n - 2) % 2];
        const store = r.slots[1 + (n + 1) % 2];
        const parity: u32 = @as(u32, r.parity) ^ @as(u32, @intCast(n % 2));
        e.pipe(b.m.kernels.get("tf_round_args"));
        e.buf(r.state, 0, 0);
        e.buf(r.args, 0, 1);
        e.bytes([12]u32{ @intCast(rows), @intCast(c.qkvDim()), @intCast(c.heads * c.head_dim), @intCast(r.c.kv[0].capacity), @intCast(c.heads / c.kv_heads), load, store, parity, @intCast(r.c.rid), st.replay_rows, @intCast(virtual), 0 }, 2);
        e.buf(b.scratch.tdepths, 0, 3);
        e.buf(b.scratch.tpaths, 0, 4);
        e.run(.{ 1, 1, 1 }, .{ 1, 1, 1 });
        const att_slot = n % tmpls;
        if (r.o.trees and r.o.siblings) {
            e.pipe(b.m.kernels.get("tf_round_attsel"));
            e.buf(r.state, 0, 0);
            e.buf(r.args, 0, 1);
            e.buf(r.tmpl_att, att_slot * per_level * 12, 2);
            e.buf(r.live_att, 0, 3);
            e.bytes(@as(u32, @intCast(r.c.attentions)), 4);
            e.run(.{ 1, 1, 1 }, .{ 1, 1, 1 });
        }

        var f = b.forward();
        f.round = .{ .buffer = r.args, .va = a_va * 4, .vb = a_vb * 4, .vc = a_vc * 4, .segi = a_segi * 4, .dims = a_md * 4, .rows = a_rows * 4, .virtual = virtual };
        const dual = @import("tree.zig").Dual{ .live = r.live_att, .tmpl = r.tmpl_att.slice(u32, tmpls * per_level * 3)[att_slot * per_level * 3 ..][0 .. per_level * 3] };
        const gpu_tree: ?@import("tree.zig").GpuTree = if (r.o.trees and r.o.siblings) .{ .args = r.args, .tp = a_tp * 4, .tt = a_tt * 4, .nca = chunks, .ncb = 2, .dual = dual } else null;
        const segs = [_]fwd.Seg{.{ .rows = rows, .cache = r.c, .gpu = .{ .buffer = r.args, .kv = a_kv * 4, .attn = a_dims * 4, .chunks = chunks }, .gpu_tree = gpu_tree }};
        f.body(e, &segs, win, 0);
        f.draw(e, &segs, .all, r.draws, 0);

        e.pipe(b.m.kernels.get("tf_round_accept"));
        e.buf(win, 0, 0);
        e.buf(r.draws, 0, 1);
        e.buf(r.state, 0, 2);
        e.buf(next, 0, 3);
        e.buf(r.recs, (n % ring) * rec * 4, 4);
        e.buf(r.args, 0, 5);
        e.bytes([12]u32{ @intCast(levels), @intCast(c.qkvDim()), @intCast(c.heads * c.head_dim), @intCast(r.c.mtp.?.capacity), @intCast(c.heads / c.kv_heads), @intCast(rows), 0, @intCast(r.c.kv[0].capacity), @intCast(c.head_dim), 0, 0, 0 }, 6);
        e.buf(r.context, 0, 7);
        e.run(.{ 1, 1, 1 }, .{ 1, 1, 1 });

        if (!r.o.split) r.draft(e, n, rows, levels, copy, tail, rate, head_chunks);
    }

    /// The window's rows into the head's cache (rows past the kept ones are overwritten later), a copy or the chain.
    fn draft(r: *Round, e: *fwd.Enc, n: u64, rows: usize, levels: usize, copy: u32, tail: bool, rate: f64, head_chunks: usize) void {
        const win = r.wins[n % 2];
        const next = r.wins[(n + 1) % 2];
        const slot = (n + 1) % tmpls;
        const b = r.b;
        const cfg = b.m.config;
        _ = win;
        // the kept path's keys and values into place, its hidden rows gathered, absorbed with the tokens after them
        for (0..r.c.attentions) |a| {
            e.pipe(b.m.kernels.get("tf_kv_compact"));
            e.buf(r.c.kv[a].k, 0, 0);
            e.buf(r.c.kv[a].v, 0, 1);
            e.buf(r.args, a_path * 4, 2);
            e.buf(r.args, a_cmp * 4, 3);
            if (a > 0) e.alongside();
            e.run(.{ cfg.head_dim, cfg.kv_heads, 1 }, .{ cfg.head_dim, 1, 1 });
        }
        e.pipe(b.m.kernels.get("tf_gather_rows"));
        e.buf(b.scratch.x, 0, 0);
        e.buf(b.scratch.kept, 0, 1);
        e.buf(r.args, a_path * 4, 2);
        e.alongside();
        e.run(.{ cfg.hidden, rows, 1 }, .{ 256, 1, 1 });
        b.head.?.absorbAt(e, r.c, rows, b.scratch.kept, 0, r.args, a_fol * 4, r.args, a_abs * 4);
        e.pipe(r.b.m.kernels.get("tf_round_copy"));
        e.buf(r.context, 0, 0);
        e.buf(r.state, 0, 1);
        e.buf(next, 0, 2);
        e.buf(r.tmpl0, slot * per_level * 12, 3);
        e.buf(r.live0, 0, 4);
        e.bytes([4]u32{ copy, r.o.min_match, per_level, if (r.o.trees and r.o.siblings) r.o.guess_match else 0 }, 5);
        e.run(.{ 1024, 1, 1 }, .{ 1024, 1, 1 });
        r.chain(e, next, slot, levels, rate, head_chunks, true, copy > 0);
        if (tail and copy > 0) {
            e.pipe(r.b.m.kernels.get("tf_round_tail"));
            e.buf(r.context, 0, 0);
            e.buf(r.state, 0, 1);
            e.buf(next, 0, 2);
            e.bytes([4]u32{ copy + 1, r.o.min_match, 0, 0 }, 3);
            e.run(.{ 1024, 1, 1 }, .{ 1024, 1, 1 });
        }
        if (r.o.trees and r.o.siblings) {
            // the head's second choice at the first draft, priced as one more row
            const w = b.costs.window;
            const row_ms = (w[@min(levels + 1, b.costs.windows - 1)].ms - w[0].ms) / @as(f64, @floatFromInt(@max(levels, 1)));
            e.pipe(b.m.kernels.get("tf_round_sib"));
            e.buf(r.state, 0, 0);
            e.buf(next, 0, 1);
            e.buf(r.args, 0, 2);
            e.buf(r.second, 0, 3);
            const bar: f64 = r.o.sib_bar orelse rate * row_ms;
            e.bytes([4]f32{ @floatCast(bar), @floatCast(r.sib_scale), @floatFromInt(@max(levels + 2, copy + 2)), 0 }, 4);
            e.run(.{ 1, 1, 1 }, .{ 1, 1, 1 });
        }
    }
};

/// Prefill `s`, then its rounds on the GPU until it finishes: its tokens in `s`, the rounds' counts in the result.
pub fn run(gpa: std.mem.Allocator, b: *Metal, s: *lanes.Stream, o: Options, hooks: Hooks) !Result {
    if (b.head == null or s.sampling != null) return error.GreedyDraftsOnly;
    if (o.depth < 1 or o.depth + 1 > backend.max_window or o.depth > st.max_levels) return error.WindowTooWide;
    const be = b.backend();
    const t0 = mtl.clock.seconds();
    const first = try be.opening(gpa, s); // a prompt pass that stops (a cancel) or fails releases the stream
    var handed = false;
    defer if (!handed) be.release(s);
    var out = Result{};
    hooks.note();
    const t1 = mtl.clock.seconds();
    out.prefill_s = t1 - t0;
    if (s.finished) return out;

    const c = try b.cacheOf(s);
    if (o.copy_max > copy_lanes.max_rows) return error.WindowTooWide;
    const leaf: usize = @intFromBool(o.trees and o.siblings); // a leaf beside the head's drafts or a copy
    const rows = @max(o.depth + 1, if (o.copy) o.copy_max + 1 else 0) + leaf; // a verify's rows at most
    const dev = b.m.device;
    var r = Round{
        .b = b,
        .c = c,
        .o = o,
        .state = try dev.buffer(128, opts),
        .args = try dev.buffer(a_end * 4, opts),
        .wins = .{ try dev.buffer(st.max_rows * 4, opts), try dev.buffer(st.max_rows * 4, opts) },
        .draws = try dev.buffer(st.max_rows * 4, opts),
        .recs = try dev.buffer(ring * rec * 4, opts),
        .h0 = try dev.buffer(b.m.config.hidden * 2, opts),
        .tmpl = try dev.buffer(tmpls * per_level * 12, opts),
        .live = try dev.buffer(@max(o.depth - 1, 1) * per_level * 12, opts),
        .tmpl0 = try dev.buffer(tmpls * per_level * 12, opts),
        .live0 = try dev.buffer(per_level * 12, opts),
        .context = try dev.buffer((b.o.capacity + st.max_rows) * 4, opts),
        .second = try dev.buffer(128, opts),
        .tmpl_att = try dev.buffer(tmpls * per_level * 12, opts),
        .live_att = try dev.buffer(per_level * 12, opts),
        .widest = rows,
        .slots = .{ c.slot, try b.pool.take(), try b.pool.take() },
        .parity = c.parity,
    };
    defer r.deinit();
    defer {
        b.drain() catch {};
        for (r.slots) |x| if (x != c.slot) b.pool.give(x);
    }
    std.debug.assert(c.replay == 0 and c.rid >= 0);
    // as if a round had kept the prompt's last row: the chain's root stays in the head's cache
    const mtp0 = c.mtp_len;
    const st0 = Round.u32s(r.state, 32);
    @memset(st0, 0);
    @memcpy(st0[0..12], &[12]u32{ @intCast(c.len), @intCast(mtp0 + 1), 0, first, 0, @intCast(c.start + c.rows - 1), 0, 1, @bitCast(@as(f32, 1)), 1, 0, 0 });
    const ctx = Round.u32s(r.context, s.prompt_len + 1);
    @memcpy(ctx[0..s.prompt_len], s.prompt());
    ctx[s.prompt_len] = first;
    Round.u32s(r.wins[0], 1)[0] = first;
    const cfg = b.m.config;
    const args = r.args.slice(i32, a_end);
    for (0..64) |i| args[a_par + i] = @as(i32, @intCast(i)) - 1; // the first window: a chain
    for (0..o.depth) |j| {
        const at: i32 = @intCast(mtp0 + j);
        args[a_step + a_stride * j ..][0..4].* = .{ @intCast(cfg.qkvDim()), @intCast(cfg.heads * cfg.head_dim), @intCast(c.mtp.?.capacity), at };
        args[a_step + a_stride * j + 64 ..][0..5].* = .{ at + 1, @divTrunc(at + 1 + 511, 512), 1, 1, @intCast(cfg.heads / cfg.kv_heads / 16) };
    }

    var rate: f64 = 2.5 / (b.costs.window[@min(4, b.costs.windows) - 1].ms + b.costs.head_ms * @as(f64, @floatFromInt(o.depth)));
    var len_known: usize = c.len;
    var mtp_known: usize = mtp0 + 1;
    var queued: u64 = 0; // rounds submitted
    var read: u64 = 0; // rounds whose records the host took
    const log = std.c.getenv("TF_ROUND_LOG") != null;
    var depth = Depth{ .levels = o.depth, .most = o.depth, .fixed = o.fixed };
    var windows: [ring]usize = undefined; // each queued round's verify rows
    var levels = depth.levels; // the levels the last queued head drafts (the next verify's rows - 1)
    var copy: u32 = 0; // the copy rows the last queued round may take for the next window
    var copies = copy_lanes.Copy{ .most = o.copy_max };
    var sib_offered: f64 = 0; // second choices' chances offered, and how many landed
    var sib_landed: f64 = 0;
    var widths: [32]usize = undefined;
    var window_ms: [32]f64 = undefined;
    for (b.costs.window[0..b.costs.windows], 0..) |w, i| {
        widths[i] = w.width;
        window_ms[i] = w.ms;
    }
    const costs = copy_lanes.Costs{ .rows = window_ms[0..b.costs.windows], .widths = widths[0..b.costs.windows] };
    const Pre = struct {
        r: *Round,
        levels: usize,
        rate: f64,
        chunks: usize,
        pub fn encode(j: @This(), _: *Metal, e: *fwd.Enc) !void {
            j.r.chain(e, j.r.wins[0], 0, j.levels, j.rate, j.chunks, false, false);
        }
    };
    try b.submit(.draft, b.next, Pre{ .r = &r, .levels = levels, .rate = rate, .chunks = (mtp_known + o.depth + 511) / 512 + 1 });
    const Job = struct {
        r: *Round,
        n: u64,
        rows: usize,
        levels: usize,
        copy: u32,
        tail: bool,
        rate: f64,
        chunks: usize,
        head_chunks: usize,
        pub fn encode(j: @This(), _: *Metal, e: *fwd.Enc) !void {
            j.r.round(e, j.n, j.rows, j.levels, j.copy, j.tail, j.rate, j.chunks, j.head_chunks);
        }
    };
    const Head = struct {
        r: *Round,
        n: u64,
        rows: usize,
        levels: usize,
        copy: u32,
        tail: bool,
        rate: f64,
        head_chunks: usize,
        pub fn encode(j: @This(), _: *Metal, e: *fwd.Enc) !void {
            j.r.draft(e, j.n, j.rows, j.levels, j.copy, j.tail, j.rate, j.head_chunks);
        }
    };
    var inflight: usize = 0; // rows the queued, unread rounds' windows take at most
    var pausing = false; // a hand-over asked: read the queued rounds, queue none
    var last_kept: u32 = 0;
    while (!s.finished) {
        // near the length limit (by the rows the rounds in flight and the next can keep), read every round first
        const sib_next: usize = @intFromBool(o.trees and o.siblings);
        const next_rows = @max(levels + 1 + sib_next, if (copy > 0) copy + 1 + sib_next else 0);
        const close = @as(i64, @intCast(s.emitted().len + inflight + next_rows)) >= s.max_new;
        const limit: u64 = if (close or pausing) 0 else o.ahead - 1;
        while (queued - read > limit) {
            const per: u64 = if (o.split) 2 else 1;
            while (b.flights.items.len > (queued - read) * per) try b.land();
            for (0..per) |_| try b.land();
            const x = Round.u32s(r.recs, ring * rec)[(read % ring) * rec ..][0..rec];
            const kept = x[0];
            out.rounds += 1;
            out.accepted += kept - 1;
            out.by_rows[x[1]] += 1;
            if (x[2] == 0) depth.observe(kept) else out.copied += 1;
            if (x[2] == 0) out.head_kept[@min(x[1], 16)][@min(kept, 16)] += 1;
            out.siblings += @intFromBool(x[58] == 1);
            out.siblings_kept += if (x[58] == 1) x[62] else 0;
            out.guesses += @intFromBool(x[58] == 2);
            out.guesses_kept += if (x[58] == 2) x[62] else 0;
            if (x[58] == 1) {
                sib_offered += @as(f32, @bitCast(x[59]));
                sib_landed += @floatFromInt(x[62]);
                r.sib_scale = (sib_landed + 1) / (sib_offered + 1);
            }
            // a copy's round: the head absorbs, drafts nothing; a head window's levels are its drafts (not its tail)
            const head_levels: f64 = if (x[2] == 0) @floatFromInt(x[1] - x[60]) else 1;
            const head_part = costs.at(x[1] - x[60]) + b.costs.head_ms * head_levels;
            copies.observe(x[2], x[60], x[1], kept, x[3] >= o.min_match, costs.at(x[1]) + b.costs.head_ms * head_levels, head_part);
            if (log) std.debug.print("round {d}: window {d} rows {d} kept {d} copy {d} tail {d} match {d} second {d}/{d} at token {d}; drafts {any}\n", .{ read, windows[read % ring], x[1], kept, x[2], x[60], x[3], x[61], x[62], s.emitted().len, x[64 .. 63 + x[1]] });
            len_known += kept;
            mtp_known += kept;
            _ = try s.commit(gpa, x[4 .. 4 + kept]);
            hooks.note();
            last_kept = kept;
            inflight -= windows[read % ring];
            read += 1;
            // the stop's bars price a head row against the head's own windows (a copy's tokens come apart)
            const ms = (mtl.clock.seconds() - t1) * 1e3 * copies.headShare();
            const head_tokens = if (copies.all_ms > copies.head_ms) copies.head_tokens else @as(f64, @floatFromInt(s.emitted().len - 1));
            if (ms > 0) rate = head_tokens / ms;
            if (s.finished) break;
        }
        if (s.finished) break;
        if (pausing) {
            try b.drain();
            r.handOver(s, read, last_kept, len_known, mtp_known);
            handed = true;
            break;
        }
        if (hooks.asked()) {
            pausing = true;
            continue;
        }
        const ahead = (queued - read + 1) * rows;
        const head_chunks = (mtp_known + ahead + rows + o.depth + 511) / 512;
        // this verify takes the rows the last round could have chosen: its head's levels, or its copy
        const sib: usize = @intFromBool(o.trees and o.siblings);
        const verify_rows = @max(levels + 1 + sib, if (copy > 0) copy + 1 + sib else 0);
        levels = depth.next();
        copy = if (o.copy) copies.rows(costs, b.costs.head_ms) else 0;
        const tail = o.copy and o.tail and copies.tails();
        if (log) std.debug.print("queue {d}: verify rows {d}, head levels {d}, copy rows {d}\n", .{ queued, verify_rows, levels, copy });
        windows[queued % ring] = verify_rows;
        inflight += verify_rows;
        try b.submit(.verify, b.next, Job{ .r = &r, .n = queued, .rows = verify_rows, .levels = levels, .copy = copy, .tail = tail, .rate = rate, .chunks = (len_known + ahead + rows + 511) / 512, .head_chunks = head_chunks });
        if (o.split) try b.submit(.draft, b.next, Head{ .r = &r, .n = queued, .rows = verify_rows, .levels = levels, .copy = copy, .tail = tail, .rate = rate, .head_chunks = head_chunks });
        queued += 1;
    }
    out.decode_s = mtl.clock.seconds() - t1;
    out.paused = handed;
    s.rounds += out.rounds;
    s.accepted += out.accepted;
    for (out.by_rows, 0..) |k, n| if (k > 0) {
        s.drafted += @as(u64, k) * @as(u64, n - 1);
        if (s.min_rows == 0 or n < s.min_rows) s.min_rows = @intCast(n);
    };
    if (handed) return out;
    if (o.profile > 0) {
        try b.drain();
        var prof = fwd.Profiler{ .queue = b.m.queue, .kernels = &b.m.kernels };
        for (0..o.profile) |i| {
            const pool = mtl.objc.Pool.push();
            defer pool.pop();
            var e = fwd.Enc{ .e = prof.begin(), .prof = &prof };
            r.round(&e, queued + i, levels + 1, levels, 0, false, rate, (len_known + 4 * rows + 511) / 512, (mtp_known + 4 * rows + 511) / 512);
            e.e.end();
            prof.cb.?.commit();
            prof.cb.?.wait();
        }
        std.debug.print("a GPU round's dispatches alone ({d} rows a verify, {d} levels):\n", .{ levels + 1, levels });
        prof.report(o.profile);
    }
    if (o.level_bench) {
        // every level live (zero bars), each chain alone in its command buffer
        r.o.stop = false;
        if (std.c.getenv("TF_HEAD_GEOMETRY")) |g| b.head.?.geometry = std.fmt.parseInt(usize, std.mem.span(g), 10) catch 0;
        var med: [2]f64 = undefined;
        for ([_]usize{ 1, @min(8, o.depth) }, 0..) |n, k| {
            var ms: [9]f64 = undefined;
            for (&ms) |*x| {
                @memcpy(Round.u32s(r.state, 10)[7..10], &[3]u32{ 1, @bitCast(@as(f32, 1)), 1 });
                try b.submit(.draft, b.next, Pre{ .r = &r, .levels = n, .rate = rate, .chunks = (mtp_known + 4 * rows + 511) / 512 });
                try b.drain();
                x.* = b.last_ms;
            }
            std.mem.sort(f64, &ms, {}, std.sort.asc(f64));
            med[k] = ms[4];
        }
        std.debug.print("head chains alone, medians of 9: 1 level {d:.3} ms, {d} levels {d:.3} ms; a live level {d:.3} ms\n", .{ med[0], @min(8, o.depth), med[1], (med[1] - med[0]) / @as(f64, @floatFromInt(@min(8, o.depth) - 1)) });
        // levels past a stop at level 0 (a bar no chance clears)
        r.o.stop = true;
        r.o.bar = 2;
        var dead: [9]f64 = undefined;
        for (&dead) |*x| {
            @memcpy(Round.u32s(r.state, 10)[7..10], &[3]u32{ 1, @bitCast(@as(f32, 1)), 1 });
            try b.submit(.draft, b.next, Pre{ .r = &r, .levels = @min(8, o.depth), .rate = rate, .chunks = (mtp_known + 4 * rows + 511) / 512 });
            try b.drain();
            x.* = b.last_ms;
        }
        std.mem.sort(f64, &dead, {}, std.sort.asc(f64));
        std.debug.print("a level past the stop: {d:.4} ms\n", .{(dead[4] - med[0]) / @as(f64, @floatFromInt(@min(8, o.depth) - 1))});
        r.o.stop = false;
        // the live levels' dispatches, each alone: GPU ms by kernel a level
        var prof = fwd.Profiler{ .queue = b.m.queue, .kernels = &b.m.kernels };
        for (0..4) |_| {
            const pool = mtl.objc.Pool.push();
            defer pool.pop();
            @memcpy(Round.u32s(r.state, 10)[7..10], &[3]u32{ 1, @bitCast(@as(f32, 1)), 1 });
            var e = fwd.Enc{ .e = prof.begin(), .prof = &prof };
            r.chain(&e, r.wins[0], 0, 8, rate, (mtp_known + 4 * rows + 511) / 512, false, false);
            e.e.end();
            prof.cb.?.commit();
            prof.cb.?.wait();
        }
        std.debug.print("8 live levels' dispatches alone (a level = 1 step here: divide by 8):\n", .{});
        prof.report(4);
    }
    return out;
}
