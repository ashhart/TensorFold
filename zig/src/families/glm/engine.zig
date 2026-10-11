//! GLM-5.3-Flash on Metal: one greedy reply at a time, the prompt in 16-row windows or tensor-unit chunks, then drafted rounds.
const std = @import("std");
const mtl = @import("metal");
const cfg = @import("config.zig");
const wts = @import("weights.zig");
const page_cache = @import("page_cache.zig");
const st = @import("state.zig");
const fwd = @import("forward.zig");
const mtp = @import("mtp.zig");
const prompt_mod = @import("prompt.zig");
const Fence = @import("../../core/fence.zig").Fence;
const kernels = @import("kernels.zig");
const ep_mod = @import("ep.zig");
const CopyIndex = @import("../../core/copy_index.zig").CopyIndex;
const checks = @import("checks.zig");
const load_plan = @import("load_plan.zig");
const plain_rounds = @import("plain_rounds.zig");
const Ref = wts.Ref;

pub const Reason = enum { stop, length, cancelled };

pub const Result = struct {
    reason: Reason,
    rounds: u64 = 0,
    drafted: u64 = 0,
    accepted: u64 = 0,
    min_rows: u32 = 0,
    prompt_seconds: f64 = 0,
    decode_seconds: f64 = 0,
    generated: u64 = 0,
    gpu_seconds: f64 = 0, // the rounds' GPU time (each command buffer's start to end)
    encode_seconds: f64 = 0, // the host's time from a round's first encode to its commit
    gap_seconds: f64 = 0, // the GPU idle between consecutive rounds
    copy_rounds: u64 = 0, // rounds whose drafts were copied from earlier in the history, and their drafts kept
    copy_accepted: u64 = 0,
    // GLM_RANKS=1: rounds by drafts kept m: [m][0] all kept, else the target's place among the head's choices at the miss
    draft_ranks: [st.max_rows][4]u32 = @splat(@splat(0)),
};

/// What a reply reports while it runs (called on the engine's thread).
pub const Out = struct {
    ctx: *anyopaque,
    prefilled: *const fn (ctx: *anyopaque) void,
    /// Tokens committed, in order; true ends the reply as a stop (a stop string matched).
    tokens: *const fn (ctx: *anyopaque, toks: []const u32) bool,
    cancelled: *const fn (ctx: *anyopaque) bool,
};

/// The outputs of a reply nobody reads (a warm-up, rank 1's half of rank 0's).
pub const Quiet = struct {
    pub fn prefilled(_: *anyopaque) void {}
    pub fn tokens(_: *anyopaque, _: []const u32) bool {
        return false;
    }
    pub fn cancelled(_: *anyopaque) bool {
        return false;
    }
};

pub const Engine = struct {
    gpa: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    event: mtl.SharedEvent,
    ev: u64 = 0,
    c: cfg.Config,
    k: *kernels.Kernels,
    w: *wts.Weights,
    arena: st.Arena,
    s: st.State,
    sc: st.Scratch,
    prompt_ids: Ref,
    pick_ring: Ref, // u32 [2]: pipelined plain rounds' picks, a round's input read from the one before's on the GPU
    round_fence: ?Fence = null, // orders pipelined plain rounds on the GPU (a shared event's wait is slower)
    pr: ?prompt_mod.Prompt, // prompt chunks on the tensor units (null: every prompt row in 16-row decode windows)
    ep: ?*ep_mod.Ep, // expert parallel with a peer Mac (GLM_EP names this Mac's link settings)
    trace_last: ?Ref = null, // a prompt's last row at every capture point (each layer's sublayers), for a path comparison
    margins: ?*std.ArrayList(f32) = null, // each emitted token's top-two logit margin (its row's logits), for a path comparison
    chunk_rows: u32 = prompt_mod.max_rows, // a prompt chunk's rows at most (GLM_CHUNK: smaller, to check chunk-size invariance)
    model_hash: u64 = 0, // the checkpoint's config and weight index, hashed (a peer's and a learned state's identity)
    cut: u32 = 0, // GLM_CUTS=N: a prompt chunk also ends at N (a server's planned start, for its served == CLI check)
    copy_min: u32 = 0, // copy drafts (GLM_COPY=N): a round copies what followed the reply's last N+ tokens earlier (0: off)
    rank_log: bool = false, // GLM_RANKS=1: each MTP depth's logits kept, and the target's rank in them where drafts miss
    ep_arena: std.heap.ArenaAllocator, // the link settings, alive as long as the link
    residency: ?mtl.ResidencySet = null,
    keepalive_sets: [1]mtl.ResidencySet = undefined, // the residency set, for the idle keepalive's commit
    keepalive_target: mtl.keepalive.Target = undefined, // the engine's queue, set at load for the server's ticker
    load_seconds: f64 = 0,
    limit: load_plan.Limit = undefined, // the one-Mac load limit, read once at load (stream admission and the cache budget read it)
    gpu: [2]f64 = .{ 0, 0 }, // the last command buffer's GPU start and end (host seconds)
    fused_route: bool = true, // GLM_ROUTE=0: the Python family's cast, router and top-k launches
    draft_vocab: u32 = 154880, // GLM_DRAFT_VOCAB: the MTP head drafts from the vocabulary's first this many tokens
    committed: u64 = 0, // when the last command buffer was committed (mach ticks)

    /// The checkpoint in `dir`, caches for `cap` tokens; GLM_LAYERS=N: the first N layers only; GLM_EP=settings: half the experts.
    pub fn load(gpa: std.mem.Allocator, dir: []const u8, cap: u32) !*Engine {
        return loadWith(gpa, dir, cap, if (std.c.getenv("GLM_EP")) |v| std.mem.span(v) else null, false);
    }

    /// `load` with expert parallel over the link in `ep_path` (this Mac's settings), or on one Mac when null.
    pub fn loadWith(gpa: std.mem.Allocator, dir: []const u8, cap: u32, ep_path: ?[]const u8, learn: bool) !*Engine {
        const e = try gpa.create(Engine); // undefined memory: every field is set below
        errdefer gpa.destroy(e);
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const t0 = std.c.mach_absolute_time();
        e.gpa = gpa;
        e.ev = 0;
        e.residency = null;
        e.gpu = .{ 0, 0 };
        e.fused_route = if (std.c.getenv("GLM_ROUTE")) |v| v[0] != '0' else true;
        e.draft_vocab = if (std.c.getenv("GLM_DRAFT_VOCAB")) |v| std.fmt.parseInt(u32, std.mem.span(v), 10) catch 0 else 0;
        e.committed = 0;
        e.ep = null;
        e.pr = null;
        e.trace_last = null;
        e.margins = null;
        e.chunk_rows = prompt_mod.max_rows;
        e.cut = if (std.c.getenv("GLM_CUTS")) |v| std.fmt.parseInt(u32, std.mem.span(v), 10) catch 0 else 0;
        e.copy_min = if (std.c.getenv("GLM_COPY")) |v| std.math.clamp(std.fmt.parseInt(u32, std.mem.span(v), 10) catch 0, 0, 8) else 0;
        e.rank_log = if (std.c.getenv("GLM_RANKS")) |v| v[0] == '1' else false;
        if (std.c.getenv("GLM_CHUNK")) |v| e.chunk_rows = std.math.clamp(std.fmt.parseInt(u32, std.mem.span(v), 10) catch prompt_mod.max_rows, st.max_rows + 1, prompt_mod.max_rows);
        e.ep_arena = .init(gpa);
        errdefer e.ep_arena.deinit();
        e.device = try mtl.Device.init();
        errdefer e.device.deinit();
        e.queue = try e.device.queue();
        errdefer e.queue.deinit();
        e.event = try e.device.sharedEvent();
        errdefer e.event.deinit();
        const path = try std.fmt.allocPrintSentinel(gpa, "{s}/config.json", .{dir}, 0);
        defer gpa.free(path);
        const f = try mtl.MappedFile.open(path);
        defer f.deinit();
        e.c = try cfg.parse(gpa, f.bytes[0..f.size]);
        if (e.draft_vocab == 0 or e.draft_vocab > e.c.vocab) e.draft_vocab = e.c.vocab;
        e.draft_vocab -= e.draft_vocab % 4; // the head's kernel takes four rows a simdgroup
        const model = try load_plan.modelHash(gpa, dir, f.bytes[0..f.size]);
        e.model_hash = model;
        if (std.c.getenv("GLM_LAYERS")) |v| try cfg.subset(&e.c, std.fmt.parseInt(u32, std.mem.span(v), 10) catch return error.BadLayerCount);
        if (std.c.getenv("GLM_MTP")) |v| if (v[0] == '0') { // no MTP head: no drafts, its ~4 GB left on disk
            e.c.mtp = 0;
        };
        const link: ?ep_mod.Settings = if (ep_path) |sp| blk: {
            const sf = try mtl.MappedFile.open(try e.ep_arena.allocator().dupeSentinel(u8, sp, 0));
            defer sf.deinit();
            const s = try ep_mod.readSettings(e.ep_arena.allocator(), sf.bytes[0..sf.size]);
            const by_rows = if (std.c.getenv("GLM_EP_SPLIT")) |v| !std.mem.eql(u8, std.mem.span(v), "experts") else true;
            if (by_rows) try cfg.splitRows(&e.c, s.rank, 2) else try cfg.split(&e.c, s.rank, 2);
            const tp = if (std.c.getenv("GLM_TP")) |v| v[0] != '0' else true; // TP2: each Mac half the KDA heads (GLM_TP=0: off)
            if (by_rows and tp) try cfg.splitHeads(&e.c, s.rank, 2);
            if (e.c.tp > 1) { // TP2's head halves: the whole vocabulary (its kernel's pitch); no host logits to rank
                e.draft_vocab = e.c.vocab;
                e.rank_log = false;
            }
            break :blk s;
        } else null;
        e.k = try kernels.load(gpa, e.device);
        errdefer {
            e.k.deinit();
            gpa.destroy(e.k);
        }
        const chunked = if (std.c.getenv("GLM_PROMPT")) |v| v[0] != '0' else true;
        e.limit = try load_plan.loadLimit(e.device);
        const limit = e.limit.bytes;
        var plan_bytes = try load_plan.planBytes(gpa, e.device, dir, &e.c);
        if (e.c.mtp > 0 and link == null and plan_bytes + try load_plan.leastArena(gpa, &e.c, cap, chunked) > limit) { // one Mac: the MTP head stays on disk when only it passes the limit
            var lean = e.c;
            lean.mtp = 0;
            const lean_bytes = try load_plan.planBytes(gpa, e.device, dir, &lean);
            if (lean_bytes + try load_plan.leastArena(gpa, &lean, cap, chunked) <= limit) {
                std.log.info("glm: the MTP head would pass this Mac's {d:.1} GB load limit ({s}); loading without it (no MTP drafts, copy drafts stay)", .{ @as(f64, @floatFromInt(limit)) / 1e9, e.limit.source() });
                e.c = lean;
                plan_bytes = lean_bytes;
            }
        }
        if (std.c.getenv("GLM_DRY") != null) return error.DryRun;
        e.arena = .{ .device = e.device, .gpa = gpa };
        errdefer e.arena.deinit();
        const both = try st.init(&e.arena, &e.c, cap);
        e.s = both.state;
        e.sc = both.scratch;
        e.prompt_ids = try e.arena.buffer(@as(usize, cap) * 4);
        e.pick_ring = try e.arena.buffer(256);
        if (chunked and (link == null or e.c.byRows())) {
            const rows = load_plan.chunkHeight(gpa, &e.c, &e.sc, cap, plan_bytes + e.arena.bytes, limit, link != null);
            e.pr = try prompt_mod.init(gpa, &e.arena, e.device, &e.c, &e.sc, e.k, cap, rows);
            e.chunk_rows = @min(e.chunk_rows, rows);
        }
        errdefer if (e.pr) |*p| p.deinit();
        if (plan_bytes + e.arena.bytes > limit) { // refused before any weight is read: the floor's one-Mac limit
            std.log.err("glm: {d:.1} GB of weights and {d:.1} GB of caches pass this Mac's {d:.1} GB load limit ({s}); load a layer subset or the expert-parallel pair, or set GLM_LOAD_LIMIT_GB (past 70% of RAM a load risks a watchdog panic)", .{ @as(f64, @floatFromInt(plan_bytes)) / 1e9, @as(f64, @floatFromInt(e.arena.bytes)) / 1e9, @as(f64, @floatFromInt(limit)) / 1e9, e.limit.source() });
            return error.OverMemoryLimit;
        }
        { // free memory short of the load: a cached copy of the checkpoint holds it, the paced reads would wait (#583)
            const dropped_at = std.c.mach_absolute_time();
            if (page_cache.dropForLoad(gpa, dir, plan_bytes + e.arena.bytes)) |d| {
                if (d.dropped > 0) std.log.info("glm: dropped {d:.1} GiB of the checkpoint from the file cache, the {d:.1} GiB free memory lacked for the load ({d:.2} s)", .{ @as(f64, @floatFromInt(d.dropped)) / (1 << 30), @as(f64, @floatFromInt(d.deficit)) / (1 << 30), @as(f64, @floatFromInt(std.c.mach_absolute_time() - dropped_at)) / 24e6 });
            } else |err| std.log.warn("glm: could not drop the checkpoint's cached pages ({s}); the load reads beside them", .{@errorName(err)});
        }
        e.w = try wts.load(gpa, e.device, dir, &e.c, 16, false);
        errdefer {
            e.w.deinit();
            gpa.destroy(e.w);
        }
        try e.prepare();
        if (link) |s| {
            const rows = e.c.byRows();
            var me: ep_mod.Identity = .{ .layers = e.c.layers, .run = e.c.run, .mtp = @intFromBool(e.w.mtp != null), .experts = if (rows) e.c.moe_inter else e.c.experts, .own_lo = if (rows) e.c.inter[0] else e.c.own[0], .own_hi = if (rows) e.c.inter[1] else e.c.own[1], .cap = cap, .model = model };
            me.rest[0] = @intFromBool(rows);
            me.rest[1] = @intCast(e.c.tp);
            me.rest[2] = @intFromBool(learn); // --learn on both Macs or neither: each holds its half of a learned state
            e.ep = try ep_mod.Ep.init(gpa, e.device, s, me);
        }
        // opt-in: wiring 181 GB leaves macOS nothing to reclaim if another model shares the Mac (Flash Next runs without)
        if (std.c.getenv("GLM_RESIDENCY") == null) {} else if (e.device.residencySet(e.w.buffers.items.len + e.arena.buffers.items.len)) |set| {
            for (e.w.buffers.items) |b| set.add(b);
            for (e.arena.buffers.items) |b| set.add(b);
            set.commit();
            set.requestResidency();
            e.queue.addResidencySet(set);
            e.residency = set;
        } else |_| {}
        // every Metal engine offers its queue to the idle keepalive; the sets list only when the model is resident
        if (e.residency) |set| e.keepalive_sets[0] = set;
        e.keepalive_target = .{ .queue = e.queue, .sets = if (e.residency != null) &e.keepalive_sets else &.{} };
        // a process's first prompt pays a one-time cost: every rank of a pair pays it here at the same step (GLM_NO_WARMUP: skip)
        if (e.pr != null and cap >= 128 and std.c.getenv("GLM_NO_WARMUP") == null) {
            var ids: [64]u32 = undefined;
            for (&ids, 0..) |*t, i| t.* = @intCast(1000 + i);
            var quiet: Quiet = .{};
            _ = try e.generate(&ids, 2, &.{}, 0, .{ .ctx = &quiet, .prefilled = Quiet.prefilled, .tokens = Quiet.tokens, .cancelled = Quiet.cancelled });
        }
        e.load_seconds = @as(f64, @floatFromInt(std.c.mach_absolute_time() - t0)) / 24e6;
        return e;
    }

    /// The KDA decay rates A = exp(A_log) with MLX's Exp, on the GPU.
    fn prepare(e: *Engine) !void {
        const cb = e.queue.commandBuffer();
        const enc = cb.compute(.serial);
        for (0..e.c.run) |li| switch (e.w.layers[li].attn) {
            .kda => |*a| {
                enc.setPipeline(e.k.exp_f32);
                enc.setBuffer(a.a_log.buf, a.a_log.off, 0);
                enc.setBuffer(a.a.buf, a.a.off, 1);
                enc.setValue(e.c.kda_heads, 2);
                enc.dispatchThreads(mtl.Size.of(e.c.kda_heads, 1, 1), mtl.Size.of(64, 1, 1));
            },
            .mla => {},
        };
        enc.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |msg| {
            std.log.err("glm: prepare failed: {s}", .{msg});
            return error.GpuFailed;
        }
    }

    pub fn deinit(e: *Engine) void {
        const gpa = e.gpa;
        if (e.residency) |set| {
            e.queue.removeResidencySet(set);
            set.deinit();
        }
        if (e.ep) |ep| ep.deinit(gpa);
        e.ep_arena.deinit();
        if (e.pr) |*p| p.deinit();
        if (e.round_fence) |f| f.deinit();
        e.arena.deinit();
        e.w.deinit();
        gpa.destroy(e.w);
        e.k.deinit();
        gpa.destroy(e.k);
        e.event.deinit();
        e.queue.deinit();
        e.device.deinit();
        gpa.destroy(e);
    }

    pub fn hasMtp(e: *const Engine) bool {
        return e.w.mtp != null;
    }

    /// Expert parallel's rank 1: it runs rank 0's requests and slot commands (mirror.follow), never its own.
    pub fn followsPeer(e: *const Engine) bool {
        return if (e.ep) |ep| ep.rank == 1 else false;
    }

    /// Rank 1's follower (mirror.follow) ends its wait for rank 0's next command.
    pub fn stopFollowing(e: *Engine) void {
        if (e.ep) |ep| ep.ctl.stop.store(true, .release);
    }

    /// This Mac's decision to stop at a step; with a peer, rank 0's decision, the one both Macs take.
    pub fn agree(e: *Engine, quit: bool) !bool {
        const ep = e.ep orelse return quit;
        return ep.ctl.agree(quit);
    }

    pub fn ctx(e: *Engine) fwd.Ctx {
        return .{ .k = e.k, .c = &e.c, .w = e.w, .s = &e.s, .sc = &e.sc, .ep = e.ep, .fused_route = e.fused_route, .draft_vocab = e.draft_vocab };
    }

    /// A command buffer ordered after the last one this engine committed.
    pub fn begin(e: *Engine) struct { cb: mtl.CommandBuffer, enc: mtl.ComputeEncoder } {
        const cb = e.queue.commandBuffer();
        if (e.ev > 0) cb.waitFor(e.event, e.ev);
        return .{ .cb = cb, .enc = cb.compute(.serial) };
    }

    /// Commit and wait: no work of this engine is in flight once it returns, whatever the caller does next.
    pub fn finish(e: *Engine, cb: mtl.CommandBuffer, enc: mtl.ComputeEncoder) !void {
        return e.finishBy(cb, enc, false);
    }

    /// finish; `spin`: the host spins on the buffer's status instead of sleeping (a window's picks read as it ends).
    pub fn finishBy(e: *Engine, cb: mtl.CommandBuffer, enc: mtl.ComputeEncoder, spin: bool) !void {
        enc.end();
        e.ev += 1;
        cb.signal(e.event, e.ev);
        cb.commit();
        e.committed = std.c.mach_absolute_time();
        if (spin) {
            while (@backingInt(cb.status()) < @backingInt(mtl.types.CommandBufferStatus.completed)) std.atomic.spinLoopHint();
        } else cb.wait();
        e.gpu = .{ cb.gpuStart(), cb.gpuEnd() };
        if (cb.failure()) |msg| {
            e.event.set(e.ev); // a failed buffer may never signal: the next one must not wait on it
            std.log.err("glm: command buffer failed: {s}", .{msg});
            if (e.ep) |ep| ep.failed.store(true, .release); // its exchanges are out of step with the peer's
            return error.GpuFailed;
        }
        if (e.ep) |ep| if (ep.failed.load(.acquire) or ep.gaveUp() > 0) return error.EpLinkFailed;
    }

    /// Wait until every command buffer this engine committed has completed (before the host touches shared state).
    pub fn sync(e: *Engine) void {
        if (e.event.value() >= e.ev) return;
        if (!e.event.wait(e.ev, 120_000)) std.log.err("glm: the GPU did not finish within 120 s", .{});
    }

    /// One bf16 row of `D` values from `src` to `dst`.
    pub fn copyRow(x: *const fwd.Ctx, enc: mtl.ComputeEncoder, src: Ref, dst: Ref, D: u32) void {
        enc.setPipeline(x.k.copy_u32);
        enc.setBuffer(src.buf, src.off, 0);
        enc.setBuffer(dst.buf, dst.off, 1);
        enc.setValue(D / 2, 2);
        enc.dispatchThreads(mtl.Size.of(D / 2, 1, 1), mtl.Size.of(256, 1, 1));
    }

    /// Where `t` falls among MTP depth m + 1's choices (its logits row m): 1 for the 2nd, 2 the 3rd, 3 later or none.
    fn rankAt(e: *const Engine, m: usize, t: u32) usize {
        if (t >= e.draft_vocab) return 3;
        const v: [*]const u16 = @ptrCast(@alignCast(e.sc.m_logits.addr()));
        const row = v[m * e.c.vocab ..][0..e.draft_vocab];
        const lt: f32 = @bitCast(@as(u32, row[t]) << 16);
        var above: usize = 0; // choices before t: higher logits, or equal ones at lower ids
        for (row, 0..) |h, i| {
            const f: f32 = @bitCast(@as(u32, h) << 16);
            if (i != t and (f > lt or (f == lt and i < t))) above += 1;
            if (above >= 3) return 3;
        }
        return @max(above, 1); // the draft holds the first place even when t ties it
    }

    /// The top two logits' difference in row `row` of the last head's logits (bf16).
    fn margin(e: *const Engine, row: u32) f32 {
        if (e.c.tp > 1) return std.math.nan(f32); // TP2: this Mac holds half of each row's logits
        const v: [*]const u16 = @ptrCast(@alignCast(e.sc.logits.addr()));
        var top = [2]f32{ -std.math.inf(f32), -std.math.inf(f32) };
        for (v[@as(usize, row) * e.c.vocab ..][0..e.c.vocab]) |h| {
            const f: f32 = @bitCast(@as(u32, h) << 16);
            if (f > top[0]) {
                top[1] = top[0];
                top[0] = f;
            } else if (f > top[1]) top[1] = f;
        }
        return top[0] - top[1];
    }

    pub fn u32s(r: Ref, n: usize) []u32 {
        return @as([*]u32, @ptrCast(@alignCast(r.addr())))[0..n];
    }

    pub const capture = checks.capture;
    pub const trace = checks.trace;
    pub const forced = checks.forced;
    pub const profile = checks.profile;
    pub const checkMatmul = checks.checkMatmul;
    pub const profilePrompt = checks.profilePrompt;

    /// One greedy reply. `depth` drafts a round (0: one token a round, the reference drafted replies must equal).
    pub fn generate(e: *Engine, prompt: []const u32, max_tokens: usize, eos: []const u32, depth: usize, out: Out) !Result {
        const c = &e.c;
        const D = c.hidden;
        const P: u32 = @intCast(prompt.len);
        if (prompt.len == 0) return error.EmptyPrompt;
        const d: u32 = @intCast(if (e.w.mtp == null) 0 else @min(depth, st.max_rows - 1));
        if (prompt.len + max_tokens + d + 1 > e.s.cap) return error.ContextFull;
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        e.sync();
        e.s.reset();
        @memcpy(u32s(e.prompt_ids, prompt.len), prompt);
        var x = e.ctx();
        const t_start = std.c.mach_absolute_time();
        // the prompt: windows of up to 16 rows, the MTP head taking each row with the token after it
        var at: u32 = 0;
        var last_n: u32 = 1;
        while (at < P) {
            if (try e.agree(out.cancelled(out.ctx))) return .{ .reason = .cancelled };
            const end = if (e.cut > at and e.cut < P) e.cut else P;
            const chunk = e.pr != null and end - at > st.max_rows;
            const n = @min(@as(u32, if (chunk) e.chunk_rows else st.max_rows), end - at);
            const last = at + n == P;
            const absorb = if (last) n - 1 else n;
            const b = e.begin();
            if (last and e.trace_last != null) {
                x.dump = e.trace_last;
                x.dump_at = 0;
                x.dump_row = n - 1;
            }
            defer x.dump = null;
            if (chunk) {
                const pr = &e.pr.?;
                var px = x;
                px.sc = &pr.streams;
                prompt_mod.backbone(pr, &px, b.enc, e.prompt_ids.at(@as(usize, at) * 4), n, at);
                fwd.flipKda(&x);
                const hidden = pr.streams.hidden;
                if (last) { // the last row into the decode scratch: the head's first token, the MTP head's first row
                    copyRow(&x, b.enc, hidden.at(@as(usize, n - 1) * D * 2), e.sc.hidden, D);
                    fwd.head(&x, b.enc, e.sc.hidden, e.sc.logits, e.sc.picks, 1);
                }
                if (d > 0 and absorb > 0) {
                    prompt_mod.mtp(pr, &px, b.enc, hidden, e.prompt_ids.at(@as(usize, at + 1) * 4), absorb, at);
                    e.s.mtp_pos = at + absorb;
                }
                last_n = 1;
            } else {
                fwd.backbone(&x, b.enc, e.prompt_ids.at(@as(usize, at) * 4), n, at);
                fwd.flipKda(&x);
                if (last) fwd.head(&x, b.enc, e.sc.hidden.at(@as(usize, n - 1) * D * 2), e.sc.logits, e.sc.picks, 1);
                if (d > 0 and absorb > 0) {
                    mtp.run(&x, b.enc, e.sc.hidden, e.prompt_ids.at(@as(usize, at + 1) * 4), absorb, at, e.sc.m_picks);
                    e.s.mtp_pos = at + absorb;
                }
                last_n = n;
            }
            try e.finish(b.cb, b.enc);
            at += n;
        }
        e.s.pos = P;
        const t_prompt = std.c.mach_absolute_time();
        out.prefilled(out.ctx);
        var res: Result = .{ .reason = .length, .prompt_seconds = @as(f64, @floatFromInt(t_prompt - t_start)) / 24e6 };
        var tok = u32s(e.sc.picks, 1)[0];
        if (e.margins) |m| try m.append(e.gpa, e.margin(0));
        res.generated = 1;
        if (try e.agree(out.tokens(out.ctx, &.{tok}) or std.mem.indexOfScalar(u32, eos, tok) != null)) {
            res.reason = .stop;
            return res;
        }
        if (max_tokens <= 1) return res;
        if (d == 0 and e.ep == null and e.copy_min == 0 and e.margins == null) {
            try plain_rounds.run(e, &x, tok, max_tokens, eos, out, &res);
            res.decode_seconds = @as(f64, @floatFromInt(std.c.mach_absolute_time() - t_prompt)) / 24e6;
            return res;
        }
        var emitted: usize = 1;
        var hist: ?CopyIndex = null; // the prompt and the reply so far: copy drafts' source
        defer if (hist) |*h| h.deinit();
        // a copy round's window: the most expected tokens a ms against the pair's measured round costs, when that beats an MTP round
        var ck: f64 = 0;
        var cr: f64 = 0;
        var mk: f64 = @floatFromInt(d);
        if (e.copy_min > 0) { // copy drafts need no MTP head: without one a round is the copied tokens or one plain row
            hist = try CopyIndex.init(e.gpa, prompt);
            try hist.?.extend(&.{tok});
        }
        var h0: u32 = last_n - 1; // the rows of `hidden` the MTP head takes next, and how many
        var keep: u32 = 1;
        var window: u32 = 0; // the last forward's rows (0: the prompt's)
        res.min_rows = d + 1;
        u32s(e.sc.next, 1)[0] = tok;
        var last_end: f64 = 0;
        while (true) {
            const t_enc = std.c.mach_absolute_time();
            const b = e.begin();
            if (window > 0) { // the last window's kept rows (the prompt's windows kept all theirs)
                fwd.keepKda(&x, b.enc, window, keep);
                fwd.flipKda(&x);
            }
            const ids = u32s(e.sc.ids, st.max_rows);
            ids[0] = tok;
            // copy drafts: what followed the reply's last tokens earlier, when they match at least copy_min deep
            var copied: u32 = 0;
            if (hist) |*h| {
                const m = h.longest(8);
                if (m.n >= e.copy_min) {
                    const room = e.s.cap - (e.s.pos + 2);
                    const avail: usize = @min(h.ctx.items.len - m.at, st.max_rows - 1, room);
                    const n: f64 = @floatFromInt(m.n);
                    const a = (ck + n) / (ck + n + cr + 1);
                    var best = (1 + mk) / (12.9 + 4.3 * @as(f64, @floatFromInt(d))); // an MTP round's tokens a ms
                    var expect: f64 = 1;
                    var p: f64 = 1;
                    for (1..avail + 1) |w| {
                        p *= a;
                        expect += p;
                        const rate = expect / (12.9 + 3.5 * @as(f64, @floatFromInt(w)));
                        if (rate > best) {
                            best = rate;
                            copied = @intCast(w);
                        }
                    }
                    if (copied < 2) copied = 0; // a one-token copy is no better than the head's draft
                    if (copied > 0) @memcpy(ids[1 .. 1 + copied], h.ctx.items[m.at .. m.at + copied]);
                }
            }
            if (d > 0) {
                const next = if (window == 0) e.sc.next else e.sc.picks;
                mtp.run(&x, b.enc, e.sc.hidden.at(@as(usize, h0) * D * 2), next, keep, e.s.mtp_pos, if (copied > 0) null else e.sc.ids.at(4));
                const base = e.s.mtp_pos + keep;
                if (copied == 0) for (1..d) |j| {
                    const h = if (j == 1) e.sc.m_x.at(@as(usize, keep - 1) * D * 2) else e.sc.m_x;
                    if (e.rank_log) x.m_row = @intCast(j);
                    mtp.chain(&x, b.enc, h, e.sc.ids.at(j * 4), base + @as(u32, @intCast(j)) - 1, e.sc.ids.at((j + 1) * 4));
                };
                x.m_row = 0;
                e.s.mtp_pos = base;
            }
            const R = if (copied > 0) copied + 1 else d + 1;
            fwd.backbone(&x, b.enc, e.sc.ids, R, e.s.pos);
            fwd.head(&x, b.enc, e.sc.hidden, e.sc.logits, e.sc.picks, R);
            try e.finish(b.cb, b.enc);
            res.encode_seconds += @as(f64, @floatFromInt(e.committed - t_enc)) / 24e6;
            res.gpu_seconds += e.gpu[1] - e.gpu[0];
            if (last_end > 0) res.gap_seconds += e.gpu[0] - last_end;
            last_end = e.gpu[1];
            const picks = u32s(e.sc.picks, R);
            const drafts = u32s(e.sc.ids, R);
            keep = 1;
            while (keep < R and picks[keep - 1] == drafts[keep]) keep += 1;
            res.rounds += 1;
            res.drafted += R - 1;
            res.accepted += keep - 1;
            if (e.rank_log and copied == 0 and d > 0) {
                const m = keep - 1;
                res.draft_ranks[m][if (m == d) 0 else e.rankAt(m, picks[m])] += 1;
            }
            if (copied > 0) {
                res.copy_rounds += 1;
                res.copy_accepted += keep - 1;
                ck += @floatFromInt(keep - 1);
                if (keep - 1 < copied) cr += 1;
            } else mk = 0.8 * mk + 0.2 * @as(f64, @floatFromInt(keep - 1));
            var take: usize = 0;
            var stop = false;
            while (take < keep and emitted + take < max_tokens) {
                take += 1;
                if (std.mem.indexOfScalar(u32, eos, picks[take - 1]) != null) {
                    stop = true;
                    break;
                }
            }
            if (e.margins) |m| for (0..take) |j| try m.append(e.gpa, e.margin(@intCast(j)));
            if (take > 0 and out.tokens(out.ctx, picks[0..take])) stop = true;
            if (hist) |*h| try h.extend(picks[0..take]);
            emitted += take;
            res.generated = emitted;
            e.s.pos += keep;
            tok = picks[keep - 1];
            window = R;
            h0 = 0;
            const quit: ?Reason = if (stop) .stop else if (emitted >= max_tokens) .length else if (out.cancelled(out.ctx)) .cancelled else if (e.s.pos + R + 1 > e.s.cap) .length else null;
            if (try e.agree(quit != null)) {
                res.reason = quit orelse .cancelled; // rank 1: rank 0's reason (a stop string, a cancel) is its own
                break;
            }
        }
        res.decode_seconds = @as(f64, @floatFromInt(std.c.mach_absolute_time() - t_prompt)) / 24e6;
        return res;
    }
};
