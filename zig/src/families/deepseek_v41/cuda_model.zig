//! One rank of the whole model for serving: backbone, head and DSpark at TP=N, forward, GPU target and pass.

const std = @import("std");
const quiet = @import("builtin").is_test; // a test's stderr is the build runner's channel
const pf = @import("forward_prefill.zig");
const cuda = @import("cuda");
const dk = @import("dsv41_kernels");
const tp = @import("tp");
const Config = @import("config.zig").Config;
const Pack = @import("pack.zig").Pack;
const plan = @import("plan.zig");
const named = @import("named.zig");
const load = @import("cuda_weights.zig");
const prepared = @import("prepared.zig");
const buffers = @import("buffers.zig");
const forms = @import("forms.zig");
const forms_inplace = @import("forms_inplace.zig");
const ex_stacks = @import("ex_stacks.zig");
const longpf = @import("longpf.zig");
const run = @import("run.zig");
const fwd = @import("forward.zig");
const eh = @import("engram_host.zig");
const boot_mem = @import("boot_mem.zig");
const gate_forward = @import("gate_forward.zig");
const target_mod = @import("target.zig");
const dspark_emit = @import("dspark_emit.zig");
const dspark_gpu = @import("dspark_gpu.zig");
const kvs = @import("cuda_kv.zig");
const sg = @import("sessions_gpu.zig");
const slots_mod = @import("slots.zig");
const batch = @import("batch.zig");
const samp = @import("sampling_gpu.zig");
const dspark_rows = @import("dspark_rows.zig");
const dspark_slots = @import("dspark_slots.zig");
const l2pf_mod = @import("l2pf.zig");
const iface = @import("draft/iface.zig");

pub const Options = struct {
    drafts: bool = true,
    /// rows a prefill window takes (the decode buckets the plan sizes for) when the prompt runs as decode windows
    prefill_rows: u32 = 16,
    /// the prompt through the forward's own prefill (2,048-row segments) instead: TF_DSV41_OWN_PREFILL (default on)
    own_prefill: bool = true,
    /// the backbone layers to load and run (null: all; a gate runs a reference's prefix, e.g. 0-24)
    layers: ?[]const u32 = null,
};

/// TF_DSV41_LAYERS=<lo>-<hi>: load and run only that backbone prefix (a reference prefix's model; null: every layer)
pub fn layersEnv(gpa: std.mem.Allocator) !?[]const u32 {
    const v = std.mem.span(std.c.getenv("TF_DSV41_LAYERS") orelse return null);
    var it = std.mem.tokenizeScalar(u8, v, '-');
    const lo = try std.fmt.parseInt(u32, it.next() orelse return error.BadLayers, 10);
    const hi = try std.fmt.parseInt(u32, it.next() orelse return error.BadLayers, 10);
    if (hi < lo) return error.BadLayers;
    const ls = try gpa.alloc(u32, hi - lo + 1);
    for (ls, 0..) |*l, i| l.* = lo + @as(u32, @intCast(i));
    return ls;
}

/// TF_DSV41_CONTEXT: the slot's positions; null: block.Options' 4,096
pub fn contextEnv() !?i64 {
    const v = std.c.getenv("TF_DSV41_CONTEXT") orelse return null;
    const c = try std.fmt.parseInt(i64, std.mem.span(v), 10);
    if (c < 1) return error.BadContext;
    return c;
}

/// TF_DSV41_OWN_PREFILL=0: prompts as 16-row decode windows (every rank must agree: the plans differ)
pub fn ownPrefillEnv() bool {
    const v = std.c.getenv("TF_DSV41_OWN_PREFILL") orelse return true;
    return !std.mem.eql(u8, std.mem.span(v), "0");
}

pub const Model = struct {
    gpa: std.mem.Allocator,
    retained_next: ?*Model = null,
    retained: bool = false,
    boot_complete: bool = false,
    /// TF_DSV41_PROFILE's writer at close (the served stop: SIGTERM / SIGINT -> serve_engine deinit -> close)
    io: std.Io,
    arena: std.heap.ArenaAllocator,
    tcfg: tp.Config,
    driver: cuda.Driver,
    ctx: cuda.Context,
    stream: cuda.Stream,
    kernels: dk.Kernels,
    session: tp.Session,
    cfg: Config,
    pack: Pack,
    pl: plan.Plan,
    weights: load.Weights,
    aot: cuda.aot.Set,
    widths: @import("block.zig").Widths,
    runner: run.Runner,
    f: fwd.Forward,
    host: eh.Host,
    lbuf: [eh.max_layers]u32 = undefined,
    gt: target_mod.GpuTarget,
    pass: ?*dspark_gpu.GpuPass = null,
    /// the session store over the paged pool (with TF_DSV41_POOL_TOKENS)
    sessions: ?*sg.Sessions = null,
    /// live slots (TF_DSV41_SLOTS; > 1: stacked state and row-mode windows, batch.zig)
    slots: u32 = 1,
    /// TF_DSV41_ARENA's sub-allocator (dev_arena.zig), for the process's lifetime
    dev_arena: ?*@import("dev_arena.zig").Installed = null,
    /// the boot check's figures before the pool, for its measured end (kv_state.afterBoot)
    boot_check: ?kvs.Boot = null,
    slot_set: ?*slots_mod.SlotSet = null,
    rows: ?*batch.Batch = null,
    /// DSpark drafts over several live slots (TF_DSV41_SLOT_DRAFTS=1 with TF_DSV41_SLOTS > 1, dspark_slots.zig)
    slot_pass: ?*dspark_slots.SlotPass = null,
    /// the keyed sampler, every rank with TF_DSV41_SAMPLING=1 (requests with T > 0; followers run its gathers)
    sampler: ?*samp.GpuSampler = null,
    /// TF_DSV41_L2PF (l2pf.zig): the filled prefetch tables and the runner's side stream
    l2pf: ?*l2pf_mod.L2pf = null,

    /// The served engine's DSpark pass: several slots' (slot_pass), else the one-slot pass; null without drafts.
    pub fn draftPass(m: *Model) ?iface.Pass {
        if (m.slot_pass) |x| return x.pass();
        return if (m.pass) |p| p.pass() else null;
    }

    pub fn rank(m: *const Model) u32 {
        return m.tcfg.rank;
    }

    /// Loads this rank (TF_TP_RANK / WORLD / DEVICE / MASTER / PORT) of `pack_dir` with `assets` (the module doc).
    pub fn open(gpa: std.mem.Allocator, io: std.Io, pack_dir: []const u8, assets: []const u8, o: Options) !*Model {
        if (@import("sys").experimental.refusal("deepseek_v41", @import("sys").getenv("TF_EXPERIMENTAL"))) |text| {
            if (!quiet) std.log.err("{s}", .{text});
            return error.ExperimentalDeepSeekRequired;
        }
        const m = try gpa.create(Model);
        var keep_owner = false;
        errdefer if (keep_owner) retain(m) else gpa.destroy(m);
        m.gpa = gpa;
        m.retained_next = null;
        m.retained = false;
        m.boot_complete = false;
        m.io = io;
        // gpa.create applies no field defaults: optional parts start null so close() never frees garbage
        m.pass = null;
        m.l2pf = null;
        m.sessions = null;
        m.slots = 1;
        m.slot_set = null;
        m.rows = null;
        m.slot_pass = null;
        m.sampler = null;
        m.dev_arena = null;
        m.arena = std.heap.ArenaAllocator.init(gpa);
        errdefer if (!keep_owner) m.arena.deinit();
        const a = m.arena.allocator();
        const cwd = std.Io.Dir.cwd();
        m.tcfg = try tp.Config.fromEnv();
        m.driver = try cuda.Driver.open();
        errdefer if (!keep_owner) m.driver.close();
        try @import("dev_arena.zig").applySchedule(&m.driver, @intCast(m.tcfg.device)); // TF_CUDA_SCHED
        m.ctx = try cuda.Context.init(&m.driver, @intCast(m.tcfg.device));
        keep_owner = true;
        m.stream = try cuda.Stream.init(&m.driver, true);
        m.kernels = try dk.Kernels.load(&m.ctx);
        // TF_TP_KERNEL_IMAGE: the mailbox / RoCE kernels (TF_COMM_BACKEND=roce: the hybrid); unset: NCCL alone
        const image: []const u8 = if (std.c.getenv("TF_TP_KERNEL_IMAGE")) |p|
            try cwd.readFileAllocOptions(io, std.mem.span(p), a, .limited(1 << 26), .@"16", null)
        else
            &.{};
        try m.session.open(&m.driver, m.tcfg, image);
        // TF_DSV41_ARENA: every device buffer from here on (weights, forms, roles, scratch) carved from large chunks
        m.dev_arena = try @import("dev_arena.zig").install(gpa, &m.ctx, m.tcfg.rank);
        // a failure from here to serving: every rank down with its reason, the session closed before `m` is freed
        m.boot(gpa, io, pack_dir, assets, o) catch |err| {
            if (m.ctx.synchronize()) |_| {
                m.session.bootFailed(err);
            } else |_| {}
            return err;
        };
        m.boot_complete = true;
        return m;
    }

    /// `open` from the session up: the pack, weights, forward, buffers, assets and serving parts of this rank.
    fn boot(m: *Model, gpa: std.mem.Allocator, io: std.Io, pack_dir: []const u8, assets: []const u8, o: Options) !void {
        const a = m.arena.allocator();
        const cwd = std.Io.Dir.cwd();
        var tr = boot_mem.Trace.fromEnv();
        tr.step("start", cuda.usage(false));
        const comm = m.session.comm();
        const world: u32 = @intCast(comm.world());
        m.cfg = try Config.read(gpa, io, pack_dir);
        if (o.own_prefill) try @import("block_admission.zig").nativePrefill(&m.cfg, world);
        m.pack = try Pack.open(gpa, io, pack_dir);
        const backbone = if (o.layers) |ls| try a.dupe(u32, ls) else blk: {
            const all = try a.alloc(u32, m.cfg.layers);
            for (all, 0..) |*l, i| l.* = @intCast(i);
            break :blk all;
        };
        var ds: [8]u32 = undefined;
        const blocks = if (o.drafts) try std.mem.concat(a, u32, &.{ backbone, dspark_emit.blocks(&m.cfg, &ds) }) else backbone;
        m.pl = try plan.build(gpa, &m.cfg, &m.pack, .{ .rank = m.tcfg.rank, .world = world, .blocks = blocks, .top = true, .dspark = o.drafts });
        try load.admitPlan(io, &m.cfg, &m.pack, &m.pl);
        m.weights = .{ .gpa = gpa, .digest = false }; // no weights check reads the digests here (load.zig)
        // TF_DSV41_PREPARED (prepared.zig): each layer's host images read back instead of rebuilt from the pack
        var cache = try prepared.Cache.fromEnv(gpa, io, &m.pack, .{ .rank = m.tcfg.rank, .world = world, .o_groups = m.cfg.o_groups, .blocks = blocks, .dspark = o.drafts });
        defer if (cache) |*c| c.deinit();
        var t_build: i96 = 0;
        var t_upload: i96 = 0;
        const t_load = std.Io.Clock.awake.now(io).toNanoseconds();
        for (m.pl.layers, 0..) |*l, i| {
            var b: named.Builder = .{ .gpa = gpa, .io = io, .pack = &m.pack };
            defer b.deinit();
            var tag_buf: [16]u8 = undefined;
            const tag = try std.fmt.bufPrint(&tag_buf, "L{d}", .{i});
            const t0 = std.Io.Clock.awake.now(io).toNanoseconds();
            var schema: named.Builder = .{ .gpa = gpa, .io = io, .pack = &m.pack, .metadata_only = true };
            defer schema.deinit();
            if (cache != null) try schema.layer(l, m.cfg.o_groups, m.tcfg.rank, world);
            if (cache == null or !try cache.?.get(&b, tag, schema.out.items)) {
                try b.layer(l, m.cfg.o_groups, m.tcfg.rank, world);
                if (cache) |*c| c.put(&b, tag) catch |e| if (!quiet) std.log.warn("dsv41 prepared: {s} not written ({t})", .{ tag, e });
            }
            const t1 = std.Io.Clock.awake.now(io).toNanoseconds();
            // the next prepared file read ahead by the kernel while this layer uploads
            if (cache) |*c| {
                var next_buf: [16]u8 = undefined;
                c.willNeed(if (i + 1 < m.pl.layers.len) std.fmt.bufPrint(&next_buf, "L{d}", .{i + 1}) catch "" else "top");
            }
            try m.weights.upload(&m.driver, &b);
            t_build += t1 - t0;
            t_upload += std.Io.Clock.awake.now(io).toNanoseconds() - t1;
        }
        {
            var b: named.Builder = .{ .gpa = gpa, .io = io, .pack = &m.pack };
            defer b.deinit();
            const t0 = std.Io.Clock.awake.now(io).toNanoseconds();
            var schema: named.Builder = .{ .gpa = gpa, .io = io, .pack = &m.pack, .metadata_only = true };
            defer schema.deinit();
            if (cache != null) {
                try schema.top(&m.pl);
                try schema.dspark(&m.pl);
            }
            if (cache == null or !try cache.?.get(&b, "top", schema.out.items)) {
                try b.top(&m.pl);
                try b.dspark(&m.pl);
                if (cache) |*c| c.put(&b, "top") catch |e| if (!quiet) std.log.warn("dsv41 prepared: top not written ({t})", .{e});
            }
            const t1 = std.Io.Clock.awake.now(io).toNanoseconds();
            try m.weights.upload(&m.driver, &b);
            t_build += t1 - t0;
            t_upload += std.Io.Clock.awake.now(io).toNanoseconds() - t1;
        }
        const ms = struct {
            fn of(ns: i96) f64 {
                return @as(f64, @floatFromInt(ns)) / 1e6;
            }
        }.of;
        if (cache) |c| std.log.info("dsv41 load: weights {d:.0} ms ({d:.0} ms images, {d:.0} ms upload + digest), {d:.1} GiB; prepared {s}: {d} read ({d:.1} GiB in {d:.0} ms), {d} built, {d} written ({d:.0} ms)", .{ ms(std.Io.Clock.awake.now(io).toNanoseconds() - t_load), ms(t_build), ms(t_upload), @as(f64, @floatFromInt(m.weights.bytes)) / (1 << 30), c.root, c.stats.hits, @as(f64, @floatFromInt(c.stats.read_bytes)) / (1 << 30), ms(c.stats.read_ns), c.stats.misses, c.stats.written, ms(c.stats.write_ns) }) else std.log.info("dsv41 load: weights {d:.0} ms ({d:.0} ms images from the pack, {d:.0} ms upload + digest), {d:.1} GiB; TF_DSV41_PREPARED unset", .{ ms(std.Io.Clock.awake.now(io).toNanoseconds() - t_load), ms(t_build), ms(t_upload), @as(f64, @floatFromInt(m.weights.bytes)) / (1 << 30) });
        tr.step("weights", cuda.usage(false));
        m.aot = try cuda.aot.Set.load(gpa, io, &m.driver, m.ctx.device, try std.fs.path.join(a, &.{ assets, "aot" }));
        try @import("aot_trace.zig").attach(gpa, io, &m.aot, @import("aot_trace.zig").pathFromEnv());
        m.widths = try gate_forward.widths(a, &m.weights, &m.pl);
        m.runner = .{ .gpa = gpa, .d = &m.driver, .stream = m.stream, .kernels = &m.kernels, .triton = &m.aot, .weights = &m.weights };
        // TF_DSV41_PROFILE=<file>: the served engine's eager windows and prefill segments timed, JSON written at close
        m.runner.prof = if (try @import("profile.zig").Profile.fromEnv(gpa, &m.driver)) |p| p.timer() else null;
        const topp: ?f64 = if (std.c.getenv("TF_DSV41_EXPERT_TOPP")) |v| try std.fmt.parseFloat(f64, std.mem.span(v)) else null;
        // TF_DSV41_R1=1, or the eight separate R1 knobs (the knob reader's r1)
        const r1 = @import("serve_knobs.zig").r1(&@import("serve_knobs.zig").env);
        m.f = fwd.Forward.init(gpa, &m.cfg, &m.widths, .{ .world = world, .rank = m.tcfg.rank, .expert_topp = topp, .r1 = r1, .taps = o.drafts }, &m.runner, comm);
        try @import("engine_hooks.zig").open(&m.f); // the forward's later-layer hooks and prefill state
        m.f.layers = backbone;
        // the rope tables cover the limit + a prefill segment (the assets' rope.bin)
        if (try contextEnv()) |c| {
            m.f.opts.limit = c;
            m.f.opts.rope_rows = c + 2048;
        }
        try @import("serve_knobs_apply.zig").apply(&m.f, a, io, m.tcfg.rank);
        m.slots = try slots_mod.countFromEnv();
        if (try kvs.optionsFromEnv(@intCast(m.f.opts.limit), world)) |ko_| {
            var ko = ko_;
            ko.slots = m.slots;
            // the long-prompt stream scratch the buffer plan will hold (not in the measured terms)
            ko.resident = if (longpf.lazyEnv()) 0 else @import("serve_knobs.zig").streamScratch(&m.cfg, m.f.opts.limit, m.f.opts.prefill_rows);
            // TF_DSV41_PF_4K / _PF_TBO: the prefill workspace a long prompt allocates, priced on top of the worst point
            if (o.own_prefill) ko.workspace = try @import("forward_prefill.zig").workspaceBytes(&m.f, @intCast(m.f.opts.prefill_rows));
            tr.step("forward", cuda.usage(false));
            m.boot_check = try kvs.bootCheck(&m.ctx, &m.cfg, ko, m.tcfg.rank, world);
            m.f.kv = try kvs.Kv.init(gpa, &m.driver, &m.ctx, &m.cfg, ko, &m.runner, comm);
            m.f.opts.pool = m.f.kv.?.options();
            tr.step("kv pool", cuda.usage(false));
        }
        // the buffer plan: every window up to the prefill rows (a verify window is at most a block + 1), DSpark's too
        var buckets: [64]u32 = undefined;
        const nb = @max(o.prefill_rows, m.cfg.dspark_block + 1);
        for (buckets[0..nb], 0..) |*b, i| b.* = @intCast(i + 1);
        var bp = try m.f.plan(buckets[0..nb]);
        // the pass ingests a prompt's prefilled tail in one call (the drafter's window) with the own prefill
        const ingest_rows: u32 = if (o.own_prefill) @max(nb, m.cfg.window) else nb;
        if (o.drafts) try dspark_gpu.GpuPass.plan(m.f.arena.allocator(), &m.f, &bp, ingest_rows);
        if (o.own_prefill) try m.f.planPrefill(&bp, @intCast(m.f.opts.prefill_rows));
        const rows_cap = try batch.capFromEnv();
        if (m.slots > 1) try batch.Batch.plan(&m.f, &bp, m.slots, rows_cap);
        // TF_DSV41_SLOT_DRAFTS=1: DSpark over every live slot (off: drafts are refused with several slots)
        const slot_drafts = o.drafts and m.slots > 1 and dspark_rows.enabled();
        const group = if (slot_drafts) try dspark_rows.groupFromEnv(m.cfg.dspark_block) else 1;
        if (slot_drafts) try dspark_slots.SlotPass.plan(m.f.arena.allocator(), &m.f, &bp, m.slots, group);
        if (m.f.kv) |k| try k.bind(&m.runner, &bp);
        // TF_DSV41_STREAM_LAZY: the stream top-k's scratch out of the plan, allocated by the prompts that stream
        if (longpf.lazyEnv()) if (bp.sizes.fetchOrderedRemove(longpf.lazy_role)) |kv| {
            pf.state(&m.f).long.lazy = .{ .bytes = kv.value };
        };
        // TF_DSV41_EX_STACKS=1: the experts' suh / svh stacks read where the weights hold them (no second copy)
        if (ex_stacks.enabled()) _ = try ex_stacks.bind(&m.runner, &m.weights, &bp);
        // TF_DSV41_FORMS_INPLACE=1: the dense forms over their trellis (not a second copy), unless a program reads it
        if (forms_inplace.enabled()) {
            const ws = &pf.state(&m.f).k4ws.plan;
            _ = try forms_inplace.bind(gpa, &m.runner, &m.weights, if (ws.*) |*w| &.{ &bp, w } else &.{&bp});
        }
        tr.plan(bp.totals());
        try m.runner.bind(&bp);
        tr.step("buffer plan", cuda.usage(false));
        const src: forms.Source = .{ .d = &m.driver, .weights = &m.weights, .plan = &m.pl };
        for (bp.sizes.keys()) |role| {
            if (buffers.scopeOf(role) != .persistent) continue;
            var ra = std.heap.ArenaAllocator.init(gpa);
            defer ra.deinit();
            const bytes = (try forms.build(ra.allocator(), &src, role)) orelse continue;
            try m.f.upload(m.runner.addressOf(role).?, bytes);
        }
        tr.step("forms", cuda.usage(false));
        // TF_DSV41_L2PF: the prefetch tables from the bound weights and forms (the expert tables above), every rank
        m.l2pf = try l2pf_mod.install(gpa, &m.f, &m.weights);
        // no size cap: a 1M limit + 2,048 rows is 538 MB
        const want: usize = @intCast(m.f.opts.rope_rows * 64 * 4);
        const RopeSink = struct {
            m: *Model,
            pub fn table(s: @This(), i: usize, bytes: []const u8) !void {
                const role = ([_][]const u8{ "s.rope.main", "s.rope.comp" })[i];
                const p = s.m.runner.addressOf(role) orelse return;
                try s.m.f.upload(p, bytes);
            }
            pub fn done(s: @This()) !void {
                try s.m.driver.synchronize();
            }
        };
        try boot_mem.uploadRope(gpa, io, try std.fs.path.join(a, &.{ assets, "rope.bin" }), want, RopeSink{ .m = m });
        tr.step("l2pf and rope", cuda.usage(false));
        const ehb = try cwd.readFileAllocOptions(io, try std.fs.path.join(a, &.{ assets, "engram-token-map.bin" }), a, .limited(1 << 26), .@"4", null);
        m.host = try eh.Host.load(ehb, m.cfg.engram_heads, m.cfg.engram_pad, m.cfg.engram_vocab, &m.lbuf);
        m.f.engram = &m.host;
        if (world > 1) m.f.link = &m.session.plan;
        m.f.on_stop = .{ .ctx = &m.session, .run = announceStop };
        const so = try sg.optionsFromEnv();
        const several_ok = m.slots == 1 or so.several;
        if (so.enabled and m.f.kv != null and !several_ok)
            if (!quiet) std.log.scoped(.dsv41).warn("sessions: off with {d} live slots (TF_DSV41_SESSION_SLOTS=1 opens them over every slot; the session-slots check has not run on hardware)", .{m.slots});
        if (so.enabled and m.f.kv == null and std.c.getenv("TF_DSV41_SESSIONS") != null)
            if (!quiet) std.log.scoped(.dsv41).warn("sessions: off, the store needs the paged pool (TF_DSV41_POOL_TOKENS)", .{});
        if (m.f.kv != null and so.enabled and several_ok) {
            // the drafter's blocks (the snapshot carries their rows: the RAM tier's measure)
            var dsb: [8]u32 = undefined;
            const ds_blocks: u32 = if (o.drafts) @intCast(dspark_emit.blocks(&m.cfg, &dsb).len) else 0;
            m.sessions = try sg.Sessions.init(gpa, io, &m.f, &bp, so, .{ .slots = m.slots, .ds_blocks = ds_blocks, .checkpoint = if (so.disk != null) try prepared.packDigest(io, &m.pack) else null });
        }
        tr.step("engram host and sessions", cuda.usage(false));
        m.gt = .{ .f = &m.f, .prefill_rows = o.prefill_rows, .max_rows = nb, .own_prefill = o.own_prefill, .sessions = m.sessions };
        // TF_DSV41_PROMPT_TAIL: the prompt's last token as the first decode row
        m.gt.prompt_tail = try @import("serve_knobs.zig").promptTail(&@import("serve_knobs.zig").env);
        // TF_DSV41_PREFILL=replay (ced.zig): own prefill over [0, n - 1), decoder replay, last token as verify row
        if (pf.state(&m.f).mode == .replay and (!o.own_prefill or m.gt.prompt_tail != .verify)) {
            std.log.scoped(.dsv41).err("TF_DSV41_PREFILL=replay needs TF_DSV41_OWN_PREFILL=1 and TF_DSV41_PROMPT_TAIL=verify, as Python's server (own prefill {}, prompt tail {t})", .{ o.own_prefill, m.gt.prompt_tail });
            return error.BadKnob;
        }
        if (o.drafts) m.pass = try dspark_gpu.GpuPass.init(gpa, &m.f, ingest_rows);
        if (m.slots > 1) try m.openRows(rows_cap, &bp);
        // TF_DSV41_PREFILL_PIECES: two slots' CED prompts interleave, so the stash ring is a slot's own (swapped)
        if (piecesEnv() and m.slot_set != null and pf.state(&m.f).mode == .replay) try m.slot_set.?.swap(&bp, &@import("ced.zig").roles);
        // sessions over several slots: the operations on the activated slot, the row windows' kept ids committed
        if (m.sessions) |x| if (m.rows) |b| {
            x.slot_set = m.slot_set;
            b.sess = x;
        };
        if (slot_drafts) m.slot_pass = try dspark_slots.SlotPass.init(gpa, &m.f, m.pass.?, m.rows.?, m.slots, group);
        // the prompt's prefilled tail's taps (CED's decoder replay, or the last segment) to the pass the lanes drive
        m.gt.pass = m.draftPass();
        // full mode: a restored entry's drafter rings start the pass's context where a restore does
        if (m.sessions) |x| x.ds_set = .{ .ctx = m, .set = setDsValid, .get = getDsValid };
        m.gt.ingest_window = m.cfg.window;
        tr.step("drafts and rows", cuda.usage(false));
        // TF_DSV41_SAMPLING=1: keyed sampling (T > 0) on every rank; off: such requests are refused (error.Sampling)
        m.sampler = if (samp.enabled()) try samp.GpuSampler.init(gpa, &m.f) else null;
        m.gt.sampler = m.sampler;
        try m.driver.synchronize();
        tr.step("end", cuda.usage(false));
        tr.trim();
        if (m.dev_arena) |x| x.endBoot(m.tcfg.rank); // TF_DSV41_ARENA=pack: boot's buffers in, the rest as before
        // the boot check on the boot's own figures: what it took, the dip below idle, the hard floor
        if (m.boot_check) |b| try kvs.afterBoot(b);
        try comm.barrier();
    }

    /// Several live slots: the stacked roles' bases, the row runner, the followers' row ops; one-slot graphs off.
    fn openRows(m: *Model, cap: u32, bp: *const buffers.Plan) !void {
        std.log.scoped(.dsv41).info("slots: {d} live, row windows of up to {d} rows, sessions {s}", .{ m.slots, cap, if (m.sessions != null) "on (every slot)" else "off" });
        const ss = try slots_mod.SlotSet.init(m.gpa, &m.f, m.slots);
        errdefer ss.deinit();
        try ss.bindBases(bp);
        m.slot_set = ss;
        m.rows = try batch.Batch.init(m.gpa, &m.f, ss, cap);
        m.f.ext = m.rows.?.ext();
        m.f.graphs_read = true;
        m.gt.rows = m.rows;
        m.gt.engram_warm = try target_mod.GpuTarget.engramWarmEnv();
    }

    fn getDsValid(ctx: *anyopaque, slot: u32) u64 {
        const m: *Model = @ptrCast(@alignCast(ctx));
        if (m.slot_pass) |x| return if (slot < x.valid.len) x.valid[slot] else 0;
        if (m.pass) |p| if (slot == 0) return p.valid;
        return 0;
    }
    fn setDsValid(ctx: *anyopaque, slot: u32, valid: u64) void {
        const m: *Model = @ptrCast(@alignCast(ctx));
        if (m.slot_pass) |x| {
            if (slot < x.valid.len) x.valid[slot] = valid;
        } else if (m.pass) |p| if (slot == 0) {
            p.valid = valid;
        };
    }

    /// TF_DSV41_PREFILL_PIECES=1: prompts prefilled in pieces between decode rounds (default off; ranks must agree).
    pub fn piecesEnv() bool {
        const v = std.c.getenv("TF_DSV41_PREFILL_PIECES") orelse return false;
        return std.mem.eql(u8, std.mem.trim(u8, std.mem.span(v), " "), "1");
    }

    fn announceStop(ctx: *anyopaque) void {
        const s: *tp.Session = @ptrCast(@alignCast(ctx));
        s.announceStop();
    }

    pub fn close(m: *Model) void {
        m.closeChecked() catch |err| {
            if (!quiet) std.log.err("DeepSeek close incomplete ({t}); model, driver and allocation owner retained", .{err});
            retain(m);
        };
    }
    pub fn closeChecked(m: *Model) !void {
        if (!m.boot_complete) return error.BootOwnerIncomplete;
        try m.ctx.synchronize();
        if (m.sessions) |x| try x.drain();
        if (m.f.kv) |k| try k.drain();
        try m.f.drain();
        if (m.rows) |b| try b.drain();
        try m.weights.close();
        const gpa = m.gpa;
        // the profile first, while the driver and its events are alive
        if (m.runner.prof) |p| {
            m.runner.prof = null;
            p.close(m.io);
        }
        if (m.slot_pass) |x| x.deinit();
        if (m.rows) |b| b.deinit();
        if (m.slot_set) |ss| ss.deinit();
        if (m.pass) |p| p.deinit();
        if (m.sampler) |x| x.deinit();
        if (m.sessions) |x| x.deinitAfterDrain();
        if (m.f.kv) |k| k.deinitAfterDrain();
        m.f.deinitAfterDrain();
        if (m.l2pf) |p| {
            m.runner.l2pf = null;
            p.deinit();
            gpa.destroy(p);
        }
        m.runner.deinit();
        m.aot.deinit();
        m.pl.deinit();
        m.pack.deinit();
        m.session.close();
        m.kernels.deinit();
        m.stream.deinit();
        if (m.dev_arena) |x| x.deinit(gpa);
        m.ctx.deinit();
        m.driver.close();
        m.arena.deinit();
        detachRetained(m);
        gpa.destroy(m);
    }
};

var retained_models: ?*Model = null;
var retained_lock: std.atomic.Mutex = .unlocked;
fn retain(m: *Model) void {
    while (!retained_lock.tryLock()) std.Thread.yield() catch {};
    defer retained_lock.unlock();
    if (m.retained) return;
    if (m.dev_arena) |x| x.detachHook();
    m.retained = true;
    m.retained_next = retained_models;
    retained_models = m;
}

fn detachRetained(m: *Model) void {
    while (!retained_lock.tryLock()) std.Thread.yield() catch {};
    defer retained_lock.unlock();
    var at = &retained_models;
    while (at.*) |x| {
        if (x == m) {
            at.* = x.retained_next;
            m.retained = false;
            return;
        }
        at = &x.retained_next;
    }
}
test "Draft state: failed model drain retains its context and allocation owner" {
    const mock = @import("load_mock.zig");
    const a = std.testing.allocator;
    const m = try a.create(Model);
    defer {
        detachRetained(m);
        a.destroy(m);
    }
    m.driver = mock.init();
    m.gpa = a;
    m.retained = false;
    m.retained_next = null;
    m.boot_complete = true;
    m.dev_arena = null;
    m.ctx = .{ .d = &m.driver, .handle = null, .device = 0 };
    mock.fail_sync = true;
    defer mock.fail_sync = false;
    m.close();
    try std.testing.expect(m.retained);
    try std.testing.expect(retained_models == m);
    try std.testing.expectEqual(@as(usize, 0), mock.freed);
}
