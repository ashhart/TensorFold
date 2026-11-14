//! Family startup kept separate from the request lifecycle.
const std = @import("std");
const cuda = @import("cuda");
const prompt = @import("cuda_prompt.zig");
const mtp = @import("cuda_mtp.zig");
const ranks = @import("cuda_rank_engine.zig");
const config = @import("cuda_config.zig");
const Glue = @import("cuda_glue.zig").Glue;
const images = @import("cuda_images.zig");
const memory = @import("exl3/pack/exl3_startup_memory.zig");
const exl3_pass = @import("exl3/model/cuda_exl3_pass.zig");
pub fn open(comptime F: type, gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, captured: ?[]const u8, o: F.Options) !F.Loaded {
    try @import("env_validate.zig").startup();
    if (o.vision) |v| try @import("qwen_vision").media.checkLimits(v);
    var load_cache = @import("core").checkpoint.cache.Load.begin(io, o.tp > 1);
    defer load_cache.deinit();
    const Cuda = F.Cuda;
    const round_streams = 32;

    if (o.context > prompt.max_tokens) return error.ContextOverWindow;
    if (o.segments != 1) return error.UnsupportedOption;
    if (o.tp != 1 and o.tp != 2) return error.FlashNextRanksOverTwo;
    if (o.rank >= o.tp) return error.BadRank;
    // refused before the stream, the glue or a captured set loads: the largest pass, wider at --parallel 1 on one rank
    const chunk = try @import("cuda_chunk.zig").soloRows(try prompt.chunkRows(), o.one_stream and o.tp == 1);
    _ = try @import("cuda_modelopt_threads.zig").configured(); // TF_FLASHNEXT_LOAD_THREADS, refused before any load
    const mode = try config.mode(gpa, io, dir);
    if (o.tp != 1 and mode != .modelopt and mode != .exl3) return error.FlashNextRanksNeedModelopt;
    if (o.tp != 1 and captured != null) return error.FlashNextRanksOnCapturedGlue;
    // the decode graph captures each layer's collectives; the dense split adds its own until a later stage qualifies it
    if (@import("cuda_round_graph.zig").enabled() and try @import("cuda_dense_split.zig").appliesDir(gpa, io, dir, o.tp)) return error.DecodeGraphWithDenseSplit;
    // sm_120 only: refused before any weight loads
    if (mode == .exl3) try @import("exl3/model/cuda_exl3_arch.zig").check(ctx.d);
    try @import("exl3/kv/cuda_exl3_kv.zig").refuseSolo(mode == .exl3, (try @import("sys").environment.read(?usize, "TF_FLASHNEXT_CHUNK_SOLO", null)) != null);
    if (o.slide and (mode != .affine or o.tp != 1)) return error.UnsupportedLearningFormat;
    if (o.slide and captured != null) return error.TrainingOnCapturedGlue;
    const page_tokens = try @import("cuda_kv_lanes.zig").paging();
    // the pool's pages are not wired to EXL3's K/V
    if (page_tokens > 0 and (o.tp != 1 or o.slide or captured != null or mode == .exl3)) return error.FlashNextPagesUnsupported;
    // pending DeltaNet rows run on one GPU only: TF_FLASHNEXT_KEEP_LATE=1 with tensor parallel is refused
    const Draft = @import("cuda_mtp_draft.zig").Draft;
    const late_fits = o.tp == 1;
    if (Draft.lateWanted() == true and !late_fits) return error.FlashNextKeepLateUnsupported;
    Draft.late_on = (Draft.lateWanted() orelse true) and late_fits;
    const driver: *cuda.Driver = @constCast(ctx.d);
    const own = try gpa.create(Cuda);
    errdefer gpa.destroy(own);
    own.* = .{ .gpa = gpa, .context = o.context, .engine = undefined, .glue = undefined, .stream = undefined, .logits = undefined };
    // fixed-size lane buffers reused across streams (cuda_lane_parts.zig; TF_FLASHNEXT_LANE_PARTS=0: allocate each)
    own.parts.on = try @import("sys").environment.read(bool, "TF_FLASHNEXT_LANE_PARTS", true);
    own.stream = try cuda.Stream.init(driver, true);
    errdefer own.stream.deinit();
    own.glue = try Glue.load(driver);
    errdefer own.glue.deinit();
    // a captured set replaces our glue, for comparison, when it holds every entry point this engine launches
    if (captured) |kernels| own.set = try F.comparison(gpa, io, driver, ctx, kernels, o.drafts);
    errdefer if (own.set) |*x| x.deinit();
    if (o.tp == 2) {
        var id = try ranks.identity(gpa, io, dir, o.context, chunk);
        id.mtp = @intFromBool(o.drafts);
        if (o.vision) |v| {
            try @import("qwen_vision").media.checkLimits(v);
            id.image_tokens = v.image_tokens;
            id.max_images = v.max_images;
        }
        own.ranks = try ranks.Session.init(gpa, io, o.master orelse "127.0.0.1", o.master_port, o.rank, id, ranks.linkWait());
        // EXL3 polls briefly before sleeping between decode collectives.
        if (mode == .exl3) own.ranks.?.collective.spin_ns = 3 * std.time.ns_per_ms;
    }
    errdefer if (own.ranks) |r| {
        r.poison();
        r.deinit();
    };
    own.engine = try prompt.Engine.initRank(gpa, io, driver, &own.stream, dir, o.tp, o.rank);
    errdefer own.engine.deinit();
    own.engine.chunk = chunk;
    if (own.ranks) |r| {
        r.engine = own.engine;
        own.engine.collective = &r.collective;
        r.benchmark_group = try @import("core").benchmark_group.cuda(gpa, io, ctx, &r.collective, own.stream);
    }
    const e = own.engine;
    if (e.chunk != e.busy_chunk) std.log.info("flashnext: prompt passes of {d} rows, {d} while another stream decodes", .{ e.chunk, e.busy_chunk });
    if (o.vision) |limits| {
        if (own.set != null) return error.VisionCapturedSetUnsupported;
        own.vision = try images.Vision.load(gpa, io, driver, own.stream, dir, limits, o.rank, own.glue.f.prep_media, own.glue.f.pool_media, own.glue.f.prep_media_paged, own.glue.f.pool_media_paged);
        own.engine.image_factory = own.vision;
    }
    errdefer if (own.vision) |v| v.deinit();
    var persistent_bytes = cuda.usage(false).device;
    // The scratch format and the chunk of an EXL3 pack.
    try @import("exl3/kv/cuda_exl3_kv.zig").setup(own.engine, mode == .exl3);
    // the pass's scratch for a full chunk, once for the engine (every prompt, window, round and head pass)
    _ = try own.engine.reserve(@max(@min(o.context, own.engine.chunk), if (o.slide) @as(usize, @import("cuda_train.zig").pack_rows) else 0));
    // the MTP head unless --no-drafts (TF_FLASHNEXT_DRAFTS and _CONFIDENCE: the chain's rule, 6 and -0.4)
    if (o.drafts and (o.tp == 1 or o.rank == 0)) {
        const head_before = cuda.usage(false).device;
        own.head = try mtp.Head.init(gpa, io, driver, &own.stream, dir);
        persistent_bytes = try std.math.add(u64, persistent_bytes, cuda.usage(false).device -| head_before);
        const drafts = try @import("sys").environment.read(usize, "TF_FLASHNEXT_DRAFTS", mtp.default_depth);
        own.depth = @max(1, @min(mtp.max_chain, drafts));
        const greedy = try @import("sys").environment.read(usize, "TF_FLASHNEXT_DRAFTS_GREEDY", mtp.default_depth_greedy);
        own.depth_greedy = @max(1, @min(mtp.max_chain, greedy));
        own.confidence = try @import("sys").environment.read(f64, "TF_FLASHNEXT_CONFIDENCE", mtp.default_confidence);
    }
    errdefer if (own.head) |h| h.deinit();
    @import("cuda_prompt_wt.zig").dropPageCache(io, dir, e.ngram.files.items);
    const window = @min(o.context + mtp.max_rows, prompt.max_tokens);
    // and an attention block's block scores at the window (64 MiB at 262,144 tokens)
    _ = try own.engine.reserveScores(window, @min(o.context, own.engine.chunk));
    // a shared round's or a shared verify's scores: up to a drafting round's rows (facts' batch_rows)
    own.logits = try cuda.DeviceBuffer.alloc(driver, @max(round_streams, mtp.round_rows) * prompt.vocabulary * 2);
    errdefer own.logits.free();
    own.lanes = .empty;
    own.next = 0;
    if (o.slide) own.lay = try @import("cuda_slide.zig").layoutOf(gpa, io, dir);
    const scratch_bytes = cuda.usage(false).device -| persistent_bytes;
    const deep_before = cuda.usage(false).device;
    // EXL3 offload: deep prefill rows before the pools
    try @import("exl3/prefill/cuda_exl3_prefill.zig").reserveFor(own.engine, o.context, o.tp);
    const deep_bytes = cuda.usage(false).device -| deep_before;
    const own_seq_bytes = try prompt.seqBytesPaged(window, try @import("cuda_pages.zig").pageTest(), own.engine.heads);
    var stream_bytes = own_seq_bytes;
    // an EXL3 rank 1 keeps no head cache
    if (own.head) |h| {
        stream_bytes = try std.math.add(u64, stream_bytes, mtp.Draft.bytes(window, h.ids.len, try @import("exl3/prefill/cuda_exl3_prefill.zig").draftRows(own.engine), own.engine.heads));
    } else if (o.drafts and o.tp == 2) stream_bytes = try std.math.add(u64, stream_bytes, mtp.Draft.bytes(if (mode == .exl3) 1 else window, 1, try @import("exl3/prefill/cuda_exl3_prefill.zig").draftRows(own.engine), own.engine.heads));
    if (o.vision) |v| stream_bytes = try std.math.add(u64, stream_bytes, try std.math.add(u64, try std.math.mul(u64, v.image_tokens, @import("cuda_prompt_wt.zig").dims * 2), try std.math.mul(u64, 16, o.context)));
    if (page_tokens > 0) {
        // the lane's head runs at the engine's chunk: its cache on the stream's pages, its buffers in the run
        own.kv.head = if (own.head) |h| .{ .draft_n = h.ids.len, .rows = own.engine.chunk } else null;
        const head_bytes = if (own.head) |h| mtp.Draft.bytes(window, h.ids.len, own.engine.chunk, own.engine.heads) else 0;
        own.kv.layout = @import("cuda_kv_lanes.zig").layout(o.context, page_tokens, own.kv.head, stream_bytes - own_seq_bytes - head_bytes);
    }
    // What a stream holds, by piece.
    try @import("exl3/kv/cuda_exl3_kv.zig").logStream(own.engine, window, stream_bytes);
    own.exl3 = mode == .exl3;
    own.stream_bytes = stream_bytes;
    const pool_budget = try exl3_pass.budgetForStreams(stream_bytes, o.lanes);
    const placement = @import("exl3/offload/cuda_exl3_frequency_state.zig");
    if (try placement.prepare(gpa, io, dir, own.engine.layers[0..own.engine.n_layers], own.head, @intCast(o.tp), @intCast(o.rank))) |prepared| {
        own.profile = prepared.state;
        own.profile_counts = prepared.counts;
    }
    errdefer if (own.profile) |*profile| profile.deinit();
    const profile_counts = if (own.profile_counts) |*counts| &counts.decoder else null;
    const attached = try exl3_pass.attach(gpa, io, driver, own.stream, own.engine.layers[0..own.engine.n_layers], if (own.head) |h| &h.layer else null, own.engine.chunk, pool_budget, profile_counts);
    if (own.profile_counts) |*counts| try placement.finish(own, counts);
    const ledger: memory.Ledger = .{
        .persistent = persistent_bytes,
        .scratch = scratch_bytes,
        .deep_prefill = try std.math.add(u64, deep_bytes, attached.prefill_window_bytes),
        .pool_scratch = attached.scratch_bytes,
        .expert_pools = attached.device_bytes,
        .reservation = .{ .stream_bytes = stream_bytes, .streams = if (stream_bytes == 0) 0 else pool_budget.other / stream_bytes },
    };
    std.log.info("flashnext rank {d} memory ledger: persistent={d} scratch={d} deep_prefill={d} pool_scratch={d} expert_pools={d} stream_bytes={d} streams={d} planned_total={d} free_after_pools={d}", .{ o.rank, ledger.persistent, ledger.scratch, ledger.deep_prefill, ledger.pool_scratch, ledger.expert_pools, stream_bytes, ledger.reservation.streams, try ledger.total(), attached.free_after });
    if (own.ranks) |r| try load_cache.finishAfter(.{ r, std.math.maxInt(u64) }, ranks.Session.room) else load_cache.finish();
    // the weights are in: the n-gram table into the page cache behind them, so the first prompt finds its rows in RAM
    own.engine.ngram.warm();
    return .{
        .layout = own.kv.layout,
        .vision = own.vision,
        .backend = own.backend(),
        .facts = own.facts(),
        .rows = if (own.head != null) mtp.max_rows else 2,
        .prompt_rows = @intCast(own.engine.chunk),
        .stream_bytes = stream_bytes,
        .host_snaps = try @import("exl3/kv/cuda_exl3_kv.zig").snapHost(o.context, mode == .exl3),
        .ctx = own,
        .deinit = F.release,
        .learns = if (o.slide) own else null,
        .learn_bytes = if (o.slide) @import("cuda_train.zig").memory() else 0,
        .target = own,
        .own_glue = own.set == null,
        .follow = if (o.rank != 0) F.follow else null,
    };
}

/// A captured set must provide every launch before it can replace the owned glue.
pub fn comparison(gpa: std.mem.Allocator, io: std.Io, driver: *cuda.Driver, ctx: *const cuda.Context, dir: []const u8, drafts: bool, entries: []const []const u8, head_entries: []const []const u8) !?cuda.aot.Set {
    var set = try cuda.aot.Set.load(gpa, io, driver, ctx.device, dir);
    const holds = for (entries) |name| {
        if (!set.has(name)) break false;
    } else for (head_entries) |name| {
        if (drafts and !set.has(name)) break false;
    } else true;
    if (holds) return set;
    set.deinit();
    std.log.info("flashnext: the captured set at {s} lacks Flash Next's kernels; its own glue runs", .{dir});
    return null;
}
