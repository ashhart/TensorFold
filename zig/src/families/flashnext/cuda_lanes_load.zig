//! Family startup kept separate from the request lifecycle.
const std = @import("std");
const cuda = @import("cuda");
const prompt = @import("cuda_prompt.zig");
const mtp = @import("cuda_mtp.zig");
const ranks = @import("cuda_rank_engine.zig");
const config = @import("cuda_config.zig");
const Glue = @import("cuda_glue.zig").Glue;
const images = @import("cuda_images.zig");
const memory = @import("exl3_startup_memory.zig");
const exl3_pass = @import("cuda_exl3_pass.zig");
pub fn open(comptime F: type, gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, captured: ?[]const u8, o: F.Options) !F.Loaded {
    if (o.vision) |v| try @import("qwen_vision").media.checkLimits(v);
    const Cuda = F.Cuda;
    const round_streams = 32;

    if (o.context > prompt.max_tokens) return error.ContextOverWindow;
    if (o.segments != 1) return error.UnsupportedOption;
    if ((o.tp != 1 and o.tp != 2) or o.rank >= o.tp) return error.BadRank;
    const mode = try config.mode(gpa, io, dir);
    if (o.tp != 1 and ((mode != .modelopt and mode != .exl3) or captured != null)) return error.UnsupportedQuantization;
    if (o.slide and (mode != .affine or o.tp != 1)) return error.UnsupportedLearningFormat;
    if (o.slide and captured != null) return error.TrainingOnCapturedGlue;
    const driver: *cuda.Driver = @constCast(ctx.d);
    const own = try gpa.create(Cuda);
    errdefer gpa.destroy(own);
    own.* = .{ .gpa = gpa, .context = o.context, .engine = undefined, .glue = undefined, .stream = undefined, .logits = undefined };
    own.stream = try cuda.Stream.init(driver, true);
    errdefer own.stream.deinit();
    own.glue = try Glue.load(driver);
    errdefer own.glue.deinit();
    // a captured set replaces our glue, for comparison, when it holds every entry point this engine launches
    if (captured) |kernels| own.set = try F.comparison(gpa, io, driver, ctx, kernels, o.drafts);
    errdefer if (own.set) |*x| x.deinit();
    if (o.tp == 2) {
        const configured_chunk = F.envUsize("TF_FLASHNEXT_CHUNK", prompt.batch.default_chunk);
        var id = try ranks.identity(gpa, io, dir, o.context, if (configured_chunk == 0) prompt.batch.default_chunk else configured_chunk);
        id.mtp = @intFromBool(o.drafts);
        if (o.vision) |v| {
            try @import("qwen_vision").media.checkLimits(v);
            id.image_tokens = v.image_tokens;
            id.max_images = v.max_images;
        }
        own.ranks = try ranks.Session.init(gpa, io, o.master orelse "127.0.0.1", o.master_port, o.rank, id);
    }
    errdefer if (own.ranks) |r| {
        r.poison();
        r.deinit();
    };
    own.engine = try prompt.Engine.initRank(gpa, io, driver, &own.stream, dir, o.tp, o.rank);
    if (own.ranks) |r| {
        r.engine = own.engine;
        own.engine.collective = &r.collective;
    }
    errdefer own.engine.deinit();
    if (o.vision) |limits| {
        if (own.set != null) return error.VisionCapturedSetUnsupported;
        own.vision = try images.Vision.load(gpa, io, driver, own.stream, dir, limits, o.rank, own.glue.f.prep_media, own.glue.f.pool_media);
        own.engine.image_factory = own.vision;
    }
    errdefer if (own.vision) |v| v.deinit();
    var persistent_bytes = cuda.usage(false).device;
    try @import("cuda_exl3_kv.zig").setup(own.engine, o.kv8, mode == .exl3); // K10: int8 KV, host index keys, chunk
    // the pass's scratch for a full chunk, once for the engine (every prompt, window, round and head pass)
    _ = try own.engine.reserve(@max(@min(o.context, own.engine.chunk), if (o.slide) @as(usize, @import("cuda_train.zig").pack_rows) else 0));
    // the MTP head unless --no-drafts (TF_FLASHNEXT_DRAFTS and _CONFIDENCE: the chain's rule, 6 and 0.60)
    if (o.drafts and (o.tp == 1 or o.rank == 0)) {
        const head_before = cuda.usage(false).device;
        own.head = try mtp.Head.init(gpa, io, driver, &own.stream, dir);
        persistent_bytes = try std.math.add(u64, persistent_bytes, cuda.usage(false).device -| head_before);
        own.depth = @max(1, @min(mtp.max_chain, F.envUsize("TF_FLASHNEXT_DRAFTS", mtp.default_depth)));
        if (std.c.getenv("TF_FLASHNEXT_CONFIDENCE")) |v| own.confidence = std.fmt.parseFloat(f64, std.mem.span(v)) catch mtp.default_confidence;
    }
    errdefer if (own.head) |h| h.deinit();
    const window = @min(o.context + mtp.max_rows, prompt.max_tokens);
    const common_kv = try own.engine.reserveSharedKv(window);
    persistent_bytes = try std.math.add(u64, persistent_bytes, common_kv);
    if (common_kv != 0) std.log.info("flashnext rank {d}: shared BF16 KV common allocation {d} bytes, charged once before lane and expert-pool admission", .{ o.rank, common_kv });
    // and an attention block's block scores at the window (64 MiB at 262,144 tokens)
    _ = try own.engine.reserveScores(window, @min(o.context, own.engine.chunk));
    own.logits = try cuda.DeviceBuffer.alloc(driver, round_streams * prompt.vocabulary * 2);
    errdefer own.logits.free();
    own.lanes = .empty;
    own.next = 0;
    if (o.slide) own.lay = try @import("cuda_slide.zig").layoutOf(gpa, io, dir);
    const scratch_bytes = cuda.usage(false).device -| persistent_bytes;
    const deep_before = cuda.usage(false).device;
    @import("cuda_exl3_prefill.zig").reserveFor(own.engine, o.context, o.tp); // EXL3 offload: deep prefill rows before the pools
    const deep_bytes = cuda.usage(false).device -| deep_before;
    var stream_bytes = try @import("cuda_exl3_kv.zig").engineSeqBytes(@import("cuda_exl3_kv.zig").of(own.engine), window, own.engine.chunk);
    if (own.head) |h| {
        stream_bytes = try std.math.add(u64, stream_bytes, mtp.Draft.bytes(window, h.ids.len, @import("cuda_exl3_prefill.zig").draftRows(own.engine)));
    } else if (o.drafts and o.tp == 2) stream_bytes = try std.math.add(u64, stream_bytes, mtp.Draft.bytes(1, 1, own.engine.chunk)); // rank 1's window (no head cache)
    if (o.vision) |v| stream_bytes = try std.math.add(u64, stream_bytes, try std.math.add(u64, try std.math.mul(u64, v.image_tokens, @import("cuda_prompt_wt.zig").dims * 2), try std.math.mul(u64, 16, o.context)));
    try @import("cuda_exl3_kv.zig").logStream(own.engine, window, stream_bytes); // K10: what a stream holds, by piece
    own.stream_bytes = stream_bytes;
    var pool_budget = try exl3_pass.budgetFromEnv(stream_bytes);
    if (mode == .exl3 and @import("cuda_exl3_graph.zig").enabled()) pool_budget.graph_reservation = try @import("cuda_exl3_graph_memory.zig").startupReservation();
    own.graph_reservation = pool_budget.graph_reservation;
    if (own.graph_reservation != 0) std.log.info("flashnext rank {d}: lane admission protects full graph reservation {d} bytes; already materialized graphs remain charged in free memory, so admission conservatively reserves those bytes twice", .{ o.rank, own.graph_reservation });
    const attached = try exl3_pass.attach(gpa, io, driver, own.stream, own.engine.layers[0..own.engine.n_layers], if (own.head) |h| &h.layer else null, own.engine.chunk, pool_budget);
    const ledger: memory.Ledger = .{
        .persistent = persistent_bytes,
        .scratch = scratch_bytes,
        .deep_prefill = deep_bytes,
        .pool_scratch = attached.scratch_bytes,
        .expert_pools = attached.device_bytes,
        .reservation = .{ .stream_bytes = stream_bytes, .streams = if (stream_bytes == 0) 0 else pool_budget.other / stream_bytes, .graph_bytes = pool_budget.graph_reservation, .after_pool_bytes = attached.after_pool_bytes },
    };
    std.log.info("flashnext rank {d} memory ledger: persistent={d} scratch={d} deep_prefill={d} pool_scratch={d} expert_pools={d} stream_bytes={d} streams={d} graph_reservation={d} planned_total={d} free_after_pools={d}", .{ o.rank, ledger.persistent, ledger.scratch, ledger.deep_prefill, ledger.pool_scratch, ledger.expert_pools, stream_bytes, ledger.reservation.streams, pool_budget.graph_reservation, try ledger.total(), attached.free_after });
    return .{
        .vision = own.vision,
        .backend = own.backend(),
        .facts = own.facts(),
        .rows = if (own.head != null) mtp.max_rows else 2,
        .prompt_rows = @intCast(own.engine.chunk),
        .stream_bytes = stream_bytes,
        .host_snaps = @import("cuda_exl3_kv.zig").snapHost(o.context, mode == .exl3),
        .ctx = own,
        .deinit = F.release,
        .learns = if (o.slide) own else null,
        .learn_bytes = if (o.slide) @import("cuda_train.zig").memory() else 0,
        .target = own,
        .own_glue = own.set == null,
        .follow = if (o.rank != 0) F.follow else null,
    };
}
