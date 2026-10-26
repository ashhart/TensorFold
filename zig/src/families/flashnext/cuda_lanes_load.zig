//! Family startup kept separate from the request lifecycle.
const std = @import("std");
const cuda = @import("cuda");
const prompt = @import("cuda_prompt.zig");
const mtp = @import("cuda_mtp.zig");
const ranks = @import("cuda_rank_engine.zig");
const config = @import("cuda_config.zig");
const Glue = @import("cuda_glue.zig").Glue;
const images = @import("cuda_images.zig");
pub fn open(comptime F: type, gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, captured: ?[]const u8, o: F.Options) !F.Loaded {
    if (o.vision) |v| try @import("qwen_vision").media.checkLimits(v);
    const Cuda = F.Cuda;
    const round_streams = 32;

    if (o.context > prompt.max_tokens) return error.ContextOverWindow;
    if (o.segments != 1) return error.UnsupportedOption;
    if ((o.tp != 1 and o.tp != 2) or o.rank >= o.tp) return error.BadRank;
    // refused before the stream, the glue or a captured set loads
    const chunk = try prompt.chunkRows();
    _ = try @import("cuda_modelopt_threads.zig").configured(); // TF_FLASHNEXT_LOAD_THREADS, refused before any load
    const mode = try config.mode(gpa, io, dir);
    if (o.tp != 1 and (mode != .modelopt or captured != null)) return error.UnsupportedQuantization;
    if (o.slide and (mode != .affine or o.tp != 1)) return error.UnsupportedLearningFormat;
    if (o.slide and captured != null) return error.TrainingOnCapturedGlue;
    const page_tokens = try @import("cuda_kv_lanes.zig").paging();
    if (page_tokens > 0 and (o.tp != 1 or o.slide or captured != null)) return error.FlashNextPagesUnsupported;
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
        var id = try ranks.identity(gpa, io, dir, o.context, chunk);
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
        own.vision = try images.Vision.load(gpa, io, driver, own.stream, dir, limits, o.rank, own.glue.f.prep_media, own.glue.f.pool_media, own.glue.f.prep_media_paged, own.glue.f.pool_media_paged);
        own.engine.image_factory = own.vision;
    }
    errdefer if (own.vision) |v| v.deinit();
    // the pass's scratch for a full chunk, once for the engine (every prompt, window, round and head pass)
    _ = try own.engine.reserve(@max(@min(o.context, own.engine.chunk), if (o.slide) @as(usize, @import("cuda_train.zig").pack_rows) else 0));
    // the MTP head unless --no-drafts (TF_FLASHNEXT_DRAFTS and _CONFIDENCE: the chain's rule, 6 and 0.60)
    if (o.drafts and (o.tp == 1 or o.rank == 0)) {
        own.head = try mtp.Head.init(gpa, io, driver, &own.stream, dir);
        own.depth = @max(1, @min(mtp.max_chain, F.envUsize("TF_FLASHNEXT_DRAFTS", mtp.default_depth)));
        if (std.c.getenv("TF_FLASHNEXT_CONFIDENCE")) |v| own.confidence = std.fmt.parseFloat(f64, std.mem.span(v)) catch mtp.default_confidence;
    }
    errdefer if (own.head) |h| h.deinit();
    const window = @min(o.context + mtp.max_rows, prompt.max_tokens);
    // and an attention block's block scores at the window (64 MiB at 262,144 tokens)
    _ = try own.engine.reserveScores(window, @min(o.context, own.engine.chunk));
    // a shared round's or a shared verify's scores: up to a drafting round's rows (facts' batch_rows)
    own.logits = try cuda.DeviceBuffer.alloc(driver, @max(round_streams, mtp.round_rows) * prompt.vocabulary * 2);
    errdefer own.logits.free();
    own.lanes = .empty;
    own.next = 0;
    if (o.slide) own.lay = try @import("cuda_slide.zig").layoutOf(gpa, io, dir);
    const own_seq_bytes = try prompt.seqBytesPaged(window, try @import("cuda_pages.zig").pageTest());
    var stream_bytes = own_seq_bytes;
    if (own.head) |h| {
        stream_bytes += mtp.Draft.bytes(window, h.ids.len, own.engine.chunk);
    } else if (o.drafts and o.tp == 2) stream_bytes += mtp.Draft.bytes(window, 1, own.engine.chunk);
    if (o.vision) |v| stream_bytes += @as(usize, v.image_tokens) * @import("cuda_prompt_wt.zig").dims * 2 + 16 * o.context;
    if (page_tokens > 0) {
        // the lane's head runs at the engine's chunk: its cache on the stream's pages, its buffers in the run
        own.kv.head = if (own.head) |h| .{ .draft_n = h.ids.len, .rows = own.engine.chunk } else null;
        const head_bytes = if (own.head) |h| mtp.Draft.bytes(window, h.ids.len, own.engine.chunk) else 0;
        own.kv.layout = @import("cuda_kv_lanes.zig").layout(o.context, page_tokens, own.kv.head, stream_bytes - own_seq_bytes - head_bytes);
    }
    return .{
        .layout = own.kv.layout,
        .vision = own.vision,
        .backend = own.backend(),
        .facts = own.facts(),
        .rows = if (own.head != null) mtp.max_rows else 2,
        .prompt_rows = @intCast(own.engine.chunk),
        .stream_bytes = stream_bytes,
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
