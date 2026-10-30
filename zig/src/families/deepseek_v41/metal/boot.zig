//! A release checkpoint folder on this Mac: config, kernels, mapped weights, DSpark, Engram rows; the served model.
const std = @import("std");
const mtl = @import("metal");
const host = @import("dsv41");
const pipelines = @import("pipelines.zig");
const model_weights = @import("model_weights.zig");
const dspark = @import("dspark.zig");
const rope = @import("rope.zig");
const scratch = @import("scratch.zig");
const forward = @import("forward.zig");
const serve = @import("serve.zig");
const target_ = @import("target.zig");
const vision = @import("vision.zig");
const vw = @import("vision_weights.zig");

const Config = host.config.Config;
const eh = host.engram_host;
const er = host.engram_rows;
const Allocator = std.mem.Allocator;

/// The files a boot reads beside the shards: the Engram token map (here or under TF_DSV41_ASSETS).
pub const token_map = "engram-token-map.bin";

pub const Options = struct {
    slots: u32 = 4,
    /// positions a slot holds (null: 262,144, at most the checkpoint's)
    context: ?u64 = null,
    drafts: bool = true,
    cache_bytes: u64 = 4 << 30,
    /// a checkpoint of the release's format at other shapes (TF_DSV41_SMALL=1: the smoke test's tiny model)
    small: bool = false,
    /// Engram row shards (null: TF_DSV41_ENGRAM_DIR, else the config's engram_table_dir beside the checkpoint)
    engram_dir: ?[]const u8 = null,
    /// --vision: load the checkpoint's image tower (its tensors read beside the backbone)
    vision: bool = false,
    /// a request's image tokens in all, and its images (--vision-image-tokens, --vision-max-images)
    image_tokens: u32 = 4096,
    max_images: u32 = 4,
};

/// Everything a served model borrows, owned here and closed after the model's target and pass.
pub const Boot = struct {
    a: Allocator,
    c: Config,
    device: mtl.Device,
    queue: mtl.Queue,
    p: pipelines.Pipelines,
    w: *model_weights.Model,
    ds: ?dspark.Weights = null,
    tables: rope.Tables,
    s: scratch.Scratch,
    map_bytes: []align(4) u8 = &.{},
    lbuf: [eh.max_layers]u32 = undefined,
    eng: eh.Host = undefined,
    rows: ?*er.Rows = null,
    kit: ?vision.Kit = null,
    vw: ?vw.Weights = null,

    /// The checkpoint in `dir` loaded and served by a `serve.Metal` (the caller's `serve_lanes.Served` closes it).
    pub fn open(a: Allocator, io: std.Io, dir: []const u8, o: Options) !*serve.Metal {
        const b = try a.create(Boot);
        errdefer a.destroy(b);
        b.a = a;
        b.ds = null;
        b.map_bytes = &.{};
        b.rows = null;
        b.kit = null;
        b.vw = null;
        b.c = try readConfig(a, io, dir, o.small);
        const c = &b.c;
        const limit: u32 = @intCast(@min(o.context orelse 262144, c.rope.max_positions));
        b.device = try mtl.Device.init();
        errdefer b.device.deinit();
        b.queue = try b.device.queue();
        errdefer b.queue.deinit();
        b.p = try pipelines.Pipelines.load(a, b.device, c);
        errdefer b.p.deinit();
        var pack = try host.pack.Pack.open(a, io, dir);
        defer pack.deinit();
        b.w = try model_weights.Model.load(a, io, b.device, b.queue, &pack, c, null);
        errdefer b.w.deinit();
        if (o.drafts and c.mtp_layers > 0) b.ds = try dspark.loadWeights(a, io, b.device, &pack, c);
        errdefer if (b.ds) |*w| w.deinit();
        try b.loadVision(io, dir, &pack, o);
        errdefer b.closeVision();
        b.tables = try rope.Tables.init(b.device, c, limit);
        errdefer b.tables.deinit();
        b.s = try scratch.Scratch.init(b.device, c, limit);
        errdefer b.s.deinit();
        const lookup = try b.engram(io, dir, o);
        errdefer b.closeEngram();
        const r = b.w.report();
        std.log.info("dsv41 metal: {d} layers, {d} shards ({d} MiB mapped, {d} MiB copied), {d} slots of {d} positions, DSpark {s}", .{ c.layers, r.shards, r.shard_bytes >> 20, r.arena_bytes >> 20, o.slots, limit, if (b.ds != null) "on" else "off" });
        const env: target_.Env = .{ .device = b.device, .queue = b.queue, .c = c, .p = &b.p, .tables = &b.tables, .s = &b.s, .model = b.w.view(), .lookup = lookup, .vision = if (b.vw) |*w| .{ .kit = &b.kit.?, .w = w } else null };
        return serve.Metal.open(a, .{ .env = env, .top = &b.w.top, .dspark = if (b.ds) |*w| w else null, .owner = .{ .ctx = b, .close = close }, .images = if (b.vw != null) .{ .io = io, .tokens = o.image_tokens, .count = o.max_images } else null }, .{ .slots = o.slots, .limit = limit, .drafts = o.drafts, .cache_bytes = o.cache_bytes });
    }

    /// The tower's kernels and weights, and each layer's image routing bias attached to the backbone's layers.
    fn loadVision(b: *Boot, io: std.Io, dir: []const u8, pack: *const host.pack.Pack, o: Options) !void {
        if (!o.vision) return;
        if (!vw.Weights.present(pack)) {
            std.log.err("dsv41 metal: --vision, but {s} has no vision tower (no vision.* tensors)", .{dir});
            return error.NoVisionTower;
        }
        const path = try std.fs.path.join(b.a, &.{ dir, "config.json" });
        defer b.a.free(path);
        const text = try std.Io.Dir.cwd().readFileAlloc(io, path, b.a, .limited(1 << 24));
        defer b.a.free(text);
        const cfg = (try vision.Config.parse(b.a, text, b.c.hidden)) orelse vision.Config{ .out_dim = b.c.hidden };
        b.kit = try vision.Kit.load(b.device);
        errdefer b.closeVision();
        b.vw = try vw.Weights.load(b.a, io, b.device, pack, cfg, b.c.layers, b.c.experts);
        for (b.w.layers, b.vw.?.bias_vl) |*l, r| l.gate_bvl = r;
        std.log.info("dsv41 metal: vision tower on ({d} blocks of {d}, patches of {d}, up to {d} tokens an image)", .{ cfg.layers, cfg.dim, cfg.patch, cfg.max_tokens });
    }

    fn closeVision(b: *Boot) void {
        if (b.vw) |*w| w.deinit();
        if (b.kit) |*k| k.deinit();
        b.vw = null;
        b.kit = null;
    }

    fn readConfig(a: Allocator, io: std.Io, dir: []const u8, small: bool) !Config {
        if (!small) return Config.read(a, io, dir);
        const path = try std.fs.path.join(a, &.{ dir, "config.json" });
        defer a.free(path);
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 24));
        defer a.free(bytes);
        return Config.parseShapes(a, bytes);
    }

    /// The Engram hashes (the canonical token map) and the packed row shards; null without Engram layers.
    fn engram(b: *Boot, io: std.Io, dir: []const u8, o: Options) !?forward.Lookup {
        const c = &b.c;
        if (c.engram_layers.len == 0) return null;
        const assets = if (std.c.getenv("TF_DSV41_ASSETS")) |v| std.mem.span(v) else dir;
        const path = try std.fs.path.join(b.a, &.{ assets, token_map });
        defer b.a.free(path);
        b.map_bytes = std.Io.Dir.cwd().readFileAllocOptions(io, path, b.a, .limited(1 << 26), .@"4", null) catch |e| {
            std.log.err("dsv41 metal: the Engram token map {s} ({t}): put it beside the checkpoint or under TF_DSV41_ASSETS", .{ path, e });
            return error.EngramTokenMap;
        };
        errdefer b.a.free(b.map_bytes);
        b.eng = try eh.Host.load(b.map_bytes, c.engram_heads, c.engram_pad, c.engram_vocab, &b.lbuf);
        const rows_dir = try b.rowsDir(dir, o);
        defer b.a.free(rows_dir);
        const cap: u32 = if (std.c.getenv("TF_DSV41_ENGRAM_CACHE")) |v| try std.fmt.parseInt(u32, std.mem.span(v), 10) else 65536;
        b.rows = try er.Rows.open(b.a, rows_dir, 0, 1, b.eng.layer_ids, cap);
        std.log.info("dsv41 metal: Engram rows from {s} ({d} records a layer cached)", .{ rows_dir, cap });
        return .{ .a = b.a, .host = &b.eng, .source = b.rows.?.source() };
    }

    /// The row shards' folder: the option, TF_DSV41_ENGRAM_DIR, else engram_table_dir (relative to the checkpoint).
    fn rowsDir(b: *Boot, dir: []const u8, o: Options) ![]u8 {
        if (o.engram_dir) |d| return b.a.dupe(u8, d);
        if (std.c.getenv("TF_DSV41_ENGRAM_DIR")) |v| return b.a.dupe(u8, std.mem.span(v));
        const t = b.c.engram_table_dir.items();
        if (t.len == 0) return error.EngramRowsUnset;
        if (std.fs.path.isAbsolute(t)) return b.a.dupe(u8, t);
        return std.fs.path.join(b.a, &.{ dir, t });
    }

    fn closeEngram(b: *Boot) void {
        if (b.rows) |r| r.close();
        b.rows = null;
        if (b.map_bytes.len > 0) b.a.free(b.map_bytes);
        b.map_bytes = &.{};
    }

    fn close(ctx: *anyopaque) void {
        const b: *Boot = @ptrCast(@alignCast(ctx));
        b.closeEngram();
        b.s.deinit();
        b.tables.deinit();
        if (b.ds) |*w| w.deinit();
        b.closeVision();
        b.w.deinit();
        b.p.deinit();
        b.queue.deinit();
        b.device.deinit();
        b.a.destroy(b);
    }
};
