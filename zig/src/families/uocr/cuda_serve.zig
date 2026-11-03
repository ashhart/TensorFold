//! Unlimited-OCR-2 on CUDA served by the lane host (serve.zig's model): one stream at a time, pages in as images,
//! the publisher's greedy reply under its page guard. A page is prepared on the HTTP thread (decode, letterbox,
//! tiles); the tower, the prompt pass and every decode step run on the lane thread with the context current.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const be = lanes.backend;
const picture = @import("core").picture;
const cfg = @import("config.zig");
const wts = @import("cuda_weights.zig");
const ks = @import("cuda_kernels.zig");
const vision = @import("cuda_vision.zig");
const preprocess = @import("preprocess.zig");
const pages = preprocess.With(@import("core").picture);
const prompt_mod = @import("prompt.zig");
const decoder_mod = @import("cuda_decoder.zig");
const guard_mod = @import("guard.zig");
const resize = @import("resize.zig");

const H = cfg.hidden;

/// Prompt rows a prefill pass takes at once (the decoder's scratch); larger prompts run in chunks of these.
pub const prompt_rows = 2048;

/// The most image tokens one page takes: the global view and the widest tile grid gundam mode picks.
pub fn maxPageTokens() u32 {
    const g = cfg.side(cfg.base_size);
    const t = cfg.side(cfg.tile_size);
    var most: usize = 0;
    for (1..preprocess.max_tiles + 1) |across| for (1..preprocess.max_tiles / across + 1) |down| {
        if (across * down < preprocess.min_tiles) continue;
        most = @max(most, (t * across + 1) * (t * down));
    };
    return @intCast(g * (g + 1) + 1 + most);
}

/// A page in base mode alone: the 1024 view's tokens.
pub fn basePageTokens() u32 {
    const g = cfg.side(cfg.base_size);
    return @intCast(g * (g + 1) + 1);
}

/// Device bytes beside the weights for `context` tokens: the caches, and the encoder's and decoder's scratch (an
/// estimate the open path checks against free memory).
pub fn cacheBytes(context: usize) usize {
    return decoder_mod.Decoder.cacheBytes(std.mem.alignForward(usize, context, cfg.block));
}

pub const Model = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    d: *const cuda.Driver,
    s: cuda.Stream,
    weights: *wts.Weights,
    kernels: ks.Kernels,
    vk: cuda.vit.Kernels,
    gemms: cuda.vit.Gemms,
    encoder: vision.Encoder,
    decoder: decoder_mod.Decoder,
    stream: ?*lanes.Stream = null,
    prompt: ?prompt_mod.Prompt = null,
    step: prompt_mod.Step = undefined,
    guard: guard_mod.Guard = undefined,

    /// The release's weights from `dir`, with caches for `context` tokens; the caller's context is current.
    pub fn load(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, dir: []const u8, context: usize) !*Model {
        const m = try gpa.create(Model);
        errdefer gpa.destroy(m);
        m.* = .{ .gpa = gpa, .io = io, .d = d, .s = undefined, .weights = undefined, .kernels = undefined, .vk = undefined, .gemms = undefined, .encoder = undefined, .decoder = undefined };
        m.s = try cuda.Stream.init(d, false);
        errdefer m.s.deinit();
        m.kernels = try ks.Kernels.load(d);
        errdefer m.kernels.unload();
        m.vk = try cuda.vit.Kernels.load(d);
        errdefer m.vk.unload();
        m.gemms = try cuda.vit.Gemms.init(gpa, d);
        errdefer m.gemms.deinit();
        m.weights = try wts.load(gpa, io, d, dir);
        errdefer m.weights.deinit();
        m.encoder = try vision.Encoder.init(gpa, d, m.s, &m.kernels, &m.vk, &m.gemms, m.weights);
        errdefer m.encoder.deinit();
        m.decoder = try decoder_mod.Decoder.init(gpa, d, m.s, &m.kernels, &m.gemms, m.weights, context, @min(prompt_rows, context));
        return m;
    }

    pub fn deinit(m: *Model) void {
        if (m.prompt) |*p| p.deinit(m.gpa);
        m.decoder.deinit();
        m.encoder.deinit();
        m.weights.deinit();
        m.gemms.deinit();
        m.vk.unload();
        m.kernels.unload();
        m.s.deinit();
        m.gpa.destroy(m);
    }

    /// One page for the tower (serve.zig's prepare): gundam tiles plus the global view when they fit `max_tokens`,
    /// else the global view alone. The views go as their bf16 values widened to f32, the global view first.
    pub fn prepare(m: *Model, a: std.mem.Allocator, bytes: []const u8, max_tokens: u32) !Prepared {
        if (max_tokens < basePageTokens()) return error.ImageTooSmall;
        var page = try pages.decode(m.io, a, bytes);
        defer page.deinit(a);
        var v = try pages.views(a, page.rgb, page.width, page.height, true);
        if (v.tokens() > max_tokens) {
            v.deinit(a);
            v = try pages.views(a, page.rgb, page.width, page.height, false);
        }
        defer v.deinit(a);
        const pixels = try a.alloc(f32, v.global.len + v.tiles.len);
        for (v.global, pixels[0..v.global.len]) |h, *f| f.* = resize.widen(h);
        for (v.tiles, pixels[v.global.len..]) |h, *f| f.* = resize.widen(h);
        const grid = [2]u32{ @intCast(v.down), @intCast(v.across) };
        return .{ .pixels = pixels, .gh = grid[0], .gw = grid[1], .tokens = @intCast(v.tokens()), .hash = picture.identity.withGrid(picture.identity.ofBytes(0x0c12, bytes), grid) };
    }

    pub const Prepared = struct { pixels: []f32, gh: u32, gw: u32, tokens: u32, hash: u64 };

    /// Every image of the stream through the tower, laid out as its image tokens in prompt order.
    fn encodeImages(m: *Model, a: std.mem.Allocator, images: []const lanes.stream.Image) ![]u16 {
        var total: usize = 0;
        for (images) |img| total += img.tokens;
        const rows = try a.alloc(u16, total * H);
        const g = cfg.side(cfg.base_size);
        const t = cfg.side(cfg.tile_size);
        const global_px = 3 * cfg.base_size * cfg.base_size;
        const tile_px = 3 * cfg.tile_size * cfg.tile_size;
        var at: usize = 0;
        for (images) |img| {
            const across: usize = img.gw;
            const down: usize = img.gh;
            const tiles: usize = if (across * down > 1) across * down else 0;
            if (img.pixels.len != global_px + tiles * tile_px) return error.ImageShape;
            const px = try a.alloc(u16, img.pixels.len);
            defer a.free(px);
            for (img.pixels, px) |f, *h| h.* = resize.bf16(f);
            const global = try a.alloc(u16, g * g * H);
            defer a.free(global);
            try m.encoder.encode(cfg.base_size, px[0..global_px], global);
            const tile_rows = try a.alloc(u16, tiles * t * t * H);
            defer a.free(tile_rows);
            for (0..tiles) |i| try m.encoder.encode(cfg.tile_size, px[global_px + i * tile_px ..][0..tile_px], tile_rows[i * t * t * H ..][0 .. t * t * H]);
            const out = rows[at * H ..][0 .. @as(usize, img.tokens) * H];
            vision.layout(m.weights, global, tile_rows, across, down, out);
            at += img.tokens;
        }
        return rows;
    }

    pub fn facts(_: *const Model) lanes.Model {
        return .{ .exact_width = 1, .copy_drafts = false, .lone_copies = false, .drafts = 1, .max_streams = 1, .mtp = false, .speculate = false, .speculate_early = false };
    }

    pub fn backend(m: *Model) be.Backend {
        return .{ .ptr = m, .vtable = &.{ .prefill = prefillFn, .first = firstFn, .queue = queueFn, .read = readFn, .verify = verifyFn, .keep = keepFn, .draft = draftFn, .release = releaseFn } };
    }

    fn cast(ptr: *anyopaque) *Model {
        return @ptrCast(@alignCast(ptr));
    }

    /// The guard's pick from the decoder's last forward: a forced token without the logits, else the GPU's argmax.
    fn pick(m: *Model) !u32 {
        if (m.guard.forcedToken()) |only| return only;
        return m.decoder.argmax(m.guard.barsEos());
    }

    fn prefillFn(ptr: *anyopaque, stream: *lanes.Stream) !void {
        const m = cast(ptr);
        if (m.stream != null) return error.UocrBusy;
        const ids = stream.prompt();
        if (ids.len == 0 or ids.len + @max(stream.max_new, 1) - 1 > m.decoder.cap) return error.UocrPromptTooLong;
        var arena: std.heap.ArenaAllocator = .init(m.gpa);
        defer arena.deinit();
        const rows = try m.encodeImages(arena.allocator(), stream.images);
        var p = try prompt_mod.positions(m.gpa, try m.gpa.dupe(u32, ids));
        errdefer p.deinit(m.gpa);
        try m.decoder.prefill(p, rows);
        m.guard = guard_mod.Guard.init(p);
        m.step = prompt_mod.Step.init(p);
        m.prompt = p;
        m.stream = stream;
        stream.cached = 0;
    }

    fn firstFn(ptr: *anyopaque, stream: *lanes.Stream, _: u64) !u64 {
        const m = cast(ptr);
        if (m.stream != stream) return error.UocrUnknownStream;
        return try m.pick();
    }

    fn queueFn(ptr: *anyopaque, stream: *lanes.Stream, f: be.Feed, _: u64) !u64 {
        const m = cast(ptr);
        if (m.stream != stream) return error.UocrUnknownStream;
        return try m.advance(switch (f) {
            .handle => |h| @intCast(h),
            .value => |v| v,
        });
    }

    fn advance(m: *Model, id: u32) !u32 {
        if (id >= cfg.vocab) return error.UocrBadToken;
        m.guard.take(id);
        const c = m.step.next(id);
        try m.decoder.step(id, c.tok, c.page, c.qpage);
        return m.pick();
    }

    fn readFn(_: *anyopaque, handle: u64) !u32 {
        return @intCast(handle);
    }

    fn verifyFn(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) !void {
        const m = cast(ptr);
        if (windows.len != out.len) return error.UocrBadWindows;
        for (windows, out) |w, o| {
            if (w.rows() != 1 or o.sampled.len != 1 or o.drafts.len != 0) return error.UocrHasNoDrafts;
            if (m.stream != w.stream) return error.UocrUnknownStream;
            o.sampled[0] = try m.advance(w.pending);
        }
    }

    fn keepFn(_: *anyopaque, windows: []const be.Window, paths: []const []const u32) !void {
        for (windows, paths) |w, path| if (w.rows() != 1 or path.len != 1 or path[0] != 0) return error.UocrHasNoDrafts;
    }

    fn draftFn(_: *anyopaque, requests: []const be.DraftRequest) !void {
        for (requests) |r| if (r.depth != 0) return error.UocrHasNoDraftHead;
    }

    fn releaseFn(ptr: *anyopaque, stream: *lanes.Stream) void {
        const m = cast(ptr);
        if (m.stream != stream) return;
        if (m.prompt) |*p| p.deinit(m.gpa);
        m.prompt = null;
        m.stream = null;
    }
};
