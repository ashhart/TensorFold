//! Unlimited-OCR-2 on the lane host: one stream, pages prepared on the HTTP thread, tower and decode on the lane's.
const std = @import("std");
const mtl = @import("metal");
const lanes = @import("lanes");
const be = lanes.backend;
const picture = @import("../../core/picture/picture.zig");
const cfg = @import("config.zig");
const wts = @import("weights.zig");
const ks = @import("kernels.zig");
const vision = @import("vision.zig");
const preprocess = @import("preprocess.zig");
const pages = preprocess.With(picture);
const prompt_mod = @import("prompt.zig");
const decoder_mod = @import("decoder.zig");
const generate = @import("generate.zig");
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

pub const Model = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    device: mtl.Device,
    queue: mtl.Queue,
    weights: wts.Weights,
    kernels: ks.Kernels,
    encoder: vision.Encoder,
    decoder: decoder_mod.Decoder,
    /// The one stream the decoder holds, and its prompt's state.
    stream: ?*lanes.Stream = null,
    prompt: ?prompt_mod.Prompt = null,
    step: prompt_mod.Step = undefined,
    guard: generate.Guard = undefined,

    /// The release's weights from `dir`, with caches for `context` tokens.
    pub fn load(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, context: usize) !*Model {
        const device = try mtl.Device.init();
        if (!device.tensorUnits()) return error.NoTensorUnits;
        const m = try gpa.create(Model);
        errdefer gpa.destroy(m);
        m.* = .{ .gpa = gpa, .io = io, .device = device, .queue = try device.queue(), .weights = undefined, .kernels = undefined, .encoder = undefined, .decoder = undefined };
        errdefer m.queue.deinit();
        m.weights = try wts.load(gpa, io, device, dir);
        errdefer m.weights.deinit();
        m.kernels = try ks.Kernels.load(gpa, device);
        errdefer m.kernels.deinit();
        m.encoder = try vision.Encoder.init(gpa, device, m.queue, &m.kernels, &m.weights);
        errdefer m.encoder.deinit();
        m.decoder = try decoder_mod.Decoder.init(gpa, device, m.queue, &m.kernels, &m.weights, context, @min(prompt_rows, context));
        return m;
    }

    pub fn deinit(m: *Model) void {
        if (m.prompt) |*p| p.deinit(m.gpa);
        m.decoder.deinit();
        m.encoder.deinit();
        m.kernels.deinit();
        m.weights.deinit();
        m.queue.deinit();
        m.gpa.destroy(m);
    }

    /// One page's views as f32-widened bf16 (global first): gundam tiles when they fit `max_tokens`, else base.
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
            try m.encoder.encode(cfg.base_size, px[0..global_px], global, .{});
            const tile_rows = try a.alloc(u16, tiles * t * t * H);
            defer a.free(tile_rows);
            for (0..tiles) |i| try m.encoder.encode(cfg.tile_size, px[global_px + i * tile_px ..][0..tile_px], tile_rows[i * t * t * H ..][0 .. t * t * H], .{});
            const out = rows[at * H ..][0 .. @as(usize, img.tokens) * H];
            vision.layout(&m.weights, global, tile_rows, across, down, out);
            at += img.tokens;
        }
        return rows;
    }

    pub fn facts(_: *const Model) lanes.Model {
        // one stream, one token a round: no drafts, no copy windows
        return .{ .exact_width = 1, .copy_drafts = false, .lone_copies = false, .drafts = 1, .max_streams = 1, .mtp = false, .speculate = false, .speculate_early = false };
    }

    pub fn backend(m: *Model) be.Backend {
        return .{ .ptr = m, .vtable = &.{ .prefill = prefillFn, .first = firstFn, .queue = queueFn, .read = readFn, .verify = verifyFn, .keep = keepFn, .draft = draftFn, .release = releaseFn } };
    }

    fn cast(ptr: *anyopaque) *Model {
        return @ptrCast(@alignCast(ptr));
    }

    fn prefillFn(ptr: *anyopaque, stream: *lanes.Stream) !void {
        const m = cast(ptr);
        if (m.stream != null) return error.UocrBusy;
        const ids = stream.prompt();
        // the prompt's rows, then a row for every reply token fed back: all but the last, which ends the reply
        if (ids.len == 0 or ids.len + @max(stream.max_new, 1) - 1 > m.decoder.cap) return error.UocrPromptTooLong;
        var arena: std.heap.ArenaAllocator = .init(m.gpa);
        defer arena.deinit();
        const rows = try m.encodeImages(arena.allocator(), stream.images);
        var p = try prompt_mod.positions(m.gpa, try m.gpa.dupe(u32, ids));
        errdefer p.deinit(m.gpa);
        try m.decoder.prefill(p, rows);
        m.guard = generate.Guard.init(p);
        m.step = prompt_mod.Step.init(p);
        m.prompt = p;
        m.stream = stream;
        stream.cached = 0;
    }

    fn firstFn(ptr: *anyopaque, stream: *lanes.Stream, _: u64) !u64 {
        const m = cast(ptr);
        if (m.stream != stream) return error.UocrUnknownStream;
        return m.guard.pick(try m.decoder.logits());
    }

    /// The fed token through one decode step; the next token's greedy pick under the page guard is the handle.
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
        return m.guard.pick(try m.decoder.logits());
    }

    fn readFn(_: *anyopaque, handle: u64) !u32 {
        return @intCast(handle);
    }

    /// A one-row window (no drafts): its pending token through a decode step, as `queue` runs it.
    fn verifyFn(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) !void {
        const m = cast(ptr);
        if (windows.len != out.len) return error.UocrBadWindows;
        for (windows, out) |w, o| {
            if (w.rows() != 1 or o.sampled.len != 1 or o.drafts.len != 0) return error.UocrHasNoDrafts;
            if (m.stream != w.stream) return error.UocrUnknownStream;
            o.sampled[0] = try m.advance(w.pending);
        }
    }

    /// One-row windows keep their row: nothing to roll back.
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

test "uocr: a page's token bounds" {
    try std.testing.expectEqual(@as(u32, 273), basePageTokens());
    try std.testing.expect(maxPageTokens() > basePageTokens());
}
