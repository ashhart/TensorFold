//! The native HIP family registry entry; the shared lane engine serves image-prefix OCR and verifies proposals.
const std = @import("std");
const hip = @import("hip");
const api = @import("engine_api");
const lanes = @import("lanes");
const serve = @import("hip_serve");
const config = @import("config.zig");
const engine = @import("engine.zig");
const Backend = @import("backend.zig").Backend;
const processor = @import("processor.zig");

pub const model_type = config.model_type;
pub const raw_text = false;
/// OCR output has no reasoning block: every reply token is answer content.
pub const reasoning = false;
pub const formats: []const []const u8 = &.{"fp32"};
pub const default_context: i64 = config.context;
pub const prefill_step: u32 = 512;
pub const slide_refusal = "--slide: Nemotron Parse has no native learner";
pub const confidence_refusal = "--mtp-confidence: Nemotron Parse uses its trained one-step proposal without a confidence cut";

/// Refuse unsupported combinations before GPU setup or a tensor-parallel join (hip_families.refused).
pub fn refusal(o: api.Open) ?[]const u8 {
    if (o.tp != 1 or o.rank != 0) return "Nemotron Parse currently serves one HIP rank (--tp 1 --rank 0)";
    if (o.vision == null) return "Nemotron Parse requires --vision and one image per request";
    if (o.context) |n| if (n > config.context) return "Nemotron Parse's visual prefix and text share a 16384-token context";
    if (o.mtp_drafts) |n| if (n > 1) return "Nemotron Parse has one trained proposal head; use --mtp-drafts 0 or 1";
    return null;
}

pub const Loaded = struct {
    backend: lanes.backend.Backend,
    facts: lanes.Model,
    rows: u32,
    ctx: *anyopaque,
    deinit: *const fn (*anyopaque) void,
    streams: usize,
    startup: []const u8,
    /// Its image tower, and its prompt cache when the budget and the slots allow one.
    offer: serve.Offer,
};

const Owned = struct {
    a: std.mem.Allocator,
    e: *engine.Engine,
    h: *Backend,
    startup: []const u8,
    image_tokens: u32,
};

fn release(ptr: *anyopaque) void {
    const own: *Owned = @ptrCast(@alignCast(ptr));
    own.h.deinit();
    own.e.deinit();
    own.a.free(own.startup);
    own.a.destroy(own);
}

fn prepare(ptr: *anyopaque, a: std.mem.Allocator, bytes: []const u8, limit: u32) anyerror!api.PreparedImage {
    const own: *Owned = @ptrCast(@alignCast(ptr));
    const p = try processor.prepare(own.e.io, a, bytes);
    errdefer a.free(p.pixels);
    const g = p.geometry;
    if (g.tokens() > @min(limit, own.image_tokens)) return error.ImageTooSmall;
    return .{ .pixels = p.pixels, .gh = @intCast(g.padded.h / 16), .gw = @intCast(g.padded.w / 16), .tokens = @intCast(g.tokens()), .content_size = .{ @intCast(g.content.h), @intCast(g.content.w) }, .hash = p.hash };
}

pub fn open(a: std.mem.Allocator, gpa: std.mem.Allocator, io: std.Io, dev: hip.Device, dir: []const u8, o: serve.Options, problem: *[]const u8) !Loaded {
    if (o.world != 1 or o.rank != 0 or o.vision == null) return error.Refused;
    const e = engine.Engine.open(gpa, io, dir, dev, o.window, o.streams, o.fixed) catch |err| {
        problem.* = try std.fmt.allocPrint(a, "cannot load native HIP Nemotron Parse ({t})", .{err});
        return error.Refused;
    };
    errdefer e.deinit();
    const h = try Backend.init(gpa, e, dev.policy.mtp.drafts != 0);
    errdefer h.deinit();
    const pool = try hip.admission.budget(gpa, io, e.ctx);
    const planned = e.streams * (engine.stateBytes(e.window) + engine.prefix_bytes) + engine.max_raw_bytes;
    const spare = pool.room(hip.usage(false).device) -| planned;
    const budget: usize = if (o.cache_gib) |g| blk: {
        if (!std.math.isFinite(g) or g < 0 or g * (1 << 30) > @as(f64, @floatFromInt(spare))) {
            problem.* = "the Parse prompt cache exceeds free memory after its admitted lanes; lower --prompt-cache-gib";
            return error.Refused;
        }
        break :blk @intFromFloat(g * (1 << 30));
    } else @min(spare, 256 << 20);
    h.cache.budget = if (dev.policy.slots == 0) 0 else budget;
    const own = try gpa.create(Owned);
    errdefer gpa.destroy(own);
    const startup = try std.fmt.allocPrint(gpa, "HIP Nemotron Parse BF16: {d} streams, {d} context rows including image; native RADIO and mBART, one-step MTP", .{ e.streams, e.window });
    own.* = .{ .a = gpa, .e = e, .h = h, .startup = startup, .image_tokens = @min(o.vision.?.image_tokens, engine.max_image_rows) };
    return .{ .backend = h.backend(), .facts = h.facts(), .rows = @import("backend.zig").max_window, .ctx = own, .deinit = release, .streams = e.streams, .startup = startup, .offer = .{
        .vision = .{ .ctx = own, .prepare = prepare, .image_token = 1, .markers = &.{1}, .image_tokens = own.image_tokens, .max_images = 1 },
        .cache = if (h.cache.budget > 0) &h.cache else null,
    } };
}

pub fn follow(_: std.mem.Allocator, _: std.mem.Allocator, _: std.Io, _: hip.Device, _: []const u8, _: serve.Options, problem: *[]const u8) !void {
    problem.* = "Nemotron Parse supports one HIP rank";
    return error.Refused;
}

test "Parse refuses missing image input, tensor parallelism and unsupported trained-head depth before GPU setup" {
    try std.testing.expect(refusal(.{ .dir = "d", .model_type = model_type }) != null);
    const base: api.Open = .{ .dir = "d", .model_type = model_type, .vision = .{} };
    try std.testing.expect(refusal(base) == null);
    var other = base;
    other.tp = 2;
    try std.testing.expect(refusal(other) != null);
    other = base;
    other.mtp_drafts = 2;
    try std.testing.expect(refusal(other) != null);
    other = base;
    other.context = 16385;
    try std.testing.expect(refusal(other) != null);
}
