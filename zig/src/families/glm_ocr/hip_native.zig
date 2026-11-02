//! Native registration for the downloaded BF16 GLM-OCR checkpoint.
const std = @import("std");
const hip = @import("hip");
const api = @import("engine_api");
const lanes = @import("lanes");
const serve = @import("hip_serve");
const c = @import("config.zig");
const Engine = @import("hip_engine.zig").Engine;
const Hip = @import("hip_lanes.zig").Hip;

pub const model_type = c.model_type;
pub const formats: []const []const u8 = &.{"bf16"};
pub const default_context: i64 = 8192;
pub const prefill_step: u32 = 0;
pub const slide_refusal = "--slide: GLM-OCR on HIP has no Sliding Weights learner";
pub const confidence_refusal = "--mtp-confidence: GLM-OCR uses shared copy drafts; its MTP head is not qualified";
pub const mtp_refusal = "--mtp-drafts: GLM-OCR's MTP head is not qualified; shared copy drafts remain available";
pub const tp_refusal = "--tp: GLM-OCR currently runs on one HIP rank";
/// GLM-OCR writes no reasoning block: replies are all answer content; the prompt still follows enable_thinking.
pub const reasoning = false;

const Owned = struct {
    gpa: std.mem.Allocator,
    e: *Engine,
    h: *Hip,
    startup: []const u8,
};
pub const Loaded = struct {
    backend: lanes.backend.Backend,
    facts: lanes.Model,
    rows: u32 = 32,
    ctx: *anyopaque,
    deinit: *const fn (*anyopaque) void,
    streams: usize,
    startup: []const u8,
    vision: ?api.Vision,
    cache: ?*api.prompt_cache.Store,
    cache_capacity: ?api.PromptCacheCapacity,
};
fn refuse(a: std.mem.Allocator, problem: *[]const u8, comptime fmt: []const u8, args: anytype) error{Refused} {
    problem.* = std.fmt.allocPrint(a, fmt, args) catch fmt;
    return error.Refused;
}
fn release(ptr: *anyopaque) void {
    const own: *Owned = @ptrCast(@alignCast(ptr));
    own.e.ctx.makeCurrent() catch {};
    own.h.deinit();
    own.e.deinit();
    own.gpa.free(own.startup);
    own.gpa.destroy(own);
}
fn prepare(ptr: *anyopaque, a: std.mem.Allocator, bytes: []const u8, max_tokens: u32) !api.PreparedImage {
    const own: *Owned = @ptrCast(@alignCast(ptr));
    const p = try @import("processor.zig").prepare(own.e.io, a, bytes, @min(max_tokens, 6144));
    return .{ .pixels = p.pixels, .gh = @intCast(p.geometry.h / 14), .gw = @intCast(p.geometry.w / 14), .tokens = @intCast(p.geometry.tokens()), .hash = p.hash };
}
const markers = [_]u32{ c.image_start, c.image_token, c.image_end };
pub fn open(a: std.mem.Allocator, gpa: std.mem.Allocator, io: std.Io, dev: hip.Device, dir: []const u8, o: serve.Options, problem: *[]const u8) !Loaded {
    if (o.world != 1 or o.rank != 0 or dev.group != null) return refuse(a, problem, "GLM-OCR currently runs on one HIP rank", .{});
    _ = @import("hip_engine.zig").cacheBudget(o.cache_gib, std.math.maxInt(usize), 1) catch return refuse(a, problem, "GLM-OCR prompt-cache budget must be finite and nonnegative", .{});
    if (o.fixed and o.streams > 4) return refuse(a, problem, "GLM-OCR currently admits at most four shared lanes; use --parallel 1 to 4", .{});
    if (o.vision) |v| if (v.max_images == 0 or v.max_images > 16 or v.image_tokens == 0) return refuse(a, problem, "GLM-OCR image limits need 1 to 16 images and a positive token budget", .{});
    var ck = @import("checkpoint.zig").Checkpoint.open(a, io, dir) catch |err| return refuse(a, problem, "cannot read this GLM-OCR checkpoint ({t})", .{err});
    ck.close();
    const e = Engine.load(gpa, io, dev, dir, o) catch |err| return refuse(a, problem, "cannot load GLM-OCR on HIP: {s}", .{@import("sys").optional_libraries.explain(err) orelse @errorName(err)});
    errdefer e.deinit();
    const h = try Hip.init(gpa, e);
    errdefer h.deinit();
    const own = try gpa.create(Owned);
    errdefer gpa.destroy(own);
    const startup = try std.fmt.allocPrint(gpa, "GLM-OCR BF16 on HIP {s}, device {d}; {d} shared lanes at context {d}; exact verification through 32 rows; full-shape prompt cache {d} bytes in at most {d} slots", .{ hip.kernels.arch, dev.index, e.streams, e.window, e.cache_bytes, e.cache_slots });
    own.* = .{ .gpa = gpa, .e = e, .h = h, .startup = startup };
    const vision: ?api.Vision = if (o.vision) |v| .{ .ctx = own, .prepare = prepare, .image_token = c.image_token, .markers = &markers, .part_type = "image", .image_tokens = if (v.image_tokens_set) v.image_tokens else 6144, .max_images = v.max_images, .explain = @import("sys").optional_libraries.explain } else null;
    return .{ .backend = h.backend(), .facts = h.facts(), .ctx = own, .deinit = release, .streams = e.streams, .startup = startup, .vision = vision, .cache = if (e.cache_bytes > 0) &h.store else null, .cache_capacity = .{ .budget_bytes = e.cache_bytes, .minimum_state_bytes = @import("hip_snapshot.zig").Snapshot.bytes(1) } };
}
pub fn follow(a: std.mem.Allocator, _: std.mem.Allocator, _: std.Io, _: hip.Device, _: []const u8, _: serve.Options, problem: *[]const u8) !void {
    return refuse(a, problem, "GLM-OCR currently runs on one HIP rank", .{});
}

test "GLM-OCR native admission: unsupported requests fail before checkpoint or driver access" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var dev: hip.Device = undefined;
    dev.group = null;
    var why: []const u8 = "";
    const cases = [_]serve.Options{
        .{ .window = 8192, .streams = 2, .world = 2 },
        .{ .window = 8192, .streams = 2, .cache_gib = std.math.inf(f64) },
        .{ .window = 8192, .streams = 5, .fixed = true },
        .{ .window = 8192, .streams = 2, .vision = .{ .max_images = 0 } },
    };
    for (cases) |o| {
        try std.testing.expectError(error.Refused, open(a, a, std.testing.io, dev, "/nonexistent/glm-ocr", o, &why));
        try std.testing.expect(why.len > 0 and std.mem.indexOf(u8, why, "checkpoint") == null);
    }
    try std.testing.expectError(error.Refused, open(a, a, std.testing.io, dev, "/nonexistent/glm-ocr", .{ .window = 8192, .streams = 2 }, &why));
    try std.testing.expect(std.mem.indexOf(u8, why, "checkpoint") != null);
}
