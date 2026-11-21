//! dots.mocr offered to the native HIP lane host; no separate scheduler or Python inference path.
const std = @import("std");
const hip = @import("hip");
const lanes = @import("lanes");
const api = @import("engine_api");
const serve = @import("hip_serve");
const cfg = @import("config.zig");
const engine = @import("hip_engine.zig");
const backend_ = @import("hip_backend.zig");
const text = @import("hip_text.zig");
const ocr = @import("ocr.zig");

pub const model_type = "dots_ocr";
pub const formats: []const []const u8 = &.{"bf16"};
pub const default_context: i64 = 32768;
pub const prefill_step: u32 = 0;
/// OCR replies carry no reasoning block: all reply text is answer content (native/hip.zig reasoning).
pub const reasoning = false;
pub const slide_refusal = "--slide: dots.mocr has no native HIP learner";
pub const confidence_refusal = "--mtp-confidence: dots.mocr has no MTP head; core copy proposals remain available";
pub const policy_line = "rocBLAS BF16; dot2 attention; graphs off; prompt rows 256; window 8; no MTP head";

pub fn policyRefusal(p: hip.Policy) ?[]const u8 {
    if (p.matrix != .auto or p.attention != .auto or p.kernels != .auto or p.stream != .auto or
        p.gemm != .auto or p.fuse != .auto or p.graphs == .on or p.graph_fail >= 0 or
        p.prefill_step < engine.plan.prompt_rows or p.fill_rows < engine.plan.prompt_rows)
        return "dots.mocr uses rocBLAS BF16, dot2 attention, no graphs, and 256-row prompt chunks; this HIP policy override is unsupported";
    return null;
}

pub fn refusal(o: api.Open) ?[]const u8 {
    if (o.tp != 1) return "dots.mocr currently serves one HIP GPU; use --tp 1";
    if (o.lanes_fixed and o.lanes > 128) return "dots.mocr supports at most 128 fixed lanes";
    if (o.mtp_drafts) |n| if (n != 0) return "--mtp-drafts: dots.mocr has no MTP head; core copy proposals remain available";
    return null;
}

pub const Loaded = struct {
    backend: lanes.backend.Backend,
    facts: lanes.Model,
    rows: u32,
    streams: usize,
    ctx: *anyopaque,
    deinit: *const fn (*anyopaque) void,
    startup: []const u8,
    /// Its prompt cache (none when the budget or the slots are zero), images under --vision, and its host settings.
    offer: serve.Offer,
};

const Owner = struct {
    e: *engine.Engine,
    back: *backend_.Backend,
    startup: []const u8,

    fn close(ptr: *anyopaque) void {
        const o: *Owner = @ptrCast(@alignCast(ptr));
        const a = o.e.gpa;
        o.back.deinit();
        o.e.deinit();
        a.free(o.startup);
        a.destroy(o);
    }
};

pub fn open(a: std.mem.Allocator, gpa: std.mem.Allocator, io: std.Io, dev: hip.Device, dir: []const u8, o: serve.Options, problem: *[]const u8) !Loaded {
    if (policyRefusal(dev.policy)) |why| return refuse(a, problem, "{s}", .{why});
    const c = cfg.Config.read(a, io, dir) catch |err| return refuse(a, problem, "dots.mocr config refused: {s}", .{@errorName(err)});
    const image_tokens: usize = if (o.vision) |v| (if (v.image_tokens_set) v.image_tokens else 14400) else 0;
    if (o.vision) |v| if (image_tokens == 0 or v.max_images == 0) return refuse(a, problem, "dots.mocr image limits must be positive", .{});
    const rows = engine.plan.max_window;
    const drawer = serve.draw.Drawer.deviceBytes(rows);
    const e = engine.Engine.load(gpa, io, dir, dev.index, .{
        .window = @intCast(o.window),
        .streams = @min(o.streams, 128),
        .fixed = o.fixed,
        .image_tokens = image_tokens,
        .cache_gib = o.cache_gib,
        .sampler_bytes = drawer,
        .sampler_host_bytes = drawer + @as(usize, c.vocab) * 2,
    }) catch |err| return refuse(a, problem, "dots.mocr HIP load refused: {s}", .{@errorName(err)});
    errdefer e.deinit();
    const back = try backend_.Backend.init(e);
    errdefer back.deinit();
    back.store.most = dev.policy.slots;
    const own = try gpa.create(Owner);
    errdefer gpa.destroy(own);
    const startup = try std.fmt.allocPrint(gpa, "dots.mocr BF16: {d} lanes, context {d}, scratch {d} bytes, prompt-cache budget {d} bytes", .{ e.p.streams, e.p.window, e.p.scratch_bytes, e.p.cache_bytes });
    own.* = .{ .e = e, .back = back, .startup = startup };
    const cached = back.store.budget > 0 and back.store.most > 0;
    const vision: ?api.Vision = if (o.vision) |v| .{
        .ctx = own,
        .prepare = prepare,
        .image_token = c.image_token,
        .markers = &.{ 151665, 151666, 151667 },
        .part_type = "image",
        .image_tokens = @intCast(image_tokens),
        .max_images = v.max_images,
    } else null;
    return .{ .backend = back.backend(), .facts = .{ .exact_width = @intCast(rows), .drafts = 0, .mtp = false, .copy_drafts = true, .streams_exact = true }, .rows = @intCast(e.p.rows), .streams = e.p.streams, .ctx = own, .deinit = Owner.close, .startup = startup, .offer = .{
        .vision = vision,
        .ocr = ocr.reader,
        .cache = if (cached) &back.store else null,
        .cache_capacity = .{ .budget_bytes = e.p.cache_bytes, .minimum_state_bytes = (text.Cache.bytes(e.c, @min(e.p.window, engine.plan.prompt_rows)) catch 0) + @as(u64, e.c.vocab) * 2 },
        .warm_turns = cached,
        .frame_key = frameKey,
        .thinking_budget = false,
        .on_thread = .{ .ctx = e, .run = bind },
        .policy = policy_line,
    } };
}

fn bind(ptr: *anyopaque) anyerror!void {
    const e: *engine.Engine = @ptrCast(@alignCast(ptr));
    try e.ctx.makeCurrent();
}
fn frameKey(images: []const api.Image) u32 {
    return if (images.len > 0) 0xd075 else 0;
}
fn prepare(ptr: *anyopaque, a: std.mem.Allocator, bytes: []const u8, max_tokens: u32) anyerror!api.PreparedImage {
    const own: *Owner = @ptrCast(@alignCast(ptr));
    const processor = own.e.processor.?;
    return ocr.prepared(try processor.prepare(own.e.io, a, bytes, max_tokens), processor.patches.patch);
}
pub fn follow(a: std.mem.Allocator, _: std.mem.Allocator, _: std.Io, _: hip.Device, _: []const u8, _: anytype, problem: *[]const u8) !void {
    return refuse(a, problem, "dots.mocr currently serves one HIP GPU", .{});
}
fn refuse(a: std.mem.Allocator, problem: *[]const u8, comptime fmt: []const u8, args: anytype) error{Refused} {
    problem.* = std.fmt.allocPrint(a, fmt, args) catch fmt;
    return error.Refused;
}
