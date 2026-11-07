//! Request-owned HIP image rows and M-RoPE positions; the shared tower never owns a lane's features.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const api = @import("engine_api");
const lanes = @import("lanes");
const shared = @import("qwen_hip_vision");
const view = @import("../model/view.zig");
const CancelCheck = @import("lanes").stream.CancelCheck;
const cancels = @import("hip_serve").cancel;
const vl = core.qwen_vl;

pub const Vision = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    c: vl.Config,
    limits: api.VisionOpen,
    markers: [3]u32,
    tower: *shared.Tower,
    pub const explain = vl.geometry.problem;

    pub fn config(a: std.mem.Allocator, io: std.Io, dir: []const u8) !vl.Config {
        var arena = std.heap.ArenaAllocator.init(a);
        defer arena.deinit();
        const aa = arena.allocator();
        const cfg = try read(aa, io, dir, "config.json") orelse return error.NoQwenVision;
        if (!declared(aa, cfg)) return error.NoQwenVision;
        // a declared vision_config the parser cannot read is a broken tower, not a text checkpoint
        return try vl.Config.parse(aa, cfg, try read(aa, io, dir, "preprocessor_config.json"), try read(aa, io, dir, "processor_config.json")) orelse error.UnsupportedQwenVision;
    }

    /// Whether config.json names a vision_config other than null: absent or null is a text-only checkpoint.
    fn declared(a: std.mem.Allocator, cfg: []const u8) bool {
        const doc = std.json.parseFromSliceLeaky(std.json.Value, a, cfg, .{}) catch return true;
        if (doc != .object) return true;
        const vc = doc.object.get("vision_config") orelse return false;
        return vc != .null;
    }

    /// Aggregate image tokens also bound each independently encoded image; scratch is fixed at open.
    pub fn patchCapacity(limits: api.VisionOpen) !usize {
        if (limits.image_tokens == 0 or limits.image_tokens > shared.max_patches / 4 or limits.max_images == 0 or limits.max_images > 64) return error.BadImageLimits;
        return @as(usize, limits.image_tokens) * 4;
    }

    pub fn bytes(c: vl.Config, limits: api.VisionOpen) !usize {
        const capacity = try patchCapacity(limits);
        return try shared.weightBytes(c) + try shared.scratchBytes(c, capacity);
    }

    pub fn open(a: std.mem.Allocator, io: std.Io, r: *const hip.Runtime, s: hip.Stream, lib: *const hip.Launcher, dir: []const u8, width: usize, limits: api.VisionOpen) !*Vision {
        const capacity = try patchCapacity(limits);
        const c = try config(a, io, dir);
        if (c.out_hidden != width) return error.UnsupportedQwenVision;
        const v = try a.create(Vision);
        errdefer a.destroy(v);
        v.* = .{ .gpa = a, .io = io, .c = c, .limits = limits, .markers = .{ c.vision_start, c.image_token, c.vision_end }, .tower = try shared.Tower.init(a, io, r, s, lib, dir, c, capacity) };
        return v;
    }

    pub fn deinit(v: *Vision) void {
        v.tower.deinit();
        v.gpa.destroy(v);
    }

    pub fn prepare(v: *const Vision, a: std.mem.Allocator, bytes_: []const u8, max_tokens: u32) !vl.geometry.Prepared {
        return vl.geometry.prepare(v.io, a, bytes_, v.c, @min(max_tokens, @min(v.limits.image_tokens, shared.max_patches / 4)));
    }

    pub fn laneBytes(v: *const Vision, capacity: usize) usize {
        return @as(usize, v.limits.image_tokens) * v.c.out_hidden * 2 + capacity * 3 * 4;
    }
};

fn read(a: std.mem.Allocator, io: std.Io, dir: []const u8, name: []const u8) !?[]const u8 {
    const path = try std.fs.path.join(a, &.{ dir, name });
    return std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(16 << 20)) catch |err| switch (err) {
        error.FileNotFound => null,
        else => err,
    };
}

pub const Request = struct {
    features: hip.DeviceBuffer,
    positions: hip.DeviceBuffer,
    layout: vl.rope.Layout,
    runs: []view.ImageRun,

    pub fn init(v: *Vision, images: []const lanes.stream.Image, prompt: []const u32, capacity: usize) !Request {
        return initCancelable(v, images, prompt, capacity, null);
    }

    pub fn initCancelable(v: *Vision, images: []const lanes.stream.Image, prompt: []const u32, capacity: usize, cancel: ?CancelCheck) !Request {
        try cancels.poll(cancel);
        const c = v.c;
        const layout = try plan(c, v.limits, images, prompt);
        var features = try hip.DeviceBuffer.alloc(v.tower.r, @as(usize, v.limits.image_tokens) * c.out_hidden * 2);
        errdefer features.free();
        var positions = try hip.DeviceBuffer.alloc(v.tower.r, capacity * 12);
        errdefer positions.free();
        const runs = try v.gpa.alloc(view.ImageRun, images.len);
        errdefer v.gpa.free(runs);
        const pos = try v.gpa.alloc([3]i32, capacity);
        defer v.gpa.free(pos);
        for (pos, 0..) |*p, i| p.* = layout.position(@intCast(i));
        try positions.upload(0, std.mem.sliceAsBytes(pos));
        var at: usize = 0;
        errdefer v.tower.s.synchronize() catch @panic("Qwen image request did not drain");
        for (images, runs) |im, *run| {
            try cancels.poll(cancel);
            const out = features.base() + at * c.out_hidden * 2;
            try v.tower.encodeCancelable(im.pixels, &.{.{ .h = im.gh, .w = im.gw }}, out, cancel);
            run.* = .{ .at = im.at, .rows = im.tokens, .features = out };
            at += im.tokens;
        }
        return .{ .features = features, .positions = positions, .layout = layout, .runs = runs };
    }

    pub fn deinit(r: *Request, a: std.mem.Allocator) void {
        r.features.free();
        r.positions.free();
        a.free(r.runs);
    }

    pub fn frame(r: *const Request) view.ImageFrame {
        return .{ .positions = r.positions.base(), .runs = r.runs };
    }
};

/// Validate host inputs before any device allocation or image launch.
pub fn plan(c: vl.Config, limits: api.VisionOpen, images: []const lanes.stream.Image, prompt: []const u32) !vl.rope.Layout {
    if (images.len > limits.max_images) return error.TooManyImages;
    const layout = try vl.rope.Layout.init(prompt, images, .{ .start = c.vision_start, .image = c.image_token, .end = c.vision_end }, c.merge);
    var count: usize = 0;
    for (images) |im| {
        _ = try shared.rowsOf(&.{.{ .h = im.gh, .w = im.gw }}, im.pixels.len);
        count += im.tokens;
        if (count > limits.image_tokens) return error.ImageTokensExceeded;
    }
    for (prompt, 0..) |id, p| if (id == c.image_token and !layout.holds(@intCast(p), 1)) return error.BadImagePlaceholders;
    return layout;
}

test "Qwen HIP image admission checks layouts, limits and orphan placeholders before device use" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const c = (try vl.Config.parse(arena.allocator(), @embedFile("../../../fixtures/lighton-ocr-0.8b.json"), null, null)).?;
    const prompt = [_]u32{ 7, c.vision_start, c.image_token, c.image_token, c.image_token, c.image_token, c.image_token, c.image_token, c.vision_end, 9 };
    var pixels: [24 * shared.patch_width]f32 = undefined;
    const im: lanes.stream.Image = .{ .at = 2, .gh = 4, .gw = 6, .tokens = 6, .pixels = &pixels };
    const l = try plan(c, .{}, &.{im}, &prompt);
    try std.testing.expectEqual(@as(i32, -3), l.shift);
    try std.testing.expectEqual([3]i32{ 7, 7, 7 }, l.position(10));
    try std.testing.expectError(error.ImageTokensExceeded, plan(c, .{ .image_tokens = 5 }, &.{im}, &prompt));
    try std.testing.expectError(error.TooManyImages, plan(c, .{ .max_images = 0 }, &.{im}, &prompt));
    try std.testing.expectError(error.BadImagePlaceholders, plan(c, .{}, &.{}, &prompt));
    var bad = im;
    bad.pixels = pixels[0..1];
    try std.testing.expectError(error.BadImagePatches, plan(c, .{}, &.{bad}, &prompt));
}

test "Qwen HIP image request cancellation stops before host or device setup" {
    const Hook = struct {
        fn check(ctx: *anyopaque) bool {
            const calls: *usize = @ptrCast(@alignCast(ctx));
            calls.* += 1;
            return true;
        }
    };
    var calls: usize = 0;
    var unavailable: Vision = undefined;
    const before = hip.usage(false);
    try std.testing.expectError(error.Cancelled, Request.initCancelable(&unavailable, &.{}, &.{}, 0, .{ .ptr = &calls, .check = Hook.check }));
    try std.testing.expectEqual(@as(usize, 1), calls);
    const after = hip.usage(false);
    try std.testing.expectEqual(before.device, after.device);
    try std.testing.expectEqual(before.host, after.host);
}

test "Qwen HIP cancellation after image allocation drains and releases every request buffer" {
    const Fake = struct {
        var allocs: usize = 0;
        var frees: usize = 0;
        var drains: usize = 0;
        var freed_before_drain = false;
        fn alloc(out: *hip.abi.DevicePtr, _: usize) callconv(.c) hip.abi.Result {
            allocs += 1;
            out.* = @ptrFromInt(0x100000 * allocs);
            return 0;
        }
        fn free(_: hip.abi.DevicePtr) callconv(.c) hip.abi.Result {
            frees += 1;
            freed_before_drain = freed_before_drain or drains == 0;
            return 0;
        }
        fn upload(_: hip.abi.DevicePtr, _: [*]const u8, _: usize) callconv(.c) hip.abi.Result {
            return 0;
        }
        fn drain(_: hip.abi.Stream) callconv(.c) hip.abi.Result {
            drains += 1;
            return 0;
        }
        fn cancel(ctx: *anyopaque) bool {
            const calls: *usize = @ptrCast(@alignCast(ctx));
            calls.* += 1;
            return calls.* == 2;
        }
    };
    Fake.allocs = 0;
    Fake.frees = 0;
    Fake.drains = 0;
    Fake.freed_before_drain = false;
    var runtime: hip.Runtime = .{ .lib = undefined, .api = undefined };
    runtime.api.hipMalloc = Fake.alloc;
    runtime.api.hipFree = Fake.free;
    runtime.api.hipMemcpyHtoD = Fake.upload;
    runtime.api.hipStreamSynchronize = Fake.drain;
    var tower: shared.Tower = undefined;
    tower.r = &runtime;
    tower.s = .{ .r = &runtime, .handle = null };
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const c = (try vl.Config.parse(arena.allocator(), @embedFile("../../../fixtures/lighton-ocr-0.8b.json"), null, null)).?;
    var v: Vision = .{ .gpa = std.testing.allocator, .io = undefined, .c = c, .limits = .{}, .markers = .{ c.vision_start, c.image_token, c.vision_end }, .tower = &tower };
    const prompt = [_]u32{ c.vision_start, c.image_token, c.vision_end, 7 };
    var pixels: [4 * shared.patch_width]f32 = undefined;
    const image: lanes.stream.Image = .{ .at = 1, .tokens = 1, .gh = 2, .gw = 2, .pixels = &pixels };
    var calls: usize = 0;
    const before = hip.usage(false);
    try std.testing.expectError(error.Cancelled, Request.initCancelable(&v, &.{image}, &prompt, prompt.len, .{ .ptr = &calls, .check = Fake.cancel }));
    try std.testing.expectEqual(@as(usize, 2), calls);
    try std.testing.expect(Fake.allocs > 0);
    try std.testing.expectEqual(Fake.allocs, Fake.frees);
    try std.testing.expect(!Fake.freed_before_drain);
    const after = hip.usage(false);
    try std.testing.expectEqual(before.device, after.device);
    try std.testing.expectEqual(before.host, after.host);
}

test "Qwen HIP vision scratch is bounded by checked configured image tokens" {
    try std.testing.expectEqual(@as(usize, shared.max_patches), try Vision.patchCapacity(.{}));
    try std.testing.expectEqual(@as(usize, 784), try Vision.patchCapacity(.{ .image_tokens = 196, .max_images = 1 }));
    try std.testing.expectEqual(@as(usize, 784), try Vision.patchCapacity(.{ .image_tokens = 196, .max_images = 64 }));
    try std.testing.expectError(error.BadImageLimits, Vision.patchCapacity(.{ .image_tokens = 0 }));
    try std.testing.expectError(error.BadImageLimits, Vision.patchCapacity(.{ .image_tokens = 4097 }));
    try std.testing.expectError(error.BadImageLimits, Vision.patchCapacity(.{ .max_images = 0 }));
    try std.testing.expectError(error.BadImageLimits, Vision.patchCapacity(.{ .max_images = 65 }));
}

test "Qwen HIP vision memory matches fixed weights and receipt/default scratch extents" {
    const configs = [_][]const u8{ @embedFile("../../../fixtures/lighton-ocr-0.8b.json"), @embedFile("../../../fixtures/lighton-ocr-4b.json") };
    const receipt_bytes = [_]usize{ 426_487_808, 1_365_398_528 };
    const default_bytes = [_]usize{ 906_343_680, 1_989_024_000 };
    for (configs, receipt_bytes, default_bytes) |json, receipt, default| {
        var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
        defer arena.deinit();
        const c = (try vl.Config.parse(arena.allocator(), json, null, null)).?;
        try std.testing.expectEqual(receipt, try Vision.bytes(c, .{ .image_tokens = 196, .max_images = 1 }));
        try std.testing.expectEqual(default, try Vision.bytes(c, .{}));
        try std.testing.expectEqual(try Vision.bytes(c, .{}), try shared.weightBytes(c) + try shared.scratchBytes(c, shared.max_patches));
        try std.testing.expectError(error.BadImageLimits, Vision.bytes(c, .{ .image_tokens = 4097 }));
    }
}

test "only an absent or null vision_config is a text-only checkpoint; another value is a broken tower" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try std.testing.expect(!Vision.declared(a, "{\"model_type\":\"qwen3_5\"}"));
    try std.testing.expect(!Vision.declared(a, "{\"vision_config\":null}"));
    try std.testing.expect(Vision.declared(a, "{\"vision_config\":[]}"));
    try std.testing.expect(Vision.declared(a, "{\"vision_config\":\"x\"}"));
    try std.testing.expect(Vision.declared(a, "{\"vision_config\":{}}"));
    try std.testing.expect(Vision.declared(a, "not json"));
}
