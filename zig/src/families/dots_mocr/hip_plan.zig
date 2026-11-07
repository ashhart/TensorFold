//! The persistent serving allocation plan, fitted inside the HIP admission pool before weight upload.
const std = @import("std");
const cfg = @import("config.zig");
const text = @import("hip_text.zig");
const vision = @import("hip_vision.zig");

pub const prompt_rows: usize = 256;
pub const max_window: usize = 8;
const gib: usize = 1 << 30;

pub const Options = struct {
    window: usize,
    streams: usize,
    fixed: bool = false,
    image_tokens: usize = 0,
    cache_gib: ?f64 = null,
    /// The shared drawer reports its allocation size; the family does not duplicate that layout here.
    sampler_bytes: usize = 0,
    sampler_host_bytes: usize = 0,
    unified: bool = false,
};

pub const Plan = struct {
    window: usize,
    streams: usize,
    rows: usize,
    prompt_rows: usize,
    image_rows: usize,
    image_tokens: usize,
    scratch_bytes: usize,
    pixels_bytes: usize,
    positions_bytes: usize,
    ids_bytes: usize,
    lane_bytes: usize,
    feature_bytes: usize,
    cache_bytes: usize,
    fixed_bytes: usize,
    pinned_bytes: usize,
    planned_bytes: usize,
};

fn cacheBytes(requested: ?f64) !?usize {
    const value = requested orelse return null;
    if (!std.math.isFinite(value) or value < 0 or value * gib >= @as(f64, @floatFromInt(std.math.maxInt(usize))))
        return error.BadCacheBudget;
    return @intFromFloat(value * gib);
}

pub fn make(c: cfg.Config, room: usize, weight_bytes: usize, o: Options) !Plan {
    if (o.window == 0 or o.window > c.context) return error.DotsContext;
    if (o.streams == 0) return error.DotsStreamsDoNotFit;
    if (o.image_tokens > c.context) return error.DotsVisionBudget;
    const chunk: usize = @min(o.window, prompt_rows);
    const width: usize = @min(o.window, max_window);
    const image_rows = try std.math.mul(usize, @min(o.image_tokens, 14400), 4);
    const scratch = @max(try text.scratchBytes(c, chunk, false), try text.scratchBytes(c, width, true));
    const arena = if (image_rows > 0) @max(scratch, try vision.scratchBytes(c, image_rows)) else scratch;
    const pixels = try std.math.mul(usize, image_rows, @as(usize, 3) * c.vision.patch * c.vision.patch * 4);
    const positions: usize = @max(chunk * 4, image_rows * 8);
    const ids: usize = chunk * 4;
    const features = try std.math.mul(usize, o.image_tokens, @as(usize, c.hidden) * 2);
    const lane = try std.math.add(usize, try text.Cache.bytes(c, o.window), try std.math.add(usize, features, @as(usize, c.vocab) * 2));
    var fixed = try std.math.add(usize, weight_bytes, arena);
    for ([_]usize{ pixels, positions, ids, o.sampler_bytes }) |bytes| fixed = try std.math.add(usize, fixed, bytes);
    var pinned: usize = 0;
    for ([_]usize{ pixels, positions, ids, o.sampler_host_bytes }) |bytes| pinned = try std.math.add(usize, pinned, bytes);
    if (o.unified) fixed = try std.math.add(usize, fixed, pinned);
    if (fixed > room or lane > room - fixed) return error.DotsMemoryBudget;
    const cache: usize = (try cacheBytes(o.cache_gib)) orelse @min(2 * gib, (room - fixed - lane) / 4);
    if (cache > room - fixed - lane) return error.DotsMemoryBudget;
    const fits = (room - fixed - cache) / lane;
    const streams: usize = @min(o.streams, fits);
    if (o.fixed and streams < o.streams) return error.DotsStreamsDoNotFit;
    const rows = try std.math.mul(usize, streams, width);
    if (rows > std.math.maxInt(u32)) return error.DotsStreamsDoNotFit;
    const total = try std.math.add(usize, fixed + cache, try std.math.mul(usize, streams, lane));
    return .{ .window = o.window, .streams = streams, .rows = rows, .prompt_rows = chunk, .image_rows = image_rows, .image_tokens = o.image_tokens, .scratch_bytes = arena, .pixels_bytes = pixels, .positions_bytes = positions, .ids_bytes = ids, .lane_bytes = lane, .feature_bytes = features, .cache_bytes = cache, .fixed_bytes = fixed, .pinned_bytes = pinned, .planned_bytes = total };
}

test "dots HIP plan: explicit lanes refuse while automatic lanes fit the same checked budget" {
    const cases = @import("config_test.zig");
    const c = try cfg.Config.parse(std.testing.allocator, cases.sample, cases.generation);
    const options: Options = .{ .window = 32768, .streams = 2, .fixed = true, .image_tokens = 14400, .cache_gib = 0 };
    const full = try make(c, 24 * gib, 6078358528, options);
    const almost = full.fixed_bytes + 2 * full.lane_bytes - 1;
    try std.testing.expectError(error.DotsStreamsDoNotFit, make(c, almost, 6078358528, options));
    var auto = options;
    auto.fixed = false;
    const reduced = try make(c, almost, 6078358528, auto);
    try std.testing.expectEqual(@as(usize, 1), reduced.streams);
    try std.testing.expect(reduced.planned_bytes <= almost);
    try std.testing.expectError(error.DotsMemoryBudget, make(c, full.fixed_bytes + full.lane_bytes - 1, 6078358528, auto));
    try std.testing.expectEqual(@as(usize, 57600), full.image_rows);
    try std.testing.expectEqual(@as(usize, 1024), full.ids_bytes);
    try std.testing.expectEqual(@as(usize, 460800), full.positions_bytes);
    try std.testing.expectEqual(@as(usize, 135475200), full.pixels_bytes);
}

test "dots HIP plan: cache and image budgets cannot bypass context or memory limits" {
    const cases = @import("config_test.zig");
    const c = try cfg.Config.parse(std.testing.allocator, cases.sample, cases.generation);
    var o: Options = .{ .window = 32768, .streams = 2, .cache_gib = 0 };
    for ([_]f64{ -1, std.math.nan(f64), std.math.inf(f64), 0x1p64 }) |g| {
        o.cache_gib = g;
        try std.testing.expectError(error.BadCacheBudget, make(c, 24 * gib, 6078358528, o));
    }
    o.cache_gib = 24;
    try std.testing.expectError(error.DotsMemoryBudget, make(c, 24 * gib, 6078358528, o));
    o.cache_gib = null;
    const default = try make(c, 24 * gib, 6078358528, o);
    try std.testing.expect(default.cache_bytes > 0 and default.cache_bytes <= 2 * gib);
    try std.testing.expect(default.planned_bytes <= 24 * gib);
    try std.testing.expectEqual(@as(usize, 0), default.pixels_bytes);
    try std.testing.expectEqual(@as(usize, 1024), default.positions_bytes);
    o.image_tokens = c.context + 1;
    try std.testing.expectError(error.DotsVisionBudget, make(c, 24 * gib, 6078358528, o));
    o.image_tokens = 0;
    o.window = c.context + 1;
    try std.testing.expectError(error.DotsContext, make(c, 24 * gib, 6078358528, o));
}

test "dots HIP plan: unified admission counts pinned staging and rejects overflow" {
    const cases = @import("config_test.zig");
    const c = try cfg.Config.parse(std.testing.allocator, cases.sample, cases.generation);
    var o: Options = .{ .window = 32768, .streams = 1, .fixed = true, .image_tokens = 14400, .cache_gib = 0, .sampler_host_bytes = 1234 };
    const discrete = try make(c, 24 * gib, 6078358528, o);
    o.unified = true;
    const unified = try make(c, 24 * gib, 6078358528, o);
    try std.testing.expectEqual(discrete.planned_bytes + discrete.pinned_bytes, unified.planned_bytes);
    try std.testing.expectError(error.DotsMemoryBudget, make(c, unified.planned_bytes - 1, 6078358528, o));
    o.sampler_host_bytes = std.math.maxInt(usize);
    try std.testing.expectError(error.Overflow, make(c, 24 * gib, 6078358528, o));
}
