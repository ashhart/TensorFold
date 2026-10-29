//! A video's frames to Qwen-VL tower patches as transformers 5.17's Qwen3VLVideoProcessor makes them from PyAV's.
const std = @import("std");
const picture = @import("../picture/picture.zig");
const Config = @import("config.zig").Config;

pub const Error = error{ VideoTooShort, VideoTooLarge, BadVideo, OutOfMemory };

/// The processor's sampling: two frames a second, at least `min_frames` and at most `max_frames`.
pub const fps = 2.0;
pub const min_frames = 4;
pub const max_frames = 768;

/// A decoded video's facts (the PyAV loader's VideoMetadata) and its frames on request.
pub const Source = struct {
    frames: u64,
    rate_num: u64,
    rate_den: u64,
    width: u32,
    height: u32,
    ctx: *anyopaque,
    /// The frames at `indices` (ascending, distinct), each handed to `sink` in order as RGB rows `stride` bytes apart.
    decode: *const fn (ctx: *anyopaque, indices: []const u32, sink: *Sink) anyerror!void,

    fn rate(s: Source) f64 {
        return @as(f64, @floatFromInt(s.rate_num)) / @as(f64, @floatFromInt(s.rate_den));
    }
};

/// Where decoded frames go: each resized to the plan's size.
pub const Sink = struct {
    plan: *const Plan,
    frames: []u8, // [distinct frames, h, w, 3]
    scratch: std.mem.Allocator,
    got: usize = 0,

    /// The plan's next distinct frame, `w` x `h` RGB rows `stride` bytes apart.
    pub fn take(s: *Sink, k: usize, rgb: []const u8, w: u32, h: u32, stride: usize) Error!void {
        const p = s.plan;
        if (k != s.got or k >= p.distinct.len or w != p.width or h != p.height) return error.BadVideo;
        const dst = s.frames[k * p.frameBytes() ..][0..p.frameBytes()];
        // torchvision's uint8 resize
        try picture.resample.into(s.scratch, .torch, .{ .pixels = rgb, .width = w, .height = h, .stride = stride }, p.w, p.h, dst);
        s.got += 1;
    }
};

/// What a video becomes before any frame is decoded.
pub const Plan = struct {
    indices: []u32, // the sampled frames in order, repeats kept (np.linspace(...).round())
    distinct: []u32, // the frames decoded: indices without repeats, ascending
    times: []f64, // one a pair: the mean of its two frames' times (_calculate_timestamps)
    width: u32,
    height: u32,
    h: u32, // the resized frame
    w: u32,

    fn frameBytes(p: Plan) usize {
        return @as(usize, p.h) * p.w * 3;
    }

    pub fn pairs(p: Plan) usize {
        return (p.indices.len + 1) / 2;
    }
};

/// numpy's round: a half goes to the even neighbour.
fn roundEven(x: f64) f64 {
    const r = @round(x);
    return if (@abs(x - @trunc(x)) == 0.5 and @mod(r, 2) != 0) r - std.math.sign(x) else r;
}

/// sample_frames: int(total / rate * fps) frames, clamped, at np.linspace(0, total - 1, n).round().
pub fn sample(a: std.mem.Allocator, s: Source, most: u32) Error![]u32 {
    if (s.frames == 0 or s.rate_num == 0 or s.rate_den == 0) return error.VideoTooShort;
    const want: u64 = @intFromFloat(@as(f64, @floatFromInt(s.frames)) / s.rate() * fps);
    const n: u64 = @min(@max(want, min_frames), @min(max_frames, most), s.frames);
    if (n < 2) return error.VideoTooShort;
    const out = try a.alloc(u32, n);
    const last: f64 = @floatFromInt(s.frames - 1);
    const steps: f64 = @floatFromInt(n - 1);
    for (out, 0..) |*o, i| o.* = @intFromFloat(roundEven(if (i == n - 1) last else @as(f64, @floatFromInt(i)) * (last / steps)));
    return out;
}

/// The video processor's smart_resize(num_frames, height, width, temporal_factor=2, factor) at `min_px`..`max_px`.
pub fn smartResize(frames: u64, height: u32, width: u32, factor: u32, temporal: u32, min_px: u64, max_px: u64) Error![2]u32 {
    if (frames < temporal) return error.VideoTooShort;
    const f: f64 = @floatFromInt(factor);
    var h: f64 = @floatFromInt(height);
    var w: f64 = @floatFromInt(width);
    if (h < f or w < f) {
        const scale = @max(f / h, f / w);
        h = @trunc(h * scale);
        w = @trunc(w * scale);
    }
    if (@max(h, w) / @min(h, w) > 200) return error.VideoTooLarge;
    const t: f64 = @floatFromInt(frames);
    const tf: f64 = @floatFromInt(temporal);
    var hb = roundEven(h / f) * f;
    var wb = roundEven(w / f) * f;
    const tb = roundEven(t / tf) * tf;
    if (tb * hb * wb > @as(f64, @floatFromInt(max_px))) {
        const beta = @sqrt(t * h * w / @as(f64, @floatFromInt(max_px)));
        hb = @max(f, @floor(h / beta / f) * f);
        wb = @max(f, @floor(w / beta / f) * f);
    } else if (tb * hb * wb < @as(f64, @floatFromInt(min_px))) {
        const beta = @sqrt(@as(f64, @floatFromInt(min_px)) / (t * h * w));
        hb = @ceil(h * beta / f) * f;
        wb = @ceil(w * beta / f) * f;
    }
    return .{ @intFromFloat(hb), @intFromFloat(wb) };
}

/// What `s` becomes at most `max_tokens` tokens and `most` sampled frames.
pub fn plan(a: std.mem.Allocator, s: Source, c: Config, max_tokens: u32, most: u32) Error!Plan {
    const indices = try sample(a, s, most);
    // a pair's token is `temporal` frames of factor x factor pixels: the budget caps the clip's pixels
    const per_token: u64 = @as(u64, c.temporal) * c.factor() * c.factor();
    const top = @min(c.max_pixels, @as(u64, max_tokens) * per_token);
    const size = try smartResize(indices.len, s.height, s.width, c.factor(), c.temporal, @min(c.min_pixels, top), top);
    var distinct: std.ArrayList(u32) = .empty;
    for (indices) |i| if (distinct.items.len == 0 or distinct.items[distinct.items.len - 1] != i) try distinct.append(a, i);
    const times = try a.alloc(f64, (indices.len + 1) / 2);
    const r = s.rate();
    for (times, 0..) |*t, g| {
        const second = indices[@min(2 * g + 1, indices.len - 1)]; // the last index repeated to an even count
        t.* = (@as(f64, @floatFromInt(indices[2 * g])) / r + @as(f64, @floatFromInt(second)) / r) / 2;
    }
    return .{ .indices = indices, .distinct = distinct.items, .times = times, .width = s.width, .height = s.height, .h = size[0], .w = size[1] };
}

/// One pair of frames ready for the tower.
pub const Pair = struct { pixels: []f32, gh: u32, gw: u32, tokens: u32, hash: u64 };
pub const Prepared = struct { groups: []Pair, times: []f64 };

/// `s` decoded, resized as it decodes, then normalized and cut into pairs (allocated in `a`).
pub fn prepare(a: std.mem.Allocator, s: Source, c: Config, max_tokens: u32, most: u32) anyerror!Prepared {
    const p = try plan(a, s, c, max_tokens, most);
    const frames = try a.alloc(u8, p.distinct.len * p.frameBytes());
    var arena: std.heap.ArenaAllocator = .init(std.heap.page_allocator);
    defer arena.deinit();
    var sink: Sink = .{ .plan = &p, .frames = frames, .scratch = arena.allocator() };
    try s.decode(s.ctx, p.distinct, &sink);
    if (sink.got != p.distinct.len) return error.BadVideo;
    return .{ .groups = try patchify(a, p, frames, c), .times = p.times };
}

/// Each pair's patches in merge order, [channel, frame, y, x], normalized by the fused rescale; odd last frame twice.
pub fn patchify(a: std.mem.Allocator, p: Plan, frames: []const u8, c: Config) Error![]Pair {
    const P = c.patch;
    const M = c.merge;
    const gh = p.h / P;
    const gw = p.w / P;
    const width = c.patchWidth();
    const lut = picture.normalize.table(.fused, c.rescale, c.mean, c.std); // the fast processors' fused rescale
    const out = try a.alloc(Pair, p.pairs());
    for (out, 0..) |*pair, g| {
        var slot: [2]usize = undefined; // each of the pair's frames' place among the decoded ones
        for (&slot, 0..) |*k, t| k.* = std.mem.indexOfScalar(u32, p.distinct, p.indices[@min(2 * g + t, p.indices.len - 1)]).?;
        const px = try a.alloc(f32, @as(usize, gh) * gw * width);
        var row: usize = 0;
        for (0..gh / M) |wy| for (0..gw / M) |wx| for (0..M) |my| for (0..M) |mx| {
            const dst = px[row * width ..][0..width];
            for (0..3) |ch| for (0..c.temporal) |t| {
                const frame = frames[slot[@min(t, 1)] * p.frameBytes() ..][0..p.frameBytes()];
                for (0..P) |dy| for (0..P) |dx| {
                    const y = (wy * M + my) * P + dy;
                    const x = (wx * M + mx) * P + dx;
                    dst[((ch * c.temporal + t) * P + dy) * P + dx] = lut[ch][frame[(y * p.w + x) * 3 + ch]];
                };
            };
            row += 1;
        };
        const id = picture.identity.withGrid(picture.identity.ofBytes(0x9e1d, std.mem.sliceAsBytes(px)), .{ gh, gw });
        pair.* = .{ .pixels = px, .gh = gh, .gw = gw, .tokens = gh * gw / (M * M), .hash = id };
    }
    return out;
}

test "sampling: two frames a second over the clip, at least four, the last frame last" {
    const a = std.testing.allocator;
    var ctx: u8 = 0;
    const decode = struct {
        fn f(_: *anyopaque, _: []const u32, _: *Sink) anyerror!void {}
    }.f;
    const ten: Source = .{ .frames = 300, .rate_num = 30, .rate_den = 1, .width = 64, .height = 64, .ctx = &ctx, .decode = decode };
    const s = try sample(a, ten, 768);
    defer a.free(s);
    try std.testing.expectEqual(@as(usize, 20), s.len);
    try std.testing.expectEqual(@as(u32, 0), s[0]);
    try std.testing.expectEqual(@as(u32, 299), s[19]);
    try std.testing.expectEqual(@as(u32, 16), s[1]); // 15.736 rounds up
    const one: Source = .{ .frames = 30, .rate_num = 30, .rate_den = 1, .width = 64, .height = 64, .ctx = &ctx, .decode = decode };
    const f = try sample(a, one, 768);
    defer a.free(f);
    try std.testing.expectEqualSlices(u32, &.{ 0, 10, 19, 29 }, f); // 9.67, 19.33 round to the nearest
}

test "smart_resize keeps the clip's pixels inside the bounds, in multiples of the factor" {
    try std.testing.expectEqual([2]u32{ 352, 640 }, try smartResize(20, 360, 640, 32, 2, 65536, 16777216));
    const big = try smartResize(20, 1080, 1920, 32, 2, 65536, 16777216); // 20 x 1088 x 1920 is past the bound
    try std.testing.expect(20 * @as(u64, big[0]) * big[1] <= 16777216 and big[0] % 32 == 0 and big[1] % 32 == 0);
    try std.testing.expectError(error.VideoTooShort, smartResize(1, 64, 64, 32, 2, 1, 1 << 30));
}
