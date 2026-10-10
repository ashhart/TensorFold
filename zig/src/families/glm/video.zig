//! A video's frames to GLM5-Next vision patches as transformers 5.19's Glm5NextVideoProcessor makes them from the
//! frames its PyAV loader decodes (video_utils.read_video_pyav): frames sampled at 2 a second (sample_frames), the
//! canvas sized under the video's token budget (smart_resize over the frame pairs), each frame resized with torch's
//! antialiased uint8 bicubic (torchvision's resize on CPU) and padded with black, normalized in float32 against
//! mean * 255 and std * 255 (the fused rescale), and each pair of frames cut into 2x2 windows of 14x14 patches: one
//! image a pair, grid (1, h, w), as get_video_features hands them to the tower.
const std = @import("std");
const vision = @import("vision.zig");

pub const Error = error{ VideoTooShort, VideoTooLarge, BadVideo, OutOfMemory };

/// The processor's frame rate and sampling cap.
pub const sample_fps = 2.0;

/// A decoded video's facts (the PyAV loader's VideoMetadata) and its frames on request.
pub const Source = struct {
    frames: u64, // the stream's frame count
    rate_num: u64, // its average frame rate, num / den
    rate_den: u64,
    width: u32,
    height: u32,
    ctx: *anyopaque,
    /// The frames at `indices` (ascending, distinct), each handed to `sink` in order as RGB rows `stride` bytes apart.
    decode: *const fn (ctx: *anyopaque, indices: []const u32, sink: *Sink) anyerror!void,

    fn fps(s: Source) f64 {
        return @as(f64, @floatFromInt(s.rate_num)) / @as(f64, @floatFromInt(s.rate_den));
    }

    /// total / fps as the loader computes it from the Fraction: the exact ratio, rounded once.
    fn duration(s: Source) f64 {
        return @as(f64, @floatFromInt(s.frames * s.rate_den)) / @as(f64, @floatFromInt(s.rate_num));
    }
};

/// Where decoded frames go: each resized into its slot of the canvas.
pub const Sink = struct {
    plan: *const Plan,
    frames: []u8, // [n, content_h, content_w, 3]
    scratch: std.mem.Allocator,
    got: usize = 0,

    /// Frame `k` of the plan, `w` x `h` RGB rows `stride` bytes apart.
    pub fn take(s: *Sink, k: usize, rgb: []const u8, w: u32, h: u32, stride: usize) Error!void {
        const p = s.plan;
        if (k != s.got or k >= p.indices.len or w != p.width or h != p.height) return error.BadVideo;
        const dst = s.frames[k * p.frameBytes() ..][0..p.frameBytes()];
        try resizeInto(s.scratch, rgb, w, h, stride, p.content_w, p.content_h, dst);
        s.got += 1;
    }
};

/// What a video becomes before any frame is decoded: the frames to decode and the canvas they land on.
pub const Plan = struct {
    indices: []u32, // distinct, ascending: the frames decoded
    times: []f64, // one a pair: the pair's first sampled frame's index over the rate (metadata.timestamps[::2])
    width: u32,
    height: u32,
    target_h: u32,
    target_w: u32,
    content_h: u32,
    content_w: u32,

    fn frameBytes(p: Plan) usize {
        return @as(usize, p.content_h) * p.content_w * 3;
    }

    pub fn pairs(p: Plan) usize {
        return (p.indices.len + 1) / 2;
    }

    pub fn tokensPerPair(p: Plan) u32 {
        return (p.target_h / vision.patch) * (p.target_w / vision.patch) / vision.patches_per_token;
    }
};

/// numpy.linspace(start, stop, num, dtype=int): the float steps, the last exactly stop, truncated toward zero.
fn linspace(a: std.mem.Allocator, start: f64, stop: f64, num: usize) ![]u32 {
    const out = try a.alloc(u32, num);
    const step = if (num > 1) (stop - start) / @as(f64, @floatFromInt(num - 1)) else 0;
    for (out, 0..) |*v, i| {
        var y = if (num > 1 and step == 0) start else @as(f64, @floatFromInt(i)) * step + start;
        if (num > 1 and i == num - 1) y = stop;
        v.* = @intFromFloat(@trunc(y));
    }
    return out;
}

/// Python's round(): halves to even.
fn pyRound(x: f64) f64 {
    if (@abs(x - @trunc(x)) == 0.5) return 2.0 * @round(x / 2.0);
    return @round(x);
}

/// Glm5NextVideoProcessor.sample_frames on the PyAV loader's metadata, then the loader's decode: the distinct indices
/// (a duplicate the processor appends to make the count even is the same frame, decoded once).
pub fn sample(a: std.mem.Allocator, s: Source, max_frames: u32) Error![]u32 {
    if (s.frames == 0 or s.rate_num == 0 or s.rate_den == 0) return error.BadVideo;
    const total = s.frames;
    const fps = s.fps();
    var duration = s.duration();
    if (duration == 0) duration = pyRound(@as(f64, @floatFromInt(total - 1)) / fps) + 1;
    const max_seconds: f64 = @trunc(duration);
    var extract_t: u64 = @intFromFloat(@trunc(duration * sample_fps));
    extract_t = @min(extract_t, max_frames);
    if (extract_t == 0) return error.VideoTooShort; // the processor samples nothing and the loader fails
    var picked: std.ArrayList(u32) = .empty;
    defer picked.deinit(a);
    var indices: []u32 = undefined;
    if (total < extract_t) {
        indices = try linspace(a, 0, @floatFromInt(total - 1), extract_t);
    } else {
        const per_frame = 1.0 / fps;
        const inv: f64 = 1.0 / sample_fps;
        var current: f64 = 0;
        for (0..total) |i| {
            if (@as(f64, @floatFromInt(i)) * per_frame >= current) {
                current += inv;
                try picked.append(a, @intCast(i));
                if (current >= max_seconds) break;
            }
        }
        if (picked.items.len < extract_t) {
            const start: f64 = if (picked.items.len == 0) 0 else @floatFromInt(picked.items[0]);
            const end: f64 = if (picked.items.len == 0) @floatFromInt(total - 1) else @floatFromInt(picked.items[picked.items.len - 1]);
            indices = try linspace(a, start, end, extract_t);
        } else if (picked.items.len > extract_t) {
            indices = try linspace(a, 0, @floatFromInt(total - 1), extract_t);
        } else indices = try a.dupe(u32, picked.items);
    }
    // the order kept, repeats dropped (linspace's are adjacent, the loop's ascending)
    var n: usize = 0;
    for (indices) |v| {
        if (n > 0 and indices[n - 1] == v) continue;
        indices[n] = v;
        n += 1;
    }
    return indices[0..n];
}

/// The processor's smart_resize: the canvas, multiples of 28, whose frame pairs fit `max_tokens` (at least
/// `min_tokens`), the aspect kept.
pub fn canvas(frames: u64, height: u32, width: u32, min_tokens: u64, max_tokens: u64) Error![2]u32 {
    const factor: u64 = vision.patch * vision.merge;
    const per_token = vision.temporal * factor * factor;
    const min_pixels = min_tokens * per_token;
    const max_pixels = max_tokens * per_token;
    const align_ = struct {
        fn f(v: u64, by: u64) u64 {
            return std.math.divCeil(u64, v, by) catch unreachable;
        }
    }.f;
    const frames_f: f64 = @floatFromInt(frames);
    const aligned_frames: u64 = @max(vision.temporal, @as(u64, @intFromFloat(pyRound(frames_f / vision.temporal))) * vision.temporal);
    var ah = align_(height, factor) * factor;
    var aw = align_(width, factor) * factor;
    if (aligned_frames * ah * aw < min_pixels) {
        const scale = @sqrt(@as(f64, @floatFromInt(min_pixels)) / (frames_f * @as(f64, @floatFromInt(height)) * @as(f64, @floatFromInt(width))));
        ah = align_(@max(1, @as(u64, @intFromFloat(@ceil(@as(f64, @floatFromInt(height)) * scale)))), factor) * factor;
        aw = align_(@max(1, @as(u64, @intFromFloat(@ceil(@as(f64, @floatFromInt(width)) * scale)))), factor) * factor;
    }
    if (aligned_frames * ah * aw > max_pixels) {
        if (max_pixels < aligned_frames * factor * factor) return error.VideoTooLarge;
        var low: i64 = 1;
        var high: i64 = height;
        var best_h: u64 = factor;
        var best_w: u64 = factor;
        while (low <= high) {
            const ch: u64 = @intCast(@divFloor(low + high, 2));
            const cw: u64 = @max(1, @as(u64, @intFromFloat(@floor(@as(f64, @floatFromInt(width)) * @as(f64, @floatFromInt(ch)) / @as(f64, @floatFromInt(height))))));
            const th = align_(ch, factor) * factor;
            const tw = align_(cw, factor) * factor;
            if (aligned_frames * th * tw <= max_pixels) {
                best_h = th;
                best_w = tw;
                low = @as(i64, @intCast(ch)) + 1;
            } else high = @as(i64, @intCast(ch)) - 1;
        }
        ah = best_h;
        aw = best_w;
    }
    return .{ @intCast(ah), @intCast(aw) };
}

/// What `s` becomes at most `max_tokens` tokens and `max_frames` sampled frames: the frames to decode and the canvas.
pub fn plan(a: std.mem.Allocator, s: Source, c: vision.Config, max_tokens: u32, max_frames: u32) Error!Plan {
    const indices = try sample(a, s, max_frames);
    const n: u64 = indices.len; // the frames decoded: the processor's num_frames
    const t = try canvas(n, s.height, s.width, @min(c.min_tokens, max_tokens), max_tokens);
    const per_token: u64 = vision.temporal * (vision.patch * vision.merge) * (vision.patch * vision.merge);
    const hf: f64 = @floatFromInt(s.height);
    const wf: f64 = @floatFromInt(s.width);
    var scale = @min(@as(f64, @floatFromInt(t[0])) / hf, @as(f64, @floatFromInt(t[1])) / wf);
    if (n * s.height * s.width >= per_token * c.min_tokens) scale = @min(1.0, scale);
    const content_h: u32 = @intCast(@max(1, @min(t[0], @as(u64, @intFromFloat(@floor(hf * scale))))));
    const content_w: u32 = @intCast(@max(1, @min(t[1], @as(u64, @intFromFloat(@floor(wf * scale))))));
    // metadata.timestamps[::2] over the sampled indices, the even-making duplicate among them
    const pairs = (indices.len + 1) / 2;
    const times = try a.alloc(f64, pairs);
    for (times, 0..) |*v, g| v.* = @as(f64, @floatFromInt(indices[2 * g])) / s.fps();
    return .{ .indices = indices, .times = times, .width = s.width, .height = s.height, .target_h = t[0], .target_w = t[1], .content_h = content_h, .content_w = content_w };
}

/// One pair of frames ready for the tower, and the video's.
pub const Pair = struct { pixels: []f32, gh: u32, gw: u32, tokens: u32, hash: u64 };
pub const Prepared = struct { groups: []Pair, times: []f64 };

/// `s` decoded, resized as it decodes, then normalized and cut into pairs (allocated in `a`).
pub fn prepare(a: std.mem.Allocator, s: Source, c: vision.Config, max_tokens: u32, max_frames: u32) anyerror!Prepared {
    const p = try plan(a, s, c, max_tokens, max_frames);
    const frames = try a.alloc(u8, p.indices.len * p.frameBytes());
    var arena: std.heap.ArenaAllocator = .init(std.heap.page_allocator);
    defer arena.deinit();
    var sink: Sink = .{ .plan = &p, .frames = frames, .scratch = arena.allocator() };
    try s.decode(s.ctx, p.indices, &sink);
    if (sink.got != p.indices.len) return error.BadVideo;
    return .{ .groups = try patchify(a, p, frames, c.mean, c.std), .times = p.times };
}

/// The frames (content size) padded with black to the canvas, normalized as torchvision's fused rescale does
/// ((x - f32(mean * 255)) / f32(std * 255) in float32), an odd last frame repeated, each pair cut into 2x2 windows
/// of 14x14 patches [channel, frame, row, column].
pub fn patchify(a: std.mem.Allocator, p: Plan, frames: []const u8, mean: [3]f32, sd: [3]f32) Error![]Pair {
    const P = vision.patch;
    const M = vision.merge;
    const gh = p.target_h / P;
    const gw = p.target_w / P;
    var m255: [3]f32 = undefined;
    var s255: [3]f32 = undefined;
    const by: f32 = @floatCast(@as(f64, 1.0) / (@as(f64, 1.0) / 255.0));
    for (0..3) |ch| {
        m255[ch] = mean[ch] * by;
        s255[ch] = sd[ch] * by;
    }
    const n = p.indices.len;
    const out = try a.alloc(Pair, p.pairs());
    for (out, 0..) |*pair, g| {
        const px = try a.alloc(f32, @as(usize, gh) * gw * vision.patch_width);
        var row: usize = 0;
        for (0..gh / M) |wy| for (0..gw / M) |wx| for (0..M) |my| for (0..M) |mx| {
            const py0 = (wy * M + my) * P;
            const px0 = (wx * M + mx) * P;
            const dst = px[row * vision.patch_width ..][0..vision.patch_width];
            for (0..3) |ch| for (0..vision.temporal) |t| {
                const f = @min(2 * g + t, n - 1);
                const frame = frames[f * p.frameBytes() ..][0..p.frameBytes()];
                for (0..P) |dy| for (0..P) |dx| {
                    const y = py0 + dy;
                    const x = px0 + dx;
                    const v: u8 = if (y < p.content_h and x < p.content_w) frame[(y * p.content_w + x) * 3 + ch] else 0;
                    dst[((ch * vision.temporal + t) * P + dy) * P + dx] = (@as(f32, @floatFromInt(v)) - m255[ch]) / s255[ch];
                };
            };
            row += 1;
        };
        var h = std.hash.Wyhash.init(0x71de0);
        h.update(std.mem.sliceAsBytes(px));
        pair.* = .{ .pixels = px, .gh = gh, .gw = gw, .tokens = gh * gw / vision.patches_per_token, .hash = h.final() };
    }
    return out;
}

// ---- torch's antialiased bicubic on uint8 (aten UpSampleKernel's separable int16 path, as torchvision calls it) ----

fn cubic(x0: f64) f64 {
    const a = -0.5;
    const x = @abs(x0);
    if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1;
    if (x < 2.0) return (((x - 5) * x + 8) * x - 4) * a;
    return 0.0;
}

const Coeffs = struct { ksize: usize, bounds: []usize, kk: []i32, precision: u5 };

/// The axis's weights (PIL's precompute_coeffs bounds and filter), as int16 fixed point at the largest precision
/// below 22 bits that keeps the largest weight in range.
fn coeffs(a: std.mem.Allocator, in_size: usize, out_size: usize) Error!Coeffs {
    const scale = @as(f64, @floatFromInt(in_size)) / @as(f64, @floatFromInt(out_size));
    const filterscale = @max(scale, 1.0);
    const support = 2.0 * filterscale;
    const ksize: usize = @as(usize, @intFromFloat(@ceil(support))) * 2 + 1;
    const bounds = try a.alloc(usize, out_size * 2);
    const kk = try a.alloc(i32, out_size * ksize);
    const pre_all = try a.alloc(f64, out_size * ksize);
    defer a.free(pre_all);
    @memset(pre_all, 0);
    for (0..out_size) |xx| {
        const pre = pre_all[xx * ksize ..][0..ksize];
        const center = (@as(f64, @floatFromInt(xx)) + 0.5) * scale;
        const ss = 1.0 / filterscale;
        const xmin_i: i64 = @max(0, @as(i64, @intFromFloat(center - support + 0.5)));
        const xmax_i: i64 = @min(@as(i64, @intCast(in_size)), @as(i64, @intFromFloat(center + support + 0.5)));
        const xmin: usize = @intCast(xmin_i);
        const xmax: usize = @intCast(xmax_i - xmin_i);
        var ww: f64 = 0;
        for (0..xmax) |x| {
            pre[x] = cubic((@as(f64, @floatFromInt(x + xmin)) - center + 0.5) * ss);
            ww += pre[x];
        }
        if (ww != 0) for (0..xmax) |x| {
            pre[x] /= ww;
        };
        bounds[xx * 2] = xmin;
        bounds[xx * 2 + 1] = xmax;
    }
    var w_max: f64 = 0;
    for (pre_all) |v| w_max = @max(w_max, @abs(v));
    var precision: u5 = 0;
    while (precision < 22) : (precision += 1) {
        const next: i64 = @intFromFloat(0.5 + w_max * @as(f64, @floatFromInt(@as(i64, 1) << (precision + 1))));
        if (next >= (1 << 15)) break;
    }
    const one: f64 = @floatFromInt(@as(i64, 1) << precision);
    for (pre_all, 0..) |v, i| kk[i] = @intFromFloat(if (v < 0) -0.5 + v * one else 0.5 + v * one);
    return .{ .ksize = ksize, .bounds = bounds, .kk = kk, .precision = precision };
}

fn clip8(v: i64, precision: u5) u8 {
    return @intCast(std.math.clamp(v >> precision, 0, 255));
}

/// `src` (w x h RGB, rows `stride` apart) resized to ow x oh into `dst`: the horizontal pass, then the vertical, each
/// rounding to bytes; an axis whose size is kept is copied.
pub fn resizeInto(a: std.mem.Allocator, src: []const u8, w: u32, h: u32, stride: usize, ow: u32, oh: u32, dst: []u8) Error!void {
    // the horizontal pass (or the rows as they are) into a packed w' x h buffer
    const mid_w: usize = ow;
    const mid = if (ow == w and oh == h) dst else try a.alloc(u8, mid_w * h * 3);
    if (ow == w) {
        for (0..h) |y| @memcpy(mid[y * mid_w * 3 ..][0 .. mid_w * 3], src[y * stride ..][0 .. mid_w * 3]);
    } else {
        const c = try coeffs(a, w, ow);
        for (0..h) |y| for (0..ow) |x| {
            const xmin = c.bounds[x * 2];
            const xmax = c.bounds[x * 2 + 1];
            const k = c.kk[x * c.ksize ..];
            const row = src[y * stride ..];
            for (0..3) |ch| {
                var ss: i64 = @as(i64, 1) << (c.precision - 1);
                for (0..xmax) |j| ss += @as(i64, row[(xmin + j) * 3 + ch]) * k[j];
                mid[(y * mid_w + x) * 3 + ch] = clip8(ss, c.precision);
            }
        };
    }
    if (oh == h) {
        if (mid.ptr != dst.ptr) @memcpy(dst[0 .. mid_w * h * 3], mid[0 .. mid_w * h * 3]);
        return;
    }
    const c = try coeffs(a, h, oh);
    for (0..oh) |y| {
        const ymin = c.bounds[y * 2];
        const ymax = c.bounds[y * 2 + 1];
        const k = c.kk[y * c.ksize ..];
        for (0..mid_w) |x| for (0..3) |ch| {
            var ss: i64 = @as(i64, 1) << (c.precision - 1);
            for (0..ymax) |j| ss += @as(i64, mid[((ymin + j) * mid_w + x) * 3 + ch]) * k[j];
            dst[(y * mid_w + x) * 3 + ch] = clip8(ss, c.precision);
        };
    }
}

/// Python's f"{t:.1f} seconds": one decimal, an exact tie rounded to even.
pub fn seconds(buf: []u8, t: f64) ![]const u8 {
    // t * 10 exactly from the double's mantissa and exponent, rounded once, an exact tie to even (no 128-bit floats:
    // the server links no compiler-rt)
    const bits: u64 = @bitCast(@abs(t));
    const biased: i32 = @intCast(bits >> 52);
    const m: u64 = (bits & ((@as(u64, 1) << 52) - 1)) | (if (biased == 0) 0 else @as(u64, 1) << 52);
    const e: i32 = (if (biased == 0) 1 else biased) - 1075;
    const x = m * 10; // under 2^57
    var r: u64 = 0;
    if (e >= 0) {
        r = x << @intCast(@min(e, 6)); // times past 2^56 seconds are not times
    } else if (e > -64) {
        const s: u6 = @intCast(-e);
        const q = x >> s;
        const rem = x & ((@as(u64, 1) << s) - 1);
        const half = @as(u64, 1) << (s - 1);
        r = q + @intFromBool(rem > half or (rem == half and q & 1 == 1));
    }
    return std.fmt.bufPrint(buf, "{d}.{d} seconds", .{ r / 10, r % 10 });
}

test "sampling as Glm5NextVideoProcessor.sample_frames, then the loader's distinct frames" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var dummy: u8 = 0;
    const src = struct {
        fn of(frames: u64, num: u64, den: u64, ctx: *anyopaque) Source {
            return .{ .frames = frames, .rate_num = num, .rate_den = den, .width = 64, .height = 48, .ctx = ctx, .decode = undefined };
        }
    }.of;
    // 80 frames at 25: 3.2 s, 6 frames at 0, 0.5, 1, 1.5, 2, 2.5 s (the reference run's indices)
    try std.testing.expectEqualSlices(u32, &.{ 0, 13, 25, 38, 50, 63 }, try sample(a, src(80, 25, 1, &dummy), 256));
    // 10 frames at 25 (0.4 s): nothing to sample
    try std.testing.expectError(error.VideoTooShort, sample(a, src(10, 25, 1, &dummy), 256));
    // a long video: 256 frames over the whole length
    const long = try sample(a, src(1500, 10, 1, &dummy), 256);
    try std.testing.expectEqual(@as(usize, 256), long.len);
    try std.testing.expectEqual(@as(u32, 1499), long[255]);
    try std.testing.expectError(error.BadVideo, sample(a, src(0, 25, 1, &dummy), 256));
}

test "seconds as Python's f\"{t:.1f}\"" {
    var buf: [32]u8 = undefined;
    try std.testing.expectEqualStrings("0.2 seconds", try seconds(&buf, 0.25));
    try std.testing.expectEqualStrings("0.8 seconds", try seconds(&buf, 0.75));
    try std.testing.expectEqualStrings("0.5 seconds", try seconds(&buf, 0.52));
    try std.testing.expectEqualStrings("12.0 seconds", try seconds(&buf, 11.96));
}
