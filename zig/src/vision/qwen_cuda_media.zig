//! A Qwen-VL prompt's image rows on the device for a family that runs the shared tower: its plan, features and splice.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const vision = @import("qwen_cuda.zig");
pub const Limits = struct { image_tokens: u32 = 4096, max_images: u32 = 4 };
pub fn checkLimits(l: Limits) !void {
    if (l.image_tokens == 0 or l.image_tokens > 4096 or l.max_images == 0 or l.max_images > 64)
        return error.BadImageLimits;
}
pub const Run = extern struct {
    at: u32,
    gh: u32,
    gw: u32,
    pub fn tokens(r: Run) usize {
        return @as(usize, r.gh / 2) * (r.gw / 2);
    }
    pub fn patches(r: Run) usize {
        return @as(usize, r.gh) * r.gw;
    }
};
pub const Plan = struct {
    a: std.mem.Allocator,
    runs: []Run,
    src: []i32,
    pos: []i32,
    delta: i32,
    rows: usize,
    pub fn init(a: std.mem.Allocator, n: usize, runs: []const Run, limits: Limits) !Plan {
        try checkLimits(limits);
        if (n == 0 or n > std.math.maxInt(i32) / 3 or runs.len == 0 or runs.len > limits.max_images)
            return error.BadImageCount;
        var end: usize = 0;
        var rows: usize = 0;
        for (runs) |r| {
            if (r.gh == 0 or r.gw == 0 or r.gh > 4096 or r.gw > 4096 or r.gh % 2 != 0 or r.gw % 2 != 0 or
                r.patches() > vision.max_patches or r.tokens() > limits.image_tokens) return error.BadImageGrid;
            if (r.at == 0 or r.at <= end or r.at >= n or r.tokens() >= n - r.at) return error.BadImagePlaceholders;
            end = r.at + r.tokens();
            rows += r.tokens();
            if (rows > limits.image_tokens) return error.BadImageGrid;
        }
        const rr = try a.dupe(Run, runs);
        errdefer a.free(rr);
        const src = try a.alloc(i32, n);
        errdefer a.free(src);
        const pos = try a.alloc(i32, 3 * n);
        errdefer a.free(pos);
        return .{ .a = a, .runs = rr, .src = src, .pos = pos, .delta = layout(n, rr, src, pos), .rows = rows };
    }
    pub fn deinit(p: *Plan) void {
        p.a.free(p.runs);
        p.a.free(p.src);
        p.a.free(p.pos);
    }
};

pub fn placeholders(a: std.mem.Allocator, c: vision.Config, ids: []const u32, images: []const lanes.stream.Image, l: Limits) !Plan {
    try checkLimits(l);
    if (images.len == 0 or images.len > l.max_images) return error.BadImageCount;
    var runs: [64]Run = undefined;
    for (images, 0..) |im, i| runs[i] = .{ .at = im.at, .gh = im.gh, .gw = im.gw };
    var p = try Plan.init(a, ids.len, runs[0..images.len], l);
    errdefer p.deinit();
    var j: usize = 0;
    for (ids, 0..) |id, at| {
        if (c.video_token) |video| if (id == video) return error.QwenVideoUnsupported;
        if (j < p.runs.len and at == p.runs[j].at) {
            const r = p.runs[j];
            const im = images[j];
            if (im.tokens != r.tokens() or im.pixels.len != r.patches() * vision.patch_width) return error.BadImagePatches;
            if (ids[at - 1] != c.vision_start or ids[at + r.tokens()] != c.vision_end) return error.BadImagePlaceholders;
            for (ids[at..][0..r.tokens()]) |x| if (x != c.image_token) return error.BadImagePlaceholders;
            j += 1;
        }
        if (id == c.image_token and p.src[at] < 0) return error.BadImagePlaceholders;
    }
    return p;
}

pub fn layout(n: usize, runs: []const Run, src: []i32, pos: []i32) i32 {
    @memset(src, -1);
    var next: i32 = 0;
    var cursor: usize = 0;
    var row: i32 = 0;
    for (runs) |img| {
        const begin = img.at;
        const h: usize = img.gh / 2;
        const w: usize = img.gw / 2;
        for (cursor..begin) |r| {
            const v = next + @as(i32, @intCast(r - cursor));
            for (0..3) |a| pos[a * n + r] = v;
        }
        const base = next + @as(i32, @intCast(begin - cursor));
        for (0..h * w) |k| {
            const r = begin + k;
            pos[r] = base;
            pos[n + r] = base + @as(i32, @intCast(k / w));
            pos[2 * n + r] = base + @as(i32, @intCast(k % w));
            src[r] = row;
            row += 1;
        }
        next = base + @as(i32, @intCast(@max(h, w)));
        cursor = begin + h * w;
    }
    for (cursor..n) |r| {
        const v = next + @as(i32, @intCast(r - cursor));
        for (0..3) |a| pos[a * n + r] = v;
    }
    return next - @as(i32, @intCast(cursor));
}

pub const Encoder = struct {
    ctx: *anyopaque,
    config: vision.Config,
    driver: *const cuda.Driver,
    stream: cuda.Stream,
    splice: cuda.Function,
    encode: *const fn (*anyopaque, []const f32, []const vision.Grid, u64) anyerror!void,
};
pub const View = struct {
    features: u64,
    src: u64,
    rope: u64,
    n: usize,
    delta: i32,
    width: usize,
    splice_fn: cuda.Function,
    // Logical rows retain cache indices. M-RoPE changes the angle alone.
    pub fn inject(m: View, s: cuda.Stream, out: u64, at: usize, rows: usize, copies: usize) !void {
        if (at >= m.n or rows == 0) return;
        var args: cuda.Args = .{};
        args.add(m.features);
        args.add(m.src);
        args.add(out);
        args.add(@as(i32, @intCast(at)));
        args.add(@as(i32, @intCast(m.n)));
        args.add(@as(i32, @intCast(m.width)));
        args.add(@as(i32, @intCast(copies)));
        try cuda.launch.launch(m.splice_fn, .{ .grid = .{ .x = @intCast(rows), .y = @intCast((m.width * copies + 255) / 256), .z = 1 }, .block = .{ .x = 256, .y = 1, .z = 1 } }, s, &args);
    }
};
pub const Media = struct {
    plan: Plan,
    features: cuda.DeviceBuffer,
    src: cuda.DeviceBuffer,
    rope: cuda.DeviceBuffer,
    width: usize,
    splice: cuda.Function,
    pub fn allocate(p: Plan, enc: Encoder) !Media {
        var plan = p;
        errdefer plan.deinit();
        var f = try cuda.DeviceBuffer.alloc(enc.driver, p.rows * enc.config.out_hidden * 2);
        errdefer f.free();
        var src = try cuda.DeviceBuffer.alloc(enc.driver, p.src.len * 4);
        errdefer src.free();
        var rope = try cuda.DeviceBuffer.alloc(enc.driver, p.pos.len * 4);
        errdefer rope.free();
        // each upload waits on the default stream; one whose wait fails drains it before the frees above run
        try @import("qwen_completion.zig").upload(src, 0, std.mem.sliceAsBytes(p.src));
        try @import("qwen_completion.zig").upload(rope, 0, std.mem.sliceAsBytes(p.pos));
        return .{ .plan = p, .features = f, .src = src, .rope = rope, .width = enc.config.out_hidden, .splice = enc.splice };
    }
    pub fn init(a: std.mem.Allocator, enc: Encoder, ids: []const u32, images: []const lanes.stream.Image, l: Limits) !Media {
        const plan = try placeholders(a, enc.config, ids, images, l);
        var m = try allocate(plan, enc);
        errdefer m.deinit();
        errdefer @import("qwen_completion.zig").drain(enc.driver, enc.stream);
        var row: usize = 0;
        // One bounded image per call, using the one shared tower and its reused scratch.
        for (images) |im| {
            try enc.encode(enc.ctx, im.pixels, &.{.{ .h = im.gh, .w = im.gw }}, m.features.ptr + row * m.width * 2);
            row += im.tokens;
        }
        try enc.stream.synchronize();
        return m;
    }
    pub fn view(m: *const Media) View {
        return .{ .features = m.features.ptr, .src = m.src.ptr, .rope = m.rope.ptr, .n = m.plan.src.len, .delta = m.plan.delta, .width = m.width, .splice_fn = m.splice };
    }
    pub fn deinit(m: *Media) void {
        m.features.free();
        m.src.free();
        m.rope.free();
        m.plan.deinit();
    }
};

test "an image plan bounds its grids, counts and limits, and lays out interleaved M-RoPE positions" {
    const a = std.testing.allocator;
    var p = try Plan.init(a, 12, &.{.{ .at = 2, .gh = 4, .gw = 6 }}, .{});
    defer p.deinit();
    try std.testing.expectEqualSlices(i32, &.{ -1, -1, 0, 1, 2, 3, 4, 5, -1, -1, -1, -1 }, p.src);
    try std.testing.expectEqual(@as(i32, -3), p.delta);
    try std.testing.expectEqual(@as(i32, 2), p.pos[2]);
    try std.testing.expectEqual(@as(i32, 3), p.pos[12 + 5]);
    try std.testing.expectEqual(@as(i32, 4), p.pos[24 + 4]);
    try std.testing.expectError(error.BadImageGrid, Plan.init(a, 12, &.{.{ .at = 2, .gh = 3, .gw = 4 }}, .{}));
    try std.testing.expectError(error.BadImageGrid, Plan.init(a, 12, &.{.{ .at = 2, .gh = 4096, .gw = 4096 }}, .{}));
    try std.testing.expectError(error.BadImagePlaceholders, Plan.init(a, 5, &.{.{ .at = 2, .gh = 4, .gw = 4 }}, .{}));
    try std.testing.expectError(error.BadImageCount, Plan.init(a, 20, &.{ .{ .at = 2, .gh = 2, .gw = 2 }, .{ .at = 8, .gh = 2, .gw = 2 } }, .{ .max_images = 1 }));
    try std.testing.expectError(error.BadImageLimits, checkLimits(.{ .max_images = 65 }));
    try std.testing.expectError(error.BadImageLimits, checkLimits(.{ .image_tokens = 4097 }));
}
