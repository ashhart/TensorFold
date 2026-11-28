//! The CUDA loop over a synthetic 2-block case and the runtime fake, against a float64 reference of its host steps.
const std = @import("std");
const bf = @import("bf16_round");
const loop = @import("cuda_loop.zig");
const mock = @import("cuda_runtime_fake.zig");
const st = loop.st;
const geometry = loop.geometry;
const Allocator = std.mem.Allocator;

const hidden = 256;
const inner = 256;
const video_in = 4;
const audio_in = 2;
const lines = 2;
const times = 2;
const steps = 2;
const rot = 2;
// prefix tiles, video tiles, keep, text, audio, video, condition (a first frame), held reference audio
const counts = [8]i32{ 1, 1, 0, 2, 3, 4, 1, 1 };
const rows = 10;

/// A case file's tensors, built in memory and indexed as geometry.read reads them.
const Built = struct {
    arena: std.heap.ArenaAllocator,
    names: st.Header = .empty,
    data: std.ArrayList(u8) = .empty,

    fn put(b: *Built, name: []const u8, dtype: st.DType, shape: []const usize, bytes: []const u8) !void {
        const a = b.arena.allocator();
        var e: st.Entry = .{ .dtype = dtype, .rank = @intCast(shape.len), .shape = undefined, .begin = b.data.items.len, .end = b.data.items.len + bytes.len };
        @memcpy(e.shape[0..shape.len], shape);
        try b.names.put(a, name, e);
        try b.data.appendSlice(a, bytes);
    }

    fn floats(b: *Built, name: []const u8, shape: []const usize, seed: u32, scale: f32) !void {
        const n = count(shape);
        const v = try b.arena.allocator().alloc(f32, n);
        for (v, 0..) |*x, i| x.* = scale * wave(seed, i);
        try b.put(name, .f32, shape, std.mem.sliceAsBytes(v));
    }

    fn halves(b: *Built, name: []const u8, shape: []const usize, seed: u32, scale: f32, bias: f32) !void {
        const n = count(shape);
        const v = try b.arena.allocator().alloc(u16, n);
        for (v, 0..) |*x, i| x.* = bf.bits(bias + scale * wave(seed, i));
        try b.put(name, .bf16, shape, std.mem.sliceAsBytes(v));
    }

    fn ints(b: *Built, name: []const u8, shape: []const usize, v: []const i32) !void {
        try b.put(name, .i32, shape, std.mem.sliceAsBytes(v));
    }

    fn case(b: *const Built) geometry.Case {
        return .{ .names = b.names, .data = b.data.items };
    }
};

fn count(shape: []const usize) usize {
    var n: usize = 1;
    for (shape) |d| n *= d;
    return n;
}

/// A fixed value in [-1, 1] for (seed, i): synthetic weights and rows.
fn wave(seed: u32, i: usize) f32 {
    const h = std.hash.Wyhash.hash(seed, std.mem.asBytes(&i));
    return @as(f32, @floatFromInt(h % 2001)) / 1000.0 - 1.0;
}

fn build(a: Allocator) !Built {
    var b: Built = .{ .arena = .init(a) };
    errdefer b.arena.deinit();
    const x = b.arena.allocator();
    try b.ints("geometry", &.{8}, &counts);
    try b.halves("text", &.{ 2, hidden }, 1, 1.0, 0);
    try b.floats("condition", &.{ 1, video_in }, 2, 1.0);
    try b.halves("final", &.{ times, 2 * hidden }, 3, 0.25, 0);
    try b.halves("final_norm.weight", &.{hidden}, 4, 0.25, 1.0);
    inline for (.{ .{ "video", video_in, 4 }, .{ "audio", audio_in, 3 } }, 0..) |k, i| {
        const seed: u32 = @intCast(10 * i);
        try b.floats(k[0] ++ "_in.weight", &.{ hidden, k[1] }, seed + 10, 0.5);
        try b.floats(k[0] ++ "_in.bias", &.{hidden}, seed + 11, 0.1);
        try b.floats(k[0] ++ "_out.weight", &.{ k[1], hidden }, seed + 12, 0.05);
        try b.floats(k[0] ++ "_out.bias", &.{k[1]}, seed + 13, 0.1);
        try b.floats(k[0], &.{ k[2], k[1] }, seed + 14, 1.0);
        try b.floats(k[0] ++ "_velocity", &.{ k[2], k[1] }, seed + 15, 1.0);
        const schedule = [_]f32{ 0.9, 0.6, 0.5, 0.0 }; // (the sigma seen, the grid's ratio) a step
        try b.put(k[0] ++ "_step", .f32, &.{ steps, 2 }, std.mem.sliceAsBytes(&schedule));
    }
    const ones = try x.alloc(f32, rows * rot);
    @memset(ones, 1);
    const zeros = try x.alloc(f32, rows * rot);
    @memset(zeros, 0);
    try b.put("cos", .f32, &.{ rows, rot }, std.mem.sliceAsBytes(ones));
    try b.put("sin", .f32, &.{ rows, rot }, std.mem.sliceAsBytes(zeros));
    var line: [steps * rows]i32 = undefined;
    for (&line, 0..) |*v, i| v.* = @intCast((i + i / rows) % lines);
    try b.ints("adaln", &.{ steps, rows }, &line);
    try b.ints("times", &.{ steps, rows }, &line);
    try b.ints("tile_slot", &.{rows}, &.{ 0, 1, 2, 3, 4, 5, 64, 65, 66, 67 });
    try b.ints("tile_sizes", &.{2}, &.{ 6, 4 });
    inline for (.{ "tables.0", "tables.1" }, 0..) |name, i| try b.halves(name, &.{ 6, lines, hidden }, @intCast(30 + i), 0.25, 0);
    return b;
}

/// The checkpoint shapes geometry.read checks: q, k and v fused (the CUDA library's layout), and fc2.
fn entry(shape: [2]usize) st.Entry {
    var e: st.Entry = .{ .dtype = .bf16, .rank = 2, .shape = undefined, .begin = 0, .end = shape[0] * shape[1] * 2 };
    e.shape[0] = shape[0];
    e.shape[1] = shape[1];
    return e;
}

/// The library's host entry without its kernel pass: the fake runs none, so the blocks leave the stream as it is.
fn staged(ctx: *anyopaque, in: *const loop.Inputs) anyerror!void {
    const m: *loop.Model = @ptrCast(@alignCast(ctx));
    const p = try m.plan(in);
    try m.drain();
    _ = try loop.host.stage(m, in, p);
    try loop.host.readback(m, in, p);
}

fn fake() !*loop.Model {
    const m = try mock.model();
    m.gpa.free(m.blocks);
    m.blocks = try m.gpa.alloc(@TypeOf(m.blocks[0]), 2);
    for (m.blocks) |*b| b.* = .{ .qkvc = none(m), .out = none(m), .fc1 = none(m), .fc2 = none(m), .norms = none(m) };
    m.hidden = hidden;
    m.inner = inner;
    m.heads = inner / 128;
    m.mlp = 256;
    return m;
}

fn none(m: *loop.Model) @TypeOf(m.blocks[0].qkvc) {
    return .{ .d = &m.driver, .ptr = 0, .len = 0, .borrowed = true };
}

/// The float64 reference of one step's host arithmetic, rounding to bf16 where the loop's float32 does.
fn reference(c: *const loop.Case, video: []const f64, audio: []const f64, step: usize, vv: []f64, av: []f64) void {
    var x: [rows * hidden]f64 = undefined;
    for (c.text, 0..) |v, i| x[i] = bf.value(v);
    const media = 2 + 1;
    const cond = [_]f64{ c.condition[0], c.condition[1], c.condition[2], c.condition[3] };
    into(&x, 2, &cond, c.video_in_w, c.video_in_b, video_in);
    into(&x, media, audio, c.audio_in_w, c.audio_in_b, audio_in);
    into(&x, media + 3, video, c.video_in_w, c.video_in_b, video_in);
    var normed: [rows * hidden]f64 = undefined;
    for (0..rows) |row| {
        var sq: f64 = 0;
        for (x[row * hidden ..][0..hidden]) |v| sq += v * v;
        const inv = 1.0 / @sqrt(sq / hidden + loop.eps);
        const t: usize = @intCast(c.times[step * rows + row]);
        const tab = c.final[t * 2 * hidden ..];
        for (0..hidden) |k| {
            const w: f64 = bf.value(c.final_norm[k]);
            const a = bf.bf16(x[row * hidden + k] * inv * w);
            const s = bf.bf16(1.0 + @as(f64, bf.value(tab[hidden + k])));
            normed[row * hidden + k] = bf.bf16(bf.bf16(a * s) + bf.value(tab[k]));
        }
    }
    out(av, &normed, media, c.audio_out_w, c.audio_out_b, audio_in);
    out(vv, &normed, media + 3, c.video_out_w, c.video_out_b, video_in);
}

fn into(x: []f64, first: usize, latents: []const f64, w: []const f32, b: []const f32, width: usize) void {
    for (0..latents.len / width) |r| for (0..hidden) |n| {
        var sum: f64 = 0;
        for (0..width) |k| sum += latents[r * width + k] * w[n * width + k];
        x[(first + r) * hidden + n] = bf.bf16(sum + b[n]);
    };
}

fn out(y: []f64, x: []const f64, first: usize, w: []const f32, b: []const f32, width: usize) void {
    for (0..y.len / width) |r| for (0..width) |n| {
        var sum: f64 = 0;
        for (0..hidden) |k| sum += x[(first + r) * hidden + k] * w[n * hidden + k];
        y[r * width + n] = sum + b[n];
    };
}

fn step64(x: []f64, v: []const f64, seen: f64, ratio: f64) void {
    for (x, v) |*o, d| o.* = ratio * o.* + (1.0 - ratio) * (o.* + seen * d);
}

fn admitted(a: Allocator, b: *const Built) !loop.Case {
    const g = try geometry.read(a, b.case(), entry(.{ 3 * inner, hidden }), entry(.{ hidden, 256 }), 2, true);
    return loop.Case.read(a, b.case(), g, 2, false);
}

test "h3 cuda loop: two blocks and every step on the runtime fake equal a float64 reference within bf16 rounding" {
    const a = std.testing.allocator;
    var b = try build(a);
    defer b.arena.deinit();
    var c = try admitted(a, &b);
    defer c.deinit();
    const m = try fake();
    defer mock.clean(m);
    var l = try loop.Loop.init(a, &c, .{ .ctx = m, .run = staged });
    defer l.deinit(a);
    var video: [4 * video_in]f32 = undefined;
    var audio: [3 * audio_in]f32 = undefined;
    @memcpy(&video, std.mem.bytesAsSlice(f32, @as([]align(1) const u8, try b.case().bytes("video"))));
    @memcpy(&audio, std.mem.bytesAsSlice(f32, @as([]align(1) const u8, try b.case().bytes("audio"))));
    var rv: [video.len]f64 = undefined;
    var ra: [audio.len]f64 = undefined;
    for (&rv, video) |*o, v| o.* = v;
    for (&ra, audio) |*o, v| o.* = v;
    const start = audio;
    try l.run(&video, &audio, 0, steps);
    for (0..steps) |s| {
        var vv: [video.len]f64 = undefined;
        var av: [audio.len]f64 = undefined;
        reference(&c, &rv, &ra, s, &vv, &av);
        step64(&rv, &vv, c.video_step[2 * s], c.video_step[2 * s + 1]);
        step64(ra[audio_in..], av[audio_in..], c.audio_step[2 * s], c.audio_step[2 * s + 1]); // one held audio row
    }
    var worst: f64 = 0;
    for (video, rv) |o, r| worst = @max(worst, @abs(o - r) / (1 + @abs(r)));
    for (audio, ra) |o, r| worst = @max(worst, @abs(o - r) / (1 + @abs(r)));
    // float32 against float64 sums can put a rounding across a bf16 boundary: within one bf16 ulp (seen: 4.4e-5)
    try std.testing.expect(worst <= 1.0 / 256.0);
    // the held reference audio row is never stepped
    try std.testing.expectEqualSlices(f32, start[0..audio_in], audio[0..audio_in]);
    // both blocks' tables went to the device: (blocks, lines, 6, hidden) floats
    try std.testing.expectEqual(@as(usize, 2 * lines * 6 * hidden * 4), m.inputs[1].?.len);
}

test "h3 cuda loop: one step is one forward and one Euler update, exactly" {
    const a = std.testing.allocator;
    var b = try build(a);
    defer b.arena.deinit();
    var c = try admitted(a, &b);
    defer c.deinit();
    const m = try fake();
    defer mock.clean(m);
    var l = try loop.Loop.init(a, &c, .{ .ctx = m, .run = staged });
    defer l.deinit(a);
    var video: [4 * video_in]f32 = undefined;
    var audio: [3 * audio_in]f32 = undefined;
    @memcpy(&video, std.mem.bytesAsSlice(f32, @as([]align(1) const u8, try b.case().bytes("video"))));
    @memcpy(&audio, std.mem.bytesAsSlice(f32, @as([]align(1) const u8, try b.case().bytes("audio"))));
    var by_hand_v = video;
    var by_hand_a = audio;
    try l.velocities(&by_hand_v, &by_hand_a, 0);
    loop.euler.advance(&by_hand_v, l.vv, c.video_step[0], c.video_step[1]);
    loop.euler.advance(by_hand_a[audio_in..], l.av[audio_in..], c.audio_step[0], c.audio_step[1]);
    try l.run(&video, &audio, 0, 1);
    try std.testing.expectEqualSlices(u32, std.mem.bytesAsSlice(u32, std.mem.asBytes(&by_hand_v)), std.mem.bytesAsSlice(u32, std.mem.asBytes(&video)));
    try std.testing.expectEqualSlices(u32, std.mem.bytesAsSlice(u32, std.mem.asBytes(&by_hand_a)), std.mem.bytesAsSlice(u32, std.mem.asBytes(&audio)));
    try std.testing.expectError(error.H3BadStep, l.run(&video, &audio, 0, steps + 1));
}
