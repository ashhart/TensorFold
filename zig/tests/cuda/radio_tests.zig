//! RADIO's tower and projector against float32 references on the same pixels; its processor against torch's.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const radio = @import("radio");
const lanes = @import("lanes");
const check = @import("check.zig");
const Gpu = check.Gpu;
const Image = lanes.stream.Image;

/// A reference's rows: the worst row's relative L2 error, the mean of them, and the least cosine.
const Err = struct {
    worst: f64 = 0,
    mean: f64 = 0,
    cos: f64 = 1,

    /// `got` [rows, width] against `want`.
    fn of(got: []const f32, want: []const f32, width: usize) Err {
        var e: Err = .{};
        const rows = want.len / width;
        for (0..rows) |r| {
            var d: f64 = 0;
            var n: f64 = 0;
            var gg: f64 = 0;
            var dot: f64 = 0;
            for (got[r * width ..][0..width], want[r * width ..][0..width]) |g, w| {
                const x: f64 = g;
                const y: f64 = w;
                d += (x - y) * (x - y);
                n += y * y;
                gg += x * x;
                dot += x * y;
            }
            const rel = @sqrt(d / n);
            e.worst = @max(e.worst, rel);
            e.mean += rel / @as(f64, @floatFromInt(rows));
            e.cos = @min(e.cos, dot / @sqrt(n * gg));
        }
        return e;
    }
};

fn readF32(a: std.mem.Allocator, io: std.Io, dir: []const u8, name: []const u8, suffix: []const u8, len: usize) ![]f32 {
    const path = try std.fmt.allocPrint(a, "{s}/{s}{s}", .{ dir, name, suffix });
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 31));
    if (bytes.len != 4 * len) {
        std.debug.print("FAIL {s}: {d} bytes, expected {d}\n", .{ path, bytes.len, 4 * len });
        return error.TestFailed;
    }
    const out = try a.alloc(f32, len);
    @memcpy(std.mem.sliceAsBytes(out), bytes);
    return out;
}

fn meta(a: std.mem.Allocator, io: std.Io, ref: []const u8) !std.json.Value {
    const path = try std.fmt.allocPrint(a, "{s}/meta.json", .{ref});
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 22));
    return std.json.parseFromSliceLeaky(std.json.Value, a, text, .{});
}

fn int(v: std.json.Value, key: []const u8) usize {
    return @intCast(v.object.get(key).?.integer);
}

/// bf16 values at a device address as f32, after the stream's work.
fn back(gpu: Gpu, a: std.mem.Allocator, s: cuda.Stream, at: u64, len: usize) ![]f32 {
    try s.synchronize();
    const raw = try a.alloc(u16, len);
    try gpu.d.check(gpu.d.api.cuMemcpyDtoH_v2(raw.ptr, at, 2 * len), "cuMemcpyDtoH");
    const out = try a.alloc(f32, len);
    for (out, raw) |*o, v| o.* = @bitCast(@as(u32, v) << 16);
    return out;
}

/// `radio REF MODEL`: each reference picture's tower rows and projector rows against float32, beside torch's bf16.
pub fn run(gpu: Gpu, ref: []const u8, model: []const u8) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var problem: []const u8 = "";
    const c = try radio.load(a, gpu.io, model, &problem) orelse {
        std.debug.print("FAIL radio: {s}\n", .{problem});
        return error.TestFailed;
    };
    const t0 = check.now(gpu.io);
    var tower = try radio.Tower.init(gpu.gpa, gpu.io, gpu.d, model, c);
    defer tower.deinit();
    std.debug.print("RESULT radio load {d:.1} s, {d:.2} GiB on the device\n", .{ seconds(gpu, t0), @as(f64, @floatFromInt(tower.bytes)) / (1 << 30) });
    var rows = try radio.Rows.init(gpu.d, c.max_patches / 4, c.text_hidden);
    defer rows.deinit();
    var s = try cuda.Stream.init(gpu.d, true);
    defer s.deinit();
    const doc = try meta(a, gpu.io, ref);
    var first: ?[]const u16 = null;
    var first_img: Image = undefined;
    for (doc.object.get("images").?.array.items) |m| {
        const name = m.object.get("name").?.string;
        const gh = int(m, "gh");
        const gw = int(m, "gw");
        const n = gh * gw;
        const tokens = int(m, "tokens");
        const planes = try readF32(a, gpu.io, ref, name, ".pixels.f32", 3 * n * c.patch * c.patch);
        const pixels = try a.alloc(f32, planes.len);
        radio.image.cut(&c, planes, .{ .h = @intCast(gh), .w = @intCast(gw) }, pixels);
        const img: Image = .{ .pixels = pixels, .gh = @intCast(gh), .gw = @intCast(gw), .at = 0, .tokens = @intCast(tokens) };
        const t1 = check.now(gpu.io);
        const f = try tower.features(s, img);
        const feats = try back(gpu, a, s, f, n * c.hidden);
        const tower_s = seconds(gpu, t1);
        try tower.project(s, f, gh, gw, rows.buf.ptr);
        const got = try back(gpu, a, s, rows.buf.ptr, tokens * c.text_hidden);
        const all_s = seconds(gpu, t1);
        const e_tower = Err.of(feats, try readF32(a, gpu.io, ref, name, ".tower.f32", n * c.hidden), c.hidden);
        const want = try readF32(a, gpu.io, ref, name, ".rows.f32", tokens * c.text_hidden);
        const e_rows = Err.of(got, want, c.text_hidden);
        const e_torch = Err.of(try readF32(a, gpu.io, ref, name, ".rows16.f32", tokens * c.text_hidden), want, c.text_hidden);
        // the same pixels again: the same bits (the rows depend on the image alone)
        try tower.picture(s, img, rows.buf.ptr);
        const again = try back(gpu, a, s, rows.buf.ptr, tokens * c.text_hidden);
        try check.sameBytes("radio rows run twice", std.mem.sliceAsBytes(again), std.mem.sliceAsBytes(got));
        std.debug.print("RESULT radio {s} {d}x{d} patches, {d} tokens: tower worst {d:.4} mean {d:.4} cos {d:.6}; rows worst {d:.4} mean {d:.4} cos {d:.6}; torch bf16 rows worst {d:.4} mean {d:.4}; {d:.3} s tower, {d:.3} s in all; rows hash {x:0>16}\n", .{ name, gh, gw, tokens, e_tower.worst, e_tower.mean, e_tower.cos, e_rows.worst, e_rows.mean, e_rows.cos, e_torch.worst, e_torch.mean, tower_s, all_s, std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(got)) });
        try check.expect(e_rows.mean <= 2 * e_torch.mean and e_rows.worst <= 2 * e_torch.worst, "radio {s}: rows' error past twice torch's own bf16 error", .{name});
        if (first == null) {
            const raw = try a.alloc(u16, tokens * c.text_hidden);
            for (raw, got) |*o, v| o.* = @intCast(@as(u32, @bitCast(v)) >> 16);
            first = raw;
            first_img = img;
        }
    }
    // the first picture once more after the others: no state carries between images
    const kept = first orelse return error.TestFailed;
    try tower.picture(s, first_img, rows.buf.ptr);
    const again = try back(gpu, a, s, rows.buf.ptr, kept.len);
    const raw = try a.alloc(u16, kept.len);
    for (raw, again) |*o, v| o.* = @intCast(@as(u32, @bitCast(v)) >> 16);
    try check.sameBytes("radio first picture after the others", std.mem.sliceAsBytes(raw), std.mem.sliceAsBytes(kept));
    check.pass("radio: every picture within twice torch's bf16 error of float32, bits repeat", .{});
}

fn seconds(gpu: Gpu, from: i96) f64 {
    return @as(f64, @floatFromInt(check.now(gpu.io) - from)) / 1e9;
}

/// `radio-processor REF PICS MODEL`: decoded RGB, the processor's pixels, torch's resizes and grids, bit for bit.
pub fn processor(gpu: Gpu, ref: []const u8, pics: []const u8, model: []const u8) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var problem: []const u8 = "";
    const c = try radio.load(a, gpu.io, model, &problem) orelse return error.TestFailed;
    const doc = try meta(a, gpu.io, ref);
    var bad: usize = 0;
    for (doc.object.get("images").?.array.items) |m| {
        const name = m.object.get("name").?.string;
        const w = int(m, "w");
        const h = int(m, "h");
        const png = try std.Io.Dir.cwd().readFileAlloc(gpu.io, try std.fmt.allocPrint(a, "{s}/{s}.png", .{ pics, name }), a, .limited(16 << 20));
        const rgb = try core.picture.decode(gpu.io, a, png);
        const ref_rgb = try std.Io.Dir.cwd().readFileAlloc(gpu.io, try std.fmt.allocPrint(a, "{s}/{s}.rgb.u8", .{ ref, name }), a, .limited(1 << 28));
        var rgb_diff: usize = 0;
        for (rgb.pixels, ref_rgb) |x, y| rgb_diff += @intFromBool(x != y);
        const prepared = try radio.image.prepareRgb(a, &c, .{ .width = w, .height = h, .pixels = ref_rgb }, 0, 4096);
        const gh = int(m, "gh");
        const gw = int(m, "gw");
        const planes = try readF32(a, gpu.io, ref, name, ".pixels.f32", 3 * gh * gw * c.patch * c.patch);
        const want = try a.alloc(f32, planes.len);
        radio.image.cut(&c, planes, .{ .h = @intCast(gh), .w = @intCast(gw) }, want);
        const px = differ(prepared.pixels, want);
        std.debug.print("RESULT radio-processor {s} {d}x{d} -> {d}x{d} patches (want {d}x{d}): decoded RGB {d} bytes differ; pixels {d} differ, worst {e}\n", .{ name, w, h, prepared.gh, prepared.gw, gh, gw, rgb_diff, px.count, px.worst });
        if (rgb_diff != 0 or px.count != 0 or prepared.gh != gh or prepared.gw != gw) bad += 1;
    }
    for (doc.object.get("resample").?.array.items) |m| {
        const name = m.object.get("name").?.string;
        const h = int(m, "h");
        const w = int(m, "w");
        const th = int(m, "th");
        const tw = int(m, "tw");
        const planar = try std.Io.Dir.cwd().readFileAlloc(gpu.io, try std.fmt.allocPrint(a, "{s}/{s}.in.u8", .{ ref, name }), a, .limited(1 << 26));
        const rgb = try a.alloc(u8, planar.len);
        for (0..3) |ch| for (0..h * w) |i| {
            rgb[i * 3 + ch] = planar[ch * h * w + i];
        };
        const got = try a.alloc(f32, 3 * th * tw);
        try core.picture.resample.bicubicPlanes(a, .of(rgb, w, h), tw, th, got);
        const d = differ(got, try readF32(a, gpu.io, ref, name, ".out.f32", got.len));
        std.debug.print("RESULT radio-processor {s}: {d} of {d} values differ from torch, worst {e}\n", .{ name, d.count, got.len, d.worst });
        if (d.count != 0) bad += 1;
    }
    const sizes_text = try std.Io.Dir.cwd().readFileAlloc(gpu.io, try std.fmt.allocPrint(a, "{s}/sizes.json", .{ref}), a, .limited(1 << 22));
    const sizes = try std.json.parseFromSliceLeaky(std.json.Value, a, sizes_text, .{});
    var grid_bad: usize = 0;
    for (sizes.array.items) |row| {
        const v = row.array.items;
        const g = radio.image.grid(&c, @intCast(v[0].integer), @intCast(v[1].integer), c.max_patches) catch {
            grid_bad += 1;
            continue;
        };
        if (g.w != v[2].integer or g.h != v[3].integer) {
            if (grid_bad < 5) std.debug.print("  grid {d}x{d}: ours {d}x{d}, the processor's {d}x{d}\n", .{ v[0].integer, v[1].integer, g.w, g.h, v[2].integer, v[3].integer });
            grid_bad += 1;
        }
    }
    std.debug.print("RESULT radio-processor grids: {d} of {d} differ from the processor's\n", .{ grid_bad, sizes.array.items.len });
    try check.expect(bad == 0 and grid_bad == 0, "radio-processor: {d} pictures or resizes and {d} grids differ", .{ bad, grid_bad });
    check.pass("radio-processor: decoded RGB, pixels, resizes and grids equal the reference bit for bit", .{});
}

/// `radio-kernels`: tf_vit_cast and tf_vit_position_add against their host references, bit for bit (synthetic rows).
pub fn kernels(gpu: Gpu) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var prng = std.Random.DefaultPrng.init(0x7ad1);
    const rng = prng.random();
    var k = try cuda.vit.Kernels.load(gpu.d);
    defer k.unload();
    var s = try cuda.Stream.init(gpu.d, true);
    defer s.deinit();
    // casts: spread values, and halfway cases that round to even both ways
    const count = 1 << 20;
    const xs = try a.alloc(f32, count);
    for (xs, 0..) |*x, i| x.* = if (i % 7 == 0) @bitCast(@as(u32, 0x3f808000) + @as(u32, @intCast(i % 3)) * 0x10000) else rng.floatNorm(f32) * 3;
    var dx = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(xs));
    defer dx.free();
    var dy = try cuda.DeviceBuffer.alloc(gpu.d, 2 * count);
    defer dy.free();
    try k.cast(s, dx.ptr, dy.ptr, count);
    try s.synchronize();
    const got = try a.alloc(u16, count);
    try dy.download(0, std.mem.sliceAsBytes(got));
    const want = try a.alloc(u16, count);
    for (want, xs) |*w, x| w.* = cuda.vit.bf16(x);
    try check.sameBytes("tf_vit_cast", std.mem.sliceAsBytes(got), std.mem.sliceAsBytes(want));
    // position rows: RADIO's table shape, grids under, at and past the table's side
    const side = 128;
    const c = 1280;
    const table = try a.alloc(u16, side * side * c);
    for (table) |*t| t.* = cuda.vit.bf16(rng.floatNorm(f32) * 0.05);
    const wide = try a.alloc(f32, table.len);
    for (wide, table) |*o, t| o.* = @bitCast(@as(u32, t) << 16);
    var dt = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(table));
    defer dt.free();
    for ([_][2]usize{ .{ 30, 40 }, .{ 46, 136 }, .{ 128, 64 }, .{ 2, 2 }, .{ 68, 182 }, .{ 76, 172 }, .{ 128, 128 } }) |grid| {
        const gh, const gw = grid;
        const n = gh * gw;
        const x = try a.alloc(u16, n * c);
        for (x) |*v| v.* = cuda.vit.bf16(rng.floatNorm(f32));
        var dxr = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(x));
        defer dxr.free();
        var dout = try cuda.DeviceBuffer.alloc(gpu.d, 2 * n * c);
        defer dout.free();
        try k.positionAdd(s, dxr.ptr, dt.ptr, dout.ptr, gh, gw, side, c);
        try s.synchronize();
        const out = try a.alloc(u16, n * c);
        try dout.download(0, std.mem.sliceAsBytes(out));
        const pos = try a.alloc(f32, n * c);
        radio.position.rows(wide, side, c, gh, gw, pos);
        const ref = try a.alloc(u16, n * c);
        for (ref, pos, x) |*r, p, v| r.* = cuda.vit.bf16(@as(f32, @bitCast(@as(u32, cuda.vit.bf16(p)) << 16)) + @as(f32, @bitCast(@as(u32, v) << 16)));
        var name: [64]u8 = undefined;
        try check.sameBytes(try std.fmt.bufPrint(&name, "tf_vit_position_add {d}x{d}", .{ gh, gw }), std.mem.sliceAsBytes(out), std.mem.sliceAsBytes(ref));
    }
    check.pass("radio-kernels: tf_vit_cast and tf_vit_position_add equal their host references bit for bit", .{});
}

/// `radio-attention`: the head-dim-80 attention's time at five tower sizes, its outputs finite (synthetic rows).
pub fn attention(gpu: Gpu) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var prng = std.Random.DefaultPrng.init(0x80);
    const rng = prng.random();
    var at = try cuda.vit.Attention.load(gpu.d);
    defer at.unload();
    var s = try cuda.Stream.init(gpu.d, true);
    defer s.deinit();
    const heads = 16;
    const dim = 80;
    const scale: f32 = @floatCast(1.0 / @sqrt(80.0));
    for ([_]usize{ 1, 77, 1210, 6276, 13082 }) |rows| {
        const qkv = try a.alloc(u16, rows * 3 * heads * dim);
        for (qkv) |*v| v.* = cuda.vit.bf16(rng.floatNorm(f32) * 1.5);
        var dq = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(qkv));
        defer dq.free();
        var dout = try cuda.DeviceBuffer.alloc(gpu.d, 2 * rows * heads * dim);
        defer dout.free();
        var times: [6]f64 = undefined;
        for (&times) |*t| {
            const t0 = check.now(gpu.io);
            try at.full(s, dq.ptr, dout.ptr, rows, heads, dim, scale);
            try s.synchronize();
            t.* = @as(f64, @floatFromInt(check.now(gpu.io) - t0)) / 1e6;
        }
        const out = try a.alloc(u16, rows * heads * dim);
        try dout.download(0, std.mem.sliceAsBytes(out));
        for (out) |v| try check.expect(v & 0x7f80 != 0x7f80, "radio-attention {d} rows: a non-finite output", .{rows});
        const ms = check.median(times[1..]);
        const flops = 4.0 * @as(f64, @floatFromInt(rows * rows * heads * dim));
        std.debug.print("RESULT radio-attention {d} rows: {d:.3} ms, {d:.1} TFLOPS\n", .{ rows, ms, flops / ms / 1e9 });
    }
    check.pass("radio-attention: timed, outputs finite", .{});
}

/// How many values differ, and by how much at most.
fn differ(got: []const f32, want: []const f32) struct { count: usize, worst: f32 } {
    var n: usize = 0;
    var worst: f32 = 0;
    for (got, want) |g, w| if (@as(u32, @bitCast(g)) != @as(u32, @bitCast(w))) {
        n += 1;
        worst = @max(worst, @abs(g - w));
    };
    return .{ .count = n, .worst = worst };
}
