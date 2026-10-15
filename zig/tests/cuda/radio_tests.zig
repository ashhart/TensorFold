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
        std.debug.print("RESULT radio {s} {d}x{d} patches, {d} tokens: tower worst {d:.4} mean {d:.4} cos {d:.6}; rows worst {d:.4} mean {d:.4} cos {d:.6}; torch bf16 rows worst {d:.4} mean {d:.4}; {d:.3} s tower, {d:.3} s in all\n", .{ name, gh, gw, tokens, e_tower.worst, e_tower.mean, e_tower.cos, e_rows.worst, e_rows.mean, e_rows.cos, e_torch.worst, e_torch.mean, tower_s, all_s });
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
