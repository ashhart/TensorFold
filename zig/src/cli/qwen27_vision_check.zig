//! `tf-qwen27-cuda vision`: the vision tower, image decode and patches against a manifest recorded from 0.6.6.
const std = @import("std");
const cuda = @import("cuda");
const qwen = @import("qwen27");

/// The tower against a 0.6.6 manifest recorded outside the tree: per image, the row cosines and largest error.
pub fn checkVision(gpa: std.mem.Allocator, arena: std.mem.Allocator, io: std.Io, model: []const u8, dir: []const u8) !u8 {
    const V = qwen.vision;
    const manifest = try readJson(arena, io, try std.fs.path.join(arena, &.{ dir, "manifest.json" }));
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var stream = try cuda.Stream.init(&driver, true);
    defer stream.deinit();
    const gpu: V.Gpu = .{ .d = &driver, .s = stream, .discrete = try ctx.attribute(.integrated) == 0 };
    const clock = std.Io.Clock.awake;
    var t0 = clock.now(io).toNanoseconds();
    const tower = try V.Tower.load(gpa, io, gpu, model);
    defer tower.deinit();
    std.debug.print("tower loaded in {d:.1}s, {d:.2} GiB\n", .{ @as(f64, @floatFromInt(clock.now(io).toNanoseconds() - t0)) / 1e9, @as(f64, @floatFromInt(tower.weight_bytes)) / (1 << 30) });
    var worst: f64 = 1;
    for (manifest.object.get("images").?.array.items) |img| {
        const name = img.object.get("name").?.string;
        const grid = img.object.get("grid").?.array.items;
        const g: V.Grid = .{ .h = @intCast(grid[0].integer), .w = @intCast(grid[1].integer) };
        const px_bytes = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(arena, &.{ dir, img.object.get("pixels").?.string }), arena, .limited(1 << 30));
        const want_bytes = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(arena, &.{ dir, img.object.get("features").?.string }), arena, .limited(1 << 30));
        const px = try arena.alloc(f32, px_bytes.len / 4);
        @memcpy(std.mem.sliceAsBytes(px), px_bytes[0 .. px.len * 4]);
        { // our decode and patches from the image's file, against 0.6.6's
            const file = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(arena, &.{ dir, img.object.get("file").?.string }), arena, .limited(1 << 30));
            const want_rgb = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(arena, &.{ dir, img.object.get("rgb").?.string }), arena, .limited(1 << 30));
            const rgb = try qwen.image.decode(gpa, file);
            defer rgb.deinit(gpa);
            var px_diff: usize = 0;
            if (rgb.px.len == want_rgb.len) {
                for (rgb.px, want_rgb) |a, b| px_diff += @intFromBool(a != b);
            } else px_diff = std.math.maxInt(usize);
            const prep = try qwen.image.prepare(gpa, file, 4096);
            defer gpa.free(prep.pixels);
            var patch_diff: usize = 0;
            if (prep.gh == g.h and prep.gw == g.w and prep.pixels.len == px.len) {
                for (prep.pixels, px) |a, b| patch_diff += @intFromBool(@as(u32, @bitCast(a)) != @as(u32, @bitCast(b)));
            } else patch_diff = std.math.maxInt(usize);
            std.debug.print("{s}: decode {d}x{d}, {d} of {d} bytes differ from PIL's; patches grid {d}x{d}, {d} of {d} values differ\n", .{ name, rgb.w, rgb.h, px_diff, want_rgb.len, prep.gh, prep.gw, patch_diff, px.len });
        }
        const rows = @as(usize, g.h) * g.w / 4;
        var out = try cuda.DeviceBuffer.alloc(&driver, rows * V.out_hidden * 2);
        defer out.free();
        t0 = clock.now(io).toNanoseconds();
        try tower.encode(gpu, px, &.{g}, out.ptr);
        const ms = @as(f64, @floatFromInt(clock.now(io).toNanoseconds() - t0)) / 1e6;
        const got = try arena.alloc(u16, rows * V.out_hidden);
        try driver.check(driver.api.cuMemcpyDtoH_v2(@ptrCast(got.ptr), out.ptr, got.len * 2), "cuMemcpyDtoH");
        const want: []align(1) const u16 = std.mem.bytesAsSlice(u16, want_bytes);
        if (want.len != got.len) {
            std.debug.print("{s}: {d} features, 0.6.6 has {d}\n", .{ name, got.len, want.len });
            return 1;
        }
        var min_cos: f64 = 1;
        var sum_cos: f64 = 0;
        var max_err: f64 = 0;
        var equal: usize = 0;
        for (0..rows) |r| {
            var dot: f64 = 0;
            var na: f64 = 0;
            var nb: f64 = 0;
            for (0..V.out_hidden) |c| {
                const a = bf(got[r * V.out_hidden + c]);
                const b = bf(want[r * V.out_hidden + c]);
                if (got[r * V.out_hidden + c] == want[r * V.out_hidden + c]) equal += 1;
                dot += a * b;
                na += a * a;
                nb += b * b;
                max_err = @max(max_err, @abs(a - b));
            }
            const cs = dot / @sqrt(na * nb);
            min_cos = @min(min_cos, cs);
            sum_cos += cs;
        }
        worst = @min(worst, min_cos);
        std.debug.print("{s}: grid {d}x{d}, {d} rows in {d:.1} ms; row cosine min {d:.5} mean {d:.5}; max error {d:.4}; {d:.1}% values equal\n", .{ name, g.h, g.w, rows, ms, min_cos, sum_cos / @as(f64, @floatFromInt(rows)), max_err, 100 * @as(f64, @floatFromInt(equal)) / @as(f64, @floatFromInt(got.len)) });
    }
    return if (worst > 0.98) 0 else 1;
}

fn bf(v: u16) f64 {
    return @as(f32, @bitCast(@as(u32, v) << 16));
}

fn readJson(a: std.mem.Allocator, io: std.Io, path: []const u8) !std.json.Value {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 30));
    return (try std.json.parseFromSliceLeaky(std.json.Value, a, bytes, .{}));
}
