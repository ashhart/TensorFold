//! Gemma 4's image tower against tools/zig/gemma4_vision_fixtures.py's dumps: patches bit for bit, then each stage.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const g = tf.gemma4;
const vision = g.vision;
const ck = tf.checkpoint;
const shards = tf.checkpoint_host;
const affine_mm = tf.affine_mm;

const usage =
    \\tf-gemma4-vision MODEL_DIR REF_DIR...
    \\Each REF_DIR (one image's dump) holds meta.json, patches.f32 and the stages' references in bf16 and fp32 (NAME.f32,
    \\NAME_fp32.f32): embed (the patch rows), block0, blocks, pooled and features (the tower's output rows). Prepares the
    \\image natively and compares its patches with patches.f32 (the PIL processor's), then runs each stage on patches.f32
    \\and prints its relative error and worst row cosine against the fp32 run beside torch's own bf16 run's. A stage fails
    \\past 1.1 times torch's error on one image (bf16 drift over 27 layers moves either run by that much), and the
    \\features fail when their error averaged over the images is past torch's average.
;

const stages = [_]struct { name: []const u8, stop: ?u32, width: usize, pooled: bool }{
    .{ .name = "embed", .stop = 0, .width = vision.width, .pooled = false },
    .{ .name = "block0", .stop = 1, .width = vision.width, .pooled = false },
    .{ .name = "blocks", .stop = vision.depth, .width = vision.width, .pooled = false },
    .{ .name = "pooled", .stop = null, .width = vision.width, .pooled = true },
    .{ .name = "features", .stop = null, .width = vision.out_hidden, .pooled = false },
};

fn readFile(a: std.mem.Allocator, dir: []const u8, name: []const u8) ![]u8 {
    const path = try std.fmt.allocPrintSentinel(a, "{s}/{s}", .{ dir, name }, 0);
    const f = try mtl.MappedFile.open(path);
    defer f.deinit();
    return a.dupe(u8, f.bytes[0..f.size]);
}

fn floats(a: std.mem.Allocator, dir: []const u8, name: []const u8) ![]f32 {
    const raw = try readFile(a, dir, name);
    const out = try a.alloc(f32, raw.len / 4);
    @memcpy(std.mem.sliceAsBytes(out), raw[0 .. out.len * 4]);
    return out;
}

/// ||got - want|| / ||want|| and the worst row's cosine, rows `width` wide.
fn compare(got: []const f32, want: []const f32, width: usize) [2]f64 {
    var d2: f64 = 0;
    var w2: f64 = 0;
    var worst: f64 = 1;
    for (0..want.len / width) |r| {
        var dot: f64 = 0;
        var ng: f64 = 0;
        var nw: f64 = 0;
        for (got[r * width ..][0..width], want[r * width ..][0..width]) |x, y| {
            d2 += (x - y) * (x - y);
            w2 += y * y;
            dot += x * y;
            ng += x * x;
            nw += y * y;
        }
        worst = @min(worst, dot / (@sqrt(ng * nw) + 1e-30));
    }
    return .{ @sqrt(d2 / w2), worst };
}

pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const gpa = std.heap.page_allocator;
    const args = try init.minimal.args.toSlice(a);
    if (args.len < 3) {
        std.debug.print("{s}\n", .{usage});
        return error.Usage;
    }
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    const queue = try device.queue();
    var c = ck.Checkpoint.init(gpa);
    defer c.deinit();
    const files = try shards.shardFiles(gpa, init.io, args[1]);
    defer shards.freeShardFiles(gpa, files);
    try vision.addTensors(&c, device, files);
    var mm: affine_mm.Pipes = undefined;
    {
        const text = try affine_mm.source(device, gpa, .{});
        defer gpa.free(text);
        const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
        defer lib.deinit();
        for (&mm, affine_mm.names) |*p, name| p.* = try mtl.Pipeline.init(device, lib, name, false);
    }
    defer for (mm) |p| p.deinit();
    const v = try vision.Vision.load(gpa, device, &c, mm);
    defer v.deinit();
    const out = try device.buffer(vision.max_patches / 9 * vision.out_hidden * 2, mtl.ResourceOptions.shared);
    defer out.deinit();
    const back = try device.buffer(vision.max_patches * vision.width * 2, mtl.ResourceOptions.shared);
    defer back.deinit();
    var failed = false;
    var sums: [2]f64 = .{ 0, 0 }; // the features' error, native and torch bf16, summed over the images
    for (args[2..]) |ref| {
        const meta = try std.json.parseFromSliceLeaky(std.json.Value, a, try readFile(a, ref, "meta.json"), .{});
        const image_path = try std.fmt.allocPrintSentinel(a, "{s}", .{meta.object.get("image").?.string}, 0);
        const f = try mtl.MappedFile.open(image_path);
        defer f.deinit();
        const max_tokens: u32 = @intCast(meta.object.get("max_tokens").?.integer);
        const prep = try g.image.prepare(a, f.bytes[0..f.size], max_tokens);
        const want_patches = try floats(a, ref, "patches.f32");
        var differ: usize = 0;
        var largest: f32 = 0;
        if (want_patches.len != prep.pixels.len) {
            std.debug.print("{s}: native grid {d}x{d} has {d} values, the reference {d}\n", .{ ref, prep.gw, prep.gh, prep.pixels.len, want_patches.len });
            failed = true;
            continue;
        }
        for (prep.pixels, want_patches) |x, y| {
            if (x != y) differ += 1;
            largest = @max(largest, @abs(x - y));
        }
        std.debug.print("{s}: grid {d}x{d}, {d} tokens; patches: {d} of {d} values differ (largest {d:.6})\n", .{ std.fs.path.basename(ref), prep.gw, prep.gh, prep.tokens, differ, want_patches.len, largest });
        if (differ != 0) failed = true;
        const P = prep.gh * prep.gw;
        for (stages) |s| {
            const cb = queue.commandBuffer();
            const e = cb.compute(.serial);
            const t0 = std.Io.Clock.awake.now(init.io).toNanoseconds();
            try v.encode(e, want_patches, prep.gh, prep.gw, .{ .buf = out }, s.stop); // the reference's input: the tower's error alone
            e.end();
            const rows: usize = if (s.stop != null) P else prep.tokens;
            const from = if (s.stop != null) v.x else if (s.pooled) v.pooled else vision.Ref{ .buf = out };
            if (from.buf.id != out.id) {
                const b = cb.blit();
                b.copy(from.buf, from.off, back, 0, rows * s.width * 2);
                b.end();
            }
            cb.commit();
            cb.wait();
            if (cb.failure()) |msg| {
                std.debug.print("command buffer failed: {s}\n", .{msg});
                return error.GpuFailed;
            }
            const ms = @as(f64, @floatFromInt(std.Io.Clock.awake.now(init.io).toNanoseconds() - t0)) / 1e6;
            const src: [*]const u16 = @ptrCast(@alignCast(if (from.buf.id != out.id) back.contents() else out.contents()));
            const got = try a.alloc(f32, rows * s.width);
            for (got, 0..) |*x, i| x.* = @bitCast(@as(u32, src[i]) << 16);
            const want = try floats(a, ref, try std.fmt.allocPrint(a, "{s}_fp32.f32", .{s.name}));
            const torch = try floats(a, ref, try std.fmt.allocPrint(a, "{s}.f32", .{s.name}));
            if (want.len != got.len) {
                std.debug.print("  {s}: native {d} values, reference {d}\n", .{ s.name, got.len, want.len });
                failed = true;
                continue;
            }
            const n = compare(got, want, s.width);
            const t = compare(torch, want, s.width);
            const ok = n[0] <= t[0] * 1.1;
            if (s.stop == null and !s.pooled) {
                sums[0] += n[0];
                sums[1] += t[0];
            }
            if (!ok) failed = true;
            std.debug.print("  {s:<8} vs fp32: native rel {d:.4} worst cos {d:.4} | torch bf16 rel {d:.4} worst cos {d:.4}  {s}  ({d:.1} ms)\n", .{ s.name, n[0], n[1], t[0], t[1], if (ok) "ok" else "WORSE", ms });
        }
    }
    const images: f64 = @floatFromInt(args.len - 2);
    const mean_ok = sums[0] <= sums[1];
    std.debug.print("features averaged over {d} images: native rel {d:.4} | torch bf16 rel {d:.4}  {s}\n", .{ args.len - 2, sums[0] / images, sums[1] / images, if (mean_ok) "ok" else "WORSE" });
    if (failed or !mean_ok) return error.GemmaVisionMismatch;
}
