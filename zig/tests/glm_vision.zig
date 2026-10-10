//! GLM-5.3-Flash's image tower against a reference dump (tools/zig/glm_vision_fixtures.py: mlx-vlm's tower).
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const vision = tf.glm.vision;

const usage =
    \\tf-glm-vision MODEL_DIR REF_DIR [STAGE] [IMAGE]
    \\REF_DIR holds meta.json (grid_thw), patches.f32 and the stage's reference: embed.f32 (STAGE embed, after the patch
    \\projection), block0.f32 (STAGE block0, after the first block), blocks.f32 (STAGE blocks, after the last) or
    \\features.f32 (STAGE features, the default: the tower's output rows). Writes native_STAGE.f32 there and prints the
    \\largest and mean absolute differences, the reference's mean magnitude and the rows' cosine similarity.
    \\IMAGE: prepare the patches from this file natively (decode, resize, pad, normalize), compare them with
    \\patches.f32 value by value, and run the tower on them instead (meta.json's max_tokens is the budget).
;

fn readFile(a: std.mem.Allocator, dir: []const u8, name: []const u8) ![]u8 {
    const path = try std.fmt.allocPrintSentinel(a, "{s}/{s}", .{ dir, name }, 0);
    const f = try mtl.MappedFile.open(path);
    defer f.deinit();
    return a.dupe(u8, f.bytes[0..f.size]);
}

fn bf16ToF32(x: u16) f32 {
    return @bitCast(@as(u32, x) << 16);
}

pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    if (args.len < 3) {
        std.debug.print("{s}\n", .{usage});
        return error.Usage;
    }
    const dir = args[1];
    const ref = args[2];
    const stage = if (args.len > 3) args[3] else "features";
    const stop: ?u32 = if (std.mem.eql(u8, stage, "embed")) 0 else if (std.mem.eql(u8, stage, "block0")) 1 else if (std.mem.eql(u8, stage, "blocks")) vision.depth else if (std.mem.eql(u8, stage, "features")) null else return error.Usage;

    const meta = try std.json.parseFromSliceLeaky(std.json.Value, a, try readFile(a, ref, "meta.json"), .{});
    const grid = meta.object.get("grid_thw").?.array.items;
    const gh: u32 = @intCast(grid[1].integer);
    const gw: u32 = @intCast(grid[2].integer);
    const raw = try readFile(a, ref, "patches.f32");
    var pixels = try a.alloc(f32, raw.len / 4);
    @memcpy(std.mem.sliceAsBytes(pixels), raw[0 .. pixels.len * 4]);
    const P = gh * gw;
    const T = P / vision.patches_per_token;

    const device = try mtl.Device.init();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    const cfg_text = try readFile(a, dir, "config.json");
    const proc_text = readFile(a, dir, "processor_config.json") catch null;
    const c = (try vision.Config.parse(a, cfg_text, proc_text)) orelse return error.NoVisionConfig;
    if (args.len > 4) {
        const img_path = try std.fmt.allocPrintSentinel(a, "{s}", .{args[4]}, 0);
        const f = try mtl.MappedFile.open(img_path);
        defer f.deinit();
        const max_tokens: u32 = @intCast(meta.object.get("max_tokens").?.integer);
        const t0i = std.Io.Clock.awake.now(init.io).toNanoseconds();
        const prep = try tf.glm.image.prepare(a, f.bytes[0..f.size], c, max_tokens);
        const t1i = std.Io.Clock.awake.now(init.io).toNanoseconds();
        std.debug.print("prepared {s}: grid {d}x{d}, {d} tokens in {d:.1} ms\n", .{ args[4], prep.gh, prep.gw, prep.tokens, @as(f64, @floatFromInt(t1i - t0i)) / 1e6 });
        if (prep.gh != gh or prep.gw != gw) {
            std.debug.print("grid differs from the reference's {d}x{d}\n", .{ gh, gw });
            return error.GridMismatch;
        }
        var differ: usize = 0;
        var max_p: f32 = 0;
        for (prep.pixels, pixels) |x, y| {
            if (x != y) differ += 1;
            max_p = @max(max_p, @abs(x - y));
        }
        std.debug.print("patches: {d} of {d} values differ from the reference (largest {d:.6})\n", .{ differ, pixels.len, max_p });
        pixels = prep.pixels;
    }
    var t0 = std.Io.Clock.awake.now(init.io).toNanoseconds();
    const v = try vision.Vision.load(std.heap.page_allocator, device, dir, c, @max(T, 1024));
    defer v.deinit();
    var t1 = std.Io.Clock.awake.now(init.io).toNanoseconds();
    std.debug.print("tower loaded in {d:.2} s ({d:.2} GiB with scratch for {d} patches, attention chunks of {d} rows)\n", .{ @as(f64, @floatFromInt(t1 - t0)) / 1e9, @as(f64, @floatFromInt(v.bytes)) / (1 << 30), v.max_patches, v.chunk });
    const out_buf = try device.buffer(@as(usize, T) * vision.out_hidden * 2, mtl.ResourceOptions.shared);
    defer out_buf.deinit();

    for (0..2) |run| { // the second run is the timed one (the first compiles pipelines' state on the GPU)
        const cb = queue.commandBuffer();
        const enc = cb.compute(.serial);
        t0 = std.Io.Clock.awake.now(init.io).toNanoseconds();
        try v.encode(enc, pixels, gh, gw, .{ .buf = out_buf }, stop);
        enc.end();
        cb.commit();
        cb.wait();
        t1 = std.Io.Clock.awake.now(init.io).toNanoseconds();
        if (cb.failure()) |msg| {
            std.debug.print("command buffer failed: {s}\n", .{msg});
            return error.GpuFailed;
        }
        if (run == 1) std.debug.print("encode: {d} patches -> {d} tokens in {d:.1} ms\n", .{ P, T, @as(f64, @floatFromInt(t1 - t0)) / 1e6 });
    }
    // the stage's rows: the residual stream for a stop, else the output rows
    const width: usize = if (stop != null) vision.hidden else vision.out_hidden;
    const rows: usize = if (stop != null) P else T;
    const src: [*]const u16 = if (stop != null) @ptrCast(@alignCast(v.x.addr())) else @ptrCast(@alignCast(out_buf.contents()));
    const got = try a.alloc(f32, rows * width);
    for (got, 0..) |*g, i| g.* = bf16ToF32(src[i]);
    const name = try std.fmt.allocPrint(a, "native_{s}.f32", .{stage});
    {
        const path = try std.fmt.allocPrint(a, "{s}/{s}", .{ ref, name });
        const f = try std.Io.Dir.cwd().createFile(init.io, path, .{});
        defer f.close(init.io);
        try f.writeStreamingAll(init.io, std.mem.sliceAsBytes(got));
    }
    const want_raw = try readFile(a, ref, try std.fmt.allocPrint(a, "{s}.f32", .{stage}));
    const want = try a.alloc(f32, want_raw.len / 4);
    @memcpy(std.mem.sliceAsBytes(want), want_raw[0 .. want.len * 4]);
    if (want.len != got.len) {
        std.debug.print("size: native {d} values, reference {d}\n", .{ got.len, want.len });
        return error.SizeMismatch;
    }
    var max_d: f64 = 0;
    var sum_d: f64 = 0;
    var sum_m: f64 = 0;
    var worst_cos: f64 = 1;
    var mean_cos: f64 = 0;
    for (0..rows) |r| {
        var dot: f64 = 0;
        var ng: f64 = 0;
        var nw: f64 = 0;
        for (0..width) |j| {
            const g: f64 = got[r * width + j];
            const w: f64 = want[r * width + j];
            const d = @abs(g - w);
            max_d = @max(max_d, d);
            sum_d += d;
            sum_m += @abs(w);
            dot += g * w;
            ng += g * g;
            nw += w * w;
        }
        const cs = dot / (@sqrt(ng * nw) + 1e-30);
        worst_cos = @min(worst_cos, cs);
        mean_cos += cs;
    }
    const n: f64 = @floatFromInt(got.len);
    std.debug.print("{s}: max |d| {d:.4}, mean |d| {d:.5}, mean |ref| {d:.4}, cosine mean {d:.6} worst {d:.6} ({d} rows of {d})\n", .{ stage, max_d, sum_d / n, sum_m / n, mean_cos / @as(f64, @floatFromInt(rows)), worst_cos, rows, width });
}
