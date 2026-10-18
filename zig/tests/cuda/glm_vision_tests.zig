//! GLM's image tower on CUDA against vision_ref.py: the patches, and features as near float32 as torch's bf16 run.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const check = @import("check.zig");
const Gpu = check.Gpu;

fn readF32(gpu: Gpu, dir: []const u8, name: []const u8) ![]f32 {
    const path = try std.fs.path.join(gpu.gpa, &.{ dir, name });
    defer gpu.gpa.free(path);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(gpu.io, path, gpu.gpa, .limited(1 << 30));
    defer gpu.gpa.free(bytes);
    const out = try gpu.gpa.alloc(f32, bytes.len / 4);
    @memcpy(std.mem.sliceAsBytes(out), bytes[0 .. out.len * 4]);
    return out;
}

/// Root-mean-square of `want - got` over the rms of `want`, and the largest difference over that rms.
fn errors(want: []const f32, got: []const f32) [2]f64 {
    var d2: f64 = 0;
    var w2: f64 = 0;
    var most: f64 = 0;
    for (want, got) |w, g| {
        const d: f64 = @as(f64, g) - w;
        d2 += d * d;
        w2 += @as(f64, w) * w;
        most = @max(most, @abs(d));
    }
    const rms = @sqrt(w2 / @as(f64, @floatFromInt(want.len)));
    return .{ @sqrt(d2 / @as(f64, @floatFromInt(want.len))) / rms, most / rms };
}

pub fn run(gpu: Gpu, model: []const u8, case: []const u8, bf16_case: []const u8, image: []const u8) !void {
    const a = gpu.gpa;
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    const cwd = std.Io.Dir.cwd();
    const cfg_path = try std.fs.path.join(arena.allocator(), &.{ model, "config.json" });
    const proc_path = try std.fs.path.join(arena.allocator(), &.{ model, "processor_config.json" });
    const cfg = try cwd.readFileAlloc(gpu.io, cfg_path, arena.allocator(), .limited(1 << 24));
    const proc = try cwd.readFileAlloc(gpu.io, proc_path, arena.allocator(), .limited(1 << 24));
    const c = (try glm.vision_config.Config.parse(arena.allocator(), cfg, proc)) orelse return error.NoVisionTower;
    const want_patches = try readF32(gpu, case, "patches.f32");
    defer a.free(want_patches);
    const want = try readF32(gpu, case, "features.f32");
    defer a.free(want);
    const own = try readF32(gpu, bf16_case, "features.f32"); // the torch tower in bf16: its own distance from float32
    defer a.free(own);
    const bound = errors(want, own);

    // preparation: core's decoder and resize, the processor's canvas and merge order
    const bytes = try cwd.readFileAlloc(gpu.io, image, arena.allocator(), .limited(64 << 20));
    const p = try glm.image.prepare(gpu.io, a, bytes, c, c.max_tokens);
    defer a.free(p.pixels);
    try check.expect(p.pixels.len == want_patches.len, "glm-vision: {d} patch values, the reference {d}", .{ p.pixels.len, want_patches.len });
    const pe = errors(want_patches, p.pixels);
    std.debug.print("RESULT glm-vision patches {d} x {d} ({d} tokens): rms difference {e:.2} of their rms, largest {e:.2}\n", .{ p.gh, p.gw, p.tokens, pe[0], pe[1] });
    try check.expect(pe[1] < 0.02, "glm-vision: the patches as the processor makes them", .{});

    // the tower on the reference's patches
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var mod = try cuda.Module.load(gpu.d, cuda.kernels.glm_vision);
    defer mod.unload();
    const ru: glm.vision.Run = .{ .stream = s, .f = try glm.vision.Funcs.load(mod) };
    var t = try glm.vision.Tower.load(a, gpu.io, gpu.d, model);
    defer t.deinit();
    var out = try cuda.DeviceBuffer.alloc(gpu.d, @as(usize, p.tokens) * glm.vision_config.out_hidden * 2);
    defer out.free();
    const grid = [3]i64{ 1, p.gh, p.gw };
    { // where the stream parts from the reference's: after the embedding, then each block
        const all = readF32(gpu, case, "all.f32") catch null;
        const embed = readF32(gpu, case, "embed.f32") catch null;
        if (all != null and embed != null) {
            defer a.free(all.?);
            defer a.free(embed.?);
            const n = embed.?.len;
            const got = try a.alloc(f32, (glm.vision.DEPTH + 1) * n);
            defer a.free(got);
            t.stages = got;
            try t.encode(&ru, want_patches, grid, out.ptr);
            t.stages = null;
            const e0 = errors(embed.?, got[0..n]);
            std.debug.print("STAGE embed: rms {d:.4}\n", .{e0[0]});
            for (0..glm.vision.DEPTH) |b| {
                const e = errors(all.?[b * n ..][0..n], got[(b + 1) * n ..][0..n]);
                std.debug.print("STAGE block {d}: rms {d:.4}, largest {d:.3}\n", .{ b, e[0], e[1] });
            }
        }
    }
    for ([_][]const f32{ want_patches, p.pixels }, [_][]const u8{ "the reference's patches", "its own patches" }) |patches, what| {
        try t.encode(&ru, patches, grid, out.ptr);
        const raw = try a.alloc(u16, @as(usize, p.tokens) * glm.vision_config.out_hidden);
        defer a.free(raw);
        try out.download(0, std.mem.sliceAsBytes(raw));
        const got = try a.alloc(f32, raw.len);
        defer a.free(got);
        for (got, raw) |*g, r| g.* = @bitCast(@as(u32, r) << 16);
        const fe = errors(want, got);
        std.debug.print("RESULT glm-vision features from {s}: rms difference {d:.4} of their rms ({d:.3}x the torch bf16 run's {d:.4}), largest {d:.3}\n", .{ what, fe[0], fe[0] / bound[0], bound[0], fe[1] });
        try check.expect(fe[0] <= 1.1 * bound[0], "glm-vision: features as close to float32 as the torch tower's bf16 run", .{});
    }
    check.pass("glm-vision: the processor's patches as the reference's, the features as near float32 as torch's bf16 run", .{});
}
