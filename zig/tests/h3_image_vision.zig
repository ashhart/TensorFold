//! tf-h3-image-vision: an edit reference's vision and DeepStack rows from Qwen3-VL's tower on Metal, or a selftest.
const std = @import("std");
const mtl = @import("metal");
const vl = @import("qwen_vl");
const copy = @import("h3_qwen_copy");
const fixture = @import("checkpoint_fixture");
const files = @import("h3_image_files.zig");
const selftest = @import("h3_image_vision_selftest.zig");

const usage =
    \\usage: tf-h3-image-vision TEXT_ENCODER_DIR MODEL_DIR reference.(png|jpg|webp) out.safetensors
    \\       tf-h3-image-vision selftest [WORK_DIR]
    \\  TEXT_ENCODER_DIR  MiniMax-H3's text_encoder/ (Qwen3-VL): config.json, its index and shards
    \\  MODEL_DIR         the MiniMax-H3-Image folder, for processor/preprocessor_config.json
    \\  reference         read upright in RGB, then its Qwen copy (short edge 1,024, LANCZOS) cut into patches
    \\  out               grid I32 [3] (t, h, w in patches), rows BF16 [tokens, 5120] and deepstack BF16
    \\                    [3, tokens, 5120], for tf-h3-image-prompt --reference
    \\  selftest          a tiny synthetic Qwen3-VL tower with DeepStack against a float64 host reference, gated;
    \\                    WORK_DIR (default under TMPDIR) holds its checkpoint while it runs
    \\Exit 0: done or every gate passed; 1: a gate failed; 2: bad arguments or input.
    \\
;

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    if (args.len >= 2 and std.mem.eql(u8, args[1], "selftest") and args.len <= 3) {
        const tmp = init.environ_map.get("TMPDIR") orelse "/tmp";
        const dir = if (args.len == 3) args[2] else try std.fs.path.join(a, &.{ tmp, "tf-h3-image-vision-selftest" });
        return selftest.run(init, a, dir);
    }
    if (args.len != 5) {
        std.debug.print("{s}", .{usage});
        return 2;
    }
    return run(init, a, args[1], args[2], args[3], args[4]);
}

pub fn read(io: std.Io, a: std.mem.Allocator, parts: []const []const u8) ![]u8 {
    const path = try std.fs.path.join(a, parts);
    return std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(64 << 20)) catch |err| {
        std.debug.print("cannot read {s}: {s}\n", .{ path, @errorName(err) });
        return err;
    };
}

/// The text encoder's tower with its DeepStack taps and the processor's settings, as a run and the selftest read them.
pub fn settings(io: std.Io, a: std.mem.Allocator, text_dir: []const u8, model_dir: []const u8) !vl.Config {
    const config = try read(io, a, &.{ text_dir, "config.json" });
    const processor = try read(io, a, &.{ model_dir, "processor", "preprocessor_config.json" });
    const c = (try vl.Config.parseDeepStack(a, config, processor, null)) orelse return error.NoVisionTower;
    if (c.taps().len != 3 or c.merge != copy.merge or c.factor() != copy.qwen_factor or c.temporal != 2) {
        std.debug.print("the vision tower is not Qwen3-VL's shape: {d} DeepStack taps, merge {d}, patch {d}\n", .{ c.taps().len, c.merge, c.patch });
        return error.UnsupportedVisionTower;
    }
    return c;
}

/// The processor's patches of a picture already at its Qwen copy's size, under the processor's own pixel bounds.
pub fn patches(a: std.mem.Allocator, c: vl.Config, rgb: vl.Rgb) !vl.patches.Prepared {
    const unit: u64 = c.factor();
    return vl.patches.prepareRgb(a, rgb, 0, c.patches(), @intCast(@min(c.max_pixels / (unit * unit), std.math.maxInt(u32))));
}

/// One tower pass: [1 + taps, tokens, out_hidden] bf16 on the host, the merged rows, then each tap's.
pub fn pass(device: mtl.Device, queue: mtl.Queue, tower: *vl.tower.Tower, a: std.mem.Allocator, p: vl.patches.Prepared, taps: usize) ![]u16 {
    const n = @as(usize, p.tokens) * tower.c.out_hidden;
    const part = std.mem.alignForward(usize, n * 2, 256);
    const buf = try device.buffer(part * (1 + taps), mtl.ResourceOptions.shared);
    defer buf.deinit();
    var deep: [vl.Config.max_deepstack]vl.tower.Ref = undefined;
    for (deep[0..taps], 1..) |*r, j| r.* = .{ .buf = buf, .off = j * part };
    try tower.imageDeep(queue, p.pixels, p.gh, p.gw, .{ .buf = buf }, deep[0..taps]);
    const out = try a.alloc(u16, (1 + taps) * n);
    for (0..1 + taps) |j| {
        const src: [*]const u16 = @ptrCast(@alignCast(buf.contents() + j * part));
        @memcpy(out[j * n ..][0..n], src[0..n]);
    }
    return out;
}

fn run(init: std.process.Init, a: std.mem.Allocator, text_dir: []const u8, model_dir: []const u8, picture_path: []const u8, out_path: []const u8) !u8 {
    const io = init.io;
    const c = try settings(io, a, text_dir, model_dir);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, picture_path, a, .limited(512 << 20));
    const rgb = vl.picture.decodeWithLimits(io, a, bytes, .{ .side = 1 << 15, .pixels = 1 << 26 }) catch |err| {
        std.debug.print("{s}: {s}\n", .{ picture_path, vl.picture.problem(err) });
        return 2;
    };
    const qwen = copy.qwenCopy(a, rgb) catch |err| {
        std.debug.print("{s}: {d}x{d} is refused: {s}\n", .{ picture_path, rgb.width, rgb.height, @errorName(err) });
        return 2;
    };
    const p = patches(a, c, qwen) catch |err| {
        std.debug.print("{s}: its Qwen copy is refused: {s}\n", .{ picture_path, vl.patches.problem(err) orelse @errorName(err) });
        return 2;
    };
    const processed = try copy.qwenProcessed(.{ .width = qwen.width, .height = qwen.height }, c.min_pixels, c.max_pixels);
    if (processed.width != p.gw * c.patch or processed.height != p.gh * c.patch) return error.ProcessorSize;

    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    const queue = try device.queue();
    const loading = mtl.clock.seconds();
    const prefix = try vl.shards.findPrefix(a, io, text_dir);
    const tower = try vl.tower.Tower.loadAt(init.gpa, io, device, text_dir, prefix, c, p.tokens, device);
    defer tower.deinit();
    std.debug.print("{s}: {s}* tower, {d} blocks, DeepStack after {any}; loaded with kernels in {d:.1} s\n", .{ device.name(), prefix, c.depth, c.taps(), mtl.clock.seconds() - loading });
    const started = mtl.clock.seconds();
    const out = try pass(device, queue, tower, a, p, 3);
    std.debug.print("{d}x{d} reference, Qwen copy {d}x{d}, grid {d} x {d} patches, {d} tokens: {d:.2} s\n", .{ rgb.width, rgb.height, qwen.width, qwen.height, p.gh, p.gw, p.tokens, mtl.clock.seconds() - started });
    const n = @as(usize, p.tokens) * c.out_hidden;
    const grid = try a.dupe(i32, &.{ 1, @intCast(p.gh), @intCast(p.gw) });
    const entries = [_]fixture.Entry{
        .{ .name = "grid", .dtype = "I32", .shape = &.{3}, .bytes = std.mem.sliceAsBytes(grid) },
        .{ .name = "rows", .dtype = "BF16", .shape = try a.dupe(usize, &.{ p.tokens, c.out_hidden }), .bytes = std.mem.sliceAsBytes(out[0..n]) },
        .{ .name = "deepstack", .dtype = "BF16", .shape = try a.dupe(usize, &.{ 3, p.tokens, c.out_hidden }), .bytes = std.mem.sliceAsBytes(out[n..]) },
    };
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = out_path, .data = try files.image(a, &entries) });
    std.debug.print("wrote {s}: grid, rows [{d}, {d}] and deepstack [3, {d}, {d}] bf16\n", .{ out_path, p.tokens, c.out_hidden, p.tokens, c.out_hidden });
    return 0;
}
