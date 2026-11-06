//! Native CogViT and connector comparison against separately captured official-model stages.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const family = @import("glm_ocr");
const platform = @import("sys");

const Capture = @import("glm_ocr_capture.zig").Capture;

test "GLM-OCR HIP vision: native CogViT and connector learned from the unchanged checkpoint" {
    const a = std.testing.allocator;
    const checkpoint = platform.getenv("TF_GLM_OCR_DIR") orelse return error.MissingCheckpointPath;
    const reference = platform.getenv("TF_GLM_OCR_REFERENCE") orelse return error.MissingReferencePath;
    const image = platform.getenv("TF_GLM_OCR_IMAGE") orelse return error.MissingImagePath;
    const bytes = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, image, a, .limited(64 << 20));
    defer a.free(bytes);
    var prepared = try family.processor.prepare(std.testing.io, a, bytes, 6144);
    defer prepared.deinit(a);
    {
        const path = try std.fs.path.join(a, &.{ reference, "input-pixel_values.npy" });
        defer a.free(path);
        const source = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, path, a, .limited(256 << 20));
        defer a.free(source);
        const pixels = try core.npy.parse(source);
        try std.testing.expectEqualStrings("<f4", pixels.descr);
        try std.testing.expectEqualSlices(u8, pixels.data, std.mem.sliceAsBytes(prepared.pixels));
    }
    var r = try hip.Runtime.open();
    defer r.close();
    if (try r.deviceCount() != 1) return error.SelectOneSharedHipDevice;
    var ctx = try hip.Context.init(&r, 0);
    defer ctx.deinit();
    const free = (try ctx.memInfo()).free;
    if (free < 5 << 30) return error.NoDeviceBudget;
    var stream = try hip.Stream.init(&r);
    defer stream.deinit();
    var blas = try hip.blas.Blas.open(stream);
    defer blas.close();
    var flash = try hip.attention_aotriton.Library.open(stream);
    defer flash.close(stream);
    var weights = try family.weights.Weights.load(a, &r, std.testing.io, checkpoint, free - (2 << 30));
    defer weights.deinit();
    var ops = try family.ops.Ops.load(&r);
    defer ops.unload();
    const pixels = try a.alloc(u16, prepared.pixels.len);
    defer a.free(pixels);
    for (pixels, prepared.pixels) |*out, x| out.* = core.quant.convert.bf16(x);
    var input = try hip.DeviceBuffer.fromHost(&r, std.mem.sliceAsBytes(pixels));
    defer input.free();
    const rows = prepared.geometry.patches();
    var output = try hip.DeviceBuffer.alloc(&r, rows / 4 * 1536 * 2);
    defer output.free();
    const held = hip.usage(false).device;
    {
        var vision = try family.vision.Vision.init(stream, &weights, &ops, &blas, &flash, rows, prepared.geometry.w / 14, (try ctx.memInfo()).free -| (512 << 20));
        defer vision.deinit();
        for (0..5) |repeat| {
            var capture: Capture = .{ .a = a, .reference = reference, .stream = stream, .quiet = repeat != 0, .required_names = &.{ "vision-patch", "vision-block0", "vision-last", "vision-norm", "vision-downsample", "vision-connector" }, .final_name = "vision-connector", .output_root = platform.getenv("TF_GLM_OCR_VISION_OUTPUT") };
            try vision.encode(input.base(), output.base(), .{ .context = &capture, .capture = Capture.tap });
            try stream.synchronize();
            try std.testing.expect(capture.compared >= 6);
            try std.testing.expectEqual(@as(usize, 0), capture.final_mismatches orelse return error.MissingConnectorReference);
            try std.testing.expectEqual(@as(usize, 0), capture.total_mismatches);
        }
    }
    try std.testing.expectEqual(held, hip.usage(false).device);
    std.debug.print("GLM-OCR HIP vision: 5 repeats, 0 BF16 differences, all owned scratch released\n", .{});
}
