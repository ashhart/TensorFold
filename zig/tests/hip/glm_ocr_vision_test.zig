//! Native CogViT and connector comparison against separately captured official-model stages.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const family = @import("glm_ocr");
const platform = @import("sys");

const Capture = struct {
    a: std.mem.Allocator,
    reference: []const u8,
    stream: hip.Stream,
    final_mismatches: ?usize = null,
    compared: usize = 0,
    total_mismatches: usize = 0,
    quiet: bool = false,
    fn tap(context: *anyopaque, name: []const u8, address: u64, values: usize) !void {
        const s: *Capture = @ptrCast(@alignCast(context));
        var path_buf: [1024]u8 = undefined;
        const path = try std.fmt.bufPrint(&path_buf, "{s}/{s}-bf16.npy", .{ s.reference, name });
        const bytes = std.Io.Dir.cwd().readFileAlloc(std.testing.io, path, s.a, .limited(256 << 20)) catch |err| switch (err) {
            error.FileNotFound => {
                inline for (.{ "vision-patch", "vision-block0", "vision-last", "vision-norm", "vision-downsample", "vision-connector" }) |required| {
                    if (std.mem.eql(u8, name, required)) return error.MissingVisionReference;
                }
                return;
            },
            else => return err,
        };
        defer s.a.free(bytes);
        try s.stream.synchronize();
        const out = try s.a.alloc(u16, values);
        defer s.a.free(out);
        const device: hip.DeviceBuffer = .{ .r = s.stream.r, .ptr = @ptrFromInt(address), .len = values * 2 };
        try device.download(0, std.mem.sliceAsBytes(out));
        if (!s.quiet) {
            if (platform.getenv("TF_GLM_OCR_VISION_OUTPUT")) |root| {
                const output_path = try std.fmt.bufPrint(&path_buf, "{s}/{s}.bf16", .{ root, name });
                try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = output_path, .data = std.mem.sliceAsBytes(out) });
            }
        }
        const expected = try core.npy.parse(bytes);
        try std.testing.expectEqualStrings("<u2", expected.descr);
        try std.testing.expectEqual(values * 2, expected.data.len);
        var mismatches: usize = 0;
        var max_abs: f32 = 0;
        for (out, std.mem.bytesAsSlice(u16, expected.data)) |x, y| {
            if (x != y) mismatches += 1;
            const xf: f32 = @bitCast(@as(u32, x) << 16);
            const yf: f32 = @bitCast(@as(u32, y) << 16);
            if (!std.math.isFinite(xf) or !std.math.isFinite(yf)) return error.NonFiniteVisionStage;
            max_abs = @max(max_abs, @abs(xf - yf));
        }
        if (!s.quiet or mismatches != 0)
            std.debug.print("{s}: {d}/{d} BF16 differences, max_abs {d}\n", .{ name, mismatches, values, max_abs });
        s.compared += 1;
        s.total_mismatches += mismatches;
        if (std.mem.eql(u8, name, "vision-connector")) s.final_mismatches = mismatches;
    }
};

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
        var vision = try family.vision.Vision.init(stream, &weights, &ops, &blas, &flash, rows, prepared.geometry.w / 14, (try ctx.memInfo()).free - (512 << 20));
        defer vision.deinit();
        for (0..5) |repeat| {
            var capture: Capture = .{ .a = a, .reference = reference, .stream = stream, .quiet = repeat != 0 };
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
