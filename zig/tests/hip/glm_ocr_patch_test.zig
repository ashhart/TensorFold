//! Real-image processor and native BF16 patch projection against the pinned independent reference.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const family = @import("glm_ocr");
const platform = @import("sys");
const c = family.config;

fn read(a: std.mem.Allocator, dir: []const u8, name: []const u8) ![]u8 {
    const path = try std.fs.path.join(a, &.{ dir, name });
    defer a.free(path);
    return std.Io.Dir.cwd().readFileAlloc(std.testing.io, path, a, .limited(256 << 20));
}

test "GLM-OCR HIP patch: real processor inputs and learned patch projection" {
    const a = std.testing.allocator;
    const dir = platform.getenv("TF_GLM_OCR_DIR") orelse return error.MissingCheckpointPath;
    const reference = platform.getenv("TF_GLM_OCR_REFERENCE") orelse return error.MissingReferencePath;
    const image = platform.getenv("TF_GLM_OCR_IMAGE") orelse return error.MissingImagePath;
    const bytes = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, image, a, .limited(64 << 20));
    defer a.free(bytes);
    var prepared = try family.processor.prepare(std.testing.io, a, bytes, 6144);
    defer prepared.deinit(a);
    const input_bytes = try read(a, reference, "input-pixel_values.npy");
    defer a.free(input_bytes);
    const inputs = try core.npy.parse(input_bytes);
    try std.testing.expectEqualStrings("<f4", inputs.descr);
    try std.testing.expectEqualSlices(u8, inputs.data, std.mem.sliceAsBytes(prepared.pixels));
    const patch_bytes = try read(a, reference, "vision-patch-bf16.npy");
    defer a.free(patch_bytes);
    const expected = try core.npy.parse(patch_bytes);
    try std.testing.expectEqualStrings("<u2", expected.descr);
    const rows = prepared.geometry.patches();
    try std.testing.expectEqual(rows * c.vision_hidden * 2, expected.data.len);
    var r = try hip.Runtime.open();
    defer r.close();
    if (try r.deviceCount() != 1) return error.SelectOneSharedHipDevice;
    var ctx = try hip.Context.init(&r, 0);
    defer ctx.deinit();
    const available = try ctx.memInfo();
    if (available.free < 4 << 30) return error.NoDeviceBudget;
    var stream = try hip.Stream.init(&r);
    defer stream.deinit();
    var blas = try hip.blas.Blas.open(stream);
    defer blas.close();
    var weights = try family.weights.Weights.load(a, &r, std.testing.io, dir, available.free - (512 << 20));
    defer weights.deinit();
    const input = try a.alloc(u16, prepared.pixels.len);
    defer a.free(input);
    for (input, prepared.pixels) |*dst, src| dst.* = core.quant.convert.bf16(src);
    var dx = try hip.DeviceBuffer.fromHost(&r, std.mem.sliceAsBytes(input));
    defer dx.free();
    const output = try a.alloc(u8, expected.data.len);
    defer a.free(output);
    var dy = try hip.DeviceBuffer.alloc(&r, output.len);
    defer dy.free();
    const w = try weights.get(family.schema.visual ++ "patch_embed.proj.weight");
    const bias = try weights.get(family.schema.visual ++ "patch_embed.proj.bias");
    var ops = try family.ops.Ops.load(&r);
    defer ops.unload();
    try blas.project(dx.base(), w, dy.base(), rows, c.vision_hidden, c.patch_values);
    try ops.addBias(stream, dy.base(), bias, rows, c.vision_hidden);
    try stream.synchronize();
    try dy.download(0, output);
    if (platform.getenv("TF_GLM_OCR_PATCH_OUTPUT")) |path|
        try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = path, .data = output });
    var mismatches: usize = 0;
    var max_abs: f32 = 0;
    for (std.mem.bytesAsSlice(u16, output), std.mem.bytesAsSlice(u16, expected.data)) |actual, want| {
        if (actual != want) mismatches += 1;
        const x: f32 = @bitCast(@as(u32, actual) << 16);
        const y: f32 = @bitCast(@as(u32, want) << 16);
        max_abs = @max(max_abs, @abs(x - y));
    }
    std.debug.print("GLM-OCR HIP patch: {d} rows, {d} BF16 mismatches, max_abs {d}\n", .{ rows, mismatches, max_abs });
    var ck = try family.checkpoint.Checkpoint.open(a, std.testing.io, dir);
    defer ck.close();
    const bias_tensor = try ck.get(family.schema.visual ++ "patch_embed.proj.bias");
    const control = try a.alloc(u8, output.len);
    defer a.free(control);
    for (0..rows) |row| @memcpy(control[row * bias_tensor.bytes.len ..][0..bias_tensor.bytes.len], bias_tensor.bytes);
    try dy.upload(0, control);
    try blas.projectAdd(dx.base(), w, dy.base(), rows, c.vision_hidden, c.patch_values);
    try stream.synchronize();
    try dy.download(0, control);
    const matrix = try ck.get(family.schema.visual ++ "patch_embed.proj.weight");
    try @import("glm_ocr_patch_reference.zig").check(input, std.mem.bytesAsSlice(u16, matrix.bytes), std.mem.bytesAsSlice(u16, bias_tensor.bytes), std.mem.bytesAsSlice(u16, output), std.mem.bytesAsSlice(u16, expected.data), std.mem.bytesAsSlice(u16, control));
    const shape = try hip.convolution.Shape.of(&.{ rows, 3, 2, 14, 14 }, &.{ 1024, 3, 2, 14, 14 }, &.{ 2, 14, 14 });
    const now = try ctx.memInfo();
    if (now.free < 512 << 20) return error.NoDeviceBudget;
    var conv = try hip.convolution.Convolution.init(stream, shape, dx.base(), w, dy.base(), now.free - (512 << 20));
    defer conv.deinit();
    for (0..5) |_| {
        try conv.run(dx.base(), w, dy.base());
        try ops.addBias(stream, dy.base(), bias, rows, c.vision_hidden);
        try stream.synchronize();
        try dy.download(0, output);
        mismatches = 0;
        for (std.mem.bytesAsSlice(u16, output), std.mem.bytesAsSlice(u16, expected.data)) |actual, want|
            if (actual != want) {
                mismatches += 1;
            };
        try std.testing.expectEqual(@as(usize, 0), mismatches);
    }
    std.debug.print("GLM-OCR HIP MIOpen {d}.{d}.{d} algorithm {d}: 5 repeats, 0 BF16 patch mismatches\n", .{ conv.version[0], conv.version[1], conv.version[2], conv.perf.algorithm });
}
