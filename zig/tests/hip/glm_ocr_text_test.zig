//! Native image/text rows, decoder prefill and cached decoding against independent model captures.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const family = @import("glm_ocr");
const platform = @import("sys");
const Capture = @import("glm_ocr_capture.zig").Capture;
const c = family.config;
const a = std.testing.allocator;

fn read(dir: []const u8, name: []const u8) ![]u8 {
    const path = try std.fs.path.join(a, &.{ dir, name });
    defer a.free(path);
    return std.Io.Dir.cwd().readFileAlloc(std.testing.io, path, a, .limited(256 << 20));
}
fn integer(array: core.npy.Array, i: usize) !i64 {
    if (!std.mem.eql(u8, array.descr, "<i8") or (i + 1) * 8 > array.data.len) return error.BadReferenceInteger;
    return std.mem.readInt(i64, array.data[i * 8 ..][0..8], .little);
}

fn compareLogits(stream: hip.Stream, address: u64, reference: []const u8, step: usize) !u32 {
    try stream.synchronize();
    const values = try a.alloc(u16, c.vocab);
    defer a.free(values);
    const device: hip.DeviceBuffer = .{ .r = stream.r, .ptr = @ptrFromInt(address), .len = c.vocab * 2 };
    try device.download(0, std.mem.sliceAsBytes(values));
    var name: [64]u8 = undefined;
    const bytes = try read(reference, try std.fmt.bufPrint(&name, "step-{d}-logits.npy", .{step}));
    defer a.free(bytes);
    const expected = try core.npy.parse(bytes);
    try std.testing.expectEqualStrings("<f4", expected.descr);
    try std.testing.expectEqual(c.vocab * 4, expected.data.len);
    var mismatches: usize = 0;
    var max_abs: f32 = 0;
    var actual_top: u32 = 0;
    var expected_top: u32 = 0;
    var actual_max: f32 = -std.math.inf(f32);
    var expected_max: f32 = -std.math.inf(f32);
    for (values, 0..) |bits, i| {
        const x: f32 = @bitCast(@as(u32, bits) << 16);
        const y: f32 = @bitCast(std.mem.readInt(u32, expected.data[i * 4 ..][0..4], .little));
        if (!std.math.isFinite(x) or !std.math.isFinite(y)) return error.NonFiniteLogits;
        if (x != y) mismatches += 1;
        max_abs = @max(max_abs, @abs(x - y));
        if (x > actual_max) {
            actual_max = x;
            actual_top = @intCast(i);
        }
        if (y > expected_max) {
            expected_max = y;
            expected_top = @intCast(i);
        }
    }
    std.debug.print("text logits step {d}: {d}/{d} differences, max_abs {d}, top {d}/{d}\n", .{ step, mismatches, c.vocab, max_abs, actual_top, expected_top });
    try std.testing.expectEqual(expected_top, actual_top);
    try std.testing.expectEqual(@as(usize, 0), mismatches);
    return actual_top;
}

test "GLM-OCR HIP text: native rows, decoder states and every generated logit match the reference" {
    const checkpoint = platform.getenv("TF_GLM_OCR_DIR") orelse return error.MissingCheckpointPath;
    const reference = platform.getenv("TF_GLM_OCR_REFERENCE") orelse return error.MissingReferencePath;
    const embed_file = try read(reference, "text-input-bf16.npy");
    defer a.free(embed_file);
    const embedding = try core.npy.parse(embed_file);
    try std.testing.expectEqualStrings("<u2", embedding.descr);
    const rows = embedding.data.len / (c.hidden * 2);
    try std.testing.expect(rows > 0 and rows < c.context - 256 and embedding.data.len == rows * c.hidden * 2);
    const position_file = try read(reference, "text-positions.npy");
    defer a.free(position_file);
    const position_ref = try core.npy.parse(position_file);
    const axes = position_ref.data.len / (rows * 8);
    try std.testing.expect((axes == 3 or axes == 4) and position_ref.data.len == rows * axes * 8);
    if (axes == 4) for (0..rows) |i| try std.testing.expectEqual(@as(i64, @intCast(i)), try integer(position_ref, i));
    const positions = try a.alloc([3]i32, rows);
    defer a.free(positions);
    for (positions, 0..) |*row, i| for (0..3) |axis| {
        const value = try integer(position_ref, (axis + axes - 3) * rows + i);
        if (value < 0 or value > c.context) return error.BadReferencePosition;
        row[axis] = @intCast(value);
    };
    const id_file = try read(reference, "input-input_ids.npy");
    defer a.free(id_file);
    const ids_ref = try core.npy.parse(id_file);
    try std.testing.expectEqual(rows * 8, ids_ref.data.len);
    const ids = try a.alloc(u32, rows);
    defer a.free(ids);
    for (ids, 0..) |*id, i| {
        const value = try integer(ids_ref, i);
        if (value < 0 or value >= c.vocab) return error.BadReferenceToken;
        id.* = @intCast(value);
    }
    const grid_file = try read(reference, "input-image_grid_thw.npy");
    defer a.free(grid_file);
    const grid_ref = try core.npy.parse(grid_file);
    const geometry: family.processor.Geometry = .{ .h = @as(usize, @intCast(try integer(grid_ref, 1))) * 14, .w = @as(usize, @intCast(try integer(grid_ref, 2))) * 14 };
    const host_positions = try a.alloc([3]i32, rows);
    defer a.free(host_positions);
    const rope = try family.positions.fill(ids, &.{geometry}, host_positions);
    try std.testing.expectEqualSlices([3]i32, positions, host_positions);
    const inverse_file = try read(reference, "text-inverse.npy");
    defer a.free(inverse_file);
    const inverse_ref = try core.npy.parse(inverse_file);
    const inverse = family.decoder.inverseFrequencies();
    try std.testing.expectEqualSlices(u8, inverse_ref.data, std.mem.asBytes(&inverse));
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
    var ops = try family.ops.Ops.load(&r);
    defer ops.unload();
    var weights = try family.weights.Weights.load(a, &r, std.testing.io, checkpoint, free - (512 << 20));
    defer weights.deinit();
    var input = try hip.DeviceBuffer.fromHost(&r, embedding.data);
    defer input.free();
    var dp = try hip.DeviceBuffer.fromHost(&r, std.mem.sliceAsBytes(positions));
    defer dp.free();
    var cache = try family.decoder.Cache.init(stream, rows + 256, (try ctx.memInfo()).free -| (256 << 20));
    defer cache.deinit();
    var graph = try family.decoder.Graph.init(stream, &weights, &ops, &blas, &flash, rows, (try ctx.memInfo()).free -| (256 << 20));
    defer graph.deinit();
    var own_input: ?family.input.Input = null;
    defer if (own_input) |*native| native.deinit();
    if (platform.getenv("TF_GLM_OCR_NATIVE_INPUT") != null) {
        const image_path = platform.getenv("TF_GLM_OCR_IMAGE") orelse return error.MissingImagePath;
        const image_bytes = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, image_path, a, .limited(64 << 20));
        defer a.free(image_bytes);
        var image = try family.processor.prepare(std.testing.io, a, image_bytes, 6144);
        defer image.deinit(a);
        try std.testing.expectEqualDeep(geometry, image.geometry);
        own_input = try .init(a, &graph, ids, &.{image}, (try ctx.memInfo()).free -| (512 << 20));
        const mixed = try a.alloc(u8, embedding.data.len);
        defer a.free(mixed);
        try own_input.?.buffer.download(0, mixed);
        try std.testing.expectEqualSlices(u8, embedding.data, mixed);
        try std.testing.expectEqualDeep(rope, own_input.?.rope);
        std.debug.print("native image/text assembly: {d} rows, 0 BF16 differences\n", .{rows});
    }
    var capture: Capture = .{ .a = a, .reference = reference, .stream = stream, .required_names = &.{ "text-block0", "text-last", "text-norm" }, .final_name = "text-norm", .output_root = platform.getenv("TF_GLM_OCR_TEXT_OUTPUT") };
    const normed = try graph.forward(&cache, if (own_input) |*native| native.embeddings() else input.base(), if (own_input) |*native| try native.rotary() else dp.base(), rows, .{ .context = &capture, .capture = Capture.tap });
    try std.testing.expectEqual(rows, cache.len);
    const logits = try graph.lastLogits(normed, rows);
    try stream.synchronize();
    var token = try compareLogits(stream, logits, reference, 0);
    try std.testing.expect(capture.compared >= 3);
    try std.testing.expectEqual(@as(usize, 0), capture.final_mismatches orelse return error.MissingTextReference);
    try std.testing.expectEqual(@as(usize, 0), capture.total_mismatches);
    const receipt_file = try read(reference, "receipt.json");
    defer a.free(receipt_file);
    const receipt = try std.json.parseFromSlice(struct { output_ids: []u32 }, a, receipt_file, .{ .ignore_unknown_fields = true });
    defer receipt.deinit();
    try std.testing.expect(receipt.value.output_ids.len > 0 and receipt.value.output_ids.len <= 256);
    try std.testing.expectEqual(receipt.value.output_ids[0], token);
    if (platform.getenv("TF_GLM_OCR_STATE_CHECK") != null)
        try @import("glm_ocr_state_check.zig").shared(&graph, &cache, receipt.value.output_ids, rope.next);
    var next_input = try hip.DeviceBuffer.alloc(&r, c.hidden * 2);
    defer next_input.free();
    var next_positions = try hip.DeviceBuffer.alloc(&r, 3 * 4);
    defer next_positions.free();
    const embedding_table = try weights.get(family.schema.text ++ "embed_tokens.weight");
    capture.required_names = &.{};
    for (1..receipt.value.output_ids.len) |step| {
        try next_input.copyFromAsync(0, embedding_table + token * c.hidden * 2, c.hidden * 2, stream);
        const coordinates: [3]i32 = @splat(rope.next + @as(i32, @intCast(step - 1)));
        try next_positions.upload(0, std.mem.asBytes(&coordinates));
        capture.name_prefix = "decode-1-";
        const hidden = try graph.forward(&cache, next_input.base(), next_positions.base(), 1, if (step == 1) .{ .context = &capture, .capture = Capture.tap } else null);
        token = try compareLogits(stream, try graph.lastLogits(hidden, 1), reference, step);
        try std.testing.expectEqual(receipt.value.output_ids[step], token);
    }
    try std.testing.expectEqual(@as(usize, 0), capture.total_mismatches);
    std.debug.print("GLM-OCR HIP text: {d} prompt rows and {d} generated IDs match the independent reference\n", .{ rows, receipt.value.output_ids.len });
}
