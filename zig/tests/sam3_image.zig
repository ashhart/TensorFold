const std = @import("std");
const sam = @import("sam3");
const st = @import("safetensors");
const Capture = @import("sam3_capture.zig").Capture;
const text_output = @import("sam3_text_output.zig");

pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    if (args.len == 2 and std.mem.eql(u8, args[1], "--help")) {
        std.debug.print(
            \\tf-sam3-image CHECKPOINT FRAMES OUTPUT_PREFIX --point X Y | --box X0 Y0 X1 Y1 | --text TEXT --tokenizer FILE
            \\  --negative-point X Y --fp32 --frame N --threshold N
            \\  --primary (visual prompts: candidate 0 instead of the best alternative candidate)
            \\  --raw-masks FILE.safetensors (visual prompts: the mask as object 1 of a one-frame mask file)
            \\  --text-features FILE.safetensors --detections FILE.safetensors (text prompts: framed ids and features;
            \\    detection masks, scores and source-pixel corner boxes in output order)
            \\
        , .{});
        return;
    }
    if (args.len < 5) return error.ExpectedCheckpointFramesOutputAndPrompt;
    var points: std.ArrayList(sam.prompts.Point) = .empty;
    var boxes: std.ArrayList(sam.prompts.Box) = .empty;
    var text: ?[]const u8 = null;
    var tokenizer: ?[]const u8 = null;
    var frame: usize = 0;
    var fp32 = false;
    var primary = false;
    var raw_path: ?[]const u8 = null;
    var features_path: ?[]const u8 = null;
    var detections_path: ?[]const u8 = null;
    var threshold: f32 = 0.5;
    var i: usize = 4;
    while (i < args.len) {
        const flag = args[i];
        i += 1;
        if (std.mem.eql(u8, flag, "--fp32")) {
            fp32 = true;
        } else if (std.mem.eql(u8, flag, "--primary")) {
            primary = true;
        } else if (std.mem.eql(u8, flag, "--point") or std.mem.eql(u8, flag, "--negative-point")) {
            if (i + 2 > args.len) return error.MissingPointCoordinates;
            try points.append(a, .{ .x = try std.fmt.parseFloat(f32, args[i]), .y = try std.fmt.parseFloat(f32, args[i + 1]), .positive = std.mem.eql(u8, flag, "--point") });
            i += 2;
        } else if (std.mem.eql(u8, flag, "--box")) {
            if (i + 4 > args.len) return error.MissingBoxCoordinates;
            try boxes.append(a, .{ .x0 = try std.fmt.parseFloat(f32, args[i]), .y0 = try std.fmt.parseFloat(f32, args[i + 1]), .x1 = try std.fmt.parseFloat(f32, args[i + 2]), .y1 = try std.fmt.parseFloat(f32, args[i + 3]) });
            i += 4;
        } else {
            if (i == args.len) return error.MissingArgument;
            const value = args[i];
            i += 1;
            if (std.mem.eql(u8, flag, "--text-features")) features_path = value else if (std.mem.eql(u8, flag, "--detections")) detections_path = value else if (std.mem.eql(u8, flag, "--raw-masks")) raw_path = value else if (std.mem.eql(u8, flag, "--text")) text = value else if (std.mem.eql(u8, flag, "--tokenizer"))
                tokenizer = value
            else if (std.mem.eql(u8, flag, "--frame")) frame = try std.fmt.parseInt(usize, value, 10) else if (std.mem.eql(u8, flag, "--threshold")) threshold = try std.fmt.parseFloat(f32, value) else return error.UnknownArgument;
        }
    }
    if (text != null and (points.items.len != 0 or boxes.items.len != 0)) return error.MixedPromptNotSupported;
    if (text == null and points.items.len == 0 and boxes.items.len == 0) return error.EmptyPrompt;
    if (text != null and (primary or raw_path != null)) return error.VisualPromptOption;
    if (text == null and (features_path != null or detections_path != null)) return error.TextPromptOption;
    var file = try st.File.open(init.gpa, init.io, args[2]);
    defer file.close(init.io);
    const frames = file.get("frames") orelse return error.MissingFramesTensor;
    if (frames.dtype != .u8 or frames.rank != 4 or frames.dim(3) != 3 or frame >= frames.dim(0))
        return error.ExpectedRgbFrames;
    const height = frames.dim(1);
    const width = frames.dim(2);
    const bytes = try std.math.mul(usize, try std.math.mul(usize, width, height), 3);
    const image = sam.Image{ .size = .{ .width = width, .height = height }, .rgb = frames.bytes[frame * bytes ..][0..bytes] };
    const started = @import("metal").clock.seconds();
    const model = try sam.Model.load(init.gpa, init.io, args[1], .{
        .tokenizer_path = tokenizer,
        .precision = if (fp32) .float32 else .checkpoint,
    });
    defer model.deinit();
    if (features_path) |path| try text_output.writeFeatures(a, init.io, model, text.?, path);
    var prediction = if (text) |prompt|
        try model.predictText(init.gpa, image, prompt, .{ .threshold = threshold })
    else
        try model.predictVisual(init.gpa, image, .{ .points = points.items, .boxes = boxes.items }, !primary);
    defer prediction.deinit();
    if (raw_path) |path| {
        var capture = try Capture.create(a, init.io, .cwd(), path, &.{1}, image.size, 1, frame, false);
        defer capture.deinit(init.io);
        var objects = [_]sam.session.TrackedMask{.{ .id = 1, .mask = prediction.masks[0] }};
        try capture.add(init.io, .{ .allocator = a, .frame_index = frame, .objects = &objects }, false);
        try capture.finish(init.io);
    }
    if (detections_path) |path| try text_output.writeDetections(a, init.io, prediction, image.size, path);
    for (prediction.masks, 0..) |mask, n| {
        const path = try std.fmt.allocPrint(a, "{s}-{d}.pgm", .{ args[3], n });
        const output = try std.Io.Dir.cwd().createFile(init.io, path, .{ .exclusive = true });
        defer output.close(init.io);
        const header = try std.fmt.allocPrint(a, "P5\n{d} {d}\n255\n", .{ width, height });
        try output.writeStreamingAll(init.io, header);
        try output.writeStreamingAll(init.io, mask.pixels);
        std.debug.print("mask {d}: score {d:.6}, box {any}, {s}\n", .{ n, mask.score, mask.box, path });
    }
    std.debug.print("SAM {s}: {d} masks, {d:.3}s wall, {d:.3}s GPU, peak Metal allocation {d} bytes\n", .{
        model.config.version.name(), prediction.masks.len, @import("metal").clock.seconds() - started,
        model.gpu.gpu_seconds,       model.gpu.peak_bytes,
    });
}
