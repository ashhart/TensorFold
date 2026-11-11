const std = @import("std");
const sam = @import("sam3");
const st = @import("safetensors");
const clock = @import("metal").clock;

pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    if (args.len < 5) return error.ExpectedCheckpointFramesOutputAndPrompt;
    var seeds: std.ArrayList(sam.session.Seed) = .empty;
    var start: usize = 0;
    var count: usize = 4;
    var reverse = false;
    var exercise = false;
    var fp32 = false;
    var reference_path: ?[]const u8 = null;
    var i: usize = 4;
    while (i < args.len) {
        const flag = args[i];
        i += 1;
        if (std.mem.eql(u8, flag, "--reverse")) {
            reverse = true;
            continue;
        }
        if (std.mem.eql(u8, flag, "--exercise-lifecycle")) {
            exercise = true;
            continue;
        }
        if (std.mem.eql(u8, flag, "--fp32")) {
            fp32 = true;
            continue;
        }
        if (std.mem.eql(u8, flag, "--box")) {
            if (i + 5 > args.len) return error.MissingBoxArguments;
            const boxes = try a.alloc(sam.prompts.Box, 1);
            boxes[0] = .{
                .x0 = try std.fmt.parseFloat(f32, args[i + 1]),
                .y0 = try std.fmt.parseFloat(f32, args[i + 2]),
                .x1 = try std.fmt.parseFloat(f32, args[i + 3]),
                .y1 = try std.fmt.parseFloat(f32, args[i + 4]),
            };
            try seeds.append(a, .{ .id = try std.fmt.parseInt(u64, args[i], 10), .prompt = .{ .boxes = boxes } });
            i += 5;
        } else if (std.mem.eql(u8, flag, "--point")) {
            if (i + 3 > args.len) return error.MissingPointArguments;
            const points = try a.alloc(sam.prompts.Point, 1);
            points[0] = .{ .x = try std.fmt.parseFloat(f32, args[i + 1]), .y = try std.fmt.parseFloat(f32, args[i + 2]) };
            try seeds.append(a, .{ .id = try std.fmt.parseInt(u64, args[i], 10), .prompt = .{ .points = points } });
            i += 3;
        } else {
            if (i == args.len) return error.MissingArgument;
            const value = args[i];
            i += 1;
            if (std.mem.eql(u8, flag, "--start")) start = try std.fmt.parseInt(usize, value, 10) else if (std.mem.eql(u8, flag, "--frames")) count = try std.fmt.parseInt(usize, value, 10) else if (std.mem.eql(u8, flag, "--seed-reference")) reference_path = value else return error.UnknownArgument;
        }
    }
    if (count == 0) return error.EmptyVideo;
    var file = try st.File.open(init.gpa, init.io, args[2]);
    defer file.close(init.io);
    const frames = file.get("frames") orelse return error.MissingFramesTensor;
    if (frames.dtype != .u8 or frames.rank != 4 or frames.dim(3) != 3) return error.ExpectedRgbFrames;
    if (start >= frames.dim(0) or (reverse and count - 1 > start) or
        (!reverse and count > frames.dim(0) - start)) return error.FrameOutOfRange;
    const height = frames.dim(1);
    const width = frames.dim(2);
    const pixels = try std.math.mul(usize, width, height);
    const bytes = try std.math.mul(usize, pixels, 3);
    var reference: ?st.File = null;
    defer if (reference) |*f| f.close(init.io);
    if (reference_path) |path| {
        if (seeds.items.len != 0) return error.MixedSeedModes;
        reference = try st.File.open(init.gpa, init.io, path);
        const masks = reference.?.get("masks") orelse return error.MissingMasks;
        if (!masks.is(.u8, &.{ frames.dim(0), height, width })) return error.BadReferenceShape;
        const values = try a.alloc(f32, pixels);
        for (values, masks.bytes[start * pixels ..][0..pixels]) |*out, byte|
            out.* = @as(f32, @floatFromInt(byte)) / 255;
        try seeds.append(a, .{ .id = 42, .prompt = .{ .mask = .{
            .size = .{ .width = width, .height = height },
            .values = values,
        } } });
    }
    if (seeds.items.len == 0) return error.EmptyPrompt;
    const loaded = try sam.Model.load(init.gpa, init.io, args[1], .{
        .precision = if (fp32) .float32 else .checkpoint,
    });
    defer loaded.deinit();
    var session = try sam.Session.init(init.gpa, loaded, .{
        .max_objects = @max(seeds.items.len, 17),
        .frame_count = frames.dim(0),
        .direction = if (reverse) .reverse else .forward,
    });
    defer session.deinit();
    var total_iou: f64 = 0;
    var minimum_iou: f64 = 1;
    const started = clock.seconds();
    for (0..count) |step| {
        const index = if (reverse) start - step else start + step;
        const image = sam.Image{ .size = .{ .width = width, .height = height }, .rgb = frames.bytes[index * bytes ..][0..bytes] };
        if (exercise and step != 0) switch (step % 5) {
            1, 3 => try session.prompt(seeds.items[0].id, seeds.items[0].prompt),
            2 => try session.removeObject(seeds.items[0].id),
            4 => try session.reset(),
            else => {},
        };
        var prediction = if (step == 0 or (exercise and step % 5 == 4))
            try session.seed(init.gpa, image, index, seeds.items)
        else
            try session.track(init.gpa, image, index);
        defer prediction.deinit();
        for (prediction.objects) |object| {
            var area: usize = 0;
            for (object.mask.pixels) |value| area += @intFromBool(value != 0);
            std.debug.print("frame {d}, object {d}: score {d:.5}, area {d}\n", .{ prediction.frame_index, object.id, object.mask.score, area });
            if (reference) |ref| {
                const expected = ref.get("masks").?.bytes[index * pixels ..][0..pixels];
                var intersection: usize = 0;
                var union_size: usize = 0;
                for (expected, object.mask.pixels) |x, y| {
                    intersection += @intFromBool(x != 0 and y != 0);
                    union_size += @intFromBool(x != 0 or y != 0);
                }
                const iou = if (union_size == 0) 1 else @as(f64, @floatFromInt(intersection)) / @as(f64, @floatFromInt(union_size));
                total_iou += iou;
                minimum_iou = @min(minimum_iou, iou);
            }
            if (step + 1 == count) {
                const path = try std.fmt.allocPrint(a, "{s}-{d}.pgm", .{ args[3], object.id });
                const output = try std.Io.Dir.cwd().createFile(init.io, path, .{ .exclusive = true });
                defer output.close(init.io);
                try output.writeStreamingAll(init.io, try std.fmt.allocPrint(a, "P5\n{d} {d}\n255\n", .{ width, height }));
                try output.writeStreamingAll(init.io, object.mask.pixels);
            }
        }
    }
    std.debug.print("SAM {s}: {d} frames, {d:.3}s wall, {d:.3}s GPU, peak Metal allocation {d} bytes\n", .{ loaded.config.version.name(), count, clock.seconds() - started, loaded.gpu.gpu_seconds, loaded.gpu.peak_bytes });
    if (reference != null) std.debug.print("reference IoU: mean {d:.6}, minimum {d:.6}\n", .{ total_iou / @as(f64, @floatFromInt(count)), minimum_iou });
}
