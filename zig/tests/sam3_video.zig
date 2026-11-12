const std = @import("std");
const sam = @import("sam3");
const st = @import("safetensors");
const clock = @import("metal").clock;
const Capture = @import("sam3_capture.zig").Capture;
const receipt = @import("sam3_receipt.zig");
const Trace = @import("sam3_trace.zig").Trace;

pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    if (args.len == 2 and std.mem.eql(u8, args[1], "--help")) {
        std.debug.print(
            \\tf-sam3-video CHECKPOINT FRAMES OUTPUT_PREFIX --point ID X Y | --box ID X0 Y0 X1 Y1 | --seed-reference MASKS
            \\  --start N --frames N --reverse --fp32 --exercise-lifecycle --baseline-attention --baseline-gemm
            \\  --reference MASKS --min-frame-iou N --min-mean-iou N
            \\  --raw-masks FILE.safetensors --benchmark FILE.json --warmup N
            \\  --trace DIR --trace-frames N,N (NHWC tensors; recomputes preprocessing and trunk)
            \\  --trace-layers (also capture each vision block)
            \\Reference comparisons exclude the seed frame and require one unchanged object.
            \\Default: exact binary masks. Approximate comparisons require --min-frame-iou.
            \\
        , .{});
        return;
    }
    if (args.len < 5) return error.ExpectedCheckpointFramesOutputAndPrompt;
    var seeds: std.ArrayList(sam.session.Seed) = .empty;
    var start: usize = 0;
    var count: usize = 4;
    var reverse = false;
    var exercise = false;
    var fp32 = false;
    var attention_two_pass = true;
    var compact_gemm = (sam.LoadOptions{}).compact_gemm;
    var reference_path: ?[]const u8 = null;
    var seed_path: ?[]const u8 = null;
    var raw_path: ?[]const u8 = null;
    var benchmark_path: ?[]const u8 = null;
    var warmup: usize = 0;
    var trace_path: ?[]const u8 = null;
    var trace_frames: std.ArrayList(usize) = .empty;
    var trace_layers = false;
    var gate = sam.comparison.Gate{};
    var i: usize = 4;
    while (i < args.len) {
        const flag = args[i];
        i += 1;
        if (std.mem.eql(u8, flag, "--trace-layers")) {
            trace_layers = true;
            continue;
        }
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
        if (std.mem.eql(u8, flag, "--baseline-attention")) {
            attention_two_pass = false;
            continue;
        }
        if (std.mem.eql(u8, flag, "--baseline-gemm")) {
            compact_gemm = false;
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
            if (std.mem.eql(u8, flag, "--trace")) {
                trace_path = value;
                continue;
            }
            if (std.mem.eql(u8, flag, "--trace-frames")) {
                var parts = std.mem.splitScalar(u8, value, ',');
                while (parts.next()) |part| {
                    const frame = try std.fmt.parseInt(usize, part, 10);
                    if (std.mem.indexOfScalar(usize, trace_frames.items, frame) != null) return error.DuplicateTraceFrame;
                    try trace_frames.append(a, frame);
                }
                continue;
            }
            if (std.mem.eql(u8, flag, "--start")) start = try std.fmt.parseInt(usize, value, 10) else if (std.mem.eql(u8, flag, "--frames")) count = try std.fmt.parseInt(usize, value, 10) else if (std.mem.eql(u8, flag, "--warmup")) warmup = try std.fmt.parseInt(usize, value, 10) else if (std.mem.eql(u8, flag, "--seed-reference")) seed_path = value else if (std.mem.eql(u8, flag, "--reference")) reference_path = value else if (std.mem.eql(u8, flag, "--raw-masks")) raw_path = value else if (std.mem.eql(u8, flag, "--benchmark")) benchmark_path = value else if (!try gate.set(flag, value)) return error.UnknownArgument;
        }
    }
    if (count == 0) return error.EmptyVideo;
    if (warmup >= count or (warmup != 0 and benchmark_path == null)) return error.InvalidWarmup;
    if (trace_path != null and benchmark_path != null) return error.TraceTimingRequiresSeparateRuns;
    if (trace_path == null and (trace_frames.items.len != 0 or trace_layers)) return error.ExpectedTraceDirectory;
    if (trace_path != null and trace_frames.items.len == 0) try trace_frames.append(a, start);
    var file = try st.File.open(init.gpa, init.io, args[2]);
    defer file.close(init.io);
    const frames = file.get("frames") orelse return error.MissingFramesTensor;
    if (frames.dtype != .u8 or frames.rank != 4 or frames.dim(3) != 3) return error.ExpectedRgbFrames;
    if (start >= frames.dim(0) or (reverse and count - 1 > start) or
        (!reverse and count > frames.dim(0) - start)) return error.FrameOutOfRange;
    for (trace_frames.items) |frame| {
        if (frame >= frames.dim(0) or (reverse and (frame > start or start - frame >= count)) or
            (!reverse and (frame < start or frame - start >= count))) return error.TraceFrameOutOfRange;
    }
    const height = frames.dim(1);
    const width = frames.dim(2);
    const pixels = try std.math.mul(usize, width, height);
    const bytes = try std.math.mul(usize, pixels, 3);
    var reference: ?st.File = null;
    defer if (reference) |*f| f.close(init.io);
    if (seed_path) |path| {
        if (seeds.items.len != 0) return error.MixedSeedModes;
        var seed_file = try st.File.open(init.gpa, init.io, path);
        defer seed_file.close(init.io);
        const masks = seed_file.get("masks") orelse return error.MissingMasks;
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
    if (benchmark_path != null and exercise) return error.LifecycleTimingRequiresSeparateRuns;
    const comparison_path = reference_path orelse seed_path;
    try sam.comparison.validateVideo(comparison_path != null, count, seeds.items.len, exercise, gate);
    if (comparison_path) |path| {
        reference = try st.File.open(init.gpa, init.io, path);
        const masks = reference.?.get("masks") orelse return error.MissingMasks;
        if (!masks.is(.u8, &.{ frames.dim(0), height, width })) return error.BadReferenceShape;
    }
    for (seeds.items) |seed| try unusedOutput(init.io, try std.fmt.allocPrint(a, "{s}-{d}.pgm", .{ args[3], seed.id }));
    if (benchmark_path) |path| try unusedOutput(init.io, path);
    if (raw_path) |path| try unusedOutput(init.io, path);
    const identity: ?receipt.Identity = if (benchmark_path != null) try receipt.Identity.load(a, init.io, args[1], frames.bytes, seeds.items) else null;
    const ids = try a.alloc(u64, seeds.items.len);
    for (seeds.items, ids) |seed, *id| id.* = seed.id;
    var capture: ?Capture = if (raw_path) |path| try Capture.create(a, init.io, .cwd(), path, ids, .{ .width = width, .height = height }, count, start, reverse) else null;
    defer if (capture) |*output| output.deinit(init.io);
    const samples: []receipt.Sample = if (benchmark_path != null) try a.alloc(receipt.Sample, count) else &.{};
    var masks_hash = std.crypto.hash.sha2.Sha256.init(.{});
    var trace: ?Trace = if (trace_path) |path| try Trace.init(init.gpa, init.io, path, trace_frames.items, trace_layers) else null;
    defer if (trace) |*output| output.deinit();
    const loaded = try sam.Model.load(init.gpa, init.io, args[1], .{
        .precision = if (fp32) .float32 else .checkpoint,
        .compact_gemm = compact_gemm,
    });
    defer loaded.deinit();
    loaded.gpu.attention_two_pass = attention_two_pass;
    const started = clock.seconds();
    if (exercise) {
        if (count < 2) return error.LifecycleNeedsTwoFrames;
        const next = if (reverse) start - 1 else start + 1;
        const size = sam.prompts.Size{ .width = width, .height = height };
        try @import("sam3_sessions.zig").check(init.gpa, init.io, loaded, .{
            .{ .size = size, .rgb = frames.bytes[start * bytes ..][0..bytes] },
            .{ .size = size, .rgb = frames.bytes[next * bytes ..][0..bytes] },
        }, .{ start, next }, frames.dim(0), seeds.items);
    }
    var session = try sam.Session.init(init.gpa, loaded, .{
        .max_objects = @max(seeds.items.len, 17),
        .frame_count = frames.dim(0),
        .direction = if (reverse) .reverse else .forward,
        .profile = benchmark_path != null,
    });
    defer session.deinit();
    var quality = sam.comparison.Quality{};
    for (0..count) |step| {
        const index = if (reverse) start - step else start + step;
        const image = sam.Image{ .size = .{ .width = width, .height = height }, .rgb = frames.bytes[index * bytes ..][0..bytes] };
        if (exercise and step != 0) switch (step % 5) {
            1, 3 => try session.prompt(seeds.items[0].id, seeds.items[0].prompt),
            2 => try session.removeObject(seeds.items[0].id),
            4 => try session.reset(),
            else => {},
        };
        const frame_start = if (benchmark_path != null) clock.seconds() else 0;
        if (trace) |*output| session.r.observer = output.observer(index);
        const is_seed = step == 0 or (exercise and step % 5 == 4);
        var prediction = if (is_seed)
            try session.seed(init.gpa, image, index, seeds.items)
        else
            try session.track(init.gpa, image, index);
        defer prediction.deinit();
        if (benchmark_path != null) {
            samples[step] = .{ .frame_index = index, .seed = is_seed, .wall_seconds = clock.seconds() - frame_start, .stages = session.last_profile };
            var index_bytes: [8]u8 = undefined;
            std.mem.writeInt(u64, &index_bytes, prediction.frame_index, .little);
            masks_hash.update(&index_bytes);
            for (prediction.objects) |object| {
                std.mem.writeInt(u64, &index_bytes, object.id, .little);
                masks_hash.update(&index_bytes);
                masks_hash.update(object.mask.pixels);
            }
        }
        if (capture) |*output| try output.add(init.io, prediction, exercise);
        if (trace) |*output| try output.frame(&session, image, index);
        if (reference != null and step != 0) {
            const expected = reference.?.get("masks").?.bytes[index * pixels ..][0..pixels];
            try quality.addPrediction(index, seeds.items[0].id, prediction, expected);
        }
        for (prediction.objects) |object| {
            var area: usize = 0;
            for (object.mask.pixels) |value| area += @intFromBool(value != 0);
            std.debug.print("frame {d}, object {d}: score {d:.5}, area {d}\n", .{ prediction.frame_index, object.id, object.mask.score, area });
            if (step + 1 == count) {
                const path = try std.fmt.allocPrint(a, "{s}-{d}.pgm", .{ args[3], object.id });
                const output = try std.Io.Dir.cwd().createFile(init.io, path, .{ .exclusive = true });
                defer output.close(init.io);
                try output.writeStreamingAll(init.io, try std.fmt.allocPrint(a, "P5\n{d} {d}\n255\n", .{ width, height }));
                try output.writeStreamingAll(init.io, object.mask.pixels);
            }
        }
    }
    if (capture) |*output| try output.finish(init.io);
    if (benchmark_path) |path| {
        var digest: [32]u8 = undefined;
        masks_hash.final(&digest);
        try receipt.write(a, init.io, path, identity.?, loaded, samples, warmup, .{
            .input_shape = frames.shape[0..frames.rank],
            .start = start,
            .frames = count,
            .reverse = reverse,
            .object_ids = ids,
            .attention_two_pass = attention_two_pass,
            .compact_gemm = compact_gemm,
        }, std.fmt.bytesToHex(digest, .lower));
    }
    std.debug.print("SAM {s}: {d} frames, {d:.3}s wall, {d:.3}s GPU, peak Metal allocation {d} bytes\n", .{ loaded.config.version.name(), count, clock.seconds() - started, loaded.gpu.gpu_seconds, loaded.gpu.peak_bytes });
    if (reference != null) {
        std.debug.print("reference IoU (propagation only, seed excluded): {d} frames, mean {d:.6}, minimum {d:.6}, worst frame {?d}\n", .{ quality.frames, quality.meanIou(), quality.minimum_iou, quality.worst_frame });
        try gate.check(quality, count - 1);
    }
}

fn unusedOutput(io: std.Io, path: []const u8) !void {
    std.Io.Dir.cwd().access(io, path, .{}) catch |err| switch (err) {
        error.FileNotFound => return,
        else => return err,
    };
    return error.PathAlreadyExists;
}

test {
    _ = @import("sam3_capture.zig");
    _ = receipt;
}
