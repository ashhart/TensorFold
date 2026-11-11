const std = @import("std");
const sam = @import("sam3");
const st = @import("safetensors");
const Masks = @import("sam3_mask_file.zig").Masks;

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len == 2 and std.mem.eql(u8, args[1], "--help")) {
        std.debug.print("tf-sam3-compare LEFT.safetensors RIGHT.safetensors [--skip-first-frame] [--min-frame-iou N] [--min-mean-iou N]\nDefault: exact binary masks. Approximate comparisons require a worst-frame IoU threshold.\n", .{});
        return;
    }
    if (args.len < 3) return error.ExpectedTwoMaskFiles;
    var gate = sam.comparison.Gate{};
    var skip_first = false;
    var at: usize = 3;
    while (at < args.len) : (at += 1) {
        if (std.mem.eql(u8, args[at], "--skip-first-frame")) {
            if (skip_first) return error.DuplicateComparisonOption;
            skip_first = true;
        } else {
            if (at + 1 == args.len) return error.MissingComparisonThreshold;
            if (!try gate.set(args[at], args[at + 1])) return error.UnknownComparisonOption;
            at += 1;
        }
    }
    try gate.validate();
    var left = try st.File.open(init.gpa, init.io, args[1]);
    defer left.close(init.io);
    var right = try st.File.open(init.gpa, init.io, args[2]);
    defer right.close(init.io);
    const x = try Masks.fromFile(&left);
    const y = try Masks.fromFile(&right);
    try x.match(y);
    const first: usize = @intFromBool(skip_first);
    if (x.frames <= first) return error.EmptyMaskComparison;
    var quality = sam.comparison.Quality{};
    for (first..x.frames) |frame| for (0..x.objects) |object| {
        try quality.add(std.math.cast(usize, x.frameIndex(frame)) orelse return error.FrameOutOfRange, x.mask(frame, object), y.mask(frame, object));
    };
    const failure: ?anyerror = if (gate.check(quality, (x.frames - first) * x.objects)) |_| null else |err| err;
    const json = try std.json.Stringify.valueAlloc(init.arena.allocator(), .{
        .scope = if (skip_first) "propagation_only" else "all_frames",
        .skipped_frames = first,
        .frames = x.frames - first,
        .objects = x.objects,
        .mask_pairs = quality.frames,
        .pixels = quality.pixels,
        .exact_pixels = quality.exact_pixels,
        .mean_iou = quality.meanIou(),
        .minimum_iou = quality.minimum_iou,
        .worst_frame = quality.worst_frame,
        .thresholds = gate,
        .passed = failure == null,
    }, .{ .whitespace = .indent_2 });
    std.debug.print("{s}\n", .{json});
    if (failure) |err| return err;
}

test {
    _ = @import("sam3_mask_file.zig");
}
