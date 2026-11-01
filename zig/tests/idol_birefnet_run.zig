//! tf-idol-birefnet's case driver for a backend `B` (safetensors, host steps, gates, Device, stage hooks).
const std = @import("std");

const usage =
    \\tf-idol-birefnet SNAPSHOT_DIR CASE_DIR PHOTO
    \\SNAPSHOT_DIR holds birefnet.safetensors (speridlabs/idol's snapshot). CASE_DIR is idol_birefnet_case.py's for PHOTO
    \\(birefnet.safetensors and .json, and birefnet_stages.safetensors with --stages). Gates, each printed:
    \\- the resized photo and the BiRefNet input: byte for byte the SDK's;
    \\- the mask logits from our own input against the float64 ruler: worst image row and 1 - cosine within twice
    \\  torch's own; with the stages, each stage on torch's own input within twice torch's own as well;
    \\- the alpha step on the SDK's own logits: byte for byte the SDK's alpha;
    \\- the uint8 alpha: differing pixels against the ruler's within twice torch's, largest difference within torch's;
    \\- the picture from the SDK's alpha and from the ruler's: byte for byte the SDK's (Pillow's steps exactly); from our
    \\  own alpha against the ruler's picture: differing pixels and largest difference within twice the SDK's;
    \\- three more forwards byte for byte the first. CASE_DIR gets tf-idol-birefnet.json.
;

const timed = 3;

/// A net's tensors the stages read and write: tokens, the picture, x1 to x4, each level's input and output, logits.
pub const Slot = union(enum) { ta, tb, x, xs: usize, lin: usize, out: usize, logits };

/// A stage the net runs alone: Swin layer (pass, i), the encoder, the squeeze, decoder block l, the whole decoder.
pub const Step = union(enum) { layer: [2]usize, encode, squeeze, block: usize, decode };

fn take(comptime T: type, a: std.mem.Allocator, f: anytype, name: []const u8) ![]T {
    const t = f.get(name) orelse {
        std.debug.print("the case has no {s}\n", .{name});
        return error.MissingTensor;
    };
    if (t.dtype != if (T == f32) .f32 else .u8) return error.BadCaseTensor;
    const out = try a.alloc(T, t.numel());
    @memcpy(std.mem.sliceAsBytes(out), t.bytes);
    return out;
}

fn print(list: *std.ArrayList(u8), a: std.mem.Allocator, comptime fmt: []const u8, args: anytype) !void {
    try list.appendSlice(a, try std.fmt.allocPrint(a, fmt, args));
}

fn verdict(list: *std.ArrayList(u8), a: std.mem.Allocator, name: []const u8, v: anytype) !void {
    std.debug.print("{s}: ours worst row {e:.3}, 1 - cos {e:.3}; torch {e:.3}, {e:.3}: {s}\n", .{ name, v.ours.worst_row, 1 - v.ours.cosine, v.torch.worst_row, 1 - v.torch.cosine, if (v.pass) "pass" else "FAIL" });
    try print(list, a, ", \"{s}\": {{\"ours_worst_row\": {e:.4}, \"ours_one_minus_cos\": {e:.4}, \"torch_worst_row\": {e:.4}, \"torch_one_minus_cos\": {e:.4}, \"pass\": {}}}", .{ name, v.ours.worst_row, 1 - v.ours.cosine, v.torch.worst_row, 1 - v.torch.cosine, v.pass });
}

fn diffLine(list: *std.ArrayList(u8), a: std.mem.Allocator, name: []const u8, d: anytype) !void {
    std.debug.print("{s}: {d} pixels differ, by at most {d}\n", .{ name, d.pixels, d.max });
    try print(list, a, ", \"{s}\": {{\"pixels\": {d}, \"max\": {d}}}", .{ name, d.pixels, d.max });
}

/// Rows [n, cols] of `src` (row length ld, from column off).
fn columns(a: std.mem.Allocator, src: []const f32, n: usize, ld: usize, off: usize, cols: usize) ![]f32 {
    const out = try a.alloc(f32, n * cols);
    for (0..n) |r| @memcpy(out[r * cols ..][0..cols], src[r * ld + off ..][0..cols]);
    return out;
}

fn put(dst: []f32, src: []const f32, ld: usize, off: usize, cols: usize) void {
    for (0..src.len / cols) |r| @memcpy(dst[r * ld + off ..][0..cols], src[r * cols ..][0..cols]);
}

fn Stages(comptime B: type) type {
    return struct {
        const S = @This();
        const cfg = B.cfg;
        a: std.mem.Allocator,
        f: *const B.st.File,
        n: *B.Net,
        out: *std.ArrayList(u8),
        pass: bool = true,

        fn check(s: *S, name: []const u8, ours: []const f32, width: usize) !void {
            var buf: [64]u8 = undefined;
            const torch = try take(f32, s.a, s.f, name);
            const ruler = try take(f32, s.a, s.f, try std.fmt.bufPrint(&buf, "ruler.{s}", .{name}));
            const v = B.gt.gate(ours, torch, ruler, width);
            s.pass = s.pass and v.pass;
            try verdict(s.out, s.a, name, v);
        }

        /// The case tensor `name` into the net's `slot`.
        fn load(s: *S, slot: Slot, name: []const u8) !void {
            try B.write(s.n, slot, try take(f32, s.a, s.f, name));
        }

        fn all(s: *S) !void {
            const n = s.n;
            var buf: [64]u8 = undefined;
            for (0..2) |pass| for (0..4) |i| {
                const h = cfg.grids[pass] >> @intCast(i);
                const C = cfg.widths[i];
                try s.load(.ta, try std.fmt.bufPrint(&buf, "layer.{d}.{d}.in", .{ i, pass }));
                try B.stage(n, .{ .layer = .{ pass, i } });
                try s.check(try std.fmt.bufPrint(&buf, "layer.{d}.{d}.out", .{ i, pass }), try B.read(n, .ta, h * h * C), C);
                if (i < 3) try s.check(try std.fmt.bufPrint(&buf, "layer.{d}.{d}.down", .{ i, pass }), try B.read(n, .tb, h * h * C / 2), 2 * C);
            };
            try s.load(.x, "enc.in");
            try B.stage(n, .encode);
            const sides = [4]usize{ 256, 128, 64, 32 };
            const widths = [4]usize{ 384, 768, 1536, cfg.squeeze_in };
            for (0..4) |i| try s.check(try std.fmt.bufPrint(&buf, "enc.out.x{d}", .{i + 1}), try B.read(n, .{ .xs = i }, sides[i] * sides[i] * widths[i]), widths[i]);
            try s.load(.{ .xs = 3 }, "squeeze.in");
            try B.stage(n, .squeeze);
            try s.check("squeeze.out", try columns(s.a, try B.read(n, .{ .lin = 0 }, 1024 * cfg.level_in[0]), 1024, cfg.level_in[0], 0, cfg.channels[0]), cfg.channels[0]);
            for (0..4) |l| {
                const side = cfg.level_side[l];
                try s.load(.{ .lin = l }, try std.fmt.bufPrint(&buf, "dec{d}.in", .{4 - l}));
                try B.stage(n, .{ .block = l });
                try s.check(try std.fmt.bufPrint(&buf, "dec{d}.out", .{4 - l}), try B.read(n, .{ .out = l }, side * side * cfg.level_out[l]), cfg.level_out[l]);
            }
            try s.load(.x, "decoder.in.x");
            for (0..3) |i| try s.load(.{ .xs = i }, try std.fmt.bufPrint(&buf, "decoder.in.x{d}", .{i + 1}));
            const lin0 = try B.read(n, .{ .lin = 0 }, 1024 * cfg.level_in[0]);
            put(lin0, try take(f32, s.a, s.f, "decoder.in.x4"), cfg.level_in[0], 0, cfg.channels[0]);
            try B.write(n, .{ .lin = 0 }, lin0);
            try B.stage(n, .decode);
            try s.check("decoder.out", try B.read(n, .logits, cfg.size * cfg.size), cfg.size);
        }
    };
}

pub fn run(comptime B: type, init: std.process.Init) !u8 {
    const pre = B.pre;
    const gt = B.gt;
    const a = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    if (args.len != 4) {
        std.debug.print("{s}\n", .{usage});
        return 2;
    }
    const cwd = std.Io.Dir.cwd();
    var case = try B.st.File.open(a, io, try std.fmt.allocPrint(a, "{s}/birefnet.safetensors", .{args[2]}));
    defer case.close(io);
    const meta_text = try cwd.readFileAlloc(io, try std.fmt.allocPrint(a, "{s}/birefnet.json", .{args[2]}), a, .limited(1 << 22));
    const meta = try std.json.parseFromSliceLeaky(struct { crop_margin: f64, has_real_alpha: bool }, a, meta_text, .{ .ignore_unknown_fields = true });
    var out: std.ArrayList(u8) = .empty;
    var pass = true;

    // the host's steps: decode and the first resize, against the SDK's resized photo
    const started = B.seconds();
    const bytes = try cwd.readFileAlloc(io, args[3], a, .limited(1 << 28));
    var photo = try pre.decode(io, a, bytes);
    var resized = try pre.resizeFirst(a, photo);
    const resize_s = B.seconds() - started;
    std.debug.print("photo {d} x {d}, {d} channels; resized to {d} x {d} in {d:.3} s\n", .{ photo.width, photo.height, photo.channels, resized.width, resized.height, resize_s });
    try print(&out, a, "{{\"photo\": [{d}, {d}, {d}], \"resized\": [{d}, {d}], \"resize_s\": {d:.4}, \"has_real_alpha\": {}", .{ photo.width, photo.height, photo.channels, resized.width, resized.height, resize_s, pre.hasRealAlpha(resized) });
    if (pre.hasRealAlpha(resized) != meta.has_real_alpha) {
        std.debug.print("the photo's real alpha differs from the SDK's\n", .{});
        return 1;
    }
    const picture = try a.alloc(u8, pre.size * pre.size * 3);
    const sdk_picture = try take(u8, a, &case, "picture");
    if (meta.has_real_alpha) {
        const finishing = B.seconds();
        try pre.finish(a, resized, null, meta.crop_margin, picture);
        const d = gt.diff(picture, sdk_picture, 3);
        try diffLine(&out, a, "picture_vs_sdk", d);
        std.debug.print("preprocess (no BiRefNet: real alpha) {d:.3} s\n", .{resize_s + B.seconds() - finishing});
        try print(&out, a, "}}\n", .{});
        try cwd.writeFile(io, .{ .sub_path = try std.fmt.allocPrint(a, "{s}/tf-idol-birefnet.json", .{args[2]}), .data = out.items });
        return if (d.pixels == 0) 0 else 1;
    }
    const sdk_resized = try take(u8, a, &case, "resized");
    const rd = gt.diff(resized.pixels, sdk_resized, 3);
    try diffLine(&out, a, "resized_vs_sdk", rd);
    pass = pass and rd.pixels == 0;
    const size = B.cfg.size;
    const input = try a.alloc(f32, size * size * 3);
    try pre.input(a, resized, input);
    const sdk_input = try take(f32, a, &case, "input");
    var in_most: f32 = 0;
    var in_count: usize = 0;
    for (input, sdk_input) |x, y| {
        in_most = @max(in_most, @abs(x - y));
        if (x != y) in_count += 1;
    }
    std.debug.print("input against the SDK's: {d} values differ, by at most {e:.3}\n", .{ in_count, in_most });
    try print(&out, a, ", \"input_vs_sdk\": {{\"values\": {d}, \"max_abs\": {e:.4}}}", .{ in_count, in_most });
    pass = pass and in_count == 0;

    var dev: B.Device = undefined;
    try dev.init();
    defer dev.deinit();
    const loading = B.seconds();
    const net = try dev.load(std.heap.page_allocator, io, try std.fmt.allocPrintSentinel(a, "{s}/birefnet.safetensors", .{args[1]}, 0));
    defer net.deinit();
    const load_s = B.seconds() - loading;
    std.debug.print("{s}: BiRefNet loaded in {d:.2} s\n", .{ dev.name(), load_s });
    try print(&out, a, ", \"device\": \"{s}\", \"load_s\": {d:.3}", .{ dev.name(), load_s });

    var stage_file: ?B.st.File = B.st.File.open(a, io, try std.fmt.allocPrint(a, "{s}/birefnet_stages.safetensors", .{args[2]})) catch null;
    defer if (stage_file) |*f| f.close(io);
    if (stage_file) |*f| {
        var s: Stages(B) = .{ .a = a, .f = f, .n = net, .out = &out };
        try s.all();
        pass = pass and s.pass;
    }

    const n = size * size;
    const logits = try a.alloc(f32, n);
    const alpha = try a.alloc(u8, resized.width * resized.height);
    const first = try net.removeBackground(a, resized, logits, alpha);
    const lv = gt.gate(logits, try take(f32, a, &case, "logits"), try take(f32, a, &case, "logits_ruler"), size);
    try verdict(&out, a, "logits", lv);
    pass = pass and lv.pass;
    const sdk_alpha = try take(u8, a, &case, "alpha");
    const ruler_alpha = try take(u8, a, &case, "alpha_ruler");
    // the alpha step alone: the SDK's own logits through ours must give the SDK's alpha
    const host_alpha = try a.alloc(u8, alpha.len);
    try pre.alpha(a, try take(f32, a, &case, "logits"), resized.width, resized.height, host_alpha);
    const hv = gt.diff(host_alpha, sdk_alpha, 1);
    try diffLine(&out, a, "alpha_from_sdk_logits_vs_sdk", hv);
    pass = pass and hv.pixels == 0;
    const av: [3]gt.Diff = .{ gt.diff(alpha, sdk_alpha, 1), gt.diff(alpha, ruler_alpha, 1), gt.diff(sdk_alpha, ruler_alpha, 1) };
    try diffLine(&out, a, "alpha_vs_sdk", av[0]);
    try diffLine(&out, a, "alpha_vs_ruler", av[1]);
    try diffLine(&out, a, "sdk_alpha_vs_ruler", av[2]);
    const alpha_pass = gt.withinDiff(av[1], av[2]);
    std.debug.print("alpha gate: {s}\n", .{if (alpha_pass) "pass" else "FAIL"});
    pass = pass and alpha_pass;

    // the steps after the alpha, on the SDK's alpha and the ruler's (Pillow's exactly), then on ours
    const ruler_picture = try take(u8, a, &case, "picture_ruler");
    try pre.finish(a, resized, sdk_alpha, meta.crop_margin, picture);
    const p_sdk = gt.diff(picture, sdk_picture, 3);
    try pre.finish(a, resized, ruler_alpha, meta.crop_margin, picture);
    const p_ruler = gt.diff(picture, ruler_picture, 3);
    try diffLine(&out, a, "picture_from_sdk_alpha_vs_sdk", p_sdk);
    try diffLine(&out, a, "picture_from_ruler_alpha_vs_ruler", p_ruler);
    pass = pass and p_sdk.pixels == 0 and p_ruler.pixels == 0;
    const finishing = B.seconds();
    try pre.finish(a, resized, alpha, meta.crop_margin, picture);
    const finish_s = B.seconds() - finishing;
    const pv: [3]gt.Diff = .{ gt.diff(picture, sdk_picture, 3), gt.diff(picture, ruler_picture, 3), gt.diff(sdk_picture, ruler_picture, 3) };
    try diffLine(&out, a, "picture_vs_sdk", pv[0]);
    try diffLine(&out, a, "picture_vs_ruler", pv[1]);
    try diffLine(&out, a, "sdk_picture_vs_ruler", pv[2]);
    const pic_pass = gt.withinPicture(pv[1], pv[2]);
    std.debug.print("picture gate: {s}\n", .{if (pic_pass) "pass" else "FAIL"});
    pass = pass and pic_pass;

    const again = try a.alloc(f32, n);
    var repeat = true;
    var times: [timed]f64 = undefined;
    var gpus: [timed]f64 = undefined;
    for (&times, &gpus) |*t, *g| {
        const tm = try net.forward(input, again);
        t.* = tm.wall;
        g.* = tm.gpu;
        repeat = repeat and std.mem.eql(u8, std.mem.sliceAsBytes(logits), std.mem.sliceAsBytes(again));
    }
    std.mem.sort(f64, &times, {}, std.sort.asc(f64));
    std.mem.sort(f64, &gpus, {}, std.sort.asc(f64));
    pass = pass and repeat;
    std.debug.print("{d} more forwards: {s}; first {d:.3} s, median {d:.4} s ({d:.4} s on the GPU); host finish {d:.3} s\n", .{ timed, if (repeat) "identical bytes" else "DIFFERENT bytes", first.wall, times[timed / 2], gpus[timed / 2], finish_s });
    if (@hasDecl(B.Device, "footprint")) try dev.footprint(net, &out, a);
    try print(&out, a, ", \"alpha_pass\": {}, \"picture_pass\": {}, \"repeat_identical\": {}, \"first_s\": {d:.4}, \"median_s\": {d:.4}, \"median_gpu_s\": {d:.4}, \"finish_s\": {d:.4}, \"pass\": {}}}\n", .{ alpha_pass, pic_pass, repeat, first.wall, times[timed / 2], gpus[timed / 2], finish_s, pass });
    try cwd.writeFile(io, .{ .sub_path = try std.fmt.allocPrint(a, "{s}/tf-idol-birefnet.json", .{args[2]}), .data = out.items });
    photo.deinit(a);
    resized.deinit(a);
    std.debug.print("{s}\n", .{if (pass) "PASS" else "FAIL"});
    return if (pass) 0 else 1;
}
