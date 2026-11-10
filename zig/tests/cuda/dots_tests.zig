//! dots.mocr on the GB10: the tower's stage taps for the oracle, and its encode twice equal against a generated page.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const dots = @import("dots_mocr_cuda");
const dense = @import("cuda_dense");
const check = @import("check.zig");
const Gpu = check.Gpu;

const Tower = dots.tower.Tower;
const Image = dense.recipe.Image;
const Prepared = core.picture.patches.Prepared;
const Args = []const [:0]const u8;

/// The stages one encode taps: the patch with its bias and its norm, every block, the post-norm, the merger.
const taps = 2 + 42 + 2;

/// dots-image's own page: the processor fixture's pixel pattern at a size of whole 28-pixel blocks, so no file is read.
const synthetic = .{ .width = 812, .height = 588, .seed = 2732 };
/// The generated page's vision-merged sha256 on the GB10: a tower drift fails dots-image without the Python oracle.
const synthetic_merged_sha256 = "ac8c66b76552c15c22af3bc776d285350c4d9596863de2a3359c9a909b971067";

const usage =
    \\  dots-stages MODEL IMAGE OUT  dots.mocr's tower: one encode, each stage's rows as OUT/<stage>.npy for the oracle
    \\  dots-image MODEL [IMAGE]  dots.mocr's tower: the generated page's vision-merged sha256 against its anchor, then
    \\                            IMAGE's; each encoded twice byte-equal, with each encode's time
    \\
;

/// tf-cuda-test dots-*: main.zig's one line for them; dots-help prints their usage.
pub fn run(gpu: Gpu, cmd: []const u8, rest: Args) !void {
    if (std.mem.eql(u8, cmd, "dots-help")) return std.debug.print("{s}", .{usage});
    if (!cuda.kernels.available) return skip(cmd, "this runner was built without CUDA kernels");
    if (std.mem.eql(u8, cmd, "dots-stages")) return stages(gpu, rest);
    if (std.mem.eql(u8, cmd, "dots-image")) return twice(gpu, rest);
    std.debug.print("{s}", .{usage});
    return error.UnknownCommand;
}

fn skip(cmd: []const u8, why: []const u8) void {
    std.debug.print("SKIP {s}: {s}\n", .{ cmd, why });
}

fn arg(rest: Args, i: usize) ![]const u8 {
    if (i >= rest.len) {
        std.debug.print("{s}", .{usage});
        return error.MissingArgument;
    }
    return rest[i];
}

/// Whether `dir` holds a checkpoint's config.json (else the command skips).
fn present(io: std.Io, dir: []const u8) bool {
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = std.fmt.bufPrint(&buf, "{s}/config.json", .{dir}) catch return false;
    std.Io.Dir.cwd().access(io, path, .{}) catch return false;
    return true;
}

fn readFile(gpu: Gpu, path: []const u8) ![]u8 {
    return std.Io.Dir.cwd().readFileAlloc(gpu.io, path, gpu.gpa, .limited(64 << 20));
}

/// The image file at `path` through the tower's processor, at the largest image's budget.
fn prepareFile(gpu: Gpu, t: *Tower, path: []const u8) !Prepared {
    const bytes = try readFile(gpu, path);
    defer gpu.gpa.free(bytes);
    return t.prepare(gpu.gpa, bytes, Tower.default_image_tokens);
}

/// The generated page through the tower's processor: (7x + 13y + 50c + xy mod 31 + seed) mod 256, as the fixture's.
fn preparePage(gpu: Gpu, t: *Tower) !Prepared {
    const rgb = try gpu.gpa.alloc(u8, synthetic.width * synthetic.height * 3);
    defer gpu.gpa.free(rgb);
    for (0..synthetic.height) |y| for (0..synthetic.width) |x| for (0..3) |c| {
        rgb[(y * synthetic.width + x) * 3 + c] = @intCast((7 * x + 13 * y + 50 * c + (x * y) % 31 + synthetic.seed) % 256);
    };
    const pixels: core.picture.Rgb = .{ .width = synthetic.width, .height = synthetic.height, .pixels = rgb };
    return t.processor.prepareRgb(gpu.gpa, pixels, synthetic.seed, Tower.default_image_tokens);
}

/// The tower alone on its own stream, one prepared image (the generated page without a path) and its output rows.
const Opened = struct {
    gpu: Gpu,
    s: cuda.Stream,
    t: *Tower,
    pixels: []f32,
    image: Image,
    out: cuda.DeviceBuffer,

    fn init(gpu: Gpu, dir: []const u8, path: ?[]const u8) !Opened {
        var s = try cuda.Stream.init(gpu.d, true);
        errdefer s.deinit();
        const t = try Tower.open(gpu.gpa, gpu.io, gpu.ctx, s, dir, .{ .image_tokens = Tower.default_image_tokens, .max_images = 1 });
        errdefer t.deinit();
        const p = if (path) |file| try prepareFile(gpu, t, file) else try preparePage(gpu, t);
        errdefer gpu.gpa.free(p.pixels);
        const out = try cuda.DeviceBuffer.alloc(gpu.d, @as(usize, p.tokens) * t.width * 2);
        return .{ .gpu = gpu, .s = s, .t = t, .pixels = p.pixels, .image = .{ .pixels = p.pixels, .gh = p.gh, .gw = p.gw, .at = 0, .tokens = p.tokens, .hash = p.hash }, .out = out };
    }

    fn deinit(o: *Opened) void {
        o.t.deinit(); // drains the stream first
        o.out.free();
        o.gpu.gpa.free(o.pixels);
        o.s.deinit();
    }

    /// One encode into `out`, waited for: its milliseconds.
    fn encode(o: *Opened, out: u64, tap: ?dense.recipe.Tap) !f64 {
        const t0 = std.Io.Clock.awake.now(o.gpu.io);
        try o.t.encode(o.s, &o.image, out, tap);
        try o.s.synchronize();
        const t1 = std.Io.Clock.awake.now(o.gpu.io);
        return @as(f64, @floatFromInt(t0.durationTo(t1).toNanoseconds())) / 1e6;
    }

    fn rows(o: *const Opened, out: u64) ![]u8 {
        const host = try o.gpu.gpa.alloc(u8, @as(usize, o.image.tokens) * o.t.width * 2);
        errdefer o.gpu.gpa.free(host);
        try cuda.memory.download(o.gpu.d, out, host);
        return host;
    }
};

/// A tap that writes each stage's rows as `<dir>/<stage>.npy`: <u2 BF16 bit patterns, shape [n, width].
const Npy = struct {
    gpu: Gpu,
    s: cuda.Stream,
    dir: []const u8,
    count: usize = 0,

    fn at(ctx: *anyopaque, stage: []const u8, rows: u64, n: usize, width: usize) anyerror!void {
        const w: *Npy = @ptrCast(@alignCast(ctx));
        const a = w.gpu.gpa;
        try w.s.synchronize(); // the stage's launches are on the tower's stream
        const host = try a.alloc(u8, n * width * 2);
        defer a.free(host);
        try cuda.memory.download(w.gpu.d, rows, host);
        const path = try std.fmt.allocPrint(a, "{s}/{s}.npy", .{ w.dir, stage });
        defer a.free(path);
        try writeNpy(w.gpu.io, path, host, n, width);
        w.count += 1;
    }
};

/// A version 1 .npy of `data` as <u2 [n, width], its header padded to 64 bytes with the preamble.
fn writeNpy(io: std.Io, path: []const u8, data: []const u8, n: usize, width: usize) !void {
    var head: [128]u8 = undefined;
    const dict = try std.fmt.bufPrint(&head, "{{'descr': '<u2', 'fortran_order': False, 'shape': ({d}, {d}), }}", .{ n, width });
    const len = std.mem.alignForward(usize, 10 + dict.len + 1, 64) - 10;
    var pad: [128]u8 = @splat(' ');
    @memcpy(pad[0..dict.len], dict);
    pad[len - 1] = '\n';
    var f = try std.Io.Dir.cwd().createFile(io, path, .{});
    defer f.close(io);
    var wb: [1 << 16]u8 = undefined;
    var w = f.writer(io, &wb);
    try w.interface.writeAll("\x93NUMPY\x01\x00");
    try w.interface.writeInt(u16, @intCast(len), .little);
    try w.interface.writeAll(pad[0..len]);
    try w.interface.writeAll(data);
    try w.interface.flush();
}

/// dots-stages MODEL_DIR IMAGE OUT_DIR: one encode, each stage's rows written for the oracle's `stages` mode.
fn stages(gpu: Gpu, rest: Args) !void {
    const dir = try arg(rest, 0);
    if (!present(gpu.io, dir)) return skip("dots-stages", "no checkpoint config.json in MODEL_DIR");
    const out = try arg(rest, 2);
    var o = try Opened.init(gpu, dir, try arg(rest, 1));
    defer o.deinit();
    try std.Io.Dir.cwd().createDirPath(gpu.io, out);
    var w: Npy = .{ .gpu = gpu, .s = o.s, .dir = out };
    const ms = try o.encode(o.out.ptr, .{ .ctx = &w, .at = Npy.at });
    try check.expect(w.count == taps, "dots-stages: {d} stages written, expected {d}", .{ w.count, taps });
    check.pass("dots-stages: {d} patches, {d} tokens, {d} stages in {s} ({d:.1} ms with the copies)", .{ @as(usize, o.image.gh) * o.image.gw, o.image.tokens, w.count, out, ms });
}

/// dots-image MODEL_DIR [IMAGE]: the generated page and IMAGE each twice byte-equal, then the page against its anchor.
fn twice(gpu: Gpu, rest: Args) !void {
    const dir = try arg(rest, 0);
    if (!present(gpu.io, dir)) return skip("dots-image", "no checkpoint config.json in MODEL_DIR");
    const generated = try mergedHash(gpu, dir, null);
    if (rest.len > 1) _ = try mergedHash(gpu, dir, rest[1]);
    try check.expect(std.mem.eql(u8, &generated, synthetic_merged_sha256), "dots-image: the generated page's vision-merged sha256 {s}, its anchor {s}", .{ &generated, synthetic_merged_sha256 });
    check.pass("dots-image: the generated page's vision-merged sha256 equals its anchor", .{});
}

/// One image (the generated page without `path`) encoded twice, byte-equal: its vision-merged rows' sha256 in hex.
fn mergedHash(gpu: Gpu, dir: []const u8, path: ?[]const u8) ![64]u8 {
    var o = try Opened.init(gpu, dir, path);
    defer o.deinit();
    var second = try cuda.DeviceBuffer.alloc(gpu.d, o.out.len);
    defer second.free();
    errdefer cuda.completion.drain(gpu.d, o.s, "dots-image: a failed encode's work did not drain before its rows were freed");
    const ms = [2]f64{ try o.encode(o.out.ptr, null), try o.encode(second.ptr, null) };
    const a = try o.rows(o.out.ptr);
    defer gpu.gpa.free(a);
    const b = try o.rows(second.ptr);
    defer gpu.gpa.free(b);
    try check.sameBytes("dots-image: the second encode's vision-merged", b, a);
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(a, &digest, .{});
    const hex = std.fmt.bytesToHex(digest, .lower);
    const what = path orelse "the generated page";
    std.debug.print("RESULT dots-image {s}: encode twice equal, {d} x {d} patches to {d} rows; vision-merged sha256 {s}; encode {d:.1} ms, {d:.1} ms\n", .{ what, o.image.gh, o.image.gw, o.image.tokens, &hex, ms[0], ms[1] });
    return hex;
}
