//! dots.mocr on the GB10: the tower's stage taps, its encode twice equal and anchored, image prompts' chunk cuts.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const lanes = @import("lanes");
const dots = @import("dots_mocr_cuda");
const dense = @import("cuda_dense");
const check = @import("check.zig");
const Gpu = check.Gpu;

const Tower = dots.tower.Tower;
const native = dots.native;
const Image = dense.recipe.Image;
const Prepared = core.picture.patches.Prepared;
const Args = []const [:0]const u8;
/// The window the engine serves here: the full image's prompt and its reply.
const context = 8192;

/// The stages one encode taps: the patch with its bias and its norm, every block, the post-norm, the merger.
const taps = 2 + 42 + 2;

/// dots-image's own page: the processor fixture's pixel pattern at a size of whole 28-pixel blocks, so no file is read.
const synthetic = .{ .width = 812, .height = 588, .seed = 2732 };
/// The generated page's vision-merged sha256 on the GB10: a tower drift fails dots-image without the Python oracle.
const synthetic_merged_sha256 = "ac8c66b76552c15c22af3bc776d285350c4d9596863de2a3359c9a909b971067";
/// The SM the pins were measured on (GB10); another device's rows are checked twice-equal, not against them.
const pin_sm = 121;

/// dots-text's greedy reply ids a prompt of our own (fixtures/own-prompts.json), fewer when EOS ends it first.
const reply_ids = 64;
/// The sha256 of each own prompt's reply ids (u32 little-endian) on the GB10: a decoder drift fails dots-text.
const short_reply_sha256 = "4136394a1b94f5d96ae2fbaab8607c3ad793e174cce0578a93b103cf6e68f480";
const page_reply_sha256 = "37b4105c553e4b6ce06c2afd7c2a175e15205d2f3afce68b1e3f9d8cf9650bef";
const chunks_reply_sha256 = "ed7e831a46ad3d4d48c3b9cb52c14bba19b7a270c70a4d623f91320212507b4e";

const usage =
    \\  dots-stages MODEL IMAGE OUT  dots.mocr's tower: one encode, each stage's rows as OUT/<stage>.npy for the oracle
    \\  dots-image MODEL [IMAGE]  dots.mocr's tower: the generated page's vision-merged sha256 against its anchor, then
    \\                            IMAGE's; each encoded twice byte-equal, with each encode's time
    \\  dots-text MODEL             dots.mocr text prompts of our own: <= 64 greedy reply ids each, sha256 against pins
    \\  dots-invariants MODEL IMAGE [N] [IDS]  dots.mocr image prompts: chunk cuts == uncut (pages, logits, N greedy ids)
    \\
;

/// tf-cuda-test dots-*: main.zig's one line for them; dots-help prints their usage.
pub fn run(gpu: Gpu, cmd: []const u8, rest: Args) !void {
    if (std.mem.eql(u8, cmd, "dots-help")) return std.debug.print("{s}", .{usage});
    if (!cuda.kernels.available) return skip(cmd, "this runner was built without CUDA kernels");
    if (std.mem.eql(u8, cmd, "dots-stages")) return stages(gpu, rest);
    if (std.mem.eql(u8, cmd, "dots-image")) return twice(gpu, rest);
    if (std.mem.eql(u8, cmd, "dots-text")) return replies(gpu, rest);
    if (std.mem.eql(u8, cmd, "dots-invariants")) return invariants(gpu, rest);
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
    if (try otherSm(gpu)) |sm| return skipPin("dots-image", sm);
    try check.expect(std.mem.eql(u8, &generated, synthetic_merged_sha256), "dots-image: the generated page's vision-merged sha256 {s}, its anchor {s}", .{ &generated, synthetic_merged_sha256 });
    check.pass("dots-image: the generated page's vision-merged sha256 equals its anchor", .{});
}

/// The device's SM when it is not the pins' (sm_121), else null.
fn otherSm(gpu: Gpu) !?u32 {
    const sm: u32 = @intCast(try gpu.ctx.capability());
    return if (sm == pin_sm) null else sm;
}

/// The pins are GB10 values: on a device of another SM, `cmd` names it and compares none.
fn skipPin(cmd: []const u8, sm: u32) void {
    std.debug.print("SKIP {s}'s pins: they are GB10 (sm_{d}) values; this device is sm_{d}\n", .{ cmd, pin_sm, sm });
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

/// One run's results: its KV rows' hash, last-row logits, greedy ids, images encoded and where its chunks ended.
const Out = struct { kv: [32]u8, logits: []u8, ids: []u32, encodes: u64, ends: []u32 };

/// The engine with --vision on one slot, its lane host and its tower.
const Served = struct {
    gpu: Gpu,
    loaded: native.Loaded,
    cfg: lanes.Config,
    clock: lanes.fake.FixedClock = .{},
    engine: lanes.Engine,
    eos: []const u32,

    fn init(s: *Served, gpu: Gpu, dir: []const u8, pages: usize, eos: []const u32) !void {
        s.gpu = gpu;
        s.eos = eos;
        s.loaded = try native.open(gpu.gpa, gpu.io, gpu.ctx, dir, null, .{ .context = context, .drafts = false, .lanes = 1, .vision = .{ .image_tokens = Tower.default_image_tokens, .max_images = 2 } });
        errdefer s.loaded.deinit(s.loaded.ctx);
        try native.attach(s.loaded.ctx, 0, pages);
        s.cfg = try lanes.Config.init(gpu.gpa, s.loaded.facts, s.loaded.rows, s.loaded.rows - 1);
        s.clock = .{};
        s.engine = lanes.Engine.init(gpu.gpa, &s.cfg, s.loaded.backend, s.clock.clock());
    }

    fn deinit(s: *Served) void {
        s.engine.deinit();
        s.cfg.deinit(s.gpu.gpa);
        s.loaded.deinit(s.loaded.ctx);
    }

    fn tower(s: *const Served) *Tower {
        return s.loaded.vision.?;
    }

    /// One stream cut at `cuts` beside the 1,024-row grid: its pages and logits after the prompt, then its reply.
    fn once(s: *Served, prompt: []const u32, images: []const Image, cuts: []const u32, n: u32) !Out {
        const gpa = s.gpu.gpa;
        const before = s.tower().encodes;
        var st = try lanes.Stream.init(gpa, .{ .id = "dots", .prompt = prompt, .max_new = n, .eos = s.eos, .drafts = false, .images = images, .reuse = .{ .marks = cuts } });
        defer st.deinit(gpa);
        // the backend's prompt steps alone, so the lane is read before the first token can end the stream
        var ends: std.ArrayList(u32) = .empty;
        errdefer ends.deinit(gpa);
        const b = s.loaded.backend;
        while (true) {
            const done = try b.vtable.prefill_step.?(b.ptr, &st, 0);
            try ends.append(gpa, @intCast(st.fill_at.load(.monotonic)));
            if (done) break;
        }
        const gb = s.loaded.target;
        try gb.e.s.synchronize();
        const l = gb.lanes.get(&st) orelse return error.UnknownStream;
        const logits = try gpa.alloc(u8, gb.m.g.vocab * 2);
        errdefer gpa.free(logits);
        try cuda.memory.download(s.gpu.d, l.last, logits);
        const kv = try pagesHash(s.gpu, gb.m, l.lease.ids.items, prompt.len);
        try s.engine.opened(&st);
        while (s.engine.activeCount() > 0) try s.engine.step();
        const ids = try gpa.dupe(u32, st.emitted());
        errdefer gpa.free(ids);
        return .{ .kv = kv, .logits = logits, .ids = ids, .encodes = s.tower().encodes - before, .ends = try ends.toOwnedSlice(gpa) };
    }
};

/// The sha256 of every layer's keys and values at positions [0, len), page by page in position order.
fn pagesHash(gpu: Gpu, m: anytype, ids: []const u32, len: usize) ![32]u8 {
    const rows_a_page = dense.pages.page_rows;
    const kvw = m.g.kvWidth();
    const page = try gpu.gpa.alloc(u8, m.pageBytes());
    defer gpu.gpa.free(page);
    var h: std.crypto.hash.sha2.Sha256 = .init(.{});
    for (ids, 0..) |id, p| {
        if (p * rows_a_page >= len) break;
        const rows = @min(rows_a_page, len - p * rows_a_page);
        try cuda.memory.download(gpu.d, m.arena + id * m.pageBytes(), page);
        for (0..2 * m.g.layers) |part| h.update(page[part * rows_a_page * kvw * 2 ..][0 .. rows * kvw * 2]);
    }
    return h.finalResult();
}

/// A prompt of the template's markers: the user mark, each image's open, pads and close, then the ask.
const Prompt = struct {
    ids: std.ArrayList(u32) = .empty,
    images: std.ArrayList(Image) = .empty,

    fn deinit(p: *Prompt, gpa: std.mem.Allocator) void {
        p.ids.deinit(gpa);
        p.images.deinit(gpa);
    }

    fn text(p: *Prompt, gpa: std.mem.Allocator, ids: []const u32) !void {
        try p.ids.appendSlice(gpa, ids);
    }

    fn image(p: *Prompt, gpa: std.mem.Allocator, prepared: Image) !void {
        const m = dots.tower.marks;
        try p.ids.append(gpa, m.open);
        var im = prepared;
        im.at = @intCast(p.ids.items.len);
        try p.ids.appendNTimes(gpa, m.image, im.tokens);
        try p.ids.append(gpa, m.close);
        try p.images.append(gpa, im);
    }
};

fn free(gpa: std.mem.Allocator, o: Out) void {
    gpa.free(o.logits);
    gpa.free(o.ids);
    gpa.free(o.ends);
}

/// Whether every cut ended a chunk of the cut run and none of the uncut run's: the schedule really changed.
fn taken(cuts: []const u32, got: []const u32, base: []const u32) bool {
    for (cuts) |c| if (std.mem.indexOfScalar(u32, got, c) == null or std.mem.indexOfScalar(u32, base, c) != null) return false;
    return true;
}

/// `x`, moved one row on when the 1,024-row grid already ends a chunk there.
fn offGrid(x: u32) u32 {
    return if (x % dense.backend.prompt_rows == 0) x + 1 else x;
}

/// Each cut run against the case's uncut (normal-grid) run: pages, last-row logits, replies and one encode an image.
fn compare(s: *Served, name: []const u8, p: *const Prompt, cuts: []const []const u32, n: u32) !usize {
    const gpa = s.gpu.gpa;
    const base = try s.once(p.ids.items, p.images.items, &.{}, n);
    defer free(gpa, base);
    var bad: usize = 0;
    const images = p.images.items.len;
    if (base.encodes != images) bad += 1;
    std.debug.print("RESULT dots {s}: {d} rows, {d} images, normal grid: {d} ids, {d} encodes\n", .{ name, p.ids.items.len, images, base.ids.len, base.encodes });
    for (cuts) |c| {
        const o = try s.once(p.ids.items, p.images.items, c, n);
        defer free(gpa, o);
        const same = [_]bool{ taken(c, o.ends, base.ends), std.mem.eql(u8, &o.kv, &base.kv), std.mem.eql(u8, o.logits, base.logits), std.mem.eql(u32, o.ids, base.ids), o.encodes == images };
        for (same) |ok| bad += @intFromBool(!ok);
        std.debug.print("RESULT dots {s}, cuts {any}: chunk ends {any} ({s}), pages {s}, last logits {s}, ids {s}, encodes {d}\n", .{ name, c, o.ends, if (same[0]) "taken" else "NOT TAKEN", word(same[1]), word(same[2]), word(same[3]), o.encodes });
    }
    return bad;
}

fn word(same: bool) []const u8 {
    return if (same) "same" else "DIFFER";
}

/// The prompt in `path` (the oracle's input-input_ids.npy, int64 [1, L]) with the demo image at its one pad run.
fn refPrompt(gpu: Gpu, path: []const u8, demo: Image) !Prompt {
    const bytes = try readFile(gpu, path);
    defer gpu.gpa.free(bytes);
    const arr = try core.npy.parse(bytes);
    if (!std.mem.eql(u8, arr.descr, "<i8")) return error.BadIds;
    var p: Prompt = .{};
    errdefer p.deinit(gpu.gpa);
    for (std.mem.bytesAsSlice(i64, arr.data)) |id| try p.ids.append(gpu.gpa, @intCast(id));
    const pad = dots.tower.marks.image;
    const at = std.mem.indexOfScalar(u32, p.ids.items, pad) orelse return error.BadIds;
    var end = at;
    while (end < p.ids.items.len and p.ids.items[end] == pad) end += 1;
    if (end - at != demo.tokens or std.mem.indexOfScalarPos(u32, p.ids.items, end, pad) != null) return error.BadIds;
    var im = demo;
    im.at = @intCast(at);
    try p.images.append(gpu.gpa, im);
    return p;
}

/// dots-invariants MODEL_DIR IMAGE [N] [IDS_NPY]: image prompts under forced chunk cuts equal their uncut runs.
fn invariants(gpu: Gpu, rest: Args) !void {
    const dir = try arg(rest, 0);
    if (!present(gpu.io, dir)) return skip("dots-invariants", "no checkpoint config.json in MODEL_DIR");
    const gpa = gpu.gpa;
    const n: u32 = if (rest.len > 2) try std.fmt.parseInt(u32, rest[2], 10) else 64;
    const c = try dots.config.Config.read(gpa, gpu.io, dir);
    var tok = try core.tokenizer.loadTokenizer(gpu.io, gpa, dir);
    defer tok.deinit();
    const user = try tok.encode(gpa, "<|user|>");
    defer gpa.free(user);
    const ask = try tok.encode(gpa, "Extract the text of this image.<|endofuser|><|assistant|>");
    defer gpa.free(ask);
    const between = try tok.encode(gpa, " and ");
    defer gpa.free(between);
    var s: Served = undefined;
    try s.init(gpu, dir, dense.pages.perSlot(context), c.end_tokens[0..c.end_count]);
    defer s.deinit();
    const bytes = try readFile(gpu, try arg(rest, 1));
    defer gpa.free(bytes);
    // three budgets: inside one 1,024-row chunk, across the grid line at 1,024, and the full (the demo's)
    var prepared: [3]Image = undefined;
    var made: usize = 0;
    defer for (prepared[0..made]) |im| gpa.free(im.pixels);
    for (&prepared, [_]u32{ 64, 1200, Tower.default_image_tokens }) |*im, budget| {
        const p = try s.tower().prepare(gpa, bytes, budget);
        im.* = .{ .pixels = p.pixels, .gh = p.gh, .gw = p.gw, .at = 0, .tokens = p.tokens, .hash = p.hash };
        made += 1;
    }
    var bad: usize = 0;
    { // a small image uncut, then cut at its first row, inside it, at its last row, and at all three
        var p: Prompt = .{};
        defer p.deinit(gpa);
        try p.text(gpa, user);
        try p.image(gpa, prepared[0]);
        try p.text(gpa, ask);
        const im = p.images.items[0];
        const first = im.at;
        const mid = im.at + im.tokens / 2;
        const last = im.at + im.tokens - 1;
        try check.expect(p.ids.items.len <= dense.backend.prompt_rows, "dots-invariants: the small prompt of {d} rows fits one chunk", .{p.ids.items.len});
        bad += try compare(&s, "small image", &p, &.{ &.{first}, &.{mid}, &.{last}, &.{ first, mid, last } }, n);
    }
    { // the full image (the oracle's prompt when given) on the 1,024-row grid, then cut inside the image off the grid
        var p: Prompt = if (rest.len > 3) try refPrompt(gpu, rest[3], prepared[2]) else .{};
        defer p.deinit(gpa);
        if (rest.len <= 3) {
            try p.text(gpa, user);
            try p.image(gpa, prepared[2]);
            try p.text(gpa, ask);
        }
        const im = p.images.items[0];
        bad += try compare(&s, "full image", &p, &.{&.{ offGrid(im.at + 333), offGrid(im.at + im.tokens - 7) }}, n);
    }
    { // two images in one prompt, the grid cutting the first, then cut inside each off the grid
        var p: Prompt = .{};
        defer p.deinit(gpa);
        try p.text(gpa, user);
        try p.image(gpa, prepared[1]);
        try p.text(gpa, between);
        try p.image(gpa, prepared[0]);
        try p.text(gpa, ask);
        const a = p.images.items[0];
        const b = p.images.items[1];
        try check.expect(a.at < dense.backend.prompt_rows and a.at + a.tokens > dense.backend.prompt_rows, "dots-invariants: the grid line at {d} cuts the first image [{d}, {d})", .{ dense.backend.prompt_rows, a.at, a.at + a.tokens });
        bad += try compare(&s, "two images", &p, &.{&.{ offGrid(a.at + a.tokens / 3), offGrid(b.at + b.tokens / 2) }}, n);
    }
    try check.expect(bad == 0, "dots-invariants: {d} differences", .{bad});
    check.pass("dots-invariants: chunk cuts inside, at the first and the last image rows equal the uncut runs (pages, logits, {d} greedy ids), each image encoded once", .{n});
}

/// dots-text MODEL_DIR: each own prompt's greedy reply ids through the served engine, their sha256 against its pin.
fn replies(gpu: Gpu, rest: Args) !void {
    const dir = try arg(rest, 0);
    if (!present(gpu.io, dir)) return skip("dots-text", "no checkpoint config.json in MODEL_DIR");
    const gpa = gpu.gpa;
    const c = try dots.config.Config.read(gpa, gpu.io, dir);
    const own = try dots.ownPrompts(gpa);
    defer own.deinit();
    var s: Served = undefined;
    try s.init(gpu, dir, dense.pages.perSlot(context), c.end_tokens[0..c.end_count]);
    defer s.deinit();
    const cases = [_]struct { []const u8, []const u32, []const u8 }{
        .{ "short", own.value.short, short_reply_sha256 },
        .{ "page", own.value.page, page_reply_sha256 },
        .{ "chunks", own.value.chunks, chunks_reply_sha256 },
    };
    var bad: usize = 0;
    for (cases) |k| {
        const o = try s.once(k[1], &.{}, &.{}, reply_ids);
        defer free(gpa, o);
        var h: std.crypto.hash.sha2.Sha256 = .init(.{});
        for (o.ids) |id| {
            var le: [4]u8 = undefined;
            std.mem.writeInt(u32, &le, id, .little);
            h.update(&le);
        }
        const hex = std.fmt.bytesToHex(h.finalResult(), .lower);
        const same = std.mem.eql(u8, &hex, k[2]);
        bad += @intFromBool(!same);
        std.debug.print("RESULT dots-text {s}: {d} prompt ids, {d} reply ids, sha256 {s}, pinned {s} ({s})\n", .{ k[0], k[1].len, o.ids.len, &hex, k[2], word(same) });
    }
    try check.expect(bad == 0, "dots-text: {d} of {d} replies differ from their pins", .{ bad, cases.len });
    check.pass("dots-text: {d} own prompts' greedy replies (at most {d} ids) equal their pins", .{ cases.len, reply_ids });
}
