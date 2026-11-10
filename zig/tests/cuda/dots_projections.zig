//! dots.mocr's projections on the GB10: each class's input and output rows, from one encode and one prompt chunk.
const std = @import("std");
const cuda = @import("cuda");
const dots = @import("dots_mocr_cuda");
const dense = @import("cuda_dense");
const check = @import("check.zig");
const t = @import("dots_tests.zig");
const Gpu = check.Gpu;

const Tower = dots.tower.Tower;
const native = dots.native;
const R = dots.recipe;
const vit = cuda.vit;
const Target = @FieldType(native.Loaded, "target");

/// Rows a class keeps: evenly spaced from its first to its last.
const sample = 256;
/// The tower blocks and decoder layers checked: the first, a middle one (tower) and the last.
const tower_blocks = [_]usize{ 0, 20, 41 };
/// The window the engine opens with: the prompt's first chunk and its pages.
const context = 4096;

const Side = enum { tower, decoder };

/// One projection's weights as the checkpoint names them, and its bias when it has one.
const Names = struct { weights: []const []const u8, bias: ?[]const u8 = null };

/// The tap that writes the checked classes' rows, their fused parts' separate GEMMs, and the manifest.
const Dump = struct {
    gpu: Gpu,
    s: cuda.Stream,
    dir: []const u8,
    arena: std.heap.ArenaAllocator,
    side: Side = .tower,
    tower: *Tower,
    gb: Target,
    last_layer: usize,
    entries: std.ArrayList([]const u8) = .empty,
    x: ?u64 = null, // the pending class's input rows (its .x tap)
    k: usize = 0,
    layer: usize = 0, // the layer the last tap named

    fn a(d: *Dump) std.mem.Allocator {
        return d.arena.allocator();
    }

    fn wanted(d: *const Dump, class: []const u8, layer: usize) bool {
        if (d.side == .decoder) return layer == 0 or layer == d.last_layer;
        if (std.mem.eql(u8, class, "patch") or std.mem.startsWith(u8, class, "mlp")) return true;
        return std.mem.indexOfScalar(usize, &tower_blocks, layer) != null;
    }

    fn at(ctx: *anyopaque, stage: []const u8, rows: u64, n: usize, width: usize) anyerror!void {
        const d: *Dump = @ptrCast(@alignCast(ctx));
        if (std.mem.startsWith(u8, stage, "vision-")) return;
        var it = std.mem.splitScalar(u8, stage, '.');
        const class = it.next() orelse return error.BadTap;
        const layer = try std.fmt.parseInt(usize, it.next() orelse return error.BadTap, 10);
        const kind = it.next() orelse return error.BadTap;
        if (!d.wanted(class, layer)) return;
        d.layer = layer;
        try d.s.synchronize(); // the rows' launches are on the engine's stream
        if (std.mem.eql(u8, kind, "x")) {
            d.x = rows;
            d.k = width;
            if (d.side == .tower and std.mem.eql(u8, class, "mlp0")) return d.mlp0(rows, n, width);
            return;
        }
        // fp32 sums tapped as two bf16 columns each
        const f32_out = d.side == .decoder and std.mem.eql(u8, class, "qkv");
        try d.entry(class, layer, d.x orelse return error.BadTap, d.k, rows, n, if (f32_out) width / 2 else width, f32_out);
        d.x = null;
    }

    /// The merger's first linear alone (its GELU is fused in the encode), on the tower's own plan for the shape.
    fn mlp0(d: *Dump, x: u64, n: usize, k: usize) !void {
        const tw = d.tower;
        var acc = try cuda.DeviceBuffer.alloc(d.gpu.d, n * k * 4);
        defer acc.free();
        var y = try cuda.DeviceBuffer.alloc(d.gpu.d, n * k * 2);
        defer y.free();
        errdefer cuda.completion.drain(d.gpu.d, d.s, "dots-projections: a failed mlp0 did not drain before its rows were freed");
        try vit.linear(&tw.gemms, &tw.k.vit, d.s, acc.ptr, x, tw.w.mlp0, tw.w.mlp0_b, y.ptr, n, k, k);
        try d.s.synchronize();
        try d.entry("mlp0", 0, x, k, y.ptr, n, k, false);
        d.x = null;
    }

    /// A class's sampled rows of x and y, its fused parts' separate GEMMs, and its manifest line.
    fn entry(d: *Dump, class: []const u8, layer: usize, x: u64, k: usize, y: u64, n: usize, width: usize, f32_out: bool) !void {
        const al = d.a();
        const name = try std.fmt.allocPrint(al, "{t}-{s}-{d}", .{ d.side, class, layer });
        const idx = try spread(al, n);
        try d.allTo(name, x, k, n); // every row: Torch runs the layer on the call's own rows
        const yd = if (f32_out) "<f4" else "<u2";
        const ys: usize = if (f32_out) 4 else 2;
        try d.rowsTo(name, "y", yd, y, width * ys, idx);
        const names = try d.namesOf(class, layer);
        var out: std.Io.Writer.Allocating = .init(al);
        const w = &out.writer;
        try w.print("{{\"name\":\"{s}\",\"side\":\"{t}\",\"class\":\"{s}\",\"layer\":{d},\"k\":{d},\"n\":{d},\"out\":\"{s}\",\"rows\":{d},\"sample\":[", .{ name, d.side, class, layer, k, width, if (f32_out) "f32" else "bf16", n });
        for (idx, 0..) |r, i| try w.print("{s}{d}", .{ if (i == 0) "" else ",", r });
        try w.writeAll("],\"weights\":");
        try json(w, names.weights);
        if (names.bias) |b| try w.print(",\"bias\":\"{s}\"", .{b}) else try w.writeAll(",\"bias\":null");
        try w.writeAll(",\"splits\":[");
        const parts = try d.fusedParts(class);
        var col: usize = 0;
        for (parts, 0..) |p, i| {
            const file = try std.fmt.allocPrint(al, "split{d}", .{i});
            var buf = try cuda.DeviceBuffer.alloc(d.gpu.d, n * p.n * ys);
            defer buf.free();
            errdefer cuda.completion.drain(d.gpu.d, d.s, "dots-projections: a failed split GEMM did not drain before its rows were freed");
            try d.separate(p, x, k, n, buf.ptr, f32_out);
            try d.s.synchronize();
            try d.rowsTo(name, file, yd, buf.ptr, p.n * ys, idx);
            try w.print("{s}{{\"y\":\"{s}.{s}.npy\",\"col\":{d},\"n\":{d}}}", .{ if (i == 0) "" else ",", name, file, col, p.n });
            col += p.n;
        }
        try w.writeAll("]}");
        try d.entries.append(al, out.written());
    }

    /// All `n` input rows [n, k] bf16 at `src` to `<dir>/<name>.x.npy`, one copy.
    fn allTo(d: *Dump, name: []const u8, src: u64, k: usize, n: usize) !void {
        const host = try d.gpu.gpa.alloc(u8, n * k * 2);
        defer d.gpu.gpa.free(host);
        try cuda.memory.download(d.gpu.d, src, host);
        const path = try std.fmt.allocPrint(d.a(), "{s}/{s}.x.npy", .{ d.dir, name });
        try t.writeNpy(d.gpu.io, path, "<u2", host, n, k);
    }

    /// Rows `idx` of [*, row_bytes] at `src` to `<dir>/<name>.<part>.npy`, one copy a row.
    fn rowsTo(d: *Dump, name: []const u8, part: []const u8, descr: []const u8, src: u64, row_bytes: usize, idx: []const usize) !void {
        const host = try d.gpu.gpa.alloc(u8, idx.len * row_bytes);
        defer d.gpu.gpa.free(host);
        for (idx, 0..) |r, i| try cuda.memory.download(d.gpu.d, src + r * row_bytes, host[i * row_bytes ..][0..row_bytes]);
        const path = try std.fmt.allocPrint(d.a(), "{s}/{s}.{s}.npy", .{ d.dir, name, part });
        try t.writeNpy(d.gpu.io, path, descr, host, idx.len, row_bytes / if (descr[1] == 'f') @as(usize, 4) else 2);
    }

    /// The checkpoint's tensors behind a class.
    fn namesOf(d: *Dump, class: []const u8, layer: usize) !Names {
        const al = d.a();
        const eq = std.mem.eql;
        if (d.side == .tower) {
            const v = "vision_tower.";
            if (eq(u8, class, "patch")) return .{ .weights = &.{v ++ "patch_embed.patchifier.proj.weight"} };
            if (eq(u8, class, "mlp0")) return .{ .weights = &.{v ++ "merger.mlp.0.weight"}, .bias = v ++ "merger.mlp.0.bias" };
            if (eq(u8, class, "mlp2")) return .{ .weights = &.{v ++ "merger.mlp.2.weight"}, .bias = v ++ "merger.mlp.2.bias" };
            const tensors: []const []const u8 = if (eq(u8, class, "qkv")) &.{"attn.qkv"} else if (eq(u8, class, "proj")) &.{"attn.proj"} else if (eq(u8, class, "fc13")) &.{ "mlp.fc1", "mlp.fc3" } else &.{"mlp.fc2"};
            return .{ .weights = try each(al, v ++ "blocks.{d}.{s}.weight", layer, tensors) };
        }
        if (eq(u8, class, "head")) return .{ .weights = &.{"lm_head.weight"} };
        const tensors: []const []const u8 = if (eq(u8, class, "qkv")) &.{ "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj" } else if (eq(u8, class, "o")) &.{"self_attn.o_proj"} else if (eq(u8, class, "gate_up")) &.{ "mlp.gate_proj", "mlp.up_proj" } else &.{"mlp.down_proj"};
        return .{ .weights = try each(al, "model.layers.{d}.{s}.weight", layer, tensors) };
    }

    /// A fused class's parts, each a GEMM of its own: fc1 | fc3 (tower), q | k | v and gate | up (decoder).
    const Part = struct { row: usize, n: usize };
    fn fusedParts(d: *Dump, class: []const u8) ![]const Part {
        const eq = std.mem.eql;
        const g = d.gb.m.g;
        if (d.side == .tower and eq(u8, class, "fc13")) return d.a().dupe(Part, &.{ .{ .row = 0, .n = d.tower.c.intermediate }, .{ .row = d.tower.c.intermediate, .n = d.tower.c.intermediate } });
        if (d.side == .decoder and eq(u8, class, "qkv")) return d.a().dupe(Part, &.{ .{ .row = 0, .n = g.qWidth() }, .{ .row = g.qWidth(), .n = g.kvWidth() }, .{ .row = g.qWidth() + g.kvWidth(), .n = g.kvWidth() } });
        if (d.side == .decoder and eq(u8, class, "gate_up")) return d.a().dupe(Part, &.{ .{ .row = 0, .n = g.mid }, .{ .row = g.mid, .n = g.mid } });
        return &.{};
    }

    /// One part's GEMM on the fused one's input rows and path: vit's (tower) or the lane's (decoder).
    fn separate(d: *Dump, p: Part, x: u64, k: usize, n: usize, out: u64, f32_out: bool) !void {
        if (d.side == .tower) {
            const b = d.tower.w.blocks[d.layer];
            return d.tower.gemms.run(d.s, x, b.fc13 + p.row * k * 2, out, n, p.n, k, .bf16);
        }
        const l = d.gb.e.w.layers[d.layer];
        const fused = if (f32_out) l.qkv else l.gate_up;
        if (p.row % 64 != 0) return error.BadSplit; // the BF16 lane packs rows in tiles of 64
        const w = cuda.modelopt.bf16Weight(fused.codes + p.row * k * 2, p.n, k);
        const lane = d.gb.e.k.lane;
        if (lane.partBytes(n, w) > d.gb.m.part_bytes) return error.LanePartTooSmall;
        if (f32_out) return lane.matmulF32(d.s, x, k, n, w, out, d.gb.m.s.part);
        return lane.matmul(d.s, x, k, n, w, out, d.gb.m.s.part);
    }
};

/// `n` rows' sample: `sample` of them evenly spaced, the first and the last included (all of them when fewer).
fn spread(a: std.mem.Allocator, n: usize) ![]usize {
    const s = @min(sample, n);
    const out = try a.alloc(usize, s);
    for (out, 0..) |*r, i| r.* = if (s == 1) 0 else i * (n - 1) / (s - 1);
    return out;
}

fn each(a: std.mem.Allocator, comptime fmt: []const u8, layer: usize, parts: []const []const u8) ![]const []const u8 {
    const out = try a.alloc([]const u8, parts.len);
    for (out, parts) |*o, p| o.* = try std.fmt.allocPrint(a, fmt, .{ layer, p });
    return out;
}

fn json(w: *std.Io.Writer, names: []const []const u8) !void {
    try w.writeAll("[");
    for (names, 0..) |s, i| try w.print("{s}\"{s}\"", .{ if (i == 0) "" else ",", s });
    try w.writeAll("]");
}

/// dots-projections MODEL_DIR IMAGE OUT_DIR: the rows tools/zig/dots_projection_check.py checks against FP64 and Torch.
pub fn run(gpu: Gpu, rest: []const [:0]const u8) !void {
    const dir = try t.arg(rest, 0);
    if (!t.present(gpu.io, dir)) return t.skip("dots-projections", "no checkpoint config.json in MODEL_DIR");
    const out = try t.arg(rest, 2);
    const gpa = gpu.gpa;
    try std.Io.Dir.cwd().createDirPath(gpu.io, out);
    var loaded = try native.open(gpa, gpu.io, gpu.ctx, dir, null, .{ .context = context, .drafts = false, .lanes = 1, .vision = .{ .image_tokens = Tower.default_image_tokens, .max_images = 1 } });
    defer loaded.deinit(loaded.ctx);
    try native.attach(loaded.ctx, 0, dense.pages.perSlot(context));
    const gb = loaded.target;
    const tw = loaded.vision.?;
    var d: Dump = .{ .gpu = gpu, .s = gb.e.s, .dir = out, .arena = .init(gpa), .tower = tw, .gb = gb, .last_layer = gb.m.g.layers - 1 };
    defer d.arena.deinit();
    const tap: dense.recipe.Tap = .{ .ctx = &d, .at = Dump.at };
    // the tower: one real encode of the image, its classes tapped as they run
    const bytes = try t.readFile(gpu, try t.arg(rest, 1));
    defer gpa.free(bytes);
    const p = try tw.prepare(gpa, bytes, Tower.default_image_tokens);
    defer gpa.free(p.pixels);
    var feats = try cuda.DeviceBuffer.alloc(gpu.d, @as(usize, p.tokens) * tw.width * 2);
    defer feats.free();
    errdefer cuda.completion.drain(gpu.d, gb.e.s, "dots-projections: a failed encode or pass did not drain before its rows were freed");
    var im: dense.recipe.Image = .{ .pixels = p.pixels, .gh = p.gh, .gw = p.gw, .at = 0, .tokens = p.tokens, .hash = p.hash };
    try tw.encode(gb.e.s, &im, feats.ptr, tap);
    try gb.e.s.synchronize();
    // the decoder: the image prompt's first chunk, its rows spliced in, every layer's classes tapped
    d.side = .decoder;
    var prompt: t.Prompt = .{};
    defer prompt.deinit(gpa);
    var tok = try @import("core").tokenizer.loadTokenizer(gpu.io, gpa, dir);
    defer tok.deinit();
    const user = try tok.encode(gpa, "<|user|>");
    defer gpa.free(user);
    const ask = try tok.encode(gpa, "Extract the text of this image.<|endofuser|><|assistant|>");
    defer gpa.free(ask);
    try prompt.text(gpa, user);
    try prompt.image(gpa, im);
    try prompt.text(gpa, ask);
    im = prompt.images.items[0];
    const rows = @min(prompt.ids.items.len, dense.backend.prompt_rows);
    var ids: [dense.backend.prompt_rows / dense.pages.page_rows]u32 = undefined;
    for (&ids, 0..) |*id, i| id.* = @intCast(i);
    try gb.m.mapPages(gb.e.s, 0, ids[0..dense.pages.pagesOf(rows)], 0);
    const inject: dense.pass.Inject = .{ .row = im.at, .rows = @min(im.tokens, rows - im.at), .src = feats.ptr };
    const chain: dense.pass.Chain = .{ .slot = 0, .p0 = 0, .tokens = prompt.ids.items[0..rows], .images = &.{inject} };
    const heads = @min(dense.backend.round_rows, rows);
    const e: dense.pass.Engine(R) = .{ .m = gb.e.m, .k = gb.e.k, .own = gb.e.own, .w = gb.e.w, .s = gb.e.s, .taps = tap };
    try dense.pass.forward(R, e, &.{chain}, true, rows - heads, heads);
    try gb.e.s.synchronize();
    // the head: the last rows' final norm against their logits
    const g = gb.m.g;
    try d.entry("head", 0, gb.m.s.x + (rows - heads) * g.hidden * 2, g.hidden, gb.m.s.logits, heads, g.vocab, false);
    try writeManifest(&d, rows, p);
    check.pass("dots-projections: {d} classes from a {d} x {d} image and {d} prompt rows in {s}", .{ d.entries.items.len, p.gh, p.gw, rows, out });
}

fn writeManifest(d: *Dump, rows: usize, p: anytype) !void {
    var out: std.Io.Writer.Allocating = .init(d.gpu.gpa);
    defer out.deinit();
    const w = &out.writer;
    try w.print("{{\"grid\":[{d},{d}],\"prompt_rows\":{d},\"entries\":[\n", .{ p.gh, p.gw, rows });
    for (d.entries.items, 0..) |e, i| try w.print("{s}{s}\n", .{ e, if (i + 1 == d.entries.items.len) "" else "," });
    try w.writeAll("]}\n");
    const path = try std.fmt.allocPrint(d.a(), "{s}/manifest.json", .{d.dir});
    try std.Io.Dir.cwd().writeFile(d.gpu.io, .{ .sub_path = path, .data = out.written() });
}
