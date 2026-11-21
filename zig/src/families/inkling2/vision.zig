//! Inkling 2's vision tower on CUDA in bf16: patch embedding, 23 ViT layers with axial rotary, the 2 x 2 projector.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const image = @import("image.zig");
const config = @import("config.zig");

const vit = cuda.vit;

pub const width = 1024;
pub const heads = 16;
pub const head_dim = 64;
pub const depth = 23;
pub const mlp_inter = 4096;
const merged = image.merge * image.merge * width; // a token's four patch rows side by side
const eps: f32 = 1e-5;
const pre = "model.visual.backbone.";
/// The checkpoint names the tower reads (source shards and the pack's vision sidecar alike).
pub const prefixes = [_][]const u8{"model.visual."};
/// Image tokens a tower's scratch covers when the caller names no limit: 1280 x 1280 pixels.
pub const default_tokens: u32 = 1024;

/// The family's tower kernels (inkling2/vision.cu).
pub const Kernels = struct {
    mod: cuda.Module,
    embed: cuda.Function,
    rope: cuda.Function,

    pub fn load(d: *const cuda.Driver) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var mod = try cuda.Module.load(d, cuda.kernels.inkling2_vision);
        errdefer mod.unload();
        const k: Kernels = .{
            .mod = mod,
            .embed = try mod.function("tf_ink2v_embed"),
            .rope = try mod.function("tf_ink2v_rope"),
        };
        return k;
    }

    pub fn unload(k: *Kernels) void {
        k.mod.unload();
        k.* = undefined;
    }
};

/// The tower's tensors: a pack's vision.safetensors sidecar (as the release stores them), else the model's shards.
fn openTower(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !core.Checkpoint {
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = try std.fmt.bufPrint(&buf, "{s}/" ++ sidecar, .{dir});
    std.Io.Dir.cwd().access(io, path, .{}) catch return core.Checkpoint.openModelPrefixes(gpa, io, dir, &prefixes);
    var ck: core.Checkpoint = .{ .gpa = gpa, .io = io };
    errdefer ck.close();
    try ck.add(dir, sidecar);
    return ck;
}

/// The file a pack carries the tower in, copied unchanged from the release's shards.
pub const sidecar = "vision.safetensors";

fn go(f: cuda.Function, s: cuda.Stream, grid: cuda.Dim3, threads: u32, shared: u32, a: *cuda.Args) !void {
    try cuda.launch.launch(f, .{ .grid = grid, .block = .{ .x = threads }, .shared = shared }, s, a);
}

const Layer = struct { ln1_w: u64, ln1_b: u64, qkv_w: u64, qkv_b: u64, out_w: u64, out_b: u64, ls1: u64, ln2_w: u64, ln2_b: u64, fc1_w: u64, fc1_b: u64, fc2_w: u64, fc2_b: u64, ls2: u64 };

/// What one image row of scratch costs: patch inputs, x/xn/att/y, qkv, mid, the f32 bias sums, tables.
const row_bytes: u64 = image.patch_width * 2 + 4 * width * 2 + 3 * width * 2 + mlp_inter * 2 + mlp_inter * 4 + 4 * 4 + 4 * 4 + 2 * image.pairs * 4;

/// Scratch for pictures of up to `tokens` decoder tokens, plus cuBLASLt's workspace.
pub fn scratchBytes(v: config.Vision, tokens: u32) u64 {
    const rows: u64 = @as(u64, tokens) * image.merge * image.merge + 1;
    return rows * row_bytes + @as(u64, tokens) * v.proj_hidden * 2 + (32 << 20);
}

/// The tower's weights and scratch on one GPU.
pub const Tower = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    k: Kernels,
    vk: vit.Kernels,
    fa: vit.Attention,
    g: vit.Gemms,
    bufs: std.ArrayList(cuda.DeviceBuffer) = .empty,
    max_tokens: u32,
    proj_inter: u32, // config.json's projection_hidden_dim
    hidden: u32, // the decoder's width, config.json's text_hidden_size
    patch_w: u64 = 0, // [width][1200]: a native patch's row, column, channel
    table: u64 = 0, // [32 * 32][width] position rows
    cls: u64 = 0,
    cls_pos: u64 = 0,
    pre_w: u64 = 0,
    pre_b: u64 = 0,
    proj_in: u64 = 0, // [proj_inter][4096], no bias
    proj_out: u64 = 0, // [hidden][proj_inter], no bias
    layers: [depth]Layer = undefined,
    px: u64 = 0, // bf16 [P][1200]
    x: u64 = 0, // bf16 [R][width]: the residual stream (row 0 the class token)
    xn: u64 = 0,
    att: u64 = 0,
    y: u64 = 0,
    qkv: u64 = 0, // bf16 [R][3 width]
    mid: u64 = 0, // bf16 [R][mlp_inter]
    acc: u64 = 0, // f32 [R][mlp_inter]: a Linear's sums when cuBLASLt cannot fuse its bias
    taps: u64 = 0, // u32 [P][4]
    wts: u64 = 0, // f32 [P][4]
    cs: u64 = 0, // f32 [R][32]
    sn: u64 = 0,
    p1: u64 = 0, // bf16 [T][proj_inter]
    bytes: u64 = 0,
    /// Layers encode runs (a parity tool stops early to read a layer's output at `x`).
    layers_run: usize = depth,

    fn alloc(t: *Tower, n: usize) !u64 {
        var b = try cuda.DeviceBuffer.alloc(t.d, @max(n, 256));
        errdefer b.free();
        try t.bufs.append(t.gpa, b);
        t.bytes += n;
        return b.ptr;
    }

    fn up(t: *Tower, bytes: []const u8) !u64 {
        const p = try t.alloc(bytes.len);
        try t.d.check(t.d.api.cuMemcpyHtoD_v2(p, bytes.ptr, bytes.len), "cuMemcpyHtoD");
        return p;
    }

    fn get(t: *Tower, ck: *core.Checkpoint, name: []const u8, dims: []const usize) !u64 {
        return t.up((try ck.expect(name, .bf16, dims)).bytes);
    }

    /// The tower from the checkpoint at `dir` (its model.visual.* tensors, bf16 as released), scratch for `max_tokens`.
    pub fn load(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, dir: []const u8, v: config.Vision, max_tokens: u32) !*Tower {
        if (max_tokens == 0) return error.ImageTokens;
        const t = try gpa.create(Tower);
        errdefer gpa.destroy(t);
        var k = try Kernels.load(d);
        errdefer k.unload();
        var vk = try vit.Kernels.load(d);
        errdefer vk.unload();
        var fa = try vit.Attention.load(d);
        errdefer fa.unload();
        t.* = .{ .gpa = gpa, .d = d, .k = k, .vk = vk, .fa = fa, .g = try vit.Gemms.init(gpa, d), .max_tokens = max_tokens, .proj_inter = v.proj_hidden, .hidden = v.text_hidden };
        errdefer {
            for (t.bufs.items) |*b| b.free();
            t.bufs.deinit(gpa);
            t.g.deinit();
        }
        var ck = try openTower(gpa, io, dir);
        defer core.checkpoint.cache.close(&ck);
        t.patch_w = try t.get(&ck, pre ++ "embeddings.patch_embedding.weight", &.{ width, image.patch_width });
        t.table = try t.get(&ck, pre ++ "embeddings.position_embedding", &.{ image.grid * image.grid, width });
        t.cls = try t.get(&ck, pre ++ "embeddings.class_embedding", &.{width});
        t.cls_pos = try t.get(&ck, pre ++ "embeddings.cls_position_embedding", &.{width});
        t.pre_w = try t.get(&ck, pre ++ "ln_pre.weight", &.{width});
        t.pre_b = try t.get(&ck, pre ++ "ln_pre.bias", &.{width});
        t.proj_in = try t.get(&ck, "model.visual.proj_in.weight", &.{ t.proj_inter, merged });
        t.proj_out = try t.get(&ck, "model.visual.proj_out.weight", &.{ t.hidden, t.proj_inter });
        const w = try gpa.alloc(u8, 3 * width * width * 2);
        defer gpa.free(w);
        for (&t.layers, 0..) |*l, i| {
            var nb: [128]u8 = undefined;
            const at = struct {
                fn f(buf: *[128]u8, li: usize, comptime rel: []const u8) ![]const u8 {
                    return std.fmt.bufPrint(buf, pre ++ "layers.{d}." ++ rel, .{li});
                }
            }.f;
            l.ln1_w = try t.get(&ck, try at(&nb, i, "ln_1.weight"), &.{width});
            l.ln1_b = try t.get(&ck, try at(&nb, i, "ln_1.bias"), &.{width});
            l.ln2_w = try t.get(&ck, try at(&nb, i, "ln_2.weight"), &.{width});
            l.ln2_b = try t.get(&ck, try at(&nb, i, "ln_2.bias"), &.{width});
            l.ls1 = try t.get(&ck, try at(&nb, i, "ls_1.gamma"), &.{width});
            l.ls2 = try t.get(&ck, try at(&nb, i, "ls_2.gamma"), &.{width});
            // q | k | v stacked by rows, their biases likewise: one Linear, heads as the attention reads them
            var b: [3 * width * 2]u8 = undefined;
            inline for (.{ "q", "k", "v" }, 0..) |which, j| {
                const wt = try ck.expect(try at(&nb, i, "self_attn." ++ which ++ "_proj.weight"), .bf16, &.{ width, width });
                @memcpy(w[j * width * width * 2 ..][0 .. width * width * 2], wt.bytes);
                const bt = try ck.expect(try at(&nb, i, "self_attn." ++ which ++ "_proj.bias"), .bf16, &.{width});
                @memcpy(b[j * width * 2 ..][0 .. width * 2], bt.bytes);
            }
            l.qkv_w = try t.up(w);
            l.qkv_b = try t.up(&b);
            l.out_w = try t.get(&ck, try at(&nb, i, "self_attn.out_proj.weight"), &.{ width, width });
            l.out_b = try t.get(&ck, try at(&nb, i, "self_attn.out_proj.bias"), &.{width});
            l.fc1_w = try t.get(&ck, try at(&nb, i, "mlp.c_fc.weight"), &.{ mlp_inter, width });
            l.fc1_b = try t.get(&ck, try at(&nb, i, "mlp.c_fc.bias"), &.{mlp_inter});
            l.fc2_w = try t.get(&ck, try at(&nb, i, "mlp.c_proj.weight"), &.{ width, mlp_inter });
            l.fc2_b = try t.get(&ck, try at(&nb, i, "mlp.c_proj.bias"), &.{width});
        }
        const P: usize = @as(usize, max_tokens) * image.merge * image.merge;
        const R = P + 1;
        t.px = try t.alloc(P * image.patch_width * 2);
        inline for (.{ "x", "xn", "att", "y" }) |f| @field(t, f) = try t.alloc(R * width * 2);
        t.qkv = try t.alloc(R * 3 * width * 2);
        t.mid = try t.alloc(R * mlp_inter * 2);
        t.acc = try t.alloc(R * mlp_inter * 4);
        t.taps = try t.alloc(P * 4 * 4);
        t.wts = try t.alloc(P * 4 * 4);
        t.cs = try t.alloc(R * image.pairs * 4);
        t.sn = try t.alloc(R * image.pairs * 4);
        t.p1 = try t.alloc(@as(usize, max_tokens) * t.proj_inter * 2);
        return t;
    }

    /// The device bytes load takes for `dir`: the tower's tensors as stored, scratch and workspace for `max_tokens`.
    pub fn bytesFor(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, v: config.Vision, max_tokens: u32) !u64 {
        var ck = try openTower(gpa, io, dir);
        defer core.checkpoint.cache.close(&ck);
        var total: u64 = scratchBytes(v, max_tokens);
        for (ck.files.items) |*f| for (f.names.keys()) |name| {
            total += f.get(name).?.bytes.len;
        };
        return total;
    }

    pub fn deinit(t: *Tower) void {
        for (t.bufs.items) |*b| b.free();
        t.bufs.deinit(t.gpa);
        t.g.deinit();
        t.fa.unload();
        t.vk.unload();
        t.k.unload();
        t.gpa.destroy(t);
    }

    fn upload(t: *Tower, dst: u64, bytes: []const u8) !void {
        try t.d.check(t.d.api.cuMemcpyHtoD_v2(dst, bytes.ptr, bytes.len), "cuMemcpyHtoD");
    }

    /// The host tables of one picture (patch rows, position taps, rotary; rotary off only for a planted control).
    fn stage(t: *Tower, s: cuda.Stream, p: image.Prepared, rope: bool) !void {
        const P = image.patchCount(p);
        const rows = try t.gpa.alloc(u16, P * image.patch_width);
        defer t.gpa.free(rows);
        const taps = try t.gpa.alloc(u32, P * 4);
        defer t.gpa.free(taps);
        const wts = try t.gpa.alloc(f32, P * 4);
        defer t.gpa.free(wts);
        const cs = try t.gpa.alloc(f32, (P + 1) * image.pairs);
        defer t.gpa.free(cs);
        const sn = try t.gpa.alloc(f32, (P + 1) * image.pairs);
        defer t.gpa.free(sn);
        image.patchRows(p, rows);
        image.positionTaps(p, taps, wts);
        image.rotary(p, cs, sn, rope);
        try s.synchronize(); // the previous picture's kernels may still read the tables
        try t.upload(t.px, std.mem.sliceAsBytes(rows));
        try t.upload(t.taps, std.mem.sliceAsBytes(taps));
        try t.upload(t.wts, std.mem.sliceAsBytes(wts));
        try t.upload(t.cs, std.mem.sliceAsBytes(cs));
        try t.upload(t.sn, std.mem.sliceAsBytes(sn));
    }

    /// One picture to its tokens' rows at `out` (bf16 [tokens][hidden]): what replaces its placeholders' embeddings.
    pub fn encode(t: *Tower, s: cuda.Stream, p: image.Prepared, out: u64) !void {
        return t.encodeWith(s, p, out, true);
    }

    pub fn encodeWith(t: *Tower, s: cuda.Stream, p: image.Prepared, out: u64, rope: bool) !void {
        try admit(p, t.max_tokens);
        const T: usize = p.tokens();
        try t.stage(s, p, rope);
        const P = image.patchCount(p);
        const R = P + 1;
        try t.g.run(s, t.px, t.patch_w, t.y, P, width, image.patch_width, .bf16);
        var a: cuda.Args = .{};
        for ([_]u64{ t.y, t.table, t.taps, t.wts, t.cls, t.cls_pos, t.xn }) |v| a.add(v);
        try go(t.k.embed, s, .{ .x = @intCast(R) }, 256, 0, &a);
        try t.vk.layerNorm(s, t.xn, t.pre_w, t.pre_b, t.x, R, width, eps);
        for (t.layers[0..@min(t.layers_run, depth)]) |l| try t.layer(s, l, R);
        if (t.layers_run < depth) return;
        // the projector: each token's four patch rows (merge order, after the class row) as one 4096-wide row
        try t.g.run(s, t.x + width * 2, t.proj_in, t.p1, T, t.proj_inter, merged, .bf16);
        try t.vk.pointwise(s, t.p1, null, t.p1, T * t.proj_inter, .gelu);
        try t.g.run(s, t.p1, t.proj_out, out, T, t.hidden, t.proj_inter, .bf16);
    }

    fn layer(t: *Tower, s: cuda.Stream, l: Layer, R: usize) !void {
        try t.vk.layerNorm(s, t.x, l.ln1_w, l.ln1_b, t.xn, R, width, eps);
        try vit.linear(&t.g, &t.vk, s, t.acc, t.xn, l.qkv_w, l.qkv_b, t.qkv, R, 3 * width, width);
        var a: cuda.Args = .{};
        for ([_]u64{ t.qkv, t.cs, t.sn }) |v| a.add(v);
        try go(t.k.rope, s, .{ .x = @intCast(R) }, 256, 0, &a);
        try t.fa.full(s, t.qkv, t.att, R, heads, head_dim, 1.0 / @sqrt(@as(f32, head_dim)));
        try vit.linear(&t.g, &t.vk, s, t.acc, t.att, l.out_w, l.out_b, t.y, R, width, width);
        try t.vk.pointwiseGamma(s, t.y, t.x, t.x, R * width, .none, l.ls1, width); // torch's `x + gamma * h`
        try t.vk.layerNorm(s, t.x, l.ln2_w, l.ln2_b, t.xn, R, width, eps);
        try vit.linearAct(&t.g, &t.vk, s, t.acc, t.xn, l.fc1_w, l.fc1_b, t.mid, R, mlp_inter, width, null, t.mid, .gelu);
        try vit.linear(&t.g, &t.vk, s, t.acc, t.mid, l.fc2_w, l.fc2_b, t.y, R, width, mlp_inter);
        try t.vk.pointwiseGamma(s, t.y, t.x, t.x, R * width, .none, l.ls2, width);
    }
};

/// A picture encode takes: pixels that fill its grid, and no more tokens than the tower's scratch covers.
fn admit(p: image.Prepared, max_tokens: u32) !void {
    const T: usize = p.tokens();
    if (T == 0 or p.pixels.len != T * image.side * image.side * 3) return error.ImageGrid;
    if (T > max_tokens) return error.ImageTooLarge;
}

const image_rows = core.image_rows;
pub const Span = image_rows.Span;

/// The placeholder runs of a prompt's ids: each maximal run of image_token is one picture's rows, in order.
pub fn spans(a: std.mem.Allocator, ids: []const u32) ![]Span {
    var out: std.ArrayList(Span) = .empty;
    errdefer out.deinit(a);
    var row: u32 = 0;
    var i: usize = 0;
    while (i < ids.len) : (i += 1) {
        if (ids[i] != image.image_token) continue;
        const at = i;
        while (i < ids.len and ids[i] == image.image_token) i += 1;
        try out.append(a, .{ .at = @intCast(at), .n = @intCast(i - at), .row = row });
        row += @intCast(i - at);
    }
    return out.toOwnedSlice(a);
}

/// The prompt's placeholders must equal the encoded pictures' rows, as transformers' get_placeholder_mask requires.
pub fn check(list: []const Span, pictures: []const image.Prepared) !void {
    var want: usize = 0;
    for (pictures) |p| want += p.tokens();
    if (image_rows.total(list) != want) return error.ImagePlaceholders;
}

/// A cut's bytes for a pass from `lo`, bf16 rows `hidden` wide: offsets into the pass and the image rows, the length.
pub fn copyBytes(c: image_rows.Cut, lo: usize, hidden: u32) [3]u64 {
    const row: u64 = @as(u64, hidden) * 2;
    return .{ (c.at - lo) * row, c.row * row, c.n * row };
}

/// A copy's bytes for bf16 rows `hidden` wide: the offset into the pass, the offset into the image rows, the length.
pub fn copyBytes(c: Copy, hidden: u32) [3]u64 {
    const row: u64 = @as(u64, hidden) * 2;
    return .{ (c.at - lo) * row, c.row * row, c.n * row };
}

/// After the normed embedding, a pass's rows at positions [lo, hi) (bf16 at `x`) take their image rows (at `rows`).
pub fn splice(d: *const cuda.Driver, s: cuda.Stream, hidden: u32, rows: u64, list: []const Span, lo: usize, hi: usize, x: u64) !void {
    for (list) |sp| {
        const b = copyBytes(image_rows.cut(sp, lo, hi) orelse continue, lo, hidden);
        try d.check(d.api.cuMemcpyDtoDAsync_v2(x + b[0], rows + b[1], b[2], s.handle), "cuMemcpyDtoDAsync");
    }
}

test "placeholder runs map to picture rows in prompt order" {
    const a = std.testing.allocator;
    const t = image.image_token;
    const got = try spans(a, &.{ 1, t, t, t, 2, 3, t, t, 4 });
    defer a.free(got);
    try std.testing.expectEqual(@as(usize, 2), got.len);
    try std.testing.expectEqual(Span{ .at = 1, .n = 3, .row = 0 }, got[0]);
    try std.testing.expectEqual(Span{ .at = 6, .n = 2, .row = 3 }, got[1]);
    var px: [2 * image.side * image.side * 3]f32 = undefined;
    const two = [_]image.Prepared{ .{ .pixels = &px, .rows = 1, .cols = 3 }, .{ .pixels = &px, .rows = 1, .cols = 2 } };
    try check(got, &two);
    try std.testing.expectError(error.ImagePlaceholders, check(got, two[0..1]));
}

test "chunked passes give every placeholder the row masked_scatter gives it" {
    const a = std.testing.allocator;
    const t = image.image_token;
    const ids = [_]u32{ 7, t, t, t, t, t, 9, 9, t, 4, t, t, t, t, t, t, t, 3, 2 };
    const list = try spans(a, &ids);
    defer a.free(list);
    for ([_]usize{ 1, 2, 3, 5, 8, ids.len }) |chunk| {
        var got: [ids.len]i32 = @splat(-1);
        var lo: usize = 0;
        while (lo < ids.len) : (lo += chunk) {
            const hi = @min(ids.len, lo + chunk);
            for (list) |sp| if (image_rows.cut(sp, lo, hi)) |c| for (0..c.n) |i| {
                got[c.at + i] = @intCast(c.row + i);
            };
        }
        var next: i32 = 0; // masked_scatter: the k-th placeholder in the prompt takes feature row k
        for (ids, got) |id, g| {
            try std.testing.expectEqual(if (id == t) next else -1, g);
            if (id == t) next += 1;
        }
    }
}

/// Inkling 2 Small's vision_config and Mini's, whose decoder is 2304 wide.
fn testVision(text_hidden: u32) config.Vision {
    return .{ .hidden = width, .layers = depth, .heads = heads, .patch = image.patch, .merge = image.merge, .grid = image.grid, .mlp_ratio = 4, .proj_hidden = 4096, .text_hidden = text_hidden, .eps = eps };
}

test "scratch for the default limit stays under 256 MiB" {
    try std.testing.expect(scratchBytes(testVision(4608), default_tokens) < 256 << 20);
}

test "the splice copies rows as wide as the config's decoder, Small's 4608 and Mini's 2304" {
    const sp: Span = .{ .at = 3, .n = 5, .row = 7 };
    for ([_]u32{ 4608, 2304 }) |h| {
        const v = testVision(h);
        const b = copyBytes(image_rows.cut(sp, 4, 6).?, 4, v.text_hidden);
        try std.testing.expectEqual([3]u64{ 0, 8 * @as(u64, h) * 2, 2 * @as(u64, h) * 2 }, b);
    }
    // the projector's scratch row is projection_hidden_dim wide (4096 in both releases)
    var v = testVision(2304);
    const wide = scratchBytes(v, 16);
    v.proj_hidden = 2048;
    try std.testing.expectEqual(@as(u64, 16 * 2048 * 2), wide - scratchBytes(v, 16));
}

test "encode refuses a picture whose pixels do not fill its grid" {
    var px: [2 * image.side * image.side * 3]f32 = undefined;
    try admit(.{ .pixels = &px, .rows = 1, .cols = 2 }, 2);
    try std.testing.expectError(error.ImageGrid, admit(.{ .pixels = &px, .rows = 1, .cols = 3 }, 4));
    try std.testing.expectError(error.ImageGrid, admit(.{ .pixels = px[0..0], .rows = 0, .cols = 2 }, 4));
}

test "encode refuses a picture with more tokens than the tower's scratch covers" {
    var px: [2 * image.side * image.side * 3]f32 = undefined;
    try std.testing.expectError(error.ImageTooLarge, admit(.{ .pixels = &px, .rows = 2, .cols = 1 }, 1));
}

test "a tower with scratch for no image tokens is refused before any device work" {
    const got = Tower.load(std.testing.allocator, std.testing.io, undefined, "", testVision(4608), 0);
    try std.testing.expectError(error.ImageTokens, got);
}

test "the splice copies rows as wide as the config's decoder, Small's 4608 and Mini's 2304" {
    const sp: Span = .{ .at = 3, .n = 5, .row = 7 };
    for ([_]u32{ 4608, 2304 }) |h| {
        const v = testVision(h);
        const b = copyBytes(cut(sp, 4, 6).?, v.text_hidden);
        try std.testing.expectEqual([3]u64{ 0, 8 * @as(u64, h) * 2, 2 * @as(u64, h) * 2 }, b);
    }
    // the projector's scratch row is projection_hidden_dim wide (4096 in both releases)
    var v = testVision(2304);
    const wide = scratchBytes(v, 16);
    v.proj_hidden = 2048;
    try std.testing.expectEqual(@as(u64, 16 * 2048 * 2), wide - scratchBytes(v, 16));
}

test "encode refuses a picture whose pixels do not fill its grid" {
    var px: [2 * image.side * image.side * 3]f32 = undefined;
    try admit(.{ .pixels = &px, .rows = 1, .cols = 2 }, 2);
    try std.testing.expectError(error.ImageGrid, admit(.{ .pixels = &px, .rows = 1, .cols = 3 }, 4));
    try std.testing.expectError(error.ImageGrid, admit(.{ .pixels = px[0..0], .rows = 0, .cols = 2 }, 4));
}

test "encode refuses a picture with more tokens than the tower's scratch covers" {
    var px: [2 * image.side * image.side * 3]f32 = undefined;
    try std.testing.expectError(error.ImageTooLarge, admit(.{ .pixels = &px, .rows = 2, .cols = 1 }, 1));
}

test "a tower with scratch for no image tokens is refused before any device work" {
    const got = Tower.load(std.testing.allocator, std.testing.io, undefined, "", testVision(4608), 0);
    try std.testing.expectError(error.ImageTokens, got);
}
