//! dots.mocr's image tower on CUDA as the dense engine's Tower: merge-ordered patches to the decoder's rows.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const dense = @import("cuda_dense");
const cfg = @import("config.zig");
const image = @import("image.zig");
const inventory = @import("weights.zig");
const weights = @import("cuda_tower_weights.zig");
const kern = @import("cuda_kernels.zig");

const vit = cuda.vit;
const bf16 = core.bf16_round;
const Allocator = std.mem.Allocator;
const Image = dense.recipe.Image;
const Tap = dense.recipe.Tap;

/// The largest image one call admits: the processor's 11,289,600 pixels in 14 x 14 patches (HIP's max_patches).
pub const max_patches = 57600;
/// vit.Gemms' own workspace (cuda/vit.zig's workspace_bytes).
pub const workspace_bytes = 32 << 20;
/// The GEMM plans the tower keeps: each image size adds its own, so past this many they go.
pub const max_plans = 32;
/// The rotary's inverse frequencies: head dim 128 over (row, column), 32 an axis.
pub const rope_pairs = 32;

pub const Marks = struct { open: u32, image: u32, close: u32 };
/// tokenizer.json's <|img|>, <|imgpad|> (config's image_token_id) and <|endofimg|>.
pub const marks: Marks = .{ .open = 151666, .image = 151665, .close = 151667 };
/// The marks the server checks are the tokenizer's added tokens.
pub const mark_ids: [3]u32 = .{ marks.open, marks.image, marks.close };

/// Transformers' own float32 rotary inv_freq for the tower (theta 10,000, dim 64), fixtures/README.md.
pub const inv_freq: [rope_pairs]f32 = blk: {
    @setEvalBranchQuota(10_000);
    const text = @embedFile("fixtures/vision_inv_freq.txt");
    var out: [rope_pairs]f32 = undefined;
    var it = std.mem.tokenizeAny(u8, text, ", \n");
    for (&out) |*v| v.* = @bitCast(std.fmt.parseInt(u32, it.next().?, 0) catch unreachable);
    if (it.next() != null) @compileError("fixtures/vision_inv_freq.txt holds more than 32 values");
    break :blk out;
};

/// One call's device buffers at `patches` rows, each 256-byte aligned in one allocation; the merger reuses them.
pub const Scratch = struct {
    px: u64 = 0, // bf16 [P, patch_k]: the patches, the columns past 588 zero
    hw: u64 = 0, // i32 [P, 2]: each patch's (row, column)
    h: u64 = 0, // bf16 [P, hidden]: the residual; then the merger's LayerNorm rows
    xn: u64 = 0, // bf16 [P, hidden]: a norm's rows
    r: u64 = 0, // bf16 [P, hidden]: a branch's output; then the merger's GELU rows
    qkv: u64 = 0, // bf16 [P, 3 hidden]; then the SwiGLU's rows [P, intermediate]
    att: u64 = 0, // bf16 [P, hidden]: attention's rows; then the merger's first linear before its GELU
    gu: u64 = 0, // bf16 [P, 2 intermediate]: fc1 | fc3; then the merger's f32 sums
    inv: u64 = 0, // f32 [32]: inv_freq
    bytes: u64 = 0,

    pub fn of(v: cfg.Vision, patches: u64) Scratch {
        const h: u64 = v.hidden;
        var s: Scratch = .{};
        const parts = [_]struct { *u64, u64 }{
            .{ &s.px, patches * weights.patch_k * 2 }, .{ &s.hw, patches * 8 },
            .{ &s.h, patches * h * 2 },                .{ &s.xn, patches * h * 2 },
            .{ &s.r, patches * h * 2 },                .{ &s.qkv, patches * 3 * h * 2 },
            .{ &s.att, patches * h * 2 },              .{ &s.gu, patches * 2 * @as(u64, v.intermediate) * 2 },
            .{ &s.inv, rope_pairs * 4 },
        };
        for (parts) |p| {
            p[0].* = s.bytes;
            s.bytes += std.mem.alignForward(u64, p[1], 256);
        }
        return s;
    }

    /// The offsets as device addresses from `base`.
    fn at(s: Scratch, base: u64) Scratch {
        var out = s;
        inline for (.{ "px", "hw", "h", "xn", "r", "qkv", "att", "gu", "inv" }) |f| @field(out, f) += base;
        return out;
    }
};

/// Each norm as Torch's RMSNorm rounds it, bf16(w bf16(x inv)), the residual sum rounded first.
const round: cuda.norms.Tail = .{ .round = true };

pub const Tower = struct {
    pub const Limits = struct { image_tokens: u32, max_images: u32 };
    /// A request's image tokens over all its images (--vision-image-tokens): one largest image, as HIP.
    pub const default_image_tokens: u32 = max_patches / 4;
    /// The chat template writes each image's open and close marks itself.
    pub const template_wraps = true;
    /// The template's name for an image part.
    pub const part_type = "image";

    marks: Marks = marks,
    mark_ids: [3]u32 = mark_ids,
    limits: Limits,
    width: usize, // the decoder's hidden size: the merger's rows
    gpa: Allocator,
    io: std.Io,
    d: *const cuda.Driver,
    s: cuda.Stream, // the engine's stream, the one every encode runs on
    c: cfg.Vision,
    processor: image.Processor,
    w: weights.Tower,
    k: dense.kernels.Kernels, // the engine's norms, vit glue and SwiGLU
    own: kern.Kernels, // the vision rotary and patch bias
    att: vit.Attention,
    gemms: vit.Gemms,
    scratch: cuda.DeviceBuffer,
    at: Scratch,
    encodes: u64 = 0, // images that passed the grid checks into encode (the GB10 runner's count)

    /// The processor, the weights, the kernels and one call's scratch at max_patches; zero limits refused first.
    pub fn open(gpa: Allocator, io: std.Io, ctx: *const cuda.Context, s: cuda.Stream, dir: []const u8, limits: Limits) !*Tower {
        if (limits.image_tokens == 0 or limits.max_images == 0) return error.DotsVisionLimits;
        const c = try cfg.Config.read(gpa, io, dir);
        if (c.image_token != marks.image) return error.UnsupportedDotsConfig;
        const processor = try image.Processor.read(gpa, io, dir, c);
        const d = ctx.d;
        const t = try gpa.create(Tower);
        errdefer gpa.destroy(t);
        var w = try weights.load(gpa, io, d, dir);
        errdefer w.deinit();
        var k = try dense.kernels.Kernels.load(ctx, @import("cuda_recipe.zig").geometryOf(c));
        errdefer k.deinit();
        var own = try kern.Kernels.load(ctx);
        errdefer own.deinit();
        var att = try vit.Attention.load(d);
        errdefer att.unload();
        var gemms = try vit.Gemms.init(gpa, d);
        errdefer gemms.deinit();
        const plan = Scratch.of(c.vision, max_patches);
        var scratch = try cuda.DeviceBuffer.alloc(d, plan.bytes);
        errdefer scratch.free();
        // runs first on failure: an upload whose wait failed lands before the scratch goes
        errdefer cuda.completion.drain(d, .{ .d = d, .handle = null }, "dots.mocr's tower upload did not drain before its scratch was freed");
        const at = plan.at(scratch.ptr);
        try cuda.copy_source.synchronousCopy(d, at.inv, std.mem.sliceAsBytes(&inv_freq), gpa);
        t.* = .{ .limits = limits, .width = c.hidden, .gpa = gpa, .io = io, .d = d, .s = s, .c = c.vision, .processor = processor, .w = w, .k = k, .own = own, .att = att, .gemms = gemms, .scratch = scratch, .at = at };
        return t;
    }

    /// The scratch at max_patches and the GEMM workspace (open's bytes beside the weights).
    pub fn openBytes(a: Allocator, io: std.Io, dir: []const u8) !u64 {
        const c = try cfg.Config.read(a, io, dir);
        return Scratch.of(c.vision, max_patches).bytes + workspace_bytes;
    }

    /// The tower's weights on the device, once every shard's header is checked.
    pub fn weightBytes(a: Allocator, io: std.Io, dir: []const u8) !u64 {
        const c = try cfg.Config.read(a, io, dir);
        _ = try inventory.inspect(a, io, dir, c);
        return weights.deviceBytes(a, c);
    }

    /// The server's processor (on the HTTP thread): the slow Qwen2VL one, at most `max_tokens` tokens.
    pub fn prepare(t: *Tower, a: Allocator, bytes: []const u8, max_tokens: u32) !image.Prepared {
        return t.processor.prepare(t.io, a, bytes, max_tokens);
    }

    pub fn explain(err: anyerror) ?[]const u8 {
        return switch (err) {
            error.DotsImageSize => "dots.mocr reads images of at least 28 pixels a side",
            error.DotsImageBudget => "the image takes more image tokens than the request's budget (--vision-image-tokens) or dots.mocr's 14,400 an image",
            error.DotsGrid => "the image's patches do not fill its grid in whole 2 x 2 merge blocks",
            error.DotsVisionLimits => "--vision: dots.mocr takes at least one image token and one image a request",
            else => core.picture.patches.problem(err),
        };
    }

    /// Waits for the engine's stream before the scratch and the weights go.
    pub fn deinit(t: *Tower) void {
        cuda.completion.drain(t.d, t.s, "dots.mocr's image tower work did not drain before its buffers were freed");
        t.scratch.free();
        t.gemms.deinit();
        t.att.unload();
        t.own.deinit();
        t.k.deinit();
        t.w.deinit();
        t.gpa.destroy(t);
    }

    /// One image's patches to its tokens' rows [tokens, width] bf16 at `out` on `s`; `taps` sees each stage's rows.
    pub fn encode(t: *Tower, s: cuda.Stream, im: *const Image, out: u64, taps: ?Tap) !void {
        const n = try rowsOf(t.c, im);
        t.encodes += 1;
        const hw = try t.gpa.alloc([2]u32, n);
        defer t.gpa.free(hw);
        try image.fillPositions(hw, im.gh, im.gw, t.c.merge);
        const px = try t.gpa.alloc(u16, n * weights.patch_k);
        defer t.gpa.free(px);
        patchRows(px, im.pixels, patchWidth(t.c));
        try t.trim(s);
        const a = t.at;
        try cuda.copy_source.enqueue(t.d, a.hw, std.mem.sliceAsBytes(hw), s);
        try cuda.copy_source.enqueue(t.d, a.px, std.mem.sliceAsBytes(px), s);
        const h: usize = t.c.hidden;
        const m: usize = t.c.intermediate;
        const heads: usize = t.c.heads;
        const eps = t.c.eps;
        const k = &t.k;
        const tap: Taps = .{ .tap = taps, .n = n, .width = h };
        try t.gemms.run(s, a.px, t.w.patch, a.r, n, h, weights.patch_k, .bf16);
        try t.own.patchBias(s, a.r, t.w.patch_bias, a.r, n, h);
        try tap.at("vision-patch-bias", a.r);
        try k.norms.addRmsTail(s, a.r, null, t.w.patch_norm, a.r, a.h, null, n, h, eps, round);
        try tap.at("vision-patch", a.h);
        const scale: f32 = @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(h / heads))));
        for (t.w.blocks, 0..) |b, i| {
            // the last block's MLP joins the residual here, as norm1's input
            try k.norms.addRmsTail(s, a.h, if (i == 0) null else a.r, b.norm1, a.h, a.xn, null, n, h, eps, round);
            if (i > 0) try tap.block(i - 1, a.h);
            try t.gemms.run(s, a.xn, b.qkv, a.qkv, n, 3 * h, h, .bf16);
            try t.own.visionRope(s, a.qkv, a.hw, a.inv, n);
            try t.att.full(s, a.qkv, a.att, n, heads, h / heads, scale);
            try t.gemms.run(s, a.att, b.proj, a.r, n, h, h, .bf16);
            try k.norms.addRmsTail(s, a.h, a.r, b.norm2, a.h, a.xn, null, n, h, eps, round);
            try t.gemms.run(s, a.xn, b.fc13, a.gu, n, 2 * m, h, .bf16);
            try k.siluMul(s, a.gu, a.qkv, n, m);
            try t.gemms.run(s, a.qkv, b.fc2, a.r, n, h, m, .bf16);
        }
        try k.norms.addRmsTail(s, a.h, a.r, t.w.post_norm, a.h, a.xn, null, n, h, eps, round);
        try tap.block(t.w.blocks.len - 1, a.h);
        try tap.at("vision-norm", a.xn);
        // the merger: LayerNorm, then each merge block's four adjacent rows as one [4 hidden] row
        try k.vit.layerNorm(s, a.xn, t.w.ln_w, t.w.ln_b, a.h, n, h, 1e-6);
        const rows = n / 4;
        const wide: usize = t.c.merged();
        try vit.linearAct(&t.gemms, &k.vit, s, a.gu, a.h, t.w.mlp0, t.w.mlp0_b, a.att, rows, wide, wide, null, a.r, .gelu);
        try vit.linear(&t.gemms, &k.vit, s, a.gu, a.r, t.w.mlp2, t.w.mlp2_b, out, rows, t.width, wide);
        if (taps) |tp| try tp.at(tp.ctx, "vision-merged", out, rows, t.width);
    }

    /// Past max_plans shapes, the stream drains and the GEMM plans go; the workspace stays.
    fn trim(t: *Tower, s: cuda.Stream) !void {
        if (t.gemms.plans.count() < max_plans) return;
        try s.synchronize();
        var it = t.gemms.plans.valueIterator();
        while (it.next()) |l| l.deinit();
        t.gemms.plans.clearRetainingCapacity();
    }
};

/// A patch's values: 3 channels of patch x patch pixels (588).
fn patchWidth(v: cfg.Vision) usize {
    return 3 * @as(usize, v.patch) * v.patch;
}

/// The image's patch rows, refused by name: a grid of whole merge blocks whose pixels and tokens it fills.
pub fn rowsOf(v: cfg.Vision, im: *const Image) error{ DotsGrid, DotsImageBudget }!usize {
    const n = @as(usize, im.gh) * im.gw;
    const blocks = @as(usize, v.merge) * v.merge;
    if (n == 0 or im.gh % v.merge != 0 or im.gw % v.merge != 0) return error.DotsGrid;
    if (n > max_patches) return error.DotsImageBudget;
    if (im.pixels.len != n * patchWidth(v) or im.tokens != n / blocks) return error.DotsGrid;
    return n;
}

/// Patches [n, k] f32 to bf16 rows of patch_k, rounded to nearest even, the columns past k zero.
pub fn patchRows(dst: []u16, src: []const f32, k: usize) void {
    @memset(dst, 0);
    for (0..dst.len / weights.patch_k) |r| for (0..k) |c| {
        dst[r * weights.patch_k + c] = bf16.bits(src[r * k + c]);
    };
}

/// The tap, when given, called with a stage's rows [n, width].
const Taps = struct {
    tap: ?Tap,
    n: usize,
    width: usize,

    fn at(x: Taps, stage: []const u8, rows: u64) !void {
        if (x.tap) |tp| try tp.at(tp.ctx, stage, rows, x.n, x.width);
    }

    /// Block i's output: the residual once its MLP has joined it.
    fn block(x: Taps, i: usize, rows: u64) !void {
        if (x.tap == null) return;
        var name: [32]u8 = undefined;
        try x.at(try std.fmt.bufPrint(&name, "vision-block{d}", .{i}), rows);
    }
};

test "the GEMM plans: below max_plans shapes they stay; at it the stream drains first, then every plan goes" {
    const lt = cuda.cublaslt;
    const Fake = struct {
        var waits: usize = 0;
        var destroyed: usize = 0;
        fn wait(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
            waits += 1;
            return 0;
        }
        fn handle(_: lt.Handle) callconv(.c) lt.Status {
            destroyed += 1;
            return 0;
        }
        fn desc(_: lt.MatmulDesc) callconv(.c) lt.Status {
            destroyed += 1;
            return 0;
        }
        fn layout(_: lt.Layout) callconv(.c) lt.Status {
            destroyed += 1;
            return 0;
        }
        fn pref(_: lt.Preference) callconv(.c) lt.Status {
            destroyed += 1;
            return 0;
        }
    };
    const a = std.testing.allocator;
    var d: cuda.Driver = .{ .lib = undefined, .api = undefined };
    d.api.cuStreamSynchronize = Fake.wait;
    var lib: lt.Library = .{ .lib = undefined, .api = undefined };
    lib.api.cublasLtDestroy = Fake.handle;
    lib.api.cublasLtMatmulDescDestroy = Fake.desc;
    lib.api.cublasLtMatrixLayoutDestroy = Fake.layout;
    lib.api.cublasLtMatmulPreferenceDestroy = Fake.pref;
    var t: Tower = undefined;
    t.gemms.gpa = a;
    t.gemms.plans = .empty;
    defer t.gemms.plans.deinit(a);
    const s: cuda.Stream = .{ .d = &d, .handle = @ptrFromInt(0x1800) };
    for (0..max_plans) |i| {
        var p: lt.Linear = undefined;
        p.lt = &lib;
        try t.gemms.plans.put(a, .{ i, 0, 0 }, p);
        if (i + 1 < max_plans) {
            try t.trim(s); // the control: under the bound, nothing drains and nothing goes
            try std.testing.expect(Fake.waits == 0 and Fake.destroyed == 0 and t.gemms.plans.count() == i + 1);
        }
    }
    try t.trim(s);
    try std.testing.expectEqual(@as(usize, 1), Fake.waits);
    try std.testing.expectEqual(@as(usize, max_plans * 6), Fake.destroyed); // handle, desc, three layouts, preference
    try std.testing.expectEqual(@as(usize, 0), t.gemms.plans.count());
    cuda.copy_source.dropSpares(&d, cuda.copy_source.current());
}
