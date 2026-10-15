//! Bolt's image tower on CUDA: C-RADIO ViT-H/16 and the mlp1 projector, bf16 weights, fp32 residual; radio_vision.cu.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");

const lt = cuda.cublaslt;
const DeviceBuffer = cuda.DeviceBuffer;
const lanes_ = @import("lanes");
const radio = "vision_model.radio_model.";
const model = radio ++ "model.";

pub const hidden = 1280;
pub const heads = 16;
pub const head_dim = 80; // radio_vision.cu's attention is built for 80
pub const depth = 32;
pub const mlp = 5120;
pub const patch = 16;
pub const patch_width = 3 * patch * patch; // 768, (c, dy, dx)
pub const prefix_tokens = 10; // 3 cls + 7 registers, dropped after the last block
pub const cpe_side = 128; // pos_embed's 16384 rows
pub const shuffled = 4 * hidden; // pixel shuffle v2: 2x2 patches a row
pub const projector_hidden = 20480;
pub const out_hidden = 4096; // the language model's width
pub const ln_eps: f32 = 1e-6;
pub const rms_eps: f32 = 1e-5;
/// One call's patches at most: max_num_patches.
pub const max_patches = 13312;
/// The prompt's placeholder ids (tokenizer.json): `<image>` repeated, between `<img>` and `</img>`.
pub const image_token: u32 = 18;
pub const image_start: u32 = 19;
pub const image_end: u32 = 20;
const workspace_bytes = 32 << 20;
/// The processor's normalization (vision/nemotron_image.zig's MEAN and STD): input_conditioner must hold the same.
const clip_mean = [3]f32{ 0.48145466, 0.4578275, 0.40821073 };
const clip_std = [3]f32{ 0.26862954, 0.26130258, 0.27577711 };

/// What config.json must say for this module to run the checkpoint's tower; anything else is refused.
pub fn check(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !void {
    const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
    defer gpa.free(path);
    // Bolt's quantization_config makes it ~6.5 MB
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 26));
    defer gpa.free(bytes);
    return checkConfig(gpa, bytes);
}

pub fn checkConfig(gpa: std.mem.Allocator, bytes: []const u8) !void {
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, bytes, .{});
    defer parsed.deinit();
    const root = parsed.value.object;
    const v = (root.get("vision_config") orelse return error.UnsupportedBoltVision).object;
    const ints = [_]struct { std.json.ObjectMap, []const u8, i64 }{
        .{ v, "hidden_size", hidden },                        .{ v, "num_attention_heads", heads },           .{ v, "num_hidden_layers", depth },
        .{ v, "patch_size", patch },                          .{ v, "num_cls_tokens", 3 },                    .{ v, "num_registers", 7 },
        .{ root, "projector_hidden_size", projector_hidden }, .{ root, "img_context_token_id", image_token }, .{ root, "vit_hidden_size", hidden },
    };
    for (ints) |w| if (num(w[0].get(w[1])) != w[2]) return error.UnsupportedBoltVision;
    const llm = (root.get("llm_config") orelse return error.UnsupportedBoltVision).object;
    if (num(llm.get("hidden_size")) != out_hidden) return error.UnsupportedBoltVision;
    if (!isString(v.get("hidden_act"), "gelu") or !isString(root.get("ps_version"), "v2")) return error.UnsupportedBoltVision;
    if (!isBool(v.get("use_swiglu_ffn"), false) or !isBool(root.get("use_thumbnail"), false) or !isBool(root.get("vision_final_layernorm"), false)) return error.UnsupportedBoltVision;
    if (!isFloat(root.get("downsample_ratio"), 0.5) or !isFloat(v.get("layer_norm_eps"), 1e-6) or !isFloat(v.get("mlp_ratio"), 4.0)) return error.UnsupportedBoltVision;
}

fn num(x: ?std.json.Value) i64 {
    const v = x orelse return -1;
    return if (v == .integer) v.integer else -1;
}

fn isString(x: ?std.json.Value, want: []const u8) bool {
    const v = x orelse return false;
    return v == .string and std.mem.eql(u8, v.string, want);
}

fn isBool(x: ?std.json.Value, want: bool) bool {
    const v = x orelse return !want; // an absent flag is false
    return v == .bool and v.bool == want;
}

fn isFloat(x: ?std.json.Value, want: f64) bool {
    const v = x orelse return false;
    const f: f64 = switch (v) {
        .float => |f| f,
        .integer => |i| @floatFromInt(i),
        else => return false,
    };
    return @abs(f - want) <= 1e-12 * @abs(want);
}

const Block = struct { norm1_w: u64, norm1_b: u64, qkv_w: u64, qkv_b: u64, proj_w: u64, proj_b: u64, norm2_w: u64, norm2_b: u64, fc1_w: u64, fc1_b: u64, fc2_w: u64, fc2_b: u64 };

const Fns = struct {
    embed: cuda.Function,
    layernorm: cuda.Function,
    qkv: cuda.Function,
    attention: cuda.Function,
    bias_gelu: cuda.Function,
    bias_residual: cuda.Function,
    shuffle_rms: cuda.Function,
    relu2: cuda.Function,
    cast: cuda.Function,
};

/// Where one call's stages live in the scratch (device pointers), for tests that read them back after `encode`.
pub const Stages = struct {
    /// fp32 [prefix_tokens + n, 1280]: the residual stream, after `encode` the last block's output (tower_out).
    x: u64,
};

/// The tower's and projector's weights and one call's scratch in one allocation, its kernels and GEMM plans.
pub const Tower = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    mod: cuda.Module,
    f: Fns,
    lib: lt.Library,
    arena: DeviceBuffer, // weights 1.681 GB, scratch for max_patches 0.702 GB
    at: usize = 0,
    scratch: u64 = 0,
    embed_w: u64 = 0,
    pos: u64 = 0,
    cls: u64 = 0,
    blocks: [depth]Block = undefined,
    norm_w: u64 = 0,
    proj1_w: u64 = 0,
    proj2_w: u64 = 0,
    gemms: Gemms = undefined,
    /// Blocks `encode` runs (tests: 0 leaves the embedded rows in Stages.x); the projector runs only after all 32.
    run_depth: usize = depth,

    /// Loads vision_model. and mlp1. (bf16) from Bolt's `dir`, synchronously, into an arena sized for `max_patches`.
    pub fn load(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, dir: []const u8) !*Tower {
        try check(gpa, io, dir);
        const t = try gpa.create(Tower);
        errdefer gpa.destroy(t);
        t.* = .{ .gpa = gpa, .d = d, .mod = undefined, .f = undefined, .lib = undefined, .arena = undefined };
        t.lib = try lt.Library.open();
        errdefer t.lib.close();
        t.gemms = .{ .lib = &t.lib };
        errdefer t.gemms.deinit();
        t.mod = try cuda.Module.load(d, cuda.kernels.radio_vision);
        errdefer t.mod.unload();
        inline for (@typeInfo(Fns).@"struct".field_names) |name| @field(t.f, name) = try t.mod.function("tf_rv_" ++ name);
        t.arena = try DeviceBuffer.alloc(d, weightBytes() + scratchBytes(max_patches));
        errdefer t.arena.free();
        try t.arena.fill8(0, null); // touched once now: the driver commits memory on first touch, slowly
        var vk = try core.Checkpoint.openModelPrefix(gpa, io, dir, "vision_model.");
        defer vk.close();
        var pk = try core.Checkpoint.openModelPrefix(gpa, io, dir, "mlp1.");
        defer pk.close();
        { // the input conditioner is the processor's normalization, already applied on the host: it must agree
            const mean = try vk.expect(radio ++ "input_conditioner.norm_mean", .f32, &.{ 3, 1, 1 });
            const stdv = try vk.expect(radio ++ "input_conditioner.norm_std", .f32, &.{ 3, 1, 1 });
            for (0..3) |c| {
                const m: f32 = @bitCast(std.mem.readInt(u32, mean.bytes[c * 4 ..][0..4], .little));
                const s: f32 = @bitCast(std.mem.readInt(u32, stdv.bytes[c * 4 ..][0..4], .little));
                if (m != clip_mean[c] or s != clip_std[c]) return error.UnsupportedBoltVision;
            }
        }
        _ = try vk.expect("vision_model.summary_idxs", .i64, &.{2}); // the summary heads' cls rows: not used for images
        _ = try vk.expect(model ++ "patch_generator.video_embedder.weight", .bf16, &.{ hidden, 2 * patch_width }); // video: checked, not loaded
        t.embed_w = try t.get(&vk, model ++ "patch_generator.embedder.weight", &.{ hidden, patch_width });
        t.pos = try t.get(&vk, model ++ "patch_generator.pos_embed", &.{ 1, cpe_side * cpe_side, hidden });
        t.cls = try t.get(&vk, model ++ "patch_generator.cls_token.token", &.{ prefix_tokens, hidden });
        for (&t.blocks, 0..) |*b, i| {
            const shapes = .{
                .{ "norm1_w", "norm1.weight", .{hidden} },                .{ "norm1_b", "norm1.bias", .{hidden} },
                .{ "qkv_w", "attn.qkv.weight", .{ 3 * hidden, hidden } }, .{ "qkv_b", "attn.qkv.bias", .{3 * hidden} },
                .{ "proj_w", "attn.proj.weight", .{ hidden, hidden } },   .{ "proj_b", "attn.proj.bias", .{hidden} },
                .{ "norm2_w", "norm2.weight", .{hidden} },                .{ "norm2_b", "norm2.bias", .{hidden} },
                .{ "fc1_w", "mlp.fc1.weight", .{ mlp, hidden } },         .{ "fc1_b", "mlp.fc1.bias", .{mlp} },
                .{ "fc2_w", "mlp.fc2.weight", .{ hidden, mlp } },         .{ "fc2_b", "mlp.fc2.bias", .{hidden} },
            };
            inline for (shapes) |sh| {
                var nb: [128]u8 = undefined;
                const shape: [sh[2].len]usize = sh[2];
                @field(b, sh[0]) = try t.get(&vk, try std.fmt.bufPrint(&nb, model ++ "blocks.{d}.{s}", .{ i, sh[1] }), &shape);
            }
        }
        t.norm_w = try t.get(&pk, "mlp1.0.weight", &.{shuffled});
        t.proj1_w = try t.get(&pk, "mlp1.1.weight", &.{ projector_hidden, shuffled });
        t.proj2_w = try t.get(&pk, "mlp1.3.weight", &.{ out_hidden, projector_hidden });
        if (vk.unused() != 0 or pk.unused() != 0) return error.UnusedVisionTensors;
        t.scratch = try t.take(scratchBytes(max_patches));
        return t;
    }

    pub fn deinit(t: *Tower) void {
        t.gemms.deinit();
        t.arena.free();
        t.mod.unload();
        t.lib.close();
        t.gpa.destroy(t);
    }

    /// The weights' bytes in the arena (each allocation 256-byte aligned): 1,680,788,480.
    pub fn weightBytes() usize {
        const a = aligned;
        var n: usize = a(hidden * patch_width * 2) + a(cpe_side * cpe_side * hidden * 2) + a(prefix_tokens * hidden * 2);
        n += depth * (4 * a(hidden * 2) + a(3 * hidden * hidden * 2) + a(3 * hidden * 2) + a(hidden * hidden * 2) + a(hidden * 2) + a(mlp * hidden * 2) + a(mlp * 2) + a(hidden * mlp * 2) + a(hidden * 2));
        return n + a(shuffled * 2) + a(projector_hidden * shuffled * 2) + a(out_hidden * projector_hidden * 2);
    }

    /// The scratch one call of `n` patches needs, in bytes (aligned pieces; 13,312 patches: 702 MB).
    pub fn scratchBytes(n: usize) usize {
        const a = aligned;
        const rows = prefix_tokens + n;
        const m4 = n / 4;
        return a(n * patch_width * 2) + // bf16 patches
            a(rows * hidden * 4) + // x, the fp32 residual
            a(@max(rows * hidden, m4 * shuffled) * 2) + // h: LayerNorm out, then the shuffled rows
            a(@max(rows * @max(3 * hidden, mlp), m4 * projector_hidden) * 4) + // fp32 GEMM out
            3 * a(rows * hidden * 2) + // q, k, v
            a(rows * hidden * 2) + // attention out
            a(@max(rows * mlp, m4 * projector_hidden) * 2) + // GELU / relu^2 rows
            a(n * 4 * 4) * 2 + // CPE taps and weights
            workspace_bytes;
    }

    fn take(t: *Tower, len: usize) !u64 {
        const at = aligned(t.at);
        if (at + len > t.arena.len) return error.ArenaExhausted;
        t.at = at + len;
        return t.arena.ptr + at;
    }

    fn get(t: *Tower, ck: *core.Checkpoint, name: []const u8, shape: []const usize) !u64 {
        const tn = try ck.expect(name, .bf16, shape);
        const p = try t.take(tn.bytes.len);
        try t.arena.upload(p - t.arena.ptr, tn.bytes);
        return p;
    }

    /// One image's patches (fp32 [gh * gw, 768]) to bf16 [gh * gw / 4, 4096] rows at `out` on `s`; inputs copied.
    pub fn encode(t: *Tower, s: cuda.Stream, patches: []const f32, gh: u32, gw: u32, out: u64) !Stages {
        const n = @as(usize, gh) * gw;
        if (gh == 0 or gw == 0 or gh % 2 != 0 or gw % 2 != 0 or n > max_patches) return error.BadImageGrid;
        if (patches.len != n * patch_width) return error.BadImagePatches;
        const gpa = t.gpa;
        const rows = prefix_tokens + n;
        const m4 = n / 4;
        const px16 = try gpa.alloc(u16, n * patch_width);
        defer gpa.free(px16);
        for (px16, patches) |*o, v| o.* = bf16(v);
        const idx = try gpa.alloc(i32, n * 4);
        defer gpa.free(idx);
        const wts = try gpa.alloc(f32, n * 4);
        defer gpa.free(wts);
        cpeTaps(gh, gw, idx, wts);

        var cur = t.scratch;
        const end = t.scratch + scratchBytes(max_patches);
        const d_px = carve(&cur, n * patch_width * 2);
        const x = carve(&cur, rows * hidden * 4);
        const h = carve(&cur, @max(rows * hidden, m4 * shuffled) * 2);
        const acc = carve(&cur, @max(rows * @max(3 * hidden, mlp), m4 * projector_hidden) * 4);
        const q = carve(&cur, rows * hidden * 2);
        const k = carve(&cur, rows * hidden * 2);
        const v = carve(&cur, rows * hidden * 2);
        const att = carve(&cur, rows * hidden * 2);
        const mid = carve(&cur, @max(rows * mlp, m4 * projector_hidden) * 2);
        const d_idx = carve(&cur, n * 4 * 4);
        const d_wts = carve(&cur, n * 4 * 4);
        const ws = cur;
        const ws_len = end - cur;
        try t.upload(s, d_px, std.mem.sliceAsBytes(px16));
        try t.upload(s, d_idx, std.mem.sliceAsBytes(idx));
        try t.upload(s, d_wts, std.mem.sliceAsBytes(wts));

        const mm = &t.gemms;
        try mm.run(n, hidden, patch_width, d_px, t.embed_w, acc, ws, ws_len, s);
        try go(t.f.embed, s, rows, 1, 256, .{ acc, t.pos, d_idx, d_wts, t.cls, x, int(prefix_tokens), int(hidden) });
        const scale_log2: f32 = std.math.log2e / @sqrt(@as(f32, head_dim));
        for (t.blocks[0..t.run_depth]) |b| {
            try go(t.f.layernorm, s, rows, 1, 256, .{ x, b.norm1_w, b.norm1_b, h, int(hidden), ln_eps });
            try mm.run(rows, 3 * hidden, hidden, h, b.qkv_w, acc, ws, ws_len, s);
            try go(t.f.qkv, s, rows, 1, 256, .{ acc, b.qkv_b, q, k, v, int(rows), int(heads), int(head_dim) });
            try go(t.f.attention, s, (rows + 63) / 64, heads, 128, .{ q, k, v, att, int(rows), int(heads), scale_log2 });
            try mm.run(rows, hidden, hidden, att, b.proj_w, acc, ws, ws_len, s);
            try go(t.f.bias_residual, s, rows, 1, 256, .{ acc, b.proj_b, x, int(hidden) });
            try go(t.f.layernorm, s, rows, 1, 256, .{ x, b.norm2_w, b.norm2_b, h, int(hidden), ln_eps });
            try mm.run(rows, mlp, hidden, h, b.fc1_w, acc, ws, ws_len, s);
            try go(t.f.bias_gelu, s, rows, 1, 256, .{ acc, b.fc1_b, mid, int(mlp) });
            try mm.run(rows, hidden, mlp, mid, b.fc2_w, acc, ws, ws_len, s);
            try go(t.f.bias_residual, s, rows, 1, 256, .{ acc, b.fc2_b, x, int(hidden) });
        }
        if (t.run_depth < depth) {
            try s.synchronize();
            return .{ .x = x };
        }
        // projector: the patch rows (prefix dropped) shuffled 2x2, RMSNorm, Linear, relu^2, Linear
        try go(t.f.shuffle_rms, s, m4, 1, 256, .{ x, t.norm_w, h, int(prefix_tokens), int(gw), int(hidden), rms_eps });
        try mm.run(m4, projector_hidden, shuffled, h, t.proj1_w, acc, ws, ws_len, s);
        try go(t.f.relu2, s, m4, 1, 256, .{ acc, mid, int(projector_hidden) });
        try mm.run(m4, out_hidden, projector_hidden, mid, t.proj2_w, acc, ws, ws_len, s);
        try go(t.f.cast, s, m4, 1, 256, .{ acc, out, int(out_hidden) });
        try s.synchronize(); // the host's patch rows and tables are freed on return
        return .{ .x = x };
    }

    fn upload(t: *Tower, s: cuda.Stream, dst: u64, bytes: []const u8) !void {
        if (bytes.len == 0) return;
        try t.d.check(t.d.api.cuMemcpyHtoDAsync_v2(dst, bytes.ptr, bytes.len, s.handle), "cuMemcpyHtoDAsync");
    }
};

fn go(f: cuda.Function, s: cuda.Stream, gx: usize, gy: usize, block: u32, args: anytype) !void {
    var a: cuda.Args = .{};
    inline for (args) |v| a.add(v);
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(gx), .y = @intCast(gy) }, .block = .{ .x = block } }, s, &a);
}

fn carve(p: *u64, len: usize) u64 {
    const r = p.*;
    p.* += aligned(len);
    return r;
}

/// cuBLASLt plans for the shapes a call runs, made on first use and kept (a new image size adds plans).
const Gemms = struct {
    lib: *const lt.Library,
    plans: [48]?struct { m: usize, n: usize, k: usize, lin: lt.Linear } = @splat(null),
    next: usize = 0,

    fn run(g: *Gemms, m: usize, n: usize, k: usize, x: u64, w: u64, out: u64, ws: u64, ws_len: usize, s: cuda.Stream) !void {
        for (&g.plans) |*p| if (p.*) |*have| {
            if (have.m == m and have.n == n and have.k == k) return have.lin.run(x, w, out, ws, ws_len, s.handle);
        };
        // a full table drops its oldest plan (8 shapes an image size: six images' worth kept)
        const slot = &g.plans[g.next];
        if (slot.*) |*old| old.lin.deinit();
        slot.* = .{ .m = m, .n = n, .k = k, .lin = try lt.Linear.init(g.lib, m, n, k, .f32, ws_len) };
        g.next = (g.next + 1) % g.plans.len;
        return slot.*.?.lin.run(x, w, out, ws, ws_len, s.handle);
    }

    fn deinit(g: *Gemms) void {
        for (&g.plans) |*p| if (p.*) |*have| {
            have.lin.deinit();
            p.* = null;
        };
    }
};

fn aligned(n: usize) usize {
    return std.mem.alignForward(usize, n, 256);
}

fn int(x: usize) c_int {
    return @intCast(x);
}

/// fp32 to bf16, round to nearest even (torch's .to(bfloat16)).
fn bf16(v: f32) u16 {
    const b: u32 = @bitCast(v);
    if (std.math.isNan(v)) return 0x7fc0;
    return @intCast((b + 0x7fff + ((b >> 16) & 1)) >> 16);
}

/// CPE taps: each patch's 4 rows and weights in the 128x128 table resized to max(gh, gw) and cropped, as aten (fp32).
pub fn cpeTaps(gh: u32, gw: u32, idx: []i32, wts: []f32) void {
    const m = @max(gh, gw);
    for (0..gh) |i| {
        const hy = axis(@intCast(i), m);
        for (0..gw) |j| {
            const wx = axis(@intCast(j), m);
            const p = i * gw + j;
            idx[p * 4 + 0] = hy.lo * cpe_side + wx.lo;
            idx[p * 4 + 1] = hy.lo * cpe_side + wx.hi;
            idx[p * 4 + 2] = hy.hi * cpe_side + wx.lo;
            idx[p * 4 + 3] = hy.hi * cpe_side + wx.hi;
            wts[p * 4 + 0] = wx.wlo;
            wts[p * 4 + 1] = wx.whi;
            wts[p * 4 + 2] = hy.wlo;
            wts[p * 4 + 3] = hy.whi;
        }
    }
}

const Axis = struct { lo: i32, hi: i32, wlo: f32, whi: f32 };

/// aten's area_pixel_compute_source_index (linear, align_corners=False) and lambdas for output `o` of `size`.
fn axis(o: u32, size: u32) Axis {
    const scale: f32 = @as(f32, cpe_side) / @as(f32, @floatFromInt(size));
    const src = @max(scale * (@as(f32, @floatFromInt(o)) + 0.5) - 0.5, 0.0);
    const lo: i32 = @intFromFloat(src);
    const hi: i32 = if (lo < cpe_side - 1) lo + 1 else lo;
    const whi = @min(@max(src - @as(f32, @floatFromInt(lo)), 0.0), 1.0);
    return .{ .lo = lo, .hi = hi, .wlo = 1.0 - whi, .whi = whi };
}

test "CPE taps: a 128 grid reads the table as is; an upsampled one interpolates and clamps at the far edge" {
    var idx: [4 * 4]i32 = undefined;
    var wts: [4 * 4]f32 = undefined;
    const a = axis(5, cpe_side);
    try std.testing.expectEqual(@as(i32, 5), a.lo);
    try std.testing.expectEqual(@as(f32, 0), a.whi);
    const last = axis(255, 256);
    try std.testing.expectEqual(@as(i32, 127), last.lo);
    try std.testing.expectEqual(@as(i32, 127), last.hi);
    cpeTaps(2, 2, &idx, &wts);
    // m = 2: scale 64, source 31.5 for row/col 0
    try std.testing.expectEqual(@as(i32, 31 * cpe_side + 31), idx[0]);
    try std.testing.expectEqual(@as(f32, 0.5), wts[0]);
}

test "Bolt's config is accepted; another tower or projector is refused" {
    const gpa = std.testing.allocator;
    const good =
        \\{"vision_config":{"hidden_size":1280,"num_attention_heads":16,"num_hidden_layers":32,"patch_size":16,
        \\"num_cls_tokens":3,"num_registers":7,"hidden_act":"gelu","use_swiglu_ffn":false,"layer_norm_eps":1e-06,
        \\"mlp_ratio":4.0},"projector_hidden_size":20480,"img_context_token_id":18,"vit_hidden_size":1280,
        \\"llm_config":{"hidden_size":4096},"ps_version":"v2","use_thumbnail":false,"vision_final_layernorm":false,
        \\"downsample_ratio":0.5}
    ;
    try checkConfig(gpa, good);
    const cases = [_][2][]const u8{ .{ "\"num_registers\":7", "\"num_registers\":4" }, .{ "\"ps_version\":\"v2\"", "\"ps_version\":\"v1\"" }, .{ "\"use_swiglu_ffn\":false", "\"use_swiglu_ffn\":true" }, .{ "\"downsample_ratio\":0.5", "\"downsample_ratio\":0.25" } };
    for (cases) |c| {
        const bad = try std.mem.replaceOwned(u8, gpa, good, c[0], c[1]);
        defer gpa.free(bad);
        try std.testing.expectError(error.UnsupportedBoltVision, checkConfig(gpa, bad));
    }
}

/// An image prompt's rows for the prompt pass: its images' features, and each prompt row's feature row (-1: text).
pub const Media = struct {
    gpa: std.mem.Allocator,
    features: DeviceBuffer, // bf16 [image rows, out_hidden]
    src: DeviceBuffer, // i32 [prompt]

    /// `images` (the request's, in prompt order) through the tower on `s`; each fills its <img><image>...</img> run.
    pub fn init(gpa: std.mem.Allocator, t: *Tower, s: cuda.Stream, prompt: []const u32, images: []const lanes_.Image) !Media {
        const host = try gpa.alloc(i32, prompt.len);
        defer gpa.free(host);
        const rows = try placeholders(prompt, images, host);
        var m: Media = .{ .gpa = gpa, .features = try DeviceBuffer.alloc(t.d, @max(rows, 1) * out_hidden * 2), .src = undefined };
        errdefer m.features.free();
        m.src = try DeviceBuffer.alloc(t.d, prompt.len * 4);
        errdefer m.src.free();
        var at: usize = 0;
        for (images) |im| {
            const gh: u32 = @intCast(im.grid[1]);
            const gw: u32 = @intCast(im.grid[2]);
            _ = try t.encode(s, im.patches, gh, gw, m.features.ptr + at * out_hidden * 2);
            at += @as(usize, gh) * gw / 4;
        }
        try m.src.upload(0, std.mem.sliceAsBytes(host));
        return m;
    }

    pub fn deinit(m: *Media) void {
        m.features.free();
        m.src.free();
    }
};

/// Each image's image_token run between image_start and image_end; `src` gets each row's feature row (-1: text).
pub fn placeholders(prompt: []const u32, images: []const lanes_.Image, src: []i32) !usize {
    @memset(src, -1);
    var next: usize = 0; // feature rows handed out
    var k: usize = 0; // images placed
    var i: usize = 0;
    while (i < prompt.len) : (i += 1) {
        if (prompt[i] != image_token) continue;
        if (k == images.len or i == 0 or prompt[i - 1] != image_start) return error.BadImagePlaceholders;
        const im = images[k];
        if (im.grid[0] != 1 or im.grid[1] <= 0 or im.grid[2] <= 0 or @mod(im.grid[1] * im.grid[2], 4) != 0) return error.BadImageGrid;
        const n: usize = @intCast(@divExact(im.grid[1] * im.grid[2], 4));
        if (im.patches.len != n * 4 * patch_width) return error.BadImagePatches;
        if (i + n >= prompt.len or prompt[i + n] != image_end) return error.BadImagePlaceholders;
        for (prompt[i..][0..n], src[i..][0..n], 0..) |id, *x, j| {
            if (id != image_token) return error.BadImagePlaceholders;
            x.* = @intCast(next + j);
        }
        next += n;
        k += 1;
        i += n;
    }
    if (k != images.len) return error.BadImagePlaceholders;
    return next;
}

test "placeholder runs map to feature rows in order; a short run is refused" {
    const px: [8 * patch_width]f32 = @splat(0);
    const ims = [_]lanes_.Image{ .{ .patches = &px, .grid = .{ 1, 2, 2 } }, .{ .patches = &px, .grid = .{ 1, 2, 4 } } };
    const p = [_]u32{ 1, image_start, image_token, image_end, 7, image_start, image_token, image_token, image_end, 9 };
    var src: [p.len]i32 = undefined;
    // the first image is 4 patches (1 token) but its patches slice is 8: refused; give each its own size
    const one: [4 * patch_width]f32 = @splat(0);
    const good = [_]lanes_.Image{ .{ .patches = &one, .grid = .{ 1, 2, 2 } }, .{ .patches = &px, .grid = .{ 1, 2, 4 } } };
    try std.testing.expectEqual(@as(usize, 3), try placeholders(&p, &good, &src));
    try std.testing.expectEqualSlices(i32, &.{ -1, -1, 0, -1, -1, -1, 1, 2, -1, -1 }, &src);
    try std.testing.expectError(error.BadImagePatches, placeholders(&p, &ims, &src));
    const short = [_]u32{ image_start, image_token, image_end, image_start, image_token, image_end };
    try std.testing.expectError(error.BadImagePlaceholders, placeholders(&short, &good, &src));
}
