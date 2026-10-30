//! The Qwen-VL tower on CUDA (transformers' Qwen3_5VisionModel, bf16) that Qwen3.5/3.8 and Flash Next both run.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");

const vl = core.qwen_vl;
const lt = cuda.cublaslt;
const DeviceBuffer = cuda.DeviceBuffer;
const weights = @import("qwen_cuda_weights.zig");

/// What every Qwen3.5/3.8 tower shares (vision_config); its width, heads, depth and MLP are the checkpoint's.
pub const patch = weights.patch;
pub const temporal = weights.temporal;
pub const merge = 2;
pub const channels = weights.channels;
pub const patch_width = weights.patch_width; // 1536
pub const grid_side = weights.grid_side; // sqrt(num_position_embeddings 2304)
pub const eps: f32 = 1e-6;
/// Blocks a tower has at most; its heads are 64 or 72 wide (qwen_vision.cu's attention instances).
pub const max_depth = 32;
/// One tower call's patches at most: a 4,096-token image's.
pub const max_patches = 16384;
const workspace_bytes = 32 << 20;

/// The tower's shape and its placeholders' ids, from core/qwen_vl's settings of the checkpoint.
pub const Config = struct {
    hidden: usize,
    heads: usize,
    depth: usize,
    intermediate: usize,
    out_hidden: usize,
    image_token: u32,
    vision_start: u32,
    vision_end: u32,
    video_token: ?u32 = null, // a prompt holding it is refused where video is not served

    pub fn headDim(c: Config) usize {
        return c.hidden / c.heads;
    }

    /// The merger's width: four merge-ordered rows read as one.
    pub fn merged(c: Config) usize {
        return c.hidden * merge * merge;
    }

    /// `vc` as this tower runs it; config.json's text M-RoPE must be qwen35.cu's (interleaved 11, 11, 10).
    pub fn of(gpa: std.mem.Allocator, vc: vl.Config, config_json: []const u8) !Config {
        const geometry = vc.patch == patch and vc.temporal == temporal and vc.merge == merge;
        if (!geometry or vc.positions != grid_side * grid_side) return error.UnsupportedQwenVision;
        if (vc.depth > max_depth or (vc.headDim() != 64 and vc.headDim() != 72)) return error.UnsupportedQwenVision;
        return .{
            .hidden = vc.hidden,
            .heads = vc.heads,
            .depth = vc.depth,
            .intermediate = vc.intermediate,
            .out_hidden = vc.out_hidden,
            .image_token = vc.image_token,
            .vision_start = vc.vision_start,
            .vision_end = vc.vision_end,
            .video_token = try textRope(gpa, config_json),
        };
    }
};

/// qwen35.cu's rotate takes interleaved M-RoPE over sections 11, 11 and 10 only; the video token's id if given.
fn textRope(gpa: std.mem.Allocator, config_json: []const u8) !?u32 {
    const parsed = std.json.parseFromSlice(std.json.Value, gpa, config_json, .{}) catch
        return error.UnsupportedQwenVision;
    defer parsed.deinit();
    if (parsed.value != .object) return error.UnsupportedQwenVision;
    const root = parsed.value.object;
    var video: ?u32 = null;
    if (root.get("video_token_id")) |v| {
        if (v != .integer or v.integer < 0 or v.integer > std.math.maxInt(u32)) return error.UnsupportedQwenVision;
        const id: u32 = @intCast(v.integer);
        video = id;
    }
    const tc = root.get("text_config");
    const text = if (tc != null and tc.? == .object) tc.?.object else root;
    const rope = text.get("rope_parameters") orelse return error.UnsupportedQwenVision;
    if (rope != .object) return error.UnsupportedQwenVision;
    const inter = rope.object.get("mrope_interleaved") orelse return error.UnsupportedQwenVision;
    const sec = rope.object.get("mrope_section") orelse return error.UnsupportedQwenVision;
    if (inter != .bool or !inter.bool or sec != .array or sec.array.items.len != 3) return error.UnsupportedQwenVision;
    for (sec.array.items, [_]i64{ 11, 11, 10 }) |x, want|
        if (x != .integer or x.integer != want) return error.UnsupportedQwenVision;
    return video;
}

/// --vision's limits (native/cuda.zig's): image tokens a request holds in all, and images.
pub const Limits = struct { image_tokens: u32, max_images: u32 };

/// A checkpoint's tower and its image processor (core/qwen_vl), read before any CUDA work.
pub const Settings = struct {
    tower: Config,
    processor: vl.Config,

    /// config.json with its processor files in `dir`; null without config.json or a vision_config.
    pub fn read(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !?Settings {
        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        const a = arena.allocator();
        const config = try readFile(a, io, dir, "config.json") orelse return null;
        const pre = try readFile(a, io, dir, "preprocessor_config.json");
        const proc = try readFile(a, io, dir, "processor_config.json");
        const parsed = vl.Config.parse(a, config, pre, proc) catch |e|
            return if (e == error.OutOfMemory) e else error.UnsupportedQwenVision;
        const processor = parsed orelse return null;
        return .{ .tower = try Config.of(gpa, processor, config), .processor = processor };
    }

    /// An image's bytes to the tower's patches, at most `max_tokens` and one tower call's tokens.
    pub fn prepare(s: *const Settings, io: std.Io, a: std.mem.Allocator, bytes: []const u8, max_tokens: u32) !vl.geometry.Prepared {
        return vl.geometry.prepare(io, a, bytes, s.processor, @min(max_tokens, max_patches / (merge * merge)));
    }
};

/// What the server's image requests reach: the checkpoint's processor (core/qwen_vl) and this tower.
pub const Vision = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    shared: vl.Config,
    tower: *Tower,
    limits: Limits,
    markers: [3]u32, // vision_start, image_pad, vision_end: text that spells one stays text

    /// The checkpoint's settings and tower for a model `width` wide; error.NoQwenVision without a vision_config.
    pub fn open(
        gpa: std.mem.Allocator,
        io: std.Io,
        d: *const cuda.Driver,
        s: cuda.Stream,
        dir: []const u8,
        width: usize,
        limits: Limits,
    ) !*Vision {
        const settings = try Settings.read(gpa, io, dir) orelse return error.NoQwenVision;
        const c = settings.tower;
        if (c.out_hidden != width) return error.UnsupportedQwenVision;
        const v = try gpa.create(Vision);
        errdefer gpa.destroy(v);
        const tower = try Tower.load(gpa, io, d, s, dir, c);
        const markers = [3]u32{ c.vision_start, c.image_token, c.vision_end };
        v.* = .{ .gpa = gpa, .io = io, .shared = settings.processor, .tower = tower, .limits = limits, .markers = markers };
        return v;
    }

    pub fn deinit(v: *Vision) void {
        v.tower.deinit();
        v.gpa.destroy(v);
    }

    pub fn ids(v: *const Vision) vl.rope.Ids {
        return .{ .start = v.shared.vision_start, .image = v.shared.image_token, .end = v.shared.vision_end };
    }

    /// An image's bytes to the tower's patches (core/qwen_vl), at most `max_tokens` and one tower call's tokens.
    pub fn prepare(v: *const Vision, a: std.mem.Allocator, bytes: []const u8, max_tokens: u32) !vl.geometry.Prepared {
        return vl.geometry.prepare(v.io, a, bytes, v.shared, @min(max_tokens, max_patches / (merge * merge)));
    }

    /// The words for a prepare error the server cannot word (a JPEG library this machine lacks).
    pub const explain = vl.geometry.problem;
};

/// `name` in `dir`, or null when it is not there.
fn readFile(a: std.mem.Allocator, io: std.Io, dir: []const u8, name: []const u8) !?[]const u8 {
    const path = try std.fs.path.join(a, &.{ dir, name });
    return std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(16 << 20)) catch |e| switch (e) {
        error.FileNotFound => null,
        else => e,
    };
}

/// One image's patch grid (rows and columns of 16x16 patches, both even).
pub const Grid = struct { h: u32, w: u32 };

const Block = struct { norm1_w: u64, norm1_b: u64, qkv_w: u64, qkv_b: u64, proj_w: u64, proj_b: u64, norm2_w: u64, norm2_b: u64, fc1_w: u64, fc1_b: u64, fc2_w: u64, fc2_b: u64 };

const Fns = struct {
    embed: cuda.Function,
    layernorm: cuda.Function,
    qkv_rope: cuda.Function,
    attention: cuda.Function,
    bias_gelu: cuda.Function,
    bias_residual: cuda.Function,
    bias_cast: cuda.Function,
};

/// The tower's weights and one call's scratch in one allocation, its kernels; `encode` turns patches into model rows.
pub const Tower = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    s: cuda.Stream,
    c: Config,
    mod: cuda.Module,
    f: Fns,
    lib: lt.Library,
    arena: DeviceBuffer,
    at: usize = 0,
    scratch: u64 = 0,
    patch_w: u64 = 0,
    patch_b: u64 = 0,
    pos: u64 = 0,
    blocks: [max_depth]Block = undefined,
    merger_norm_w: u64 = 0,
    merger_norm_b: u64 = 0,
    fc1_w: u64 = 0,
    fc1_b: u64 = 0,
    fc2_w: u64 = 0,
    fc2_b: u64 = 0,
    inv: u64 = 0,

    /// The checkpoint's vision tensors (bf16) on stream `s`; the patch convolution's weight as [C, T, y, x] rows.
    pub fn load(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, s: cuda.Stream, dir: []const u8, c: Config) !*Tower {
        const t = try gpa.create(Tower);
        errdefer gpa.destroy(t);
        t.* = .{ .gpa = gpa, .d = d, .s = s, .c = c, .mod = undefined, .f = undefined, .lib = undefined, .arena = undefined };
        t.lib = try lt.Library.open();
        errdefer t.lib.close();
        t.mod = try cuda.Module.load(d, cuda.kernels.qwen_vision);
        errdefer t.mod.unload();
        inline for (@typeInfo(Fns).@"struct".field_names) |name| @field(t.f, name) = try t.mod.function(if (comptime std.mem.eql(u8, name, "attention")) "tf_qv_attention72" else "tf_qv_" ++ name);
        if (c.headDim() == 64) t.f.attention = try t.mod.function("tf_qv_attention64");
        t.arena = try DeviceBuffer.alloc(d, weightBytes(c) + scratchBytes(c, max_patches));
        errdefer t.arena.free();
        try t.arena.fill8(0, null); // touched once now: the driver commits memory on first touch, slowly
        try weights.load(gpa, io, dir, c, t);
        var inv: [72 / 4]f32 = undefined;
        invFreq(inv[0 .. c.headDim() / 4], c.headDim());
        t.inv = try t.put(std.mem.sliceAsBytes(inv[0 .. c.headDim() / 4]));
        t.scratch = try t.take(scratchBytes(c, max_patches));
        return t;
    }

    pub fn deinit(t: *Tower) void {
        t.arena.free();
        t.mod.unload();
        t.lib.close();
        t.gpa.destroy(t);
    }

    /// The weights' bytes in the arena (each allocation 256-byte aligned).
    pub fn weightBytes(c: Config) usize {
        const a = aligned;
        const hidden = c.hidden;
        const intermediate = c.intermediate;
        const merged = c.merged();
        const out_hidden = c.out_hidden;
        var n: usize = a(hidden * patch_width * 2) + a(hidden * 2) + a(grid_side * grid_side * hidden * 2);
        n += c.depth * (4 * a(hidden * 2) + a(3 * hidden * hidden * 2) + a(3 * hidden * 2) + a(hidden * hidden * 2) + a(hidden * 2) + a(intermediate * hidden * 2) + a(intermediate * 2) + a(hidden * intermediate * 2) + a(hidden * 2));
        n += 2 * a(hidden * 2) + a(merged * merged * 2) + a(merged * 2) + a(out_hidden * merged * 2) + a(out_hidden * 2);
        return n + a(c.headDim() / 4 * 4);
    }

    /// The scratch one tower call of `n` patches needs, in bytes (the patches' bf16 copy included).
    pub fn scratchBytes(c: Config, n: usize) usize {
        const hidden = c.hidden;
        const intermediate = c.intermediate;
        return n * patch_width * 2 + // patches
            2 * n * hidden * 2 + // x, h
            n * @max(3 * hidden, intermediate) * 4 + // fp32 Linear out
            3 * n * hidden * 2 + // q, k, v
            n * hidden * 2 + // attention out
            n * @max(intermediate, hidden) * 2 + // gelu rows
            n * (2 + 4) * 4 + n * 4 * 4 + // hw, idx, weights
            16 * 256 + workspace_bytes;
    }

    fn take(t: *Tower, len: usize) !u64 {
        const at = aligned(t.at);
        if (at + len > t.arena.len) return error.ArenaExhausted;
        t.at = at + len;
        return t.arena.ptr + at;
    }

    /// `bytes` into the arena; a failed wait drains before load's frees of the host rows or the mapped checkpoint.
    pub fn put(t: *Tower, bytes: []const u8) !u64 {
        const p = try t.take(bytes.len);
        try @import("qwen_completion.zig").upload(t.arena, p - t.arena.ptr, bytes);
        return p;
    }

    fn upload(t: *Tower, dst: u64, bytes: []const u8) !void {
        if (bytes.len == 0) return;
        try cuda.copy_source.enqueue(t.d, dst, bytes, t.s);
    }

    /// Images' patches (fp32, merge order) to merged bf16 [sum(h * w) / 4, out_hidden] at `out`, each image on its own.
    pub fn encode(t: *Tower, pixels: []const f32, grids: []const Grid, out: u64) !void {
        return t.encodeWith(pixels, grids, out, Gemms);
    }

    /// `encode` over `Matmuls`, cuBLASLt's or a test's; a failure drains the stream before an upload's source is freed.
    pub fn encodeWith(t: *Tower, pixels: []const f32, grids: []const Grid, out: u64, comptime Matmuls: type) !void {
        var n: usize = 0;
        for (grids) |g| {
            if (g.h == 0 or g.w == 0 or g.h % merge != 0 or g.w % merge != 0) return error.BadImageGrid;
            n += @as(usize, g.h) * g.w;
        }
        if (n == 0 or n > max_patches or pixels.len != n * patch_width) return error.BadImagePatches;
        const gpa = t.gpa;
        const c = t.c;
        const out_hidden = c.out_hidden;
        const hidden = c.hidden;
        const heads = c.heads;
        const head_dim = c.headDim();
        const intermediate = c.intermediate;
        const merged = c.merged();
        // host side: bf16 patches, each patch's (row, column) and its 4 position-grid taps
        const px16 = try gpa.alloc(u16, n * patch_width);
        defer gpa.free(px16);
        for (px16, pixels) |*o, v| o.* = bf16(v);
        const hw = try gpa.alloc(i32, n * 2);
        defer gpa.free(hw);
        const idx = try gpa.alloc(i32, n * 4);
        defer gpa.free(idx);
        const wts = try gpa.alloc(f32, n * 4);
        defer gpa.free(wts);
        var at: usize = 0;
        for (grids) |g| {
            const len = @as(usize, g.h) * g.w;
            positions(g, hw[at * 2 ..][0 .. len * 2], idx[at * 4 ..][0 .. len * 4], wts[at * 4 ..][0 .. len * 4]);
            at += len;
        }
        var cur: u64 = t.scratch;
        const end = t.scratch + scratchBytes(c, max_patches);
        const take_ = struct {
            fn f(p: *u64, len: usize) u64 {
                const r = p.*;
                p.* += std.mem.alignForward(usize, len, 256);
                return r;
            }
        }.f;
        const d_px = take_(&cur, n * patch_width * 2);
        const x = take_(&cur, n * hidden * 2);
        const h = take_(&cur, n * hidden * 2);
        const acc = take_(&cur, n * @max(3 * hidden, intermediate) * 4);
        const q = take_(&cur, n * hidden * 2);
        const k = take_(&cur, n * hidden * 2);
        const v = take_(&cur, n * hidden * 2);
        const att = take_(&cur, n * hidden * 2);
        const mid = take_(&cur, n * @max(intermediate, hidden) * 2);
        const d_hw = take_(&cur, n * 2 * 4);
        const d_idx = take_(&cur, n * 4 * 4);
        const d_wts = take_(&cur, n * 4 * 4);
        const ws = cur;
        const ws_len = end - cur;
        var mm: Matmuls = .{ .lib = &t.lib };
        defer mm.deinit();
        // runs before the defers above: the GPU may still read px16, hw, idx and wts or run a plan
        errdefer @import("qwen_completion.zig").drain(t.d, t.s);
        try t.upload(d_px, std.mem.sliceAsBytes(px16));
        try t.upload(d_hw, std.mem.sliceAsBytes(hw));
        try t.upload(d_idx, std.mem.sliceAsBytes(idx));
        try t.upload(d_wts, std.mem.sliceAsBytes(wts));

        const m4 = n / (merge * merge);
        try mm.run(n, hidden, patch_width, d_px, t.patch_w, acc, ws, ws_len, t.s);
        try t.go(t.f.embed, n, 1, 256, .{ acc, t.patch_b, t.pos, d_idx, d_wts, x, int(hidden) });
        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));
        for (t.blocks[0..c.depth]) |b| {
            try t.go(t.f.layernorm, n, 1, 256, .{ x, b.norm1_w, b.norm1_b, h, int(hidden), eps });
            try mm.run(n, 3 * hidden, hidden, h, b.qkv_w, acc, ws, ws_len, t.s);
            try t.go(t.f.qkv_rope, n, heads, 128, .{ acc, b.qkv_b, d_hw, t.inv, q, k, v, int(n), int(heads), int(head_dim) });
            var s0: usize = 0;
            for (grids) |g| {
                const len = @as(usize, g.h) * g.w;
                try t.go(t.f.attention, (len + 63) / 64, heads, 64, .{ q, k, v, att, int(n), int(heads), int(s0), int(len), scale });
                s0 += len;
            }
            try mm.run(n, hidden, hidden, att, b.proj_w, acc, ws, ws_len, t.s);
            try t.go(t.f.bias_residual, n, 1, 256, .{ acc, b.proj_b, x, int(hidden) });
            try t.go(t.f.layernorm, n, 1, 256, .{ x, b.norm2_w, b.norm2_b, h, int(hidden), eps });
            try mm.run(n, intermediate, hidden, h, b.fc1_w, acc, ws, ws_len, t.s);
            try t.go(t.f.bias_gelu, n, 1, 256, .{ acc, b.fc1_b, mid, int(intermediate), int(1) });
            try mm.run(n, hidden, intermediate, mid, b.fc2_w, acc, ws, ws_len, t.s);
            try t.go(t.f.bias_residual, n, 1, 256, .{ acc, b.fc2_b, x, int(hidden) });
        }
        // merger: LayerNorm over 1152, four merge-ordered rows read as one of 4608
        try t.go(t.f.layernorm, n, 1, 256, .{ x, t.merger_norm_w, t.merger_norm_b, h, int(hidden), eps });
        try mm.run(m4, merged, merged, h, t.fc1_w, acc, ws, ws_len, t.s);
        try t.go(t.f.bias_gelu, m4, 1, 256, .{ acc, t.fc1_b, mid, int(merged), int(0) });
        try mm.run(m4, out_hidden, merged, mid, t.fc2_w, acc, ws, ws_len, t.s);
        try t.go(t.f.bias_cast, m4, 1, 256, .{ acc, t.fc2_b, out, int(out_hidden) });
        try t.s.synchronize(); // the host's patch rows and tables are freed on return
    }

    fn go(t: *Tower, f: cuda.Function, gx: usize, gy: usize, block: u32, args: anytype) !void {
        var a: cuda.Args = .{};
        inline for (args) |v| a.add(v);
        try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(gx), .y = @intCast(gy) }, .block = .{ .x = block } }, t.s, &a);
    }
};

/// cuBLASLt plans for the shapes one tower call runs, made on first use.
const Gemms = struct {
    lib: *const lt.Library,
    plans: [8]?struct { m: usize, n: usize, k: usize, lin: lt.Linear } = @splat(null),

    fn run(g: *Gemms, m: usize, n: usize, k: usize, x: u64, w: u64, out: u64, ws: u64, ws_len: usize, s: cuda.Stream) !void {
        for (&g.plans) |*p| {
            if (p.*) |*have| {
                if (have.m == m and have.n == n and have.k == k) return have.lin.run(x, w, out, ws, ws_len, s.handle);
                continue;
            }
            p.* = .{ .m = m, .n = n, .k = k, .lin = try lt.Linear.init(g.lib, m, n, k, .f32, ws_len) };
            return p.*.?.lin.run(x, w, out, ws, ws_len, s.handle);
        }
        return error.TooManyShapes;
    }

    fn deinit(g: *Gemms) void {
        for (&g.plans) |*p| if (p.*) |*have| have.lin.deinit();
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

/// The axial rotary's frequencies: 1 / 10000 ** (i / (head_dim / 2)) for even i, in fp32 as torch computes them.
pub fn invFreq(out: []f32, head_dim: usize) void {
    const dim: f32 = @floatFromInt(head_dim / 2);
    for (out, 0..) |*f, j| {
        const e: f32 = @as(f32, @floatFromInt(2 * j)) / dim;
        f.* = 1.0 / std.math.pow(f32, 10000.0, e);
    }
}

/// One image's patches in merge order: each one's (row, column) for the rotary and 4 bilinear taps into the 48x48 grid.
pub fn positions(g: Grid, hw: []i32, idx: []i32, wts: []f32) void {
    const bw = g.w / merge;
    const n = @as(usize, g.h) * g.w;
    for (0..n) |i| {
        const in_col: u32 = @intCast(i % merge);
        const in_row: u32 = @intCast((i / merge) % merge);
        const block_col: u32 = @intCast((i / (merge * merge)) % bw);
        const block_row: u32 = @intCast(i / (merge * merge * bw));
        const r = block_row * merge + in_row;
        const c = block_col * merge + in_col;
        hw[i * 2] = @intCast(r);
        hw[i * 2 + 1] = @intCast(c);
        const ht = taps(r, g.h);
        const wt = taps(c, g.w);
        for (0..2) |a| for (0..2) |b| {
            idx[i * 4 + a * 2 + b] = ht.i[a] * grid_side + wt.i[b];
            wts[i * 4 + a * 2 + b] = ht.w[a] * wt.w[b];
        };
    }
}

const Taps = struct { i: [2]i32, w: [2]f32 };

fn taps(index: u32, size: u32) Taps {
    const side_m1: f32 = grid_side - 1;
    const denom: f32 = @floatFromInt(@max(size, 2) - 1);
    const src = @as(f32, @floatFromInt(index)) * side_m1 / denom;
    const fl = @floor(src);
    var out: Taps = undefined;
    for (0..2) |o| {
        const raw: i32 = @as(i32, @intFromFloat(fl)) + @as(i32, @intCast(o));
        out.i[o] = std.math.clamp(raw, 0, grid_side - 1);
        const dist = @abs(src - fl - @as(f32, @floatFromInt(o)));
        out.w[o] = @max(1.0 - dist, 0.0);
    }
    return out;
}

test "bilinear taps match align_corners resampling at the ends and middle" {
    const a = taps(0, 10);
    try std.testing.expectEqual(@as(i32, 0), a.i[0]);
    try std.testing.expectEqual(@as(f32, 1), a.w[0]);
    const b = taps(9, 10);
    try std.testing.expectEqual(@as(i32, 47), b.i[0]);
    try std.testing.expectEqual(@as(f32, 1), b.w[0]);
}

test "a vision_config's tower is taken; other patches, depths, head widths and M-RoPE are refused" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const good =
        \\{"vision_config":{"depth":27,"hidden_size":1152,"num_heads":16,"intermediate_size":4304,"patch_size":16,
        \\"temporal_patch_size":2,"spatial_merge_size":2,"in_channels":3,"num_position_embeddings":2304,
        \\"hidden_act":"gelu_pytorch_tanh","deepstack_visual_indexes":[],"out_hidden_size":5120},
        \\"text_config":{"rope_parameters":{"mrope_interleaved":true,"mrope_section":[11,11,10]}},
        \\"image_token_id":248056,"video_token_id":248057,"vision_start_token_id":248053,"vision_end_token_id":248054}
    ;
    const parse = struct {
        fn f(al: std.mem.Allocator, text: []const u8) !Config {
            return Config.of(al, (try vl.Config.parse(al, text, null, null)).?, text);
        }
    }.f;
    const c = try parse(a, good);
    try std.testing.expectEqual(@as(usize, 5120), c.out_hidden);
    try std.testing.expectEqual(@as(u32, 248056), c.image_token);
    try std.testing.expectEqual(@as(?u32, 248057), c.video_token);
    try std.testing.expectEqual(@as(usize, 72), c.headDim());
    try std.testing.expectEqual(@as(?vl.Config, null), try vl.Config.parse(a, "{\"hidden_size\":2048}", null, null));
    const refused = [_][2][]const u8{
        .{ "\"depth\":27", "\"depth\":33" },
        .{ "\"num_heads\":16", "\"num_heads\":12" },
        .{ "\"patch_size\":16", "\"patch_size\":14" },
        .{ "\"num_position_embeddings\":2304", "\"num_position_embeddings\":1024" },
        .{ "[11,11,10]", "[16,24,24]" },
        .{ "\"mrope_interleaved\":true", "\"mrope_interleaved\":false" },
    };
    for (refused) |r| {
        const text = try std.mem.replaceOwned(u8, a, good, r[0], r[1]);
        try std.testing.expectError(error.UnsupportedQwenVision, parse(a, text));
    }
    const invalid_video = try std.mem.replaceOwned(u8, a, good, "\"video_token_id\":248057", "\"video_token_id\":-1");
    try std.testing.expectError(error.VisionConfig, parse(a, invalid_video));
    // the 2B's tower: 24 blocks of 1024, heads of 64
    const big = "\"depth\":27,\"hidden_size\":1152,\"num_heads\":16,\"intermediate_size\":4304";
    const two = "\"depth\":24,\"hidden_size\":1024,\"num_heads\":16,\"intermediate_size\":4096";
    const small = try std.mem.replaceOwned(u8, a, good, big, two);
    const c2 = try parse(a, small);
    try std.testing.expectEqual(@as(usize, 64), c2.headDim());
    try std.testing.expectEqual(@as(usize, 24), c2.depth);
}

/// A Qwen-VL prompt's image rows on the device, shared by every CUDA family that runs this tower.
pub const media = @import("qwen_cuda_media.zig");
/// The completion policy the tower's owners follow: drain the stream, then the context, or end the process.
pub const completion = @import("qwen_completion.zig");

test {
    _ = media;
    _ = @import("qwen_cuda_test.zig");
}
