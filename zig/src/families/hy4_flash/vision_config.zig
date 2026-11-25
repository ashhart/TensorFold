//! Hy4-flash vision config, image preprocessing, 128x128 bilinear position taps, and 3D M-RoPE layout.
const std = @import("std");
const core = @import("core");

const picture = core.picture;
const patches = picture.patches;

pub const patch: u32 = 16;
pub const merge: u32 = 2;
pub const channels: u32 = 3;
pub const patch_width: usize = channels * patch * patch; // 768
pub const position_edge: u32 = 128; // 2048 / 16
pub const num_positions: usize = 1 + @as(usize, position_edge) * position_edge; // 16,385 (row 0 = unused CLS)
pub const max_patches: usize = 16_384; // 128 x 128 patches (up to 2048 x 2048 image)

pub const Limits = struct {
    image_tokens: u32 = 4224, // >= 64 x (64 + 1) = 4,160 tokens for a full 2048x2048 image
    max_images: u32 = 16,
};

pub const Config = struct {
    hidden: usize = 1152,
    heads: usize = 16,
    depth: usize = 27,
    intermediate: usize = 4304,
    out_hidden: usize = 3072,
    eps: f32 = 1e-5,
    image_start: u32 = 120118,
    image_token: u32 = 120120,
    image_end: u32 = 120119,
    video_token: u32 = 120124,
    min_pixels: u64 = 4096,
    max_pixels: u64 = 4_194_304,
    rescale: f64 = 1.0 / 255.0,
    mean: [3]f32 = .{ 0.5, 0.5, 0.5 },
    std_: [3]f32 = .{ 0.5, 0.5, 0.5 },

    pub fn headDim(c: Config) usize {
        return c.hidden / c.heads;
    }

    pub fn mergedWidth(c: Config) usize {
        return c.hidden * merge * merge;
    }

    /// Reads `vision_config` from `config.json` and optional normalization/bounds from `processor_config.json`.
    pub fn read(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !?Config {
        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        const a = arena.allocator();
        const cfg_txt = try readOptFile(a, io, dir, "config.json") orelse return null;
        const proc_txt = try readOptFile(a, io, dir, "processor_config.json");
        const pre_txt = try readOptFile(a, io, dir, "preprocessor_config.json");
        return parse(a, cfg_txt, proc_txt orelse pre_txt);
    }

    /// Parses `vision_config` from JSON and validates Hy4-flash SigLIP2 geometry and processor bounds.
    pub fn parse(a: std.mem.Allocator, cfg_txt: []const u8, proc_txt: ?[]const u8) !?Config {
        var parsed = try std.json.parseFromSlice(std.json.Value, a, cfg_txt, .{});
        defer parsed.deinit();
        if (parsed.value != .object) return error.BadVisionConfig;
        const top = parsed.value.object;
        const vv = top.get("vision_config") orelse return null;
        if (vv != .object) return error.BadVisionConfig;
        const vc = vv.object;

        var c: Config = .{
            .hidden = try optUsize(vc, "hidden_size", 1152),
            .heads = try optUsize(vc, "num_attention_heads", 16),
            .depth = try optUsize(vc, "num_hidden_layers", 27),
            .intermediate = try optUsize(vc, "intermediate_size", 4304),
            .out_hidden = try optUsize(vc, "out_hidden_size", 3072),
            .eps = @floatCast(try optFloat(vc, "rms_norm_eps", 1e-5)),
            .image_start = try optToken(top, "image_start_token_id", 120118),
            .image_token = try optToken(top, "image_token_id", 120120),
            .image_end = try optToken(top, "image_end_token_id", 120119),
            .video_token = try optToken(top, "video_token_id", 120124),
        };
        const p_sz = try optUsize(vc, "patch_size", 16);
        const m_sz = try optUsize(vc, "spatial_merge_size", 2);
        const ch = try optUsize(vc, "num_channels", 3);
        const max_im = try optUsize(vc, "max_image_size", 2048);
        if (p_sz != patch or m_sz != merge or ch != channels or max_im != position_edge * patch) {
            return error.UnsupportedHy4Vision;
        }
        if (c.hidden != 1152 or c.heads != 16 or c.depth == 0 or c.depth > 32 or c.intermediate != 4304 or c.out_hidden != 3072) {
            return error.UnsupportedHy4Vision;
        }
        if (!std.math.isFinite(c.eps) or c.eps <= 0) return error.BadVisionConfig;
        if (vc.get("hidden_act")) |ha| if (ha != .string or !std.mem.eql(u8, ha.string, "gelu")) return error.UnsupportedHy4Vision;

        if (proc_txt) |pt| {
            var pp = try std.json.parseFromSlice(std.json.Value, a, pt, .{});
            defer pp.deinit();
            if (pp.value != .object) return error.BadVisionConfig;
            const root = pp.value.object;
            const ip = if (root.get("image_processor")) |sub| (if (sub == .object) sub.object else root) else root;
            if (ip.get("min_pixels")) |v| {
                if (v != .integer or v.integer <= 0) return error.BadVisionConfig;
                c.min_pixels = @intCast(v.integer);
            }
            if (ip.get("max_pixels")) |v| {
                if (v != .integer or v.integer <= 0) return error.BadVisionConfig;
                c.max_pixels = @intCast(v.integer);
            }
            if (ip.get("size")) |sv| if (sv == .object) {
                if (sv.object.get("shortest_edge")) |v| {
                    if (v != .integer or v.integer <= 0) return error.BadVisionConfig;
                    c.min_pixels = @intCast(v.integer);
                }
                if (sv.object.get("longest_edge")) |v| {
                    if (v != .integer or v.integer <= 0) return error.BadVisionConfig;
                    c.max_pixels = @intCast(v.integer);
                }
            };
            c.rescale = try optFloat(ip, "rescale_factor", c.rescale);
            if (ip.get("image_mean")) |mv| c.mean = try parseVec3(mv);
            if (ip.get("image_std")) |sv| c.std_ = try parseVec3(sv);
        }
        if (c.min_pixels == 0 or c.max_pixels < c.min_pixels) return error.BadVisionConfig;
        if (!std.math.isFinite(c.rescale) or c.rescale <= 0) return error.BadVisionConfig;
        for (c.mean, c.std_) |m, s| {
            if (!std.math.isFinite(m) or !std.math.isFinite(s) or s <= 0) return error.BadVisionConfig;
        }
        return c;
    }
};

/// Merged LLM tokens for patch grid `(gh, gw)`: `(gh / 2) * (gw / 2 + 1)` (one `image_newline` per row).
pub fn tokensForGrid(gh: u32, gw: u32) u32 {
    return (gh / merge) * (gw / merge + 1);
}

/// Decodes and preprocesses raw image bytes into flat raster patches `[gh * gw, 768]` and `(gh, gw, tokens, hash)`.
pub fn prepare(io: std.Io, gpa: std.mem.Allocator, bytes: []const u8, c: Config, max_tokens: u32) patches.Error!patches.Prepared {
    var rgb = try picture.decode(io, gpa, bytes);
    defer rgb.deinit(gpa);
    return prepareRgb(gpa, rgb, picture.identity.ofBytes(0x68793476, bytes), c, max_tokens);
}

/// Smart-resizes an RGB image onto 32-pixel multiples within `max_tokens` and patchifies in raster order.
pub fn prepareRgb(gpa: std.mem.Allocator, img: picture.Rgb, bytes_hash: u64, c: Config, max_tokens: u32) patches.Error!patches.Prepared {
    if (img.width == 0 or img.height == 0) return error.Invalid;
    const expected_pixels = std.math.mul(usize, img.width, img.height) catch return error.Invalid;
    const expected_bytes = std.math.mul(usize, expected_pixels, 3) catch return error.Invalid;
    if (img.pixels.len != expected_bytes) return error.Invalid;
    if (max_tokens < 2) return error.ImageTooSmall;
    const factor: u32 = patch * merge; // 32
    const unit_px: u64 = @as(u64, factor) * factor; // 1024 px per merged spatial token
    var ceiling: u64 = @min(c.max_pixels, @as(u64, max_tokens) * unit_px);
    var floor_px: u64 = @min(c.min_pixels, ceiling);
    var size = try patches.smartResize(@intCast(img.height), @intCast(img.width), factor, floor_px, ceiling);
    while (tokensForGrid(size.h / patch, size.w / patch) > max_tokens and (size.h > factor or size.w > factor)) {
        const cur_px: u64 = @as(u64, size.h) * size.w;
        if (cur_px <= unit_px) break;
        ceiling = cur_px - unit_px;
        floor_px = @min(c.min_pixels, ceiling);
        size = try patches.smartResize(@intCast(img.height), @intCast(img.width), factor, floor_px, ceiling);
    }
    const gh = size.h / patch;
    const gw = size.w / patch;
    if (@as(usize, gh) * gw > max_patches) return error.Oversized;
    const resized = try picture.resample.resize(gpa, .pillow, .of(img.pixels, img.width, img.height), size.w, size.h);
    defer gpa.free(resized);
    const px = try patchifyRaster(gpa, .{ .width = size.w, .height = size.h, .pixels = resized }, c);
    return .{
        .pixels = px,
        .gh = gh,
        .gw = gw,
        .tokens = tokensForGrid(gh, gw),
        .hash = picture.identity.withGrid(bytes_hash, .{ gh, gw }),
    };
}

/// Raster patchify matching `HYV4FlashImageProcessorPil.patchify`: `(gh, gw, C=3, ph=16, pw=16)`.
pub fn patchifyRaster(gpa: std.mem.Allocator, img: picture.Rgb, c: Config) error{OutOfMemory}![]f32 {
    const gh = img.height / patch;
    const gw = img.width / patch;
    const out = try gpa.alloc(f32, gh * gw * patch_width);
    var at: usize = 0;
    for (0..gh) |py| for (0..gw) |px| {
        const top = py * patch;
        const left = px * patch;
        for (0..3) |ch| for (0..patch) |y| for (0..patch) |x| {
            const raw = img.pixels[((top + y) * img.width + left + x) * 3 + ch];
            const scaled: f32 = @floatCast(@as(f64, @floatFromInt(raw)) * c.rescale);
            out[at] = (scaled - c.mean[ch]) / c.std_[ch];
            at += 1;
        };
    };
    return out;
}

const Taps = struct { i: [2]i32, w: [2]f32 };

fn axisTap(dst: u32, size: u32) Taps {
    const scale: f64 = (@as(f64, @floatFromInt(size)) + 0.1) / @as(f64, position_edge);
    const inv_scale: f64 = 1.0 / scale;
    const raw: f64 = (@as(f64, @floatFromInt(dst)) + 0.5) * inv_scale - 0.5;
    const clamped: f32 = @floatCast(std.math.clamp(raw, 0.0, @as(f64, position_edge - 1)));
    const fl = @floor(clamped);
    const lo: i32 = @intFromFloat(fl);
    const hi: i32 = @min(lo + 1, @as(i32, position_edge - 1));
    const d: f32 = clamped - fl;
    return .{ .i = .{ lo, hi }, .w = .{ 1.0 - d, d } };
}

/// Bilinear 4-tap indices & weights into the `128 x 128` position grid (`scale_factor = (size + 0.1) / 128`).
pub fn positions(gh: u32, gw: u32, idx: []i32, wts: []f32) void {
    const total = @as(usize, gh) * gw;
    std.debug.assert(gh > 0 and gw > 0 and idx.len == total * 4 and wts.len == total * 4);
    const edge: i32 = position_edge;
    if (gh == position_edge and gw == position_edge) {
        for (0..total) |p| {
            idx[p * 4] = @intCast(p);
            idx[p * 4 + 1] = 0;
            idx[p * 4 + 2] = 0;
            idx[p * 4 + 3] = 0;
            wts[p * 4] = 1.0;
            wts[p * 4 + 1] = 0.0;
            wts[p * 4 + 2] = 0.0;
            wts[p * 4 + 3] = 0.0;
        }
        return;
    }
    var p: usize = 0;
    for (0..gh) |r| {
        const ht = axisTap(@intCast(r), gh);
        for (0..gw) |col| {
            const wt = axisTap(@intCast(col), gw);
            for (0..2) |a| for (0..2) |b| {
                idx[p * 4 + a * 2 + b] = ht.i[a] * edge + wt.i[b];
                wts[p * 4 + a * 2 + b] = ht.w[a] * wt.w[b];
            };
            p += 1;
        }
    }
}

pub const Run = struct {
    at: u32,
    gh: u32,
    gw: u32,

    pub fn tokens(r: Run) usize {
        return tokensForGrid(r.gh, r.gw);
    }

    pub fn patchesCount(r: Run) usize {
        return @as(usize, r.gh) * r.gw;
    }
};

/// Computes `src_idx [n]` (`-1` on text, `0..total_img_tokens` on image pads) and `pos3 [3, n]` + `rope_delta`.
pub fn layout(n: usize, runs: []const Run, src: []i32, pos3: []i32) error{BadVisionLayout}!i32 {
    if (src.len != n or pos3.len != (std.math.mul(usize, 3, n) catch return error.BadVisionLayout)) {
        return error.BadVisionLayout;
    }
    @memset(src, -1);
    var next: i32 = 0;
    var cursor: usize = 0;
    var feat_row: i32 = 0;
    for (runs) |im| {
        if (im.gh == 0 or im.gw == 0 or im.gh % merge != 0 or im.gw % merge != 0) return error.BadVisionLayout;
        const begin: usize = im.at;
        const mh: usize = im.gh / merge;
        const mw_nl: usize = im.gw / merge + 1;
        const span = std.math.mul(usize, mh, mw_nl) catch return error.BadVisionLayout;
        const end = std.math.add(usize, begin, span) catch return error.BadVisionLayout;
        if (begin < cursor or end > n) return error.BadVisionLayout;
        for (cursor..begin) |r| {
            const v = next + @as(i32, @intCast(r - cursor));
            pos3[r] = v;
            pos3[n + r] = v;
            pos3[2 * n + r] = v;
        }
        const base = next + @as(i32, @intCast(begin - cursor));
        for (0..mh) |oh| for (0..mw_nl) |ow| {
            const r = begin + oh * mw_nl + ow;
            pos3[r] = base;
            pos3[n + r] = base + @as(i32, @intCast(oh));
            pos3[2 * n + r] = base + @as(i32, @intCast(ow));
            src[r] = feat_row;
            feat_row += 1;
        };
        next = base + @as(i32, @intCast(@max(mh, mw_nl)));
        cursor = end;
    }
    for (cursor..n) |r| {
        const v = next + @as(i32, @intCast(r - cursor));
        pos3[r] = v;
        pos3[n + r] = v;
        pos3[2 * n + r] = v;
    }
    return next - @as(i32, @intCast(cursor));
}

fn readOptFile(a: std.mem.Allocator, io: std.Io, dir: []const u8, name: []const u8) !?[]const u8 {
    const p = try std.fs.path.join(a, &.{ dir, name });
    return std.Io.Dir.cwd().readFileAlloc(io, p, a, .limited(8 << 20)) catch |e| switch (e) {
        error.FileNotFound => null,
        else => e,
    };
}

fn optUsize(o: std.json.ObjectMap, k: []const u8, d: usize) !usize {
    const v = o.get(k) orelse return d;
    if (v != .integer or v.integer <= 0) return error.BadVisionConfig;
    return @intCast(v.integer);
}

fn optFloat(o: std.json.ObjectMap, k: []const u8, d: f64) !f64 {
    const v = o.get(k) orelse return d;
    return switch (v) {
        .float => |f| f,
        .integer => |i| @floatFromInt(i),
        else => error.BadVisionConfig,
    };
}

fn optToken(o: std.json.ObjectMap, k: []const u8, d: u32) !u32 {
    const v = o.get(k) orelse return d;
    if (v != .integer or v.integer < 0 or v.integer > std.math.maxInt(u32)) return error.BadVisionConfig;
    return @intCast(v.integer);
}

fn parseVec3(v: std.json.Value) ![3]f32 {
    if (v != .array or v.array.items.len != 3) return error.BadVisionConfig;
    var out: [3]f32 = undefined;
    for (v.array.items, 0..) |x, i| out[i] = switch (x) {
        .float => |f| @floatCast(f),
        .integer => |n| @floatFromInt(n),
        else => return error.BadVisionConfig,
    };
    return out;
}
