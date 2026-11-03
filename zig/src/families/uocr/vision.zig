//! Unlimited-OCR-2's image encoder on Metal: a view's pixels through SAM ViT-B, its neck and CLIP-L to the projector's
//! rows, one view (the 1024 global view or a 640 tile) a pass.
const std = @import("std");
const mtl = @import("metal");
const cfg = @import("config.zig");
const wts = @import("weights.zig");
const ks = @import("kernels.zig");
const resize = @import("resize.zig");
const Ref = ks.Ref;

const S = cfg.sam_width;
const C = cfg.clip_width;
const H = cfg.hidden;
const N = cfg.neck_width;

/// A view's sizes: its patch grid, the padded window grid, and the feature side after the neck.
const View = struct {
    size: usize,
    grid: usize,
    padded: usize,
    side: usize,

    fn of(size: usize) View {
        const grid = size / cfg.patch;
        return .{ .size = size, .grid = grid, .padded = (std.math.divCeil(usize, grid, cfg.sam_window) catch unreachable) * cfg.sam_window, .side = cfg.side(size) };
    }
};

/// Intermediate rows a check may ask for: SAM's output [side^2, 1024] and CLIP's [side^2 + 1, 1024].
pub const Taps = struct { sam: ?[]u16 = null, clip: ?[]u16 = null };

pub const Encoder = struct {
    gpa: std.mem.Allocator,
    queue: mtl.Queue,
    k: *const ks.Kernels,
    w: *const wts.Weights,
    scratch: Scratch,
    /// A 640 tile's tables: SAM's positions [40, 40, 768], CLIP's [101, 1024], and the global blocks' relative
    /// positions [79, 64] (h then w, block by block).
    tile: mtl.Buffer,

    /// An encoder on `queue` (the caller's, which outlives it).
    pub fn init(gpa: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, k: *const ks.Kernels, w: *const wts.Weights) !Encoder {
        var e: Encoder = .{ .gpa = gpa, .queue = queue, .k = k, .w = w, .scratch = undefined, .tile = undefined };
        e.scratch = try Scratch.make(device, View.of(cfg.base_size));
        errdefer e.scratch.buf.deinit();
        e.tile = try tileTables(gpa, device, w);
        return e;
    }

    pub fn deinit(e: *Encoder) void {
        e.tile.deinit();
        e.scratch.buf.deinit();
    }

    /// The projector's rows [side^2, 1280] bf16 for one view's normalized pixels [3, size, size] bf16 (size 1024 or 640).
    pub fn encode(e: *Encoder, size: usize, pixels: []const u16, out: []u16, taps: Taps) !void {
        if (size != cfg.base_size and size != cfg.tile_size) return error.ViewSize;
        const v = View.of(size);
        if (pixels.len != 3 * size * size or out.len != v.side * v.side * H) return error.ViewRows;
        const s = e.scratch;
        @memcpy(halves(s.at(.pix), pixels.len), pixels);
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = e.queue.commandBuffer();
        const enc = cb.compute(.serial);
        e.sam(enc, v);
        e.clip(enc, v);
        const n = v.side * v.side;
        ks.Kernels.bind(enc, e.k.get("uv_concat"), &.{ s.at(.cx).at(C * 2), s.at(.m3), s.at(.fused) }, [3]u32{ @intCast(n), C, cfg.net3_width });
        enc.dispatchThreads(mtl.Size.of(cfg.fused, n, 1), mtl.Size.of(256, 1, 1));
        e.k.linearBias(enc, s.at(.fused), e.w.projector_w, e.w.projector_b, s.at(.sum), s.at(.proj_out), n, H, cfg.fused);
        enc.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |msg| {
            std.log.err("unlimited ocr: the image encoder failed on the GPU: {s}", .{msg});
            return error.GpuFailure;
        }
        @memcpy(out, halves(s.at(.proj_out), out.len));
        if (taps.sam) |t| @memcpy(t, halves(s.at(.m3), n * cfg.net3_width));
        if (taps.clip) |t| @memcpy(t, halves(s.at(.cx), (n + 1) * C));
    }

    fn sam(e: *Encoder, enc: mtl.ComputeEncoder, v: View) void {
        const s = e.scratch;
        const k = e.k;
        const g2 = v.grid * v.grid;
        const x = s.at(.x);
        im2col(enc, k, s.at(.pix), s.at(.cols), .{ v.size, v.size, 3, cfg.patch, cfg.patch, 0, v.grid, v.grid, 1 });
        k.linearBias(enc, s.at(.cols), e.w.sam.patch_w, e.w.sam.patch_b, s.at(.sum), x, g2, S, 3 * cfg.patch * cfg.patch);
        const pos = if (v.size == cfg.base_size) e.w.sam.pos else Ref{ .buf = e.tile, .off = 0 };
        add(enc, k, x, pos, x, g2 * S);
        var global_index: usize = 0;
        for (e.w.sam.blocks, 0..) |b, i| {
            norm(enc, k.get("uv_layer_norm"), x, b.norm1_w, b.norm1_b, s.at(.ln), g2, S, cfg.sam_eps);
            const global = cfg.isGlobal(i);
            const win: UvWindow = .{ .side = @intCast(v.grid), .padded = @intCast(v.padded), .window = cfg.sam_window, .channels = S };
            const rows = if (global) g2 else v.padded * v.padded;
            const src = if (global) s.at(.ln) else s.at(.win);
            if (!global) {
                ks.Kernels.bind(enc, k.get("uv_window"), &.{ s.at(.ln), s.at(.win) }, win);
                enc.dispatchThreads(mtl.Size.of(S, rows, 1), mtl.Size.of(256, 1, 1));
            }
            k.linearBias(enc, src, b.qkv_w, b.qkv_b, s.at(.sum), s.at(.qkv), rows, 3 * S, S);
            const side: usize = if (global) v.grid else cfg.sam_window;
            const groups: usize = if (global) 1 else (v.padded / cfg.sam_window) * (v.padded / cfg.sam_window);
            var tab_h = b.rel_h;
            var tab_w = b.rel_w;
            if (global and v.size != cfg.base_size) {
                const at = tileRelOffset(global_index);
                tab_h = .{ .buf = e.tile, .off = at };
                tab_w = .{ .buf = e.tile, .off = at + (2 * v.grid - 1) * cfg.sam_head_dim * 2 };
            }
            if (global) global_index += 1;
            const rel: UvRel = .{ .groups = @intCast(groups), .side = @intCast(side), .heads = cfg.sam_heads, .width = S };
            ks.Kernels.bind(enc, k.get("uv_rel_pos"), &.{ s.at(.qkv), tab_h, tab_w, s.at(.rel_h), s.at(.rel_w) }, rel);
            enc.dispatchThreads(mtl.Size.of(2 * side, rows, cfg.sam_heads), mtl.Size.of(32, 8, 1));
            attention(enc, k, s.at(.qkv), s.at(.rel_h), s.at(.rel_w), s.at(.att), .{ .groups = @intCast(groups), .len = @intCast(side * side), .heads = cfg.sam_heads, .width = S, .side = @intCast(side), .scale = 1.0 / @sqrt(@as(f32, cfg.sam_head_dim)) });
            k.linearBias(enc, s.at(.att), b.proj_w, b.proj_b, s.at(.sum), s.at(.proj), rows, S, S);
            if (global) add(enc, k, x, s.at(.proj), x, g2 * S) else {
                ks.Kernels.bind(enc, k.get("uv_unwindow_add"), &.{ x, s.at(.proj) }, win);
                enc.dispatchThreads(mtl.Size.of(S, g2, 1), mtl.Size.of(256, 1, 1));
            }
            norm(enc, k.get("uv_layer_norm"), x, b.norm2_w, b.norm2_b, s.at(.ln), g2, S, cfg.sam_eps);
            k.linearBias(enc, s.at(.ln), b.lin1_w, b.lin1_b, s.at(.sum), s.at(.hid), g2, cfg.sam_ffn, S);
            pointwise(enc, k.get("uv_gelu"), s.at(.hid), g2 * cfg.sam_ffn);
            k.linearBias(enc, s.at(.hid), b.lin2_w, b.lin2_b, s.at(.sum), s.at(.proj), g2, S, cfg.sam_ffn);
            add(enc, k, x, s.at(.proj), x, g2 * S);
        }
        // the neck: 1x1 to 256, LayerNorm2d, 3x3 (pad 1), LayerNorm2d; then two 3x3 stride-2 convolutions
        k.linear(enc, x, e.w.sam.neck0, s.at(.n0), g2, N, S);
        norm(enc, k.get("uv_layer_norm2d"), s.at(.n0), e.w.sam.neck1_w, e.w.sam.neck1_b, s.at(.n1), g2, N, cfg.sam_eps);
        im2col(enc, k, s.at(.n1), s.at(.cols), .{ v.grid, v.grid, N, 3, 1, 1, v.grid, v.grid, 0 });
        k.linear(enc, s.at(.cols), e.w.sam.neck2, s.at(.n0), g2, N, N * 9);
        norm(enc, k.get("uv_layer_norm2d"), s.at(.n0), e.w.sam.neck3_w, e.w.sam.neck3_b, s.at(.n1), g2, N, cfg.sam_eps);
        const half = v.grid / 2;
        im2col(enc, k, s.at(.n1), s.at(.cols), .{ v.grid, v.grid, N, 3, 2, 1, half, half, 0 });
        k.linear(enc, s.at(.cols), e.w.sam.net2, s.at(.m2), half * half, cfg.net2_width, N * 9);
        im2col(enc, k, s.at(.m2), s.at(.cols), .{ half, half, cfg.net2_width, 3, 2, 1, v.side, v.side, 0 });
        k.linear(enc, s.at(.cols), e.w.sam.net3, s.at(.m3), v.side * v.side, cfg.net3_width, cfg.net2_width * 9);
    }

    fn clip(e: *Encoder, enc: mtl.ComputeEncoder, v: View) void {
        const s = e.scratch;
        const k = e.k;
        const L = v.side * v.side + 1;
        const x = s.at(.cx);
        const pos = if (v.size == cfg.base_size) e.w.clip.pos else Ref{ .buf = e.tile, .off = clipPosOffset() };
        ks.Kernels.bind(enc, k.get("uv_clip_embed"), &.{ e.w.clip.cls, s.at(.m3), pos, s.at(.cln) }, @as(u32, C));
        enc.dispatchThreads(mtl.Size.of(C, L, 1), mtl.Size.of(256, 1, 1));
        norm(enc, k.get("uv_layer_norm"), s.at(.cln), e.w.clip.pre_w, e.w.clip.pre_b, x, L, C, cfg.clip_eps);
        for (e.w.clip.blocks) |b| {
            norm(enc, k.get("uv_layer_norm"), x, b.ln1_w, b.ln1_b, s.at(.cln), L, C, cfg.clip_eps);
            k.linearBias(enc, s.at(.cln), b.qkv_w, b.qkv_b, s.at(.sum), s.at(.qkv), L, 3 * C, C);
            attention(enc, k, s.at(.qkv), s.at(.rel_h), s.at(.rel_w), s.at(.att), .{ .groups = 1, .len = @intCast(L), .heads = cfg.clip_heads, .width = C, .side = 0, .scale = 1.0 / @sqrt(@as(f32, cfg.clip_head_dim)) });
            k.linearBias(enc, s.at(.att), b.out_w, b.out_b, s.at(.sum), s.at(.proj), L, C, C);
            add(enc, k, x, s.at(.proj), x, L * C);
            norm(enc, k.get("uv_layer_norm"), x, b.ln2_w, b.ln2_b, s.at(.cln), L, C, cfg.clip_eps);
            k.linearBias(enc, s.at(.cln), b.fc1_w, b.fc1_b, s.at(.sum), s.at(.hid), L, cfg.clip_ffn, C);
            pointwise(enc, k.get("uv_quick_gelu"), s.at(.hid), L * cfg.clip_ffn);
            k.linearBias(enc, s.at(.hid), b.fc2_w, b.fc2_b, s.at(.sum), s.at(.proj), L, C, cfg.clip_ffn);
            add(enc, k, x, s.at(.proj), x, L * C);
        }
    }
};

/// A page's image-token rows [tokens, 1280] in the publisher's order: with tiles, the tiles' rows first (10 per tile
/// row, running across every tile of that row, each followed by the newline row), then the global view's 16 rows of
/// 16 with their newlines, then the separator. `tiles` holds `across` x `down` tiles' 100 rows each, row-major.
pub fn layout(w: *const wts.Weights, global: []const u16, tiles: []const u16, across: usize, down: usize, out: []u16) void {
    const newline = halves(w.newline, H);
    var at: usize = 0;
    const put = struct {
        fn row(o: []u16, i: *usize, r: []const u16) void {
            @memcpy(o[i.* * H ..][0..H], r);
            i.* += 1;
        }
    }.row;
    const t = cfg.side(cfg.tile_size);
    if (across * down > 1) for (0..t * down) |r| {
        for (0..across) |tc| for (0..t) |c| {
            const tile = (r / t) * across + tc;
            put(out, &at, tiles[(tile * t * t + (r % t) * t + c) * H ..][0..H]);
        };
        put(out, &at, newline);
    };
    const g = cfg.side(cfg.base_size);
    for (0..g) |r| {
        for (0..g) |c| put(out, &at, global[(r * g + c) * H ..][0..H]);
        put(out, &at, newline);
    }
    put(out, &at, halves(w.separator, H));
    std.debug.assert(at * H == out.len);
}

const UvWindow = extern struct { side: u32, padded: u32, window: u32, channels: u32 };
const UvRel = extern struct { groups: u32, side: u32, heads: u32, width: u32 };
const UvAttn = extern struct { groups: u32, len: u32, heads: u32, width: u32, side: u32, scale: f32 };

fn im2col(enc: mtl.ComputeEncoder, k: *const ks.Kernels, x: Ref, y: Ref, p: [9]usize) void {
    var q: [9]u32 = undefined;
    for (p, &q) |a, *b| b.* = @intCast(a);
    ks.Kernels.bind(enc, k.get("uv_im2col"), &.{ x, y }, q);
    enc.dispatchThreads(mtl.Size.of(p[2] * p[3] * p[3], p[6] * p[7], 1), mtl.Size.of(256, 1, 1));
}

fn add(enc: mtl.ComputeEncoder, k: *const ks.Kernels, a: Ref, b: Ref, y: Ref, n: usize) void {
    ks.Kernels.bind(enc, k.get("uv_add"), &.{ a, b, y }, @as(u32, @intCast(n)));
    enc.dispatchThreads(mtl.Size.of(n, 1, 1), mtl.Size.of(256, 1, 1));
}

fn pointwise(enc: mtl.ComputeEncoder, p: mtl.Pipeline, x: Ref, n: usize) void {
    ks.Kernels.bind(enc, p, &.{x}, @as(u32, @intCast(n)));
    enc.dispatchThreads(mtl.Size.of(n, 1, 1), mtl.Size.of(256, 1, 1));
}

fn norm(enc: mtl.ComputeEncoder, p: mtl.Pipeline, x: Ref, w: Ref, b: Ref, y: Ref, rows: usize, width: usize, eps: f32) void {
    ks.Kernels.bind(enc, p, &.{ x, w, b, y }, extern struct { width: u32, eps: f32 }{ .width = @intCast(width), .eps = eps });
    enc.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(256, 1, 1));
}

fn attention(enc: mtl.ComputeEncoder, k: *const ks.Kernels, qkv: Ref, rel_h: Ref, rel_w: Ref, o: Ref, p: UvAttn) void {
    ks.Kernels.bind(enc, k.get("uv_attention"), &.{ qkv, rel_h, rel_w, o }, p);
    enc.dispatchGroups(mtl.Size.of(p.groups * p.len, p.heads, 1), mtl.Size.of(32, 1, 1));
}

const tile_grid = cfg.tile_size / cfg.patch;
const tile_rel = 2 * tile_grid - 1;

fn clipPosOffset() usize {
    return std.mem.alignForward(usize, tile_grid * tile_grid * S * 2, 256);
}

fn tileRelOffset(global_index: usize) usize {
    const clip_bytes = std.mem.alignForward(usize, (cfg.side(cfg.tile_size) * cfg.side(cfg.tile_size) + 1) * C * 2, 256);
    return clipPosOffset() + clip_bytes + global_index * 2 * tile_rel * cfg.sam_head_dim * 2;
}

/// A 640 tile's position tables and relative positions, resized from the release's as its torch code does.
fn tileTables(gpa: std.mem.Allocator, device: mtl.Device, w: *const wts.Weights) !mtl.Buffer {
    const total = tileRelOffset(cfg.sam_global.len);
    const buf = try device.buffer(total, mtl.ResourceOptions.shared);
    errdefer buf.deinit();
    const base = buf.contents();
    const sam_pos = @as([*]u16, @ptrCast(@alignCast(base)))[0 .. tile_grid * tile_grid * S];
    try resize.bicubicGrid(gpa, halves(w.sam.pos, cfg.sam_grid * cfg.sam_grid * S), cfg.sam_grid, S, tile_grid, sam_pos);
    const clip_pos = @as([*]u16, @ptrCast(@alignCast(base + clipPosOffset())))[0 .. (cfg.side(cfg.tile_size) * cfg.side(cfg.tile_size) + 1) * C];
    const src_pos = halves(w.clip.pos, cfg.clip_positions * C);
    @memcpy(clip_pos[0..C], src_pos[0..C]);
    try resize.bicubicGrid(gpa, src_pos[C..], cfg.side(cfg.base_size), C, cfg.side(cfg.tile_size), clip_pos[C..]);
    for (cfg.sam_global, 0..) |bi, gi| {
        const b = w.sam.blocks[bi];
        const at = tileRelOffset(gi);
        const h_out = @as([*]u16, @ptrCast(@alignCast(base + at)))[0 .. tile_rel * cfg.sam_head_dim];
        const w_out = @as([*]u16, @ptrCast(@alignCast(base + at + tile_rel * cfg.sam_head_dim * 2)))[0 .. tile_rel * cfg.sam_head_dim];
        resize.linearTable(halves(b.rel_h, b.rel_len * cfg.sam_head_dim), b.rel_len, cfg.sam_head_dim, tile_rel, h_out);
        resize.linearTable(halves(b.rel_w, b.rel_len * cfg.sam_head_dim), b.rel_len, cfg.sam_head_dim, tile_rel, w_out);
    }
    return buf;
}

fn halves(r: Ref, n: usize) []u16 {
    return @as([*]u16, @ptrCast(@alignCast(r.addr())))[0..n];
}

/// One buffer's parts for the largest view, each 256-aligned.
const Scratch = struct {
    const Part = enum { pix, cols, x, ln, win, qkv, att, proj, hid, sum, rel_h, rel_w, n0, n1, m2, m3, cx, cln, fused, proj_out };
    buf: mtl.Buffer,
    offs: [std.enums.values(Part).len]usize,

    fn bytes(p: Part, v: View) usize {
        const g2 = v.grid * v.grid;
        const p2 = v.padded * v.padded;
        const L = v.side * v.side + 1;
        return switch (p) {
            .pix => 3 * v.size * v.size * 2,
            .cols => @max(g2 * N * 9, g2 * 3 * cfg.patch * cfg.patch) * 2,
            .x => g2 * S * 2,
            .ln, .win, .att, .proj => @max(p2 * S, L * C) * 2,
            .qkv => @max(p2 * 3 * S, L * 3 * C) * 2,
            .hid => @max(g2 * cfg.sam_ffn, L * cfg.clip_ffn) * 2,
            .sum => @max(p2 * 3 * S, g2 * cfg.sam_ffn) * 4,
            .rel_h, .rel_w => cfg.sam_heads * @max(g2 * v.grid, p2 * cfg.sam_window) * 2,
            .n0, .n1 => g2 * N * 2,
            .m2 => (v.grid / 2) * (v.grid / 2) * cfg.net2_width * 2,
            .m3 => v.side * v.side * cfg.net3_width * 2,
            .cx, .cln => L * C * 2,
            .fused => (L - 1) * cfg.fused * 2,
            .proj_out => (L - 1) * H * 2,
        };
    }

    fn make(device: mtl.Device, v: View) !Scratch {
        var s: Scratch = undefined;
        var total: usize = 0;
        for (std.enums.values(Part), 0..) |p, i| {
            s.offs[i] = total;
            total += std.mem.alignForward(usize, bytes(p, v), 256);
        }
        s.buf = try device.buffer(total, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        return s;
    }

    fn at(s: Scratch, p: Part) Ref {
        return .{ .buf = s.buf, .off = s.offs[@backingInt(p)] };
    }
};
