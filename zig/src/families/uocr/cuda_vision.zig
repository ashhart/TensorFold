//! Unlimited-OCR-2's image encoder on CUDA: a view's pixels through SAM ViT-B, its neck and CLIP-L to the projector's
//! rows, one view (the 1024 global view or a 640 tile) a pass; vision.zig's steps, cuBLASLt for the projections.
const std = @import("std");
const cuda = @import("cuda");
const cfg = @import("config.zig");
const wts = @import("cuda_weights.zig");
const ks = @import("cuda_kernels.zig");
const resize = @import("resize.zig");

const S = cfg.sam_width;
const C = cfg.clip_width;
const H = cfg.hidden;
const N = cfg.neck_width;
const T = wts.T;

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

pub const Encoder = struct {
    d: *const cuda.Driver,
    s: cuda.Stream,
    k: *const ks.Kernels,
    vk: *const cuda.vit.Kernels,
    gemms: *cuda.vit.Gemms,
    w: *const wts.Weights,
    scratch: Scratch,
    /// A 640 tile's tables: SAM's positions [40, 40, 768], CLIP's [101, 1024], and the global blocks' relative
    /// positions [79, 64] (h then w, block by block).
    tile: cuda.DeviceBuffer,

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, s: cuda.Stream, k: *const ks.Kernels, vk: *const cuda.vit.Kernels, gemms: *cuda.vit.Gemms, w: *const wts.Weights) !Encoder {
        var e: Encoder = .{ .d = d, .s = s, .k = k, .vk = vk, .gemms = gemms, .w = w, .scratch = undefined, .tile = undefined };
        e.scratch = try Scratch.make(d, View.of(cfg.base_size));
        errdefer e.scratch.buf.free();
        e.tile = try tileTables(gpa, d, w);
        return e;
    }

    pub fn deinit(e: *Encoder) void {
        e.tile.free();
        e.scratch.buf.free();
    }

    /// The projector's rows [side^2, 1280] bf16 for one view's normalized pixels [3, size, size] bf16 (size 1024 or 640).
    pub fn encode(e: *Encoder, size: usize, pixels: []const u16, out: []u16) !void {
        if (size != cfg.base_size and size != cfg.tile_size) return error.ViewSize;
        const v = View.of(size);
        if (pixels.len != 3 * size * size or out.len != v.side * v.side * H) return error.ViewRows;
        const sc = e.scratch;
        try e.d.check(e.d.api.cuMemcpyHtoDAsync_v2(sc.at(.pix), pixels.ptr, pixels.len * 2, e.s.handle), "cuMemcpyHtoDAsync");
        try e.sam(v);
        try e.clip(v);
        const n = v.side * v.side;
        try e.k.run("concat", e.s, .{ n, ks.blocks(cfg.fused), 1 }, 256, .{ sc.at(.cx) + C * 2, sc.at(.m3), sc.at(.fused), n, C, cfg.net3_width });
        try e.linearBias(sc.at(.fused), e.w.projector_w, e.w.projector_b, sc.at(.proj_out), n, H, cfg.fused);
        try e.d.check(e.d.api.cuMemcpyDtoHAsync_v2(out.ptr, sc.at(.proj_out), out.len * 2, e.s.handle), "cuMemcpyDtoHAsync");
        try e.s.synchronize();
    }

    fn linear(e: *Encoder, x: u64, w: T, y: u64, rows: usize, n: usize, k: usize) !void {
        try e.gemms.run(e.s, x, w.dev, y, rows, n, k, .bf16);
    }

    fn linearBias(e: *Encoder, x: u64, w: T, b: T, y: u64, rows: usize, n: usize, k: usize) !void {
        try cuda.vit.linear(e.gemms, e.vk, e.s, e.scratch.at(.sum), x, w.dev, b.dev, y, rows, n, k);
    }

    fn add(e: *Encoder, a: u64, b: u64, y: u64, n: usize) !void {
        try e.k.run("add", e.s, .{ ks.blocks(n), 1, 1 }, 256, .{ a, b, y, @as(i64, @intCast(n)) });
    }

    fn norm(e: *Encoder, comptime name: []const u8, x: u64, w: T, b: T, y: u64, rows: usize, width: usize, eps: f32) !void {
        try e.k.run(name, e.s, .{ rows, 1, 1 }, 256, .{ x, w.dev, b.dev, y, width, eps });
    }

    fn im2col(e: *Encoder, x: u64, y: u64, p: [9]usize) !void {
        const width = p[2] * p[3] * p[3];
        try e.k.run("im2col", e.s, .{ p[6] * p[7], ks.blocks(width), 1 }, 256, .{ x, y, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8] });
    }

    fn attention(e: *Encoder, qkv: u64, rel_h: u64, rel_w: u64, o: u64, groups: usize, len: usize, heads: usize, width: usize, side: usize, scale: f32) !void {
        try e.k.run("attention", e.s, .{ groups * ((len + 7) / 8), heads, 1 }, 256, .{ qkv, rel_h, rel_w, o, groups, len, width, side, scale });
    }

    fn sam(e: *Encoder, v: View) !void {
        const sc = e.scratch;
        const g2 = v.grid * v.grid;
        const x = sc.at(.x);
        try e.im2col(sc.at(.pix), sc.at(.cols), .{ v.size, v.size, 3, cfg.patch, cfg.patch, 0, v.grid, v.grid, 1 });
        try e.linearBias(sc.at(.cols), e.w.sam.patch_w, e.w.sam.patch_b, x, g2, S, 3 * cfg.patch * cfg.patch);
        const pos = if (v.size == cfg.base_size) e.w.sam.pos.dev else e.tile.ptr;
        try e.add(x, pos, x, g2 * S);
        var global_index: usize = 0;
        for (e.w.sam.blocks, 0..) |b, i| {
            try e.norm("layer_norm", x, b.norm1_w, b.norm1_b, sc.at(.ln), g2, S, cfg.sam_eps);
            const global = cfg.isGlobal(i);
            const rows = if (global) g2 else v.padded * v.padded;
            const src = if (global) sc.at(.ln) else sc.at(.win);
            if (!global) try e.k.run("window", e.s, .{ rows, ks.blocks(S), 1 }, 256, .{ sc.at(.ln), sc.at(.win), v.grid, v.padded, cfg.sam_window, S });
            try e.linearBias(src, b.qkv_w, b.qkv_b, sc.at(.qkv), rows, 3 * S, S);
            const side: usize = if (global) v.grid else cfg.sam_window;
            const groups: usize = if (global) 1 else (v.padded / cfg.sam_window) * (v.padded / cfg.sam_window);
            var tab_h = b.rel_h.dev;
            var tab_w = b.rel_w.dev;
            if (global and v.size != cfg.base_size) {
                const at = tileRelOffset(global_index);
                tab_h = e.tile.ptr + at;
                tab_w = e.tile.ptr + at + (2 * v.grid - 1) * cfg.sam_head_dim * 2;
            }
            if (global) global_index += 1;
            try e.k.run("rel_pos", e.s, .{ rows, cfg.sam_heads, 1 }, @intCast(std.mem.alignForward(usize, 2 * side, 32)), .{ sc.at(.qkv), tab_h, tab_w, sc.at(.rel_h), sc.at(.rel_w), groups, side, S });
            try e.attention(sc.at(.qkv), sc.at(.rel_h), sc.at(.rel_w), sc.at(.att), groups, side * side, cfg.sam_heads, S, side, 1.0 / @sqrt(@as(f32, cfg.sam_head_dim)));
            try e.linearBias(sc.at(.att), b.proj_w, b.proj_b, sc.at(.proj), rows, S, S);
            if (global) try e.add(x, sc.at(.proj), x, g2 * S) else try e.k.run("unwindow_add", e.s, .{ g2, ks.blocks(S), 1 }, 256, .{ x, sc.at(.proj), v.grid, v.padded, cfg.sam_window, S });
            try e.norm("layer_norm", x, b.norm2_w, b.norm2_b, sc.at(.ln), g2, S, cfg.sam_eps);
            try e.linearBias(sc.at(.ln), b.lin1_w, b.lin1_b, sc.at(.hid), g2, cfg.sam_ffn, S);
            try e.k.run("gelu", e.s, .{ ks.blocks(g2 * cfg.sam_ffn), 1, 1 }, 256, .{ sc.at(.hid), @as(i64, @intCast(g2 * cfg.sam_ffn)) });
            try e.linearBias(sc.at(.hid), b.lin2_w, b.lin2_b, sc.at(.proj), g2, S, cfg.sam_ffn);
            try e.add(x, sc.at(.proj), x, g2 * S);
        }
        // the neck: 1x1 to 256, LayerNorm2d, 3x3 (pad 1), LayerNorm2d; then two 3x3 stride-2 convolutions
        try e.linear(x, e.w.sam.neck0, sc.at(.n0), g2, N, S);
        try e.norm("layer_norm2d", sc.at(.n0), e.w.sam.neck1_w, e.w.sam.neck1_b, sc.at(.n1), g2, N, cfg.sam_eps);
        try e.im2col(sc.at(.n1), sc.at(.cols), .{ v.grid, v.grid, N, 3, 1, 1, v.grid, v.grid, 0 });
        try e.linear(sc.at(.cols), e.w.sam.neck2, sc.at(.n0), g2, N, N * 9);
        try e.norm("layer_norm2d", sc.at(.n0), e.w.sam.neck3_w, e.w.sam.neck3_b, sc.at(.n1), g2, N, cfg.sam_eps);
        const half = v.grid / 2;
        try e.im2col(sc.at(.n1), sc.at(.cols), .{ v.grid, v.grid, N, 3, 2, 1, half, half, 0 });
        try e.linear(sc.at(.cols), e.w.sam.net2, sc.at(.m2), half * half, cfg.net2_width, N * 9);
        try e.im2col(sc.at(.m2), sc.at(.cols), .{ half, half, cfg.net2_width, 3, 2, 1, v.side, v.side, 0 });
        try e.linear(sc.at(.cols), e.w.sam.net3, sc.at(.m3), v.side * v.side, cfg.net3_width, cfg.net2_width * 9);
    }

    fn clip(e: *Encoder, v: View) !void {
        const sc = e.scratch;
        const L = v.side * v.side + 1;
        const x = sc.at(.cx);
        const pos = if (v.size == cfg.base_size) e.w.clip.pos.dev else e.tile.ptr + clipPosOffset();
        try e.k.run("clip_embed", e.s, .{ L, ks.blocks(C), 1 }, 256, .{ e.w.clip.cls.dev, sc.at(.m3), pos, sc.at(.cln), C });
        try e.norm("layer_norm", sc.at(.cln), e.w.clip.pre_w, e.w.clip.pre_b, x, L, C, cfg.clip_eps);
        for (e.w.clip.blocks) |b| {
            try e.norm("layer_norm", x, b.ln1_w, b.ln1_b, sc.at(.cln), L, C, cfg.clip_eps);
            try e.linearBias(sc.at(.cln), b.qkv_w, b.qkv_b, sc.at(.qkv), L, 3 * C, C);
            try e.attention(sc.at(.qkv), sc.at(.rel_h), sc.at(.rel_w), sc.at(.att), 1, L, cfg.clip_heads, C, 0, 1.0 / @sqrt(@as(f32, cfg.clip_head_dim)));
            try e.linearBias(sc.at(.att), b.out_w, b.out_b, sc.at(.proj), L, C, C);
            try e.add(x, sc.at(.proj), x, L * C);
            try e.norm("layer_norm", x, b.ln2_w, b.ln2_b, sc.at(.cln), L, C, cfg.clip_eps);
            try e.linearBias(sc.at(.cln), b.fc1_w, b.fc1_b, sc.at(.hid), L, cfg.clip_ffn, C);
            try e.k.run("quick_gelu", e.s, .{ ks.blocks(L * cfg.clip_ffn), 1, 1 }, 256, .{ sc.at(.hid), @as(i64, @intCast(L * cfg.clip_ffn)) });
            try e.linearBias(sc.at(.hid), b.fc2_w, b.fc2_b, sc.at(.proj), L, C, cfg.clip_ffn);
            try e.add(x, sc.at(.proj), x, L * C);
        }
    }
};

/// A page's image-token rows [tokens, 1280] in the publisher's order (vision.layout's): with tiles, the tiles' rows
/// first (10 per tile row, running across every tile of that row, each followed by the newline row), then the
/// global view's 16 rows of 16 with their newlines, then the separator.
pub fn layout(w: *const wts.Weights, global: []const u16, tiles: []const u16, across: usize, down: usize, out: []u16) void {
    const newline = w.newline.host;
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
    put(out, &at, w.separator.host);
    std.debug.assert(at * H == out.len);
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

/// A 640 tile's position tables and relative positions, resized from the release's on the host as its torch code
/// does, then uploaded.
fn tileTables(gpa: std.mem.Allocator, d: *const cuda.Driver, w: *const wts.Weights) !cuda.DeviceBuffer {
    const total = tileRelOffset(cfg.sam_global.len);
    const host = try gpa.alignedAlloc(u8, .@"16", total);
    defer gpa.free(host);
    @memset(host, 0);
    const base = host.ptr;
    const sam_pos = @as([*]u16, @ptrCast(@alignCast(base)))[0 .. tile_grid * tile_grid * S];
    try resize.bicubicGrid(gpa, w.sam.pos.host, cfg.sam_grid, S, tile_grid, sam_pos);
    const clip_pos = @as([*]u16, @ptrCast(@alignCast(base + clipPosOffset())))[0 .. (cfg.side(cfg.tile_size) * cfg.side(cfg.tile_size) + 1) * C];
    const src_pos = w.clip.pos.host;
    @memcpy(clip_pos[0..C], src_pos[0..C]);
    try resize.bicubicGrid(gpa, src_pos[C..], cfg.side(cfg.base_size), C, cfg.side(cfg.tile_size), clip_pos[C..]);
    for (cfg.sam_global, 0..) |bi, gi| {
        const b = w.sam.blocks[bi];
        const at = tileRelOffset(gi);
        const h_out = @as([*]u16, @ptrCast(@alignCast(base + at)))[0 .. tile_rel * cfg.sam_head_dim];
        const w_out = @as([*]u16, @ptrCast(@alignCast(base + at + tile_rel * cfg.sam_head_dim * 2)))[0 .. tile_rel * cfg.sam_head_dim];
        resize.linearTable(b.rel_h.host, b.rel_len, cfg.sam_head_dim, tile_rel, h_out);
        resize.linearTable(b.rel_w.host, b.rel_len, cfg.sam_head_dim, tile_rel, w_out);
    }
    return cuda.DeviceBuffer.fromHost(d, host);
}

/// One device buffer's parts for the largest view, each 256-aligned.
const Scratch = struct {
    const Part = enum { pix, cols, x, ln, win, qkv, att, proj, hid, sum, rel_h, rel_w, n0, n1, m2, m3, cx, cln, fused, proj_out };
    buf: cuda.DeviceBuffer,
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
            .sum => @max(p2 * 3 * S, g2 * cfg.sam_ffn, (L - 1) * H) * 4,
            .rel_h, .rel_w => cfg.sam_heads * @max(g2 * v.grid, p2 * cfg.sam_window) * 2,
            .n0, .n1 => g2 * N * 2,
            .m2 => (v.grid / 2) * (v.grid / 2) * cfg.net2_width * 2,
            .m3 => v.side * v.side * cfg.net3_width * 2,
            .cx, .cln => L * C * 2,
            .fused => (L - 1) * cfg.fused * 2,
            .proj_out => (L - 1) * H * 2,
        };
    }

    fn make(d: *const cuda.Driver, v: View) !Scratch {
        var s: Scratch = undefined;
        var total: usize = 0;
        for (std.enums.values(Part), 0..) |p, i| {
            s.offs[i] = total;
            total += std.mem.alignForward(usize, bytes(p, v), 256);
        }
        s.buf = try cuda.DeviceBuffer.alloc(d, total);
        return s;
    }

    fn at(s: Scratch, p: Part) u64 {
        return s.buf.ptr + s.offs[@backingInt(p)];
    }
};
