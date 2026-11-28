//! Unlimited-OCR-2's image encoder on CUDA: SAM ViT-B, its neck and CLIP-L to the projector, a view a pass.
const std = @import("std");
const cuda = @import("cuda");
const cfg = @import("config.zig");
const wts = @import("cuda_weights.zig");
const ks = @import("cuda_kernels.zig");
const resize = @import("resize.zig");
const steps = ks.vision;

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
    /// A 640 tile's position tables and the global blocks' resized relative positions.
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

    /// The projector's rows [side^2, 1280] for one view's pixels [3, size, size] (size 1024 or 640).
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

    fn attention(e: *Encoder, qkv: u64, rel_h: u64, rel_w: u64, o: u64, groups: usize, len: usize, heads: usize, width: usize, side: usize, scale: f32) !void {
        try steps.attention(e.k, e.s, qkv, rel_h, rel_w, o, .{ .groups = groups, .len = len, .heads = heads, .width = width, .side = side, .scale = scale });
    }

    /// SAM's tower and neck on the view's pixels (`.pix`), its output in `.m3` [side^2, 1024].
    fn sam(e: *Encoder, v: View) !void {
        const sc = e.scratch;
        const k = e.k;
        const g2 = v.grid * v.grid;
        const x = sc.at(.x);
        // the 16 x 16 patches as rows, their convolution, then the position table (resized for a tile)
        try steps.cols(k, e.s, sc.at(.pix), sc.at(.cols), .{ .channels = 3, .in_side = v.size, .out_side = v.grid, .span = cfg.patch, .stride = cfg.patch, .pad = 0, .cstride = v.size * v.size, .pstride = 1 });
        try e.linearBias(sc.at(.cols), e.w.sam.patch_w, e.w.sam.patch_b, x, g2, S, 3 * cfg.patch * cfg.patch);
        try e.add(x, if (v.size == cfg.base_size) e.w.sam.pos.dev else e.tile.ptr, x, g2 * S);
        const scale = 1.0 / @sqrt(@as(f32, cfg.sam_head_dim));
        var global: usize = 0;
        for (e.w.sam.blocks, 0..) |b, i| {
            try e.norm("layer_norm", x, b.norm1_w, b.norm1_b, sc.at(.ln), g2, S, cfg.sam_eps);
            const win: steps.Window = .{ .side = v.grid, .padded = v.padded, .window = cfg.sam_window, .channels = S };
            if (cfg.isGlobal(i)) {
                const rel = if (v.size == cfg.base_size) [2]u64{ b.rel_h.dev, b.rel_w.dev } else relTiles(e.tile, global);
                global += 1;
                try e.linearBias(sc.at(.ln), b.qkv_w, b.qkv_b, sc.at(.qkv), g2, 3 * S, S);
                try steps.relPos(k, e.s, sc.at(.qkv), rel[0], rel[1], sc.at(.rel_h), sc.at(.rel_w), .{ .groups = 1, .side = v.grid, .heads = cfg.sam_heads, .width = S });
                try e.attention(sc.at(.qkv), sc.at(.rel_h), sc.at(.rel_w), sc.at(.att), 1, g2, cfg.sam_heads, S, v.grid, scale);
                try e.linearBias(sc.at(.att), b.proj_w, b.proj_b, sc.at(.proj), g2, S, S);
                try e.add(x, sc.at(.proj), x, g2 * S);
            } else {
                const p2 = v.padded * v.padded;
                const groups = p2 / (cfg.sam_window * cfg.sam_window);
                try steps.window(k, e.s, sc.at(.ln), sc.at(.win), win);
                try e.linearBias(sc.at(.win), b.qkv_w, b.qkv_b, sc.at(.qkv), p2, 3 * S, S);
                try steps.relPos(k, e.s, sc.at(.qkv), b.rel_h.dev, b.rel_w.dev, sc.at(.rel_h), sc.at(.rel_w), .{ .groups = groups, .side = cfg.sam_window, .heads = cfg.sam_heads, .width = S });
                try e.attention(sc.at(.qkv), sc.at(.rel_h), sc.at(.rel_w), sc.at(.att), groups, cfg.sam_window * cfg.sam_window, cfg.sam_heads, S, cfg.sam_window, scale);
                try e.linearBias(sc.at(.att), b.proj_w, b.proj_b, sc.at(.proj), p2, S, S);
                try steps.unwindowAdd(k, e.s, x, sc.at(.proj), win);
            }
            try e.norm("layer_norm", x, b.norm2_w, b.norm2_b, sc.at(.ln), g2, S, cfg.sam_eps);
            try e.linearBias(sc.at(.ln), b.lin1_w, b.lin1_b, sc.at(.hid), g2, cfg.sam_ffn, S);
            try k.run("gelu", e.s, .{ ks.blocks(g2 * cfg.sam_ffn), 1, 1 }, 256, .{ sc.at(.hid), @as(i64, @intCast(g2 * cfg.sam_ffn)) });
            try e.linearBias(sc.at(.hid), b.lin2_w, b.lin2_b, sc.at(.proj), g2, S, cfg.sam_ffn);
            try e.add(x, sc.at(.proj), x, g2 * S);
        }
        // the neck on rows (channels last, so its channel norms are row norms), then the two stride-2 convolutions
        const w = &e.w.sam;
        try e.linear(x, w.neck0, sc.at(.n0), g2, N, S);
        try e.norm("layer_norm", sc.at(.n0), w.neck1_w, w.neck1_b, sc.at(.n1), g2, N, cfg.sam_eps);
        try steps.cols(k, e.s, sc.at(.n1), sc.at(.cols), .{ .channels = N, .in_side = v.grid, .out_side = v.grid, .span = 3, .stride = 1, .pad = 1, .cstride = 1, .pstride = N });
        try e.linear(sc.at(.cols), w.neck2, sc.at(.n0), g2, N, N * 9);
        try e.norm("layer_norm", sc.at(.n0), w.neck3_w, w.neck3_b, sc.at(.n1), g2, N, cfg.sam_eps);
        const half = v.grid / 2;
        try steps.cols(k, e.s, sc.at(.n1), sc.at(.cols), .{ .channels = N, .in_side = v.grid, .out_side = half, .span = 3, .stride = 2, .pad = 1, .cstride = 1, .pstride = N });
        try e.linear(sc.at(.cols), w.net2, sc.at(.m2), half * half, cfg.net2_width, N * 9);
        try steps.cols(k, e.s, sc.at(.m2), sc.at(.cols), .{ .channels = cfg.net2_width, .in_side = half, .out_side = v.side, .span = 3, .stride = 2, .pad = 1, .cstride = 1, .pstride = cfg.net2_width });
        try e.linear(sc.at(.cols), w.net3, sc.at(.m3), v.side * v.side, cfg.net3_width, cfg.net2_width * 9);
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

/// A page's image-token rows in the publisher's order: tile rows with newlines, the global view's, the separator.
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

/// A 640 tile's tables, 256-aligned in `Encoder.tile`: SAM's positions, CLIP's, each global block's offsets h and w.
const tile_grid = cfg.tile_size / cfg.patch;
const tile_rel = 2 * tile_grid - 1;
const tile_clip = cfg.side(cfg.tile_size) * cfg.side(cfg.tile_size) + 1;
const tile_rels = 2 * cfg.sam_global.len;
const tile_rel_parts: [tile_rels]usize = @splat(tile_rel * cfg.sam_head_dim);
const tile_parts = [_]usize{ tile_grid * tile_grid * S, tile_clip * C } ++ tile_rel_parts;

fn tileOffset(part: usize) usize {
    var at: usize = 0;
    for (tile_parts[0..part]) |n| at += std.mem.alignForward(usize, n * 2, 256);
    return at;
}

/// Where the 640 tile's CLIP position table starts in `Encoder.tile`.
fn clipPosOffset() usize {
    return tileOffset(1);
}

/// The 640 tile's relative position tables (h, w) of the `nth` global block.
fn relTiles(tile: cuda.DeviceBuffer, nth: usize) [2]u64 {
    return .{ tile.ptr + tileOffset(2 + 2 * nth), tile.ptr + tileOffset(3 + 2 * nth) };
}

/// A 640 tile's tables (SAM's and CLIP's positions, the global blocks' relative positions) in one buffer.
fn tileTables(gpa: std.mem.Allocator, d: *const cuda.Driver, w: *const wts.Weights) !cuda.DeviceBuffer {
    const host = try gpa.alloc(u16, tileOffset(tile_parts.len) / 2);
    defer gpa.free(host);
    @memset(host, 0);
    const at = struct {
        fn f(h: []u16, part: usize) []u16 {
            return h[tileOffset(part) / 2 ..][0..tile_parts[part]];
        }
    }.f;
    try resize.bicubicGrid(gpa, w.sam.pos.host, cfg.sam_grid, tile_grid, S, at(host, 0));
    // CLIP's class position is kept; its grid (the 1024 view's 16 x 16) shrinks to the tile's 10 x 10
    const clip_grid = cfg.side(cfg.base_size);
    const cpos = w.clip.pos.host;
    const ctile = at(host, 1);
    @memcpy(ctile[0..C], cpos[0..C]);
    try resize.bicubicGrid(gpa, cpos[C..], clip_grid, cfg.side(cfg.tile_size), C, ctile[C..]);
    for (cfg.sam_global, 0..) |bi, n| {
        const b = w.sam.blocks[bi];
        for ([2]T{ b.rel_h, b.rel_w }, 0..) |r, hw| {
            try resize.linearRows(gpa, r.host, b.rel_len, tile_rel, cfg.sam_head_dim, at(host, 2 + 2 * n + hw));
        }
    }
    return cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(host));
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
