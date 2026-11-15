//! Unlimited-OCR-2's image encoder on Metal: SAM ViT-B, its neck and CLIP-L to the projector's rows, a view a pass.
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
    /// A 640 tile's tables: SAM's positions, CLIP's, and the global blocks' relative positions (h then w).
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

    /// The projector's rows [side^2, 1280] bf16 for one view's normalized pixels [3, size, size] bf16.
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
        e.project(enc, v);
        const n = v.side * v.side;
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

    /// One piece of `encode`: a SAM block, the neck, CLIP's front, a CLIP layer, or the projector.
    pub const Step = union(enum) { sam: usize, neck, front, clip: usize, project };

    /// A step's bf16 input rows: SAM's x [grid^2, 768], its output m3 [side^2, 1024], CLIP's cx [side^2 + 1, 1024].
    pub const Rows = struct { x: ?[]const u16 = null, m3: ?[]const u16 = null, cx: ?[]const u16 = null };

    /// `st` alone for a view of `size` on `in` (the rows it reads), its rows into `out`: the code `encode` runs.
    pub fn step(e: *Encoder, size: usize, st: Step, in: Rows, out: []u16) !void {
        if (size != cfg.base_size and size != cfg.tile_size) return error.ViewSize;
        const v = View.of(size);
        const s = e.scratch;
        const want: struct { x: bool, m3: bool, cx: bool }, const part: Scratch.Part = switch (st) {
            .sam => .{ .{ .x = true, .m3 = false, .cx = false }, .x },
            .neck => .{ .{ .x = true, .m3 = false, .cx = false }, .m3 },
            .front => .{ .{ .x = false, .m3 = true, .cx = false }, .cx },
            .clip => .{ .{ .x = false, .m3 = false, .cx = true }, .cx },
            .project => .{ .{ .x = false, .m3 = true, .cx = true }, .proj_out },
        };
        inline for (.{ .{ "x", Scratch.Part.x }, .{ "m3", Scratch.Part.m3 }, .{ "cx", Scratch.Part.cx } }) |p| {
            if (@field(want, p[0])) {
                const rows = @field(in, p[0]) orelse return error.ViewRows;
                if (rows.len != rowsOf(p[1], v)) return error.ViewRows;
                @memcpy(halves(s.at(p[1]), rows.len), rows);
            }
        }
        if (out.len != rowsOf(part, v)) return error.ViewRows;
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = e.queue.commandBuffer();
        const enc = cb.compute(.serial);
        switch (st) {
            .sam => |i| e.samBlock(enc, v, i),
            .neck => e.neck(enc, v),
            .front => e.clipFront(enc, v),
            .clip => |i| e.clipLayer(enc, v, i),
            .project => e.project(enc, v),
        }
        enc.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |msg| {
            std.log.err("unlimited ocr: the image encoder failed on the GPU: {s}", .{msg});
            return error.GpuFailure;
        }
        @memcpy(out, halves(s.at(part), out.len));
    }

    /// The bf16 values a step reads or leaves in `part` for a view.
    fn rowsOf(part: Scratch.Part, v: View) usize {
        const n = v.side * v.side;
        return switch (part) {
            .x => v.grid * v.grid * S,
            .m3 => n * cfg.net3_width,
            .cx => (n + 1) * C,
            .proj_out => n * H,
            else => unreachable,
        };
    }

    /// The projector's rows from CLIP's (`.cx`, its class row left out) beside SAM's (`.m3`), into `.proj_out`.
    fn project(e: *Encoder, enc: mtl.ComputeEncoder, v: View) void {
        const s = e.scratch;
        const n = v.side * v.side;
        ks.Kernels.bind(enc, e.k.get("uv_concat"), &.{ s.at(.cx).at(C * 2), s.at(.m3), s.at(.fused) }, [3]u32{ @intCast(n), C, cfg.net3_width });
        enc.dispatchThreads(mtl.Size.of(cfg.fused, n, 1), mtl.Size.of(256, 1, 1));
        e.k.linearBias(enc, s.at(.fused), e.w.projector_w, e.w.projector_b, s.at(.sum), s.at(.proj_out), n, H, cfg.fused);
    }

    /// SAM's tower and neck on the view's pixels (`.pix`), its output in `.m3` [side^2, 1024].
    fn sam(e: *Encoder, enc: mtl.ComputeEncoder, v: View) void {
        const s = e.scratch;
        const k = e.k;
        const g2 = v.grid * v.grid;
        const x = s.at(.x);
        // the 16 x 16 patches as rows, their convolution, then the position table (resized for a tile)
        cols(enc, k, s.at(.pix), s.at(.cols), .{ .channels = 3, .in_side = @intCast(v.size), .out_side = @intCast(v.grid), .span = cfg.patch, .stride = cfg.patch, .pad = 0, .cstride = @intCast(v.size * v.size), .pstride = 1 });
        k.linearBias(enc, s.at(.cols), e.w.sam.patch_w, e.w.sam.patch_b, s.at(.sum), x, g2, S, 3 * cfg.patch * cfg.patch);
        add(enc, k, x, if (v.size == cfg.base_size) e.w.sam.pos else Ref{ .buf = e.tile, .off = 0 }, x, g2 * S);
        for (0..e.w.sam.blocks.len) |i| e.samBlock(enc, v, i);
        e.neck(enc, v);
    }

    /// The neck on `.x` (channels last: its channel norms are row norms), then two stride-2 convolutions, into `.m3`.
    fn neck(e: *Encoder, enc: mtl.ComputeEncoder, v: View) void {
        const s = e.scratch;
        const k = e.k;
        const g2 = v.grid * v.grid;
        const x = s.at(.x);
        const w = &e.w.sam;
        k.linear(enc, x, w.neck0, s.at(.n0), g2, N, S);
        norm(enc, k.get("uv_layer_norm"), s.at(.n0), w.neck1_w, w.neck1_b, s.at(.n1), g2, N, cfg.sam_eps);
        cols(enc, k, s.at(.n1), s.at(.cols), .{ .channels = N, .in_side = @intCast(v.grid), .out_side = @intCast(v.grid), .span = 3, .stride = 1, .pad = 1, .cstride = 1, .pstride = N });
        k.linear(enc, s.at(.cols), w.neck2, s.at(.n0), g2, N, N * 9);
        norm(enc, k.get("uv_layer_norm"), s.at(.n0), w.neck3_w, w.neck3_b, s.at(.n1), g2, N, cfg.sam_eps);
        const half = v.grid / 2;
        cols(enc, k, s.at(.n1), s.at(.cols), .{ .channels = N, .in_side = @intCast(v.grid), .out_side = @intCast(half), .span = 3, .stride = 2, .pad = 1, .cstride = 1, .pstride = N });
        k.linear(enc, s.at(.cols), w.net2, s.at(.m2), half * half, cfg.net2_width, N * 9);
        cols(enc, k, s.at(.m2), s.at(.cols), .{ .channels = cfg.net2_width, .in_side = @intCast(half), .out_side = @intCast(v.side), .span = 3, .stride = 2, .pad = 1, .cstride = 1, .pstride = cfg.net2_width });
        k.linear(enc, s.at(.cols), w.net3, s.at(.m3), v.side * v.side, cfg.net3_width, cfg.net2_width * 9);
    }

    /// SAM block `i` on `.x` [grid^2, 768], in place.
    fn samBlock(e: *Encoder, enc: mtl.ComputeEncoder, v: View, i: usize) void {
        const s = e.scratch;
        const k = e.k;
        const b = e.w.sam.blocks[i];
        const g2 = v.grid * v.grid;
        const x = s.at(.x);
        norm(enc, k.get("uv_layer_norm"), x, b.norm1_w, b.norm1_b, s.at(.ln), g2, S, cfg.sam_eps);
        const win: UvWindow = .{ .side = @intCast(v.grid), .padded = @intCast(v.padded), .window = cfg.sam_window, .channels = S };
        const scale = 1.0 / @sqrt(@as(f32, cfg.sam_head_dim));
        if (std.mem.indexOfScalar(usize, &cfg.sam_global, i)) |global| {
            const rel = if (v.size == cfg.base_size) [2]Ref{ b.rel_h, b.rel_w } else relTiles(e.tile, global);
            k.linearBias(enc, s.at(.ln), b.qkv_w, b.qkv_b, s.at(.sum), s.at(.qkv), g2, 3 * S, S);
            relPos(enc, k, s.at(.qkv), rel, s.at(.rel_h), s.at(.rel_w), .{ .groups = 1, .side = @intCast(v.grid), .heads = cfg.sam_heads, .width = S });
            attention(enc, k, s.at(.qkv), s.at(.rel_h), s.at(.rel_w), s.at(.att), .{ .groups = 1, .len = @intCast(g2), .heads = cfg.sam_heads, .width = S, .side = @intCast(v.grid), .scale = scale });
            k.linearBias(enc, s.at(.att), b.proj_w, b.proj_b, s.at(.sum), s.at(.proj), g2, S, S);
            add(enc, k, x, s.at(.proj), x, g2 * S);
        } else {
            const p2 = v.padded * v.padded;
            const groups = p2 / (cfg.sam_window * cfg.sam_window);
            ks.Kernels.bind(enc, k.get("uv_window"), &.{ s.at(.ln), s.at(.win) }, win);
            enc.dispatchThreads(mtl.Size.of(S, p2, 1), mtl.Size.of(256, 1, 1));
            k.linearBias(enc, s.at(.win), b.qkv_w, b.qkv_b, s.at(.sum), s.at(.qkv), p2, 3 * S, S);
            relPos(enc, k, s.at(.qkv), .{ b.rel_h, b.rel_w }, s.at(.rel_h), s.at(.rel_w), .{ .groups = @intCast(groups), .side = cfg.sam_window, .heads = cfg.sam_heads, .width = S });
            attention(enc, k, s.at(.qkv), s.at(.rel_h), s.at(.rel_w), s.at(.att), .{ .groups = @intCast(groups), .len = cfg.sam_window * cfg.sam_window, .heads = cfg.sam_heads, .width = S, .side = cfg.sam_window, .scale = scale });
            k.linearBias(enc, s.at(.att), b.proj_w, b.proj_b, s.at(.sum), s.at(.proj), p2, S, S);
            ks.Kernels.bind(enc, k.get("uv_unwindow_add"), &.{ x, s.at(.proj) }, win);
            enc.dispatchThreads(mtl.Size.of(S, g2, 1), mtl.Size.of(256, 1, 1));
        }
        norm(enc, k.get("uv_layer_norm"), x, b.norm2_w, b.norm2_b, s.at(.ln), g2, S, cfg.sam_eps);
        k.linearBias(enc, s.at(.ln), b.lin1_w, b.lin1_b, s.at(.sum), s.at(.hid), g2, cfg.sam_ffn, S);
        pointwise(enc, k.get("uv_gelu"), s.at(.hid), g2 * cfg.sam_ffn);
        k.linearBias(enc, s.at(.hid), b.lin2_w, b.lin2_b, s.at(.sum), s.at(.proj), g2, S, cfg.sam_ffn);
        add(enc, k, x, s.at(.proj), x, g2 * S);
    }

    fn clip(e: *Encoder, enc: mtl.ComputeEncoder, v: View) void {
        e.clipFront(enc, v);
        for (0..e.w.clip.blocks.len) |i| e.clipLayer(enc, v, i);
    }

    /// CLIP's rows from SAM's (`.m3`): the class row, the positions, then the pre-norm, into `.cx`.
    fn clipFront(e: *Encoder, enc: mtl.ComputeEncoder, v: View) void {
        const s = e.scratch;
        const k = e.k;
        const L = v.side * v.side + 1;
        const pos = if (v.size == cfg.base_size) e.w.clip.pos else Ref{ .buf = e.tile, .off = clipPosOffset() };
        ks.Kernels.bind(enc, k.get("uv_clip_embed"), &.{ e.w.clip.cls, s.at(.m3), pos, s.at(.cln) }, @as(u32, C));
        enc.dispatchThreads(mtl.Size.of(C, L, 1), mtl.Size.of(256, 1, 1));
        norm(enc, k.get("uv_layer_norm"), s.at(.cln), e.w.clip.pre_w, e.w.clip.pre_b, s.at(.cx), L, C, cfg.clip_eps);
    }

    /// CLIP layer `i` on `.cx` [side^2 + 1, 1024], in place.
    fn clipLayer(e: *Encoder, enc: mtl.ComputeEncoder, v: View, i: usize) void {
        const s = e.scratch;
        const k = e.k;
        const b = e.w.clip.blocks[i];
        const L = v.side * v.side + 1;
        const x = s.at(.cx);
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
};

/// A page's image-token rows in the publisher's order: tile rows with newlines, the global view's, the separator.
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
const UvCols = extern struct { channels: u32, in_side: u32, out_side: u32, span: u32, stride: u32, pad: u32, cstride: u32, pstride: u32 };

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

/// A convolution's input rows [out_side^2, channels x span^2] from a square map (see uv_im2col).
fn cols(enc: mtl.ComputeEncoder, k: *const ks.Kernels, x: Ref, y: Ref, p: UvCols) void {
    ks.Kernels.bind(enc, k.get("uv_im2col"), &.{ x, y }, p);
    enc.dispatchThreads(mtl.Size.of(p.channels * p.span * p.span, p.out_side * p.out_side, 1), mtl.Size.of(256, 1, 1));
}

/// The relative-position terms of every query row against each key row and column, bf16 [heads, rows, side] each.
fn relPos(enc: mtl.ComputeEncoder, k: *const ks.Kernels, qkv: Ref, table: [2]Ref, rel_h: Ref, rel_w: Ref, p: UvRel) void {
    ks.Kernels.bind(enc, k.get("uv_rel_pos"), &.{ qkv, table[0], table[1], rel_h, rel_w }, p);
    enc.dispatchThreads(mtl.Size.of(2 * p.side, p.groups * p.side * p.side, p.heads), mtl.Size.of(32, 8, 1));
}

const UvAttn = extern struct { groups: u32, len: u32, heads: u32, width: u32, side: u32, scale: f32 };

/// Attention within groups of `len` rows of qkv, heads of 64, into o; side > 0 adds relPos's terms to the scores.
fn attention(enc: mtl.ComputeEncoder, k: *const ks.Kernels, qkv: Ref, rel_h: Ref, rel_w: Ref, o: Ref, p: UvAttn) void {
    std.debug.assert(p.len <= 4096); // uv_attention keeps 8 blocks of 512 keys a row
    ks.Kernels.bind(enc, k.get("uv_attention"), &.{ qkv, rel_h, rel_w, o }, p);
    enc.dispatchGroups(mtl.Size.of((p.len + 31) / 32, p.heads, p.groups), mtl.Size.of(128, 1, 1));
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
fn relTiles(tile: mtl.Buffer, nth: usize) [2]Ref {
    return .{ .{ .buf = tile, .off = tileOffset(2 + 2 * nth) }, .{ .buf = tile, .off = tileOffset(3 + 2 * nth) } };
}

/// A 640 tile's tables (SAM's and CLIP's positions, the global blocks' relative positions) in one shared buffer.
fn tileTables(gpa: std.mem.Allocator, device: mtl.Device, w: *const wts.Weights) !mtl.Buffer {
    const buf = try device.buffer(tileOffset(tile_parts.len), mtl.ResourceOptions.shared);
    errdefer buf.deinit();
    const at = struct {
        fn f(b: mtl.Buffer, part: usize) []u16 {
            return halves(.{ .buf = b, .off = tileOffset(part) }, tile_parts[part]);
        }
    }.f;
    try resize.bicubicGrid(gpa, halves(w.sam.pos, cfg.sam_grid * cfg.sam_grid * S), cfg.sam_grid, tile_grid, S, at(buf, 0));
    // CLIP's class position is kept; its grid (the 1024 view's 16 x 16) shrinks to the tile's 10 x 10
    const clip_grid = cfg.side(cfg.base_size);
    const cpos = halves(w.clip.pos, cfg.clip_positions * C);
    const ctile = at(buf, 1);
    @memcpy(ctile[0..C], cpos[0..C]);
    try resize.bicubicGrid(gpa, cpos[C..], clip_grid, cfg.side(cfg.tile_size), C, ctile[C..]);
    for (cfg.sam_global, 0..) |bi, n| {
        const b = w.sam.blocks[bi];
        for ([2]Ref{ b.rel_h, b.rel_w }, 0..) |r, hw| {
            try resize.linearRows(gpa, halves(r, b.rel_len * cfg.sam_head_dim), b.rel_len, tile_rel, cfg.sam_head_dim, at(buf, 2 + 2 * n + hw));
        }
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
