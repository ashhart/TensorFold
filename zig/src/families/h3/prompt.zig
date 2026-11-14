//! MiniMax-H3-Image's prompt path on Metal: Qwen3-VL's text model to layer 50, then condition_proj and the refiner.
const std = @import("std");
const mtl = @import("metal");
pub const rules = @import("picture_prompt.zig");
pub const picture = @import("picture.zig");
pub const tensors = @import("tensors.zig");
pub const weights = @import("prompt_weights.zig");
pub const reference = @import("prompt_reference.zig");
pub const kernels = @import("prompt_kernels.zig");

const Ref = tensors.Ref;
const Heads = kernels.Heads;
const D = rules.head_dim;

/// One buffer's parts, each 256-aligned, for a pass over T rows.
fn Scratch(comptime Part: type) type {
    return struct {
        const Self = @This();
        buf: mtl.Buffer,
        offs: [std.enums.values(Part).len]usize,

        fn make(device: mtl.Device, sizes: std.enums.EnumArray(Part, usize)) !Self {
            var s: Self = undefined;
            var total: usize = 0;
            for (std.enums.values(Part), 0..) |p, i| {
                s.offs[i] = total;
                total += std.mem.alignForward(usize, @max(sizes.get(p), 1), 256);
            }
            s.buf = try device.buffer(total, mtl.ResourceOptions.shared);
            return s;
        }

        fn at(s: Self, p: Part) Ref {
            return .{ .buffer = s.buf, .offset = s.offs[@backingInt(p)] };
        }

        fn slice(s: Self, comptime T: type, p: Part, n: usize) []T {
            return @as([*]T, @ptrCast(@alignCast(s.buf.contents() + s.offs[@backingInt(p)])))[0..n];
        }
    };
}

/// A presentation's vision rows: the tower's merged rows at its image pads, and DeepStack's added after layers 0 to 2.
pub const Vision = struct {
    /// The image pads' rows, rising.
    pads: []const u32,
    /// [pads, hidden] bf16 each.
    rows: []const u16,
    deepstack: [3][]const u16,

    /// The pad count, once every row is inside `T` rows of `H` and every block's size agrees.
    fn check(v: Vision, T: usize, H: usize) !usize {
        const n = v.pads.len;
        for (v.pads, 0..) |at, i| if (at >= T or (i > 0 and at <= v.pads[i - 1])) return error.H3BadVisionRows;
        if (v.rows.len != n * H) return error.H3BadVisionRows;
        for (v.deepstack) |d| if (d.len != n * H) return error.H3BadVisionRows;
        return n;
    }
};

pub const Runner = struct {
    device: mtl.Device,
    queue: mtl.Queue,
    k: kernels.Kernels,
    /// GPU seconds of the last encode and refine.
    gpu: f64 = 0,

    pub fn init(device: mtl.Device) !Runner {
        const queue = try device.queue();
        errdefer queue.deinit();
        return .{ .device = device, .queue = queue, .k = try kernels.Kernels.load(device) };
    }

    pub fn deinit(r: *Runner) void {
        r.k.deinit();
        r.queue.deinit();
    }

    /// Command buffers committed in order, each a few layers' work; `finish` waits for the last and checks them all.
    const Passes = struct {
        list: [80]mtl.CommandBuffer = undefined,
        n: usize = 0,

        fn next(p: *Passes, q: mtl.Queue) !mtl.ComputeEncoder {
            if (p.n == p.list.len) return error.H3TooManyPasses;
            p.list[p.n] = q.commandBuffer();
            p.n += 1;
            return p.list[p.n - 1].compute(.serial);
        }

        fn commit(p: *Passes, e: mtl.ComputeEncoder) void {
            e.end();
            p.list[p.n - 1].commit();
        }

        fn finish(p: *Passes) !f64 {
            var gpu: f64 = 0;
            for (p.list[0..p.n]) |cb| {
                cb.wait();
                if (cb.failure()) |msg| {
                    std.log.err("h3 prompt: a pass failed on the GPU: {s}", .{msg});
                    return error.GpuFailed;
                }
                gpu += cb.gpuSeconds();
            }
            return gpu;
        }
    };

    const EncodePart = enum { x, xn, y, q, k, v, att, gate, up, act, cos, sin, pads, deep };

    /// The text model's rows leaving its last run layer [T, hidden] bf16 for `ids` at M-RoPE positions `pos`.
    pub fn encode(r: *Runner, w: *const weights.Text, ids: []const u32, pos: []const [3]i32, vision: ?Vision, out: []u16) !void {
        const g = w.g;
        const T = ids.len;
        const H = g.hidden;
        const qh: Heads = .{ .count = g.heads, .row = g.heads * D, .head = D };
        const kh: Heads = .{ .count = g.kv_heads, .row = g.kv_heads * D, .head = D };
        if (T == 0 or pos.len != T or out.len != T * H) return error.H3PromptRows;
        var sizes = std.enums.EnumArray(EncodePart, usize).initFill(T * H * 2);
        inline for (.{ .{ .q, qh.row }, .{ .att, qh.row }, .{ .k, kh.row }, .{ .v, kh.row }, .{ .gate, g.ffn }, .{ .up, g.ffn }, .{ .act, g.ffn } }) |p| sizes.set(p[0], T * p[1] * 2);
        sizes.set(.cos, T * 64 * 4);
        sizes.set(.sin, T * 64 * 4);
        const n = if (vision) |v| try v.check(T, H) else 0;
        sizes.set(.pads, n * 4);
        sizes.set(.deep, 3 * n * H * 2);
        const s = try Scratch(EncodePart).make(r.device, sizes);
        defer s.buf.deinit();
        rules.rotary(s.slice(f32, .cos, T * 64), s.slice(f32, .sin, T * 64), pos, g.theta, g.sections);
        const x = s.slice(u8, .x, T * H * 2);
        for (ids, 0..) |id, i| @memcpy(x[i * H * 2 ..][0 .. H * 2], try w.row(id));
        if (vision) |v| {
            for (v.pads, 0..) |at, i| @memcpy(s.slice(u16, .x, T * H)[at * H ..][0..H], v.rows[i * H ..][0..H]);
            @memcpy(s.slice(u32, .pads, n), v.pads);
            for (v.deepstack, 0..) |d, i| @memcpy(s.slice(u16, .deep, 3 * n * H)[i * n * H ..][0 .. n * H], d);
        }
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        var passes: Passes = .{};
        for (w.layers, 0..) |l, li| {
            const e = try passes.next(r.queue);
            const k = &r.k;
            k.norm(e, s.at(.x), l.norm1, s.at(.xn), T, H, g.eps, .qwen);
            k.gemm(e, s.at(.xn), l.q, null, s.at(.q), T, qh.row, H);
            k.gemm(e, s.at(.xn), l.k, null, s.at(.k), T, kh.row, H);
            k.gemm(e, s.at(.xn), l.v, null, s.at(.v), T, kh.row, H);
            k.heads(e, s.at(.q), qh, s.at(.k), kh, l.q_norm, l.k_norm, .{ s.at(.cos), s.at(.sin) }, T, g.eps, .qwen);
            k.attention(e, s.at(.q), qh, s.at(.k), s.at(.v), kh, s.at(.att), T, true);
            k.gemm(e, s.at(.att), l.o, null, s.at(.y), T, H, qh.row);
            k.add(e, s.at(.x), s.at(.y), T * H);
            k.norm(e, s.at(.x), l.norm2, s.at(.xn), T, H, g.eps, .qwen);
            k.gemm(e, s.at(.xn), l.gate, null, s.at(.gate), T, g.ffn, H);
            k.gemm(e, s.at(.xn), l.up, null, s.at(.up), T, g.ffn, H);
            k.glu(e, s.at(.gate), s.at(.up), s.at(.act), T, g.ffn, g.ffn, g.ffn, .qwen);
            k.gemm(e, s.at(.act), l.down, null, s.at(.y), T, H, g.ffn);
            k.add(e, s.at(.x), s.at(.y), T * H);
            if (n > 0 and li < 3) k.scatterAdd(e, s.at(.x), .{ .buffer = s.buf, .offset = s.at(.deep).offset + li * n * H * 2 }, s.at(.pads), n, H);
            passes.commit(e);
        }
        r.gpu = try passes.finish();
        @memcpy(out, s.slice(u16, .x, T * H));
    }

    const RefinePart = enum { in, x, xn, y, qkv, att, wide, act, out };

    /// condition_proj, the refiner's blocks over every row as one document, its final norm: [T, text] to [T, hidden].
    pub fn refine(r: *Runner, w: *const weights.Refiner, rows: []const u16, out: []u16) !void {
        const g = w.g;
        const H = g.hidden;
        const inner = g.heads * D;
        const T = rows.len / g.text;
        if (T == 0 or rows.len != T * g.text or out.len != T * H) return error.H3PromptRows;
        var sizes = std.enums.EnumArray(RefinePart, usize).initFill(T * H * 2);
        inline for (.{ .{ .in, g.text }, .{ .qkv, 3 * inner }, .{ .att, inner }, .{ .wide, 2 * g.mlp }, .{ .act, g.mlp } }) |p| sizes.set(p[0], T * p[1] * 2);
        const s = try Scratch(RefinePart).make(r.device, sizes);
        defer s.buf.deinit();
        @memcpy(s.slice(u16, .in, rows.len), rows);
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        var passes: Passes = .{};
        const k = &r.k;
        var e = try passes.next(r.queue);
        k.gemm(e, s.at(.in), w.cond_w, w.cond_b, s.at(.x), T, H, g.text);
        // the fused rows hold each head's q, k and v together: 384 values a head, q at 0, k at 128, v at 256
        const fused: Heads = .{ .count = g.heads, .row = 3 * inner, .head = 3 * D };
        const qkv = s.at(.qkv);
        const kr: Ref = .{ .buffer = qkv.buffer, .offset = qkv.offset + D * 2 };
        const vr: Ref = .{ .buffer = qkv.buffer, .offset = qkv.offset + 2 * D * 2 };
        for (w.blocks) |b| {
            k.norm(e, s.at(.x), b.norm1, s.at(.xn), T, H, g.eps, .h3);
            k.gemm(e, s.at(.xn), b.qkv, null, qkv, T, 3 * inner, H);
            k.heads(e, qkv, fused, kr, fused, b.q_norm, b.k_norm, null, T, g.qk_eps, .h3);
            k.attention(e, qkv, fused, kr, vr, fused, s.at(.att), T, false);
            k.gemm(e, s.at(.att), b.out, null, s.at(.y), T, H, inner);
            k.add(e, s.at(.x), s.at(.y), T * H);
            k.norm(e, s.at(.x), b.norm2, s.at(.xn), T, H, g.eps, .h3);
            k.gemm(e, s.at(.xn), b.fc1, null, s.at(.wide), T, 2 * g.mlp, H);
            const value: Ref = .{ .buffer = s.buf, .offset = s.at(.wide).offset + g.mlp * 2 };
            k.glu(e, s.at(.wide), value, s.at(.act), T, g.mlp, 2 * g.mlp, 2 * g.mlp, .h3);
            k.gemm(e, s.at(.act), b.fc2, null, s.at(.y), T, H, g.mlp);
            k.add(e, s.at(.x), s.at(.y), T * H);
            passes.commit(e);
            e = try passes.next(r.queue);
        }
        k.norm(e, s.at(.x), w.final_norm, s.at(.out), T, H, g.final_eps, .h3);
        passes.commit(e);
        r.gpu = try passes.finish();
        @memcpy(out, s.slice(u16, .out, T * H));
    }
};
