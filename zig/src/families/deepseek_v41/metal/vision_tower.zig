//! One image through the ViT and aligner: patches in, the aligner's rows (bf16 [tokens][out_dim]) out.
const std = @import("std");
const mtl = @import("metal");
const vision = @import("vision.zig");
const vw = @import("vision_weights.zig");
const glue = @import("glue.zig");

const Ref = vision.Ref;

/// What a run covers: the first `blocks` ViT blocks (null: all), then the final norm and aligner (`align`).
pub const Run = struct { blocks: ?u32 = null, finish: bool = true };

pub const Tower = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kit: *const vision.Kit,
    w: *const vw.Weights,
    /// scratch for the largest image so far (grown, never shrunk)
    buf: ?mtl.Buffer = null,
    cap_n: u32 = 0,
    cap_t: u32 = 0,
    cs: ?mtl.Buffer = null,
    table: std.ArrayList([2]f32) = .empty,
    /// patches in, the residual stream and the aligner's output of the last run
    refs: Refs = undefined,

    const Refs = struct { pin: Ref, x: Ref, xn: Ref, acc: Ref, qkv: Ref, o: Ref, h: Ref, u: Ref, t: Ref, out: Ref };

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kit: *const vision.Kit, w: *const vw.Weights) Tower {
        return .{ .a = a, .device = device, .queue = queue, .kit = kit, .w = w };
    }

    pub fn deinit(t: *Tower) void {
        if (t.buf) |b| b.deinit();
        if (t.cs) |b| b.deinit();
        t.table.deinit(t.a);
    }

    /// The scratch's places for `n` patches and `tokens` aligner rows; the total bytes.
    fn layout(t: *Tower, n: u32, tokens: u32) usize {
        const c = t.w.cfg;
        var off: usize = 0;
        const take = struct {
            fn f(o: *usize, len: usize) usize {
                const at = std.mem.alignForward(usize, o.*, 256);
                o.* = at + @max(len, 4);
                return at;
            }
        }.f;
        const N: usize = n;
        const L: usize = tokens;
        const at = [10]usize{
            take(&off, N * c.patchValues() * 2),
            take(&off, N * c.dim * 2),
            take(&off, N * c.dim * 2),
            take(&off, @max(N * 2 * c.inter, @max(N * 3 * c.dim, L * c.out_dim)) * 4),
            take(&off, N * 3 * c.dim * 2),
            take(&off, N * c.dim * 2),
            take(&off, N * c.inter * 2),
            take(&off, L * c.alignIn() * 2),
            take(&off, L * c.out_dim * 2),
            take(&off, L * c.out_dim * 2),
        };
        if (t.buf) |b| {
            const f = struct {
                fn r(bb: mtl.Buffer, o: usize) Ref {
                    return .{ .buf = bb, .off = o };
                }
            }.r;
            t.refs = .{ .pin = f(b, at[0]), .x = f(b, at[1]), .xn = f(b, at[2]), .acc = f(b, at[3]), .qkv = f(b, at[4]), .o = f(b, at[5]), .h = f(b, at[6]), .u = f(b, at[7]), .t = f(b, at[8]), .out = f(b, at[9]) };
        }
        return off;
    }

    /// The scratch for `n` patches and `tokens` aligner rows, regrown when this run is larger than any before.
    fn ensure(t: *Tower, n: u32, tokens: u32) !void {
        if (n <= t.cap_n and tokens <= t.cap_t) return;
        if (t.buf) |b| b.deinit();
        t.buf = null;
        t.cap_n = @max(n, t.cap_n);
        t.cap_t = @max(tokens, t.cap_t);
        t.buf = try t.device.buffer(t.layout(t.cap_n, t.cap_t), mtl.ResourceOptions.shared);
        _ = t.layout(t.cap_n, t.cap_t);
    }

    pub fn tokensOf(t: *const Tower, gh: u32, gw: u32) u32 {
        const r = t.w.cfg.ratio;
        return ((gh + r - 1) / r) * ((gw + r - 1) / r);
    }

    /// The aligner's rows of the last `run` (bf16 [tokens][out_dim]); the residual stream after the last block is `x`.
    pub fn output(t: *const Tower) Ref {
        return t.refs.out;
    }

    pub fn stream(t: *const Tower) Ref {
        return t.refs.x;
    }

    /// `patches` [gh * gw][3 * patch^2] (values bf16 holds exactly) through the tower into `output()`.
    pub fn run(t: *Tower, patches: []const f32, gh: u32, gw: u32, o: Run) !void {
        const c = t.w.cfg;
        const n = gh * gw;
        if (n == 0 or patches.len != @as(usize, n) * c.patchValues()) return error.BadPatches;
        const tokens = t.tokensOf(gh, gw);
        try t.ensure(n, tokens);
        const r = t.refs;
        const pin: []u16 = @alignCast(std.mem.bytesAsSlice(u16, (r.pin.buf.contents() + r.pin.off)[0 .. patches.len * 2]));
        for (pin, patches) |*d, v| d.* = bf16(v);
        try t.cosSin(gh, gw);
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = t.queue.commandBuffer();
        const e = cb.compute(.serial);
        t.encode(e, gh, gw, tokens, o);
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |msg| {
            std.debug.print("dsv41 vision command buffer failed: {s}\n", .{msg});
            return error.CommandBufferFailed;
        }
    }

    fn cosSin(t: *Tower, gh: u32, gw: u32) !void {
        const half = t.w.cfg.headDim() / 2;
        try t.table.resize(t.a, @as(usize, gh) * gw * half);
        vision.ropeTable(t.table.items, gh, gw, t.w.cfg.headDim(), t.w.cfg.theta);
        const bytes = std.mem.sliceAsBytes(t.table.items);
        if (t.cs == null or t.cs.?.length() < bytes.len) {
            if (t.cs) |b| b.deinit();
            t.cs = try t.device.buffer(bytes.len, mtl.ResourceOptions.shared);
        }
        @memcpy(t.cs.?.contents()[0..bytes.len], bytes);
    }

    fn encode(t: *Tower, e: mtl.ComputeEncoder, gh: u32, gw: u32, tokens: u32, o: Run) void {
        const c = t.w.cfg;
        const w = t.w;
        const v = &t.kit.v;
        const gp = &t.kit.g;
        const r = t.refs;
        const n = gh * gw;
        const D = c.dim;
        glue.mv(gp, e, r.pin, w.patch_w, r.acc, n, c.patchValues(), D, true);
        vision.epilogue(v, e, r.acc, w.patch_b, null, r.x, n, D, false);
        const cs: Ref = .{ .buf = t.cs.? };
        for (w.blocks[0..@min(o.blocks orelse c.layers, c.layers)]) |b| {
            glue.rmsnorm(gp, e, r.x, b.norm1, r.xn, n, D, c.eps);
            glue.mv(gp, e, r.xn, b.wqkv, r.acc, n, D, 3 * D, true);
            vision.qkv(v, e, r.acc, b.bqkv, cs, r.qkv, n, c.heads, c.headDim());
            vision.attention(v, e, r.qkv, r.o, n, c.heads);
            glue.mv(gp, e, r.o, b.wo, r.acc, n, D, D, true);
            vision.epilogue(v, e, r.acc, b.bo, r.x, r.x, n, D, false);
            glue.rmsnorm(gp, e, r.x, b.norm2, r.xn, n, D, c.eps);
            glue.mv(gp, e, r.xn, b.w1, r.acc, n, D, 2 * c.inter, true);
            vision.swiglu(v, e, r.acc, r.h, n, c.inter);
            glue.mv(gp, e, r.h, b.w2, r.acc, n, c.inter, D, true);
            vision.epilogue(v, e, r.acc, null, r.x, r.x, n, D, false);
        }
        if (!o.finish) return;
        glue.rmsnorm(gp, e, r.x, w.norm, r.xn, n, D, c.eps);
        vision.unshuffle(v, e, r.xn, r.u, gh, gw, D, c.ratio);
        glue.mv(gp, e, r.u, w.al_w1, r.acc, tokens, c.alignIn(), c.out_dim, true);
        vision.epilogue(v, e, r.acc, w.al_b1, null, r.t, tokens, c.out_dim, true);
        glue.mv(gp, e, r.t, w.al_w2, r.acc, tokens, c.out_dim, c.out_dim, true);
        vision.epilogue(v, e, r.acc, w.al_b2, null, r.out, tokens, c.out_dim, false);
    }
};

pub fn bf16(v: f32) u16 {
    return @import("weights.zig").bf16Round(v);
}
