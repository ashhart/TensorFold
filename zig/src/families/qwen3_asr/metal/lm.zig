//! The Qwen3 language model's binding on Metal: RMS norms, QK norm with RoPE into the caches, causal GQA attention.
const std = @import("std");
const cfg = @import("../config.zig");
const shared = @import("../lm.zig");
const gpu = @import("gpu.zig");
const mt = @import("tower.zig");

const QkRope = extern struct { h: u32, hk: u32, dim: u32, eps: f32, cap: u32 };

fn u(x: usize) u32 {
    return @intCast(x);
}

pub const Lm = struct {
    t: cfg.Text,
    g: *gpu.Gpu,
    cap: usize, // positions the caches hold
    first: usize = 0, // the pass's first position
    b: Buffers,
    logits: []u16,
    last_margin: f32 = 0, // the last pick's lead over the runner-up
    sink: ?mt.Dump = null,

    const Buffers = struct { ids: u64, pos: u64, q: u64, logits: u64, cos: u64, sin: u64, kc: u64, vc: u64, zero: u64 };

    /// Passes of up to `max_rows` rows over up to `span` positions.
    pub fn init(gpa: std.mem.Allocator, g: *gpu.Gpu, t: cfg.Text, max_rows: usize, span: usize) !Lm {
        if (t.head_dim > 256 or t.head_dim % 32 != 0 or t.heads % t.kv_heads != 0) return error.UnsupportedQwen3AsrGeometry;
        if (span > t.context) return error.ContextTooLong;
        if (span > mt.max_keys) return error.AttentionWindow;
        const r = max_rows;
        const qd = t.heads * t.head_dim;
        const kvd = t.kv_heads * t.head_dim;
        const kv = t.layers * span * kvd * 2;
        var lm: Lm = .{ .t = t, .g = g, .cap = span, .b = .{
            .ids = try g.alloc(4 * r),
            .pos = try g.alloc(4 * r),
            .q = try g.alloc(2 * r * qd),
            .logits = try g.alloc(2 * t.vocab),
            .cos = try g.alloc(4 * span * t.head_dim),
            .sin = try g.alloc(4 * span * t.head_dim),
            .kc = try g.alloc(kv),
            .vc = try g.alloc(kv),
            .zero = try g.alloc(2 * @max(t.vocab, 2 * t.intermediate, qd + 2 * kvd, t.hidden)),
        }, .logits = undefined };
        const cs = std.mem.bytesAsSlice(f32, g.host(lm.b.cos, 4 * span * t.head_dim));
        const sn = std.mem.bytesAsSlice(f32, g.host(lm.b.sin, 4 * span * t.head_dim));
        shared.ropeTables(t, @alignCast(cs), @alignCast(sn), span);
        lm.logits = try gpa.alloc(u16, t.vocab);
        return lm;
    }

    pub fn deinit(lm: *Lm, gpa: std.mem.Allocator) void {
        gpa.free(lm.logits);
    }

    /// The pass's ids and positions [pos, pos + ids.len) on the device.
    pub fn positions(lm: *Lm, ids: []const u32, pos: usize) !void {
        const rows = ids.len;
        if (4 * rows > lm.g.ref(lm.b.ids).buf.length()) return error.PassRows;
        if (pos + rows > lm.cap) return error.ContextFull;
        try lm.g.flush(); // the last pass may still read them
        const hi = std.mem.bytesAsSlice(i32, lm.g.host(lm.b.ids, 4 * rows));
        const hp = std.mem.bytesAsSlice(i32, lm.g.host(lm.b.pos, 4 * rows));
        for (ids, 0..) |id, i| {
            hi[i] = @intCast(id);
            hp[i] = @intCast(pos + i);
        }
        lm.first = pos;
    }

    pub fn dump(lm: *Lm, stage: []const u8, ptr: u64, rows: usize, cols: usize) !void {
        if (lm.sink) |dm| {
            try lm.g.flush();
            try dm.put(dm.ctx, stage, ptr, rows, cols, null);
        }
    }

    pub fn embed(lm: *Lm, table: u64, out: u64, rows: usize, hidden: usize) !void {
        lm.g.go(.q3a_embed, .{ (hidden + 255) / 256, rows, 1 }, 256, &.{ lm.b.ids, table, out }, [2]u32{ u(rows), u(hidden) });
    }

    pub fn copy(lm: *Lm, dst: u64, src: u64, n: usize) !void {
        lm.g.go(.q3a_copy, .{ (n / 2 + 255) / 256, 1, 1 }, 256, &.{ src, dst }, u(n / 2));
    }

    /// h = bf16(x + res) (x when res is null, h unwritten), y = bf16(w bf16(h rsqrt(mean h^2 + eps))).
    pub fn addNorm(lm: *Lm, x: u64, res: ?u64, w: u64, h: u64, y: u64, rows: usize) !void {
        const p: mt.Rows = .{ .rows = u(rows), .width = u(lm.t.hidden), .eps = lm.t.eps, .has_res = @intFromBool(res != null) };
        lm.g.go(.q3a_add_rmsnorm, .{ rows, 1, 1 }, 256, &.{ x, res orelse x, w, h, y }, p);
    }

    pub fn gemm(lm: *Lm, x: u64, w: u64, y: u64, rows: usize, n: usize, k: usize) !void {
        mt.gemm(lm.g, x, w, lm.b.zero, null, y, rows, n, k, .none);
    }

    fn kvAt(lm: *const Lm, base: u64, layer: usize) u64 {
        return base + layer * lm.cap * lm.t.kv_heads * lm.t.head_dim * 2;
    }

    /// q and k RMS-normed per head and rotated, k and v into layer's caches at the pass's positions.
    pub fn qkRope(lm: *Lm, qkv: u64, q_norm: u64, k_norm: u64, layer: usize, rows: usize) !void {
        const t = lm.t;
        const p: QkRope = .{ .h = u(t.heads), .hk = u(t.kv_heads), .dim = u(t.head_dim), .eps = t.eps, .cap = u(lm.cap) };
        const bufs = [_]u64{ qkv, q_norm, k_norm, lm.b.cos, lm.b.sin, lm.b.pos, lm.b.q, lm.kvAt(lm.b.kc, layer), lm.kvAt(lm.b.vc, layer) };
        lm.g.go(.q3a_qk_rope, .{ rows, t.heads + 2 * t.kv_heads, 1 }, t.head_dim, &bufs, p);
    }

    /// Causal GQA attention of the pass's q rows over layer's caches into out [rows, heads x dim].
    pub fn attention(lm: *Lm, layer: usize, out: u64, rows: usize) !void {
        const t = lm.t;
        const qd = t.heads * t.head_dim;
        const kvd = t.kv_heads * t.head_dim;
        const p: mt.Attn = .{
            .q_ld = u(qd),
            .k_ld = u(kvd),
            .v_ld = u(kvd),
            .o_ld = u(qd),
            .heads = u(t.heads),
            .group = u(t.heads / t.kv_heads),
            .dim = u(t.head_dim),
            .keys = 0,
            .first = u(lm.first),
            .scale = @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(t.head_dim)))),
        };
        lm.g.go(.q3a_attn, .{ rows, t.heads, 1 }, 128, &.{ lm.b.q, lm.kvAt(lm.b.kc, layer), lm.kvAt(lm.b.vc, layer), out }, p);
    }

    /// out [rows, i] = bf16(silu(gate) up) of gu's [gate | up] rows.
    pub fn swiglu(lm: *Lm, gu: u64, out: u64, rows: usize, i: usize) !void {
        lm.g.go(.q3a_swiglu, .{ (i + 255) / 256, rows, 1 }, 256, &.{ gu, out }, [2]u32{ u(rows), u(i) });
    }

    /// The tied head over one normed row, downloaded: its greedy pick.
    pub fn greedy(lm: *Lm, row: u64, embed_table: u64) !u32 {
        mt.gemm(lm.g, row, embed_table, lm.b.zero, null, lm.b.logits, 1, lm.t.vocab, lm.t.hidden, .none);
        try lm.g.get(lm.b.logits, std.mem.sliceAsBytes(lm.logits));
        const best = shared.argmax(lm.logits);
        lm.last_margin = shared.margin(lm.logits, best);
        return best;
    }
};
