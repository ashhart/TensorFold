//! MiMo-V2's native row ops and attention (ops.cu, attention.cu): typed launches, one per kernel.

const std = @import("std");
const cuda = @import("cuda");
const Stream = cuda.Stream;

pub const CK = 256; // keys a chunk (attention.cu)
pub const PROMPT_ROWS = 64; // rows a mimo_attn_prompt block (a warp 16), one query head
const NORM_THREADS = 256;
const SILU_THREADS = 256;
const ROUTER_WARPS = 8;
const TILE_WORDS = 32 * (97 + 64); // attention.cu's K and V tile, 32-bit words
const QUERY_WORDS = 96; // a 192-dim fp16 query

fn u(x: usize) u32 {
    return @intCast(x);
}
fn ci(x: usize) c_int {
    return @intCast(x);
}

/// Where a window's attention reads and writes (the decode buffers or a prompt chunk's).
pub const AttnIo = struct { q: u64, k: u64, v: u64, qo: u64, att: u64, pos: u64, slot: u64, po: u64, pm: u64, pl: u64 };

/// One attention layer's cache, RoPE tables and shape.
pub const Layer = struct { kc: u64, vc: u64, cos: u64, sin: u64, len: usize, hk: usize, win: usize, sinks: u64 };

pub const Ops = struct {
    ops: cuda.Module,
    attn: cuda.Module,
    f_rms: cuda.Function,
    f_silu: cuda.Function,
    f_router: cuda.Function,
    f_topk: cuda.Function,
    f_wsum: cuda.Function,
    f_qkv: cuda.Function,
    f_rows: cuda.Function,
    f_merge: cuda.Function,
    f_prompt: cuda.Function,

    pub fn load(d: *const cuda.Driver) !Ops {
        var o: Ops = undefined;
        o.ops = try cuda.Module.load(d, cuda.kernels.mimo_ops);
        errdefer o.ops.unload();
        o.attn = try cuda.Module.load(d, cuda.kernels.mimo_attention);
        errdefer o.attn.unload();
        inline for (.{ .{ "f_rms", "mimo_rms" }, .{ "f_silu", "mimo_silu_mul" }, .{ "f_router", "mimo_router" }, .{ "f_topk", "mimo_topk" }, .{ "f_wsum", "mimo_wsum" } }) |f|
            @field(o, f[0]) = try o.ops.function(f[1]);
        inline for (.{ .{ "f_qkv", "mimo_qkv" }, .{ "f_rows", "mimo_attn_rows" }, .{ "f_merge", "mimo_attn_merge" }, .{ "f_prompt", "mimo_attn_prompt" } }) |f|
            @field(o, f[0]) = try o.attn.function(f[1]);
        try o.f_prompt.allowDynamicShared(PROMPT_SHARED);
        return o;
    }

    pub fn unload(o: *Ops) void {
        o.attn.unload();
        o.ops.unload();
    }

    fn go(f: cuda.Function, s: Stream, grid: [3]usize, threads: usize, shared: usize, a: *cuda.Args) !void {
        try cuda.launch.launch(f, .{ .grid = .{ .x = u(grid[0]), .y = u(grid[1]), .z = u(grid[2]) }, .block = .{ .x = u(threads), .y = 1, .z = 1 }, .shared = u(shared) }, s, a);
    }

    /// out [n, D] fp16 = rms(v) * w, v = x [n, D] fp32, or res += x and v = res when `res` is set.
    pub fn rms(o: *const Ops, s: Stream, x: u64, res: u64, w: u64, out: u64, n: usize, D: usize, eps: f64) !void {
        if (D > 16 * NORM_THREADS) return error.NormTooWide;
        var a: cuda.Args = .{};
        inline for (.{ x, res, w, out }) |p| a.add(p);
        a.add(ci(D));
        a.add(@as(f32, @floatCast(eps)));
        try go(o.f_rms, s, .{ n, 1, 1 }, NORM_THREADS, 0, &a);
    }

    /// out [n, D] fp32 = sum over the K slots of w[row, k] * y[row, k, :], slot order.
    pub fn wsum(o: *const Ops, s: Stream, y: u64, w: u64, out: u64, n: usize, K: usize, D: usize) !void {
        var a: cuda.Args = .{};
        inline for (.{ y, w, out }) |p| a.add(p);
        a.add(ci(K));
        a.add(ci(D));
        try go(o.f_wsum, s, .{ n, (D + SILU_THREADS - 1) / SILU_THREADS, 1 }, SILU_THREADS, 0, &a);
    }

    /// out [n, N] fp16 = silu(g) * u.
    pub fn silu(o: *const Ops, s: Stream, g: u64, up: u64, out: u64, n: usize, N: usize) !void {
        var a: cuda.Args = .{};
        inline for (.{ g, up, out }) |p| a.add(p);
        a.add(ci(N));
        try go(o.f_silu, s, .{ n, (N + SILU_THREADS - 1) / SILU_THREADS, 1 }, SILU_THREADS, 0, &a);
    }

    /// The MoE's routing: fp32 logits [n, E] of x [n, D] fp16, then the top K (pick, wts [n, K]).
    pub fn route(o: *const Ops, s: Stream, x: u64, router: u64, bias: u64, logits: u64, pick: u64, wts: u64, n: usize, D: usize, E: usize, K: usize, scale: f32) !void {
        if (E > 1024 or K > 32 or D % 2 != 0) return error.RouterShape;
        {
            var a: cuda.Args = .{};
            inline for (.{ x, router, logits }) |p| a.add(p);
            a.add(ci(D));
            a.add(ci(E));
            try go(o.f_router, s, .{ (E + ROUTER_WARPS - 1) / ROUTER_WARPS, n, 1 }, 32 * ROUTER_WARPS, D * 4, &a);
        }
        var a: cuda.Args = .{};
        a.add(logits);
        a.add(bias);
        a.add(ci(E));
        a.add(ci(K));
        a.add(scale);
        a.add(pick);
        a.add(wts);
        const t = std.math.ceilPowerOfTwo(usize, @max(E, 32)) catch unreachable;
        try go(o.f_topk, s, .{ n, 1, 1 }, t, 0, &a);
    }

    /// q, k, v [n, heads * 192] -> rotated queries in io.qo; keys and values into the layer's cache.
    pub fn qkv(o: *const Ops, s: Stream, io: AttnIo, L: Layer, H: usize, n: usize) !void {
        var a: cuda.Args = .{};
        inline for (.{ io.q, io.k, io.v, L.cos, L.sin, io.pos, io.slot, io.qo, L.kc, L.vc }) |p| a.add(p);
        a.add(ci(L.len));
        a.add(ci(H));
        a.add(ci(L.hk));
        try go(o.f_qkv, s, .{ n, H + 2 * L.hk, 1 }, 64, 0, &a);
    }

    /// Decode rows (any slots and positions): `nch` chunks a row in parallel, then folded with the sinks into io.att.
    pub fn rows(o: *const Ops, s: Stream, io: AttnIo, L: Layer, H: usize, n: usize, nch: usize, scale: f32) !void {
        const G = H / L.hk;
        if (H % L.hk != 0 or G > 16 or H % 4 != 0) return error.AttentionShape;
        {
            var a: cuda.Args = .{};
            inline for (.{ io.qo, L.kc, L.vc, io.pos, io.slot, io.po, io.pm, io.pl }) |p| a.add(p);
            inline for (.{ L.len, nch, H, L.hk, L.win }) |v| a.add(ci(v));
            a.add(scale);
            try go(o.f_rows, s, .{ n, L.hk, nch }, 32 * G, (TILE_WORDS + G * QUERY_WORDS) * 4, &a);
        }
        var a: cuda.Args = .{};
        inline for (.{ io.po, io.pm, io.pl, L.sinks, io.att }) |p| a.add(p);
        a.add(ci(nch));
        a.add(ci(H));
        try go(o.f_merge, s, .{ n, H / 4, 1 }, 128, 0, &a);
    }

    /// A full layer's prompt rows of one slot (no sinks) on tensor cores: key tiles of 64 from 0, whatever the chunk.
    pub fn prompt(o: *const Ops, s: Stream, io: AttnIo, L: Layer, H: usize, n: usize, scale: f32) !void {
        if (H % L.hk != 0 or L.win != 0 or L.sinks != 0) return error.AttentionShape;
        var a: cuda.Args = .{};
        inline for (.{ io.qo, L.kc, L.vc, io.pos, io.slot, io.att }) |p| a.add(p);
        inline for (.{ n, L.len, H, L.hk }) |v| a.add(ci(v));
        a.add(scale);
        try go(o.f_prompt, s, .{ (n + PROMPT_ROWS - 1) / PROMPT_ROWS, H, 1 }, 128, PROMPT_SHARED, &a);
    }
};

/// attention.cu's tc::SMEM: two stages of 64 keys, K rows of 192 + 8 halves and V rows of 128 + 8.
pub const PROMPT_SHARED: u32 = 2 * 64 * ((192 + 8) + (128 + 8)) * 2;

test "attention shared memory fits a GB10 block" {
    try std.testing.expect(PROMPT_SHARED <= 99 * 1024);
    try std.testing.expectEqual(@as(u32, 86016), PROMPT_SHARED);
}
