//! A sequence's state (KDA states by slot, MLA caches, the MTP head's last) and a window's scratch.
const std = @import("std");
const mtl = @import("metal");
const cfg = @import("config.zig");
const Ref = @import("weights.zig").Ref;
const hc = @import("../../core/hc.zig");

pub const max_rows = 16;
/// MLA caches: the backbone's 11 layers, then the MTP head's.
pub const max_mla = 12;
pub const max_kda = 34;

/// Bump allocation inside shared buffers, 256-byte aligned.
pub const Arena = struct {
    device: mtl.Device,
    gpa: std.mem.Allocator,
    buffers: std.ArrayList(mtl.Buffer) = .empty,
    bytes: usize = 0,

    pub fn buffer(a: *Arena, len: usize) !Ref {
        const b = try a.device.buffer(@max(len, 16), mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        a.buffers.append(a.gpa, b) catch |err| {
            b.deinit();
            return err;
        };
        a.bytes += @max(len, 16);
        @memset(b.contents()[0..@max(len, 16)], 0);
        return .{ .buf = b };
    }

    pub fn deinit(a: *Arena) void {
        for (a.buffers.items) |b| b.deinit();
        a.buffers.deinit(a.gpa);
    }
};

const Carve = struct {
    base: Ref,
    at: usize = 0,

    fn take(c: *Carve, len: usize) Ref {
        const r = c.base.at(c.at);
        c.at = std.mem.alignForward(usize, c.at + len, 256);
        return r;
    }
};

pub const Kda = struct {
    st: [2]Ref, // fp32 [heads, dim, dim]
    cs: [2]Ref, // bf16 [taps - 1, 3 width]
    cur: u1 = 0, // which of each pair holds the state now
    proj: Ref, // the last window's stacked projections [rows, proj] (a partial keep replays them)
};

pub const Mla = struct { keys: Ref, ik: Ref, ig: Ref, pool: Ref };

pub const State = struct {
    cap: u32,
    pos: u32 = 0, // backbone tokens in the caches
    mtp_pos: u32 = 0, // the MTP head's rows in its cache
    kda: [max_kda]Kda = undefined,
    mla: [max_mla]Mla = undefined,

    pub fn reset(s: *State) void {
        s.pos = 0;
        s.mtp_pos = 0;
        for (&s.kda) |*k| {
            k.cur = 0;
            @memset(k.st[0].addr()[0 .. 64 * 128 * 128 * 4], 0);
            @memset(k.cs[0].addr()[0 .. 3 * 3 * 8192 * 2], 0);
        }
    }
};

pub const Scratch = struct {
    ids: Ref, // u32 [rows]: the window's tokens
    next: Ref, // u32 [rows]: tokens after each row (the MTP head's inputs)
    x: [2]Ref, // bf16 [rows, 4, D] streams, alternating at each write-back
    h: Ref, // bf16 [rows, D] embedding rows
    normed: Ref,
    branch: Ref,
    post: Ref,
    comb: Ref,
    inv: Ref,
    mixes: Ref, // fp32 [rows, slices, 25]: each boundary slice's partial mixes and squares
    z: Ref,
    y: Ref, // KDA output [rows, width]
    xp: Ref,
    qr: Ref,
    qp: Ref,
    ql: Ref, // [rows * heads, rank]
    qls: Ref,
    scores: Ref, // bf16 [rows, heads, 2048]
    probs: Ref,
    att: Ref, // [rows * heads, rank]
    vals: Ref, // [rows, heads * v]
    iw: Ref,
    sscore: Ref, // fp32 [rows, cap / 4]
    indices: Ref, // i32 [rows, key width]
    sel: Ref, // GlmSelectArgs [rows]
    xf: Ref,
    logits_r: Ref, // router fp32 [rows, experts]
    pick: Ref,
    wts: Ref,
    uids: Ref,
    umem: Ref,
    ucount: Ref,
    none: Ref,
    acts: Ref,
    ys: Ref,
    act: Ref,
    ye: Ref,
    gu: Ref,
    actd: Ref,
    raw: Ref,
    hidden: Ref, // bf16 [rows, D]: final-normed rows (the LM head's and the MTP head's input)
    logits: Ref, // bf16 [rows, vocab]
    picks: Ref, // u32 [rows]
    // the MTP head
    m_emb: Ref,
    m_eh: Ref,
    m_x: Ref,
    m_xn: Ref,
    m_out: Ref,
    m_hn: Ref,
    m_logits: Ref,
    m_picks: Ref,
    rl: Ref, // one Mac's route lists: the picks it computes, the peer's (none), their counts and the count's word
    yp: Ref, // fp32 [rows * topk, D]: expert parallel by rows, each pick's down partial
};

/// The bytes `initState` takes for `cap` tokens (`shared`: with another state's projections).
pub fn stateBytes(c: *const cfg.Config, cap: u32, shared: bool) usize {
    const st_bytes: usize = @as(usize, c.kda_heads) * c.kda_dim * c.kda_dim * 4;
    const cs_bytes: usize = @as(usize, c.conv - 1) * 3 * c.kdaWidth() * 2;
    const proj_bytes: usize = if (shared) 0 else max_rows * c.kdaProj() * 2;
    const n_mla = c.countKind(.mla) + @as(u32, if (c.mtp > 0) 1 else 0);
    const mla: usize = @as(usize, cap) * (c.kv_lora + 2 * c.i_dim) * 2 + @as(usize, cap / c.kpool + 1) * c.i_dim * 2 + 4 * 256;
    return c.countKind(.kda) * (2 * st_bytes + 2 * cs_bytes + proj_bytes + 5 * 256) + n_mla * mla;
}

/// One sequence's caches for `cap` tokens; with `share`, its KDA projections are `share`'s (any stream's window rows).
pub fn initState(arena: *Arena, c: *const cfg.Config, cap: u32, share: ?*const State) !State {
    var s: State = .{ .cap = cap };
    const n_kda = c.countKind(.kda);
    const st_bytes: usize = @as(usize, c.kda_heads) * c.kda_dim * c.kda_dim * 4;
    const cs_bytes: usize = @as(usize, c.conv - 1) * 3 * c.kdaWidth() * 2;
    const proj_bytes: usize = if (share == null) max_rows * c.kdaProj() * 2 else 0;
    var kc: Carve = .{ .base = try arena.buffer(n_kda * (2 * st_bytes + 2 * cs_bytes + proj_bytes + 5 * 256)) };
    for (0..n_kda) |i| s.kda[i] = .{
        .st = .{ kc.take(st_bytes), kc.take(st_bytes) },
        .cs = .{ kc.take(cs_bytes), kc.take(cs_bytes) },
        .proj = if (share) |o| o.kda[i].proj else kc.take(proj_bytes),
    };
    const n_mla = c.countKind(.mla) + @as(u32, if (c.mtp > 0) 1 else 0);
    for (0..n_mla) |i| {
        var m: Carve = .{ .base = try arena.buffer(@as(usize, cap) * (c.kv_lora + 2 * c.i_dim) * 2 + @as(usize, cap / c.kpool + 1) * c.i_dim * 2 + 4 * 256) };
        s.mla[i] = .{
            .keys = m.take(@as(usize, cap) * c.kv_lora * 2),
            .ik = m.take(@as(usize, cap) * c.i_dim * 2),
            .ig = m.take(@as(usize, cap) * c.i_dim * 2),
            .pool = m.take(@as(usize, cap / c.kpool + 1) * c.i_dim * 2),
        };
    }
    return s;
}

/// The sequence state for `cap` tokens and the scratch for windows of up to 16 rows.
pub fn init(arena: *Arena, c: *const cfg.Config, cap: u32) !struct { state: State, scratch: Scratch } {
    const R: usize = max_rows;
    const D: usize = c.hidden;
    const s = try initState(arena, c, cap, null);
    const H: usize = c.mla_heads;
    const V: usize = c.vocab;
    const sizes = [_]usize{
        R * 4,                                                                   R * 4,                        R * 4 * D * 2,            R * 4 * D * 2,               R * D * 2,
        R * D * 2,                                                               R * D * 2,                    R * 4 * 4,                R * 16 * 4,                  R * 4,
        R * hc.partBytes(.{ .width = @intCast(D), .sinkhorn = 0, .eps_e9 = 0 }), 64,                           R * c.kdaWidth() * 2,     R * c.xProj() * 2,           R * c.q_lora * 2,
        R * c.qrProj() * 2,                                                      R * H * c.kv_lora * 2,        R * H * c.kv_lora * 2,    R * H * c.i_topk * 2,        R * H * c.i_topk * 2,
        R * H * c.kv_lora * 2,                                                   R * H * c.v_dim * 2,          R * c.i_heads * 2,        R * (cap / c.kpool + 1) * 4, R * c.keyWidth() * 4,
        R * 6 * 4,                                                               R * D * 4,                    R * c.experts * 4,        R * c.topk * 4,              R * c.topk * 4,
        R * c.topk * 4 + 64,                                                     R * c.topk * R * 4,           64,                       64,                          R * c.moe_inter * 2,
        R * D * 2,                                                               R * c.topk * c.moe_inter * 2, R * c.topk * D * 2,       R * 2 * c.dense_inter * 2,   R * c.dense_inter * 2,
        R * D * 2,                                                               R * D * 2,                    R * V * 2,                R * 4,                       R * D * 2,
        R * 2 * D * 2,                                                           R * D * 2,                    R * D * 2,                R * D * 2,                   R * D * 2,
        R * V * 2,                                                               R * 4,                        R * c.topk * 4 * 2 + 256, R * c.topk * D * 4,
    };
    var total: usize = 0;
    for (sizes) |n| total += std.mem.alignForward(usize, n, 256);
    var sc: Carve = .{ .base = try arena.buffer(total) };
    var refs: [sizes.len]Ref = undefined;
    for (sizes, 0..) |n, i| refs[i] = sc.take(n);
    var scratch: Scratch = undefined;
    const info = @typeInfo(Scratch).@"struct";
    inline for (info.field_names, info.field_types, 0..) |name, T, i| {
        if (T == Ref) {
            @field(scratch, name) = refs[flat(i)];
        } else {
            @field(scratch, name) = .{ refs[flat(i)], refs[flat(i) + 1] };
        }
    }
    return .{ .state = s, .scratch = scratch };
}

/// The `sizes` index of Scratch field i (the stream pair `x` takes two).
fn flat(comptime i: usize) usize {
    return if (i <= 2) i else i + 1;
}

test "the scratch sizes line up with its fields" {
    const n = @typeInfo(Scratch).@"struct".field_names.len;
    try std.testing.expectEqual(@as(usize, 53), n);
}
