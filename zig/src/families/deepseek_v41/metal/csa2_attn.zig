//! DeepSeek V4.1 CSA2 attention on Metal: pipelines and the launch encoders, wired into no forward.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");

pub const head_dim = 512;
pub const window = 128;
pub const max_selected = 512;
pub const rope_dims = 64;
/// Heads a simdgroup carries, and the groups of the two launch shapes.
pub const heads_per_sg = 1;
pub const chunks = 5;
pub const prompt_rows = 2;
pub const prompt_head_groups = 4;
pub const prompt_tile = 16;
pub const Ref = struct { buf: mtl.Buffer, off: usize = 0 };
pub const Out = enum { f32, bf16 };

/// One row's launch record: `wbase` -1 reads the window as a ring of 128 slots; decode's RoPE row is pos + rope.
pub const Meta = extern struct { pos: i32, wofs: i32, wbase: i32, cofs: i32, ncomp: i32, rope: i32 = 0, pad: [2]i32 = .{ 0, 0 } };

pub const Call = struct { rows: u32, heads: u32, out: Out = .bf16 };

const Args = extern struct { rows: u32, heads: u32, scale: f32, pad: u32 = 0 };
const dec_names = [_][:0]const u8{ "tf_dsv41_attn_dec_f32", "tf_dsv41_attn_dec_bf16" };
const pf_names = [_][:0]const u8{ "tf_dsv41_attn_dec_pf_f32", "tf_dsv41_attn_dec_pf_bf16" };
/// Decode launches of at most this many rows load each step's keys a step ahead (the same math, so the same bits).
pub const prefetch_rows = 2;
const pre_names = [_][:0]const u8{ "tf_dsv41_attn_pre_f32", "tf_dsv41_attn_pre_bf16" };

pub const Pipelines = struct {
    dec: [2]mtl.Pipeline,
    pf: [2]mtl.Pipeline,
    pre: [2]mtl.Pipeline,

    pub fn load(device: mtl.Device) !Pipelines {
        return loadSource(device, sources.dsv41.csa2_attn);
    }

    /// The same kernels from other text: the contract's planted controls prefix a TF_CTL define.
    pub fn loadSource(device: mtl.Device, text: []const u8) !Pipelines {
        const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
        defer lib.deinit();
        var p: Pipelines = undefined;
        for (dec_names, 0..) |name, i| p.dec[i] = try mtl.Pipeline.init(device, lib, name, false);
        for (pf_names, 0..) |name, i| p.pf[i] = try mtl.Pipeline.init(device, lib, name, false);
        for (pre_names, 0..) |name, i| p.pre[i] = try mtl.Pipeline.init(device, lib, name, false);
        return p;
    }

    pub fn deinit(p: *Pipelines) void {
        for (p.dec) |x| x.deinit();
        for (p.pf) |x| x.deinit();
        for (p.pre) |x| x.deinit();
    }
};

pub fn check(c: Call) !void {
    if (c.rows == 0 or c.heads == 0 or c.heads % heads_per_sg != 0) return error.BadAttnShape;
}

/// The softmax scale, head_dim^-0.5 in fp32.
pub fn scale() f32 {
    return 1.0 / @sqrt(@as(f32, head_dim));
}

/// Q [rows][heads][512] bf16 (RoPE'd), W and C bf16 rows, I [rows][512] selected, M Meta, SK sinks, CS [pos][64] fp32.
pub const Bind = struct { q: Ref, w: Ref, c: Ref, i: Ref, m: Ref, sk: Ref, cs: Ref, o: Ref };

fn bind(e: mtl.ComputeEncoder, b: Bind, c: Call) void {
    const refs = [_]Ref{ b.q, b.w, b.c, b.i, b.m, b.sk, b.cs, b.o };
    for (refs, 0..) |r, n| e.setBuffer(r.buf, r.off, n);
    e.setValue(Args{ .rows = c.rows, .heads = c.heads, .scale = scale() }, 8);
}

/// Decode rows: any positions and windows, ring or linear; a threadgroup is one row by 2 heads, a simdgroup a chunk.
pub fn encodeDecode(p: *const Pipelines, e: mtl.ComputeEncoder, b: Bind, c: Call) !void {
    return encodeDecodeAs(p, e, b, c, c.rows <= prefetch_rows);
}

/// `encodeDecode` with the key prefetch chosen by the caller.
pub fn encodeDecodeAs(p: *const Pipelines, e: mtl.ComputeEncoder, b: Bind, c: Call, prefetch: bool) !void {
    try check(c);
    e.setPipeline((if (prefetch) p.pf else p.dec)[@backingInt(c.out)]);
    bind(e, b, c);
    e.dispatchGroups(mtl.Size.of(c.rows, c.heads / heads_per_sg, 1), mtl.Size.of(32 * chunks, 1, 1));
}

/// Prompt rows: consecutive positions of one stream over a linear window buffer (`wbase` >= 0); bits equal decode's.
pub fn encodePrompt(p: *const Pipelines, e: mtl.ComputeEncoder, b: Bind, c: Call) !void {
    try check(c);
    e.setPipeline(p.pre[@backingInt(c.out)]);
    bind(e, b, c);
    const per = heads_per_sg * prompt_head_groups;
    e.dispatchGroups(mtl.Size.of((c.rows + prompt_rows - 1) / prompt_rows, (c.heads + per - 1) / per, 1), mtl.Size.of(32 * prompt_rows * prompt_head_groups, 1, 1));
}
