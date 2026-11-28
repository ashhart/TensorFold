//! Small DSpark's fixed-shape Metal entry points (7 Small-geometry kernels + 4 shared DSpark kernels).
const std = @import("std");
const mtl = @import("metal");
const sources = @import("inkling2_metal");
const shared = @import("dspark_kernels.zig");

pub const hidden: u32 = 4608;
pub const query_width: u32 = 6144;
pub const kv_width: u32 = 1024;
pub const q_heads: u32 = 48;
pub const kv_heads: u32 = 8;
pub const block_rows: u32 = 8;
pub const rank: u32 = 256;
pub const rnn_width: u32 = 768;
pub const intermediate: u32 = 18432;
pub const vocab: u32 = 200058;
pub const taps_count: u32 = 8;
pub const fusions_count: u32 = 6;
pub const Ref = @import("inkling2_metal_ref").Ref;
pub const FuseArgs = shared.FuseArgs;
pub const PositionArgs = shared.PositionArgs;
pub const ContextArgs = shared.ContextArgs;
pub const AttentionArgs = shared.AttentionArgs;
pub const KappaArgs = shared.KappaArgs;
pub const TokenArgs = shared.TokenArgs;
pub const ResidualArgs = shared.ResidualArgs;
pub const CountArgs = shared.CountArgs;
pub const BaseArgs = shared.BaseArgs;
pub const LogitsArgs = shared.LogitsArgs;
pub const GroupStat = shared.GroupStat;

pub const roundBf16 = shared.roundBf16;

pub const Kind = enum {
    init_rows,
    fuse_taps,
    block_qkv,
    attention,
    residual_norm,
    kappa,
    base,

    pub fn symbol(k: Kind) []const u8 {
        return switch (k) {
            .init_rows => "tf_ink2_small_dspark_init_rows",
            .fuse_taps => "tf_ink2_small_dspark_fuse_taps",
            .block_qkv => "tf_ink2_small_dspark_block_qkv",
            .attention => "tf_ink2_small_dspark_attention",
            .residual_norm => "tf_ink2_small_dspark_residual_norm",
            .kappa => "tf_ink2_small_dspark_kappa",
            .base => "tf_ink2_small_dspark_base",
        };
    }

    pub fn label(k: Kind) []const u8 {
        return switch (k) {
            inline else => |v| std.fmt.comptimePrint("metal inkling2 small_dspark {s} {s}", .{ @tagName(v), comptime v.symbol() }),
        };
    }

    pub fn requiredThreads(k: Kind) u32 {
        return switch (k) {
            .kappa => 64,
            .block_qkv => 128,
            .attention => 384,
            else => 256,
        };
    }
};

pub const Pipelines = struct {
    init_rows: mtl.Pipeline,
    fuse_taps: mtl.Pipeline,
    block_qkv: mtl.Pipeline,
    attention: mtl.Pipeline,
    residual_norm: mtl.Pipeline,
    kappa: mtl.Pipeline,
    base: mtl.Pipeline,
    context_kv: mtl.Pipeline,
    swiglu: mtl.Pipeline,
    recurrent: mtl.Pipeline,
    logits: mtl.Pipeline,

    pub fn load(device: mtl.Device) !Pipelines {
        return loadSource(device, sources.inkling2_small_dspark);
    }

    pub fn loadSource(device: mtl.Device, text: []const u8) !Pipelines {
        const prefix = "#define TF_SIMD_FRAGS 1\n";
        const owned = if (device.tensorUnits()) null else try std.mem.concat(std.heap.page_allocator, u8, &.{ prefix, text });
        defer if (owned) |buf| std.heap.page_allocator.free(buf);
        const lib = try mtl.Library.fromSource(device, owned orelse text, mtl.CompileOptions.mlx());
        defer lib.deinit();
        const shared_lib = try mtl.Library.fromSource(device, sources.inkling2_dspark_pre_m5, mtl.CompileOptions.mlx());
        defer shared_lib.deinit();
        var out: Pipelines = undefined;
        var loaded: usize = 0;
        errdefer inline for (@typeInfo(Pipelines).@"struct".field_names, 0..) |name, i| {
            if (loaded > i) @field(out, name).deinit();
        };
        inline for (std.enums.values(Kind)) |kind| {
            const name = @tagName(kind);
            @field(out, name) = try mtl.Pipeline.init(device, lib, kind.symbol(), false);
            loaded += 1;
            if (@field(out, name).maxThreads() < kind.requiredThreads()) return error.DsparkThreads;
        }
        const shared_specs = .{
            .{ "context_kv", "tf_ink2_dspark_context_kv", @as(u32, 128) },
            .{ "swiglu", "tf_ink2_dspark_swiglu", @as(u32, 256) },
            .{ "recurrent", "tf_ink2_dspark_recurrent", @as(u32, 256) },
            .{ "logits", "tf_ink2_dspark_logits", @as(u32, 256) },
        };
        inline for (shared_specs) |spec| {
            @field(out, spec[0]) = try mtl.Pipeline.init(device, shared_lib, spec[1], false);
            loaded += 1;
            if (@field(out, spec[0]).maxThreads() < spec[2]) return error.DsparkThreads;
        }
        return out;
    }

    pub fn deinit(p: *Pipelines) void {
        inline for (@typeInfo(Pipelines).@"struct".field_names) |name| @field(p, name).deinit();
        p.* = undefined;
    }
};

fn bind(e: mtl.ComputeEncoder, refs: []const Ref) void {
    for (refs, 0..) |r, slot| e.setBuffer(r.buf, r.off, slot);
}
fn groups(e: mtl.ComputeEncoder, count: usize, width: usize) void {
    e.dispatchGroups(mtl.Size.of(count, 1, 1), mtl.Size.of(width, 1, 1));
}
fn threads(e: mtl.ComputeEncoder, count: usize, width: usize) void {
    e.dispatchThreads(mtl.Size.of(count, 1, 1), mtl.Size.of(width, 1, 1));
}

pub fn initializeRows(p: *const Pipelines, e: mtl.ComputeEncoder, anchor: Ref, mask_embed: Ref, out: Ref, streams: u32) void {
    std.debug.assert(streams > 0 and streams <= 8);
    e.setPipeline(p.init_rows);
    bind(e, &.{ anchor, mask_embed, out });
    groups(e, @as(usize, streams) * block_rows, 256);
}

pub fn fuseTaps(p: *const Pipelines, e: mtl.ComputeEncoder, taps: Ref, fuse_weight: Ref, norm_weight: Ref, c: Ref, newest_f: Ref, rows: u32, stride_rows: u32) void {
    std.debug.assert(rows > 0 and stride_rows >= rows);
    e.setPipeline(p.fuse_taps);
    bind(e, &.{ taps, fuse_weight, norm_weight, c, newest_f });
    e.setValue(FuseArgs{ .count = rows, .stride_rows = stride_rows }, 5);
    groups(e, @as(usize, fusions_count) * @as(usize, rows), 256);
}

pub fn blockQkv(p: *const Pipelines, e: mtl.ComputeEncoder, q_raw: Ref, k_raw: Ref, v_raw: Ref, q_norm: Ref, k_norm: Ref, q: Ref, k: Ref, v: Ref, bases: []const u32) void {
    std.debug.assert(bases.len > 0 and bases.len <= 8);
    var pos: PositionArgs = .{ .base = @splat(0) };
    @memcpy(pos.base[0..bases.len], bases);
    e.setPipeline(p.block_qkv);
    bind(e, &.{ q_raw, k_raw, v_raw, q_norm, k_norm, q, k, v });
    e.setValue(pos, 8);
    groups(e, bases.len * block_rows * 64, 128);
}

pub fn commitContext(p: *const Pipelines, e: mtl.ComputeEncoder, k_raw: Ref, v_raw: Ref, k_norm_ctx: Ref, keys: Ref, values: Ref, base_position: u32, rows: u32, ring_capacity: u32) void {
    std.debug.assert(rows > 0 and ring_capacity > 0);
    e.setPipeline(p.context_kv);
    bind(e, &.{ k_raw, v_raw, k_norm_ctx, keys, values });
    e.setValue(ContextArgs{ .base = base_position, .rows = rows, .ring = ring_capacity }, 5);
    groups(e, @as(usize, rows) * kv_heads, 128);
}

pub fn attend(p: *const Pipelines, e: mtl.ComputeEncoder, q: Ref, block_k: Ref, block_v: Ref, out: Ref, ctx_k: []const Ref, ctx_v: []const Ref, committed: []const u32, ring_capacity: []const u32, from: []const u32, local: bool) void {
    const n = ctx_k.len;
    std.debug.assert(n > 0 and n <= 8 and ctx_v.len == n and committed.len == n and ring_capacity.len == n and from.len == n);
    e.setPipeline(p.attention);
    bind(e, &.{ q, block_k, block_v, out });
    for (0..8) |i| {
        const idx = if (i < n) i else 0;
        e.setBuffer(ctx_k[idx].buf, ctx_k[idx].off, 4 + i);
        e.setBuffer(ctx_v[idx].buf, ctx_v[idx].off, 12 + i);
    }
    var args: AttentionArgs = .{ .local = @intFromBool(local), .streams = @intCast(n), .committed = @splat(0), .ring = @splat(1), .from = @splat(0) };
    for (0..n) |i| {
        std.debug.assert(ring_capacity[i] > 0 and from[i] <= committed[i]);
        args.committed[i] = committed[i];
        args.ring[i] = ring_capacity[i];
        args.from[i] = from[i];
    }
    e.setValue(args, 20);
    e.dispatchGroups(mtl.Size.of(kv_heads, n, 3), mtl.Size.of(384, 1, 1));
}

pub fn residualNorm(p: *const Pipelines, e: mtl.ComputeEncoder, x: Ref, residual: Ref, weight: Ref, new_residual: Ref, normalized: Ref, rows: u32, init: bool) void {
    std.debug.assert(rows > 0);
    e.setPipeline(p.residual_norm);
    bind(e, &.{ x, residual, weight, new_residual, normalized });
    e.setValue(ResidualArgs{ .init = @intFromBool(init) }, 5);
    groups(e, rows, 256);
}

pub fn swiglu(p: *const Pipelines, e: mtl.ComputeEncoder, gate: Ref, up: Ref, out: Ref, count: u32) void {
    std.debug.assert(count > 0 and count % (block_rows * intermediate) == 0);
    e.setPipeline(p.swiglu);
    bind(e, &.{ gate, up, out });
    e.setValue(CountArgs{ .count = count }, 3);
    threads(e, count, 256);
}

pub fn kappa(p: *const Pipelines, e: mtl.ComputeEncoder, y: Ref, newest_fs: []const Ref, rnn_kappa: Ref, rnn_bias: Ref, out: Ref, rows: u32, stride_s: u32, row_off: u32) void {
    const n = newest_fs.len;
    std.debug.assert(rows > 0 and rows <= 7 and n > 0 and n <= 8 and stride_s >= row_off + rows);
    e.setPipeline(p.kappa);
    e.setBuffer(y.buf, y.off, 0);
    for (0..8) |i| {
        const f = newest_fs[if (i < n) i else 0];
        e.setBuffer(f.buf, f.off, 1 + i);
    }
    e.setBuffer(rnn_kappa.buf, rnn_kappa.off, 9);
    e.setBuffer(rnn_bias.buf, rnn_bias.off, 10);
    e.setBuffer(out.buf, out.off, 11);
    e.setValue(KappaArgs{ .rows = rows, .stride_s = stride_s, .row_off = row_off, .streams = @intCast(n) }, 12);
    e.dispatchGroups(mtl.Size.of(12, n, 1), mtl.Size.of(64, 1, 1));
}

pub fn recurrentStep(p: *const Pipelines, e: mtl.ComputeEncoder, state: Ref, kappa_step: Ref, rnn_rec: Ref, markov_w1: Ref, z: Ref, next_state: Ref, lambda: Ref, held: Ref, streams: u32, token: u32, step: u32, use_held: bool) void {
    std.debug.assert(streams > 0 and streams <= 8);
    e.setPipeline(p.recurrent);
    bind(e, &.{ state, kappa_step, rnn_rec, markov_w1, z, next_state, lambda, held });
    e.setValue(TokenArgs{ .token = token, .step = step, .use_held = @intFromBool(use_held) }, 8);
    groups(e, streams, 256);
}

pub fn base(p: *const Pipelines, e: mtl.ComputeEncoder, unembed: Ref, y: Ref, out: Ref, rows: u32, vocabulary_count: u32) void {
    std.debug.assert(rows > 0 and vocabulary_count > 0 and vocabulary_count <= vocab);
    e.setPipeline(p.base);
    bind(e, &.{ unembed, y, out });
    e.setValue(BaseArgs{ .rows = rows, .vocab = vocabulary_count }, 3);
    if (rows < 4) {
        groups(e, (vocabulary_count + 7) / 8, 256);
    } else {
        e.dispatchGroups(mtl.Size.of((vocabulary_count + 31) / 32, (rows + 15) / 16, 1), mtl.Size.of(256, 1, 1));
    }
}

pub fn finalLogits(p: *const Pipelines, e: mtl.ComputeEncoder, base_values: Ref, phi: Ref, beta: Ref, lambda: Ref, out: Ref, stats: Ref, held: Ref, probs: Ref, sample_tok: Ref, args: LogitsArgs) void {
    std.debug.assert(args.streams > 0 and args.streams <= 8 and args.count > 0 and args.count <= vocab);
    e.setPipeline(p.logits);
    bind(e, &.{ base_values, phi, beta, lambda, out, stats, held, probs, sample_tok });
    e.setValue(args, 9);
    if (args.mode == 0) {
        groups(e, (args.count + 255) / 256, 256);
    } else {
        groups(e, args.streams, 256);
    }
}
