//! What the loader hands the GPU, still on the host: the Python TextModel's layers with the bytes its kernels read.

const std = @import("std");
const quant = @import("core").quant;
const table = @import("table.zig");

pub const Tensor = table.Tensor;

/// A projection as its format read it from the checkpoint: the shared formats, or ModelOpt's NVFP4.
pub const Projection = union(enum) {
    mlx: quant.mlx.Host,
    dense: quant.dense.Host,
    bf16: quant.bf16.Host,
    nvfp4: quant.nvfp4.Host,

    /// Output rows N.
    pub fn rows(p: Projection) usize {
        return switch (p) {
            .mlx => |h| quant.mlx.rows(h),
            .dense => |h| quant.dense.rows(h),
            .bf16 => |h| quant.bf16.rows(h),
            .nvfp4 => |h| quant.nvfp4.rows(h),
        };
    }

    /// The host bytes the projection holds.
    pub fn bytes(p: Projection) usize {
        return switch (p) {
            .mlx => |h| quant.mlx.bytes(h),
            .dense => |h| quant.dense.bytes(h),
            .bf16 => |h| quant.bf16.bytes(h),
            .nvfp4 => |h| quant.nvfp4.bytes(h),
        };
    }
};

/// The shared formats' projection as this family reads it (a ModelOpt checkpoint goes through `modelopt.zig`).
pub fn read(a: std.mem.Allocator, t: anytype, key: []const u8) !Projection {
    return switch (try quant.read(a, t, key)) {
        .mlx => |x| .{ .mlx = x },
        .dense => |x| .{ .dense = x },
        .bf16 => |x| .{ .bf16 = x },
    };
}

/// A fused projection as its two halves along K.
pub fn halves(a: std.mem.Allocator, p: Projection) ![2]Projection {
    return switch (p) {
        .mlx => |x| pair(.mlx, try quant.mlx.halves(a, x)),
        .dense => |x| pair(.dense, try quant.dense.halves(a, x)),
        .bf16 => |x| pair(.bf16, try quant.bf16.halves(a, x)),
        .nvfp4 => |x| pair(.nvfp4, try quant.nvfp4.halves(a, x)),
    };
}

/// Output rows `spans`: a column-split projection's share on one rank.
pub fn sliceRows(a: std.mem.Allocator, p: Projection, spans: []const quant.Span) !Projection {
    return switch (p) {
        .mlx => |x| one(.mlx, try quant.mlx.sliceRows(a, x, spans)),
        .dense => |x| one(.dense, try quant.dense.sliceRows(a, x, spans)),
        .bf16 => |x| one(.bf16, try quant.bf16.sliceRows(a, x, spans)),
        .nvfp4 => |x| one(.nvfp4, try quant.nvfp4.sliceRows(a, x, spans)),
    };
}

/// One rank's input groups of a row-split projection.
pub fn sliceCols(a: std.mem.Allocator, p: Projection, r: quant.Rank, what: []const u8) !Projection {
    return switch (p) {
        .mlx => |x| one(.mlx, try quant.mlx.sliceCols(a, x, r, what)),
        .dense => |x| one(.dense, try quant.dense.sliceCols(a, x, r, what)),
        .bf16 => |x| one(.bf16, try quant.bf16.sliceCols(a, x, r, what)),
        .nvfp4 => |x| one(.nvfp4, try quant.nvfp4.sliceCols(a, x, r, what)),
    };
}

fn one(comptime tag: std.meta.Tag(Projection), v: anytype) Projection {
    return @unionInit(Projection, @tagName(tag), v);
}

fn pair(comptime tag: std.meta.Tag(Projection), v: anytype) [2]Projection {
    return .{ one(tag, v[0]), one(tag, v[1]) };
}

/// A layer's E + 1 experts, shared last, and its router rows (core/quant/experts.zig).
pub const Experts = quant.experts.Experts;
pub const Routed = quant.experts.Routed;

pub const DenseMlp = struct { gate: Projection, up: Projection, down: Projection };

pub const Mlp = union(enum) { dense: DenseMlp, routed: Routed };

/// A gated delta-net layer. Norms, conv [channels, kernel], a_log, dt_bias and gnorm are fp32.
pub const Linear = struct {
    input_norm: Tensor,
    post_norm: Tensor,
    qkv: Projection,
    z: Projection,
    a: Projection,
    b: Projection,
    conv: Tensor,
    a_log: Tensor,
    dt_bias: Tensor,
    gnorm: Tensor,
    out: Projection,
    mlp: Mlp,
};

pub const Full = struct {
    input_norm: Tensor,
    post_norm: Tensor,
    q: Projection,
    k: Projection,
    v: Projection,
    o: Projection,
    q_norm: Tensor,
    k_norm: Tensor,
    mlp: Mlp,
};

pub const Body = union(enum) { linear: Linear, full: Full };

/// One layer's tensors; `arena` holds what was converted or stacked, the rest points into the mapped files.
pub const Layer = struct {
    arena: std.heap.ArenaAllocator,
    body: Body,

    pub fn deinit(l: *Layer) void {
        l.arena.deinit();
        l.* = undefined;
    }
};

/// The MTP head: the Flash Next shape, or the Qwen3 one (`gated`) with input and post norms and an MLP.
pub const Mtp = struct {
    arena: std.heap.ArenaAllocator,
    /// The side files' table when the head came from them: the head's tensors point into its maps.
    owned: ?table.Table = null,
    io: std.Io,
    fc_e_norm: Tensor,
    fc_h_norm: Tensor,
    fc_e: Projection,
    fc_h: Projection,
    q_norm: Tensor,
    k_norm: Tensor,
    q: Projection,
    k: Projection,
    v: Projection,
    o: Projection,
    final_norm: Tensor,
    /// `null` ties with the main model's output head.
    head: ?Projection,
    input_norm: ?Tensor,
    post_norm: ?Tensor,
    mlp: ?Mlp,
    gated: bool,

    pub fn deinit(m: *Mtp) void {
        if (m.owned) |*t| t.close(m.io);
        m.arena.deinit();
        m.* = undefined;
    }
};
