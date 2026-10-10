//! What the loader hands the GPU, still on the host: the Python TextModel's layers with the bytes its kernels read.

const std = @import("std");
const quant = @import("core").quant;
const table = @import("table.zig");

pub const Tensor = table.Tensor;

/// A projection as its format read it from the checkpoint.
pub const Projection = quant.Host;

/// A layer's E + 1 experts, shared last: `fused` is gate then up along N per expert, or up alone when not `gated`.
pub const Experts = struct {
    fused: Projection,
    gated: bool,
    down: Projection,
    /// The expert MLP's inner width (rows of the gate, and of the up).
    width: usize,
    /// The model width (rows of the down).
    dims: usize,
    /// E + 1.
    count: usize,
};

/// The router rows [E + 1, D] with the shared gate last, in bf16 and widened to `rows32`; `remap` is a rank's own ids.
pub const Routed = struct { router: Tensor, rows32: Tensor, experts: Experts, top_k: usize, remap: ?Tensor = null };

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
