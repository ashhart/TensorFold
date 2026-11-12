//! The joint schema head on the GPU: its weights, the shared scratch, and the kernels and products it runs through.
const std = @import("std");
const js = @import("core").joint_schema;
const Driver = @import("cuda").Driver;
const DeviceBuffer = @import("cuda").DeviceBuffer;
const Stream = @import("cuda").Stream;
const Fns = @import("launch.zig").Fns;
const Blas = @import("blas.zig").Blas;
const weights = @import("weights.zig");
const plan = js.plan;

pub const Head = struct {
    gpa: std.mem.Allocator,
    d: *const Driver,
    blas: Blas,
    fns: Fns,
    cfg: js.Config,
    w: weights.Weights,
    bounds: plan.Bounds,
    lane: plan.Lane,
    layout: plan.Scratch,
    scratch: DeviceBuffer,
    /// The backbone's lm_head as the qmmf BF16 lane packs it (vocab x hidden): the options' lexical rows.
    table: u64,
    vocab: usize,

    /// The head over uploaded weights (it owns them now), with its scratch: `plan.bytes(cfg, bounds, 0)` bytes.
    pub fn init(gpa: std.mem.Allocator, d: *const Driver, w: weights.Weights, bounds: plan.Bounds, table: u64, vocab: usize) !*Head {
        const cfg = w.w.cfg;
        if (cfg.width % cfg.heads != 0 or cfg.hidden_size % 64 != 0 or bounds.rows == 0 or bounds.questions == 0 or bounds.options == 0 or bounds.piece == 0) return error.Invalid;
        const h = try gpa.create(Head);
        errdefer gpa.destroy(h);
        var fns = try Fns.load(d);
        errdefer fns.unload();
        h.* = .{ .gpa = gpa, .d = d, .blas = try Blas.init(gpa, d), .fns = fns, .cfg = cfg, .w = w, .bounds = bounds, .lane = plan.Lane.of(cfg, bounds), .layout = plan.Scratch.of(cfg, bounds), .scratch = undefined, .table = table, .vocab = vocab };
        errdefer h.blas.deinit();
        h.scratch = try DeviceBuffer.alloc(d, h.layout.total);
        return h;
    }

    pub fn deinit(h: *Head) void {
        h.scratch.free();
        h.blas.deinit();
        h.fns.unload();
        h.w.deinit();
        h.gpa.destroy(h);
    }

    /// A scratch region's device address.
    pub fn at(h: *const Head, offset: usize) u64 {
        return h.scratch.ptr + offset;
    }

    /// x rows times w transposed, plus the bias broadcast first so it adds inside the one BF16 rounding.
    pub fn linear(h: *Head, s: Stream, x: u64, rows: usize, w: u64, bias: ?u64, out: u64, n: usize, k: usize) !void {
        if (rows == 0) return;
        if (bias) |b| {
            try h.fns.broadcast(s, b, out, rows, n);
            return h.blas.projectAdd(s, x, w, out, rows, n, k);
        }
        return h.blas.project(s, x, w, out, rows, n, k);
    }

    /// out = bf16(x + y) over n values (a residual add).
    pub fn add(h: *const Head, s: Stream, x: u64, y: u64, out: u64, n: usize) !void {
        try h.fns.add(s, x, y, out, n);
    }

    /// Rows of `n` BF16 values from one source row, as many as `rows`.
    pub fn broadcast(h: *const Head, s: Stream, row: u64, out: u64, rows: usize, n: usize) !void {
        try h.fns.broadcast(s, row, out, rows, n);
    }

    pub fn norm(h: *const Head, s: Stream, x: u64, n: js.tensors.NormOf(u64), y: u64, rows: usize, dim: usize) !void {
        try h.fns.layerNorm(s, x, n.weight, n.bias, y, rows, dim, 1e-5);
    }
};
