//! The joint schema head on the GPU: its weights, the shared scratch, and the device and library it runs through.
const std = @import("std");
const js = @import("core").joint_schema;
const runtime = @import("../runtime.zig");
const DeviceBuffer = @import("../memory.zig").DeviceBuffer;
const Stream = @import("../stream.zig").Stream;
const Launcher = @import("../launches.zig").Launcher;
const Blas = @import("../blas.zig").Blas;
const Fns = @import("launch.zig").Fns;
const weights = @import("weights.zig");
const plan = @import("core").joint_schema.plan;

pub const Head = struct {
    gpa: std.mem.Allocator,
    r: *const runtime.Runtime,
    l: *const Launcher,
    blas: *const Blas,
    fns: Fns,
    cfg: js.Config,
    w: weights.Weights,
    bounds: plan.Bounds,
    lane: plan.Lane,
    layout: plan.Scratch,
    scratch: DeviceBuffer,
    /// The backbone's lm_head rows (BF16, vocab x hidden): the options' lexical rows.
    table: u64,
    vocab: usize,

    /// The head over uploaded weights (it owns them now), with its scratch: `plan.bytes(cfg, bounds, 0)` bytes.
    pub fn init(gpa: std.mem.Allocator, r: *const runtime.Runtime, l: *const Launcher, blas: *const Blas, w: weights.Weights, bounds: plan.Bounds, table: u64, vocab: usize) !*Head {
        const cfg = w.w.cfg;
        if (cfg.width % cfg.heads != 0 or bounds.rows == 0 or bounds.questions == 0 or bounds.options == 0 or bounds.piece == 0) return error.Invalid;
        const h = try gpa.create(Head);
        errdefer gpa.destroy(h);
        h.* = .{ .gpa = gpa, .r = r, .l = l, .blas = blas, .fns = try Fns.load(l), .cfg = cfg, .w = w, .bounds = bounds, .lane = plan.Lane.of(cfg, bounds), .layout = plan.Scratch.of(cfg, bounds), .scratch = undefined, .table = table, .vocab = vocab };
        h.scratch = try DeviceBuffer.alloc(r, h.layout.total);
        return h;
    }

    pub fn deinit(h: *Head) void {
        h.scratch.free();
        h.w.deinit();
        h.gpa.destroy(h);
    }

    /// A scratch region's device address.
    pub fn at(h: *const Head, offset: usize) u64 {
        return h.scratch.base() + offset;
    }

    /// x rows times w transposed, plus the bias broadcast first so it adds inside the one BF16 rounding.
    pub fn linear(h: *const Head, s: Stream, x: u64, rows: usize, w: u64, bias: ?u64, out: u64, n: usize, k: usize) !void {
        if (rows == 0) return;
        if (bias) |b| {
            try h.l.tf_copy_cols(@ptrFromInt(b), 0, 0, @ptrFromInt(out), 2, @intCast(rows), @intCast(n), s.handle);
            return h.blas.projectAdd(x, w, out, rows, n, k);
        }
        return h.blas.project(x, w, out, rows, n, k);
    }

    /// out = bf16(x + y) over n values (a residual add).
    pub fn add(h: *const Head, s: Stream, x: u64, y: u64, out: u64, n: usize) !void {
        try h.l.tf_add(@ptrFromInt(x), @ptrFromInt(y), @ptrFromInt(out), 2, @intCast(n), s.handle);
    }

    /// Rows of `n` BF16 values from one source row, as many as `rows` (stride 0).
    pub fn broadcast(h: *const Head, s: Stream, row: u64, out: u64, rows: usize, n: usize) !void {
        if (rows == 0) return;
        try h.l.tf_copy_cols(@ptrFromInt(row), 0, 0, @ptrFromInt(out), 2, @intCast(rows), @intCast(n), s.handle);
    }

    pub fn norm(h: *const Head, s: Stream, x: u64, n: js.tensors.NormOf(u64), y: u64, rows: usize, dim: usize) !void {
        try h.fns.layerNorm(s, x, n.weight, n.bias, y, rows, dim, 1e-5);
    }
};
