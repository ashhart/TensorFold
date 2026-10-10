//! An adapter owns real buffers and arithmetic; this protocol checks shapes and keeps preparation claims explicit.
const std = @import("std");
const ck = @import("../checkpoint.zig");
const Config = @import("config.zig").Config;
pub const Mode = enum { bf16_reference, prepared_q4_reference, python_q4g64 };
pub const Preparation = struct {
    mode: Mode = .python_q4g64,
    quantization_verified: bool = false,
    default_proposal_parity: bool = false,
    source_sha256: [32]u8 = @splat(0),
    pub fn check(p: Preparation) !void {
        if (p.mode == .prepared_q4_reference and (!p.quantization_verified or std.mem.allEqual(u8, &p.source_sha256, 0))) return error.DraftPreparationUnqualified;
        if (p.mode == .python_q4g64 and (!p.quantization_verified or !p.default_proposal_parity or std.mem.allEqual(u8, &p.source_sha256, 0))) return error.DraftPreparationUnqualified;
    }
};
pub const View = struct {
    handle: u64,
    rows: u32,
    width: u32,
    stride: u32,
    dtype: ck.DType,
    pub fn check(v: View, rows: u32, width: u32, dtype: ck.DType) !void {
        if (v.handle == 0 or v.rows != rows or v.width != width or v.stride < width or v.dtype != dtype) return error.DraftOperatorShape;
    }
};
pub const Head = struct { logits: View, token_ids: ?View = null };
pub const Lattice = struct {
    gpa: std.mem.Allocator,
    depth: u32,
    topk: u32,
    rank: u32,
    candidates: []u32,
    unary: []f32,
    projected: []f32,
    pub fn check(l: Lattice, c: Config) !void {
        if (l.depth == 0 or l.depth > 15 or l.topk != c.topk or l.rank != c.rank or l.candidates.len != @as(usize, l.depth) * l.topk or l.unary.len != l.candidates.len or l.projected.len != @as(usize, l.depth) * l.rank) return error.DraftLatticeShape;
        for (l.candidates) |id| if (id >= c.vocab) return error.DraftCandidateId;
        for (l.unary) |v| if (!std.math.isFinite(v)) return error.DraftLatticeFinite;
        for (l.projected) |v| if (!std.math.isFinite(v)) return error.DraftLatticeFinite;
        for (0..l.depth) |d| for (0..l.topk) |i| for (0..i) |j| if (l.candidates[d * l.topk + i] == l.candidates[d * l.topk + j]) return error.DraftCandidateId;
    }
    pub fn deinit(l: *Lattice) void {
        l.gpa.free(l.candidates);
        l.gpa.free(l.unary);
        l.gpa.free(l.projected);
        l.* = undefined;
    }
};
pub const Context = struct { cache: u64, begin: u64, end: u64 };
pub const Norm = struct { weight: ck.Tensor, eps: f32, heads: u32 = 1, head_dim: u32 };
pub const Rotary = struct { positions: []const u64, heads: u32, dim: u32, theta: f64 };
pub const Conv = struct { dynamic: View, base: ck.Tensor, branch: u32, group: u32, block: u32, residual: ?View = null };
pub const BlockMask = enum { all_slots };
pub const Attention = struct { context: Context, config: Config, layer: u32, query: View, keys: View, values: View, positions: []const u64, mask: BlockMask = .all_slots };
pub const Ops = struct {
    ptr: *anyopaque,
    vtable: *const VTable,
    pub const VTable = struct {
        begin: *const fn (*anyopaque) anyerror!u64,
        release: *const fn (*anyopaque, u64) void,
        project: *const fn (*anyopaque, u64, View, ck.Tensor, Mode) anyerror!View,
        norm: *const fn (*anyopaque, u64, View, Norm) anyerror!View,
        embed: *const fn (*anyopaque, u64, ck.Linear, []const u32) anyerror!View,
        conv: *const fn (*anyopaque, u64, View, Conv) anyerror!View,
        rope: *const fn (*anyopaque, u64, View, Rotary) anyerror!View,
        attention: *const fn (*anyopaque, u64, Attention) anyerror!View,
        swiglu: *const fn (*anyopaque, u64, View, View) anyerror!View,
        mask_rows: *const fn (*anyopaque, u64, View, u32, u32) anyerror!View,
        head: *const fn (*anyopaque, u64, ck.Linear, View) anyerror!Head,
        // Readback owns its arrays beyond frame release; top-k token ids include any head-subset remapping.
        lattice: *const fn (*anyopaque, u64, Head, View, Config) anyerror!Lattice,
        context_begin: *const fn (*anyopaque, Context) anyerror!u64,
        context_append: *const fn (*anyopaque, u64, u64, u32, View, View, []const u64) anyerror!void,
        context_commit: *const fn (*anyopaque, u64, u64, Context) anyerror!void,
        // context_commit without finishing the frame: the frame's later ops read the appended rows behind their fences.
        context_stage: *const fn (*anyopaque, u64, u64, Context) anyerror!void,
        context_abort: *const fn (*anyopaque, u64) void,
    };
};

test "unprepared default cannot claim parity while explicit BF16 reference remains possible" {
    try std.testing.expectError(error.DraftPreparationUnqualified, (Preparation{}).check());
    try (Preparation{ .mode = .bf16_reference }).check();
    try (Preparation{ .quantization_verified = true, .default_proposal_parity = true, .source_sha256 = @splat(1) }).check();
}
test "operator dtype and physical width cannot be silently cast" {
    const v = View{ .handle = 1, .rows = 8, .width = 5120, .stride = 5120, .dtype = .bf16 };
    try v.check(8, 5120, .bf16);
    try std.testing.expectError(error.DraftOperatorShape, v.check(8, 5120, .f32));
    try std.testing.expectError(error.DraftOperatorShape, v.check(8, 25600, .bf16));
}
