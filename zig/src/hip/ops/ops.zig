//! Typed launches of the model-free kernels: shapes checked here, as the kernels trust them; `Ops` holds them.

const std = @import("std");
const abi = @import("../abi.zig");
const Launcher = @import("../launches.zig").Launcher;
const Arena = @import("../arena.zig").Arena;
const types = @import("types.zig");
const attention = @import("attention.zig");
const recurrence = @import("recurrence.zig");
const norms = @import("norms.zig");
const elementwise = @import("elementwise.zig");
const moe = @import("moe.zig");
const draw = @import("draw.zig");

pub const Kind = types.Kind;
pub const Error = types.Error;
pub const Tensor = types.Tensor;

pub const Ops = struct {
    l: *const Launcher,
    /// The activations are bf16 (fp16 where the GPU has no bf16 dot2: RDNA2).
    bf16: bool,
    stream: abi.Stream,
    arena: *Arena,
    /// A prompt's span: products, router and recurrence take prefill's kernel, so rows have the same bits however cut.
    prefill: bool = false,
    /// A lane round's forward: decode tiles at any row count, so a row's bits ignore the rows it shares a round with.
    window: bool = false,

    /// Whether decode.hip's merged launches are on.
    pub fn fused(o: Ops) bool {
        return o.l.fuse;
    }

    pub const attnGate = attention.attnGate;
    pub const ropePrefill = attention.ropePrefill;
    pub const qkRope = attention.qkRope;
    pub const ropeDecode = attention.ropeDecode;
    pub const kvWrite = attention.kvWrite;
    pub const kvWriteAt = attention.kvWriteAt;
    pub const Cache = attention.Cache;
    pub const Paged = attention.Paged;
    pub const pageWrite = attention.pageWrite;
    pub const causalPaged = attention.causalPaged;
    pub const causalPrefill = attention.causalPrefill;
    pub const causalAt = attention.causalAt;
    pub const convPrefill = recurrence.convPrefill;
    pub const convDecode = recurrence.convDecode;
    pub const convRows = recurrence.convRows;
    pub const gdnGatePrefill = recurrence.gdnGatePrefill;
    pub const gdnGate = recurrence.gdnGate;
    pub const gatedDelta = recurrence.gatedDelta;
    pub const rms = norms.rms;
    pub const rms2 = norms.rms2;
    pub const gnormOut = norms.gnormOut;
    pub const addRms = norms.addRms;
    pub const gnormSilu = norms.gnormSilu;
    pub const cast = elementwise.cast;
    pub const siluMul = elementwise.siluMul;
    pub const add = elementwise.add;
    pub const copyCols = elementwise.copyCols;
    pub const moeTail = moe.moeTail;
    pub const moeRouter = moe.moeRouter;
    pub const moeSelect = moe.moeSelect;
    pub const moeRoute = moe.moeRoute;
    pub const moeAct = moe.moeAct;
    pub const moeCombine = moe.moeCombine;
    pub const argmaxRows = draw.argmaxRows;
    pub const topkRows = draw.topkRows;
};

test "dtype numberings of the three kernel families" {
    try std.testing.expectEqual(@as(c_int, 1), @backingInt(Kind.f16));
    try std.testing.expectEqual(@as(c_int, 0), Kind.f16.cache());
}

test "every launch compiles" {
    std.testing.refAllDecls(Ops);
}
