//! Resident Engram FP8 gathers and signed-sqrt gates; WKV is projected separately by the caller.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
pub const View = @import("view.zig").View;
const Dtype = @import("ops.zig").Dtype;
pub const hash = @import("engram_hash.zig");

pub const head_dim = 256;
pub const gather_dim = hash.columns * head_dim;
pub const history_bytes = 16;

/// ROUND-527: engram_exl3.metal's `EngramHistory`, field for field: three token slots and the count.
pub const History = extern struct { tail: [3]i32, available: u32 };
comptime {
    if (@sizeOf(History) != history_bytes) @compileError("engram.History must stay 16 bytes");
}

/// Borrowed full-resident E4M3FN rows and UE8M0 rowwise scale blocks, never a 32x32 scale matrix.
pub const Table = struct {
    weight: View,
    scale: View,
    rows: u32,
    weight_stride_bytes: usize = head_dim,
    scale_stride_bytes: usize = head_dim / 32,
    scale_block: u32 = 32,

    pub fn validate(self: Table) !void {
        if (self.rows == 0 or self.weight_stride_bytes < head_dim or self.scale_stride_bytes < head_dim / 32 or self.scale_block != 32) return error.InvalidEngramTable;
        try self.weight.requireRows(self.rows, self.weight_stride_bytes, head_dim, 1);
        try self.scale.requireRows(self.rows, self.scale_stride_bytes, head_dim / 32, 1);
    }
};

/// BF16 hidden [tokens,copies,dim], WKV [tokens,(copies+1)*dim]; norm weights contiguous FP32 [copies,dim].
pub const Gate = struct {
    q_weight: View,
    k_weight: View,
    /// The released checkpoints store these f32; rank-split repacks store bf16.
    weight_dtype: Dtype = .f32,
    dim: u32,
    copies: u32,
    hidden_stride: usize,
    kv_stride: usize,
    output_stride: usize,
    eps: f32,

    pub fn validate(self: Gate) !void {
        const width = try std.math.mul(usize, self.dim, self.copies);
        const projected_width = try std.math.add(usize, width, self.dim);
        if (self.dim == 0 or self.copies == 0 or self.hidden_stride < width or self.output_stride < width or self.kv_stride < projected_width or !std.math.isFinite(self.eps) or self.eps <= 0 or
            self.dim > std.math.maxInt(u32) - 255) return error.InvalidEngramGate;
        const weight_bytes = try std.math.mul(usize, width, 4);
        try self.q_weight.require(weight_bytes, 4);
        try self.k_weight.require(weight_bytes, 4);
    }
};

pub const Pipelines = struct {
    hash_ids: mtl.Pipeline,
    commit: mtl.Pipeline,
    reset: mtl.Pipeline,
    gather: mtl.Pipeline,
    gate: mtl.Pipeline,

    pub fn load(device: mtl.Device) !Pipelines {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const lib = try mtl.Library.fromSource(device, sources.deepseek_v41_engram, mtl.CompileOptions.mlx());
        defer lib.deinit();
        const h = try mtl.Pipeline.init(device, lib, "tf_v41_engram_hash", false);
        errdefer h.deinit();
        const c = try mtl.Pipeline.init(device, lib, "tf_v41_engram_commit", false);
        errdefer c.deinit();
        const r = try mtl.Pipeline.init(device, lib, "tf_v41_engram_reset", false);
        errdefer r.deinit();
        const g = try mtl.Pipeline.init(device, lib, "tf_v41_engram_gather", false);
        errdefer g.deinit();
        const gate = try mtl.Pipeline.init(device, lib, "tf_v41_engram_gate", false);
        return .{ .hash_ids = h, .commit = c, .reset = r, .gather = g, .gate = gate };
    }

    pub fn deinit(self: *Pipelines) void {
        self.hash_ids.deinit();
        self.commit.deinit();
        self.reset.deinit();
        self.gather.deinit();
        self.gate.deinit();
    }
};

pub const HashArgs = extern struct { tokens: u32, pad: u32, masked: u32 };
pub const GatherArgs = extern struct {
    weight_stride: u64,
    scale_stride: u64,
    output_stride: u64,
    tokens: u32,
    columns: u32,
    head_dim: u32,
    scale_block: u32,
    comptime {
        @import("metal/index_mirror.zig").check(GatherArgs, sources.deepseek_v41_engram, "EngramGatherArgs");
    }
};
pub const GateArgs = extern struct {
    hidden_stride: u64,
    kv_stride: u64,
    output_stride: u64,
    dim: u32,
    copies: u32,
    masked: u32,
    weight_dtype: u32,
    eps: f32,
};

fn bind(e: mtl.ComputeEncoder, view: View, at: usize) void {
    e.setBuffer(view.buffer, view.offset, at);
}

/// Each history View is one sequence's 16-byte GPU record; initialize before its first hash.
pub fn encodeReset(e: mtl.ComputeEncoder, p: *const Pipelines, history: View) void {
    e.setPipeline(p.reset);
    bind(e, history, 0);
    e.dispatchGroups(.{ .width = 1 }, .{ .width = 1 });
    e.barrier();
}

/// Snapshot/rollback are GPU copies; caller ends compute encoding before beginning a blit encoder.
pub fn encodeSnapshot(e: mtl.BlitEncoder, history: View, saved: View) void {
    e.copy(history.buffer, history.offset, saved.buffer, saved.offset, history_bytes);
}

pub fn encodeRollback(e: mtl.BlitEncoder, saved: View, history: View) void {
    encodeSnapshot(e, saved, history);
}

/// Raw u32 IDs -> [tokens,24] hashes; map covers valid IDs, optional 0=DEAD mask, history is unchanged.
pub fn encodeHash(e: mtl.ComputeEncoder, p: *const Pipelines, meta: hash.Layer, ids: View, token_map: View, mask: ?View, history: View, out: View, tokens: u32) !void {
    if (tokens == 0) return error.EmptyEngramChunk;
    if (!std.mem.eql(u8, std.mem.asBytes(&meta), std.mem.asBytes(&hash.layers[0])) and
        !std.mem.eql(u8, std.mem.asBytes(&meta), std.mem.asBytes(&hash.layers[1]))) return error.InvalidEngramMetadata;
    try ids.requireRows(tokens, 1, 1, 4);
    try token_map.require(4, 4);
    try history.require(history_bytes, 4);
    try out.requireRows(tokens, hash.columns, hash.columns, 4);
    if (mask) |m| try m.require(tokens, 1);
    e.setPipeline(p.hash_ids);
    bind(e, ids, 0);
    bind(e, token_map, 1);
    bind(e, mask orelse ids, 2);
    bind(e, history, 3);
    bind(e, out, 4);
    e.setValue(meta, 5);
    e.setValue(HashArgs{ .tokens = tokens, .pad = hash.pad_id, .masked = @intFromBool(mask != null) }, 6);
    e.dispatchThreads(.{ .width = tokens }, .{ .width = 64 });
    e.barrier();
}

/// Commit exactly once per chunk after both layers have hashed; remains bounded at three past tokens.
pub fn encodeCommit(e: mtl.ComputeEncoder, p: *const Pipelines, ids: View, token_map: View, mask: ?View, history: View, tokens: u32) !void {
    if (tokens == 0) return error.EmptyEngramChunk;
    try ids.requireRows(tokens, 1, 1, 4);
    try token_map.require(4, 4);
    try history.require(history_bytes, 4);
    if (mask) |m| try m.require(tokens, 1);
    e.barrier();
    e.setPipeline(p.commit);
    bind(e, ids, 0);
    bind(e, token_map, 1);
    bind(e, mask orelse ids, 2);
    bind(e, history, 3);
    e.setValue(HashArgs{ .tokens = tokens, .pad = hash.pad_id, .masked = @intFromBool(mask != null) }, 4);
    e.dispatchGroups(.{ .width = 1 }, .{ .width = 1 });
    e.barrier();
}

/// ROUND-488: the host-side hash pass. The same integer hash and the same bounded history as the
/// `hash_ids`/`commit` kernels, computed where the ids are consumed: a decode-sized window no longer
/// pays a command buffer + wait before its rows can be staged. The 16-byte history record keeps its
/// shipped layout (`int tail[3]; uint available`), so snapshot, rollback and the cache receipt stand
/// on the same bytes. `TF_DSV41_CPU_HASH=0` restores the GPU pass (the control).
pub fn encodeHashHost(history: View, ids: View, token_map: View, out_0: View, out_1: View, tokens: u32) !void {
    if (tokens == 0) return error.EmptyEngramChunk;
    try ids.requireRows(tokens, 1, 1, 4);
    try token_map.require(hash.raw_vocab * 4, 4);
    try history.require(history_bytes, 4);
    const hb: [*]u8 = @ptrCast(@alignCast(history.buffer.contents() + history.offset));
    const tail: [*]const i32 = @ptrCast(@alignCast(hb));
    const available = std.mem.readInt(u32, hb[12..16], .little);
    if (available > 3) return error.InvalidEngramMetadata;
    // The GPU hashes both layers before its single commit, so both layers see the SAME pre-chunk
    // history; each advances its own copy and the record takes the chunk's single advance.
    const start = hash.History{ .tail = .{ tail[0], tail[1], tail[2] }, .position = available };
    const idp: [*]const u32 = @ptrCast(@alignCast(ids.buffer.contents() + ids.offset));
    const map = hash.TokenMap{ .bytes = (token_map.buffer.contents() + token_map.offset)[0 .. hash.raw_vocab * 4] };
    var advanced = start;
    for ([_]View{ out_0, out_1 }, hash.layers) |out, meta| {
        try out.requireRows(tokens, hash.columns, hash.columns, 4);
        const op: [*]u32 = @ptrCast(@alignCast(out.buffer.contents() + out.offset));
        const rows: [*][hash.columns]u32 = @ptrCast(op);
        var state = start;
        try hash.hashChunk(&state, map, meta, idp[0..tokens], null, rows[0..tokens]);
        advanced = state;
    }
    for (0..3) |i| std.mem.writeInt(i32, hb[i * 4 ..][0..4], advanced.tail[i], .little);
    std.mem.writeInt(u32, hb[12..16], @intCast(@min(3, advanced.position)), .little);
}

/// ROUND-493: the ids a one-token host hash would produce from the record AS IT STANDS, without advancing
/// it. `proposeTokens` knows the pending token before its own forward runs; the next verify's hash pass is
/// what consumes that row, so warming its pages now overlaps the page-in with the drafter's GPU window.
pub fn peekHashHost(history: View, token_map: View, pending: u32, out: *[2][hash.columns]u32) !void {
    try token_map.require(hash.raw_vocab * 4, 4);
    try history.require(history_bytes, 4);
    const hb: [*]u8 = @ptrCast(@alignCast(history.buffer.contents() + history.offset));
    const tail: [*]const i32 = @ptrCast(@alignCast(hb));
    const available = std.mem.readInt(u32, hb[12..16], .little);
    if (available > 3) return error.InvalidEngramMetadata;
    const start = hash.History{ .tail = .{ tail[0], tail[1], tail[2] }, .position = available };
    const map = hash.TokenMap{ .bytes = (token_map.buffer.contents() + token_map.offset)[0 .. hash.raw_vocab * 4] };
    for ([_]*[hash.columns]u32{ &out[0], &out[1] }, hash.layers) |dst, meta| {
        var state = start;
        var rows: [1][hash.columns]u32 = undefined;
        const tok = [1]u32{pending};
        try hash.hashChunk(&state, map, meta, &tok, null, &rows);
        dst.* = rows[0];
    }
}

/// Gather uint32 [tokens,24] row IDs into BF16 [tokens,6144]; IDs must be within the borrowed table.
pub fn encodeGather(e: mtl.ComputeEncoder, p: *const Pipelines, table: Table, ids: View, out: View, tokens: u32, output_stride: usize) !void {
    try table.validate();
    if (tokens == 0 or output_stride < gather_dim) return error.InvalidEngramGather;
    try ids.requireRows(tokens, hash.columns, hash.columns, 4);
    try out.requireRows(tokens, output_stride, gather_dim, 2);
    e.setPipeline(p.gather);
    bind(e, ids, 0);
    bind(e, table.weight, 1);
    bind(e, table.scale, 2);
    bind(e, out, 3);
    e.setValue(GatherArgs{ .weight_stride = table.weight_stride_bytes, .scale_stride = table.scale_stride_bytes, .output_stride = output_stride, .tokens = tokens, .columns = hash.columns, .head_dim = head_dim, .scale_block = table.scale_block }, 4);
    e.dispatchThreads(.{ .width = gather_dim, .height = tokens }, .{ .width = 256 });
    e.barrier();
}

/// Add per-copy gate times shared value to BF16 hidden; out may alias hidden, never the projected KV.
pub fn encodeGate(e: mtl.ComputeEncoder, p: *const Pipelines, gate: Gate, hidden: View, projected_kv: View, mask: ?View, out: View, tokens: u32) !void {
    try gate.validate();
    if (tokens == 0) return error.EmptyEngramChunk;
    const width = @as(usize, gate.dim) * gate.copies;
    try hidden.requireRows(tokens, gate.hidden_stride, width, 2);
    try projected_kv.requireRows(tokens, gate.kv_stride, width + gate.dim, 2);
    try out.requireRows(tokens, gate.output_stride, width, 2);
    if (mask) |m| try m.require(tokens, 1);
    e.setPipeline(p.gate);
    bind(e, hidden, 0);
    bind(e, projected_kv, 1);
    bind(e, gate.q_weight, 2);
    bind(e, gate.k_weight, 3);
    bind(e, mask orelse hidden, 4);
    bind(e, out, 5);
    e.setValue(GateArgs{ .hidden_stride = gate.hidden_stride, .kv_stride = gate.kv_stride, .output_stride = gate.output_stride, .dim = gate.dim, .copies = gate.copies, .masked = @intFromBool(mask != null), .weight_dtype = @backingInt(gate.weight_dtype), .eps = gate.eps }, 6);
    e.dispatchGroups(.{ .width = tokens, .height = gate.copies }, .{ .width = 256 });
    e.barrier();
}

test {
    _ = @import("engram_test.zig");
}
