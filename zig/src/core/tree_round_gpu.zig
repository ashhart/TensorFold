//! Encode-only BF16 target picks and deterministic verified-tree matching.
const std = @import("std");
const mtl = @import("metal");
pub const contract = @import("tree_round");
const source = @import("tree_round_sources").text;
pub const Ref = struct { buf: mtl.Buffer, off: usize = 0 };
/// `Ops.argmaxChunks` takes a row's argmax over chunks of its vocabulary, on every core; off, it is `argmax` (checks).
pub var chunked_picks = true;
const chunk_tokens = 1024;
const max_chunks = 512;
const chunk_best_bytes = 12;
pub const Ops = struct {
    argmax_pipe: mtl.Pipeline,
    match_pipe: mtl.Pipeline,
    chunks_pipe: mtl.Pipeline,
    rows_pipe: mtl.Pipeline,
    chunk_best: mtl.Buffer,
    pub fn init(device: mtl.Device) !Ops {
        const lib = try mtl.Library.fromSource(device, source, mtl.CompileOptions.mlx());
        defer lib.deinit();
        const argmax_pipe = try mtl.Pipeline.init(device, lib, "tf_round_bf16_argmax", false);
        errdefer argmax_pipe.deinit();
        const match_pipe = try mtl.Pipeline.init(device, lib, "tf_tree_round_match", false);
        errdefer match_pipe.deinit();
        const chunks_pipe = try mtl.Pipeline.init(device, lib, "tf_round_pick_chunks", false);
        errdefer chunks_pipe.deinit();
        const rows_pipe = try mtl.Pipeline.init(device, lib, "tf_round_pick_rows", false);
        errdefer rows_pipe.deinit();
        const bytes = contract.max_rows * max_chunks * chunk_best_bytes;
        const chunk_best = try device.buffer(bytes, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        return .{ .argmax_pipe = argmax_pipe, .match_pipe = match_pipe, .chunks_pipe = chunks_pipe, .rows_pipe = rows_pipe, .chunk_best = chunk_best };
    }
    pub fn deinit(ops: Ops) void {
        ops.argmax_pipe.deinit();
        ops.match_pipe.deinit();
        ops.chunks_pipe.deinit();
        ops.rows_pipe.deinit();
        ops.chunk_best.deinit();
    }
    /// The same picks as `argmax`, through a scratch buffer: the caller keeps one call at a time on an encoder.
    pub fn argmaxChunks(ops: Ops, e: mtl.ComputeEncoder, logits: Ref, out: Ref, p: contract.Argmax) !void {
        const chunks = (p.vocab + chunk_tokens - 1) / chunk_tokens;
        if (!chunked_picks or chunks > max_chunks or chunks < 8) return ops.argmax(e, logits, out, p);
        try contract.checkArgmax(p);
        const cells = try std.math.add(usize, try std.math.mul(usize, p.rows - 1, p.stride), p.vocab);
        const bytes = try std.math.mul(usize, cells, 2);
        try check(logits, bytes, 2);
        try check(out, @as(usize, p.rows) * @sizeOf(contract.Pick), 4);
        try distinct(out, @as(usize, p.rows) * @sizeOf(contract.Pick), logits, bytes);
        e.setPipeline(ops.chunks_pipe);
        e.setBuffer(logits.buf, logits.off, 0);
        e.setBuffer(ops.chunk_best, 0, 1);
        e.setValue(p, 2);
        e.dispatchGroups(mtl.Size.of(chunks, p.rows, 1), mtl.Size.of(32, 1, 1));
        e.barrier();
        e.setPipeline(ops.rows_pipe);
        e.setBuffer(ops.chunk_best, 0, 0);
        e.setBuffer(out.buf, out.off, 1);
        e.setValue(p, 2);
        e.dispatchGroups(mtl.Size.of(p.rows, 1, 1), mtl.Size.of(32, 1, 1));
    }
    pub fn argmax(ops: Ops, e: mtl.ComputeEncoder, logits: Ref, picks: Ref, p: contract.Argmax) !void {
        try contract.checkArgmax(p);
        const cells = try std.math.add(usize, try std.math.mul(usize, p.rows - 1, p.stride), p.vocab);
        const bytes = try std.math.mul(usize, cells, 2);
        try check(logits, bytes, 2);
        try check(picks, @as(usize, p.rows) * @sizeOf(contract.Pick), 4);
        try distinct(picks, @as(usize, p.rows) * @sizeOf(contract.Pick), logits, bytes);
        e.setPipeline(ops.argmax_pipe);
        e.setBuffer(logits.buf, logits.off, 0);
        e.setBuffer(picks.buf, picks.off, 1);
        e.setValue(p, 2);
        e.dispatchGroups(mtl.Size.of(p.rows, 1, 1), mtl.Size.of(256, 1, 1));
    }
    pub fn match(ops: Ops, e: mtl.ComputeEncoder, tokens: Ref, parents: Ref, picks: Ref, eos: ?Ref, result: Ref, p: contract.Match) !void {
        try contract.checkMatch(p);
        if ((eos == null) != (p.eos_count == 0)) return error.BadRoundShape;
        const rows = @as(usize, p.rows);
        try check(result, @sizeOf(contract.Result), 4);
        for ([_]Ref{ tokens, parents, picks }, [_]usize{ rows * 4, rows * 4, rows * @sizeOf(contract.Pick) }) |ref, bytes| {
            try check(ref, bytes, 4);
            try distinct(result, @sizeOf(contract.Result), ref, bytes);
        }
        if (eos) |ref| {
            try check(ref, @as(usize, p.eos_count) * 4, 4);
            try distinct(result, @sizeOf(contract.Result), ref, @as(usize, p.eos_count) * 4);
        }
        e.setPipeline(ops.match_pipe);
        for ([_]Ref{ tokens, parents, picks, eos orelse tokens, result }, 0..) |ref, i| e.setBuffer(ref.buf, ref.off, i);
        e.setValue(p, 5);
        e.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(1, 1, 1));
    }
};
fn check(ref: Ref, bytes: usize, alignment: usize) !void {
    if (ref.off % alignment != 0 or ref.off > ref.buf.length() or bytes > ref.buf.length() - ref.off) return error.BadRoundBuffer;
}
fn distinct(out: Ref, bytes: usize, input: Ref, input_bytes: usize) !void {
    if (out.buf.id == input.buf.id and out.off < input.off + input_bytes and input.off < out.off + bytes) return error.RoundAlias;
}
