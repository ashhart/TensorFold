//! Encode-only BF16 target picks and deterministic verified-tree matching.
const std = @import("std");
const mtl = @import("metal");
pub const contract = @import("tree_round");
const source = @import("tree_round_sources").text;
pub const Ref = struct { buf: mtl.Buffer, off: usize = 0 };
pub const Ops = struct {
    argmax_pipe: mtl.Pipeline,
    match_pipe: mtl.Pipeline,
    pub fn init(device: mtl.Device) !Ops {
        const lib = try mtl.Library.fromSource(device, source, mtl.CompileOptions.mlx());
        defer lib.deinit();
        const argmax_pipe = try mtl.Pipeline.init(device, lib, "tf_round_bf16_argmax", false);
        errdefer argmax_pipe.deinit();
        return .{ .argmax_pipe = argmax_pipe, .match_pipe = try mtl.Pipeline.init(device, lib, "tf_tree_round_match", false) };
    }
    pub fn deinit(ops: Ops) void {
        ops.argmax_pipe.deinit();
        ops.match_pipe.deinit();
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
