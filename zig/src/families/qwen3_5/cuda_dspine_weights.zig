//! DSpine's code bridge on the device: read, gate, sender and write projections in affine-4, codebook and TG in bf16.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const engine = @import("cuda_engine.zig");
const dw = @import("cuda_draft_weights.zig");
const precision = @import("cuda_draft_precision.zig");
const Config = @import("cuda_draft_config.zig").Config;

const Affine4 = cuda.qlinear.Affine4;
const DeviceBuffer = cuda.DeviceBuffer;
const aligned = engine.aligned;
const prefix = "selected_prefix_bridge.";

pub const Bridge = struct {
    read: [8]Affine4 = undefined, // layer i's read: hidden -> code
    write: [8]Affine4 = undefined, // layer i's feature write: rank -> hidden
    gate: Affine4 = undefined, // code -> rank (the row's own gate term)
    pv: Affine4 = undefined, // code -> [predecessor gate | value] (the sender's terms), 2 x rank
    gate_b: u64 = 0, // gate.bias, rank bf16
    codebook: u64 = 0, // (vocab, code) bf16
    tg: u64 = 0, // (vocab, rank) bf16: W_write[last]^T (final_norm * head row)
    arena: DeviceBuffer,
    arena_at: usize = 0,

    pub fn alloc(b: *Bridge, bytes: usize) !u64 {
        const at = aligned(b.arena_at);
        if (at + bytes > b.arena.len) return error.ArenaExhausted;
        b.arena_at = at + bytes;
        return b.arena.ptr + at;
    }

    pub fn deinit(b: *Bridge) void {
        b.arena.free();
    }
};

/// Device bytes of everything `load` keeps.
pub fn total(c: Config) usize {
    const L = c.layers;
    return L * dw.packedBytes(c.code, c.hidden) + L * dw.packedBytes(c.hidden, c.rank) + dw.packedBytes(c.rank, c.code) +
        dw.packedBytes(2 * c.rank, c.code) + aligned(c.rank * 2) + aligned(c.vocab * c.code * 2) + aligned(c.vocab * c.rank * 2);
}

fn version(ck: *core.Checkpoint, name: []const u8) !void {
    const t = try ck.expect(name, .i64, &.{});
    if (std.mem.readInt(i64, t.bytes[0..8], .little) != 1) return error.UnsupportedDraftConfig;
}

/// The bridge's tensors from `dir`; `transpose` (tf_q35d_transpose) turns the target head's TG rows into the table.
pub fn load(gpa: std.mem.Allocator, io: std.Io, e: *engine.Engine, dir: []const u8, c: Config, transpose: cuda.Function) !Bridge {
    var ck = try core.Checkpoint.openModel(gpa, io, dir);
    defer core.checkpoint.cache.close(&ck);
    for ([_][]const u8{ "draft_block_causal_version", "last_write_refine_version", prefix ++ "independent_write_version", prefix ++ "terminal_disabled_version" }) |n| try version(&ck, n);
    var b: Bridge = .{ .arena = try DeviceBuffer.alloc(e.d, total(c)) };
    errdefer b.arena.free();
    const most = @max(dw.rawBytes(c.code, c.hidden), dw.rawBytes(c.hidden, c.rank), dw.rawBytes(2 * c.rank, c.code));
    var stage = try DeviceBuffer.alloc(e.d, most);
    defer stage.free();
    const L = c.layers;
    const early = try ck.expect(prefix ++ "early_read_weights", .bf16, &.{ L - 1, c.code, c.hidden });
    const last = try ck.expect(prefix ++ "read_weight", .bf16, &.{ c.code, c.hidden });
    const writes = try ck.expect(prefix ++ "feature_write_weights", .bf16, &.{ L, c.hidden, c.rank });
    const one_read = c.code * c.hidden;
    const one_write = c.hidden * c.rank;
    for (0..L) |i| {
        const rows = if (i + 1 < L) std.mem.bytesAsSlice(u16, early.bytes)[i * one_read ..][0..one_read] else std.mem.bytesAsSlice(u16, last.bytes);
        b.read[i] = try dw.linearRows(gpa, e, &stage, &b, rows, c.code, c.hidden);
        b.write[i] = try dw.linearRows(gpa, e, &stage, &b, std.mem.bytesAsSlice(u16, writes.bytes)[i * one_write ..][0..one_write], c.hidden, c.rank);
    }
    const gate = try ck.expect(prefix ++ "gate.weight", .bf16, &.{ c.rank, c.code });
    b.gate = try dw.linearRows(gpa, e, &stage, &b, std.mem.bytesAsSlice(u16, gate.bytes), c.rank, c.code);
    const pv = try gpa.alloc(u16, 2 * c.rank * c.code);
    defer gpa.free(pv);
    for ([_][]const u8{ prefix ++ "predecessor_gate.weight", prefix ++ "prev.weight" }, 0..) |name, i| {
        const t = try ck.expect(name, .bf16, &.{ c.rank, c.code });
        @memcpy(pv[i * c.rank * c.code ..][0 .. c.rank * c.code], std.mem.bytesAsSlice(u16, t.bytes));
    }
    b.pv = try dw.linearRows(gpa, e, &stage, &b, pv, 2 * c.rank, c.code);
    const bias = try ck.expect(prefix ++ "gate.bias", .bf16, &.{c.rank});
    b.gate_b = try b.alloc(bias.bytes.len);
    try b.arena.upload(b.gate_b - b.arena.ptr, bias.bytes);
    const book = try ck.expect(prefix ++ "codebook", .bf16, &.{ c.vocab, c.code });
    b.codebook = try b.alloc(book.bytes.len);
    try b.arena.upload(b.codebook - b.arena.ptr, book.bytes);
    b.tg = try b.alloc(c.vocab * c.rank * 2);
    const norm = try ck.expect("norm.weight", .bf16, &.{c.hidden});
    const wl = std.mem.bytesAsSlice(u16, writes.bytes)[(L - 1) * one_write ..][0..one_write];
    try table(gpa, e, c, std.mem.bytesAsSlice(u16, norm.bytes), wl, b.tg, transpose);
    return b;
}

fn bf16(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}

/// TG[v][k] = head row v . bf16(norm * W[:, k]): the target head over `rank` rows, transposed to vocab rows.
fn table(gpa: std.mem.Allocator, e: *engine.Engine, c: Config, norm: []align(1) const u16, w: []align(1) const u16, out: u64, transpose: cuda.Function) !void {
    const rows = try gpa.alloc(u16, c.rank * c.hidden);
    defer gpa.free(rows);
    for (0..c.rank) |k| for (0..c.hidden) |h| {
        rows[k * c.hidden + h] = core.bf16_round.bits(bf16(norm[h]) * bf16(w[h * c.rank + k]));
    };
    var x = try DeviceBuffer.alloc(e.d, rows.len * 2 + c.rank * (c.hidden / 64) * 4);
    defer x.free();
    try x.upload(0, std.mem.sliceAsBytes(rows));
    var logits = try DeviceBuffer.alloc(e.d, c.rank * c.vocab * 2);
    defer logits.free();
    const xs = x.ptr + rows.len * 2;
    try precision.project(e, e.head, x.ptr, xs, logits.ptr, c.rank, .prompt, precision.dflash_projection_prompt_cutoff);
    try e.run(transpose, .{ (c.vocab + 31) / 32, (c.rank + 31) / 32, 1 }, .{ 32, 8, 1 }, .{ logits.ptr, out, @as(c_int, @intCast(c.rank)), @as(c_int, @intCast(c.vocab)) });
    try e.stream.synchronize();
}
