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
const prefix = core.dspine.bridge_prefix;

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

/// A tensor as core.dspine's table states it: its dtype and its dims for `c`.
fn tensor(ck: *core.Checkpoint, c: Config, name: []const u8) !core.checkpoint.Tensor {
    const want = core.dspine.find(name, c.layers) orelse return error.UnsupportedDraftConfig;
    var dims: [3]usize = undefined;
    return ck.expect(name, switch (want.kind) {
        .bf16 => .bf16,
        .i64 => .i64,
    }, core.dspine.dims(c, want, &dims));
}

fn version(ck: *core.Checkpoint, c: Config, name: []const u8) !void {
    const t = try tensor(ck, c, name);
    if (std.mem.readInt(i64, t.bytes[0..8], .little) != 1) return error.UnsupportedDraftConfig;
}

/// The bridge's tensors from `dir`; `transpose` (tf_q35d_transpose) turns the target head's TG rows into the table.
pub fn load(gpa: std.mem.Allocator, io: std.Io, e: *engine.Engine, dir: []const u8, c: Config, transpose: cuda.Function) !Bridge {
    var ck = try core.Checkpoint.openModel(gpa, io, dir);
    defer core.checkpoint.cache.close(&ck);
    inline for (core.dspine.globals ++ core.dspine.bridge_tensors, 0..) |t, i| if (t.kind == .i64) try version(&ck, c, if (i < core.dspine.globals.len) t.name else prefix ++ t.name);
    var b: Bridge = .{ .arena = try DeviceBuffer.alloc(e.d, total(c)) };
    errdefer b.arena.free();
    const most = @max(dw.rawBytes(c.code, c.hidden), dw.rawBytes(c.hidden, c.rank), dw.rawBytes(2 * c.rank, c.code));
    var stage = try DeviceBuffer.alloc(e.d, most);
    defer stage.free();
    const L = c.layers;
    const early = try tensor(&ck, c, prefix ++ "early_read_weights");
    const last = try tensor(&ck, c, prefix ++ "read_weight");
    const writes = try tensor(&ck, c, prefix ++ "feature_write_weights");
    const one_read = c.code * c.hidden;
    const one_write = c.hidden * c.rank;
    for (0..L) |i| {
        const rows = if (i + 1 < L) std.mem.bytesAsSlice(u16, early.bytes)[i * one_read ..][0..one_read] else std.mem.bytesAsSlice(u16, last.bytes);
        b.read[i] = try dw.linearRows(gpa, e, &stage, &b, rows, c.code, c.hidden);
        b.write[i] = try dw.linearRows(gpa, e, &stage, &b, std.mem.bytesAsSlice(u16, writes.bytes)[i * one_write ..][0..one_write], c.hidden, c.rank);
    }
    const gate = try tensor(&ck, c, prefix ++ "gate.weight");
    b.gate = try dw.linearRows(gpa, e, &stage, &b, std.mem.bytesAsSlice(u16, gate.bytes), c.rank, c.code);
    const pv = try gpa.alloc(u16, 2 * c.rank * c.code);
    defer gpa.free(pv);
    for ([_][]const u8{ prefix ++ "predecessor_gate.weight", prefix ++ "prev.weight" }, 0..) |name, i| {
        const t = try tensor(&ck, c, name);
        @memcpy(pv[i * c.rank * c.code ..][0 .. c.rank * c.code], std.mem.bytesAsSlice(u16, t.bytes));
    }
    b.pv = try dw.linearRows(gpa, e, &stage, &b, pv, 2 * c.rank, c.code);
    const bias = try tensor(&ck, c, prefix ++ "gate.bias");
    b.gate_b = try b.alloc(bias.bytes.len);
    try b.arena.upload(b.gate_b - b.arena.ptr, bias.bytes);
    const book = try tensor(&ck, c, prefix ++ "codebook");
    b.codebook = try b.alloc(book.bytes.len);
    try b.arena.upload(b.codebook - b.arena.ptr, book.bytes);
    b.tg = try b.alloc(c.vocab * c.rank * 2);
    const norm = try tensor(&ck, c, "norm.weight");
    const wl = std.mem.bytesAsSlice(u16, writes.bytes)[(L - 1) * one_write ..][0..one_write];
    try table(gpa, e, c, std.mem.bytesAsSlice(u16, norm.bytes), wl, b.tg, transpose);
    return b;
}

/// TG[v][k] = head row v . bf16(norm * W[:, k]): the target head over `rank` rows, transposed to vocab rows.
fn table(gpa: std.mem.Allocator, e: *engine.Engine, c: Config, norm: []align(1) const u16, w: []align(1) const u16, out: u64, transpose: cuda.Function) !void {
    const rows = try gpa.alloc(u16, c.rank * c.hidden);
    defer gpa.free(rows);
    for (0..c.rank) |k| core.dspine.tgInput(norm, w, c.rank, k, rows[k * c.hidden ..][0..c.hidden]);
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
