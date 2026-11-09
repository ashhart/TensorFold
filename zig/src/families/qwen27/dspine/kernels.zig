//! DSpine's Metal kernels (kernels/metal/qwen27/dspine.metal): bridge read, code, message and write, senders, table.
const std = @import("std");
const mtl = @import("metal");
pub const Ref = @import("core").draft_ops.Ref;
const source = @import("kernel_sources").qwen27_dspine;

pub const Rows = extern struct { rows: u32, width: u32, block: u32, eps: f32 = 0, scale: f32 = 0 };
pub const Senders = extern struct { rows: u32, k: u32, code: u32, row_words: u32, anchor: u32, scale: f32 };
pub const Table = extern struct { rows: u32, k: u32, rank: u32, row_words: u32, cols: u32, rms_first: u32 };
const names = [_][]const u8{ "tf_dspine_add", "tf_dspine_read", "tf_dspine_code", "tf_dspine_message", "tf_dspine_write", "tf_dspine_senders", "tf_dspine_table" };
const Kind = enum { add, read, code, message, write, senders, table };
const group = 256;

fn bind(e: mtl.ComputeEncoder, refs: []const Ref, first: usize) void {
    for (refs, first..) |r, i| e.setBuffer(r.buf, r.off, i);
}

pub const Kernels = struct {
    pipes: [names.len]mtl.Pipeline,
    pub fn init(device: mtl.Device) !Kernels {
        const lib = try mtl.Library.fromSource(device, source, mtl.CompileOptions.mlx());
        defer lib.deinit();
        var pipes: [names.len]mtl.Pipeline = undefined;
        var made: usize = 0;
        errdefer for (pipes[0..made]) |p| p.deinit();
        for (&pipes, names) |*p, name| {
            p.* = try mtl.Pipeline.init(device, lib, name, false);
            made += 1;
        }
        return .{ .pipes = pipes };
    }
    pub fn deinit(k: Kernels) void {
        for (k.pipes) |p| p.deinit();
    }
    fn use(k: Kernels, e: mtl.ComputeEncoder, kind: Kind) void {
        e.setPipeline(k.pipes[@intFromEnum(kind)]);
    }
    /// y = bf16(a + b) over rows x width.
    pub fn add(k: Kernels, e: mtl.ComputeEncoder, a: Ref, b: Ref, y: Ref, rows: u32, width: u32) void {
        k.use(e, .add);
        bind(e, &.{ a, b }, 0);
        e.setValue(Rows{ .rows = rows, .width = width, .block = 1 }, 2);
        e.setBuffer(y.buf, y.off, 3);
        e.dispatchThreads(mtl.Size.of(width, rows, 1), mtl.Size.of(256, 1, 1));
    }
    /// rms (f32 per row) and xn = bf16(raw / (rms + eps)).
    pub fn read(k: Kernels, e: mtl.ComputeEncoder, raw: Ref, xn: Ref, rms: Ref, rows: u32, width: u32, eps: f32) void {
        k.use(e, .read);
        e.setBuffer(raw.buf, raw.off, 0);
        e.setValue(Rows{ .rows = rows, .width = width, .block = 1, .eps = eps }, 1);
        bind(e, &.{ xn, rms }, 2);
        e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(group, 1, 1));
    }
    /// Codes C wide from the read projection; each block's first row from the codebook at ids[row].
    pub fn code(k: Kernels, e: mtl.ComputeEncoder, projected: Ref, codebook: Ref, ids: Ref, out: Ref, rows: u32, c: u32, block: u32, eps: f32) void {
        k.use(e, .code);
        bind(e, &.{ projected, codebook, ids }, 0);
        e.setValue(Rows{ .rows = rows, .width = c, .block = block, .eps = eps, .scale = @sqrt(@as(f32, @floatFromInt(c))) }, 3);
        e.setBuffer(out.buf, out.off, 4);
        e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(group, 1, 1));
    }
    /// The gated message R wide from the packed [Q|S|V] rows.
    pub fn message(k: Kernels, e: mtl.ComputeEncoder, packed_rows: Ref, gate_bias: Ref, out: Ref, rows: u32, r: u32, block: u32) void {
        k.use(e, .message);
        bind(e, &.{ packed_rows, gate_bias }, 0);
        e.setValue(Rows{ .rows = rows, .width = r, .block = block }, 2);
        e.setBuffer(out.buf, out.off, 3);
        e.dispatchThreads(mtl.Size.of(r, rows, 1), mtl.Size.of(256, 1, 1));
    }
    /// The write: res = raw + rms * delta off the anchors, xn its next norm; `u` the base rows bf16(raw * w).
    pub fn write(k: Kernels, e: mtl.ComputeEncoder, raw: Ref, delta: Ref, rms: Ref, weight: Ref, res: Ref, xn: Ref, u: ?Ref, rows: u32, width: u32, block: u32, eps: f32) void {
        k.use(e, .write);
        bind(e, &.{ raw, delta, rms, weight }, 0);
        e.setValue(Rows{ .rows = rows, .width = width, .block = block, .eps = eps, .scale = if (u != null) 1 else 0 }, 4);
        bind(e, &.{ res, xn, u orelse xn }, 5);
        e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(group, 1, 1));
    }
    /// Each head row's K senders' scaled codebook rows [rows K, C] and ids [rows K].
    pub fn senders(k: Kernels, e: mtl.ComputeEncoder, topk: Ref, codebook: Ref, out: Ref, sids: Ref, p: Senders) void {
        k.use(e, .senders);
        bind(e, &.{ topk, codebook }, 0);
        e.setValue(p, 2);
        bind(e, &.{ out, sids }, 3);
        e.dispatchGroups(mtl.Size.of(p.k, p.rows, 1), mtl.Size.of(128, 1, 1));
    }
    /// The greedy score table [rows][K senders][K candidates], f32.
    pub fn table(k: Kernels, e: mtl.ComputeEncoder, packed_rows: Ref, tsv: Ref, gate_bias: Ref, tg: Ref, topk: Ref, logits: Ref, rms: Ref, out: Ref, p: Table) void {
        k.use(e, .table);
        bind(e, &.{ packed_rows, tsv, gate_bias, tg, topk, logits, rms }, 0);
        e.setValue(p, 7);
        e.setBuffer(out.buf, out.off, 8);
        e.dispatchGroups(mtl.Size.of(p.k, p.rows, 1), mtl.Size.of(group, 1, 1));
    }
};

test "DSpine kernel parameter blocks keep their Metal layouts" {
    try std.testing.expectEqual(@as(usize, 20), @sizeOf(Rows));
    try std.testing.expectEqual(@as(usize, 24), @sizeOf(Senders));
    try std.testing.expectEqual(@as(usize, 24), @sizeOf(Table));
}
