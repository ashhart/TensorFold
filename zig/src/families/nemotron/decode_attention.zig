//! Chain verification with the decode attention pipelines and causal extent for each row.
const std = @import("std");
const mtl = @import("metal");

pub const Ref = struct { buf: mtl.Buffer, off: usize = 0 };
pub const Pipelines = struct { q: mtl.Pipeline, partial: mtl.Pipeline, merge: mtl.Pipeline, out: mtl.Pipeline };
pub const Call = struct {
    head_dim: usize,
    heads: usize,
    kv_heads: usize,
    qkv_n: usize,
    capacity: usize,
    len: usize,
    first: usize,
    rows: usize,
    qkv: Ref,
    qp: Ref,
    k: Ref,
    v: Ref,
    po: Ref,
    pm: Ref,
    pl: Ref,
    att: Ref,
    out: Ref,
    table: ?Ref = null,
    page_bytes: u64 = 0,
};

pub fn enabled(tensor_units: bool, bolt: bool) bool {
    return !tensor_units and !bolt;
}

pub fn serialForSegment(verify: bool, paged: bool, lone: bool, tree: bool, gpu: bool) bool {
    return verify and paged and !lone and !tree and !gpu;
}

fn bind(e: anytype, r: Ref, index: usize) void {
    e.buf(r.buf, r.off, index);
}

/// Reuses scratch between rows; the caller's encoder keeps its normal data-flow barriers.
pub fn encode(e: anytype, p: Pipelines, c: Call) void {
    const hd = c.head_dim;
    const g = c.heads / c.kv_heads;
    std.debug.assert(g == 16 and hd == 128 and c.len + c.rows <= c.capacity);
    const strides = [4]i64{ @intCast(c.kv_heads * c.capacity * hd), @intCast(c.capacity * hd), @intCast(hd), 1 };
    const scale: f32 = @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(hd))));
    for (0..c.rows) |r| {
        const keys = c.len + r + 1;
        const chunks = (keys + 511) / 512;
        const dims = [5]i32{ @intCast(keys), @intCast(chunks), 1, 1, 1 };
        e.pipe(p.q);
        e.buf(c.qkv.buf, c.qkv.off + (c.first + r) * c.qkv_n * 2, 0);
        bind(e, c.qp, 1);
        e.bytes([2]u32{ @intCast(c.qkv_n), @intCast(g) }, 2);
        e.run(.{ hd, g, c.kv_heads }, .{ hd, 1, 1 });

        e.pipe(p.partial);
        bind(e, c.qp, 0);
        bind(e, c.k, 1);
        e.bytes(strides, 2);
        bind(e, c.v, 3);
        e.bytes(strides, 4);
        e.bytes(scale, 5);
        e.bytes(dims, 6);
        bind(e, c.po, 7);
        bind(e, c.pm, 8);
        bind(e, c.pl, 9);
        if (c.table) |t| {
            bind(e, t, 10);
            e.bytes(c.page_bytes, 11);
        }
        e.run(.{ c.kv_heads * 32, chunks, 1 }, .{ 32, 1, 1 });

        e.pipe(p.merge);
        bind(e, c.po, 0);
        bind(e, c.pm, 1);
        bind(e, c.pl, 2);
        e.bytes(dims, 3);
        bind(e, c.att, 4);
        e.run(.{ c.kv_heads * 32, g, 1 }, .{ 32, 1, 1 });

        e.pipe(p.out);
        bind(e, c.att, 0);
        e.buf(c.out.buf, c.out.off + (c.first + r) * c.heads * hd * 2, 1);
        e.run(.{ hd, c.heads, 1 }, .{ hd, 1, 1 });
    }
}
