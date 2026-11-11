//! The CUDA side: a prefill's host snapshot copies become the hand-off's neutral bf16/f32 parts; absent layers error.
const std = @import("std");
const g = @import("geometry.zig");
const c = @import("codec.zig");
const plan = @import("plan.zig");

pub const Error = error{ MissingSource, MissingRing, MissingCarry, MissingShard, ShortPool, ShortRing, BadCarry, BadRingRows, UnsupportedFormat, OutOfMemory } || c.Error || error{ BadGeometry, BadShape };

pub const IndexFormat = enum { fp8, bf16 };

/// A ring's two arrays: [rows][576] values and [rows][8] scales.
pub const Ring = struct { values: []const u8, scales: []const u8 };

/// One pool family's bytes by rank: replicated holds every page on `ranks[0]`; split puts page p on rank p % ranks.len.
pub const Paged = struct {
    split: bool,
    ranks: []const []const u8,

    fn row(v: Paged, row_bytes: usize, page_rows: u32, r: u64) Error![]const u8 {
        const pg = r / page_rows;
        const owner: usize = if (v.split) @intCast(pg % v.ranks.len) else 0;
        const local = if (v.split) pg / v.ranks.len else pg;
        const bytes = v.ranks[owner];
        if (bytes.len == 0) return error.MissingShard;
        const off = std.math.mul(usize, @intCast(local * page_rows + r % page_rows), row_bytes) catch return error.ShortPool;
        if (off > bytes.len or bytes.len - off < row_bytes) return error.ShortPool;
        return bytes[off..][0..row_bytes];
    }
};

pub const SourceState = struct { layer: u32, comp: Paged, index: Paged };

pub const Cuda = struct {
    index_format: IndexFormat = .fp8,
    rings: g.Rings = .{},
    sources: []const SourceState,
    /// the backbone layers' SWA rings by layer; null where the snapshot does not hold one
    swa: []const ?Ring,
    /// each geometry source's carry (f32 [2][512] bytes: kv then gate), empty for a ratio-1 source
    carries: []const []const u8,
    /// the DSpark blocks' rings, in order
    drafter: []const Ring = &.{},
};

pub const Part = struct { spec: plan.Spec, bytes: []u8 };

fn ringRow(r: Ring, row: u64) Error!struct { *const [g.cuda_values_bytes]u8, *const [g.cuda_scale_bytes]u8 } {
    const v = std.math.mul(usize, @intCast(row), g.cuda_values_bytes) catch return error.ShortRing;
    const s = std.math.mul(usize, @intCast(row), g.cuda_scale_bytes) catch return error.ShortRing;
    if (r.values.len < v + g.cuda_values_bytes or r.scales.len < s + g.cuda_scale_bytes) return error.ShortRing;
    return .{ r.values[v..][0..g.cuda_values_bytes], r.scales[s..][0..g.cuda_scale_bytes] };
}

/// The rows of the last `n` positions before `upto`, oldest first, from a ring with position p at row p % ring_rows.
fn ringPart(out: []u8, r: Ring, upto: u64, n: u32, ring_rows: u32) Error!void {
    var dec: [512]u16 = undefined;
    for (0..n) |i| {
        const pos = upto - n + i;
        const v, const s = try ringRow(r, pos % ring_rows);
        try c.decodeComp(v, s, &dec);
        c.storeRow(512, &dec, out[i * 1024 ..]);
    }
}

fn sourceOf(cuda: Cuda, layer: u32) ?SourceState {
    for (cuda.sources) |s| if (s.layer == layer) return s;
    return null;
}

/// All of a shard's parts, converted, in plan order. `shape` says which producer this is (shard, split, drafter rows).
pub fn build(arena: std.mem.Allocator, shape: plan.Shape, cuda: Cuda) Error![]Part {
    const list = try plan.specs(arena, shape);
    const geo = shape.geo;
    if (cuda.rings.rows < geo.window or !std.math.isPowerOfTwo(cuda.rings.rows) or !std.math.isPowerOfTwo(cuda.rings.drafter_rows)) return error.BadRingRows;
    if (cuda.rings.drafter_rows < shape.ds_rows) return error.BadRingRows;
    const parts = try arena.alloc(Part, list.len);
    for (list, parts) |spec, *p| {
        p.* = .{ .spec = spec, .bytes = try arena.alloc(u8, @intCast(spec.bytes())) };
        switch (spec.kind) {
            .swa => {
                const ring = if (spec.layer < cuda.swa.len) cuda.swa[spec.layer] else null;
                try ringPart(p.bytes, ring orelse return error.MissingRing, shape.upto, spec.rows, cuda.rings.rows);
            },
            .ring => {
                const i = spec.layer - geo.layers;
                if (i >= cuda.drafter.len) return error.MissingRing;
                try ringPart(p.bytes, cuda.drafter[i], shape.upto, spec.rows, cuda.rings.drafter_rows);
            },
            .carry => {
                const s = geo.sourceIndex(spec.layer).?;
                if (s >= cuda.carries.len) return error.MissingCarry;
                if (cuda.carries[s].len != p.bytes.len) return error.BadCarry;
                @memcpy(p.bytes, cuda.carries[s]);
            },
            .comp, .ixk => try pages(p, shape, cuda, sourceOf(cuda, spec.layer) orelse return error.MissingSource),
        }
    }
    return parts;
}

fn pages(p: *Part, shape: plan.Shape, cuda: Cuda, src: SourceState) Error!void {
    const si = shape.geo.sourceIndex(src.layer).?;
    const page_rows = shape.geo.pageRows(si);
    var it = plan.partPages(shape, si, p.spec.kind, p.spec.first_page).?;
    var at: usize = 0;
    var comp: [512]u16 = undefined;
    var idx: [128]u16 = undefined;
    while (it.next()) |run| {
        for (0..run.n) |k| {
            const r = run.row + k;
            const out = p.bytes[at..];
            if (p.spec.kind == .comp) {
                const rec = try src.comp.row(g.cuda_record_bytes, page_rows, r);
                try c.decodeRecord(rec[0..g.cuda_record_bytes], &comp);
                c.storeRow(512, &comp, out);
                at += 1024;
            } else {
                switch (cuda.index_format) {
                    .fp8 => try c.decodeIndexFp8((try src.index.row(g.cuda_index_fp8_bytes, page_rows, r))[0..g.cuda_index_fp8_bytes], &idx),
                    .bf16 => c.decodeIndexBf16((try src.index.row(g.cuda_index_bf16_bytes, page_rows, r))[0..g.cuda_index_bf16_bytes], &idx),
                }
                c.storeRow(128, &idx, out);
                at += 256;
            }
        }
    }
}
