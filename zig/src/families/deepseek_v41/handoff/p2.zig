//! The hand-off as an MKVH protocol 2 export: a `recurrent` layer per model layer, a named part per frame.
const std = @import("std");
const fabric = @import("fabric");
const plan = @import("plan.zig");
const export_cuda = @import("export.zig");
const import_metal = @import("import.zig");
const manifest = fabric.manifest;
const frames = fabric.frames;

pub const Error = error{ BadManifest, UnsupportedDtype, BadPart, OutOfMemory };

const dims = [_]manifest.PartDim{ .tap, .channel };

fn partOf(arena: std.mem.Allocator, s: plan.Spec) !manifest.Part {
    var buf: [40]u8 = undefined;
    return .{
        .name = try arena.dupe(u8, s.name(&buf)),
        .shape = try arena.dupe(u64, &.{ s.rows, s.cols }),
        .dims = &dims,
        .dtype = if (s.dtype == .bf16) .bfloat16 else .float32,
    };
}

/// The manifest layers of a plan's specs (a layer is a run of specs of one layer index), without data.
pub fn describe(arena: std.mem.Allocator, specs: []const plan.Spec) ![]manifest.Layer {
    var out: std.ArrayList(manifest.Layer) = .empty;
    var i: usize = 0;
    while (i < specs.len) {
        var j = i;
        while (j < specs.len and specs[j].layer == specs[i].layer) j += 1;
        const parts = try arena.alloc(manifest.Part, j - i);
        for (specs[i..j], parts) |s, *p| p.* = try partOf(arena, s);
        try out.append(arena, .{ .index = specs[i].layer, .kind = .recurrent, .shape = &.{}, .dims = &.{}, .dtype = .float32, .parts = parts });
        i = j;
    }
    return out.items;
}

pub const Built = struct { layers: []manifest.Layer, sources: []frames.Source };

/// A producer's export layers and their sources over the converted parts (nothing is copied).
pub fn layers(arena: std.mem.Allocator, parts: []const export_cuda.Part) !Built {
    const specs = try arena.alloc(plan.Spec, parts.len);
    for (parts, specs) |p, *s| s.* = p.spec;
    const ls = try describe(arena, specs);
    const sources = try arena.alloc(frames.Source, ls.len);
    var at: usize = 0;
    for (ls, sources) |l, *s| {
        const pieces = try arena.alloc([]const u8, l.parts.len);
        for (pieces, parts[at..][0..l.parts.len]) |*piece, p| piece.* = p.bytes;
        at += l.parts.len;
        s.* = .{ .bytes = &.{}, .shape = &.{}, .block_dim = 0, .elem = 1, .rows = &.{}, .parts = pieces };
    }
    return .{ .layers = ls, .sources = sources };
}

/// The plan a manifest announces: `upto` from the prompt length, drafter rows from ring parts, shards from tp_size.
pub fn shapeOf(geo: @import("geometry.zig").Geometry, m: manifest.Manifest) Error!plan.Shape {
    if (m.protocol != fabric.wire.recurrent_version or m.tp_size > plan.max_world) return error.BadManifest;
    var ds_rows: u32 = 0;
    for (m.layers) |l| if (l.index >= geo.layers and l.parts.len > 0 and l.parts[0].shape.len == 2) {
        ds_rows = std.math.cast(u32, l.parts[0].shape[0]) orelse return error.BadManifest;
    };
    return .{ .geo = geo, .upto = m.prompt_tokens, .shard = @intCast(m.tp_rank), .shards = @intCast(m.tp_size), .split = m.tp_size > 1, .ds_rows = ds_rows };
}

/// A manifest must be exactly the plan of its own announced shape: names, shapes, dims, dtypes, nothing more or less.
pub fn check(arena: std.mem.Allocator, shape: plan.Shape, m: manifest.Manifest) Error!void {
    if (m.protocol != fabric.wire.recurrent_version or m.prompt_tokens != shape.upto or m.tp_rank != shape.shard or m.tp_size != shape.shards) return error.BadManifest;
    const specs = plan.specs(arena, shape) catch return error.BadManifest;
    const want = try describe(arena, specs);
    if (want.len != m.layers.len) return error.BadManifest;
    for (want, m.layers) |w, l| {
        if (w.index != l.index or l.kind != .recurrent or w.parts.len != l.parts.len) return error.BadManifest;
        for (w.parts, l.parts) |a, b| {
            if (!std.mem.eql(u8, a.name, b.name) or a.dtype != b.dtype or !std.mem.eql(u64, a.shape, b.shape) or !std.mem.eql(manifest.PartDim, a.dims, b.dims)) return error.BadManifest;
        }
    }
}

/// A decoder Sink for one shard's pulls, landing each part in the import.
pub const ShardSink = struct {
    imp: *import_metal.Import,
    shard: u32,

    pub fn sink(s: *ShardSink) fabric.decoder.Sink {
        return .{ .ptr = s, .put_fn = put };
    }

    fn put(ptr: *anyopaque, layer: *const manifest.Layer, row_start: u64, rows: u64, bytes: []const u8) anyerror!void {
        const s: *ShardSink = @ptrCast(@alignCast(ptr));
        if (layer.kind != .recurrent or rows != 1 or row_start >= layer.parts.len) return error.BadPart;
        const p = layer.parts[@intCast(row_start)];
        if (p.shape.len != 2) return error.BadPart;
        const dtype: plan.Dtype = switch (p.dtype) {
            .bfloat16 => .bf16,
            .float32 => .f32,
            .float16 => return error.UnsupportedDtype,
        };
        try s.imp.put(s.shard, layer.index, p.name, dtype, p.shape[0], p.shape[1], bytes);
    }
};
