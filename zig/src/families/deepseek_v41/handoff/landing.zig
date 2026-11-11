//! Lands a pulled or file-fed hand-off on a Metal slot's Packed buffers; the drafter's rings land if the slot has them.
const std = @import("std");
const fabric = @import("fabric");
const g = @import("geometry.zig");
const plan = @import("plan.zig");
const metal = @import("metal.zig");
const import_metal = @import("import.zig");
const p2 = @import("p2.zig");
const ps = fabric.prefill_source;
const manifest = fabric.manifest;

pub const Landing = struct {
    imp: import_metal.Import,
    sink: p2.ShardSink,
    /// the prompt tokens the state covers
    upto: u64,
    /// ring parts read and dropped (the slot has no drafter rings)
    dropped: u32 = 0,
    /// the drafter stages whose rings landed (bit i: stage i), and the rows of each
    rings: u8 = 0,
    ring_rows: u64 = 0,
    /// time spent in `put`, ns
    land_ns: u64 = 0,

    /// In place: the sink points at `imp`. `upto` tokens land in `layout` (a blank Packed slot).
    pub fn init(l: *Landing, gpa: std.mem.Allocator, upto: u64, layout: metal.Layout) !void {
        l.* = .{ .imp = try import_metal.Import.init(gpa, .{ .upto = upto }, layout, .{}), .sink = undefined, .upto = upto };
        l.sink = .{ .imp = &l.imp, .shard = 0 };
    }

    pub fn deinit(l: *Landing) void {
        l.imp.deinit();
    }

    pub fn state(l: *Landing) ps.StateSink {
        return .{ .ptr = l, .put_fn = put, .commit_fn = commit, .abort_fn = abort };
    }

    /// The decoder's side of the same calls (a file feeds them directly).
    pub fn decoderSink(l: *Landing) @import("fabric").decoder.Sink {
        return .{ .ptr = l, .put_fn = put };
    }

    fn put(ptr: *anyopaque, layer: *const manifest.Layer, row_start: u64, rows: u64, bytes: []const u8) anyerror!void {
        const l: *Landing = @ptrCast(@alignCast(ptr));
        const began = fabric.words.nowNs();
        defer l.land_ns += fabric.words.nowNs() - began;
        if (layer.index >= g.release.layers) return l.ring(layer, row_start, rows, bytes);
        return l.sink.sink().put_fn(l.sink.sink().ptr, layer, row_start, rows, bytes);
    }

    /// A drafter ring part: landed in the slot's rings if it has them (a wrong ring costs drafts, never a token).
    fn ring(l: *Landing, layer: *const manifest.Layer, row_start: u64, rows: u64, bytes: []const u8) anyerror!void {
        const geo = g.release;
        if (!l.imp.layout.hasDrafter(geo.mtp_layers)) {
            l.dropped += 1;
            return;
        }
        if (layer.kind != .recurrent or rows != 1 or row_start >= layer.parts.len or layer.index >= geo.layers + geo.mtp_layers) return error.BadPart;
        const part = layer.parts[@intCast(row_start)];
        if (part.shape.len != 2 or part.shape[1] != geo.head_dim or part.dtype != .bfloat16 or !std.mem.eql(u8, part.name, "ring")) return error.BadPart;
        const n = part.shape[0];
        if (n == 0 or n > @min(l.upto, geo.window) or bytes.len != n * 1024) return error.BadPart;
        const bit = @as(u8, 1) << @intCast(layer.index - geo.layers);
        if (l.rings & bit != 0 or (l.rings != 0 and n != l.ring_rows)) return error.BadPart;
        try import_metal.Import.landRing(l.imp.layout, geo, l.upto, layer.index, n, bytes);
        l.rings |= bit;
        l.ring_rows = n;
    }

    fn commit(_: *anyopaque, _: u64, _: ps.Provenance) void {} // `complete` is the check, after the pull
    fn abort(_: *anyopaque) void {} // the caller drops the blank slot

    /// Every part of the plan landed; the result says which drafter key positions did (all three stages, or none).
    pub fn complete(l: *const Landing) !import_metal.Result {
        var r = try l.imp.complete();
        r.drafter_dropped = l.dropped > 0;
        if (l.rings == 0) return r;
        if (l.rings != (@as(u8, 1) << @intCast(g.release.mtp_layers)) - 1) return error.Incomplete;
        r.ds_first = l.upto - l.ring_rows;
        return r;
    }
};
