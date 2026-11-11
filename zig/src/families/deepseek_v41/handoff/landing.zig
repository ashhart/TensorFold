//! Lands a pulled or file-fed hand-off on a Metal slot's Packed buffers; ring layers (drafter) are read and dropped.
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
    /// shard 0's sink (the whole state when one producer exports it)
    sink: p2.ShardSink,
    /// every shard's sink, when KV_SPLIT's producers each export theirs
    shards: [plan.max_world]p2.ShardSink = undefined,
    /// ring parts read and dropped
    dropped: u32 = 0,

    /// In place: the sink points at `imp`. `upto` tokens land in `layout` (a blank Packed slot).
    pub fn init(l: *Landing, gpa: std.mem.Allocator, upto: u64, layout: metal.Layout) !void {
        return l.initShards(gpa, upto, 1, layout);
    }

    /// `init` for a state that `shards` producers export (KV_SPLIT): `state().shard_fn` gives each its sink.
    pub fn initShards(l: *Landing, gpa: std.mem.Allocator, upto: u64, shards: u32, layout: metal.Layout) !void {
        const shape: plan.Shape = .{ .upto = upto, .shards = shards, .split = shards > 1 };
        l.* = .{ .imp = try import_metal.Import.init(gpa, shape, layout, .{ .drafter = false }), .sink = undefined };
        l.sink = .{ .imp = &l.imp, .shard = 0 };
        for (l.shards[0..shards], 0..) |*s, k| s.* = .{ .imp = &l.imp, .shard = @intCast(k) };
    }

    pub fn deinit(l: *Landing) void {
        l.imp.deinit();
    }

    pub fn state(l: *Landing) ps.StateSink {
        return .{ .ptr = l, .put_fn = put, .commit_fn = commit, .abort_fn = abort, .shard_fn = shardSink };
    }

    fn shardSink(ptr: *anyopaque, shard: u32) @import("fabric").decoder.Sink {
        const l: *Landing = @ptrCast(@alignCast(ptr));
        return .{ .ptr = &l.shards[shard], .put_fn = putShard };
    }

    /// One shard's part, on its pull's thread: ring layers (shard 0's drafter) are read and dropped, uncounted.
    fn putShard(ptr: *anyopaque, layer: *const manifest.Layer, row_start: u64, rows: u64, bytes: []const u8) anyerror!void {
        const s: *p2.ShardSink = @ptrCast(@alignCast(ptr));
        if (layer.index >= g.release.layers) return;
        return s.sink().put_fn(s.sink().ptr, layer, row_start, rows, bytes);
    }

    /// The decoder's side of the same calls (a file feeds them directly).
    pub fn decoderSink(l: *Landing) @import("fabric").decoder.Sink {
        return .{ .ptr = l, .put_fn = put };
    }

    fn put(ptr: *anyopaque, layer: *const manifest.Layer, row_start: u64, rows: u64, bytes: []const u8) anyerror!void {
        const l: *Landing = @ptrCast(@alignCast(ptr));
        if (layer.index >= g.release.layers) {
            l.dropped += 1;
            return;
        }
        return l.sink.sink().put_fn(l.sink.sink().ptr, layer, row_start, rows, bytes);
    }

    fn commit(_: *anyopaque, _: u64, _: ps.Provenance) void {} // `complete` is the check, after the pull
    fn abort(_: *anyopaque) void {} // the caller drops the blank slot

    /// Every part of the plan landed.
    pub fn complete(l: *const Landing) !import_metal.Result {
        return l.imp.complete();
    }
};
