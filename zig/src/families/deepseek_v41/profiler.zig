//! Family stage labels over the generic same-command-buffer counter recorder; no extra storage or GPU waits.
const std = @import("std");
const mtl = @import("metal");
const counters = @import("../../core/gpu_counters.zig");

pub const Stage = enum { engram, mhc, attention, moe, head, other };
pub const stage_count = @typeInfo(Stage).@"enum".field_names.len;
pub const Options = counters.Options;
pub const Aggregate = counters.Aggregate;
pub const Span = counters.Span;

pub fn class(stage: Stage) counters.Class {
    return switch (stage) {
        .engram => .embedding,
        .mhc => .norm,
        .attention => .attention,
        .moe, .head => .dense,
        .other => .copy,
    };
}
pub fn stageOf(span: Span) !Stage {
    if (span.metadata >= stage_count) return error.InvalidProfileStage;
    return @fromBackingInt(@intCast(span.metadata));
}

/// Unique owner of the recorder; stage metadata and class totals share the recorder's bounded span storage.
pub const Recorder = struct {
    recorder: counters.Recorder,
    totals: [stage_count]Aggregate = @splat(.{}),

    pub fn init(allocator: std.mem.Allocator, device: mtl.Device, options: Options) !Recorder {
        return .{ .recorder = try counters.Recorder.init(allocator, device, options) };
    }
    pub fn enabled(self: *const Recorder) bool {
        return self.recorder.enabled();
    }
    pub fn capability(self: *const Recorder) mtl.counters.Capability {
        return self.recorder.capability();
    }
    pub fn activeSpan(self: *const Recorder) bool {
        return self.recorder.active_span != null;
    }
    pub fn beginCommand(self: *Recorder, cb: mtl.CommandBuffer) !void {
        try self.recorder.beginCommand(cb);
        self.totals = @splat(.{});
    }
    pub fn beginSpan(self: *Recorder, encoder: *mtl.ComputeEncoder, stage: Stage) !void {
        try self.recorder.beginSpan(encoder, class(stage), @intCast(@backingInt(stage)));
    }
    pub fn endSpan(self: *Recorder, encoder: *mtl.ComputeEncoder) !void {
        try self.recorder.endSpan(encoder);
    }
    pub fn finishEncoding(self: *Recorder, encoder: *mtl.ComputeEncoder) !void {
        try self.recorder.finishEncoding(encoder);
    }
    pub fn cancelEncoding(self: *Recorder, encoder: *mtl.ComputeEncoder) !void {
        try self.recorder.cancelEncoding(encoder);
        self.totals = @splat(.{});
    }
    pub fn discardAfterCompletion(self: *Recorder) !void {
        try self.recorder.discardAfterCompletion();
        self.totals = @splat(.{});
    }
    pub fn readAfterCompletion(self: *Recorder) ![]const Span {
        const spans = try self.recorder.readAfterCompletion();
        var totals: [stage_count]Aggregate = @splat(.{});
        for (spans) |span| {
            const total = &totals[@backingInt(try stageOf(span))];
            total.calls += 1;
            total.gpu_seconds += span.gpu_seconds;
        }
        self.totals = totals;
        return spans;
    }
    pub fn aggregates(self: *const Recorder) [stage_count]Aggregate {
        return self.totals;
    }
    pub fn deinit(self: *Recorder) !void {
        try self.recorder.deinit();
        self.totals = @splat(.{});
    }
};

test "disabled family profiling allocates nothing and leaves no stage receipts" {
    var recorder = try Recorder.init(std.testing.failing_allocator, undefined, .{});
    defer recorder.deinit() catch unreachable;
    var encoder: mtl.ComputeEncoder = undefined;
    try recorder.beginSpan(&encoder, .mhc);
    try recorder.endSpan(&encoder);
    try std.testing.expect(!recorder.activeSpan());
    try std.testing.expectEqual(@as(usize, 0), (try recorder.readAfterCompletion()).len);
    for (recorder.aggregates()) |total| try std.testing.expectEqual(@as(u64, 0), total.calls);
}
