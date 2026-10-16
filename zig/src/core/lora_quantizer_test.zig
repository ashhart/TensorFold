//! `tf-merge-check --quantizer` on synthetic dense and shipped folders: validation of both inputs, no model, no GPU.
const std = @import("std");
const se = @import("shard_edit.zig");
const fx = @import("lora_fixture.zig");
const dir = @import("lora_merge_dir.zig");
const Io = std.Io;

const io = fx.io;
const module = fx.module;

fn contains(hay: []const u8, needle: []const u8) bool {
    return std.mem.indexOf(u8, hay, needle) != null;
}

/// The quantizer check over the scratch folders `dense` and `base`: its result and what it printed.
fn check(s: *fx.Scratch, msg: []u8) !struct { res: anyerror!bool, out: []const u8 } {
    var w = Io.Writer.fixed(msg);
    const res = dir.gate.quantizerCheck(std.testing.allocator, s.arena.allocator(), io, &w, try s.path(&.{"dense"}), try s.path(&.{"base"}), &.{module});
    return .{ .res = res, .out = w.buffered() };
}

test "the quantizer check passes on a dense source and its own quantization, and fails on another weight" {
    var s = try fx.Scratch.init();
    defer s.deinit();
    _ = try fx.makeSingle(&s, 8, 128, try fx.randomValues(s.arena.allocator(), 8 * 128, 4));
    var buf: [4096]u8 = undefined;
    var r = try check(&s, &buf);
    try std.testing.expect(try r.res);
    try std.testing.expect(contains(r.out, "identical") and contains(r.out, "byte-identical on every tensor"));
    // another dense source: the shipped tensor no longer comes from it
    const a = s.arena.allocator();
    var dense: std.ArrayList(se.New) = .empty;
    var quantized: std.ArrayList(se.New) = .empty;
    try fx.leaves(a, module, 8, 128, try fx.randomValues(a, 8 * 128, 9), &dense, &quantized);
    try se.create(std.testing.allocator, try s.path(&.{ "dense", "model.safetensors" }), &.{}, dense.items);
    r = try check(&s, &buf);
    try std.testing.expect(!(try r.res));
    try std.testing.expect(contains(r.out, "DIFFERS"));
}

/// A dense file holding `module`.weight with the given dtype, shape and zeroed payload.
fn denseOf(s: *fx.Scratch, dtype: @import("safetensors.zig").DType, shape: []const usize) !void {
    var count: usize = dtype.size();
    for (shape) |d| count *= d;
    const bytes = try s.arena.allocator().alloc(u8, count);
    @memset(bytes, 0);
    try se.create(std.testing.allocator, try s.path(&.{ "dense", "model.safetensors" }), &.{}, &.{.{ .name = module ++ ".weight", .dtype = dtype, .shape = shape, .bytes = bytes }});
}

test "a dense tensor with fewer or more elements, reshaped, or of an unsupported dtype is refused naming it" {
    const Case = struct { dtype: @import("safetensors.zig").DType, shape: []const usize, why: []const u8 };
    const cases = [_]Case{
        .{ .dtype = .bf16, .shape = &.{ 8, 64 }, .why = "dense shape" },
        .{ .dtype = .bf16, .shape = &.{ 8, 256 }, .why = "dense shape" },
        .{ .dtype = .bf16, .shape = &.{ 16, 64 }, .why = "dense shape" },
        .{ .dtype = .bf16, .shape = &.{8 * 128}, .why = "dense shape" },
        .{ .dtype = .f64, .shape = &.{ 8, 128 }, .why = "dense dtype" },
    };
    for (cases) |c| {
        var s = try fx.Scratch.init();
        defer s.deinit();
        _ = try fx.makeSingle(&s, 8, 128, try fx.randomValues(s.arena.allocator(), 8 * 128, 4));
        try denseOf(&s, c.dtype, c.shape);
        var buf: [4096]u8 = undefined;
        const r = try check(&s, &buf);
        try std.testing.expectError(error.Refused, r.res);
        try std.testing.expect(contains(r.out, module ++ ".weight") and contains(r.out, c.why));
    }
}

test "a directory standing in for the index, a shard or the config is refused, not read" {
    const Stand = enum { dense_index, dense_shard, shipped_index, shipped_shard, config };
    for ([_]Stand{ .dense_index, .dense_shard, .shipped_index, .shipped_shard, .config }) |t| {
        var s = try fx.Scratch.init();
        defer s.deinit();
        _ = try fx.makeSingle(&s, 8, 128, try fx.randomValues(s.arena.allocator(), 8 * 128, 4));
        const target = switch (t) {
            .dense_index => .{ "dense", "model.safetensors.index.json" },
            .dense_shard => .{ "dense", "model.safetensors" },
            .shipped_index => .{ "base", "model.safetensors.index.json" },
            .shipped_shard => .{ "base", "model.safetensors" },
            .config => .{ "base", "config.json" },
        };
        if (t != .dense_index and t != .shipped_index) try Io.Dir.cwd().deleteFile(io, try s.path(&.{ target[0], target[1] }));
        try Io.Dir.cwd().createDirPath(io, try s.path(&.{ target[0], target[1] }));
        var buf: [4096]u8 = undefined;
        const r = try check(&s, &buf);
        try std.testing.expectError(error.Refused, r.res);
        try std.testing.expect(contains(r.out, target[1]) and contains(r.out, "not a regular file"));
    }
}

fn checkWith(gpa: std.mem.Allocator, dense: []const u8, shipped: []const u8) !void {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    var buf: [8192]u8 = undefined;
    var w = Io.Writer.fixed(&buf);
    _ = try dir.gate.quantizerCheck(gpa, arena.allocator(), io, &w, dense, shipped, &.{module});
}

test "an allocation that fails at any step is unwound without a leak or a close of an unopened shard" {
    var s = try fx.Scratch.init();
    defer s.deinit();
    _ = try fx.makeSingle(&s, 8, 128, try fx.randomValues(s.arena.allocator(), 8 * 128, 4));
    const dense = try s.path(&.{"dense"});
    const shipped = try s.path(&.{"base"});
    var at: usize = 0;
    while (at < 1000) : (at += 1) {
        var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = at });
        if (checkWith(failing.allocator(), dense, shipped)) |_| {
            try std.testing.expect(!failing.has_induced_failure);
            break;
        } else |e| try std.testing.expectEqual(error.OutOfMemory, e);
    }
    try std.testing.expect(at > 5 and at < 1000); // it did fail at some steps, and ran clean once none failed
}
