//! Constructor and shared-forward ownership use a recording backend; these are not numeric inference tests.
const std = @import("std");
const t = std.testing;
const a = t.allocator;
const cfg = @import("config.zig");
pub const B = struct {
    pub const Buffer = []align(16) u8;
    held: usize = 0,
    allocs: usize = 0,
    fail_drain: bool = false,
    pub fn allocationBytes(n: usize) !u64 {
        return n;
    }
    pub fn allocate(b: *B, n: usize) !Buffer {
        const out = try a.alignedAlloc(u8, .@"16", n);
        b.held += n;
        b.allocs += 1;
        return out;
    }
    pub fn upload(b: *B, bytes: []const u8) !Buffer {
        const out = try b.allocate(bytes.len);
        @memcpy(out, bytes);
        return out;
    }
    pub fn release(b: *B, bytes: Buffer) void {
        b.held -= bytes.len;
        a.free(bytes);
    }
    pub fn drain(b: *B) !void {
        if (b.fail_drain) return error.PlantedDrain;
    }
    pub fn zero(_: *B, to: Buffer, off: usize, n: usize) !void {
        @memset(to[off..][0..n], 0);
    }
    pub fn copy(_: *B, to: Buffer, to_off: usize, from: Buffer, from_off: usize, n: usize) !void {
        @memcpy(to[to_off..][0..n], from[from_off..][0..n]);
    }
    pub fn write(_: *B, to: Buffer, off: usize, bytes: []const u8) !void {
        @memcpy(to[off..][0..bytes.len], bytes);
    }
};
const Model = @import("model.zig").Model(B);
const Arena = @import("arena.zig").Arena(B);
const Tensor = Arena.Tensor;
pub const G = struct {
    pub const Storage = B;
    pub const kind = @import("native_admission.zig").Backend.cuda;
    b: *B,
    project_rows: usize = 0,
    begins: usize = 0,
    submits: usize = 0,
    pub fn deinitDrained(_: *G) void {}
    pub fn begin(g: *G) !void {
        g.begins += 1;
    }
    pub fn submit(g: *G) !void {
        g.submits += 1;
    }
    pub fn discard(_: *G) void {}
    fn zero(x: Tensor) !void {
        @memset(x.buffer[x.offset..][0..try x.bytes()], 0);
    }
    pub fn project(g: *G, w: Model.Matrix, x: Tensor, y: Tensor) !void {
        try t.expectEqual(@as(usize, @intCast(w.params.k)), x.width);
        try t.expectEqual(x.rows, y.rows);
        g.project_rows = @max(g.project_rows, x.rows);
        try zero(y);
    }
    pub fn hcNorm(_: *G, _: Model.Norm, _: Tensor, _: Tensor, y: Tensor, _: f32) !void {
        try zero(y);
    }
    pub fn hcActivation(_: *G, _: Tensor, y: Tensor, _: usize, _: bool) !void {
        try zero(y);
    }
    pub fn hcMix(_: *G, _: Tensor, _: Tensor, y: Tensor, _: Tensor) !void {
        try zero(y);
    }
    pub fn writeback(_: *G, _: Tensor, _: Tensor, _: Tensor, _: Tensor) !void {}
    pub fn swiglu(_: *G, _: Tensor, _: Tensor, y: Tensor) !void {
        try zero(y);
    }
    pub fn delta(g: *G, _: Model.Linear, state: anytype, layer: usize, _: Tensor, y: Tensor, _: Tensor) !void {
        try zero(y);
        const l = state.layout.layers[layer];
        for (0..y.rows) |r| for ([_]@import("state_layout.zig").Range{ l.conv, l.recurrent }) |range| {
            const src = try state.current(range);
            const dst = try state.row(r, range);
            try g.b.copy(dst.buffer, dst.offset, src.buffer, src.offset, src.len);
        };
    }
    pub fn indexed(_: *G, _: *Arena, _: Model.Indexed, _: anytype, _: usize, _: Tensor, y: Tensor) !void {
        try zero(y);
    }
    pub fn ple(g: *G, _: *Arena, _: Model.Ple, state: anytype, layer: usize, _: Tensor, h: Tensor, _: Tensor, _: Tensor, _: Tensor) !void {
        const range = state.layout.layers[layer].ple;
        const src = try state.current(range);
        for (0..h.rows) |r| {
            const dst = try state.row(r, range);
            try g.b.copy(dst.buffer, dst.offset, src.buffer, src.offset, src.len);
        }
    }
};
pub const Loader = struct {
    pub fn load(_: Loader, _: std.mem.Allocator, b: *B, _: *const cfg.Config, _: *cfg.Problem) !G {
        return .{ .b = b };
    }
};
test "Qwen4 engine owner: whole-path budget before allocation, shared rows, exact slot keeps and failed drain retention" {
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    const mem = arena.allocator();
    const text = @embedFile("fixtures/native-dense.json");
    var p: cfg.Problem = .{};
    var c = try cfg.parse(a, text, &p);
    defer c.deinit();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "config.json", .data = text });
    try @import("reader_fixture.zig").write(tmp, mem, &c, false);
    const path = try std.fmt.allocPrint(mem, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    const E = @import("engine.zig").Engine(G);
    var b: B = .{};
    var o: @import("native_admission.zig").Options = .{ .capacity = 32, .window = 4, .streams = 2, .budget = 0 };
    try t.expectError(error.Qwen4PathDoesNotFit, E.open(a, std.testing.io, path, &b, o, &p, Loader{}));
    try t.expectEqual(@as(usize, 0), b.allocs);
    o.budget = std.math.maxInt(u64);
    o.config_digest = @splat(0);
    try t.expectError(error.UnsupportedQwen4, E.open(a, std.testing.io, path, &b, o, &p, Loader{}));
    try t.expectEqual(@as(usize, 0), b.allocs);
    try t.expectEqualStrings("config.json changed during admission", p.name());
    o.config_digest = null;
    const e = try E.open(a, std.testing.io, path, &b, o, &p, Loader{});
    const left = try e.lease();
    const right = try e.lease();
    try t.expectError(error.Qwen4NoStreamSlot, e.lease());
    const begins = e.gpu.begins;
    try t.expectError(error.TokenOutOfRange, e.forward(&.{.{ .state = &left.state, .tokens = &.{999} }}, E.NoDraw{}, &p));
    try t.expectEqual(begins, e.gpu.begins);
    try t.expect(e.healthy());
    try e.forward(&.{ .{ .state = &left.state, .tokens = &.{ 3, 5 } }, .{ .state = &right.state, .tokens = &.{7} } }, E.NoDraw{}, &p);
    try t.expect(e.scratch.peak <= e.plan.scratch);
    try t.expect(b.held <= e.plan.device);
    try t.expectEqual(@as(usize, 3), e.gpu.project_rows);
    try t.expectEqual(@as(usize, 1), e.gpu.submits);
    try e.keep(left, 1);
    try e.keep(right, 1);
    try t.expectEqual(@as(usize, 1), left.state.pos);
    try t.expect(!left.has_logits);
    try t.expect(right.has_logits);
    try e.release(left);
    try t.expectEqual(left, try e.lease());
    try t.expectEqual(@as(usize, 0), left.state.pos);
    const held = b.held;
    b.fail_drain = true;
    try t.expectError(error.PlantedDrain, e.close());
    try t.expectEqual(held, b.held);
    b.fail_drain = false;
    try e.close();
    try t.expectEqual(@as(usize, 0), b.held);
}
