//! Failure witnesses prove native pocket publication and trainer reclamation require successful device completion.
const std = @import("std");
const a = std.testing.allocator;
const B = struct {
    const Base = @import("engine_test.zig").B;
    pub const Buffer = Base.Buffer;
    base: Base = .{},
    fail_upload: bool = false,
    uploads: usize = 0,
    drains: usize = 0,
    pub const allocationBytes = Base.allocationBytes;
    pub fn upload(b: *@This(), bytes: []const u8) !Buffer {
        b.uploads += 1;
        if (b.fail_upload) return error.PlantedUpload;
        return b.base.upload(bytes);
    }
    pub fn release(b: *@This(), bytes: Buffer) void {
        b.base.release(bytes);
    }
    pub fn drain(b: *@This()) !void {
        b.drains += 1;
        try b.base.drain();
    }
};
test "Qwen4 learning: pocket replacement is atomic under budget drain upload and nonfinite refusals" {
    var b: B = .{};
    var p = try @import("pocket.zig").Pockets(B).init(a, &b, 64, 2);
    defer p.deinitDrained();
    const n = 64 * 64;
    const values = try a.alloc(f32, 3 * n);
    defer a.free(values);
    @memset(values, 0);
    values[0] = 1;
    values[n + 1] = 2;
    values[2 * n + 2] = 3;
    const charge = values.len * 4 * 2;
    p.budget = charge;
    try p.set(0, values[0..n], values[n..][0..n], values[2 * n ..], 1);
    const pointer = p.layers[0].?.buffer.ptr;
    const held = b.base.held;
    try std.testing.expectError(error.Qwen4PocketBudget, p.set(0, values[0..n], values[n..][0..n], values[2 * n ..], 1));
    p.budget = charge * 2;
    b.base.fail_drain = true;
    try std.testing.expectError(error.PlantedDrain, p.set(0, values[0..n], values[n..][0..n], values[2 * n ..], 1));
    b.base.fail_drain = false;
    b.fail_upload = true;
    try std.testing.expectError(error.PlantedUpload, p.set(0, values[0..n], values[n..][0..n], values[2 * n ..], 1));
    b.fail_upload = false;
    const drains = b.drains;
    values[0] = std.math.nan(f32);
    try std.testing.expectError(error.Qwen4PocketNonfinite, p.set(0, values[0..n], values[n..][0..n], values[2 * n ..], 1));
    try std.testing.expectEqual(drains, b.drains);
    try std.testing.expectEqual(pointer, p.layers[0].?.buffer.ptr);
    try std.testing.expectEqual(held, b.base.held);
    try std.testing.expectEqual(@as(f32, 1), p.layers[0].?.values[0]);
    values[0] = 4;
    try p.set(0, values[0..n], values[n..][0..n], values[2 * n ..], 1);
    try std.testing.expectEqual(held, b.base.held);
    try std.testing.expectEqual(@as(f32, 4), p.layers[0].?.values[0]);
}
test "Qwen4 learning: a failed trainer drain retains every buffer until runtime completion and preflight allocates none" {
    var f = try @import("learning_native_fixture.zig").Fixture.open();
    defer f.deinit();
    const native = @import("train_test.zig");
    const r = try native.R.init(a, f.e);
    const old = r.pockets.budget;
    r.pockets.budget = 0;
    const before = f.b.held;
    try std.testing.expectError(error.Qwen4LearningBudget, native.T.init(a, std.testing.io, f.e, &r.pockets, r));
    try std.testing.expectEqual(before, f.b.held);
    try std.testing.expectEqual(@as(u64, 0), r.training_reserved);
    r.pockets.budget = old;
    const t = try native.T.init(a, std.testing.io, f.e, &r.pockets, r);
    const held = f.b.held;
    f.b.fail_drain = true;
    t.deinit();
    try std.testing.expect(r.retained != null and r.training_reserved > 0);
    try std.testing.expect(!f.e.healthy());
    try std.testing.expectEqual(held, f.b.held);
    try std.testing.expectError(error.PlantedDrain, r.close());
    try std.testing.expectEqual(held, f.b.held);
    f.b.fail_drain = false;
    try r.close();
    try std.testing.expectEqual(before, f.b.held);
}
