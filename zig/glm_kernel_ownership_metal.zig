//! Count loader pipeline ownership without a device, shader compiler or driver.
const std = @import("std");
pub var created = std.atomic.Value(usize).init(0);
pub var released = std.atomic.Value(usize).init(0);
var ended: [4096]std.atomic.Value(bool) = @splat(std.atomic.Value(bool).init(false));
pub fn reset() void {
    created.store(0, .monotonic);
    released.store(0, .monotonic);
    for (&ended) |*v| v.store(false, .monotonic);
}
pub const Device = struct {
    tensor: bool,
    pub fn tensorUnits(d: Device) bool {
        return d.tensor;
    }
    pub fn name(_: Device) [*:0]const u8 {
        return "ownership fixture";
    }
};
pub const CompileOptions = struct {
    pub fn mlx() CompileOptions {
        return .{};
    }
};
pub const Library = struct {
    pub fn fromSource(_: Device, _: []const u8, _: CompileOptions) !Library {
        return .{};
    }
    pub fn deinit(_: Library) void {}
};
pub const Pipeline = struct {
    id: usize,
    pub fn init(_: Device, _: Library, _: [:0]const u8, _: bool) !Pipeline {
        const id = created.fetchAdd(1, .monotonic);
        std.debug.assert(id < ended.len);
        return .{ .id = id };
    }
    pub fn deinit(p: Pipeline) void {
        std.debug.assert(!ended[p.id].swap(true, .monotonic));
        _ = released.fetchAdd(1, .monotonic);
    }
};
pub const Buffer = struct {
    pub fn deinit(_: Buffer) void {}
};
pub const ComputeEncoder = struct {};
pub const Queue = struct {};
pub const Size = @import("gemma4_recording_metal.zig").Size;
pub const objc = struct {
    pub const Pool = struct {
        pub fn push() Pool {
            return .{};
        }
        pub fn pop(_: Pool) void {}
    };
};
