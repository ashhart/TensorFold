const std = @import("std");

pub const Profile = struct {
    allocator: std.mem.Allocator,
    totals: std.StringHashMapUnmanaged(Entry) = .empty,

    const Entry = struct { name: []const u8, calls: usize = 0, gpu_seconds: f64 = 0 };

    pub fn deinit(self: *Profile) void {
        var keys = self.totals.keyIterator();
        while (keys.next()) |key| self.allocator.free(key.*);
        self.totals.deinit(self.allocator);
    }

    pub fn record(context: *anyopaque, name: []const u8, seconds: f64) !void {
        const self: *Profile = @ptrCast(@alignCast(context));
        if (self.totals.getPtr(name)) |entry| {
            entry.calls += 1;
            entry.gpu_seconds += seconds;
        } else {
            const key = try self.allocator.dupe(u8, name);
            errdefer self.allocator.free(key);
            try self.totals.put(self.allocator, key, .{ .name = key, .calls = 1, .gpu_seconds = seconds });
        }
    }

    fn before(_: void, left: Entry, right: Entry) bool {
        return left.gpu_seconds > right.gpu_seconds;
    }

    pub fn write(self: *const Profile, a: std.mem.Allocator, io: std.Io, path: []const u8, identity: anytype, frames: usize, profiled_frames: usize, selected_frame: ?usize, gpu_seconds: f64) !void {
        const entries = try a.alloc(Entry, self.totals.count());
        var values = self.totals.valueIterator();
        var total: f64 = 0;
        for (entries) |*entry| {
            entry.* = values.next().?.*;
            total += entry.gpu_seconds;
        }
        std.mem.sort(Entry, entries, {}, before);
        const report = try std.json.Stringify.valueAlloc(a, .{
            .schema = 1,
            .scope = "per-operation GPU synchronization; perturbs batching; not whole-video timing",
            .checkpoint_sha256 = identity.checkpoint_sha256[0..],
            .frames_tensor_sha256 = identity.frames_tensor_sha256[0..],
            .executable_sha256 = identity.executable_sha256[0..],
            .frames = frames,
            .profiled_frames = profiled_frames,
            .selected_frame = selected_frame,
            .total_gpu_seconds = gpu_seconds,
            .profiled_gpu_seconds = total,
            .operators = entries,
        }, .{ .whitespace = .indent_2 });
        const file = try std.Io.Dir.cwd().createFile(io, path, .{ .exclusive = true });
        defer file.close(io);
        try file.writeStreamingAll(io, report);
        try file.writeStreamingAll(io, "\n");
    }
};
