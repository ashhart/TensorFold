const std = @import("std");
const runtime = @import("runtime.zig");
const recall_plan = @import("recall_plan.zig");

const Runtime = runtime.Runtime;
const Tensor = runtime.Tensor;

pub const KeyValue = struct { key: Tensor, value: Tensor };

pub const Entry = struct {
    frame: usize,
    prompted: bool,
    encoding: Tensor,
    coarse_image: Tensor,
    pointer: Tensor,
    confidence: f32 = 0,
    projections: ?[4]KeyValue = null,
    grid_live: bool = true,

    pub fn tensors(self: Entry, buffer: *[11]Tensor) []const Tensor {
        buffer[0] = self.pointer;
        if (!self.grid_live) return buffer[0..1];
        buffer[1] = self.encoding;
        buffer[2] = self.coarse_image;
        var count: usize = 3;
        if (self.projections) |pairs| for (pairs) |pair| {
            buffer[count] = pair.key;
            buffer[count + 1] = pair.value;
            count += 2;
        };
        return buffer[0..count];
    }

    pub fn release(self: Entry, rt: *Runtime) void {
        var buffer: [11]Tensor = undefined;
        for (self.tensors(&buffer)) |tensor| rt.release(tensor);
    }

    pub fn dropGrid(self: *Entry, rt: *Runtime) void {
        if (!self.grid_live) return;
        var buffer: [11]Tensor = undefined;
        for (self.tensors(&buffer)[1..]) |tensor| rt.release(tensor);
        self.grid_live = false;
        self.projections = null;
    }
};

pub const Bank = struct {
    entries: std.ArrayList(Entry) = .empty,

    pub fn deinit(self: *Bank, allocator: std.mem.Allocator) void {
        self.entries.deinit(allocator);
        self.* = undefined;
    }

    /// Guarantees that the next `put` cannot fail.
    pub fn reserve(self: *Bank, allocator: std.mem.Allocator) !void {
        try self.entries.ensureUnusedCapacity(allocator, 1);
    }

    pub fn clear(self: *Bank, rt: *Runtime) void {
        for (self.entries.items) |entry| entry.release(rt);
        self.entries.clearRetainingCapacity();
    }

    pub fn put(self: *Bank, rt: *Runtime, entry: Entry) void {
        for (self.entries.items) |*existing| if (existing.frame == entry.frame) {
            existing.release(rt);
            existing.* = entry;
            return;
        };
        self.entries.appendAssumeCapacity(entry);
    }

    // Sequential evaluation equals evaluation on the unpruned bank: a removal never changes another entry's decision.
    pub fn prune(self: *Bank, rt: *Runtime, current: usize, frames: usize, grouped: bool) void {
        var i: usize = 0;
        while (i < self.entries.items.len) {
            const decision = recall_plan.keep(self.entries.items, i, current, frames, grouped);
            if (!decision.entry) {
                self.entries.orderedRemove(i).release(rt);
                continue;
            }
            if (!decision.grid) self.entries.items[i].dropGrid(rt);
            i += 1;
        }
    }
};

test "SAM memory entry lists its pointer first and its grid tensors only while live" {
    const fake = struct {
        fn tensor(count: usize) Tensor {
            return .{ .buffer = undefined, .count = count, .dtype = .f32 };
        }
    };
    var pairs: [4]KeyValue = undefined;
    for (&pairs, 0..) |*pair, i| pair.* = .{ .key = fake.tensor(10 + 2 * i), .value = fake.tensor(11 + 2 * i) };
    var entry = Entry{ .frame = 3, .prompted = false, .encoding = fake.tensor(1), .coarse_image = fake.tensor(2), .pointer = fake.tensor(0) };
    var buffer: [11]Tensor = undefined;
    const plain = entry.tensors(&buffer);
    try std.testing.expectEqual(@as(usize, 3), plain.len);
    for (plain, [_]usize{ 0, 1, 2 }) |tensor, count| try std.testing.expectEqual(count, tensor.count);
    entry.projections = pairs;
    const cached = entry.tensors(&buffer);
    try std.testing.expectEqual(@as(usize, 11), cached.len);
    for (cached[3..], 10..) |tensor, count| try std.testing.expectEqual(count, tensor.count);
    entry.grid_live = false;
    try std.testing.expectEqual(@as(usize, 1), entry.tensors(&buffer).len);
}
