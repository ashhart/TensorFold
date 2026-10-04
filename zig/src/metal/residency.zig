//! MTLResidencySet (macOS 15+): keeps a fixed set of allocations (weights, KV pages) resident for a queue.
const objc = @import("objc.zig");
const Buffer = @import("device.zig").Buffer;

const Id = objc.Id;

pub const ResidencySet = struct {
    id: Id,

    pub fn add(self: ResidencySet, buffer: Buffer) void {
        objc.msg(void, self.id, "addAllocation:", .{buffer.id});
    }

    pub fn remove(self: ResidencySet, buffer: Buffer) void {
        objc.msg(void, self.id, "removeAllocation:", .{buffer.id});
    }

    /// Apply the adds and removes since the last commit.
    pub fn commit(self: ResidencySet) void {
        objc.msg(void, self.id, "commit", .{});
    }

    pub fn requestResidency(self: ResidencySet) void {
        objc.msg(void, self.id, "requestResidency", .{});
    }

    pub fn allocatedSize(self: ResidencySet) u64 {
        return objc.msg(u64, self.id, "allocatedSize", .{});
    }

    pub fn count(self: ResidencySet) usize {
        return objc.msg(usize, self.id, "allocationCount", .{});
    }

    pub fn deinit(self: ResidencySet) void {
        objc.release(self.id);
    }
};
