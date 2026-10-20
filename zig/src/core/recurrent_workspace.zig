//! Recurrent scratch leases share one ordered queue; physical disposal requires its completion watermark.
const std = @import("std");

pub fn rowClass(rows: usize, maximum: usize) !usize {
    if (rows == 0 or rows > maximum) return error.InvalidRecurrentWindow;
    return @min(maximum, try std.math.ceilPowerOfTwo(usize, rows));
}

pub fn Workspace(comptime Resource: type) type {
    return struct {
        const Self = @This();
        pub const Lease = struct { slot: usize, generation: u64 };
        const Entry = struct {
            resource: ?Resource = null,
            bytes: usize = 0,
            generation: u64 = 0,
            leased: bool = false,
            last_use: u64 = 0,
        };
        gpa: std.mem.Allocator,
        domain: usize,
        entries: std.ArrayList(Entry) = .empty,
        issued: u64 = 0,
        completed: u64 = 0,

        pub fn init(gpa: std.mem.Allocator, domain: usize) Self {
            return .{ .gpa = gpa, .domain = domain };
        }

        fn check(self: *const Self, domain: usize) !void {
            if (domain != self.domain) return error.ForeignRecurrentQueue;
        }

        fn entry(self: *Self, domain: usize, lease: Lease) !*Entry {
            try self.check(domain);
            if (lease.slot >= self.entries.items.len) return error.InvalidRecurrentLease;
            const e = &self.entries.items[lease.slot];
            if (e.generation != lease.generation) return error.StaleRecurrentLease;
            if (!e.leased or e.resource == null) return error.RecurrentLeaseReleased;
            return e;
        }

        pub fn acquire(self: *Self, domain: usize, bytes: usize, provider: anytype) !Lease {
            try self.check(domain);
            if (bytes == 0) return error.EmptyRecurrentScratch;
            for (self.entries.items, 0..) |*e, i| {
                if (e.leased or e.resource == null or e.bytes != bytes) continue;
                const generation = try std.math.add(u64, e.generation, 1);
                e.generation = generation;
                e.leased = true;
                return .{ .slot = i, .generation = generation };
            }
            try self.collect(domain, provider);
            var at: ?usize = null;
            for (self.entries.items, 0..) |e, i| if (e.resource == null) {
                at = i;
                break;
            };
            if (at == null) try self.entries.ensureUnusedCapacity(self.gpa, 1);
            const generation = try std.math.add(u64, if (at) |i| self.entries.items[i].generation else 0, 1);
            const resource = try provider.allocate(bytes);
            const value: Entry = .{ .resource = resource, .bytes = bytes, .generation = generation, .leased = true };
            const i = at orelse self.entries.items.len;
            if (at != null) self.entries.items[i] = value else self.entries.appendAssumeCapacity(value);
            return .{ .slot = i, .generation = generation };
        }

        pub fn view(self: *Self, domain: usize, lease: Lease) !Resource {
            return (try self.entry(domain, lease)).resource.?;
        }

        pub fn markUsed(self: *Self, domain: usize, lease: Lease) !void {
            const e = try self.entry(domain, lease);
            const epoch = try std.math.add(u64, self.issued, 1);
            self.issued = epoch;
            e.last_use = epoch;
        }

        pub fn reclaimable(self: *Self, domain: usize, lease: Lease) !bool {
            return (try self.entry(domain, lease)).last_use <= self.completed;
        }

        pub fn release(self: *Self, domain: usize, lease: Lease) !void {
            (try self.entry(domain, lease)).leased = false;
        }

        /// Call only after a successful wait covering every previously marked use on this queue.
        pub fn complete(self: *Self, domain: usize) !void {
            try self.check(domain);
            self.completed = self.issued;
        }

        pub fn collect(self: *Self, domain: usize, provider: anytype) !void {
            try self.check(domain);
            for (self.entries.items) |*e| {
                if (e.leased or e.resource == null or e.last_use > self.completed) continue;
                provider.dispose(&e.resource.?);
                e.resource = null;
                e.bytes = 0;
            }
        }

        pub fn residentBytes(self: *const Self) usize {
            var total: usize = 0;
            for (self.entries.items) |e| if (e.resource != null) {
                total += e.bytes;
            };
            return total;
        }

        pub fn deinit(self: *Self, domain: usize, provider: anytype) !void {
            try self.check(domain);
            for (self.entries.items) |e| {
                if (e.last_use > self.completed) return error.RecurrentScratchInFlight;
                if (e.leased) return error.RecurrentScratchLeased;
            }
            try self.collect(domain, provider);
            self.entries.deinit(self.gpa);
            self.* = undefined;
        }
    };
}

const Fake = struct {
    const Resource = struct { id: usize, bytes: usize };
    allocs: usize = 0,
    frees: usize = 0,
    fail: bool = false,
    pub fn allocate(f: *Fake, bytes: usize) !Resource {
        if (f.fail) return error.InjectedAllocation;
        f.allocs += 1;
        return .{ .id = f.allocs, .bytes = bytes };
    }
    pub fn dispose(f: *Fake, _: *Resource) void {
        f.frees += 1;
    }
};

test "recurrent workspace: pending leases reuse one queue but cannot be freed before its completion" {
    var f: Fake = .{};
    var p = Workspace(Fake.Resource).init(std.testing.allocator, 7);
    const one = try p.acquire(7, 1024, &f);
    const id = (try p.view(7, one)).id;
    try p.markUsed(7, one);
    try p.release(7, one);
    try p.collect(7, &f);
    try std.testing.expectEqual(@as(usize, 0), f.frees);
    try std.testing.expectError(error.ForeignRecurrentQueue, p.acquire(8, 1024, &f));
    const two = try p.acquire(7, 1024, &f);
    try std.testing.expectEqual(id, (try p.view(7, two)).id);
    try std.testing.expectError(error.StaleRecurrentLease, p.view(7, one));
    try std.testing.expectError(error.RecurrentScratchInFlight, p.deinit(7, &f));
    try p.complete(7);
    try std.testing.expectError(error.RecurrentScratchLeased, p.deinit(7, &f));
    try p.release(7, two);
    try std.testing.expectError(error.RecurrentLeaseReleased, p.release(7, two));
    try p.collect(7, &f);
    try std.testing.expectEqual(@as(usize, 1), f.allocs);
    try std.testing.expectEqual(@as(usize, 1), f.frees);
    try p.deinit(7, &f);
}

test "recurrent workspace: failed growth preserves a live lease and metadata allocation precedes resources" {
    var f: Fake = .{};
    var p = Workspace(Fake.Resource).init(std.testing.allocator, 9);
    const old = try p.acquire(9, 32, &f);
    f.fail = true;
    try std.testing.expectError(error.InjectedAllocation, p.acquire(9, 64, &f));
    try std.testing.expectEqual(@as(usize, 32), (try p.view(9, old)).bytes);
    try p.release(9, old);
    try p.deinit(9, &f);
    var fail = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = 0 });
    var empty = Workspace(Fake.Resource).init(fail.allocator(), 9);
    const before = f.allocs;
    try std.testing.expectError(error.OutOfMemory, empty.acquire(9, 64, &f));
    try std.testing.expectEqual(before, f.allocs);
    try empty.deinit(9, &f);
}

test "recurrent workspace: every admitted window fits its power-of-two log class" {
    for (1..65) |rows| {
        const n = try rowClass(rows, 64);
        try std.testing.expect(n >= rows and n <= 64);
        try std.testing.expect(std.math.isPowerOfTwo(n));
        if (rows > 1) try std.testing.expect(n < 2 * rows);
    }
    try std.testing.expectError(error.InvalidRecurrentWindow, rowClass(0, 64));
    try std.testing.expectError(error.InvalidRecurrentWindow, rowClass(65, 64));
}
