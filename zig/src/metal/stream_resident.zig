//! Storage-backed serving families share initial enrollment and paired callbacks for later stream state.
const std = @import("std");
pub fn Streams(comptime mtl: type) type {
    return struct {
        pub fn set(b: anytype, hook: ?mtl.ResidencySet.Follow) void {
            b.scratch.memory.setResident(hook);
            var it = b.caches.valueIterator();
            while (it.next()) |cache| cache.*.memory.setResident(hook);
            b.residency = hook;
        }
        pub fn hold(h: anytype, io: std.Io, label: []const u8) !void {
            var list: std.ArrayList(mtl.Buffer) = .empty;
            defer list.deinit(h.gpa);
            for (h.model.checkpoint.shards.items) |shard| try list.append(h.gpa, shard.buffer);
            try list.appendSlice(h.gpa, h.metal.scratch.memory.buffers.items);
            var it = h.metal.caches.valueIterator();
            while (it.next()) |cache| try list.appendSlice(h.gpa, cache.*.memory.buffers.items);
            if (@hasDecl(@TypeOf(h.model.*), "dropResident")) h.model.dropResident();
            h.resident = mtl.resident.Resident.hold(h.gpa, io, h.model.device, h.model.queue, list.items, label) catch |err| blk: {
                std.log.warn("{s}: residency skipped ({s})", .{ label, @errorName(err) });
                break :blk null;
            };
            if (h.resident) |*r| set(h.metal, r.follow());
            h.warm = if (h.resident) |*r| r.target() else .{ .queue = h.model.queue };
        }
        pub fn drop(h: anytype) void {
            set(h.metal, null);
            if (h.resident) |*r| r.deinit();
            h.resident = null;
            h.warm = .{ .queue = h.model.queue };
        }
    };
}
