//! MTLResidencySet (macOS 15+): keeps a set of allocations (weights, KV pages) resident for a queue.
const std = @import("std");
const objc = @import("objc.zig");
const Buffer = @import("device.zig").Buffer;

const Id = objc.Id;

pub const ResidencySet = struct {
    id: Id,

    /// The share of RAM, in percent, a process's sets may wire: the rest stays with macOS and other processes.
    pub const share_percent = 70;
    /// Kept back inside that share for buffers that grow with long prompts.
    pub const margin: u64 = 2 << 30;

    /// Whether a set holding `held` bytes may take `added` more, the margin inside the share of `ram` (null: never).
    pub fn fits(held: u64, added: u64, ram: ?u64) bool {
        return fitsUnder(held, added, ram, null);
    }

    /// As fits, under a validated load limit when supplied; the margin stays inside it and RAM must be known.
    pub fn fitsUnder(held: u64, added: u64, ram: ?u64, limit: ?u64) bool {
        const total = ram orelse return false;
        const cap = if (limit) |n| @min(n, total) else total / 100 * share_percent;
        return held <= cap and added <= cap - held and margin <= cap - held - added;
    }

    /// Why `fits` said no, logged for `name`: RAM that could not be read, or the bytes past the share.
    pub fn refused(name: []const u8, held: u64, added: u64, ram: ?u64) void {
        const gib = struct {
            fn of(x: u64) f64 {
                return @as(f64, @floatFromInt(x)) / (1 << 30);
            }
        }.of;
        const unread = "{s}: no residency set: this Mac's RAM could not be read, so its 70% load limit is unknown";
        const total = ram orelse return std.log.warn(unread, .{name});
        const past = "{s}: no residency set: its {d:.1} GiB and a {d} GiB margin pass 70% of {d:.0} GiB of RAM";
        if (held == 0) return std.log.warn(past, .{ name, gib(added), margin >> 30, gib(total) });
        const more = "{s}: {d:.1} GiB more would take its residency set past 70% of {d:.0} GiB of RAM: it stays out";
        std.log.info(more, .{ name, gib(added), gib(total) });
    }

    /// `fits` on this Mac's RAM, read now; a refusal is logged for `name`.
    pub fn within(name: []const u8, held: u64, added: u64) bool {
        return withinUnder(name, held, added, null);
    }

    /// As within, using the family's already validated load limit rather than parsing its environment again.
    pub fn withinUnder(name: []const u8, held: u64, added: u64, limit: ?u64) bool {
        const ram = std.process.totalSystemMemory() catch null;
        if (fitsUnder(held, added, ram, limit)) return true;
        refusedUnder(name, held, added, ram, limit);
        return false;
    }

    /// A refusal against the selected bound; the default keeps its existing RAM-share diagnostic.
    pub fn refusedUnder(name: []const u8, held: u64, added: u64, ram: ?u64, limit: ?u64) void {
        const cap = if (limit) |n| @min(n, ram orelse return refused(name, held, added, ram)) else return refused(name, held, added, ram);
        const gib: f64 = 1 << 30;
        if (held == 0) {
            std.log.warn("{s}: no residency set: its {d:.1} GiB and a {d} GiB margin pass the {d:.1} GiB load limit", .{ name, @as(f64, @floatFromInt(added)) / gib, margin >> 30, @as(f64, @floatFromInt(cap)) / gib });
        } else {
            std.log.info("{s}: {d:.1} GiB more would take its residency set past the {d:.1} GiB load limit: it stays out", .{ name, @as(f64, @floatFromInt(added)) / gib, @as(f64, @floatFromInt(cap)) / gib });
        }
    }

    /// Where buffers a model makes or frees after its set was built report, so the set follows them.
    pub const Follow = struct {
        ctx: *anyopaque,
        join: *const fn (ctx: *anyopaque, buffer: Buffer) void,
        leave: *const fn (ctx: *anyopaque, buffer: Buffer) void,
    };

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
