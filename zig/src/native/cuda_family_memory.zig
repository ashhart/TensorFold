//! Family-aware checkpoint and stream admission; rank coordination runs before the host admits any stream.
const std = @import("std");
const cuda = @import("cuda");
const api = @import("engine_api");
const budget = @import("cuda_memory.zig");
const getenv = @import("cuda_device.zig").getenv;
const Allocator = std.mem.Allocator;

/// The pool now, read on the thread whose context is current; `problem` names a bad variable.
pub fn pool(a: Allocator, io: std.Io, ctx: *const cuda.Context, problem: *[]const u8) !?budget.Pool {
    return poolFor(@import("sys").windows, Sources, a, io, ctx, problem);
}
const Sources = struct {
    pub const procText = budget.procText;
    pub const env = getenv;
};
/// The served pool decision with only the OS and host probes supplied by a fixture.
pub fn poolFor(comptime windows: bool, comptime Host: type, a: Allocator, io: std.Io, ctx: *const cuda.Context, problem: *[]const u8) !?budget.Pool {
    const unified = (try ctx.attribute(.integrated)) != 0;
    const card = try ctx.memInfo();
    const host_counts = budget.hostCounts(unified, windows);
    const text = if (host_counts) Host.procText(a, io, "/proc/meminfo") else null;
    defer if (text) |bytes| a.free(bytes);
    // read after MemAvailable, which the lists drain into: a page moving in between is missed, not counted twice
    const zones = if (host_counts) Host.procText(a, io, "/proc/zoneinfo") else null;
    defer if (zones) |bytes| a.free(bytes);
    const measured = budget.counts(host_counts, .{ .total = card.total, .available = card.free }, text) catch {
        problem.* = "cannot read or parse /proc/meminfo's MemTotal and MemAvailable; refusing CUDA unified-memory admission";
        return null;
    };
    const available = budget.counting(measured, zones);
    const free = available.available;
    const total = available.total;
    const reserve = budget.reserveBytes(Host.env("TENSORFOLD_MEMORY_RESERVE_GIB"), total, unified) catch {
        problem.* = try std.fmt.allocPrint(a, "TENSORFOLD_MEMORY_RESERVE_GIB={s}: a number of GiB from {d} to the memory's size", .{ Host.env("TENSORFOLD_MEMORY_RESERVE_GIB").?, budget.floor(unified).least });
        return null;
    };
    const limit = budget.limitBytes(Host.env("TENSORFOLD_CUDA_MEMORY_LIMIT_GB")) catch {
        problem.* = try std.fmt.allocPrint(a, "TENSORFOLD_CUDA_MEMORY_LIMIT_GB={s}: a positive number of GiB whose byte count fits in a 64-bit size", .{Host.env("TENSORFOLD_CUDA_MEMORY_LIMIT_GB").?});
        return null;
    };
    return .{ .free = free, .total = total, .reserve = reserve, .limit = limit, .unified = unified, .parked = free - measured.available };
}

/// `F`'s weight bytes on this rank: tensors or files, a split's share, a tower beside; null if tensors are unreadable.
pub fn weights(comptime F: type, a: Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !?u64 {
    const whole = if (@hasDecl(F, "tensorWeightBytes")) F.tensorWeightBytes(a, io, o.dir, o.vision != null) catch |err| {
        problem.* = try std.fmt.allocPrint(a, "cannot read checkpoint tensor sizes ({s})", .{@errorName(err)});
        return null;
    } else budget.weightBytes(a, io, o.dir) catch |err| {
        problem.* = try std.fmt.allocPrint(a, "cannot estimate checkpoint weight files ({s}); refusing CUDA admission", .{@errorName(err)});
        return null;
    };
    const text = if (@hasDecl(F, "placementWeightBytes")) try F.placementWeightBytes(a, io, o.dir, o.tp, o.rank, o.drafts) else if (@hasDecl(F, "weightShare")) F.weightShare(a, io, o.dir, whole) else whole;
    return text + try @import("cuda_vision_offer.zig").weightBytes(F, a, io, o);
}

/// Logs, from 0.25 GiB, the free pages on the per-CPU lists that a pool's reading counted.
pub fn noteParked(p: budget.Pool, what: []const u8) void {
    const line = "{s} counts the {d:.2} GiB free on the per-CPU page lists beyond their resting level (MemAvailable does not)";
    if (p.parked >= 1 << 28) std.log.info(line, .{ what, toGib(p.parked) });
}

pub fn room(comptime F: type, ptr: *anyopaque, local: u64) !u64 {
    if (@hasDecl(F, "rankRoom")) return F.rankRoom(ptr, local);
    return local;
}

pub fn read(_: ?*anyopaque, reset_peak: bool) ?api.Memory {
    const u = cuda.usage(reset_peak);
    return .{ .active = u.device, .cache = 0, .peak = u.peak, .pinned = u.host };
}

pub fn learnBytes(loaded: anytype) u64 {
    return if (@hasField(@TypeOf(loaded), "learn_bytes")) loaded.learn_bytes else 0;
}

pub fn toGib(bytes: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / @as(f64, 1 << 30);
}

test "rank room is agreed before stream admission and preserves a single-rank family's pool" {
    const Single = struct {};
    const Pair = struct {
        pub fn rankRoom(_: *anyopaque, local: u64) !u64 {
            return @min(local, 19);
        }
    };
    var context: u8 = 0;
    try std.testing.expectEqual(@as(u64, 30), try room(Single, &context, 30));
    try std.testing.expectEqual(@as(u64, 19), try room(Pair, &context, 30));
    try std.testing.expectEqual(@as(u64, 7), try room(Pair, &context, 7));
}

test {
    _ = @import("cuda_family_memory_test.zig");
}

test "weight-files admission: an unreadable named shard refuses before family loading" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data =
        \\{"weight_map":{"x":"missing.safetensors"}}
    });
    var path: [64]u8 = undefined;
    const dir = try std.fmt.bufPrint(&path, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    var problem: []const u8 = "";
    const got = try weights(struct {}, a, io, .{ .dir = dir, .model_type = "test" }, &problem);
    defer if (problem.len > 0) a.free(problem);
    try std.testing.expectEqual(@as(?u64, null), got);
    try std.testing.expect(std.mem.indexOf(u8, problem, "refusing CUDA admission") != null);
}
