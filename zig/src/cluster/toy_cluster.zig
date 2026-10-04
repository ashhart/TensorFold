//! In-process toy clusters for tests: rank 0 leads in the caller's thread, the others follow on threads.
const std = @import("std");
const fabric = @import("fabric");
const exchange = @import("exchange.zig");
const round = @import("round.zig");
const toy = @import("toy.zig");
const backend = @import("backend.zig");
const fabric_link = @import("fabric_link.zig");

/// How the ranks compute and exchange: every variant but `naive` gives the same bits.
pub const Variant = struct {
    naive: bool = false,
    split_sums: bool = false,
    by_streams: bool = false,
    fabric: bool = false,
    /// Thunderbolt's two-sided links (the fabric's fake, messages delivered in random interleavings by a pump thread).
    two_sided: bool = false,
    split_experts: bool = false,
};

pub fn shardOf(r: *toy.Rank) backend.Shard {
    return .{ .ptr = r, .vtable = &.{ .forward = forward, .keep = keep, .release = release, .vocab = vocab } };
}

fn forward(ptr: *anyopaque, a: std.mem.Allocator, x: exchange.Exchange, rows: []const round.Row, logits: ?[]f32) anyerror!void {
    const r: *toy.Rank = @ptrCast(@alignCast(ptr));
    return r.forward(a, x, rows, logits);
}

fn keep(ptr: *anyopaque, slot: u32, len: u32) void {
    const r: *toy.Rank = @ptrCast(@alignCast(ptr));
    r.keep(slot, len);
}

fn release(ptr: *anyopaque, slot: u32) void {
    const r: *toy.Rank = @ptrCast(@alignCast(ptr));
    r.release(slot);
}

fn vocab(ptr: *anyopaque) u32 {
    const r: *toy.Rank = @ptrCast(@alignCast(ptr));
    return r.w.dims.vocab;
}

pub const Cluster = struct {
    gpa: std.mem.Allocator,
    mem: *exchange.Mem,
    fab: ?*fabric.fake.Cluster = null,
    links: []fabric_link.Link = &.{},
    sr: ?*fabric.sendrecv_fake.Cluster = null,
    sides: []fabric_link.TwoSided = &.{},
    pump: ?std.Thread = null,
    stop: std.atomic.Value(bool) = .init(false),
    ranks: []toy.Rank,
    threads: []std.Thread,
    leader: backend.Leader,

    fn endpoint(c: *Cluster, r: u32) exchange.Exchange {
        if (c.sr != null) return c.sides[r].steps();
        return if (c.fab != null) c.links[r].steps() else c.mem.endpoint(r);
    }

    pub fn init(gpa: std.mem.Allocator, w: *const toy.Weights, n: u32, v: Variant) !*Cluster {
        const c = try gpa.create(Cluster);
        c.* = .{ .gpa = gpa, .mem = try exchange.Mem.init(gpa, n), .ranks = &.{}, .threads = &.{}, .leader = undefined };
        if (v.fabric) {
            const layout = fabric_link.Layout.init(n, 1 << 20, 4096);
            c.fab = try fabric.fake.Cluster.init(gpa, n, layout.total, .immediate, null);
            c.links = try gpa.alloc(fabric_link.Link, n);
            for (c.links, 0..) |*l, i| l.* = try fabric_link.Link.init(gpa, c.fab.?.endpoint(@intCast(i)), layout);
        }
        if (v.two_sided) {
            c.sr = try fabric.sendrecv_fake.Cluster.init(gpa, n, true);
            c.sides = try gpa.alloc(fabric_link.TwoSided, n);
            for (c.sides, 0..) |*t, i| t.* = fabric_link.TwoSided.init(c.sr.?.link(@intCast(i)), c.sr.?.link(@intCast(i)));
            c.pump = try std.Thread.spawn(.{}, fabric.sendrecv_fake.Cluster.pump, .{ c.sr.?, &c.stop });
        }
        c.ranks = try gpa.alloc(toy.Rank, n);
        for (c.ranks, 0..) |*r, i| {
            r.* = try toy.Rank.init(gpa, w, @intCast(i), n);
            r.naive = v.naive;
            r.split_sums = v.split_sums;
            r.by_streams = v.by_streams;
            r.split_experts = v.split_experts;
        }
        c.threads = try gpa.alloc(std.Thread, n - 1);
        for (c.threads, 1..) |*t, i| t.* = try std.Thread.spawn(.{}, follow, .{ gpa, c.endpoint(@intCast(i)), &c.ranks[i] });
        c.leader = .{ .gpa = gpa, .x = c.endpoint(0), .shard = shardOf(&c.ranks[0]) };
        return c;
    }

    fn follow(gpa: std.mem.Allocator, x: exchange.Exchange, r: *toy.Rank) void {
        backend.follow(gpa, x, shardOf(r)) catch |err| std.debug.panic("follower: {s}", .{@errorName(err)});
    }

    pub fn deinit(c: *Cluster) void {
        const gpa = c.gpa;
        c.leader.stop() catch unreachable;
        for (c.threads) |t| t.join();
        c.leader.deinit();
        for (c.ranks) |*r| r.deinit();
        gpa.free(c.ranks);
        gpa.free(c.threads);
        c.mem.deinit();
        for (c.links) |*l| l.deinit();
        if (c.fab) |f| {
            gpa.free(c.links);
            f.deinit();
        }
        if (c.sr) |f| {
            c.stop.store(true, .release);
            c.pump.?.join();
            gpa.free(c.sides);
            f.deinit();
        }
        gpa.destroy(c);
    }
};
