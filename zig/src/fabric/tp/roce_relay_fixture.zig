//! Deterministic actors over actual verbs fakes and shared-memory posts, with consumers advancing independently.
const std = @import("std");
const t = std.testing;
const Fabric = @import("roce_multi_fake.zig").Fabric;
const Shm = @import("roce_multi_shm.zig").ShmWire;
const relay = @import("roce_relay.zig");
const guard = @import("roce_multi_guard.zig");
pub const Harness = struct {
    fabric: Fabric = undefined,
    states: [4]relay.Relay = undefined,
    shm: [4]Shm = undefined,
    oracle: [4]Shm = undefined,
    memory: [4][]align(4096) u8 = undefined,
    consumed: [4]u32 = @splat(0),
    doorbell: [4]u32 = @splat(0),
    sizes: [4]u32 = @splat(16),
    ticks: u64 = 0,
    pub fn init(h: *Harness, slot: usize) !void {
        h.* = .{};
        try h.fabric.init(4, 2, slot);
        h.fabric.ring();
        h.fabric.deferred = true;
        var records: [4]@import("roce_multi_verbs.zig").Record = undefined;
        for (0..4) |r| {
            h.fabric.wires[r].relay = true;
            records[r] = h.fabric.wires[r].record();
            h.memory[r] = try t.allocator.alignedAlloc(u8, .fromByteUnits(4096), h.fabric.layout.len);
            @memset(h.memory[r], 0);
        }
        for (0..4) |r| {
            try h.fabric.wires[r].connect(&records);
            h.states[r] = try relay.Relay.init(@intCast(r), h.fabric.layout, h.fabric.regions[r], 10000);
            var regions: [8][]u8 = undefined;
            var oracle: [8][]u8 = undefined;
            for (0..4) |p| {
                regions[p] = h.fabric.regions[p];
                oracle[p] = h.memory[p];
            }
            h.shm[r] = .{ .regions = regions, .local = regions[r], .me = @intCast(r), .layout = h.fabric.layout, .stripes = 2 };
            h.oracle[r] = .{ .regions = oracle, .local = oracle[r], .me = @intCast(r), .layout = h.fabric.layout, .stripes = 2 };
        }
    }
    pub fn deinit(h: *Harness) void {
        h.fabric.deinit();
        for (h.memory) |m| t.allocator.free(m);
    }
    pub fn stage(h: *Harness, rank: u32, seq: u32, n: u32) !void {
        // This is the existing stream contract: a producer cannot begin another collective before consuming its own.
        if (h.doorbell[rank] != h.consumed[rank]) return error.ConsumerStillReading;
        if (seq != h.doorbell[rank] +% 1) return error.WrongSequence;
        h.doorbell[rank] = seq;
        h.sizes[rank] = n;
        const at = h.fabric.layout.sendAt(seq & 1);
        for (0..n / 4) |i| {
            const value = pattern(seq, rank, @intCast(i));
            std.mem.writeInt(u32, h.fabric.regions[rank][at + i * 4 ..][0..4], value, .little);
            std.mem.writeInt(u32, h.memory[rank][at + i * 4 ..][0..4], value, .little);
        }
        try h.oracle[rank].post(seq & 1, seq, n);
    }
    pub fn tick(h: *Harness, comptime verbs: bool, rank: u32) !void {
        h.ticks += 1;
        const wire = if (verbs) &h.fabric.wires[rank] else &h.shm[rank];
        _ = try h.states[rank].progress(wire, h.doorbell[rank], h.consumed[rank], h.sizes[rank], h.ticks);
    }
    pub fn ready(h: *const Harness, rank: u32, seq: u32) bool {
        for (0..4) |src| if (src != rank) for (0..2) |d| {
            const at = h.fabric.layout.flagAt(@intCast(src), seq & 1, @intCast(d));
            const flag: *std.atomic.Value(u32) = @ptrCast(@alignCast(h.fabric.regions[rank][at..].ptr));
            if (flag.load(.acquire) != seq) return false;
        };
        return true;
    }
    pub fn check(h: *Harness, rank: u32, seq: u32) !void {
        try t.expect(h.ready(rank, seq));
        for (0..4) |src| if (src != rank) {
            const at = h.fabric.layout.recvAt(@intCast(src), seq & 1);
            const n = h.sizes[src];
            for (0..n / 4) |i| {
                const got = std.mem.readInt(u32, h.fabric.regions[rank][at + i * 4 ..][0..4], .little);
                if (got != pattern(seq, @intCast(src), @intCast(i))) return error.StaleWord;
            }
            try t.expectEqualSlices(u8, h.memory[rank][at..][0..n], h.fabric.regions[rank][at..][0..n]);
        };
        h.consumed[rank] = seq;
    }
};
pub fn pattern(seq: u32, source: u32, index: u32) u32 {
    var x = seq *% 0x9e37_79b9 ^ source *% 0x85eb_ca6b ^ index *% 0xc2b2_ae35;
    x ^= x >> 16;
    return x *% 0x27d4_eb2d;
}
