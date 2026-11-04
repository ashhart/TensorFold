const std = @import("std");
const telemetry = @import("../model/exl3_telemetry.zig");

pub const Source = enum { demand, initial_prefetch, cross_layer_prefetch };
pub const Counts = struct {
    copies: u64 = 0,
    requested_bytes: u64 = 0,
    unused_copies: u64 = 0,
    unused_requested_bytes: u64 = 0,
};
pub const Totals = [3]Counts;

const Residency = struct { source: Source, bytes: u64 };

pub const State = struct {
    slots: []Residency,
    totals: Totals = @splat(.{}),

    pub fn init(allocator: std.mem.Allocator, slots: usize, enabled: bool) !?State {
        if (!enabled) return null;
        return .{ .slots = try allocator.alloc(Residency, slots) };
    }

    pub fn deinit(state: *State, allocator: std.mem.Allocator) void {
        allocator.free(state.slots);
    }

    pub fn copied(state: *State, slot: usize, source: Source, bytes: u64) void {
        state.slots[slot] = .{ .source = source, .bytes = bytes };
        const counts = &state.totals[@backingInt(source)];
        counts.copies += 1;
        counts.requested_bytes += bytes;
    }

    pub fn unused(state: *State, slot: usize) void {
        const residency = state.slots[slot];
        const counts = &state.totals[@backingInt(residency.source)];
        counts.unused_copies += 1;
        counts.unused_requested_bytes += residency.bytes;
    }
};

pub fn publish(totals: Totals, experts: *telemetry.Experts) void {
    inline for (std.enums.values(Source), 0..) |source, index| {
        inline for (@typeInfo(Counts).@"struct".field_names) |field| {
            @field(experts.*, "ring_" ++ @tagName(source) ++ "_" ++ field) = @field(totals[index], field);
        }
    }
}
