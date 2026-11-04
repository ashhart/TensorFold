const std = @import("std");

pub const Class = enum { decode, prefill, chain };
pub const Charge = struct {
    exec: u64,
    template: u64,
    transient: u64,

    pub fn peak(self: Charge) !u64 {
        return std.math.add(u64, try std.math.add(u64, self.exec, self.template), self.transient);
    }
};

pub fn Budget(comptime capacity: usize) type {
    return struct {
        const Self = @This();
        pub const Ticket = struct { slot: usize, generation: u64 };
        const Entry = struct {
            generation: u64,
            cohort: u64,
            class: Class,
            charge: Charge,
            phase: enum { pending, retained, retiring } = .pending,
            quiescent: bool = false,
            used: u64,
        };
        pub const Plan = struct {
            victims: [capacity]Ticket = undefined,
            len: usize = 0,
        };

        limit: u64,
        headroom: u64,
        entries: [capacity]?Entry = @splat(null),
        generation: u64 = 0,

        pub fn init(reservation: u64, headroom: u64) !Self {
            if (reservation <= headroom) return error.InvalidGraphReservation;
            return .{ .limit = reservation - headroom, .headroom = headroom };
        }

        pub fn charged(self: *const Self) u64 {
            var total: u64 = 0;
            for (self.entries) |maybe| if (maybe) |entry| {
                total += entry.charge.peak() catch unreachable;
            };
            return total;
        }

        pub fn pending(self: *const Self) u64 {
            var total: u64 = 0;
            for (self.entries) |maybe| if (maybe) |entry| {
                if (entry.phase == .pending) total += entry.charge.peak() catch unreachable;
            };
            return total;
        }

        pub fn localCharged(self: *const Self) u64 {
            var total: u64 = 0;
            for (self.entries) |maybe| if (maybe) |entry| {
                if (entry.cohort == 0) total += entry.charge.peak() catch unreachable;
            };
            return total;
        }

        pub fn plan(self: *const Self, charge: Charge) !Plan {
            const needed = try charge.peak();
            if (charge.exec == 0 or needed > self.limit) return error.GraphBudgetExceeded;
            var result: Plan = .{};
            var remaining = self.charged();
            var free_slot = false;
            var selected: [capacity]bool = @splat(false);
            for (self.entries) |entry| free_slot = free_slot or entry == null;
            while (!free_slot or needed > self.limit - remaining) {
                var oldest: ?usize = null;
                for (self.entries, 0..) |maybe, index| if (maybe) |entry| {
                    if (selected[index] or entry.phase != .retained or !entry.quiescent) continue;
                    if (oldest == null or entry.used < self.entries[oldest.?].?.used) oldest = index;
                };
                const index = oldest orelse return error.GraphBudgetBusy;
                const entry = self.entries[index].?;
                result.victims[result.len] = .{ .slot = index, .generation = entry.generation };
                result.len += 1;
                selected[index] = true;
                remaining -= try entry.charge.peak();
                free_slot = true;
            }
            return result;
        }

        pub fn reserve(self: *Self, class: Class, cohort: u64, charge: Charge, used: u64) !Ticket {
            const proposed = try self.plan(charge);
            if (proposed.len != 0) return error.GraphEvictionRequired;
            const generation = try std.math.add(u64, self.generation, 1);
            for (&self.entries, 0..) |*entry, index| if (entry.* == null) {
                entry.* = .{ .generation = generation, .cohort = cohort, .class = class, .charge = charge, .used = used };
                self.generation = generation;
                return .{ .slot = index, .generation = generation };
            };
            unreachable;
        }

        fn get(self: *Self, ticket: Ticket) !*Entry {
            if (ticket.slot >= capacity) return error.StaleGraphTicket;
            const entry = &(self.entries[ticket.slot] orelse return error.StaleGraphTicket);
            if (entry.generation != ticket.generation) return error.StaleGraphTicket;
            return &self.entries[ticket.slot].?;
        }

        pub fn installed(self: *Self, ticket: Ticket) !void {
            const entry = try self.get(ticket);
            if (entry.phase != .pending) return error.InvalidGraphPhase;
            entry.phase = .retained;
            entry.charge.transient = 0;
        }

        pub fn getForRelease(self: *Self, ticket: Ticket) !bool {
            return (try self.get(ticket)).phase == .retained;
        }

        pub fn cohortOf(self: *Self, ticket: Ticket) !u64 {
            return (try self.get(ticket)).cohort;
        }

        pub fn touch(self: *Self, ticket: Ticket, used: u64, quiescent: bool) !void {
            const entry = try self.get(ticket);
            if (entry.phase != .retained) return error.InvalidGraphPhase;
            entry.used = used;
            entry.quiescent = quiescent;
        }

        pub fn beginRetire(self: *Self, ticket: Ticket, agreed_cohort: u64) !void {
            const entry = try self.get(ticket);
            if (entry.cohort != agreed_cohort) return error.GraphCohortMismatch;
            if (entry.phase != .retained or !entry.quiescent) return error.GraphBudgetBusy;
            entry.phase = .retiring;
        }

        pub fn destroyed(self: *Self, ticket: Ticket) !void {
            const entry = try self.get(ticket);
            if (entry.phase != .retiring and entry.phase != .pending) return error.InvalidGraphPhase;
            self.entries[ticket.slot] = null;
        }
    };
}

test "exl3: graph admission reserves pending peak and explicit headroom" {
    var budget = try Budget(4).init(100, 20);
    const ticket = try budget.reserve(.decode, 7, .{ .exec = 30, .template = 10, .transient = 20 }, 1);
    try std.testing.expectEqual(@as(u64, 60), budget.charged());
    try std.testing.expectError(error.GraphBudgetBusy, budget.plan(.{ .exec = 30, .template = 0, .transient = 0 }));
    try budget.installed(ticket);
    try std.testing.expectEqual(@as(u64, 40), budget.charged());
    try std.testing.expectError(error.GraphBudgetExceeded, budget.plan(.{ .exec = 81, .template = 0, .transient = 0 }));
}

test "exl3: pre eviction requires quiescence cohort agreement and completed destruction" {
    var budget = try Budget(2).init(120, 20);
    const first = try budget.reserve(.prefill, 11, .{ .exec = 40, .template = 10, .transient = 0 }, 1);
    try budget.installed(first);
    const second = try budget.reserve(.chain, 12, .{ .exec = 40, .template = 10, .transient = 0 }, 2);
    try budget.installed(second);
    try std.testing.expectError(error.GraphBudgetBusy, budget.plan(.{ .exec = 50, .template = 0, .transient = 0 }));
    try budget.touch(first, 1, true);
    const plan = try budget.plan(.{ .exec = 50, .template = 0, .transient = 0 });
    try std.testing.expectEqual(@as(usize, 1), plan.len);
    try std.testing.expectEqual(first, plan.victims[0]);
    try std.testing.expectError(error.GraphCohortMismatch, budget.beginRetire(first, 12));
    try budget.beginRetire(first, 11);
    try std.testing.expectEqual(@as(u64, 100), budget.charged());
    try std.testing.expectError(error.GraphBudgetBusy, budget.reserve(.decode, 13, .{ .exec = 50, .template = 0, .transient = 0 }, 3));
    try budget.destroyed(first);
    const replacement = try budget.reserve(.decode, 13, .{ .exec = 50, .template = 0, .transient = 0 }, 3);
    try std.testing.expectError(error.StaleGraphTicket, budget.destroyed(first));
    try budget.destroyed(replacement);
    try std.testing.expectEqual(@as(u64, 50), budget.charged());
}

test "exl3: graph estimates reject overflow and invalid reservations" {
    try std.testing.expectError(error.InvalidGraphReservation, Budget(1).init(10, 10));
    var budget = try Budget(1).init(100, 10);
    try std.testing.expectError(error.Overflow, budget.plan(.{ .exec = std.math.maxInt(u64), .template = 1, .transient = 0 }));
    try std.testing.expectError(error.GraphBudgetExceeded, budget.plan(.{ .exec = 0, .template = 0, .transient = 0 }));
}

test "exl3: cross class LRU planning is read only and launch revokes retirement eligibility" {
    var budget = try Budget(3).init(130, 10);
    const charge: Charge = .{ .exec = 30, .template = 10, .transient = 0 };
    const decode = try budget.reserve(.decode, 1, charge, 3);
    const prefill = try budget.reserve(.prefill, 2, charge, 1);
    const chain = try budget.reserve(.chain, 3, charge, 2);
    for ([_]Budget(3).Ticket{ decode, prefill, chain }, [_]u64{ 3, 1, 2 }) |ticket, used| {
        try budget.installed(ticket);
        try budget.touch(ticket, used, true);
    }
    const request: Charge = .{ .exec = 70, .template = 0, .transient = 0 };
    const plan = try budget.plan(request);
    try std.testing.expectEqual(@as(usize, 2), plan.len);
    try std.testing.expectEqual(prefill, plan.victims[0]);
    try std.testing.expectEqual(chain, plan.victims[1]);
    try std.testing.expectEqual(@as(u64, 120), budget.charged());
    try std.testing.expectError(error.GraphEvictionRequired, budget.reserve(.decode, 4, request, 4));
    try budget.touch(prefill, 4, false);
    try std.testing.expectError(error.GraphBudgetBusy, budget.beginRetire(prefill, 2));
    try std.testing.expectError(error.InvalidGraphPhase, budget.destroyed(chain));
}
