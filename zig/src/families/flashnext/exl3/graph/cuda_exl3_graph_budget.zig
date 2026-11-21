//! The graph memory budget of the EXL3 decode graphs: admission, eviction and cohort retirement.

const std = @import("std");
const cuda = @import("cuda");
const collective = @import("../../cuda_collective.zig");
const memory = @import("cuda_exl3_graph_memory.zig");
const hub = @import("cuda_exl3_graph.zig");
const gcache = @import("cuda_exl3_graph_cache.zig");
const gprefill = @import("cuda_exl3_graph_prefill.zig");

pub fn freeMemory(stream: cuda.Stream) !u64 {
    var free: usize = 0;
    var total: usize = 0;
    try stream.d.check(stream.d.api.cuMemGetInfo_v2(&free, &total), "cuMemGetInfo");
    return free;
}

pub fn uploadResident(exec: *cuda.graph.Exec, stream: cuda.Stream) !void {
    try exec.upload(stream);
    try stream.synchronize();
}

pub fn reserveMemory(stream: cuda.Stream, class: memory.Class, cohort: u64) !memory.Ticket {
    _ = try memory.startupReservation();
    while (true) {
        return memory.admitWithQuery(class, cohort, stream, freeMemory) catch |err| switch (err) {
            error.GraphBudgetBusy, error.GraphMemoryHeadroom, error.GraphEvictionRequired => {
                if (cohort != 0) return err;
                try stream.d.check(stream.d.api.cuCtxSynchronize(), "cuCtxSynchronize(graph retirement)");
                if (!evictOne()) return err;
                continue;
            },
            else => return err,
        };
    }
}

pub fn retirementBarrier(c: *collective.Collective, signature: u64, cohort: u64) !void {
    const decision = try c.graphAgree(signature, .{ .state = .ready, .cohort = cohort });
    if (decision.choice != .replay) return error.GraphCohortMismatch;
}

pub fn oldestCohort() u64 {
    var cohort: u64 = 0;
    var used: u64 = std.math.maxInt(u64);
    for (gcache.entries) |maybe| if (maybe) |entry| {
        if (entry.cohort != 0 and (entry.used < used or (entry.used == used and (cohort == 0 or entry.cohort < cohort)))) {
            cohort = entry.cohort;
            used = entry.used;
        }
    };
    return cohort;
}

pub fn retireCohort(cohort: u64) void {
    for (&gcache.entries) |*maybe| if (maybe.*) |*entry| {
        if (entry.cohort == cohort) {
            gcache.drop(entry);
            maybe.* = null;
        }
    };
}

pub fn reserveTogether(c: *collective.Collective, stream: cuda.Stream, signature: u64, cohort: u64) !memory.Ticket {
    _ = try memory.startupReservation();
    var retired: usize = 0;
    while (true) {
        var ticket: ?memory.Ticket = memory.admitWithQuery(.decode, cohort, stream, freeMemory) catch |err| switch (err) {
            error.GraphBudgetBusy, error.GraphBudgetExceeded, error.GraphMemoryHeadroom, error.GraphEvictionRequired => null,
            else => return err,
        };
        errdefer if (ticket) |held| memory.release(held) catch unreachable;
        var free_slot = false;
        for (gcache.entries) |entry| free_slot = free_slot or entry == null;
        const agreement = try c.graphAgree(signature ^ 0x61646d6974, .{ .state = if (ticket != null and free_slot) .ready else .eager, .cohort = cohort });
        if (agreement.choice == .replay) return ticket.?;
        if (ticket) |held| {
            try memory.release(held);
            ticket = null;
        }
        const victim = try c.graphVictim(signature ^ 0x76696374696d, oldestCohort());
        if (victim == 0) {
            try stream.d.check(stream.d.api.cuCtxSynchronize(), "cuCtxSynchronize(TP local graph retirement)");
            try retirementBarrier(c, signature ^ 0x6c6f63616c7175, cohort);
            const reclaimed = evictOne();
            const any_reclaimed = try c.graphVictim(signature ^ 0x6c6f63616c646f, @intFromBool(reclaimed));
            if (any_reclaimed == 0) return error.GraphBudgetExceeded;
            continue;
        }
        if (retired >= gcache.entries.len * 2) return error.GraphBudgetExceeded;
        try stream.d.check(stream.d.api.cuCtxSynchronize(), "cuCtxSynchronize(TP graph retirement)");
        try retirementBarrier(c, signature ^ 0x71756965736365, victim);
        retireCohort(victim);
        try retirementBarrier(c, signature ^ 0x72657469726564, victim);
        retired += 1;
    }
}

pub fn evictOne() bool {
    var oldest: u64 = std.math.maxInt(u64);
    var decode: ?usize = null;
    var prefill: ?usize = null;
    for (gcache.entries, 0..) |maybe, index| if (maybe) |entry| {
        if (entry.cohort == 0 and entry.used < oldest) {
            oldest = entry.used;
            decode = index;
        }
    };
    for (gprefill.pentries, 0..) |maybe, index| if (maybe) |entry| {
        if (entry.used < oldest) {
            oldest = entry.used;
            decode = null;
            prefill = index;
        }
    };
    if (decode) |index| {
        gcache.drop(&gcache.entries[index].?);
        gcache.entries[index] = null;
        return true;
    }
    if (prefill) |index| {
        const entry = &gprefill.pentries[index].?;
        gprefill.dropPrefill(entry);
        gprefill.pentries[index] = null;
        return true;
    }
    return @import("../model/cuda_exl3_chain.zig").evictOne();
}
