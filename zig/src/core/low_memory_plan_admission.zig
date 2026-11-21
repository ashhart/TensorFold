//! Serialized admission promises and two-phase plan replacement with rollback to the last admitted receipt.
const std = @import("std");
const planner = @import("low_memory_plan.zig");

pub const Lease = struct {
    bytes: u64 = 0,
    active: bool = false,
    owner: ?*Admission = null,
};

pub const Admission = struct {
    active: planner.Plan,
    pending: ?planner.Plan = null,
    promised: u64 = 0,
    leases: u32 = 0,

    pub fn init(plan: planner.Plan) Admission {
        return .{ .active = plan };
    }

    pub fn remaining(a: *const Admission) u64 {
        return a.active.deviceHeadroom() - a.promised;
    }

    pub fn prepare(
        a: *Admission,
        envelope: planner.Envelope,
        measured: planner.MeasuredLimits,
        model: planner.Model,
        placements: []const planner.Placement,
        request: planner.Request,
    ) !planner.Decision {
        if (a.pending != null) return error.PlanTransactionPending;
        if (!sameArtifact(a.active.model, model)) return error.ArtifactChanged;
        const decision = try planner.choose(envelope, measured, model, placements, request);
        switch (decision) {
            .refused => return decision,
            .admitted => |next| {
                if (a.leases != 0 and next.streams != a.active.streams) return error.ConcurrencyChangeWithLeases;
                if (a.promised > next.deviceHeadroom()) return error.AdmittedBytesExceedPlan;
                a.pending = next;
                return decision;
            },
        }
    }

    pub fn commit(a: *Admission, expected: [32]u8) !void {
        const next = a.pending orelse return error.NoPlanTransaction;
        if (!std.mem.eql(u8, &next.id, &expected)) return error.PlanReceiptChanged;
        if (a.leases != 0 and next.streams != a.active.streams) return error.ConcurrencyChangeWithLeases;
        if (a.promised > next.deviceHeadroom()) return error.AdmittedBytesExceedPlan;
        a.active = next;
        a.pending = null;
    }

    pub fn rollback(a: *Admission) void {
        a.pending = null;
    }

    pub fn promise(a: *Admission, lease: *Lease, bytes: u64) !void {
        if (a.pending != null) return error.PlanTransactionPending;
        if (lease.active) return error.LeaseAlreadyAdmitted;
        const promised = std.math.add(u64, a.promised, bytes) catch return error.AdmissionAccountingOverflow;
        if (promised > a.active.deviceHeadroom()) return error.LaneMemory;
        const leases = std.math.add(u32, a.leases, 1) catch return error.AdmissionAccountingOverflow;
        a.promised = promised;
        a.leases = leases;
        lease.* = .{ .bytes = bytes, .active = true, .owner = a };
    }

    pub fn finish(a: *Admission, lease: *Lease) !void {
        if (!lease.active) return;
        if (lease.owner != a) return error.ForeignLease;
        a.promised -= lease.bytes;
        a.leases -= 1;
        lease.* = .{};
    }
};

fn sameArtifact(a: planner.Model, b: planner.Model) bool {
    if (!std.mem.eql(u8, &a.checkpoint, &b.checkpoint) or a.weight_bytes != b.weight_bytes) return false;
    if ((a.drafter == null) != (b.drafter == null)) return false;
    if (a.drafter) |draft| {
        const other = b.drafter.?;
        return draft.weight_bytes == other.weight_bytes and std.mem.eql(u8, &draft.checkpoint, &other.checkpoint);
    }
    return true;
}
