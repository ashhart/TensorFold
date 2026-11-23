//! Whole-path residency prices computed before device buffers or checkpoint payloads are loaded.
const std = @import("std");
pub const plan = @import("../../core/fsr/plan.zig");
const memory = @import("flashnext_memory");
const prompt = memory.prompt;
pub const Plan = struct { pool: ?plan.Pool = null, ple: ?u64 = null, raw: u64, extra: u64 = 0 };

pub fn of(inventory: anytype, settings: anytype) !Plan {
    var p: Plan = .{ .raw = inventory.raw, .extra = if (settings.ssd_experts != null or settings.ple_on_ssd or settings.fsr_trace != null) 1024 * 1024 else 0 };
    if (settings.ssd_experts) |bytes| {
        // One full layer's union fits at every admitted prompt and concurrent window width.
        p.pool = try plan.pool(bytes, inventory.expert_parts, 49 * 512, 512);
        // Bounded demand IDs, forecast logits/routes and metadata for the source descriptor owner.
        p.extra = 1024 * 1024;
    }
    if (settings.ple_on_ssd) {
        p.ple = try std.math.add(u64, try plan.pleRows(prompt.PMAX * 16, 160, 8, 32), 128 * 3 * 32);
    }
    p.raw = try plan.resident(inventory.raw, inventory.routed, inventory.ple, p.pool, p.ple);
    return p;
}

/// Only a refused maximum shrinks; every admitted setting and every impossible fit keeps its original policy.
pub fn fit(inventory: anytype, settings: anytype, context: anytype) !@TypeOf(settings) {
    if (context.paired or settings.ssd_experts == null) return settings;
    const current = try of(inventory, settings);
    if ((try budget(inventory, settings, context)).forPhase(context.needs_build) <= context.limit) return settings;
    const pool = current.pool.?;
    var next = settings;
    next.ssd_experts = try std.math.mul(u64, 512, pool.stride);
    var growth: u64 = if (@hasField(@TypeOf(context), "rotary_growth")) context.rotary_growth else 0;
    if (!try fits(inventory, next, context, growth)) {
        if (!try fits(inventory, next, context, 0)) return settings;
        growth = 0; // the complete-demand minimum still admits when the optional cache cannot fit
    }
    var lo: usize = 512;
    var hi = pool.slots;
    while (lo < hi) {
        const mid = lo + (hi - lo + 1) / 2;
        next.ssd_experts = try std.math.mul(u64, mid, pool.stride);
        if (try fits(inventory, next, context, growth))
            lo = mid
        else
            hi = mid - 1;
    }
    next.ssd_experts = try std.math.mul(u64, lo, pool.stride);
    return next;
}

fn budget(inventory: anytype, settings: anytype, context: anytype) !memory.Budget {
    const p = try of(inventory, settings);
    var resident = inventory;
    resident.raw = p.raw;
    const extra = try std.math.add(u64, context.extra, try std.math.add(u64, context.existing, p.extra));
    return memory.budgetLiveFor(0, settings.capacity, resident, extra, context.existing, settings.sibling(context.paired));
}

fn fits(inventory: anytype, settings: anytype, context: anytype, growth: u64) !bool {
    var b = try budget(inventory, settings, context);
    b.steady = try std.math.add(u64, b.steady, growth);
    return b.forPhase(context.needs_build) <= context.limit;
}
