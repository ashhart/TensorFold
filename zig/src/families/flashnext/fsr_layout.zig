//! Whole-path residency prices computed before device buffers or checkpoint payloads are loaded.
const std = @import("std");
pub const plan = @import("../../core/fsr/plan.zig");
const prompt = @import("flashnext_memory").prompt;
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
