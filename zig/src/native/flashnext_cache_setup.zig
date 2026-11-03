//! Flash Next's startup prompt-cache budget and learned-state store.
const std = @import("std");
const Allocator = std.mem.Allocator;
const api = @import("engine_api");
const fx = @import("tensorfold").flashnext_engine;
const cache_fit = @import("cache_fit.zig");
const disk = @import("flashnext_disk.zig");

/// The prompt cache's plan (cache_fit.zig); without an OS reading, `gib` or the Metal working set's spare.
pub fn cacheBudget(eng: *fx.Engine, gib: ?f64, over: bool, limit: ?u64, a: Allocator, why: *[]const u8) !api.PromptCachePlan {
    const rank = if (eng.followsPeer()) " (rank 1 keeps its halves of rank 0's)" else if (eng.r.tp != null) " (rank 1 mirrors them)" else "";
    const dev = eng.r.device;
    if (limit) |cap| {
        const ready = cache_fit.footprint() orelse return error.NoMemoryReading;
        if (ready +| cache_fit.MARGIN > cap) return error.OverMemoryLimit;
        return cache_fit.budgetUnder(cap, gib, over, dev.maxWorkingSet() -| dev.allocated(), a, why, rank);
    }
    return cache_fit.budget(gib, over, dev.maxWorkingSet() -| dev.allocated() -| (8 << 30), a, why, rank);
}

/// Learned states under `root` by checkpoint, kernel sources and modes, probe bits, OS build and chip.
pub fn learnedStates(gpa: Allocator, io: std.Io, eng: *fx.Engine, root: []const u8, cap: u64) !api.prompt_imprint.Imprint {
    const m = try api.prompt_imprint.Imprint.open(gpa, root, (try disk.identity(gpa, io, eng)).id, cap);
    std.log.info("flash next: learned prompt states in {s} ({d} known, {d} of {d} MiB)", .{ m.dir, m.metas.items.len, m.total() >> 20, cap >> 20 });
    return m;
}
