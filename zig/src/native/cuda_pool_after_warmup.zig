//! Rebuild only an idle startup pool through the family's allocator, after the existing warm-up has completed.
const std = @import("std");
const api = @import("engine_api");
const cuda = @import("cuda");
const lanes = @import("lanes");
const memory = @import("cuda_family_memory.zig");
const budget = @import("cuda_memory.zig");
const streams = @import("cuda_streams.zig");
const qwen = @import("qwen35").native;
const Allocator = std.mem.Allocator;

/// Current pages stay resident at the probe: only whole pages of additional measured room can be added.
pub fn size(current: usize, useful: usize, page_bytes: u64, measured: budget.Pool, allocated: u64) !usize {
    if (page_bytes == 0 or useful < current) return error.InvalidPoolGeometry;
    if (measured.free < measured.reserve) return error.PoolReserveUnavailable;
    if (measured.limit) |limit| if (allocated > limit) return error.PoolReserveUnavailable;
    return current + @as(usize, @intCast(@min(useful - current, measured.room(allocated) / page_bytes)));
}

/// Calibration publishes through the same validated cost-table path used by normal synchronous startup.
pub fn prepare(served: anytype, gpa: Allocator, rows: u32, engine: anytype) !void {
    var cfg = try lanes.Config.init(gpa, served.facts, rows, rows - 1);
    defer cfg.deinit(gpa);
    try served.calibrate(gpa, &cfg);
    try engine.stream.synchronize();
}

/// No descriptors may be borrowed and no page may be leased, promised or retained before replacing an empty arena.
pub fn idle(l: anytype) !void {
    if (l.caches.count() != 0 or l.pool.items.len != 0 or l.prefix != null) return error.PoolInUse;
    const p = l.pages orelse return error.MissingPagePool;
    if (p.pool.available() != p.pool.total() or p.pool.promised != 0 or p.pool.reclaimable() != 0) return error.PoolInUse;
    for (p.lanes) |lane| if (lane.used) return error.PoolInUse;
}

pub fn replace(comptime F: type, l: anytype, count: u32, pages: usize) !void {
    try idle(l);
    const old = l.pages.?;
    l.pages = null;
    old.deinit();
    try F.attach(l, count - 1, pages);
}

/// Other modes retain their current startup order and capacity even when the switch is enabled.
pub fn apply(comptime F: type, a: Allocator, gpa: Allocator, io: std.Io, ctx: *const cuda.Context, loaded: anytype, o: api.Open, served: anytype, weights: u64, held0: u64, estimated: budget.Pool, counted: *streams.Count, report: *?api.CudaPoolSizing, problem: *[]const u8) !bool {
    if (!try @import("sys").environment.read(bool, "TF_CUDA_POOL_AFTER_WARMUP", false)) return true;
    if (comptime F == qwen) {
        if (!loaded.engine_paged or served.prepare == null or loaded.own_prompt_cache or counted.extra_device != 0 or
            o.tp != 1 or o.rank != 0 or o.slide or o.vision != null or (o.prompt_cache_gib orelse 1) != 0) return true;
        const l: *qwen.Lanes = @ptrCast(@alignCast(loaded.ctx));
        try idle(l);
        try prepare(served, gpa, loaded.rows, l.e);
        const measured = try memory.poolCached(a, io, ctx, weights, problem) orelse return false;
        const allocated = cuda.usage(false).device - held0;
        var p = counted.paged.?;
        const layout = counted.layout.?;
        // The existing planner's useful-page cap: one window a lane, plus the other lanes' state runs.
        const useful = @as(usize, p.lanes) * layout.pages(@as(u64, l.capacity) + F.window_rows) + (p.lanes - 1) * layout.runPages();
        const pages = size(p.pages, useful, layout.pageBytes(), measured, allocated) catch |err| {
            problem.* = try std.fmt.allocPrint(a, "TF_CUDA_POOL_AFTER_WARMUP cannot keep the current pool and reserve ({s}); free memory or lower --context", .{@errorName(err)});
            return false;
        };
        report.* = .{ .estimated_free_bytes = estimated.free, .measured_free_bytes = measured.free, .final_free_bytes = measured.free, .reserve_bytes = measured.reserve, .growth_room_bytes = measured.room(allocated), .estimated_pages = p.pages, .measured_pages = pages, .page_bytes = layout.pageBytes(), .page_tokens = layout.page_tokens };
        if (pages > p.pages) {
            try replace(F, l, p.lanes, pages);
            const final = try memory.poolCached(a, io, ctx, weights, problem) orelse return false;
            _ = size(pages, pages, layout.pageBytes(), final, cuda.usage(false).device - held0) catch {
                problem.* = "TF_CUDA_POOL_AFTER_WARMUP: memory changed during pool allocation; the reserve no longer fits";
                return false;
            };
            report.*.?.final_free_bytes = final.free;
            p.pages = pages;
            counted.paged = p;
        }
    }
    return true;
}

pub fn text(a: Allocator, report: ?api.CudaPoolSizing) ![]const u8 {
    const s = report orelse return "";
    const gib: f64 = 1 << 30;
    return std.fmt.allocPrint(a, "; pool after warm-up: estimated-plan free {d:.2} GiB before allocation, measured free {d:.2} GiB with initial pool resident, reserve {d:.2} GiB, pages {d} -> {d}", .{ @as(f64, @floatFromInt(s.estimated_free_bytes)) / gib, @as(f64, @floatFromInt(s.measured_free_bytes)) / gib, @as(f64, @floatFromInt(s.reserve_bytes)) / gib, s.estimated_pages, s.measured_pages });
}

test {
    _ = @import("cuda_pool_after_warmup_test.zig");
}
