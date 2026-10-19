//! TF_DSV41_ARENA (default off): device buffers carved from a few large chunks instead of one cuMemAlloc each.

const std = @import("std");
const quiet = @import("builtin").is_test; // a test's stderr is the build runner's channel
const cuda = @import("cuda");

const log = std.log.scoped(.dsv41);

pub const Mode = enum { off, vmm, plain, pack }; // 1 / `vmm`: CUDA VMM (else `plain`); `plain`: one cuMemAlloc a chunk

/// TF_CUDA_SCHED (unset: driver `auto`): how the host waits for the GPU; host scheduling only, same launches and bits.
pub fn schedule(get: *const fn ([]const u8) ?[]const u8) !?cuda.Context.Schedule {
    const raw = std.mem.trim(u8, get("TF_CUDA_SCHED") orelse return null, " \t");
    if (raw.len == 0) return null;
    return std.meta.stringToEnum(cuda.Context.Schedule, raw) orelse {
        if (!quiet) log.warn("TF_CUDA_SCHED={s}: expected auto, spin, yield or blocking", .{raw});
        return error.BadKnob;
    };
}

/// Sets TF_CUDA_SCHED on `ordinal`'s primary context; call before cuda.Context.init.
pub fn applySchedule(d: *const cuda.Driver, ordinal: c_int) !void {
    const s = try schedule(&env) orelse return;
    try cuda.Context.setSchedule(d, ordinal, s);
    log.info("cuda schedule: {t}", .{s});
}

pub fn mode(get: *const fn ([]const u8) ?[]const u8) !Mode {
    const raw = std.mem.trim(u8, get("TF_DSV41_ARENA") orelse return .off, " \t");
    if (raw.len == 0 or std.mem.eql(u8, raw, "0")) return .off;
    if (std.mem.eql(u8, raw, "1") or std.mem.eql(u8, raw, "vmm")) return .vmm;
    if (std.mem.eql(u8, raw, "plain")) return .plain;
    // boot's permanent buffers packed (cuda/pack.zig); the rest as without an arena
    if (std.mem.eql(u8, raw, "pack")) return .pack;
    if (!quiet) log.warn("TF_DSV41_ARENA={s}: expected 0, 1 / vmm, plain or pack", .{raw});
    return error.BadKnob;
}

/// The arena and its backend, for the process's lifetime (never freed: buffers it holds may outlive any owner).
pub const Installed = struct {
    vmm: cuda.arena.Vmm = undefined,
    plain: cuda.arena.Plain,
    arena: cuda.arena.Arena = undefined,
    backend: Mode,
    /// TF_DSV41_ARENA=pack: open from install to the end of boot (`endBoot`)
    pack: ?cuda.pack.Pack = null,
    vmm_open: bool = false,

    pub fn detachHook(x: *Installed) void {
        if (x.pack) |*p| p.uninstall() else x.arena.uninstall();
    }
    // Called after every allocation from this owner is freed and the context is drained.
    pub fn deinit(x: *Installed, gpa: std.mem.Allocator) void {
        if (x.pack) |*p| {
            for (p.segments.items) |s| {
                const be = if (x.vmm_open and s[0] >= x.vmm.va and s[0] < x.vmm.va + x.vmm.va_len) x.vmm.backend() else x.plain.backend();
                be.release(be.ctx, s[0], @intCast(s[1]));
            }
            p.deinit();
        } else x.arena.deinit();
        if (x.vmm_open) x.vmm.close();
        gpa.destroy(x);
    }
    /// Boot's buffers are in: later ones (workspaces, images, sessions) are allocated and freed as before.
    pub fn endBoot(x: *Installed, rank: u32) void {
        const p = &(x.pack orelse return);
        p.end();
        const gib = @as(f64, @floatFromInt(p.stats.bytes)) / (1 << 30);
        const slack = @as(f64, @floatFromInt(p.slack())) / (1 << 20);
        // a buffer freed during boot stays mapped to the end (`freed`): a count that should stay small
        log.info("device arena: rank {d}: {d} boot buffers packed, {d:.2} GiB in {d} mapping(s), {d:.1} MiB slack, {d} freed during boot", .{ rank, p.stats.buffers, gib, p.segments.items.len, slack, p.stats.freed });
        if (x.vmm_open) if (x.vmm.failed) |f| log.warn("device arena: rank {d}: VMM's chunks failed ({s}: CUresult {d}); packed on cuMemAlloc segments of {d} MiB", .{ rank, f.call, f.code, p.seg_min >> 20 });
        if (p.stats.buffers == 0) log.warn("device arena: rank {d}: TF_DSV41_ARENA=pack packed nothing: every boot buffer went to cuMemAlloc", .{rank});
    }
};

fn env(name: []const u8) ?[]const u8 {
    return @import("serve_knobs.zig").env(name);
}

/// Reads the knob and installs the arena under DeviceBuffer.alloc; null when off.
pub fn install(gpa: std.mem.Allocator, ctx: *const cuda.Context, rank: u32) !?*Installed {
    const m = try mode(&env);
    if (m == .off) return null;
    const d = ctx.d;
    const x = try gpa.create(Installed);
    errdefer gpa.destroy(x);
    x.* = .{ .plain = .{ .d = d }, .backend = .plain };
    if (m == .pack) {
        var total: usize = 0;
        try d.check(d.api.cuDeviceTotalMem_v2(&total, ctx.device), "cuDeviceTotalMem");
        x.backend = .pack;
        // a mapping of its own, so every growth follows the last; where VMM fails (logged): cuMemAlloc segments
        if (cuda.arena.Vmm.open(d, ctx.device, 2 * total)) |v| {
            x.vmm = v;
            x.vmm_open = true;
            x.pack = cuda.pack.Pack.init(gpa, x.vmm.backend());
            x.pack.?.fallback = x.plain.backend();
        } else |e| {
            log.warn("device arena: rank {d}: no CUDA virtual memory ({t}): packed on cuMemAlloc segments", .{ rank, e });
            x.pack = cuda.pack.Pack.init(gpa, x.plain.backend());
            x.pack.?.contiguous = false;
        }
        x.pack.?.install();
        x.pack.?.begin();
        log.info("device arena: rank {d}: pack, {d} KiB granularity, boot's buffers back to back", .{ rank, x.pack.?.be.granularity >> 10 });
        return x;
    }
    var be = x.plain.backend();
    if (m == .vmm) {
        var total: usize = 0;
        try d.check(d.api.cuDeviceTotalMem_v2(&total, ctx.device), "cuDeviceTotalMem");
        if (cuda.arena.Vmm.open(d, ctx.device, 2 * total)) |v| {
            x.vmm = v;
            x.backend = .vmm;
            x.vmm_open = true;
            be = x.vmm.backend();
        } else |e| if (!quiet) log.warn("TF_DSV41_ARENA: no CUDA virtual memory ({t}): one cuMemAlloc a chunk", .{e});
    }
    x.arena = cuda.arena.Arena.init(gpa, be, .{});
    x.arena.install();
    if (rank == 0) log.info("device arena: {s}, {d} KiB granularity, shared chunks of {d} MiB for buffers up to {d} MiB", .{ be.name, be.granularity >> 10, x.arena.o.small_chunk >> 20, x.arena.o.small_max >> 20 });
    return x;
}

test "TF_CUDA_SCHED reader" {
    const t = std.testing;
    const F = struct {
        var v: ?[]const u8 = null;
        fn get(_: []const u8) ?[]const u8 {
            return v;
        }
    };
    F.v = null;
    try t.expectEqual(@as(?cuda.Context.Schedule, null), try schedule(&F.get));
    F.v = "blocking";
    try t.expectEqual(@as(?cuda.Context.Schedule, .blocking), try schedule(&F.get));
    F.v = " yield ";
    try t.expectEqual(@as(?cuda.Context.Schedule, .yield), try schedule(&F.get));
    F.v = "sleep";
    try t.expectError(error.BadKnob, schedule(&F.get));
}

test "TF_DSV41_ARENA reader" {
    const t = std.testing;
    const F = struct {
        var v: ?[]const u8 = null;
        fn get(_: []const u8) ?[]const u8 {
            return v;
        }
    };
    for ([_]?[]const u8{ null, "0", "" }) |v| {
        F.v = v;
        try t.expectEqual(Mode.off, try mode(&F.get));
    }
    F.v = "1";
    try t.expectEqual(Mode.vmm, try mode(&F.get));
    F.v = " vmm ";
    try t.expectEqual(Mode.vmm, try mode(&F.get));
    F.v = "plain";
    try t.expectEqual(Mode.plain, try mode(&F.get));
    F.v = "pack";
    try t.expectEqual(Mode.pack, try mode(&F.get));
    F.v = "yes";
    try t.expectError(error.BadKnob, mode(&F.get));
}
