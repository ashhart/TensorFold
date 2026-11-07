//! What the HIP engine may serve: the CUDA path's memory rules (cuda_memory.zig) and the window and weights it reads.
const std = @import("std");
const builtin = @import("builtin");
const memory = @import("cuda_memory");
const Context = @import("context.zig").Context;
const Allocator = std.mem.Allocator;

/// Windows has no /proc/meminfo and no sysfs VRAM file; the driver's own count is the budget.
const windows = builtin.os.tag == .windows;

pub const MemInfo = memory.MemInfo;
pub const Pool = memory.Pool;
pub const meminfo = memory.meminfo;
pub const reserveBytes = memory.reserveBytes;
pub const limitBytes = memory.limitBytes;
pub const admit = memory.admit;
pub const counts = memory.counts;

/// The model's window (config.json's max_position_embeddings, text_config's first), 0 when it names none.
pub fn modelContext(a: Allocator, io: std.Io, dir: []const u8) i64 {
    const path = std.fs.path.join(a, &.{ dir, "config.json" }) catch return 0;
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(16 << 20)) catch return 0;
    const doc = std.json.parseFromSliceLeaky(std.json.Value, a, bytes, .{}) catch return 0;
    if (doc != .object) return 0;
    const text = if (doc.object.get("text_config")) |t| (if (t == .object) t else doc) else doc;
    const limit = text.object.get("max_position_embeddings") orelse doc.object.get("max_position_embeddings") orelse return 0;
    return if (limit == .integer and limit.integer > 0) limit.integer else 0;
}

/// Prompt plus reply tokens a request may use: --context (0: the model's window), else the family default within it.
pub fn contextWindow(requested: ?i64, native: i64, default: i64) error{ Negative, NoNative, PastNative }!i64 {
    const r = requested orelse return if (native > 0) @min(default, native) else default;
    if (r < 0) return error.Negative;
    if (r == 0) return if (native > 0) native else error.NoNative;
    if (native > 0 and r > native) return error.PastNative;
    return r;
}

/// The bytes of the checkpoint's safetensors files: what its weights need on the device, near enough to refuse early.
pub fn weightBytes(io: std.Io, dir: []const u8) u64 {
    var d = std.Io.Dir.cwd().openDir(io, dir, .{ .iterate = true }) catch return 0;
    defer d.close(io);
    var total: u64 = 0;
    var it = d.iterate();
    while (it.next(io) catch null) |e| {
        if (!std.mem.endsWith(u8, e.name, ".safetensors")) continue;
        const st = d.statFile(io, e.name, .{}) catch continue;
        total += st.size;
    }
    return total;
}

/// What a plan may take on `ctx`'s GPU: free (the host's when integrated without a carve-out), reserve, cap.
pub fn budget(gpa: Allocator, io: std.Io, ctx: Context) !Pool {
    const integrated = (try ctx.attribute(.integrated)) != 0;
    const card = try ctx.memInfo();
    const own: MemInfo = .{ .total = card.total, .available = card.free };
    const env: Env = .{
        .reserve = getenv("TENSORFOLD_MEMORY_RESERVE_GIB"),
        .limit = getenv("TENSORFOLD_HIP_MEMORY_LIMIT_GB"),
    };
    if (windows) {
        // WDDM reports the whole budget through the driver: no host text, no dedicated-VRAM file.
        const p = try plan(false, own, null, null, env);
        announce(p);
        return p.pool;
    }
    const vram = if (integrated) vramBytes(gpa, io, ctx) else null;
    const text = if (sourceOf(integrated, own, vram) == .host) procText(gpa, io, "/proc/meminfo") else null;
    defer if (text) |t| gpa.free(t);
    const p = try plan(integrated, own, text, vram, env);
    announce(p);
    return p.pool;
}

/// Where a budget's count comes from: a discrete card, an integrated GPU's firmware carve-out, or the host's pages.
pub const Source = enum { card, carve_out, host };

/// The reserve and cap variables' text, unset as null.
pub const Env = struct { reserve: ?[]const u8 = null, limit: ?[]const u8 = null };

pub const Plan = struct { pool: Pool, source: Source };

pub const PlanError = error{ HostMemoryUnavailable, BadReserve, BadLimit };

/// User-facing malformed budget text, kept separate from the pure admission error for host tests.
pub fn badSetting(a: std.mem.Allocator, err: PlanError, env: Env) ![]const u8 {
    const name: []const u8 = switch (err) {
        error.BadReserve => "TENSORFOLD_MEMORY_RESERVE_GIB",
        error.BadLimit => "TENSORFOLD_HIP_MEMORY_LIMIT_GB",
        else => unreachable,
    };
    const raw = if (err == error.BadReserve) env.reserve else env.limit;
    const forms: []const u8 = if (err == error.BadReserve)
        "a number of GiB from 0.5 (2 on shared memory) to the memory's size"
    else
        "a positive number of GiB whose byte count fits in a 64-bit size";
    return std.fmt.allocPrint(a, "{s}=\"{f}\": expected {s}", .{ name, std.zig.fmtString(raw orelse ""), forms });
}

test "HIP environment: reserve and limit refusals name the actual text and accepted form" {
    const a = std.testing.allocator;
    const card: MemInfo = .{ .total = 16 << 30, .available = 15 << 30 };
    try std.testing.expectError(error.BadReserve, plan(false, card, null, null, .{ .reserve = "bad-reserve" }));
    try std.testing.expectError(error.BadLimit, plan(false, card, null, null, .{ .limit = "bad-limit" }));
    for ([_]PlanError{ error.BadReserve, error.BadLimit }) |err| {
        const text = try badSetting(a, err, .{ .reserve = "bad-reserve", .limit = "bad-limit" });
        defer a.free(text);
        try std.testing.expect(std.mem.indexOf(u8, text, if (err == error.BadReserve) "TENSORFOLD_MEMORY_RESERVE_GIB=\"bad-reserve\"" else "TENSORFOLD_HIP_MEMORY_LIMIT_GB=\"bad-limit\"") != null);
        try std.testing.expect(std.mem.indexOf(u8, text, "expected") != null);
        try std.testing.expect(std.mem.indexOf(u8, text, "GiB") != null);
    }
}

/// The pool from the card's count, /proc/meminfo's text and the device's dedicated VRAM (sysfs), without I/O.
pub fn plan(integrated: bool, card: MemInfo, text: ?[]const u8, vram: ?u64, env: Env) PlanError!Plan {
    const source = sourceOf(integrated, card, vram);
    const c = if (source == .host) counts(true, card, text) catch return error.HostMemoryUnavailable else card;
    const shared = source == .host;
    const reserve = reserveBytes(env.reserve, c.total, shared) catch return error.BadReserve;
    const limit = limitBytes(env.limit) catch return error.BadLimit;
    const pool: Pool = .{
        .free = c.available,
        .total = c.total,
        .reserve = reserve,
        .limit = limit,
        .unified = shared,
    };
    return .{ .pool = pool, .source = source };
}

/// An integrated GPU counts its own memory only when the card's total is the firmware's dedicated VRAM, within a 32nd.
pub fn sourceOf(integrated: bool, card: MemInfo, vram: ?u64) Source {
    if (!integrated) return .card;
    const v = vram orelse return .host;
    return if (v > 0 and card.total <= v and card.total >= v - v / 32) .carve_out else .host;
}

/// The device's dedicated VRAM from sysfs (mem_info_vram_total under its PCI address), null when unreadable.
fn vramBytes(gpa: Allocator, io: std.Io, ctx: Context) ?u64 {
    const domain = ctx.attribute(.pci_domain_id) catch return null;
    const bus = ctx.attribute(.pci_bus_id) catch return null;
    const dev = ctx.attribute(.pci_device_id) catch return null;
    var buf: [96]u8 = undefined;
    const path = vramPath(&buf, domain, bus, dev) catch return null;
    const text = procText(gpa, io, path) orelse return null;
    defer gpa.free(text);
    return std.fmt.parseInt(u64, std.mem.trim(u8, text, " \n"), 10) catch null;
}

/// sysfs's dedicated-VRAM file for the GPU at PCI domain:bus:device, function 0.
pub fn vramPath(buf: []u8, domain: c_int, bus: c_int, dev: c_int) ![]const u8 {
    if (domain < 0 or bus < 0 or dev < 0) return error.Invalid;
    const fmt = "/sys/bus/pci/devices/{x:0>4}:{x:0>2}:{x:0>2}.0/mem_info_vram_total";
    const at = .{ @as(u32, @intCast(domain)), @as(u32, @intCast(bus)), @as(u32, @intCast(dev)) };
    return std.fmt.bufPrint(buf, fmt, at);
}

var announced = std.atomic.Value(bool).init(false);

/// Logs once per process which count the budget uses, so a GTT-backed or carve-out pool is visible at start-up.
fn announce(p: Plan) void {
    if (announced.swap(true, .monotonic)) return;
    const g = @as(f64, 1 << 30);
    const what = switch (p.source) {
        .card => "the card's memory",
        .carve_out => "the card's memory (firmware VRAM carve-out)",
        .host => "the host's /proc/meminfo (integrated, shared memory)",
    };
    const f = @as(f64, @floatFromInt(p.pool.free)) / g;
    const t = @as(f64, @floatFromInt(p.pool.total)) / g;
    const r = @as(f64, @floatFromInt(p.pool.reserve)) / g;
    std.log.info("HIP memory budget from {s}: {d:.1} of {d:.1} GiB free, {d:.1} GiB reserve", .{ what, f, t, r });
}

fn getenv(name: [:0]const u8) ?[]const u8 {
    return std.mem.span(std.c.getenv(name) orelse return null);
}

/// A /proc file's text, streamed: procfs reports size 0, and a positional read (readFileAlloc) stops there.
pub fn procText(a: Allocator, io: std.Io, path: []const u8) ?[]u8 {
    var file = std.Io.Dir.cwd().openFile(io, path, .{}) catch return null;
    defer file.close(io);
    var buf: [4096]u8 = undefined;
    var r = file.readerStreaming(io, &buf);
    return r.interface.allocRemaining(a, .limited(1 << 20)) catch null;
}

test "context windows: the family default inside the model's, 0 for the model's, never past it" {
    try std.testing.expectEqual(@as(i64, 16384), try contextWindow(null, 262144, 16384));
    try std.testing.expectEqual(@as(i64, 4096), try contextWindow(null, 4096, 16384));
    try std.testing.expectEqual(@as(i64, 262144), try contextWindow(0, 262144, 16384));
    try std.testing.expectEqual(@as(i64, 65536), try contextWindow(65536, 262144, 16384));
    try std.testing.expectError(error.PastNative, contextWindow(300000, 262144, 16384));
    try std.testing.expectError(error.Negative, contextWindow(-5, 262144, 16384));
    try std.testing.expectError(error.NoNative, contextWindow(0, 0, 16384));
}

test "a budget counts an integrated GPU's own memory only when it fits the firmware's dedicated VRAM" {
    const g: u64 = 1 << 30;
    const pages = "MemTotal: 32505856 kB\nMemAvailable: 26214400 kB\n"; // 31 GiB, 25 GiB available
    const carve = try plan(true, .{ .total = 96 * g, .available = 95 * g }, pages, 96 * g, .{});
    try std.testing.expectEqual(Source.carve_out, carve.source);
    try std.testing.expectEqual(@as(u64, 95 * g), carve.pool.free);
    try std.testing.expectEqual(@as(u64, 3 * g), carve.pool.reserve); // a discrete card's share, not unified memory's
    try std.testing.expect(!carve.pool.unified);
    const big = "MemTotal: 127926272 kB\nMemAvailable: 115343360 kB\n"; // 122 GiB, 110 GiB available
    // a GTT pool past MemTotal beside a small carve-out
    const gtt = try plan(true, .{ .total = 128 * g, .available = 127 * g }, big, g / 2, .{});
    try std.testing.expectEqual(Source.host, gtt.source);
    try std.testing.expectEqual(@as(u64, 110 * g), gtt.pool.free);
    try std.testing.expect(gtt.pool.unified);
    const mid = "MemTotal: 98566144 kB\nMemAvailable: 94371840 kB\n"; // 94 GiB, 90 GiB available
    const medium = try plan(true, .{ .total = 32 * g, .available = 31 * g }, mid, 32 * g, .{}); // a 32 GiB carve-out
    try std.testing.expectEqual(Source.carve_out, medium.source);
    try std.testing.expectEqual(@as(u64, 31 * g), medium.pool.free);
    const unread = try plan(true, .{ .total = 96 * g, .available = 95 * g }, pages, null, .{});
    try std.testing.expectEqual(Source.host, unread.source); // no sysfs reading: the host's pages
    const discrete = try plan(false, .{ .total = 48 * g, .available = 47 * g }, null, null, .{ .reserve = "3" });
    try std.testing.expectEqual(Source.card, discrete.source);
    try std.testing.expectEqual(@as(u64, 3 * g), discrete.pool.reserve);
    const blind = plan(true, .{ .total = 8 * g, .available = 8 * g }, null, null, .{});
    try std.testing.expectError(error.HostMemoryUnavailable, blind);
}

test "sysfs's dedicated-VRAM path names the GPU's PCI address" {
    var buf: [96]u8 = undefined;
    const want = "/sys/bus/pci/devices/0000:c3:00.0/mem_info_vram_total";
    try std.testing.expectEqualStrings(want, try vramPath(&buf, 0, 0xc3, 0));
    try std.testing.expectError(error.Invalid, vramPath(&buf, 0, -1, 0));
}

test "a carve-out needs the card's total to be the dedicated VRAM: not past it, not a smaller GTT pool beside it" {
    const g: u64 = 1 << 30;
    const T = struct {
        fn of(total: u64, vram: u64) Source {
            return sourceOf(true, .{ .total = total, .available = 0 }, vram);
        }
    };
    try std.testing.expectEqual(Source.carve_out, T.of(96 * g, 96 * g));
    try std.testing.expectEqual(Source.carve_out, T.of(95 * g, 96 * g));
    try std.testing.expectEqual(Source.host, T.of(96 * g + 1, 96 * g));
    try std.testing.expectEqual(Source.host, T.of(15 * g, 96 * g)); // a GTT-sized pool beside a large carve-out
    try std.testing.expectEqual(Source.host, T.of(60 * g, g / 2)); // a GTT pool beside a small carve-out
    try std.testing.expectEqual(Source.host, T.of(0, 0));
    try std.testing.expectEqual(Source.card, sourceOf(false, .{ .total = 48 * g, .available = 0 }, 96 * g));
    const one: MemInfo = .{ .total = 96 * g, .available = 95 * g };
    const blind = try plan(true, one, null, 96 * g, .{}); // a carve-out needs no /proc/meminfo
    try std.testing.expectEqual(@as(u64, 95 * g), blind.pool.free);
    const half = try plan(true, one, null, 96 * g, .{ .reserve = "0.5" });
    try std.testing.expectEqual(@as(u64, g / 2), half.pool.reserve); // a carve-out takes the discrete floor
    const big = "MemTotal: 127926272 kB\nMemAvailable: 115343360 kB\n";
    const shared = plan(true, .{ .total = 60 * g, .available = 59 * g }, big, g / 2, .{ .reserve = "1" });
    try std.testing.expectError(error.BadReserve, shared); // shared memory keeps its floor of 2
}
