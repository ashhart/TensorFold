//! TF_DSV41_L2PF: the decode windows' L2 prefetch tables, filled at boot; prefetch launches on a side stream.

const std = @import("std");
const cuda = @import("cuda");
const block = @import("block.zig");
const fwd = @import("forward.zig");
const run = @import("run.zig");
const load = @import("cuda_weights.zig");

const log = std.log.scoped(.dsv41);

pub const Mode = enum { off, bulk };

pub const Settings = struct {
    mode: Mode = .off,
    rows: u32 = 16,
};

pub const Error = error{ BadL2pf, L2pfTable, L2pfSpan };

fn get(name: [:0]const u8) ?[]const u8 {
    const v = std.mem.span(std.c.getenv(name) orelse return null);
    const t = std.mem.trim(u8, v, " \t");
    return if (t.len == 0) null else t;
}

/// The knobs as l2pf.settings reads them; the emitter's fixed values checked against their own knobs.
pub fn settings(o: block.Options) !Settings {
    var s: Settings = .{};
    const raw = get("TF_DSV41_L2PF") orelse return s;
    var low: [16]u8 = undefined;
    if (raw.len > low.len) return error.BadL2pf;
    const m = std.ascii.lowerString(&low, raw);
    if (eq(m, "0") or eq(m, "off") or eq(m, "none") or eq(m, "false")) return s;
    if (!(eq(m, "1") or eq(m, "on") or eq(m, "true") or eq(m, "bulk"))) {
        log.err("TF_DSV41_L2PF={s}: the Zig engine runs the reference's bulk prefetch (0 | 1 | bulk)", .{raw});
        return error.BadL2pf;
    }
    s.mode = .bulk;
    if (get("TF_DSV41_L2PF_ROWS")) |v| s.rows = std.fmt.parseInt(u32, v, 10) catch return error.BadL2pf;
    // the emitter's launches carry these values (block.Options defaults): another value is refused
    const fixed = [_]struct { name: [:0]const u8, want: f64 }{
        .{ .name = "TF_DSV41_L2PF_MB", .want = @as(f64, @floatFromInt(o.l2pf_budget)) / (1 << 20) },
        .{ .name = "TF_DSV41_L2PF_CHUNK_KB", .want = @as(f64, @floatFromInt(o.l2pf_chunk)) / 1024 },
        .{ .name = "TF_DSV41_L2PF_PACE_GBPS", .want = o.l2pf_pace_gbps },
        .{ .name = "TF_DSV41_L2PF_PACE_CTAS", .want = @floatFromInt(o.l2pf_pace_ctas) },
        .{ .name = "TF_DSV41_L2PF_DELAY_US", .want = 0 },
    };
    for (fixed) |k| if (get(k.name)) |v| {
        const x = std.fmt.parseFloat(f64, v) catch return error.BadL2pf;
        if (x != k.want) {
            log.err("{s}={s}: the Zig emitter's L2 prefetch launches use {d} (the reference's)", .{ k.name, v, k.want });
            return error.BadL2pf;
        }
    };
    for ([_]struct { name: [:0]const u8, want: []const u8 }{ .{ .name = "TF_DSV41_L2PF_SITES", .want = "x,o,f" }, .{ .name = "TF_DSV41_L2PF_PACE_SITES", .want = "x,f" }, .{ .name = "TF_DSV41_L2PF_AT", .want = "before" } }) |k| if (get(k.name)) |v| {
        if (!sameSet(v, k.want)) {
            log.err("{s}={s}: the Zig emitter's L2 prefetch sites are {s} (the reference's)", .{ k.name, v, k.want });
            return error.BadL2pf;
        }
    };
    return s;
}

fn eq(a: []const u8, b: []const u8) bool {
    return std.mem.eql(u8, a, b);
}

/// The same comma-separated items, in any order.
fn sameSet(a: []const u8, b: []const u8) bool {
    var n: usize = 0;
    var it = std.mem.tokenizeScalar(u8, a, ',');
    while (it.next()) |x| {
        const t = std.mem.trim(u8, x, " ");
        var found = false;
        var jt = std.mem.tokenizeScalar(u8, b, ',');
        while (jt.next()) |y| found = found or std.ascii.eqlIgnoreCase(t, y);
        if (!found) return false;
        n += 1;
    }
    return n == std.mem.count(u8, b, ",") + 1;
}

/// A span to prefetch: a 16-byte-aligned start and a multiple of 16 bytes.
pub const Span = struct { addr: u64, bytes: u64 };

/// Where a source lives: its first byte and the end of the allocation that holds it (the span may not pass it).
pub const Located = struct { addr: u64, end: u64 };

/// l2pf.take over located sources: each whole, the last one's head, `budget` bytes in all; unresolved sources skipped.
pub fn take(out: *std.ArrayList(Span), a: std.mem.Allocator, srcs: []const block.Emitter.L2Src, at: []const ?Located, budget: u64) !void {
    var left = budget;
    for (srcs, at) |s, loc| {
        if (left < 16) break;
        const l = loc orelse continue;
        const start = std.mem.alignForward(u64, l.addr, 16);
        const end = @min(l.addr + s.bytes, l.end);
        if (end <= start) continue;
        const n = @min(left, end - start) / 16 * 16;
        if (n == 0) continue;
        try out.append(a, .{ .addr = start, .bytes = n });
        left -= n;
    }
}

/// The emitter's own row count for a site (its table's shape): `take` over the declared sizes.
fn rowsOf(srcs: []const block.Emitter.L2Src, budget: u64) usize {
    var left = budget;
    var rows: usize = 0;
    for (srcs) |s| {
        if (left < 16) break;
        const n = @min(left, s.bytes) / 16 * 16;
        if (n == 0) continue;
        rows += 1;
        left -= n;
    }
    return rows;
}

/// The side stream and the filled tables (rank-local; every rank runs the same launches on its own weights).
pub const L2pf = struct {
    side: run.Runner.Side,
    tables: usize,
    bytes: [3]u64,

    pub fn deinit(p: *L2pf) void {
        p.side.stream.deinit();
        p.side.fork.deinit();
        p.side.join.deinit();
    }
};

/// Fills every "s.L<i>.pf.{o,f,x}" table and turns the side stream on; null when off; a bad table leaves it off.
pub fn install(gpa: std.mem.Allocator, f: *fwd.Forward, weights: *const load.Weights) !?*L2pf {
    const s = try settings(f.opts);
    if (s.mode == .off) return null;
    return fill(gpa, f, weights, s) catch |e| switch (e) {
        error.L2pfTable, error.L2pfSpan => {
            log.err("l2pf: OFF ({t}): the decode windows run without the L2 prefetch (TF_DSV41_L2PF=1 asked for it)", .{e});
            return null;
        },
        else => return e,
    };
}

fn fill(gpa: std.mem.Allocator, f: *fwd.Forward, weights: *const load.Weights, s: Settings) !*L2pf {
    const r = f.runner;
    var la = std.heap.ArenaAllocator.init(gpa);
    defer la.deinit();
    const a = la.allocator();
    var e: block.Emitter = .{ .a = a, .cfg = f.cfg, .w = f.widths, .o = f.opts, .n = 1, .start = 0 };
    const p = try gpa.create(L2pf);
    errdefer gpa.destroy(p);
    p.* = .{ .side = .{ .stream = try cuda.Stream.init(r.d, true), .fork = try cuda.Event.init(r.d, false), .join = try cuda.Event.init(r.d, false), .rows = s.rows }, .tables = 0, .bytes = @splat(0) };
    errdefer p.deinit();
    var nb: [64]u8 = undefined;
    for (try f.backbone(a)) |L| for (std.enums.values(block.Emitter.L2Site)) |site| {
        const role = try std.fmt.bufPrint(&nb, "s.L{d}.pf.{t}", .{ L, site });
        const table = r.addressOf(role) orelse continue;
        const srcs = try e.l2pfSources(L, site);
        const at = try a.alloc(?Located, srcs.len);
        for (srcs, at) |x, *y| y.* = try locate(r, weights, x);
        var spans: std.ArrayList(Span) = .empty;
        try take(&spans, a, srcs, at, f.opts.l2pf_budget);
        const want = rowsOf(srcs, f.opts.l2pf_budget);
        if (spans.items.len != want) {
            log.err("l2pf: {s} resolves {d} spans, the emitter's table holds {d}", .{ role, spans.items.len, want });
            return error.L2pfTable;
        }
        const words = try a.alloc(i64, 2 * spans.items.len);
        for (spans.items, 0..) |sp, i| {
            words[2 * i] = @intCast(sp.addr);
            words[2 * i + 1] = @intCast(sp.bytes);
            p.bytes[@backingInt(site)] += sp.bytes;
        }
        try cuda.DeviceBuffer.upload(.{ .d = r.d, .ptr = table, .len = 16 * spans.items.len }, 0, std.mem.sliceAsBytes(words));
        p.tables += 1;
    };
    r.l2pf = &p.side;
    log.info("l2pf: bulk, {d} MiB a site, {d} tables (o {d:.1} / f {d:.1} / x {d:.1} MiB), sites x / f paced at {d} GB/s on {d} CTAs, windows <= {d} rows, a side stream", .{ f.opts.l2pf_budget >> 20, p.tables, mib(p.bytes[0]), mib(p.bytes[1]), mib(p.bytes[2]), f.opts.l2pf_pace_gbps, f.opts.l2pf_pace_ctas, s.rows });
    return p;
}

fn mib(b: u64) f64 {
    return @as(f64, @floatFromInt(b)) / (1 << 20);
}

/// A source's device address and the end of the allocation holding it; null: the rank has no such tensor (left out).
fn locate(r: *const run.Runner, weights: *const load.Weights, x: block.Emitter.L2Src) !?Located {
    switch (x.kind) {
        .weight => {
            const t = weights.get(x.name) orelse return null;
            return .{ .addr = t.ptr, .end = t.ptr + t.len };
        },
        .words => {
            var nb: [160]u8 = undefined;
            const lanes = try std.fmt.bufPrint(&nb, "s.{s}.lanes", .{x.name});
            // in place (TF_DSV41_FORMS_INPLACE) the form lives in its trellis weight's buffer
            if (r.addressOf(lanes)) |p| return .{ .addr = p, .end = ownedEnd(r, p) orelse weightEnd(weights, p) orelse return error.L2pfSpan };
            const tr = try std.fmt.bufPrint(&nb, "{s}.trellis", .{x.name});
            const t = weights.get(tr) orelse return null;
            return .{ .addr = t.ptr, .end = t.ptr + t.len };
        },
        .expert => {
            const tab = r.addressOf(x.name) orelse return null;
            var ptr: [8]u8 = undefined;
            try cuda.DeviceBuffer.download(.{ .d = r.d, .ptr = tab, .len = 8 * (@as(usize, x.index) + 1) }, 8 * @as(usize, x.index), &ptr);
            const addr = std.mem.readInt(u64, &ptr, .little);
            var it = weights.map.valueIterator();
            while (it.next()) |t| if (t.ptr <= addr and addr < t.ptr + t.len) return .{ .addr = addr, .end = t.ptr + t.len };
            if (ownedEnd(r, addr)) |end| return .{ .addr = addr, .end = end };
            log.err("l2pf: {s}[{d}] = 0x{x} is inside no allocation", .{ x.name, x.index, addr });
            return error.L2pfSpan;
        },
    }
}

/// The end of the runner-owned buffer holding `addr`.
fn ownedEnd(r: *const run.Runner, addr: u64) ?u64 {
    for (r.owned.items) |b| if (b.ptr <= addr and addr < b.ptr + b.len) return b.ptr + b.len;
    return null;
}

fn weightEnd(weights: *const load.Weights, addr: u64) ?u64 {
    var it = weights.map.valueIterator();
    while (it.next()) |t| if (t.ptr <= addr and addr < t.ptr + t.len) return t.ptr + t.len;
    return null;
}

test "l2pf.take: l2pf.py's spans (aligned starts, whole sources, the last one's head, missing ones left out)" {
    const a = std.testing.allocator;
    const S = block.Emitter.L2Src;
    const srcs = [_]S{
        .{ .kind = .weight, .name = "a", .bytes = 64 },
        .{ .kind = .weight, .name = "gone", .bytes = 32 },
        .{ .kind = .words, .name = "b", .bytes = 100 },
        .{ .kind = .weight, .name = "c", .bytes = 4096 },
        .{ .kind = .weight, .name = "d", .bytes = 64 },
    };
    const at = [_]?Located{ .{ .addr = 0x1000, .end = 0x1040 }, null, .{ .addr = 0x2008, .end = 0x3000 }, .{ .addr = 0x4000, .end = 0x9000 }, .{ .addr = 0xa000, .end = 0xb000 } };
    var out: std.ArrayList(Span) = .empty;
    defer out.deinit(a);
    try take(&out, a, &srcs, &at, 256);
    // a whole; b from 0x2010 to its end 0x206c (92 bytes -> 80); c's head: 256 - 64 - 80 = 112; budget spent before d
    try std.testing.expectEqual(@as(usize, 3), out.items.len);
    try std.testing.expectEqual(Span{ .addr = 0x1000, .bytes = 64 }, out.items[0]);
    try std.testing.expectEqual(Span{ .addr = 0x2010, .bytes = 80 }, out.items[1]);
    try std.testing.expectEqual(Span{ .addr = 0x4000, .bytes = 112 }, out.items[2]);
    // with every source located and aligned, the spans are the emitter's rows
    const all = [_]?Located{ .{ .addr = 0x1000, .end = 0x1040 }, .{ .addr = 0x1100, .end = 0x1120 }, .{ .addr = 0x2000, .end = 0x3000 }, .{ .addr = 0x4000, .end = 0x9000 }, .{ .addr = 0xa000, .end = 0xb000 } };
    out.clearRetainingCapacity();
    try take(&out, a, &srcs, &all, 1 << 20);
    try std.testing.expectEqual(rowsOf(&srcs, 1 << 20), out.items.len);
    try std.testing.expect(sameSet("f, x", "x,f") and !sameSet("x", "x,f") and sameSet("o,x,f", "x,o,f"));
}
