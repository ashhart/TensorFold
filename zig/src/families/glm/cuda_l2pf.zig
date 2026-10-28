//! TF_GLM_L2PF: a decode window's next weights read into L2 on a side stream (DeepSeek V4.1's l2pf and l2pace kernels).

const std = @import("std");
const cuda = @import("cuda");
const weights = @import("cuda_weights.zig");
const projection = @import("cuda_proj.zig");

/// TF_GLM_L2PF and its knobs at DeepSeek V4.1's defaults (TF_DSV41_L2PF's), the knobs read only when it is 1.
pub const Options = struct {
    on: bool = false,
    mb: u32 = 12, // TF_GLM_L2PF_MB: a site's budget in MiB
    ctas: u32 = 2, // TF_GLM_L2PF_CTAS: the CTAs a site's prefetch spreads over
    pace_gbps: u32 = 150, // TF_GLM_L2PF_PACE_GBPS: the sites beside a gather read at this rate (0: unpaced)

    pub fn read(get: *const fn ([:0]const u8) ?[]const u8, problem: *[]const u8) !Options {
        var o: Options = .{};
        const v = get("TF_GLM_L2PF") orelse return o;
        if (std.mem.eql(u8, v, "0")) return o;
        if (!std.mem.eql(u8, v, "1")) {
            problem.* = "TF_GLM_L2PF: 0 or 1";
            return error.BadSetting;
        }
        o.on = true;
        const knobs = .{
            .{ "TF_GLM_L2PF_MB", "mb", 1, "TF_GLM_L2PF_MB: a site's budget in MiB, 1 or more" },
            .{ "TF_GLM_L2PF_CTAS", "ctas", 1, "TF_GLM_L2PF_CTAS: 1 or more" },
            .{ "TF_GLM_L2PF_PACE_GBPS", "pace_gbps", 0, "TF_GLM_L2PF_PACE_GBPS: GB/s, or 0 for unpaced" },
        };
        inline for (knobs) |k| if (get(k[0])) |x| {
            const n = std.fmt.parseInt(u32, x, 10) catch null;
            if (n == null or n.? < k[2]) {
                problem.* = k[3];
                return error.BadSetting;
            }
            @field(o, k[1]) = n.?;
        };
        return o;
    }

    /// What the ranks agree on: zeros when off, so knobs nobody reads never differ.
    pub fn words(o: Options) [4]u32 {
        return if (o.on) .{ 1, o.mb, o.ctas, o.pace_gbps } else @splat(0);
    }
};

/// A layer's sites: n before gather_b (the next mixer's input), f before gather_a (the ffn's head), o (KDA's chain).
pub const Site = enum(u2) { n, f, o };

/// One site's rows of the table and their bytes.
pub const Span = struct { at: u32 = 0, rows: u32 = 0, bytes: u64 = 0 };

const max_sources = 8;

const Sources = struct {
    ptrs: [max_sources]u64 = undefined,
    len: usize = 0,

    fn add(s: *Sources, p: u64) void {
        if (p != 0) {
            s.ptrs[s.len] = p;
            s.len += 1;
        }
    }

    fn lane(s: *Sources, q: cuda.qmmf.Weight) void {
        s.add(q.scales);
        s.add(q.codes);
    }

    fn proj(s: *Sources, w: projection.Proj) void {
        switch (w) {
            .bf16 => |p| s.add(p),
            .fp8g, .mxfp8 => |q| s.lane(q),
        }
    }
};

/// The buffers site `site` of layer `l` reads first, in its matmuls' order: a lane weight's scales, then its codes.
fn sources(l: weights.Layer, site: Site) Sources {
    var s: Sources = .{};
    switch (site) {
        .n => switch (l.mixer) {
            .kda => |k| switch (k.proj) {
                .bf16 => |p| s.add(p),
                .split => |sp| {
                    s.lane(sp.qkv);
                    s.add(sp.rest);
                },
            },
            .mla => |a| {
                s.proj(a.proj);
                s.proj(a.q_b);
            },
        },
        .f => switch (l.ffn) {
            .moe => |m| s.add(m.router),
            .dense => |m| switch (m.w) {
                .nvfp4 => |x| {
                    s.lane(x.gate);
                    s.lane(x.up);
                },
                .bf16 => |x| s.add(x.gu),
                .mxfp8 => |x| s.lane(x.gu),
            },
        },
        .o => switch (l.mixer) {
            .kda => |k| s.proj(k.o),
            .mla => {},
        },
    }
    return s;
}

/// l2pf.take: the bytes of each of `sizes` in order within `budget`, 16-byte multiples, the last one's head, 0 past it.
pub fn take(sizes: []const u64, budget: u64, out: []u64) void {
    var left = budget;
    for (sizes, out) |s, *n| {
        n.* = if (left < 16) 0 else @min(left, s) / 16 * 16;
        left -= n.*;
    }
}

/// Every site's rows of (address, bytes) from `bufs`, the weights' allocations; false when a source starts none.
pub fn table(gpa: std.mem.Allocator, bufs: []const cuda.DeviceBuffer, layers: []const weights.Layer, budget: u64, rows: *std.ArrayList([2]i64), spans: []Span) !bool {
    std.debug.assert(spans.len == layers.len * 3);
    for (layers, 0..) |_, j| for ([_]Site{ .n, .f, .o }) |site| {
        const sp = &spans[j * 3 + @backingInt(site)];
        sp.* = .{ .at = @intCast(rows.items.len) };
        if (site == .n and j + 1 == layers.len) continue; // the last layer's next is the head
        const src = sources(layers[if (site == .n) j + 1 else j], site);
        var sizes: [max_sources]u64 = undefined;
        for (src.ptrs[0..src.len], sizes[0..src.len]) |p, *n| n.* = for (bufs) |b| {
            if (b.ptr == p) break b.len;
        } else {
            std.log.warn("TF_GLM_L2PF: layer {d}'s site {t} reads 0x{x}, which starts no weight buffer; prefetch off", .{ j, site, p });
            return false;
        };
        var got: [max_sources]u64 = undefined;
        take(sizes[0..src.len], budget, got[0..src.len]);
        for (src.ptrs[0..src.len], got[0..src.len]) |p, b| if (b > 0) {
            try rows.append(gpa, .{ @intCast(p), @intCast(b) });
            sp.rows += 1;
            sp.bytes += b;
        };
    };
    return true;
}

/// DeepSeek V4.1's launch: 32 KiB pieces, segments over 128 threads a CTA, paced as one piece per `chunk / rate` ns.
const chunk: u32 = 32 << 10;

/// The prefetch's own stream, events, kernels and table; the main stream forks at each site and joins once a window.
pub const L2pf = struct {
    gpa: std.mem.Allocator,
    o: Options,
    side: cuda.Stream,
    fork_ev: cuda.Event,
    join_ev: cuda.Event,
    mods: [2]cuda.Module,
    segments: cuda.Function,
    paced: cuda.Function,
    rows: cuda.DeviceBuffer,
    spans: []Span,

    /// Built after the weights load and tile (tiling moves packed linears); null when a source is not a buffer.
    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, w: *const weights.Weights, o: Options) !?L2pf {
        const spans = try gpa.alloc(Span, w.layers.len * 3);
        errdefer gpa.free(spans);
        var host: std.ArrayList([2]i64) = .empty;
        defer host.deinit(gpa);
        if (!try table(gpa, w.buffers.items, w.layers, @as(u64, o.mb) << 20, &host, spans)) {
            gpa.free(spans);
            return null;
        }
        if (host.items.len == 0) return error.NoPrefetchRows;
        var total: u64 = 0;
        for (spans) |sp| total += sp.bytes;
        std.log.info("TF_GLM_L2PF: {d} rows, {d} MiB a decode window, {d} CTAs a site, the gathers' sites at {d} GB/s (0: unpaced)", .{ host.items.len, total >> 20, o.ctas, o.pace_gbps });
        var rows = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(host.items));
        errdefer rows.free();
        var mods: [2]cuda.Module = .{ try cuda.Module.load(d, cuda.kernels.l2pf), undefined };
        errdefer mods[0].unload();
        mods[1] = try cuda.Module.load(d, cuda.kernels.l2pace);
        errdefer mods[1].unload();
        var side = try cuda.Stream.init(d, true);
        errdefer side.deinit();
        var fork_ev = try cuda.Event.init(d, false);
        errdefer fork_ev.deinit();
        var join_ev = try cuda.Event.init(d, false);
        errdefer join_ev.deinit();
        const segments = try mods[0].function("_ZN6tfl2pf15segments_kernelILb1EEEvPKlij");
        const paced = try mods[1].function("_ZN12dsv41_l2pace12paced_kernelEPKlijmm");
        return .{ .gpa = gpa, .o = o, .side = side, .fork_ev = fork_ev, .join_ev = join_ev, .mods = mods, .segments = segments, .paced = paced, .rows = rows, .spans = spans };
    }

    /// Before the weights free: the side stream drains first (Stream.deinit synchronizes).
    pub fn deinit(p: *L2pf) void {
        p.side.deinit();
        p.fork_ev.deinit();
        p.join_ev.deinit();
        p.rows.free();
        for (&p.mods) |*m| m.unload();
        p.gpa.free(p.spans);
        p.* = undefined;
    }

    /// Site `site` of layer `j` read into L2 on the side stream from this point of `main` on.
    pub fn fork(p: *const L2pf, main: cuda.Stream, j: usize, site: Site) !void {
        if (j * 3 >= p.spans.len) return;
        const sp = p.spans[j * 3 + @backingInt(site)];
        if (sp.rows == 0) return;
        try p.fork_ev.record(main);
        try p.side.wait(p.fork_ev);
        var a: cuda.Args = .{};
        a.add(p.rows.ptr + @as(u64, sp.at) * 16);
        a.add(@as(c_int, @intCast(sp.rows)));
        a.add(chunk);
        if (site != .o and p.o.pace_gbps > 0) {
            a.add(@as(u64, @max(1, (chunk + p.o.pace_gbps / 2) / p.o.pace_gbps))); // GB/s is bytes a ns
            a.add(@as(u64, 0));
            return cuda.launch.launch(p.paced, .{ .grid = .{ .x = p.o.ctas }, .block = .{ .x = 32 } }, p.side, &a);
        }
        const pieces = std.math.divCeil(u64, sp.bytes, chunk) catch unreachable;
        const grid: u32 = @intCast(std.math.clamp(std.math.divCeil(u64, pieces, 128) catch unreachable, 1, p.o.ctas));
        try cuda.launch.launch(p.segments, .{ .grid = .{ .x = grid }, .block = .{ .x = 128 } }, p.side, &a);
    }

    /// A window's end: `main` waits for every site forked on it, so nothing is in flight past the window.
    pub fn join(p: *const L2pf, main: cuda.Stream) !void {
        try p.join_ev.record(p.side);
        try main.wait(p.join_ev);
    }
};

/// Site o's place in a KDA block: between its projections and its chain.
pub const At = struct {
    l2: *const L2pf,
    layer: usize,

    pub fn fork(at: At, main: cuda.Stream) !void {
        return at.l2.fork(main, at.layer, .o);
    }
};

fn Env(comptime pairs: anytype) type {
    return struct {
        fn get(key: [:0]const u8) ?[]const u8 {
            inline for (pairs) |p| if (std.mem.eql(u8, key, p[0])) return p[1];
            return null;
        }
    };
}

test "TF_GLM_L2PF: off unless 1, DeepSeek's knobs by default, refusals naming a bad value, and unread knobs never differ" {
    const t = std.testing;
    var problem: []const u8 = "";
    try t.expectEqual([4]u32{ 0, 0, 0, 0 }, (try Options.read(Env(.{}).get, &problem)).words());
    try t.expectEqual([4]u32{ 0, 0, 0, 0 }, (try Options.read(Env(.{ .{ "TF_GLM_L2PF", "0" }, .{ "TF_GLM_L2PF_MB", "0" } }).get, &problem)).words());
    try t.expectEqual([4]u32{ 1, 12, 2, 150 }, (try Options.read(Env(.{.{ "TF_GLM_L2PF", "1" }}).get, &problem)).words());
    const set = Env(.{ .{ "TF_GLM_L2PF", "1" }, .{ "TF_GLM_L2PF_MB", "4" }, .{ "TF_GLM_L2PF_CTAS", "4" }, .{ "TF_GLM_L2PF_PACE_GBPS", "0" } });
    try t.expectEqual([4]u32{ 1, 4, 4, 0 }, (try Options.read(set.get, &problem)).words());
    const bad = .{
        .{ Env(.{.{ "TF_GLM_L2PF", "yes" }}), "TF_GLM_L2PF: 0 or 1" },
        .{ Env(.{ .{ "TF_GLM_L2PF", "1" }, .{ "TF_GLM_L2PF_MB", "0" } }), "TF_GLM_L2PF_MB: a site's budget in MiB, 1 or more" },
        .{ Env(.{ .{ "TF_GLM_L2PF", "1" }, .{ "TF_GLM_L2PF_CTAS", "x" } }), "TF_GLM_L2PF_CTAS: 1 or more" },
        .{ Env(.{ .{ "TF_GLM_L2PF", "1" }, .{ "TF_GLM_L2PF_PACE_GBPS", "-1" } }), "TF_GLM_L2PF_PACE_GBPS: GB/s, or 0 for unpaced" },
    };
    inline for (bad) |b| {
        try t.expectError(error.BadSetting, Options.read(b[0].get, &problem));
        try t.expectEqualStrings(b[1], problem);
    }
}

test "take: each size in order within the budget, in 16-byte multiples, the last one's head and nothing past it" {
    const t = std.testing;
    var out: [3]u64 = undefined;
    take(&.{ 1 << 20, 34 << 20, 2 << 20 }, 12 << 20, &out);
    try t.expectEqual([3]u64{ 1 << 20, 11 << 20, 0 }, out);
    take(&.{ 1000, 50, 7 }, 4096, &out);
    try t.expectEqual([3]u64{ 992, 48, 0 }, out);
    take(&.{ 5 << 20, 1 << 20, 1 << 20 }, (4 << 20) + 8, &out);
    try t.expectEqual([3]u64{ 4 << 20, 0, 0 }, out);
}

test "the table: each site's sources in its matmuls' order, site n from the next layer, none past the last; an unknown buffer turns it off" {
    const t = std.testing;
    const kda_block = @import("cuda_kda_block.zig");
    const lane: cuda.qmmf.Weight = .{ .mode = .mxfp8, .codes = 0x2000_0000, .scales = 0x1000_0000, .n = 64, .k = 64, .npad = 64 };
    var k: kda_block.Layer = undefined;
    k.proj = .{ .split = .{ .qkv = lane, .rest = 0x3000_0000 } };
    k.o = .{ .bf16 = 0x4000_0000 };
    var d: weights.Layer = undefined; // layer 0: DSA and a dense bf16 MLP
    d.mixer = .{ .mla = undefined };
    d.mixer.mla.proj = .{ .bf16 = 0x6000_0000 };
    d.mixer.mla.q_b = .{ .bf16 = 0x7000_0000 };
    d.ffn = .{ .dense = undefined };
    d.ffn.dense.w = .{ .bf16 = .{ .gu = 0x9000_0000, .down = 0xa000_0000 } };
    var e: weights.Layer = undefined; // layer 1: KDA on the lane and MoE
    e.mixer = .{ .kda = k };
    e.ffn = .{ .moe = undefined };
    e.ffn.moe.router = 0x5000_0000;
    const MiB = 1 << 20;
    var bufs = [_]cuda.DeviceBuffer{
        .{ .d = undefined, .ptr = 0x1000_0000, .len = 1 * MiB },
        .{ .d = undefined, .ptr = 0x2000_0000, .len = 34 * MiB },
        .{ .d = undefined, .ptr = 0x3000_0000, .len = 2 * MiB },
        .{ .d = undefined, .ptr = 0x4000_0000, .len = 23 * MiB },
        .{ .d = undefined, .ptr = 0x5000_0000, .len = 2 * MiB },
        .{ .d = undefined, .ptr = 0x6000_0000, .len = 16 * MiB },
        .{ .d = undefined, .ptr = 0x7000_0000, .len = 17 * MiB },
        .{ .d = undefined, .ptr = 0x9000_0000, .len = 34 * MiB },
    };
    var rows: std.ArrayList([2]i64) = .empty;
    defer rows.deinit(t.allocator);
    var spans: [6]Span = undefined;
    try t.expect(try table(t.allocator, &bufs, &.{ d, e }, 12 * MiB, &rows, &spans));
    const want = [_][2]i64{
        .{ 0x1000_0000, 1 * MiB }, // layer 0's n: layer 1's qkv scales whole, then its codes' head
        .{ 0x2000_0000, 11 * MiB },
        .{ 0x9000_0000, 12 * MiB }, // layer 0's f: its gate and up's head
        .{ 0x5000_0000, 2 * MiB }, // layer 1's f: its router; its n would be the head's, its o is KDA's o
        .{ 0x4000_0000, 12 * MiB },
    };
    try t.expectEqualSlices([2]i64, &want, rows.items);
    try t.expectEqualSlices(Span, &.{
        .{ .at = 0, .rows = 2, .bytes = 12 * MiB },
        .{ .at = 2, .rows = 1, .bytes = 12 * MiB },
        .{ .at = 3 }, // DSA: no site o
        .{ .at = 3 },
        .{ .at = 3, .rows = 1, .bytes = 2 * MiB },
        .{ .at = 4, .rows = 1, .bytes = 12 * MiB },
    }, &spans);
    rows.clearRetainingCapacity();
    try t.expect(!try table(t.allocator, bufs[1..], &.{ d, e }, 12 * MiB, &rows, &spans));
}
