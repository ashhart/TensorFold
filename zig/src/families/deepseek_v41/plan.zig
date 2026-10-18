//! A rank's weights planned from the pack's headers alone: group TP parts, routed layouts, byte totals, coverage.

const std = @import("std");
const quiet = @import("builtin").is_test; // a test's stderr is the build runner's channel
const log = std.log.scoped(.dsv41);
const Config = @import("config.zig").Config;
const exl3 = @import("core").exl3_layout;
const names = @import("names.zig");
const Pack = @import("pack.zig").Pack;
const Info = @import("pack.zig").Info;

/// One EXL3 group of the rank: its stored name and the rank's part.
pub const GroupPlan = struct {
    proj: names.Proj,
    prefix: []const u8,
    split: exl3.Split,
    full: exl3.Group,
    part: exl3.Part,
    /// marker tensor (mul1 / mcg) whose value picks the codebook at load; null: the 3-instruction default
    marker: ?[]const u8,
    /// wo_a: the global group index this slice is
    slice: u32 = 0,
};

pub const NativePlan = struct {
    name: []const u8,
    info: Info,
    slice: names.Slice,
    /// bytes of the rank's slice as stored
    bytes: u64,
};

/// One routed projection: each expert's group part, their layout on the device, and the stored prefixes.
pub const ExpertsPlan = struct {
    proj: names.ExpertProj,
    parts: []exl3.Part,
    prefixes: [][]const u8,
    layout: exl3.Layout,
};

pub const LayerPlan = struct {
    index: u32,
    prefix: []const u8,
    mode: @import("config.zig").Mode,
    groups: []GroupPlan,
    natives: []NativePlan,
    experts: [3]ExpertsPlan,
    /// experts whose gate (w1) and up (w3) widths differ: x3gm cannot fuse them, the block goes to x3pf / grouped
    gate_up_mismatch: u32,

    /// Routed trellis bytes plus the shared expert's: the quant repo's `rank_bytes`.
    pub fn expertBytes(l: *const LayerPlan) u64 {
        var n: u64 = 0;
        for (l.experts) |e| n += e.layout.trellisBytes();
        for (l.groups) |g| switch (g.proj) {
            .shared_w1, .shared_w2, .shared_w3 => n += g.part.trellis.bytes(),
            else => {},
        };
        return n;
    }

    /// Everything the rank holds for the block as stored: trellises, suh/svh, natives.
    pub fn bytes(l: *const LayerPlan) u64 {
        var n: u64 = 0;
        for (l.groups) |g| n += g.part.bytes();
        for (l.natives) |v| n += v.bytes;
        for (l.experts) |e| {
            n += e.layout.trellisBytes();
            for (e.parts) |p| n += p.suh.bytes() + p.svh.bytes();
        }
        return n;
    }
};

pub const Dspark = struct {
    main_proj: GroupPlan,
    natives: []NativePlan,
    confidence: ?NativePlan,
};

pub const Plan = struct {
    arena: std.heap.ArenaAllocator,
    rank: u32,
    world: u32,
    layers: []LayerPlan,
    top: []NativePlan,
    /// null when only some blocks were planned
    head: ?GroupPlan,
    dspark: ?Dspark,
    /// stored names neither planned nor on the ignore list (the loader refuses a pack with any)
    leftovers: [][]const u8,

    pub fn deinit(p: *Plan) void {
        p.arena.deinit();
        p.* = undefined;
    }

    pub fn bytes(p: *const Plan) u64 {
        var n: u64 = if (p.head) |h| h.part.bytes() else 0;
        for (p.top) |v| n += v.bytes;
        for (p.layers) |*l| n += l.bytes();
        if (p.dspark) |d| {
            n += d.main_proj.part.bytes();
            for (d.natives) |v| n += v.bytes;
            if (d.confidence) |c| n += c.bytes;
        }
        return n;
    }
};

pub const Options = struct {
    rank: u32,
    world: u32,
    /// plan only these blocks (single-layer runs, header dumps): no top tensors, head, DSpark or coverage; null: all
    blocks: ?[]const u32 = null,
    /// with `blocks`: also the vocabulary rows, the final norm and the head (a window's finish)
    top: bool = false,
    /// with `blocks` and `top`: also DSpark's heads (main_proj, its norms, the Markov and confidence heads)
    dspark: bool = false,
};

const Builder = struct {
    a: std.mem.Allocator,
    c: *const Config,
    pack: *const Pack,
    o: Options,
    used: std.StringHashMapUnmanaged(void) = .empty,
    buf: [256]u8 = undefined,

    fn mark(b: *Builder, name: []const u8) !void {
        try b.used.put(b.a, b.pack.names.getKey(name) orelse return error.MissingTensor, {});
    }

    fn dupe(b: *Builder, s: []const u8) ![]const u8 {
        return b.a.dupe(u8, s);
    }

    fn group(b: *Builder, proj: names.Proj, prefix: []const u8, split: exl3.Split, k: u32, n: u32) !GroupPlan {
        var shared = try CheckpointPlan.init(b.a, b.pack, &.{prefix});
        defer shared.deinit();
        const storage = shared.projections[0].storage;
        const g = try storage.cudaGroup();
        if ((k != 0 and g.k() != k) or (n != 0 and g.n() != n)) {
            if (!quiet) log.warn("{s}: K={d} N={d}, expected K={d} N={d}", .{ prefix, g.k(), g.n(), k, n });
            return error.UnexpectedShape;
        }
        inline for (.{ "trellis", "suh", "svh" }) |part| try b.mark(try std.fmt.bufPrint(&b.buf, "{s}." ++ part, .{prefix}));
        const marker = try b.pack.marker(prefix, &b.buf);
        if (marker) |m| try b.mark(m);
        return .{ .proj = proj, .prefix = try b.dupe(prefix), .split = split, .full = g, .part = try g.part(split, b.o.rank, b.o.world), .marker = if (marker) |m| try b.dupe(m) else null };
    }

    fn native(b: *Builder, name: []const u8, spec: names.NativeSpec) !NativePlan {
        const info = try b.pack.need(name);
        const want = spec.dims();
        var ok = spec.store.accepts(info.dtype) and info.rank == want.len;
        if (ok) for (want, info.dims()) |w, d| {
            ok = ok and w == d;
        };
        if (!ok) {
            if (!quiet) log.warn("{s}: {t} {any}, expected {t} storage of {any}", .{ name, info.dtype, info.dims(), spec.store, want });
            return error.UnexpectedTensor;
        }
        try b.mark(name);
        const bytes = switch (spec.slice) {
            .whole => info.nbytes,
            .heads, .vocab_rows => blk: {
                if (info.shape[0] % b.o.world != 0) return error.SplitNotEven;
                break :blk info.nbytes / b.o.world;
            },
        };
        return .{ .name = try b.dupe(name), .info = info, .slice = spec.slice, .bytes = bytes };
    }

    fn experts(b: *Builder, block: []const u8, index: u32, proj: names.ExpertProj) !ExpertsPlan {
        const count = b.c.expertsOf(index).count;
        const parts = try b.a.alloc(exl3.Part, count);
        const prefixes = try b.a.alloc([]const u8, count);
        const kn = proj.kn(b.c);
        var codebook_marker: ?bool = null;
        for (parts, prefixes, 0..) |*p, *pre, e| {
            var nb: [128]u8 = undefined;
            const name = try names.expertName(block, @intCast(e), proj, &nb);
            const g = try b.group(.wq_a, name, proj.split(), kn[0], kn[1]);
            // the codebook is one a projection; here the marker's presence, its value at load
            const has = g.marker != null;
            if (codebook_marker) |m| if (m != has) return error.CodebooksDiffer;
            codebook_marker = has;
            p.* = g.part;
            pre.* = g.prefix;
        }
        return .{ .proj = proj, .parts = parts, .prefixes = prefixes, .layout = try exl3.Layout.of(b.a, parts) };
    }

    fn layer(b: *Builder, index: u32) !LayerPlan {
        var pb: [32]u8 = undefined;
        const block = try b.dupe(try names.blockPrefix(b.c, index, &pb));
        const has_shared = b.pack.has(try std.fmt.bufPrint(&b.buf, "{s}.ffn.shared_experts.w1.trellis", .{block}));
        const specs = try names.blockGroups(b.c, index, b.o.rank, b.o.world, has_shared);
        const groups = try b.a.alloc(GroupPlan, specs.len);
        for (specs.items(), groups) |s, *g| {
            var nb: [128]u8 = undefined;
            g.* = try b.group(s.proj, try names.groupName(s, block, &nb), s.split, s.k, s.n);
            g.slice = s.slice;
        }
        const nspecs = try names.blockNatives(b.c, index);
        const natives = try b.a.alloc(NativePlan, nspecs.len);
        for (nspecs.items(), natives) |s, *v| {
            var nb: [128]u8 = undefined;
            v.* = try b.native(try std.fmt.bufPrint(&nb, "{s}.{s}", .{ block, s.suffix }), s);
        }
        // the other ranks' wo_a groups are theirs to read: mark them so coverage holds on every rank
        var g: u32 = 0;
        while (g < b.c.o_groups) : (g += 1) {
            var nb: [128]u8 = undefined;
            const pre = try std.fmt.bufPrint(&nb, "{s}.attn.wo_a.slice.{d}", .{ block, g });
            inline for (.{ "trellis", "suh", "svh" }) |part| try b.mark(try std.fmt.bufPrint(&b.buf, "{s}." ++ part, .{pre}));
            if (try b.pack.marker(pre, &b.buf)) |m| try b.mark(m);
        }
        var out: LayerPlan = .{ .index = index, .prefix = block, .mode = b.c.mode(index), .groups = groups, .natives = natives, .experts = undefined, .gate_up_mismatch = 0 };
        inline for (.{ .w1, .w2, .w3 }, 0..) |p, i| out.experts[i] = try b.experts(block, index, p);
        for (out.experts[0].layout.words, out.experts[2].layout.words) |w1, w3| out.gate_up_mismatch += @intFromBool(w1 != w3);
        return out;
    }

    /// The latest mtp.i holding `suffix` (as a tensor, a group or a .weight).
    fn findDspark(b: *Builder, suffix: []const u8, buf: []u8) !?[]const u8 {
        var i = b.c.mtp_layers;
        while (i > 0) {
            i -= 1;
            const base = try std.fmt.bufPrint(buf, "mtp.{d}.{s}", .{ i, suffix });
            for ([_][]const u8{ "", ".trellis", ".weight" }) |ext| {
                var nb: [160]u8 = undefined;
                if (b.pack.has(try std.fmt.bufPrint(&nb, "{s}{s}", .{ base, ext }))) return base;
            }
        }
        return null;
    }

    fn dspark(b: *Builder) !?Dspark {
        if (b.c.mtp_layers == 0) return null;
        var fb: [160]u8 = undefined;
        const mp = (try b.findDspark(names.dspark_main_proj, &fb)) orelse return null;
        const main_proj = try b.group(.wq_a, mp, .whole, 3 * b.c.hidden, b.c.hidden);
        const specs = try names.dsparkNatives(b.c);
        const natives = try b.a.alloc(NativePlan, specs.len);
        for (specs.items(), natives) |s, *v| {
            const stem = s.suffix[0 .. s.suffix.len - ".weight".len];
            const base = (try b.findDspark(stem, &fb)) orelse return error.MissingTensor;
            var nb: [160]u8 = undefined;
            v.* = try b.native(try std.fmt.bufPrint(&nb, "{s}.weight", .{base}), s);
        }
        var confidence: ?NativePlan = null;
        if (try b.findDspark("confidence_head.proj", &fb)) |base| {
            var nb: [160]u8 = undefined;
            const name = try std.fmt.bufPrint(&nb, "{s}.weight", .{base});
            const info = try b.pack.need(name);
            var n: u64 = 1;
            for (info.dims()) |d| n *= d;
            if (info.dtype != .f32 and info.dtype != .bf16 or n != b.c.hidden + b.c.dspark_markov_rank) return error.UnexpectedTensor;
            try b.mark(name);
            confidence = .{ .name = try b.dupe(name), .info = info, .slice = .whole, .bytes = info.nbytes };
        }
        return .{ .main_proj = main_proj, .natives = natives, .confidence = confidence };
    }
};

/// The rank's plan; heads, groups and the vocabulary must split over `world` (blocks of 128).
pub fn build(gpa: std.mem.Allocator, c: *const Config, pack: *const Pack, o: Options) !Plan {
    if (o.world == 0 or o.rank >= o.world or c.heads % o.world != 0 or c.o_groups % o.world != 0 or c.vocab % (128 * o.world) != 0) return error.BadWorld;
    var arena = std.heap.ArenaAllocator.init(gpa);
    errdefer arena.deinit();
    var b: Builder = .{ .a = arena.allocator(), .c = c, .pack = pack, .o = o };
    const full = o.blocks == null;
    const layers = try b.a.alloc(LayerPlan, if (o.blocks) |bl| bl.len else c.allLayers());
    for (layers, 0..) |*l, i| {
        const index: u32 = if (o.blocks) |bl| bl[i] else @intCast(i);
        if (index >= c.allLayers()) return error.LayerOutOfBounds;
        l.* = try b.layer(index);
    }
    var top: []NativePlan = &.{};
    var head: ?GroupPlan = null;
    var ds: ?Dspark = null;
    if (full or o.top) {
        const tspecs = try names.topNatives(c);
        top = try b.a.alloc(NativePlan, tspecs.len);
        for (tspecs.items(), top) |s, *v| v.* = try b.native(s.suffix, s);
        head = try b.group(.wq_a, names.head.suffix, .col, c.hidden, c.vocab);
        if (full or o.dspark) ds = try b.dspark();
    }
    // coverage: every stored name was planned or is knowingly left (vision, image routing, bf16 copies of groups)
    var left: std.ArrayList([]const u8) = .empty;
    if (full) for (pack.names.keys()) |k| {
        if (b.used.contains(k) or names.ignored(k, pack)) continue;
        try left.append(b.a, try b.dupe(k));
    };
    return .{ .arena = arena, .rank = o.rank, .world = o.world, .layers = layers, .top = top, .head = head, .dspark = ds, .leftovers = left.items };
}

/// Shared storage owners borrow the Pack; the plan owns its list and prefixes. Backends check capabilities explicitly.
pub const CheckpointPlan = struct {
    arena: std.heap.ArenaAllocator,
    projections: []const Item,
    pub const Item = struct { prefix: []const u8, storage: Pack.Projection };
    pub fn init(a: std.mem.Allocator, pack: *const Pack, prefixes: []const []const u8) !CheckpointPlan {
        var arena = std.heap.ArenaAllocator.init(a);
        errdefer arena.deinit();
        const items = try arena.allocator().alloc(Item, prefixes.len);
        for (items, prefixes) |*item, prefix| item.* = .{ .prefix = try arena.allocator().dupe(u8, prefix), .storage = try pack.projection(prefix) };
        return .{ .arena = arena, .projections = items };
    }
    pub fn deinit(p: *CheckpointPlan) void {
        p.arena.deinit();
        p.* = undefined;
    }
};
