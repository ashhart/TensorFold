//! An EXL3 pack's trellis groups on the Metal lane: queued by weights.zig's Loader beside the natives, then viewed.
const std = @import("std");
const mtl = @import("metal");
const host = @import("dsv41");
const x3 = @import("core").exl3_projection;
const xe = @import("experts_exl3.zig");
const lane = @import("exl3_lane.zig");
const weights = @import("weights.zig");

const Config = host.config.Config;
const Loader = weights.Loader;
const Place = weights.Place;
pub const Matrix = x3.Matrix;
pub const Book = x3.format.Codebook;
pub const Descriptor = x3.ExpertDescriptor;

/// The kernels read trellises and scales as ushort; a marker is read once, on the host.
const align_x3: u32 = 2;

/// A group queued in the arena or a mapped shard: its tensors, K x N, width in half-bits, codebook and scale format.
pub const Placed = struct { trellis: Place, su: Place, sv: Place, marker: ?Place, book: Book, k: u32, n: u32, halves: u16, signs: bool };

/// A block's dense linears by the forward's names; T is a queued group, then its Matrix.
pub fn Dense(comptime T: type) type {
    return struct {
        wq_a: T,
        wkv: T,
        wq_b: T,
        /// the output projection's groups, `attn.wo_a.slice.{g}`; null past the config's o_groups
        wo_a: [lane.max_groups]?T,
        wo_b: T,
        comp_wkv: ?T,
        comp_wgate: ?T,
        ix_wk: ?T,
        ix_wq_b: ?T,
        /// w1, w2, w3
        shared: [3]T,
        engram_wkv: ?T,
    };
}

/// A layer's EXL3 half: its dense linears, and gate, up and down descriptors of the routed experts in its table.
pub const Layer = struct { m: Dense(Matrix), experts: xe.Tables };

/// What `place` queued: the dense groups, and the gate (w1), up (w3) and down (w2) of each routed expert.
pub const Places = struct { dense: Dense(Placed), experts: [][3]Placed };

/// `prefix`'s trellis group, refused unless it is one of K x N in a width and codebook the kernels decode.
pub fn group(l: *Loader, prefix: []const u8, k: u32, n: u32) !Placed {
    const p = try host.projection.read(l.pack, prefix);
    if (p != .exl3) return error.UnsupportedProjection;
    const g = p.exl3;
    if (g.geometry.k() != k or g.geometry.n() != n) return error.BadProjectionShape;
    var buf: [256]u8 = undefined;
    const marker = try l.pack.marker(prefix, &buf);
    const book: Book = if (marker) |name| (if (std.mem.endsWith(u8, name, ".mul1")) .mul1 else .mcg) else .inst3;
    const halves: u16 = @intCast(g.geometry.k2());
    const m: Matrix = .{ .trellis = undefined, .suh = undefined, .svh = undefined, .k = k, .n = n, .width = .{ .halves = halves }, .codebook = book };
    try m.validate();
    return .{
        .trellis = try l.add(g.trellis, .raw, align_x3),
        .su = try l.add(g.scales.input, .raw, align_x3),
        .sv = try l.add(g.scales.output, .raw, align_x3),
        .marker = if (g.marker) |info| try l.add(info, .raw, 4) else null,
        .book = book,
        .k = k,
        .n = n,
        .halves = halves,
        .signs = g.scales.format == .packed_signs,
    };
}

fn named(l: *Loader, buf: []u8, block: []const u8, suffix: []const u8, k: u32, n: u32) !Placed {
    return group(l, try std.fmt.bufPrint(buf, "{s}.{s}", .{ block, suffix }), k, n);
}

/// Layer `layer`'s trellis groups under `block`, each against the config's K x N; `experts` is the caller's to free.
pub fn place(l: *Loader, a: std.mem.Allocator, block: []const u8, c: *const Config, layer: u32) !Places {
    if (c.o_groups > lane.max_groups) return error.UnsupportedShape;
    var nb: [160]u8 = undefined;
    const kg = c.heads / c.o_groups * c.head_dim;
    const W = c.expert_width;
    var d: Dense(Placed) = .{
        .wq_a = try named(l, &nb, block, "attn.wq_a", c.hidden, c.q_lora),
        .wkv = try named(l, &nb, block, "attn.wkv", c.hidden, c.head_dim),
        .wq_b = try named(l, &nb, block, "attn.wq_b", c.q_lora, c.heads * c.head_dim),
        .wo_a = @splat(null),
        .wo_b = try named(l, &nb, block, "attn.wo_b", c.o_groups * c.o_lora, c.hidden),
        .comp_wkv = null,
        .comp_wgate = null,
        .ix_wk = null,
        .ix_wq_b = null,
        .shared = .{
            try named(l, &nb, block, "ffn.shared_experts.w1", c.hidden, W),
            try named(l, &nb, block, "ffn.shared_experts.w2", W, c.hidden),
            try named(l, &nb, block, "ffn.shared_experts.w3", c.hidden, W),
        },
        .engram_wkv = null,
    };
    for (d.wo_a[0..c.o_groups], 0..) |*g, gi| g.* = try group(l, try std.fmt.bufPrint(&nb, "{s}.attn.wo_a.slice.{d}", .{ block, gi }), kg, c.o_lora);
    if (c.isKvSource(layer)) {
        d.comp_wkv = try named(l, &nb, block, "attn.compressor.wkv", c.hidden, c.head_dim);
        if (c.compressRatio(layer) > 1) d.comp_wgate = try named(l, &nb, block, "attn.compressor.wgate", c.hidden, c.head_dim);
        d.ix_wk = try named(l, &nb, block, "attn.indexer.wk", c.head_dim, c.index_dim);
    }
    if (c.isIndexSource(layer)) d.ix_wq_b = try named(l, &nb, block, "attn.indexer.wq_b", c.q_lora, c.index_heads * c.index_dim);
    if (c.isEngram(layer)) d.engram_wkv = try named(l, &nb, block, "engram.wkv", c.hashCols() * c.engram_head_dim, (c.hc_mult + 1) * c.hidden);
    const ex = try a.alloc([3]Placed, c.expertsOf(layer).count);
    errdefer a.free(ex);
    for (ex, 0..) |*e, i| {
        var pb: [96]u8 = undefined;
        const pre = try std.fmt.bufPrint(&pb, "{s}.ffn.experts.{d}", .{ block, i });
        e.* = .{ try named(l, &nb, pre, "w1", c.hidden, W), try named(l, &nb, pre, "w3", c.hidden, W), try named(l, &nb, pre, "w2", W, c.hidden) };
    }
    return .{ .dense = d, .experts = ex };
}

/// The stored value of each codebook's marker: its multiplier (format.py's MARKERS).
fn magic(b: Book) u32 {
    return switch (b) {
        .mul1 => x3.format.Mul1Mul,
        .mcg => x3.format.McgMul,
        .inst3 => unreachable,
    };
}

/// A group's marker holds its codebook's value; `arena`: the bytes the copied tensors were filled into.
pub fn checkMarker(p: Placed, arena: []const u8) !void {
    const mk = p.marker orelse return;
    const at: [*]const u8 = if (mk.buf) |b| b.contents() + mk.off else arena[mk.off..].ptr;
    if (std.mem.readInt(u32, at[0..4], .little) != magic(p.book)) return error.BadMarker;
}

/// Every queued group's marker, after the loader's fill.
pub fn checkMarkers(p: Places, arena: []const u8) !void {
    const s = @typeInfo(Dense(Placed)).@"struct";
    inline for (s.field_names, s.field_types) |name, T| {
        const v = @field(p.dense, name);
        switch (T) {
            Placed => try checkMarker(v, arena),
            ?Placed => if (v) |g| try checkMarker(g, arena),
            [3]Placed => for (v) |g| try checkMarker(g, arena),
            [lane.max_groups]?Placed => for (v) |o| if (o) |g| try checkMarker(g, arena),
            else => @compileError("an unhandled dense field"),
        }
    }
    for (p.experts) |e| for (e) |g| try checkMarker(g, arena);
}

fn viewOf(p: Place, arena: mtl.Buffer) x3.View {
    return .{ .buffer = p.buf orelse arena, .offset = p.off };
}

/// The group as the kernels read it, over the arena and the shards its tensors were placed in.
pub fn matrix(p: Placed, arena: mtl.Buffer) Matrix {
    const sf: x3.ScaleFormat = if (p.signs) .packed_signs else .f16;
    return .{ .trellis = viewOf(p.trellis, arena), .suh = viewOf(p.su, arena), .svh = viewOf(p.sv, arena), .k = p.k, .n = p.n, .width = .{ .halves = p.halves }, .codebook = p.book, .suh_format = sf, .svh_format = sf };
}

/// Bytes of a layer's expert table: gate, up and down descriptors of each expert.
pub fn tableBytes(count: usize) usize {
    return 3 * count * @sizeOf(Descriptor);
}

/// After the fill: the markers checked, the dense views, and the experts' descriptors written into `table`.
pub fn finish(p: Places, arena: mtl.Buffer, table: mtl.Buffer) !Layer {
    try checkMarkers(p, arena.contents()[0..arena.length()]);
    var d: Dense(Matrix) = undefined;
    const s = @typeInfo(Dense(Placed)).@"struct";
    inline for (s.field_names, s.field_types) |name, T| {
        const v = @field(p.dense, name);
        @field(d, name) = switch (T) {
            Placed => matrix(v, arena),
            ?Placed => if (v) |g| matrix(g, arena) else null,
            [3]Placed => .{ matrix(v[0], arena), matrix(v[1], arena), matrix(v[2], arena) },
            [lane.max_groups]?Placed => blk: {
                var o: [lane.max_groups]?Matrix = @splat(null);
                for (v, &o) |g, *m| if (g) |x| {
                    m.* = matrix(x, arena);
                };
                break :blk o;
            },
            else => @compileError("an unhandled dense field"),
        };
    }
    const count = p.experts.len;
    const all = table.slice(Descriptor, 3 * count);
    for (p.experts, 0..) |e, i| for (e, 0..) |g, v| {
        all[v * count + i] = try x3.descriptor(matrix(g, arena));
    };
    const one = count * @sizeOf(Descriptor);
    return .{ .m = d, .experts = .{ .gate = .{ .buf = table }, .up = .{ .buf = table, .off = one }, .down = .{ .buf = table, .off = 2 * one }, .count = @intCast(count) } };
}

/// Layer `layer`'s trellises into `w`'s arena and table; its FP8 fields stay empty, so no FP8 kernel reads them.
pub fn load(w: *weights.Layer, l: *Loader, a: std.mem.Allocator, io: std.Io, device: mtl.Device, block: []const u8, c: *const Config, layer: u32) !void {
    const p = try place(l, a, block, c, layer);
    defer a.free(p.experts);
    const arena = try device.buffer(@max(l.size, 256), mtl.ResourceOptions.shared);
    errdefer arena.deinit();
    try l.fill(io, arena.contents()[0..l.size]);
    const table = try device.buffer(@max(tableBytes(p.experts.len), 256), mtl.ResourceOptions.shared);
    errdefer table.deinit();
    w.x3 = try finish(p, arena, table);
    w.arena = arena;
    w.table = table;
    w.back[0] = arena;
    @memcpy(w.back[1 .. 1 + l.nback], l.back[0..l.nback]);
    w.nback = l.nback + 1;
    const none: weights.Fp8 = .{ .w = .{ .buf = arena }, .s = .{ .buf = arena }, .k = 0, .n = 0 };
    inline for (.{ "wq_a", "wkv", "wq_b", "wo_a", "wo_b" }) |f| @field(w, f) = none;
    w.shared = @splat(none);
    inline for (.{ "ix_wq_b", "comp_wkv", "comp_wgate", "ix_wk", "comp_norm", "ix_k_norm", "ix_wp" }) |f| @field(w, f) = null;
    w.engram = if (c.isEngram(layer)) .{ .wkv = none, .q = undefined, .k = undefined } else null;
}

test {
    _ = @import("weights_exl3_test.zig");
}
