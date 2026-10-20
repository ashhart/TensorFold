//! A MoE layer's MLP on the frame's normed rows: router logits, top-k, routed and shared experts, one weighted sum.
const std = @import("std");
const mtl = @import("metal");
const core = @import("core");
const tile = core.tile_qmm;
const route = core.moe_route;
const sources = @import("qwen_runtime_sources");
const Config = @import("config.zig").Config;
const Ref = @import("projection.zig").Ref;
const metal_experts = core.qwen35_experts;
pub const Experts = metal_experts.Experts;

/// A buffer and byte offset as core/moe_route binds them.
const Bind = struct { buf: mtl.Buffer, off: usize = 0 };

const kernel_names = [_][:0]const u8{ "q27_moe_select", "q27_moe_sort", "q27_moe_combine" };
pub const Kernel = enum { select, sort, combine };

/// The scratch rows: one buffer a field, sized for the frame's rows.
pub const Field = enum { logits, picks, wts, starts, pos, rows, one, work, work_one, x_sums, act_sums, shared_sums, act, y, shared_act, shared_y, zeros, ident, groups };

pub const Stage = struct {
    config: Config,
    capacity: u32,
    tiles: tile.Pipelines,
    logits: mtl.Pipeline,
    kernels: [kernel_names.len]mtl.Pipeline,
    /// Whether the twin equals the tiles at both expert shapes on this GPU; otherwise decode rows take the tiles too.
    twin: bool,
    buffers: [std.enums.values(Field).len]mtl.Buffer,

    /// The pipelines and scratch for `capacity` rows a call; the twin check reads layer 0's experts.
    pub fn init(a: std.mem.Allocator, device: mtl.Device, c: Config, capacity: u32, first: Experts) !*Stage {
        const s = try a.create(Stage);
        errdefer a.destroy(s);
        s.config = c;
        s.capacity = capacity;
        s.tiles = try tile.Pipelines.load(device);
        errdefer s.tiles.deinit();
        const shape = routeShape(c);
        const route_src = try route.source(a, shape);
        defer a.free(route_src);
        const route_lib = try mtl.Library.fromSource(device, route_src, mtl.CompileOptions.mlx());
        defer route_lib.deinit();
        s.logits = try mtl.Pipeline.init(device, route_lib, route.names[0], false);
        errdefer s.logits.deinit();
        const src = try source(a, c);
        defer a.free(src);
        const lib = try mtl.Library.fromSource(device, src, mtl.CompileOptions.mlx());
        defer lib.deinit();
        var made: usize = 0;
        errdefer for (s.kernels[0..made]) |p| p.deinit();
        for (kernel_names, &s.kernels) |name, *p| {
            p.* = try mtl.Pipeline.init(device, lib, name, false);
            made += 1;
        }
        var count: usize = 0;
        errdefer for (s.buffers[0..count]) |b| b.deinit();
        for (sizes(c, capacity), &s.buffers) |bytes, *b| {
            b.* = try device.buffer(@max(bytes, 16), mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
            count += 1;
        }
        for (s.get(.ident).slice(u32, capacity * c.topk), 0..) |*v, i| v.* = @intCast(i);
        @memset(s.get(.zeros).slice(u32, capacity), 0);
        s.twin = try tile.twinExact(&s.tiles, device, first.gate) and try tile.twinExact(&s.tiles, device, first.down) and
            try tile.twinExact(&s.tiles, device, first.shared_gate) and try tile.twinExact(&s.tiles, device, first.shared_down);
        if (!s.twin) std.log.info("qwen27: the expert twin differs from the tiles on this GPU; decode rows take the tiles", .{});
        return s;
    }

    pub fn deinit(s: *Stage, a: std.mem.Allocator) void {
        for (s.buffers) |b| b.deinit();
        for (s.kernels) |p| p.deinit();
        s.logits.deinit();
        s.tiles.deinit();
        a.destroy(s);
    }

    pub fn get(s: *const Stage, f: Field) mtl.Buffer {
        return s.buffers[@backingInt(f)];
    }

    pub fn pipe(s: *const Stage, k: Kernel) mtl.Pipeline {
        return s.kernels[@backingInt(k)];
    }

    /// out [rows, hidden] = the layer's experts on `x` (normed rows, offset 0); `prompt` rows take the tiles.
    pub fn encode(s: *const Stage, e: mtl.ComputeEncoder, x: Experts, input: Ref, out: Ref, rows: u32, prompt: bool) !void {
        if (rows == 0 or rows > s.capacity or input.offset != 0) return error.BadMoeRows;
        const c = s.config;
        const t = &s.tiles;
        const slots: u32 = rows * @as(u32, @intCast(c.topk));
        const in = input.buffer;
        route.logits(e, s.logits, routeShape(c), Bind{ .buf = in }, Bind{ .buf = x.router }, Bind{ .buf = s.get(.logits) }, rows);
        e.barrier();
        e.setPipeline(s.pipe(.select));
        e.setBuffer(s.get(.logits), 0, 0);
        e.setBuffer(s.get(.picks), 0, 1);
        e.setBuffer(s.get(.wts), 0, 2);
        e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(32, 1, 1));
        e.barrier();
        const twin = !prompt and s.twin and slots <= tile.window_slots;
        if (twin) {
            // a row's picks are distinct experts; wide windows read each picked expert once for its rows
            const groups: ?mtl.Buffer = if (rows >= tile.group_rows) s.get(.groups) else null;
            if (groups) |g| {
                try tile.encodeTwinGroup(e, t, s.get(.picks), slots, g);
                e.barrier();
            }
            try tile.encodeTwinGlu(e, t, x.gate, x.up, in, s.get(.act), s.get(.picks), groups, slots, c.topk);
            try tile.encodeTwinGlu(e, t, x.shared_gate, x.shared_up, in, s.get(.shared_act), s.get(.zeros), null, rows, 1);
            e.barrier();
            try tile.encodeTwinIndexed(e, t, x.down, s.get(.act), s.get(.y), s.get(.picks), groups, slots, 1);
            try tile.encodeTwinIndexed(e, t, x.shared_down, s.get(.shared_act), s.get(.shared_y), s.get(.zeros), null, rows, 1);
            e.barrier();
        } else {
            e.setPipeline(s.pipe(.sort));
            e.setBuffer(s.get(.picks), 0, 0);
            e.setValue(slots, 1);
            e.setBuffer(s.get(.starts), 0, 2);
            e.setBuffer(s.get(.pos), 0, 3);
            e.setBuffer(s.get(.rows), 0, 4);
            e.setBuffer(s.get(.one), 0, 5);
            e.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(c.experts * sort_chunks, 1, 1));
            tile.encodeSums(e, t, c.hidden, in, 0, s.get(.x_sums), rows);
            e.barrier();
            tile.encodeWork(e, t, s.get(.starts), c.experts, s.get(.work));
            tile.encodeWork(e, t, s.get(.one), 1, s.get(.work_one));
            e.barrier();
            try tile.encodeGroupedGlu(e, t, x.gate, x.up, c.experts, in, s.get(.x_sums), s.get(.rows), s.get(.act), slots, s.get(.starts), s.get(.work));
            try tile.encodeGroupedGlu(e, t, x.shared_gate, x.shared_up, 1, in, s.get(.x_sums), s.get(.ident), s.get(.shared_act), rows, s.get(.one), s.get(.work_one));
            e.barrier();
            tile.encodeSums(e, t, c.moe_intermediate, s.get(.act), 0, s.get(.act_sums), slots);
            tile.encodeSums(e, t, c.shared_intermediate, s.get(.shared_act), 0, s.get(.shared_sums), rows);
            e.barrier();
            try tile.encodeGrouped(e, t, x.down, c.experts, s.get(.act), s.get(.act_sums), s.get(.ident), s.get(.y), slots, s.get(.starts), s.get(.work));
            try tile.encodeGrouped(e, t, x.shared_down, 1, s.get(.shared_act), s.get(.shared_sums), s.get(.ident), s.get(.shared_y), rows, s.get(.one), s.get(.work_one));
            e.barrier();
        }
        e.setPipeline(s.pipe(.combine));
        e.setBuffer(s.get(.y), 0, 0);
        e.setBuffer(s.get(.shared_y), 0, 1);
        e.setBuffer(s.get(.wts), 0, 2);
        e.setBuffer(s.get(.pos), 0, 3);
        e.setValue(@as(u32, @intFromBool(!twin)), 4);
        e.setBuffer(out.buffer, out.offset, 5);
        e.dispatchThreads(mtl.Size.of(c.hidden, rows, 1), mtl.Size.of(256, 1, 1));
        e.barrier();
    }
};

/// The sort kernel's chunks an expert (moe.metal's SORT_CHUNKS): one threadgroup of experts * chunks threads.
const sort_chunks = 4;

fn routeShape(c: Config) route.Shape {
    return .{ .hidden = @intCast(c.hidden), .experts = @intCast(metal_experts.columns(c.experts)), .topk = @intCast(c.topk) };
}

/// moe.metal with the config's constants at its Q27_SPECIALIZE mark; the caller frees it.
pub fn source(a: std.mem.Allocator, c: Config) ![]u8 {
    if (c.experts * sort_chunks > 1024 or c.topk == 0 or c.topk > 16 or c.hidden % 256 != 0 or c.hidden > 4096) return error.UnsupportedMoeShape; // tf_route_logits covers at most 4096
    const text = sources.moe;
    const mark = std.mem.indexOf(u8, text, "Q27_SPECIALIZE") orelse return error.BadKernelSource;
    const defs = try std.fmt.allocPrint(a, "constant constexpr uint E = {d}, TOPK = {d}, LS = {d}, D = {d};", .{ c.experts, c.topk, metal_experts.columns(c.experts), c.hidden });
    defer a.free(defs);
    return std.mem.concat(a, u8, &.{ text[0..mark], defs, text[mark + "Q27_SPECIALIZE".len ..] });
}

/// Bytes a field takes for `r` rows.
fn sizes(c: Config, r: usize) [std.enums.values(Field).len]usize {
    const slots = r * c.topk;
    return .{
        r * metal_experts.columns(c.experts) * 4, // logits
        slots * 4, // picks
        r * (c.topk + 1) * 4, // wts
        (c.experts + 1) * 4, // starts
        slots * 4, // pos
        slots * 4, // rows
        2 * 4, // one
        (tile.items(slots, c.experts) + 1) * 4, // work
        (tile.items(r, 1) + 1) * 4, // work_one
        r * (c.hidden / 64) * 4, // x_sums
        slots * (c.moe_intermediate / 64) * 4, // act_sums
        r * (c.shared_intermediate / 64) * 4, // shared_sums
        slots * c.moe_intermediate * 2, // act
        slots * c.hidden * 2, // y
        r * c.shared_intermediate * 2, // shared_act
        r * c.hidden * 2, // shared_y
        r * 4, // zeros
        slots * 4, // ident
        tile.groupWords(tile.window_slots) * 4, // groups
    };
}

test "the sort's chunks are moe.metal's and the source takes the config's constants" {
    const line = std.fmt.comptimePrint("constant constexpr uint SORT_CHUNKS = {d};", .{sort_chunks});
    try std.testing.expect(std.mem.indexOf(u8, sources.moe, line) != null);
    var c: Config = undefined;
    c.experts = 256;
    c.topk = 8;
    c.hidden = 2048;
    const src = try source(std.testing.allocator, c);
    defer std.testing.allocator.free(src);
    try std.testing.expect(std.mem.indexOf(u8, src, "E = 256, TOPK = 8, LS = 260, D = 2048;") != null);
    c.experts = 512;
    try std.testing.expectError(error.UnsupportedMoeShape, source(std.testing.allocator, c));
}
