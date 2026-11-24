//! Per-stream CPU recurrent state and MLA cache; weights and temporary rows stay outside this arena.
const std = @import("std");
const config = @import("config.zig");
const mla = @import("cpu_mla.zig");
pub const Kda = struct { matrix: []f32, history: [][3]f32 };
pub const Layer = union(enum) { kda: Kda, mla: mla.Cache };
pub const State = struct {
    arena: std.heap.ArenaAllocator,
    layers: []Layer,
    capacity: u32,
    position: u32 = 0,
    poisoned: bool = false,

    pub fn init(a: std.mem.Allocator, c: config.Config, capacity: u32) !State {
        if (capacity == 0 or capacity > 4096) return error.CpuContextCapacity;
        var arena = std.heap.ArenaAllocator.init(a);
        errdefer arena.deinit();
        const alloc = arena.allocator();
        const layers = try alloc.alloc(Layer, c.layers);
        for (layers, 0..) |*layer, i| {
            layer.* = if (c.kind(@intCast(i)) == .kda) .{ .kda = .{
                .matrix = try alloc.alloc(f32, c.kda_heads * 128 * 128),
                .history = try alloc.alloc([3]f32, 3 * c.kda_heads * 128),
            } } else .{ .mla = try mla.Cache.init(alloc, c.mla_heads, capacity) };
        }
        var s = State{ .arena = arena, .layers = layers, .capacity = capacity };
        s.reset();
        return s;
    }

    pub fn reset(self: *State) void {
        for (self.layers) |*layer| switch (layer.*) {
            .kda => |k| {
                @memset(k.matrix, 0);
                @memset(k.history, .{ 0, 0, 0 });
            },
            .mla => |*m| m.position = 0,
        };
        self.position = 0;
        self.poisoned = false;
    }

    /// Copy only live cache rows; dead MLA storage is never read or published.
    pub fn copyFrom(self: *State, source: *const State) !void {
        if (source.position > self.capacity or source.position > source.capacity or self.layers.len != source.layers.len)
            return error.CpuStateShape;
        for (self.layers, source.layers) |*dst, src| {
            if (std.meta.activeTag(dst.*) != std.meta.activeTag(src)) return error.CpuStateShape;
            switch (src) {
                .kda => |k| {
                    if (dst.kda.matrix.len != k.matrix.len or dst.kda.history.len != k.history.len)
                        return error.CpuStateShape;
                },
                .mla => |m| {
                    if (m.capacity != source.capacity or dst.mla.capacity != self.capacity or
                        dst.mla.k_nope.len / dst.mla.capacity != m.k_nope.len / m.capacity or
                        dst.mla.value.len / dst.mla.capacity != m.value.len / m.capacity or
                        dst.mla.rope.len / dst.mla.capacity != m.rope.len / m.capacity or m.position > self.capacity)
                        return error.CpuStateShape;
                },
            }
        }
        for (self.layers, source.layers) |*dst, src| switch (src) {
            .kda => |k| {
                @memcpy(dst.kda.matrix, k.matrix);
                @memcpy(dst.kda.history, k.history);
            },
            .mla => |m| {
                const n = m.position;
                const keys = m.k_nope.len / m.capacity * n;
                const values = m.value.len / m.capacity * n;
                const rope = m.rope.len / m.capacity * n;
                @memcpy(dst.mla.k_nope[0..keys], m.k_nope[0..keys]);
                @memcpy(dst.mla.value[0..values], m.value[0..values]);
                @memcpy(dst.mla.rope[0..rope], m.rope[0..rope]);
                dst.mla.position = n;
            },
        };
        self.position = source.position;
        self.poisoned = source.poisoned;
    }

    pub fn deinit(self: *State) void {
        self.arena.deinit();
    }
};

test "CPU committed state restores recurrent values and only live attention rows" {
    const a = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const alloc = arena.allocator();
    const source_layers = try alloc.alloc(Layer, 2);
    const target_layers = try alloc.alloc(Layer, 2);
    for ([_][]Layer{ source_layers, target_layers }) |layers| {
        layers[0] = .{ .kda = .{
            .matrix = try alloc.alloc(f32, 4),
            .history = try alloc.alloc([3]f32, 2),
        } };
        layers[1] = .{ .mla = try mla.Cache.init(alloc, 1, 3) };
    }
    var src = State{ .arena = undefined, .layers = source_layers, .capacity = 3, .position = 1 };
    var dst = State{ .arena = undefined, .layers = target_layers, .capacity = 3 };
    @memset(source_layers[0].kda.matrix, 7);
    @memset(source_layers[0].kda.history, .{ 1, 2, 3 });
    @memset(source_layers[1].mla.k_nope, 11);
    @memset(source_layers[1].mla.value, 12);
    @memset(source_layers[1].mla.rope, 13);
    source_layers[1].mla.position = 1;
    @memset(target_layers[1].mla.k_nope, -1);
    @memset(target_layers[1].mla.value, -1);
    @memset(target_layers[1].mla.rope, -1);
    try dst.copyFrom(&src);
    try std.testing.expectEqual(@as(u32, 1), dst.position);
    try std.testing.expectEqualSlices(f32, source_layers[0].kda.matrix, target_layers[0].kda.matrix);
    try std.testing.expectEqualDeep(source_layers[0].kda.history, target_layers[0].kda.history);
    try std.testing.expectEqualSlices(f32, source_layers[1].mla.k_nope[0..128], target_layers[1].mla.k_nope[0..128]);
    try std.testing.expectEqual(@as(f32, -1), target_layers[1].mla.k_nope[128]);
    try std.testing.expectEqual(@as(f32, 12), target_layers[1].mla.value[127]);
    try std.testing.expectEqual(@as(f32, -1), target_layers[1].mla.value[128]);
    try std.testing.expectEqual(@as(f32, 13), target_layers[1].mla.rope[63]);
    try std.testing.expectEqual(@as(f32, -1), target_layers[1].mla.rope[64]);
    dst.reset();
    try std.testing.expectEqual(@as(u32, 0), dst.position);
    try std.testing.expectEqual(@as(f32, 0), target_layers[0].kda.matrix[0]);
    const compact_layers = try alloc.alloc(Layer, 2);
    compact_layers[0] = .{ .kda = .{
        .matrix = try alloc.alloc(f32, 4),
        .history = try alloc.alloc([3]f32, 2),
    } };
    compact_layers[1] = .{ .mla = try mla.Cache.init(alloc, 1, 1) };
    var compact = State{ .arena = undefined, .layers = compact_layers, .capacity = 1 };
    try compact.copyFrom(&src);
    try dst.copyFrom(&compact);
    try std.testing.expectEqual(@as(u32, 1), dst.position);
    try std.testing.expectEqualSlices(f32, compact_layers[0].kda.matrix, target_layers[0].kda.matrix);
    try std.testing.expectEqualSlices(f32, compact_layers[1].mla.k_nope, target_layers[1].mla.k_nope[0..128]);
    src.poisoned = true;
    try dst.copyFrom(&src);
    try std.testing.expect(dst.poisoned);
    dst.capacity = 2;
    try std.testing.expectError(error.CpuStateShape, dst.copyFrom(&src));
}
