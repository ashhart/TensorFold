//! Mutate actual inputs to the production identity builder; no CUDA calls or user-cache writes.
const std = @import("std");
const q = @import("qwen35");
const adapter = q.native.cost_adapter;
const a = std.testing.allocator;
const io = std.testing.io;
const Driver = struct {
    version_number: i32 = 13000,
    pub fn version(d: Driver) !i32 {
        return d.version_number;
    }
};
const Context = struct {
    d: Driver = .{},
    product: []const u8 = "product A",
    cc: u32 = 121,
    sms: i32 = 48,
    pub fn name(c: Context, _: []u8) ![]const u8 {
        return c.product;
    }
    pub fn capability(c: Context) !u32 {
        return c.cc;
    }
    pub fn attribute(c: Context, _: enum { multiprocessor_count }) !i32 {
        return c.sms;
    }
};
const Draft = struct {
    c: q.draft.Config,
    trees: bool = true,
    nodes: u32 = 32,
    pub fn drafter(d: *Draft) *Draft {
        return d;
    }
    pub fn facts(d: *Draft) struct { trees: bool, tree_nodes: u32 } {
        return .{ .trees = d.trees, .tree_nodes = d.nodes };
    }
};

pub fn check() !void {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(root);
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = "{}" });
    const header = "{}";
    var bytes: [8 + header.len]u8 = @splat(0);
    std.mem.writeInt(u64, bytes[0..8], header.len, .little);
    @memcpy(bytes[8..], header);
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = &bytes });
    var context: Context = .{};
    var e: q.engine.Engine = undefined;
    e.kit = null;
    e.dir = root;
    e.learning = null;
    e.features = 0;
    e.pockets.layers = &.{};
    e.g = .{ .hidden = 5120, .intermediate = 17408, .layers = 64, .vocab = 248320, .k_heads = 16, .v_heads = 48, .q_heads = 24, .kv_heads = 4, .eps = 1e-6, .tied = false, .context = 262144, .linear = @splat(false) };
    e.kv8 = false;
    e.verify_rows = 128;
    e.window_rows = 64;
    e.n_taps = 2;
    e.taps[0..2].* = .{ 10, 20 };
    const Lanes = struct { gpa: std.mem.Allocator = a, io: std.Io, e: *q.engine.Engine, cost_context: ?*Context, cost_identity: ?[32]u8 = @splat(1), cost_kind: adapter.Kind = .dflash };
    var l: Lanes = .{ .io = io, .e = &e, .cost_context = &context };
    var d: Draft = .{ .c = try q.draft.Config.parse(a, @embedFile("tests/fixtures/qwen35_cost_dflash.json"), e.g) };
    var cache: q.engine.Cache = undefined;
    cache.cap = 64;
    const first = try adapter.identity(&l, &d, &cache);
    for (0..18) |change| {
        for (0..2) |pass| {
            switch (change) {
                0 => context.product = if (pass == 0) "product B" else "product A",
                1 => context.sms ^= 1,
                2 => context.cc ^= 1,
                3 => context.d.version_number ^= 1,
                4 => e.g.hidden ^= 1,
                5 => e.g.linear[0] = !e.g.linear[0],
                6 => e.g.bits.proj[0][0] ^= 1,
                7 => e.g.expert_bits ^= 1,
                8 => e.kv8 = !e.kv8,
                9 => e.taps[1] ^= 1,
                10 => cache.cap ^= 128,
                11 => e.verify_rows ^= 64,
                12 => e.window_rows ^= 96,
                13 => d.c.block ^= 8,
                14 => d.nodes ^= 16,
                15 => d.trees = !d.trees,
                16 => l.cost_kind = if (pass == 0) .mtp else .dflash,
                else => l.cost_identity.?[0] ^= 1,
            }
            const changed = try adapter.identity(&l, &d, &cache);
            try std.testing.expectEqual(pass == 1, std.mem.eql(u8, &first.current, &changed.current));
            try std.testing.expectEqual(pass == 1, std.mem.eql(u8, &first.reference, &changed.reference));
        }
    }
    cache.cap = 32;
    try std.testing.expectEqualDeep(first, try adapter.identity(&l, &d, &cache));
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = "{\"revision\":2}" });
    const changed_target = try adapter.identity(&l, &d, &cache);
    try std.testing.expect(!std.mem.eql(u8, &first.current, &changed_target.current));
    e.features = 1;
    try std.testing.expectError(error.NonTextCostProfile, adapter.identity(&l, &d, &cache));
}
