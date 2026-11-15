//! The release format's rank plans on a header-only release at real shapes: coverage, tiling, balance, refusals.
const std = @import("std");
const testing = std.testing;
const Config = @import("config.zig").Config;
const pk = @import("pack.zig");
const plan = @import("plan.zig");
const names = @import("names.zig");
const named = @import("named.zig");
const Share = @import("partition.zig").Share;

const Specs = std.ArrayList(pk.Spec);

fn add(a: std.mem.Allocator, out: *Specs, comptime f: []const u8, args: anytype, dtype: []const u8, shape: []const usize) !void {
    try out.append(a, .{ .name = try std.fmt.allocPrint(a, f, args), .dtype = dtype, .shape = try a.dupe(usize, shape) });
}

fn fp8(a: std.mem.Allocator, out: *Specs, prefix: []const u8, n: usize, k: usize) !void {
    try add(a, out, "{s}.weight", .{prefix}, "F8_E4M3", &.{ n, k });
    try add(a, out, "{s}.scale", .{prefix}, "F8_E8M0", &.{ (n + 31) / 32, (k + 31) / 32 });
}

fn wide(a: std.mem.Allocator, dims: []const u32) ![]const usize {
    const out = try a.alloc(usize, dims.len);
    for (out, dims) |*o, d| o.* = d;
    return out;
}

fn storeDtype(s: names.Store) []const u8 {
    return switch (s) {
        .f32, .bias => "F32",
        .norm, .half, .param => "BF16",
    };
}

/// Every tensor the text engine reads from the release, at its real dtype and shape (an arena owns the strings).
fn releaseSpecs(a: std.mem.Allocator, c: *const Config) ![]pk.Spec {
    var out: Specs = .empty;
    for (0..c.allLayers()) |li| {
        const L: u32 = @intCast(li);
        var pb: [32]u8 = undefined;
        const block = try a.dupe(u8, try names.blockPrefix(c, L, &pb));
        for ((try names.releaseBlocks(c, L, true)).items()) |s| {
            const prefix = try std.fmt.allocPrint(a, "{s}.{s}", .{ block, s.suffix });
            switch (s.proj) {
                .comp_wkv, .comp_wgate, .ix_wk => try add(a, &out, "{s}.weight", .{prefix}, "BF16", &.{ s.n, s.k }),
                .wo_a => try fp8(a, &out, prefix, c.o_groups * c.o_lora, s.k),
                .engram_wkv => try fp8(a, &out, prefix, (c.hc_mult + 1) * c.hidden, (c.engram_max_ngram - 1) * c.engram_heads * c.engram_head_dim),
                .engram_embed => {
                    const rows = c.engram_rows.items()[std.mem.indexOfScalar(u32, c.engram_layers.items(), L).?];
                    try add(a, &out, "{s}.weight", .{prefix}, "F8_E4M3", &.{ rows, 256 });
                    try add(a, &out, "{s}.scale", .{prefix}, "F8_E8M0", &.{ rows, 8 });
                },
                else => try fp8(a, &out, prefix, s.n, s.k),
            }
        }
        for ((try names.blockNatives(c, L)).items()) |s| try add(a, &out, "{s}.{s}", .{ block, s.suffix }, storeDtype(s.store), try wide(a, s.dims()));
        for (0..c.expertsOf(L).count) |e| inline for (.{ .w1, .w2, .w3 }) |p| {
            const kn = names.ExpertProj.kn(p, c);
            var nb: [128]u8 = undefined;
            const prefix = try a.dupe(u8, try names.expertName(block, @intCast(e), p, &nb));
            try add(a, &out, "{s}.weight", .{prefix}, "I8", &.{ kn[1], kn[0] / 2 });
            try add(a, &out, "{s}.scale", .{prefix}, "F8_E8M0", &.{ kn[1], kn[0] / 32 });
        };
    }
    try add(a, &out, "embed.weight", .{}, "BF16", &.{ c.vocab, c.hidden });
    try add(a, &out, "norm.weight", .{}, "BF16", &.{c.hidden});
    try add(a, &out, "head.weight", .{}, "BF16", &.{ c.vocab, c.hidden });
    const last = c.mtp_layers - 1;
    try fp8(a, &out, try std.fmt.allocPrint(a, "mtp.{d}.main_proj", .{last}), c.hidden, 3 * c.hidden);
    for ((try names.dsparkNatives(c)).items()) |s| try add(a, &out, "mtp.{d}.{s}", .{ last, s.suffix }, storeDtype(s.store), try wide(a, s.dims()));
    return out.items;
}

fn releasePack(a: std.mem.Allocator, arena: std.mem.Allocator, c: *const Config) !pk.Pack {
    const header = try pk.headerOnly(a, try releaseSpecs(arena, c));
    defer a.free(header);
    var p = pk.Pack.init(a);
    errdefer p.deinit();
    try p.addHeader("model.safetensors", header[8..], std.math.maxInt(u64)); // the JSON after its length
    return p;
}

fn releaseConfig() Config {
    var c: Config = .{};
    c.format = .release_fp8;
    return c;
}

test "release fixture: every rank of worlds 1, 2 and 4 plans the whole release on 32-blocks, nothing left over" {
    const a = testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    const c = releaseConfig();
    var pack = try releasePack(a, arena.allocator(), &c);
    defer pack.deinit();
    var experts_one: u64 = 0;
    for ([_]u32{ 1, 2, 4 }) |world| {
        var experts_sum: u64 = 0;
        var wq_b_rows: u32 = 0;
        var wo_b_cols: u32 = 0;
        var shared_rows: u32 = 0;
        for (0..world) |rank| {
            var p = try plan.build(a, &c, &pack, .{ .rank = @intCast(rank), .world = world });
            defer p.deinit();
            try testing.expectEqual(@as(usize, 0), p.leftovers.len);
            try @import("cuda_weights.zig").admitPlan(testing.io, &c, &pack, &p);
            var experts: u64 = 0;
            for (p.layers) |*l| {
                experts += l.expertBytes();
                for (l.groups) |g| switch (g.form) {
                    .fp8_block => |f| for ([_]u32{ f.rows.lo, f.rows.len, f.cols.lo, f.cols.len }) |x| try testing.expect(x % 32 == 0),
                    .bf16 => {},
                    .exl3 => return error.TestUnexpectedResult,
                };
                if (l.index != 2) continue;
                for (l.groups) |g| switch (g.proj) {
                    .wq_b => wq_b_rows += g.form.fp8_block.rows.len,
                    .wo_b => wo_b_cols += g.form.fp8_block.cols.len,
                    .shared_w1 => shared_rows += g.form.fp8_block.rows.len,
                    else => {},
                };
            }
            try testing.expect(p.head.?.form == .bf16 and p.dspark.?.main_proj.form == .fp8_block);
            experts_sum += experts;
            if (world == 1) experts_one = experts;
            if (world == 4) try testing.expectEqual(experts_one / 4, experts);
        }
        try testing.expectEqual(experts_one, experts_sum);
        try testing.expectEqual(c.heads * c.head_dim, wq_b_rows);
        try testing.expectEqual(c.o_groups * c.o_lora, wo_b_cols);
        try testing.expectEqual(c.expert_width, shared_rows);
    }
}

test "release fixture: the Engram tables are checked but never named for upload nor counted in the rank's bytes" {
    const a = testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    const c = releaseConfig();
    var pack = try releasePack(a, arena.allocator(), &c);
    defer pack.deinit();
    var p = try plan.build(a, &c, &pack, .{ .rank = 1, .world = 4, .blocks = &.{ 1, 14 } });
    defer p.deinit();
    var b: named.Builder = .{ .gpa = a, .io = testing.io, .pack = &pack, .metadata_only = true };
    defer b.deinit();
    for (p.layers) |*l| try b.layer(l, c.o_groups, p.rank, p.world);
    for (b.out.items) |n| try testing.expect(std.mem.indexOf(u8, n.name, "engram.embed") == null);
    try testing.expect(b.find("L14.engram.wkv.w") != null and b.find("L14.engram.wkv.s") != null);
    const w2 = b.find("L1.moe.w2.w").?;
    try testing.expectEqualSlices(usize, &.{ 384, 5120, 288 }, w2.shape[0..3]);
    // one block's routed experts dominate; the 2 x ~94 GiB tables would dwarf them if they were counted
    for (p.layers) |*l| try testing.expect(l.bytes() < 4 << 30);
}

test "release fixture: a misshaped scale and an F16 projection are refused by name" {
    const a = testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    const c = releaseConfig();
    const specs = try releaseSpecs(arena.allocator(), &c);
    for (specs) |*s| if (std.mem.eql(u8, s.name, "layers.0.attn.wq_a.scale")) {
        s.shape = &.{ 40, 161 };
    };
    for ([_][]const u8{ "scale", "dtype", "engram" }) |what| {
        if (std.mem.eql(u8, what, "dtype")) for (specs) |*s| {
            if (std.mem.eql(u8, s.name, "layers.0.attn.wq_a.scale")) s.shape = &.{ 40, 160 };
            if (std.mem.eql(u8, s.name, "layers.0.attn.wkv.weight")) s.dtype = "F16";
        };
        // an aligned Engram wkv of the wrong size: refused by its geometry, though the name list gives no shape
        if (std.mem.eql(u8, what, "engram")) for (specs) |*s| {
            if (std.mem.eql(u8, s.name, "layers.0.attn.wkv.weight")) s.dtype = "F8_E4M3";
            if (std.mem.eql(u8, s.name, "layers.1.engram.wkv.weight")) s.shape = &.{ 64, 64 };
            if (std.mem.eql(u8, s.name, "layers.1.engram.wkv.scale")) s.shape = &.{ 2, 2 };
        };
        const header = try pk.headerOnly(a, specs);
        defer a.free(header);
        var pack = pk.Pack.init(a);
        defer pack.deinit();
        try pack.addHeader("model.safetensors", header[8..], std.math.maxInt(u64));
        const want: anyerror = if (std.mem.eql(u8, what, "scale")) error.BadScales else if (std.mem.eql(u8, what, "dtype")) error.UnsupportedProjection else error.UnexpectedShape;
        const block: u32 = if (std.mem.eql(u8, what, "engram")) 1 else 0;
        try testing.expectError(want, plan.build(a, &c, &pack, .{ .rank = 0, .world = 4, .blocks = &.{block} }));
    }
}
