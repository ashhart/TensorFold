//! SenseNova Loop config rules: the shared case table through the SenseNova reader and Metal qwen27 admission.
const std = @import("std");
const t = std.testing;

pub const checkpoint = @import("src/core/checkpoint.zig");
pub const checkpoint_host = checkpoint;
pub const Checkpoint = checkpoint.Checkpoint;
pub const quant = @import("src/core/quant/quant.zig");
pub const rope_yarn = @import("src/core/rope_yarn.zig");
pub const sensenova_loop = struct {
    pub const config = @import("src/families/sensenova_loop/config.zig");
};

const sn_cfg = sensenova_loop.config;
const qwen27_cfg = @import("src/families/qwen27/config.zig");
const table = @embedFile("tests/fixtures/sensenova_loop_config.json");

const Expect = enum { accept, config, admission };
const Case = struct { name: []const u8, set: ?std.json.Value = null, remove: []const []const u8 = &.{}, expect: Expect };
const Table = struct { base: std.json.Value, cases: []const Case };

/// JSON merge: objects merge key by key; any other value, null included, replaces.
fn merge(a: std.mem.Allocator, into: *std.json.Value, patch: std.json.Value) !void {
    if (into.* != .object or patch != .object) {
        into.* = patch;
        return;
    }
    var it = patch.object.iterator();
    while (it.next()) |entry| {
        if (into.object.getPtr(entry.key_ptr.*)) |slot| try merge(a, slot, entry.value_ptr.*) else try into.object.put(a, entry.key_ptr.*, entry.value_ptr.*);
    }
}

fn remove(root: *std.json.Value, path: []const u8) !void {
    var node = root;
    var it = std.mem.splitScalar(u8, path, '.');
    while (it.next()) |key| {
        if (it.peek() == null) {
            try t.expect(node.object.orderedRemove(key));
        } else node = node.object.getPtr(key) orelse return error.UnknownPath;
    }
}

/// The error classes: loop, layer, and tied contradictions refuse as config; engine limits refuse as admission.
fn outcome(err: anyerror) ?Expect {
    return switch (err) {
        error.BadLoopSettings,
        error.BadMhcConfig,
        error.MhcCrossesLoopBoundary,
        error.BadLayerTypes,
        error.UnsupportedLayerTypes,
        error.BadFullAttentionInterval,
        error.LayerTypesContradictFullAttentionInterval,
        error.MissingFullAttention,
        error.BadTiedHead,
        => .config,
        error.UnsupportedSenseNovaMhc, error.TiedEmbeddingHead => .admission,
        else => null,
    };
}

fn patchedBytes(a: std.mem.Allocator, base: std.json.Value, patch: ?std.json.Value, rem: []const []const u8) ![]u8 {
    var doc = try std.json.parseFromSliceLeaky(std.json.Value, a, try std.json.Stringify.valueAlloc(a, base, .{}), .{});
    if (patch) |p| try merge(a, &doc, p);
    for (rem) |path| try remove(&doc, path);
    return std.json.Stringify.valueAlloc(a, doc, .{});
}

test "SenseNova loop config table: every case reaches its outcome" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const parsed = try std.json.parseFromSliceLeaky(Table, a, table, .{});
    try t.expect(parsed.cases.len > 0);
    for (parsed.cases) |c| {
        const bytes = try patchedBytes(a, parsed.base, c.set, c.remove);
        if (qwen27_cfg.parse(t.allocator, bytes, null)) |cfg| {
            t.expectEqual(Expect.accept, c.expect) catch |e| {
                std.debug.print("case '{s}' was accepted\n", .{c.name});
                return e;
            };
            if (std.mem.eql(u8, c.name, "mhc range of one layer")) {
                try t.expectEqual(sn_cfg.MhcRole.both, cfg.mhc_roles[4]);
                try t.expectEqual(sn_cfg.MhcRole.both, cfg.mhc_roles[28]);
                try t.expectEqual(sn_cfg.MhcRole.none, cfg.mhc_roles[5]);
            }
            if (std.mem.eql(u8, c.name, "mhc range of one layer at the loop end")) {
                try t.expectEqual(sn_cfg.MhcRole.both, cfg.mhc_roles[27]);
                try t.expectEqual(sn_cfg.MhcRole.both, cfg.mhc_roles[51]);
                try t.expectEqual(sn_cfg.MhcRole.none, cfg.mhc_roles[26]);
            }
        } else |err| {
            t.expectEqual(c.expect, outcome(err) orelse {
                std.debug.print("case '{s}' failed with {s}\n", .{ c.name, @errorName(err) });
                return err;
            }) catch |e| {
                std.debug.print("case '{s}' failed with {s}\n", .{ c.name, @errorName(err) });
                return e;
            };
        }
        // The SenseNova reader alone accepts both accept and admission cases and refuses only config cases.
        if (sn_cfg.Config.parse(t.allocator, bytes)) |sn| {
            t.expect(c.expect != .config) catch |e| {
                std.debug.print("case '{s}' passed sensenova reader unexpectedly\n", .{c.name});
                return e;
            };
            if (std.mem.eql(u8, c.name, "mhc range of one layer")) try t.expectEqual(sn_cfg.MhcRole.both, sn.mhcRole(4));
        } else |sn_err| {
            try t.expectEqual(Expect.config, c.expect);
            try t.expectEqual(Expect.config, outcome(sn_err) orelse return sn_err);
        }
    }
}

test "SenseNova loop config table: pre_scale FP32 overflow and root sensenova_loop_text" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const parsed = try std.json.parseFromSliceLeaky(Table, a, table, .{});

    const over = try patchedBytes(a, parsed.base, (try std.json.parseFromSliceLeaky(std.json.Value, a, "{\"mhc\":{\"pre_scale\":1e39}}", .{})), &.{});
    const sn_over = try sn_cfg.Config.parse(t.allocator, over);
    try t.expect(sn_over.mhc.?.pre_scale > std.math.floatMax(f32));
    try t.expectError(error.UnsupportedSenseNovaMhc, qwen27_cfg.parse(t.allocator, over, null));

    const root_text = try patchedBytes(a, parsed.base, (try std.json.parseFromSliceLeaky(std.json.Value, a, "{\"model_type\":\"sensenova_loop_text\"}", .{})), &.{});
    try t.expectError(error.UnsupportedModel, sn_cfg.Config.parse(t.allocator, root_text));
    try t.expectError(error.NotQwenDense, qwen27_cfg.parse(t.allocator, root_text, null));

    const inner_text = try patchedBytes(a, parsed.base, (try std.json.parseFromSliceLeaky(std.json.Value, a, "{\"text_config\":{\"model_type\":\"sensenova_loop_text\"}}", .{})), &.{});
    _ = try sn_cfg.Config.parse(t.allocator, inner_text);
    _ = try qwen27_cfg.parse(t.allocator, inner_text, null);
}
