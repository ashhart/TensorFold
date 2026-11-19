//! SenseNova Loop config rules: the shared case table through the loop reader and CUDA's geometry admission.
const std = @import("std");
const geometry = @import("src/families/qwen3_5/cuda_geometry.zig");
const loop_config = @import("src/families/qwen3_5/loop_config.zig");
const t = std.testing;

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

/// The error classes: the loop and layer readers refuse as config, the engine's limits as admission.
fn outcome(err: anyerror) ?Expect {
    return switch (err) {
        error.BadLoopConfig, error.BadLayerTypes, error.UnsupportedLayerTypes, error.BadFullAttentionInterval, error.LayerTypesContradictFullAttentionInterval, error.BadTiedHead => .config,
        error.UnsupportedSenseNovaMhc, error.UnsupportedQwenConfig => .admission,
        else => null,
    };
}

test "SenseNova loop config table: every case reaches its outcome" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const parsed = try std.json.parseFromSliceLeaky(Table, a, table, .{});
    try t.expect(parsed.cases.len > 0);
    for (parsed.cases) |c| {
        var doc = try std.json.parseFromSliceLeaky(std.json.Value, a, try std.json.Stringify.valueAlloc(a, parsed.base, .{}), .{});
        if (c.set) |patch| try merge(a, &doc, patch);
        for (c.remove) |path| try remove(&doc, path);
        const bytes = try std.json.Stringify.valueAlloc(a, doc, .{});
        const result = geometry.parse(t.allocator, bytes);
        if (result) |_| {
            t.expectEqual(Expect.accept, c.expect) catch |e| {
                std.debug.print("case '{s}' was accepted\n", .{c.name});
                return e;
            };
        } else |err| {
            t.expectEqual(c.expect, outcome(err) orelse {
                std.debug.print("case '{s}' failed with {s}\n", .{ c.name, @errorName(err) });
                return err;
            }) catch |e| {
                std.debug.print("case '{s}' failed with {s}\n", .{ c.name, @errorName(err) });
                return e;
            };
            // the loop reader alone agrees: it refuses exactly the config cases
            const root = (try std.json.parseFromSliceLeaky(std.json.Value, a, bytes, .{})).object;
            const layers: usize = @intCast(root.get("text_config").?.object.get("num_hidden_layers").?.integer);
            if (loop_config.parse(root, layers)) |_| {
                try t.expect(err != error.BadLoopConfig);
            } else |_| try t.expectEqual(Expect.config, c.expect);
        }
    }
}
