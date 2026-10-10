//! Saved lessons in a tensor file beside the shards, each a block at every layer and the router that decides when it
//! acts on a reply; both backends pass in the `core` that reads it.
const std = @import("std");
const Config = @import("config.zig").Config;
const shape = @import("learn_shape.zig");

pub const file_name = "living-weights.safetensors";
const format = "tensorfold living weights";
const version = "2"; // 2: a router a lesson, deciding once a reply (1 gated every row)

/// One layer's blocks as a backend holds them: `ranks` rows of a [rank, in] and b [rank, out], each block's router
/// threshold here (tau, ranks / block of them).
pub const Layer = struct { layer: usize, in: usize, out: usize, a: []const f32, b: []const f32, tau: []const f32 };

pub fn Io(comptime core: type) type {
    return struct {
        /// Writes every layer where a block acts, and the votes each lesson's router needs, as the living-weights file;
        /// its bytes returned. Layers where every router is shut and every output zero are left out.
        pub fn write(gpa: std.mem.Allocator, dir: []const u8, layers: []const Layer, need: []const f32, ranks: usize) !usize {
            var arena_state = std.heap.ArenaAllocator.init(gpa);
            defer arena_state.deinit();
            const arena = arena_state.allocator();
            const blocks = ranks / shape.block;
            var tensors: std.ArrayList(core.shard_edit.New) = .empty;
            var bytes: usize = 0;
            for (layers) |l| {
                const b = l.b[0 .. ranks * l.out];
                const gates = l.tau[0..blocks];
                var acting = false;
                for (gates) |g| acting = acting or g < shape.shut;
                if (!acting and std.mem.allEqual(f32, b, 0)) continue;
                const parts = [_]struct { leaf: []const u8, data: []const f32, dims: []const usize }{
                    .{ .leaf = "a", .data = l.a[0 .. ranks * l.in], .dims = try arena.dupe(usize, &.{ ranks, l.in }) },
                    .{ .leaf = "b", .data = b, .dims = try arena.dupe(usize, &.{ ranks, l.out }) },
                    .{ .leaf = "tau", .data = gates, .dims = try arena.dupe(usize, &.{blocks}) },
                };
                for (parts) |p| {
                    try tensors.append(arena, .{ .name = try std.fmt.allocPrint(arena, "layers.{d}.{s}", .{ l.layer, p.leaf }), .dtype = .f32, .shape = p.dims, .bytes = std.mem.sliceAsBytes(p.data) });
                    bytes += p.data.len * 4;
                }
            }
            try tensors.append(arena, .{ .name = "need", .dtype = .f32, .shape = try arena.dupe(usize, &.{blocks}), .bytes = std.mem.sliceAsBytes(need[0..blocks]) });
            bytes += blocks * 4;
            const meta = [_][2][]const u8{
                .{ "format", format },
                .{ "version", version },
                .{ "block", std.fmt.comptimePrint("{d}", .{shape.block}) },
                .{ "scale", std.fmt.comptimePrint("{d}", .{shape.scale}) },
                .{ "a_norm", std.fmt.comptimePrint("{d}", .{shape.a_norm}) },
            };
            const path = try std.fs.path.join(arena, &.{ dir, file_name });
            try core.shard_edit.create(gpa, path, &meta, tensors.items);
            return bytes;
        }

        /// The model's living-weights file, mapped, or null when it has none.
        pub const Saved = struct {
            file: core.safetensors.File,
            ranks: usize,

            /// The votes each lesson's router needs at a decision row to act on the reply.
            pub fn need(s: *const Saved) ![]const f32 {
                const t = s.file.get("need") orelse return error.BadLivingWeights;
                if (!t.is(.f32, &.{s.ranks / shape.block})) return error.BadLivingWeights;
                return floats(t.bytes);
            }

            /// Layer i's blocks (a, b, tau as written), or null when none acts there.
            pub fn layer(s: *const Saved, c: Config, i: usize) !?struct { a: []const f32, b: []const f32, tau: []const f32 } {
                var buf: [48]u8 = undefined;
                const a = s.file.get(try std.fmt.bufPrint(&buf, "layers.{d}.a", .{i})) orelse return null;
                const b = s.file.get(try std.fmt.bufPrint(&buf, "layers.{d}.b", .{i})) orelse return error.BadLivingWeights;
                const tau = s.file.get(try std.fmt.bufPrint(&buf, "layers.{d}.tau", .{i})) orelse return error.BadLivingWeights;
                const in = shape.inWidth(c, i);
                if (!a.is(.f32, &.{ s.ranks, in }) or !b.is(.f32, &.{ s.ranks, c.hidden }) or !tau.is(.f32, &.{s.ranks / shape.block})) return error.BadLivingWeights;
                return .{ .a = floats(a.bytes), .b = floats(b.bytes), .tau = floats(tau.bytes) };
            }

            pub fn close(s: *Saved, io: std.Io) void {
                s.file.close(io);
            }
        };

        pub fn open(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !?Saved {
            const path = try std.fs.path.join(gpa, &.{ dir, file_name });
            defer gpa.free(path);
            var file = core.safetensors.File.open(gpa, io, path) catch |e| return if (e == error.FileNotFound) null else e;
            errdefer file.close(io);
            try checkMeta(gpa, file.map.memory[8..file.data]);
            var ranks: ?usize = null;
            for (file.names.keys(), file.names.values()) |name, e| {
                if (!std.mem.endsWith(u8, name, ".a")) continue;
                if (ranks) |r| if (r != e.shape[0]) return error.BadLivingWeights;
                ranks = e.shape[0];
            }
            const r = ranks orelse return error.BadLivingWeights;
            if (r == 0 or r % shape.block != 0 or r > shape.max_rank) return error.BadLivingWeights;
            return .{ .file = file, .ranks = r };
        }
    };
}

/// The file's header names this format and version, and the block size and scale its blocks were learned with.
fn checkMeta(gpa: std.mem.Allocator, header: []const u8) !void {
    const parsed = std.json.parseFromSlice(std.json.Value, gpa, header, .{}) catch return error.BadLivingWeights;
    defer parsed.deinit();
    if (parsed.value != .object) return error.BadLivingWeights;
    const meta = parsed.value.object.get("__metadata__") orelse return error.BadLivingWeights;
    if (meta != .object) return error.BadLivingWeights;
    const want = [_][2][]const u8{
        .{ "format", format },
        .{ "version", version },
        .{ "block", std.fmt.comptimePrint("{d}", .{shape.block}) },
        .{ "scale", std.fmt.comptimePrint("{d}", .{shape.scale}) },
        .{ "a_norm", std.fmt.comptimePrint("{d}", .{shape.a_norm}) },
    };
    for (want) |kv| {
        const v = meta.object.get(kv[0]) orelse return error.BadLivingWeights;
        if (v != .string or !std.mem.eql(u8, v.string, kv[1])) return error.LivingWeightsFromAnotherVersion;
    }
}

fn floats(bytes: []const u8) []const f32 {
    return @alignCast(std.mem.bytesAsSlice(f32, bytes));
}
