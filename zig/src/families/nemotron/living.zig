//! Saved lessons as shards beside the model's, a file a lesson (its blocks and router); both backends pass in `core`.
const std = @import("std");
const platform = @import("sys");
const Config = @import("config.zig").Config;
const shape = @import("learn_shape.zig");

const prefix = "living-weights-"; // lesson k's file: living-weights-0000k.safetensors, counted from 1
const legacy = "living-weights.safetensors"; // every lesson in one file, up to version 3
const format = "tensorfold living weights";
const version = "4"; // 4: a file a lesson (3 every lesson in one file, 2 counted layers, 1 gated every row)

/// One layer's blocks as a backend holds them: a [rank, in], b [rank, out] and tau (each block's router offset).
pub const Layer = struct { layer: usize, in: usize, out: usize, a: []const f32, b: []const f32, tau: []const f32 };

/// Lesson k's file name (k from 0), in `buf`.
pub fn fileName(buf: []u8, k: usize) ![]const u8 {
    return std.fmt.bufPrint(buf, prefix ++ "{d:0>5}.safetensors", .{k + 1});
}

pub fn Io(comptime core: type) type {
    return struct {
        /// Writes lessons `first`..`last` a file each, removes any past `last`, skips idle layers; the bytes written.
        pub fn write(gpa: std.mem.Allocator, dir: []const u8, layers: []const Layer, need: []const f32, first: usize, last: usize) !usize {
            var bytes: usize = 0;
            for (first..last) |k| bytes += try writeOne(gpa, dir, layers, need[k], k);
            var buf: [64]u8 = undefined;
            var k = last;
            while (true) : (k += 1) {
                const path = try std.fs.path.joinZ(gpa, &.{ dir, try fileName(&buf, k) });
                defer gpa.free(path);
                if (platform.c.unlink(path) != 0) break;
            }
            return bytes;
        }

        fn writeOne(gpa: std.mem.Allocator, dir: []const u8, layers: []const Layer, need: f32, k: usize) !usize {
            var arena_state = std.heap.ArenaAllocator.init(gpa);
            defer arena_state.deinit();
            const arena = arena_state.allocator();
            const rows = shape.block;
            var tensors: std.ArrayList(core.shard_edit.New) = .empty;
            var bytes: usize = 0;
            for (layers) |l| {
                const a = l.a[k * rows * l.in ..][0 .. rows * l.in];
                const b = l.b[k * rows * l.out ..][0 .. rows * l.out];
                // the block's first rank is its router's direction, zero where it reads nothing
                if (std.mem.allEqual(f32, a[0..l.in], 0) and std.mem.allEqual(f32, b, 0)) continue;
                const parts = [_]struct { leaf: []const u8, data: []const f32, dims: []const usize }{
                    .{ .leaf = "a", .data = a, .dims = try arena.dupe(usize, &.{ rows, l.in }) },
                    .{ .leaf = "b", .data = b, .dims = try arena.dupe(usize, &.{ rows, l.out }) },
                    .{ .leaf = "tau", .data = l.tau[k..][0..1], .dims = try arena.dupe(usize, &.{1}) },
                };
                for (parts) |p| {
                    try tensors.append(arena, .{ .name = try std.fmt.allocPrint(arena, "layers.{d}.{s}", .{ l.layer, p.leaf }), .dtype = .f32, .shape = p.dims, .bytes = std.mem.sliceAsBytes(p.data) });
                    bytes += p.data.len * 4;
                }
            }
            const one = try arena.dupe(f32, &.{need});
            try tensors.append(arena, .{ .name = "need", .dtype = .f32, .shape = try arena.dupe(usize, &.{1}), .bytes = std.mem.sliceAsBytes(one) });
            bytes += 4;
            var buf: [64]u8 = undefined;
            const path = try std.fs.path.join(arena, &.{ dir, try fileName(&buf, k) });
            try core.shard_edit.create(gpa, path, &meta, tensors.items);
            return bytes;
        }

        /// Lessons `sites.saved` to `blocks` written beside the shards, any past `blocks` removed; the bytes written.
        pub fn save(gpa: std.mem.Allocator, dir: []const u8, sites: anytype, blocks: usize) !usize {
            const layers = try gpa.alloc(Layer, sites.list.len);
            defer gpa.free(layers);
            const ranks = sites.cap * shape.block;
            for (sites.list, layers) |*s, *l| l.* = .{ .layer = s.layer, .in = s.in, .out = s.out, .a = s.a.slice(f32, ranks * s.in), .b = s.b.slice(f32, ranks * s.out), .tau = s.tau.slice(f32, sites.cap) };
            const bytes = try write(gpa, dir, layers, sites.needs(), @min(sites.saved, blocks), blocks);
            sites.saved = blocks;
            return bytes;
        }

        /// Each saved lesson into new `sites` (either backend, room for all), a block each in order; the lessons read.
        pub fn fill(saved: *const Saved, c: Config, sites: anytype) !usize {
            for (saved.lessons, 0..) |*lesson, k| {
                for (sites.list) |*site| {
                    const got = (try lesson.layer(c, site.layer)) orelse continue;
                    @memcpy(site.a.slice(f32, sites.cap * shape.block * site.in)[k * shape.block * site.in ..][0..got.a.len], got.a);
                    @memcpy(site.b.slice(f32, sites.cap * shape.block * site.out)[k * shape.block * site.out ..][0..got.b.len], got.b);
                    site.tau.slice(f32, sites.cap)[k] = got.tau;
                }
                sites.needs()[k] = try lesson.need();
            }
            sites.rank = saved.lessons.len * shape.block;
            sites.saved = saved.lessons.len;
            return saved.lessons.len;
        }

        /// One saved lesson's file, mapped.
        pub const Lesson = struct {
            file: core.safetensors.File,

            /// The score its router needs at a decision row to act on the reply.
            pub fn need(s: *const Lesson) !f32 {
                const t = s.file.get("need") orelse return error.BadLivingWeights;
                if (!t.is(.f32, &.{1})) return error.BadLivingWeights;
                return floats(t.bytes)[0];
            }

            /// Its block at layer i (a, b and the router's offset there), or null when the lesson leaves that layer be.
            pub fn layer(s: *const Lesson, c: Config, i: usize) !?struct { a: []const f32, b: []const f32, tau: f32 } {
                var buf: [48]u8 = undefined;
                const a = s.file.get(try std.fmt.bufPrint(&buf, "layers.{d}.a", .{i})) orelse return null;
                const b = s.file.get(try std.fmt.bufPrint(&buf, "layers.{d}.b", .{i})) orelse return error.BadLivingWeights;
                const tau = s.file.get(try std.fmt.bufPrint(&buf, "layers.{d}.tau", .{i})) orelse return error.BadLivingWeights;
                if (!a.is(.f32, &.{ shape.block, shape.inWidth(c, i) }) or !b.is(.f32, &.{ shape.block, c.hidden }) or !tau.is(.f32, &.{1})) return error.BadLivingWeights;
                return .{ .a = floats(a.bytes), .b = floats(b.bytes), .tau = floats(tau.bytes)[0] };
            }
        };

        /// The model's saved lessons in order.
        pub const Saved = struct {
            gpa: std.mem.Allocator,
            lessons: []Lesson,

            pub fn close(s: *Saved, io: std.Io) void {
                for (s.lessons) |*l| l.file.close(io);
                s.gpa.free(s.lessons);
            }
        };

        /// Lesson files from the first until one is missing, or null; an old one-file save is reported, not read.
        pub fn open(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !?Saved {
            const old = try std.fs.path.joinZ(gpa, &.{ dir, legacy });
            defer gpa.free(old);
            if (platform.c.access(old, platform.c.F_OK) == 0) std.log.warn("slide: {s} holds lessons saved by an earlier version; they are not read", .{legacy});
            var lessons: std.ArrayList(Lesson) = .empty;
            errdefer {
                for (lessons.items) |*l| l.file.close(io);
                lessons.deinit(gpa);
            }
            var buf: [64]u8 = undefined;
            while (lessons.items.len < shape.max_blocks) {
                const path = try std.fs.path.join(gpa, &.{ dir, try fileName(&buf, lessons.items.len) });
                defer gpa.free(path);
                var file = core.safetensors.File.open(gpa, io, path) catch |e| if (e == error.FileNotFound) break else return e;
                errdefer file.close(io);
                try checkMeta(gpa, file.map.memory[8..file.data]);
                try lessons.append(gpa, .{ .file = file });
            }
            if (lessons.items.len == 0) {
                lessons.deinit(gpa);
                return null;
            }
            return .{ .gpa = gpa, .lessons = try lessons.toOwnedSlice(gpa) };
        }
    };
}

const meta = [_][2][]const u8{
    .{ "format", format },
    .{ "version", version },
    .{ "block", std.fmt.comptimePrint("{d}", .{shape.block}) },
    .{ "scale", std.fmt.comptimePrint("{d}", .{shape.scale}) },
    .{ "a_norm", std.fmt.comptimePrint("{d}", .{shape.a_norm}) },
};

/// The file's header names this format and version, and the block size and scale its block was learned with.
fn checkMeta(gpa: std.mem.Allocator, header: []const u8) !void {
    const parsed = std.json.parseFromSlice(std.json.Value, gpa, header, .{}) catch return error.BadLivingWeights;
    defer parsed.deinit();
    if (parsed.value != .object) return error.BadLivingWeights;
    const got = parsed.value.object.get("__metadata__") orelse return error.BadLivingWeights;
    if (got != .object) return error.BadLivingWeights;
    for (meta) |kv| {
        const v = got.object.get(kv[0]) orelse return error.BadLivingWeights;
        if (v != .string or !std.mem.eql(u8, v.string, kv[1])) return error.LivingWeightsFromAnotherVersion;
    }
}

fn floats(bytes: []const u8) []const f32 {
    return @alignCast(std.mem.bytesAsSlice(f32, bytes));
}
