//! One tensor table over a checkpoint's safetensors files, with the Python loader's `_Shards` key stripping and widths.

const std = @import("std");
const Io = std.Io;
const st = @import("core").safetensors;
const formats = @import("core").quant;
const config = @import("config.zig");

const Shard = @import("shard.zig").Shard;
pub const Tensor = st.Tensor;
pub const DType = st.DType;
pub const Error = error{ MissingTensor, NoSafetensors } || config.Error || std.mem.Allocator.Error;

const Where = struct { file: usize, stored: []const u8 };

pub const Table = struct {
    gpa: std.mem.Allocator,
    files: []Shard,
    where: std.StringHashMapUnmanaged(Where) = .empty,
    /// A key prefix the table drops (`language_model.`), as `_Shards(strip=)`.
    strip: []const u8,
    quant: formats.Config,
    /// A bf16 chat checkpoint's projections read as fp32 dense (row-invariant kernels), not as rocBLAS BF16.
    widen: bool = false,

    /// Maps `paths` (later files win a repeated name, as the Python dict does).
    pub fn open(gpa: std.mem.Allocator, io: Io, paths: []const []const u8, strip: []const u8, quant: formats.Config) !Table {
        const files = try gpa.alloc(Shard, paths.len);
        var opened: usize = 0;
        errdefer {
            for (files[0..opened]) |*f| f.close(io);
            gpa.free(files);
        }
        for (paths) |p| {
            files[opened] = try Shard.open(gpa, io, p);
            opened += 1;
        }
        var t: Table = .{ .gpa = gpa, .files = files, .strip = strip, .quant = quant };
        errdefer t.where.deinit(gpa);
        for (files, 0..) |*f, i| for (f.names.keys()) |key| {
            const short = if (strip.len > 0 and std.mem.startsWith(u8, key, strip)) key[strip.len..] else key;
            try t.where.put(gpa, short, .{ .file = i, .stored = key });
        };
        return t;
    }

    pub fn close(t: *Table, io: Io) void {
        for (t.files) |*f| f.close(io);
        t.gpa.free(t.files);
        t.where.deinit(t.gpa);
        t.* = undefined;
    }

    /// Releases the mapped pages of every tensor whose key starts with `key_prefix` (`Shard.release`).
    pub fn release(t: *const Table, key_prefix: []const u8) void {
        var it = t.where.iterator();
        while (it.next()) |kv| {
            if (!std.mem.startsWith(u8, kv.key_ptr.*, key_prefix)) continue;
            const shard = &t.files[kv.value_ptr.file];
            const e = shard.names.get(kv.value_ptr.stored) orelse continue;
            shard.release(e.begin, e.end);
        }
    }

    pub fn has(t: *const Table, key: []const u8) bool {
        return t.where.contains(key);
    }

    /// The tensor under `key`; a missing one is `error.MissingTensor`, logged by name.
    pub fn get(t: *const Table, key: []const u8) Error!Tensor {
        const w = t.where.get(key) orelse {
            std.log.err("checkpoint has no tensor {s}", .{key});
            return error.MissingTensor;
        };
        return t.files[w.file].get(w.stored).?;
    }

    /// Every key of the table (stripped names), in no order.
    pub fn keys(t: *const Table) std.StringHashMapUnmanaged(Where).KeyIterator {
        return t.where.keyIterator();
    }

    /// The tensor's own width when the config names it (`key`, then the stripped prefix back on), else the global one.
    pub fn width(t: *const Table, key: []const u8) config.Error!formats.mlx.Width {
        const q = switch (t.quant) {
            .mlx => |m| m,
            .dense, .bf16, .nvfp4 => return error.UnsupportedQuantization,
        };
        if (q.overrides.get(key)) |w| return w orelse error.UnsupportedQuantization;
        if (t.strip.len > 0) {
            var buf: [512]u8 = undefined;
            const full = std.fmt.bufPrint(&buf, "{s}{s}", .{ t.strip, key }) catch return q.global;
            if (q.overrides.get(full)) |w| return w orelse error.UnsupportedQuantization;
        }
        return q.global;
    }
};

/// The checkpoint's safetensors paths, sorted: the model's own files, and the `mtp*.safetensors` side files.
pub const Files = struct {
    arena: std.heap.ArenaAllocator,
    model: []const []const u8,
    side: []const []const u8,

    pub fn deinit(f: *Files) void {
        f.arena.deinit();
        f.* = undefined;
    }

    /// `root.glob("*.safetensors")`: a file whose name holds "mtp" is not the model's; `mtp*` ones are the side files.
    pub fn list(gpa: std.mem.Allocator, io: Io, dir: []const u8) !Files {
        var arena: std.heap.ArenaAllocator = .init(gpa);
        errdefer arena.deinit();
        const a = arena.allocator();
        var model: std.ArrayList([]const u8) = .empty;
        var side: std.ArrayList([]const u8) = .empty;
        var d = try Io.Dir.cwd().openDir(io, dir, .{ .iterate = true });
        defer d.close(io);
        var it = d.iterate();
        while (try it.next(io)) |e| {
            if (e.kind != .file and e.kind != .sym_link) continue;
            if (!std.mem.endsWith(u8, e.name, ".safetensors")) continue;
            const path = try std.fs.path.join(a, &.{ dir, e.name });
            if (std.mem.startsWith(u8, e.name, "mtp")) try side.append(a, path);
            if (!containsLower(e.name, "mtp")) try model.append(a, path);
        }
        const less = struct {
            fn f(_: void, x: []const u8, y: []const u8) bool {
                return std.mem.order(u8, x, y) == .lt;
            }
        }.f;
        std.mem.sort([]const u8, model.items, {}, less);
        std.mem.sort([]const u8, side.items, {}, less);
        return .{ .arena = arena, .model = model.items, .side = side.items };
    }

    /// The shards of `model.safetensors.index.json` (else `model.safetensors`) holding a text-tower or head tensor.
    pub fn indexed(gpa: std.mem.Allocator, io: Io, dir: []const u8) !Files {
        var arena: std.heap.ArenaAllocator = .init(gpa);
        errdefer arena.deinit();
        const a = arena.allocator();
        const index = try std.fs.path.join(a, &.{ dir, "model.safetensors.index.json" });
        const text = Io.Dir.cwd().readFileAlloc(io, index, a, .limited(1 << 26)) catch |e| switch (e) {
            error.FileNotFound => {
                const one = try a.alloc([]const u8, 1);
                one[0] = try std.fs.path.join(a, &.{ dir, "model.safetensors" });
                return .{ .arena = arena, .model = one, .side = &.{} };
            },
            else => return e,
        };
        const doc = try std.json.parseFromSliceLeaky(std.json.Value, a, text, .{});
        const map = if (doc == .object) doc.object.get("weight_map") else null;
        if (map == null or map.? != .object) return error.NoSafetensors;
        var names: std.StringArrayHashMapUnmanaged(void) = .empty;
        for (map.?.object.keys(), map.?.object.values()) |k, v| {
            if (v != .string or std.mem.indexOfScalar(u8, v.string, '/') != null) return error.NoSafetensors;
            // a shard of only vision tensors is not the backbone's: it need not exist or parse
            if (std.mem.startsWith(u8, k, "model.language_model.") or std.mem.startsWith(u8, k, "lm_head.")) try names.put(a, v.string, {});
        }
        const model = try a.alloc([]const u8, names.count());
        for (model, names.keys()) |*m, n| m.* = try std.fs.path.join(a, &.{ dir, n });
        std.mem.sort([]const u8, model, {}, struct {
            fn f(_: void, x: []const u8, y: []const u8) bool {
                return std.mem.order(u8, x, y) == .lt;
            }
        }.f);
        return .{ .arena = arena, .model = model, .side = &.{} };
    }
};

/// The bytes of a bf16 checkpoint's text tower and head (its index's shards): what loads, not the vision tower.
pub fn textBytes(gpa: std.mem.Allocator, io: Io, dir: []const u8) !u64 {
    var files = try Files.indexed(gpa, io, dir);
    defer files.deinit();
    var total: u64 = 0;
    for (files.model) |path| {
        var shard = try Shard.open(gpa, io, path);
        defer shard.close(io);
        var it = shard.names.iterator();
        while (it.next()) |kv| {
            const key = kv.key_ptr.*;
            if (!std.mem.startsWith(u8, key, "model.language_model.") and !std.mem.eql(u8, key, "lm_head.weight")) continue;
            total += kv.value_ptr.end - kv.value_ptr.begin;
        }
    }
    return total;
}

/// The bytes a ModelOpt checkpoint takes on the device: U8 codes as stored, an E4M3 tensor at 2 bytes a value.
pub fn deviceBytes(gpa: std.mem.Allocator, io: Io, dir: []const u8) !u64 {
    var files = try Files.indexed(gpa, io, dir);
    defer files.deinit();
    var total: u64 = 0;
    for (files.model) |path| {
        var shard = try Shard.open(gpa, io, path);
        defer shard.close(io);
        var it = shard.names.iterator();
        while (it.next()) |kv| {
            const key = kv.key_ptr.*;
            if (!std.mem.startsWith(u8, key, "model.language_model.") and !std.mem.eql(u8, key, "lm_head.weight")) continue;
            const stored: u64 = @intCast(kv.value_ptr.end - kv.value_ptr.begin);
            total += if (kv.value_ptr.dtype == .f8_e4m3) 2 * stored else stored;
        }
    }
    return total;
}
fn containsLower(name: []const u8, needle: []const u8) bool {
    var i: usize = 0;
    while (i + needle.len <= name.len) : (i += 1) {
        if (std.ascii.eqlIgnoreCase(name[i..][0..needle.len], needle)) return true;
    }
    return false;
}

test "mtp in a file name keeps it out of the model's files" {
    try std.testing.expect(containsLower("Model-MTP-4bit.safetensors", "mtp"));
    try std.testing.expect(!containsLower("model-00001-of-00004.safetensors", "mtp"));
}
