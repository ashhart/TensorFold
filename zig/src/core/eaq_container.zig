//! EAQ packs (an EXL3 checkpoint plus a manifest): `open` proves a directory is the complete pack its writer published.

const std = @import("std");
const Io = std.Io;
const Sha256 = std.crypto.hash.sha2.Sha256;
const safetensors = @import("safetensors.zig");
const exl3_format = @import("exl3_format.zig");

pub const quant_method = "eaq";
pub const manifest_name = "eaq_manifest.json";
pub const marker_name = "EAQ_COMPLETE.json";
pub const layout = "eaq-container/v0-provisional";
pub const pack_versions = [_][]const u8{ "eaq.pack/0", "eaq.pack/1" };
const max_header = 1 << 28;

pub const Format = enum {
    trellis,
    fp8,
    dense,

    fn parse(text: []const u8) ?Format {
        inline for (.{ .{ "eaq.trellis/0", Format.trellis }, .{ "eaq.fp8/0", Format.fp8 }, .{ "eaq.dense/0", Format.dense } }) |n|
            if (std.mem.eql(u8, text, n[0])) return n[1];
        return null;
    }
};

pub const Error = error{
    NotEaq, // config.json does not say quant_method "eaq"
    BadConfig, // the quantization_config block lacks a usable manifest name or a 64-digit manifest_sha256
    ManifestMissing,
    ManifestHashMismatch, // the manifest differs from the hash config.json states
    BadManifest, // JSON that is not the layout's shape
    MarkerMissing, // EAQ_COMPLETE.json absent: the pack was not published complete
    MarkerMismatch, // the marker vouches for another manifest
    UnsupportedLayout,
    UnsupportedPackVersion,
    FamilyMismatch, // the manifest was written for another model family
    MissingFeature, // the pack requires a runtime feature the caller does not provide
    UnknownDictionary,
    DictionaryMismatch, // a codebook table hash differs from TensorFold's own table
    UnknownFormat,
    ShardMissing,
    ShardOutside, // a shard resolves outside the pack's folder (and, for a Hugging Face snapshot, its blobs)
    UnknownShard, // an entry names a shard the manifest does not list
    ShardSizeMismatch,
    ShardHashMismatch,
    BadShard, // a shard whose safetensors header is truncated, implausible or inconsistent
    DuplicateTensor, // one tensor name in two shards
    TensorMissing, // the manifest lists a tensor no shard holds
    TensorMismatch, // a tensor's dtype or shape differs from the manifest's
    TensorExtra, // a shard holds a tensor the manifest does not list
    TensorMisplaced, // a part's tensor sits in another shard than its entry names
    DuplicateEntry, // two entries share a name
};

/// The name of whatever a refusal is about (a file or tensor), filled when the caller passes one.
pub const Why = struct {
    buf: [256]u8 = undefined,
    len: usize = 0,

    pub fn text(self: *const Why) []const u8 {
        return self.buf[0..self.len];
    }
    fn set(why: ?*Why, comptime fmt: []const u8, args: anytype) void {
        const w = why orelse return;
        w.len = if (std.fmt.bufPrint(&w.buf, fmt, args)) |t| t.len else |_| 0;
    }
};

pub const Options = struct {
    /// the model family being served (the manifest's model.adapter.family must equal it); null accepts any
    family: ?[]const u8 = null,
    /// runtime features the caller reads; a manifest's required_features must all be listed here
    features: []const []const u8 = &.{},
    /// sha256 every shard (the default); false still checks sizes, headers and the inventory
    hash_shards: bool = true,
    threads: usize = 4,
    why: ?*Why = null,
};

pub const Part = struct { name: []const u8, tensor: []const u8, dtype: safetensors.DType, shape: []const usize };

pub const Entry = struct {
    name: []const u8,
    format: Format,
    params: std.json.ObjectMap,
    parts: []const Part,
    shard: []const u8,
};

/// `file` is the manifest's bare name; `path` the real path verified, inside the pack's allowed folders.
pub const Shard = struct { file: []const u8, bytes: u64, sha256: [32]u8, path: []const u8 = "" };

/// A verified pack; every slice lives in `arena`.
pub const Pack = struct {
    arena: std.heap.ArenaAllocator,
    manifest_sha256: [32]u8,
    family: []const u8,
    version: []const u8,
    features: []const []const u8,
    shards: []const Shard,
    entries: []const Entry,
    n_tensors: usize,

    pub fn deinit(self: *Pack) void {
        self.arena.deinit();
        self.* = undefined;
    }

    pub fn entry(self: *const Pack, name: []const u8) ?*const Entry {
        for (self.entries) |*e| if (std.mem.eql(u8, e.name, name)) return e;
        return null;
    }
};

/// Whether config.json's quantization_config (top level or text_config) names quant_method "eaq".
pub fn isEaq(gpa: std.mem.Allocator, io: Io, dir: []const u8) !bool {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const path = try std.fs.path.join(arena.allocator(), &.{ dir, "config.json" });
    const text = Io.Dir.cwd().readFileAlloc(io, path, arena.allocator(), .limited(64 << 20)) catch return false;
    const root = std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), text, .{}) catch return false;
    return configBlock(root) != null;
}

fn configBlock(root: std.json.Value) ?std.json.ObjectMap {
    if (root != .object) return null;
    const texts = [_]?std.json.Value{ root, root.object.get("text_config") };
    for (texts) |src| {
        const s = src orelse continue;
        if (s != .object) continue;
        const b = s.object.get("quantization_config") orelse continue;
        if (b != .object) continue;
        const m = str(b.object.get("quant_method")) orelse continue;
        if (std.ascii.eqlIgnoreCase(m, quant_method)) return b.object;
    }
    return null;
}

/// Refuses, before any weight is read, a pack that is not complete and unmodified (checks run in code order).
pub fn open(gpa: std.mem.Allocator, io: Io, dir: []const u8, o: Options) !Pack {
    var arena = std.heap.ArenaAllocator.init(gpa);
    errdefer arena.deinit();
    const a = arena.allocator();
    const cfg_text = try Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ dir, "config.json" }), a, .limited(64 << 20));
    const cfg = std.json.parseFromSliceLeaky(std.json.Value, a, cfg_text, .{}) catch return error.BadConfig;
    const block = configBlock(cfg) orelse return error.NotEaq;
    const name = str(block.get("manifest")) orelse return error.BadConfig;
    if (!std.mem.eql(u8, name, manifest_name)) {
        Why.set(o.why, "quantization_config.manifest {s}", .{name});
        return error.BadConfig;
    }
    var want: [32]u8 = undefined;
    const hex = str(block.get("manifest_sha256")) orelse return error.BadConfig;
    if (hex.len != 64) return error.BadConfig;
    _ = std.fmt.hexToBytes(&want, hex) catch return error.BadConfig;

    const raw = Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ dir, manifest_name }), a, .limited(1 << 30)) catch return error.ManifestMissing;
    var got: [32]u8 = undefined;
    Sha256.hash(raw, &got, .{});
    if (!std.mem.eql(u8, &got, &want)) return error.ManifestHashMismatch;
    const m = std.json.parseFromSliceLeaky(std.json.Value, a, raw, .{}) catch return error.BadManifest;
    if (m != .object) return error.BadManifest;
    const mo = m.object;

    const shards_v = array(mo.get("shards")) orelse return error.BadManifest;
    try checkMarker(a, io, dir, got, mo, shards_v.len);
    if (!std.mem.eql(u8, str(mo.get("eaq_layout")) orelse "", layout)) return error.UnsupportedLayout;
    const version = str(mo.get("eaq_pack")) orelse pack_versions[0];
    if (!inList(&pack_versions, version)) return error.UnsupportedPackVersion;
    const family = str(obj(obj(mo.get("model"), "adapter"), "family")) orelse return error.BadManifest;
    if (o.family) |f| if (!std.mem.eql(u8, f, family)) {
        Why.set(o.why, "pack is for {s}, serving {s}", .{ family, f });
        return error.FamilyMismatch;
    };
    var features: std.ArrayList([]const u8) = .empty;
    for (array(mo.get("required_features")) orelse &.{}) |v| {
        const f = if (v == .string) v.string else return error.BadManifest;
        if (!inList(o.features, f)) {
            Why.set(o.why, "{s}", .{f});
            return error.MissingFeature;
        }
        try features.append(a, f);
    }
    const dicts = mo.get("dictionaries") orelse return error.BadManifest;
    try checkDictionaries(dicts, o.why);

    const shards = try a.alloc(Shard, shards_v.len);
    for (shards_v, shards) |v, *s| s.* = try shardOf(a, v);
    // each shard is opened once: its header and its hash are read from the same open file
    const roots = try allowedRoots(a, io, dir);
    const files = try a.alloc(Io.File, shards.len);
    var opened: usize = 0;
    defer Io.File.closeMany(io, files[0..opened]);
    for (shards, files) |*s, *f| {
        s.path = try resolve(a, io, dir, s.file, roots, o.why);
        f.* = Io.Dir.cwd().openFile(io, s.path, .{}) catch {
            Why.set(o.why, "{s}", .{s.file});
            return error.ShardMissing;
        };
        opened += 1;
    }
    var tensors: std.StringArrayHashMapUnmanaged(Seen) = .empty;
    const metas = try a.alloc(?[]const u8, shards.len);
    for (shards, files, metas, 0..) |s, f, *meta, i| meta.* = try readShard(a, io, f, s, i, &tensors, o.why);

    const entries = try readEntries(a, mo, metas, shards, o.why);
    if (mo.get("n_entries")) |n| if (n != .integer or n.integer != entries.len) return error.BadManifest;
    try checkUsedDictionaries(entries, dicts, o.why);
    try checkInventory(entries, shards, &tensors, o.why);
    if (o.hash_shards) try hashShards(gpa, io, shards, files, o);
    return .{ .arena = arena, .manifest_sha256 = got, .family = family, .version = version, .features = try features.toOwnedSlice(a), .shards = shards, .entries = entries, .n_tensors = tensors.count() };
}

fn checkMarker(a: std.mem.Allocator, io: Io, dir: []const u8, manifest_sha: [32]u8, mo: std.json.ObjectMap, n_shards: usize) !void {
    const text = Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ dir, marker_name }), a, .limited(1 << 20)) catch return error.MarkerMissing;
    const mk = std.json.parseFromSliceLeaky(std.json.Value, a, text, .{}) catch return error.MarkerMismatch;
    if (mk != .object) return error.MarkerMismatch;
    const hex = std.fmt.bytesToHex(manifest_sha, .lower);
    const same = std.mem.eql(u8, str(mk.object.get("manifest_sha256")) orelse "", &hex) and
        std.mem.eql(u8, str(mk.object.get("eaq_layout")) orelse "", str(mo.get("eaq_layout")) orelse "\x00") and
        intEq(mk.object.get("n_shards"), n_shards) and intEq(mk.object.get("n_entries"), intOf(mo.get("n_entries")) orelse return error.BadManifest);
    if (!same) return error.MarkerMismatch;
}

/// Each dictionary's recorded table hash must equal the hash of TensorFold's own codebook table.
fn checkDictionaries(v: std.json.Value, why: ?*Why) !void {
    if (v != .object) return error.BadManifest;
    const ids = [_]struct { []const u8, exl3_format.Codebook }{
        .{ "exl3.3inst.v1", .inst3 }, .{ "exl3.mcg.v1", .mcg }, .{ "exl3.mul1.v1", .mul1 },
    };
    var it = v.object.iterator();
    next: while (it.next()) |kv| {
        for (ids) |id| if (std.mem.eql(u8, id[0], kv.key_ptr.*)) {
            const hex = str(obj(kv.value_ptr.*, "table_sha256")) orelse return error.BadManifest;
            var d: [32]u8 = undefined;
            codebookSha(id[1], &d);
            const have = std.fmt.bytesToHex(d, .lower);
            if (!std.ascii.eqlIgnoreCase(hex, &have)) {
                Why.set(why, "{s}", .{id[0]});
                return error.DictionaryMismatch;
            }
            continue :next;
        };
        Why.set(why, "{s}", .{kv.key_ptr.*});
        return error.UnknownDictionary;
    }
}

/// sha256 of the fp16 value of all 65536 states, little endian: the table the manifest's `table_sha256` names.
pub fn codebookSha(cb: exl3_format.Codebook, out: *[32]u8) void {
    var h = Sha256.init(.{});
    var chunk: [4096]u8 = undefined;
    var s: usize = 0;
    while (s < 65536) : (s += chunk.len / 2) {
        for (0..chunk.len / 2) |i| {
            const bits: u16 = @bitCast(exl3_format.codebookValue(cb, @intCast(s + i)));
            std.mem.writeInt(u16, chunk[2 * i ..][0..2], bits, .little);
        }
        h.update(&chunk);
    }
    h.final(out);
}

fn shardOf(a: std.mem.Allocator, v: std.json.Value) !Shard {
    const file = str(obj(v, "file")) orelse return error.BadManifest;
    if (!std.mem.eql(u8, std.fs.path.basename(file), file) or !std.mem.endsWith(u8, file, ".safetensors")) return error.BadManifest;
    const bytes = intOf(obj(v, "bytes")) orelse return error.BadManifest;
    var s: Shard = .{ .file = try a.dupe(u8, file), .bytes = bytes, .sha256 = undefined };
    const hex = str(obj(v, "sha256")) orelse return error.BadManifest;
    if (hex.len != 64) return error.BadManifest;
    _ = std.fmt.hexToBytes(&s.sha256, hex) catch return error.BadManifest;
    return s;
}

/// The folders a shard may resolve into: the pack's own, and its repository's blobs when it is a Hugging Face snapshot.
fn allowedRoots(a: std.mem.Allocator, io: Io, dir: []const u8) ![2]?[]const u8 {
    const root = try Io.Dir.cwd().realPathFileAlloc(io, dir, a);
    const snapshots = std.fs.path.dirname(root) orelse return .{ root, null };
    if (!std.mem.eql(u8, std.fs.path.basename(snapshots), "snapshots")) return .{ root, null };
    const repo = std.fs.path.dirname(snapshots) orelse return .{ root, null };
    return .{ root, try std.fs.path.join(a, &.{ repo, "blobs" }) };
}

/// A shard's real path (links followed), refused unless it lies inside one of `roots`.
fn resolve(a: std.mem.Allocator, io: Io, dir: []const u8, file: []const u8, roots: [2]?[]const u8, why: ?*Why) ![]const u8 {
    const real = Io.Dir.cwd().realPathFileAlloc(io, try std.fs.path.join(a, &.{ dir, file }), a) catch {
        Why.set(why, "{s}", .{file});
        return error.ShardMissing;
    };
    for (roots) |r| {
        const root = r orelse continue;
        if (real.len > root.len and std.mem.startsWith(u8, real, root) and std.fs.path.isSep(real[root.len])) return real;
    }
    Why.set(why, "{s}", .{file});
    return error.ShardOutside;
}

const Seen = struct { dtype: safetensors.DType, rank: u8, shape: [safetensors.max_rank]usize, shard: usize, used: bool = false };

/// Checks a shard's size and header, adds its tensors to `seen`, and returns its `eaq.entries` metadata if it has any.
fn readShard(a: std.mem.Allocator, io: Io, file: Io.File, s: Shard, index: usize, seen: *std.StringArrayHashMapUnmanaged(Seen), why: ?*Why) !?[]const u8 {
    const len = try file.length(io);
    if (len != s.bytes) {
        Why.set(why, "{s}", .{s.file});
        return error.ShardSizeMismatch;
    }
    var prefix: [8]u8 = undefined;
    if (len < 8 or try file.readPositionalAll(io, &prefix, 0) != 8) return badShard(why, s.file);
    const n = std.mem.readInt(u64, &prefix, .little);
    if (n > max_header or n > len - 8) return badShard(why, s.file);
    const json = try a.alloc(u8, @intCast(n));
    if (try file.readPositionalAll(io, json, 8) != json.len) return badShard(why, s.file);
    const header = safetensors.parseHeader(a, json, @intCast(len - 8 - n)) catch return badShard(why, s.file);
    for (header.keys(), header.values()) |name, e| {
        const put = try seen.getOrPut(a, name);
        if (put.found_existing) {
            Why.set(why, "{s}", .{name});
            return error.DuplicateTensor;
        }
        put.value_ptr.* = .{ .dtype = e.dtype, .rank = e.rank, .shape = e.shape, .shard = index };
    }
    const root = std.json.parseFromSliceLeaky(std.json.Value, a, json, .{}) catch return badShard(why, s.file);
    return str(obj(obj(root, "__metadata__"), "eaq.entries"));
}

fn badShard(why: ?*Why, file: []const u8) error{BadShard} {
    Why.set(why, "{s}", .{file});
    return error.BadShard;
}

fn readEntries(a: std.mem.Allocator, mo: std.json.ObjectMap, metas: []const ?[]const u8, shard_files: []const Shard, why: ?*Why) ![]const Entry {
    var out: std.ArrayList(Entry) = .empty;
    if (mo.get("entries_inline")) |inl| {
        if (inl != .bool or !inl.bool) return error.BadManifest;
        for (array(mo.get("entries")) orelse return error.BadManifest) |v| {
            const e = try entryOf(a, v, null);
            if (shardIndex(shard_files, e.shard) == null) {
                Why.set(why, "{s}", .{e.shard});
                return error.UnknownShard;
            }
            try out.append(a, e);
        }
    } else {
        for (metas, 0..) |meta, i| {
            const list = std.json.parseFromSliceLeaky(std.json.Value, a, meta orelse return error.BadManifest, .{}) catch return error.BadManifest;
            for (array(list) orelse return error.BadManifest) |v| try out.append(a, try entryOf(a, v, shard_files[i].file));
        }
    }
    var names: std.StringHashMapUnmanaged(void) = .empty;
    for (out.items) |e| if ((try names.getOrPut(a, e.name)).found_existing) {
        Why.set(why, "{s}", .{e.name});
        return error.DuplicateEntry;
    };
    return out.items;
}

fn shardIndex(shards: []const Shard, file: []const u8) ?usize {
    for (shards, 0..) |s, i| if (std.mem.eql(u8, s.file, file)) return i;
    return null;
}

/// A trellis entry's codebook must be one the manifest declares, which checkDictionaries held to TensorFold's tables.
fn checkUsedDictionaries(entries: []const Entry, dicts: std.json.Value, why: ?*Why) !void {
    for (entries) |e| if (e.format == .trellis) {
        const d = str(e.params.get("dictionary")) orelse {
            Why.set(why, "{s}", .{e.name});
            return error.BadManifest;
        };
        if (dicts.object.get(d) == null) {
            Why.set(why, "{s}", .{d});
            return error.UnknownDictionary;
        }
    };
}

fn entryOf(a: std.mem.Allocator, v: std.json.Value, shard: ?[]const u8) !Entry {
    const format = Format.parse(str(obj(v, "format")) orelse return error.BadManifest) orelse return error.UnknownFormat;
    const parts_v = obj(v, "parts") orelse return error.BadManifest;
    if (parts_v != .object) return error.BadManifest;
    var parts: std.ArrayList(Part) = .empty;
    var it = parts_v.object.iterator();
    while (it.next()) |kv| {
        const p = kv.value_ptr.*;
        const dtype = safetensors.DType.parse(str(obj(p, "dtype")) orelse return error.BadManifest) orelse return error.BadManifest;
        const dims = array(obj(p, "shape")) orelse return error.BadManifest;
        const shape = try a.alloc(usize, dims.len);
        for (dims, shape) |d, *x| x.* = std.math.cast(usize, intOf(d) orelse return error.BadManifest) orelse return error.BadManifest;
        try parts.append(a, .{ .name = kv.key_ptr.*, .tensor = str(obj(p, "tensor")) orelse return error.BadManifest, .dtype = dtype, .shape = shape });
    }
    const params = obj(v, "params") orelse return error.BadManifest;
    return .{ .name = str(obj(v, "name")) orelse return error.BadManifest, .format = format, .params = if (params == .object) params.object else return error.BadManifest, .parts = parts.items, .shard = shard orelse str(obj(v, "shard")) orelse "" };
}

/// Every manifest part is in its entry's shard with its dtype and shape, and every shard tensor is a manifest part.
fn checkInventory(entries: []const Entry, shards: []const Shard, seen: *std.StringArrayHashMapUnmanaged(Seen), why: ?*Why) !void {
    for (entries) |e| {
        const at = shardIndex(shards, e.shard) orelse return error.UnknownShard;
        for (e.parts) |p| {
            const t = seen.getPtr(p.tensor) orelse {
                Why.set(why, "{s}", .{p.tensor});
                return error.TensorMissing;
            };
            if (t.dtype != p.dtype or !std.mem.eql(usize, t.shape[0..t.rank], p.shape)) {
                Why.set(why, "{s}", .{p.tensor});
                return error.TensorMismatch;
            }
            if (t.shard != at) {
                Why.set(why, "{s}", .{p.tensor});
                return error.TensorMisplaced;
            }
            t.used = true; // one tensor may serve two entries (tied weights); each must serve at least one
        }
    }
    for (seen.keys(), seen.values()) |name, t| if (!t.used) {
        Why.set(why, "{s}", .{name});
        return error.TensorExtra;
    };
}

const HashJob = struct {
    io: Io,
    shards: []const Shard,
    files: []const Io.File,
    next: std.atomic.Value(usize) = .init(0),
    failed: std.atomic.Value(usize) = .init(std.math.maxInt(usize)),
    oom: std.atomic.Value(bool) = .init(false), // a worker had no buffer: reported apart from any shard

    fn work(self: *HashJob, gpa: std.mem.Allocator) void {
        const buf = gpa.alloc(u8, 4 << 20) catch return self.oom.store(true, .monotonic);
        defer gpa.free(buf);
        while (true) {
            const i = self.next.fetchAdd(1, .monotonic);
            if (i >= self.shards.len) return;
            if (!self.ok(self.files[i], self.shards[i], buf)) self.fail(i);
        }
    }
    fn fail(self: *HashJob, i: usize) void {
        _ = self.failed.fetchMin(i, .monotonic);
    }
    fn ok(self: *HashJob, file: Io.File, s: Shard, buf: []u8) bool {
        var h = Sha256.init(.{});
        var at: u64 = 0;
        while (at < s.bytes) {
            const n = file.readPositionalAll(self.io, buf[0..@intCast(@min(buf.len, s.bytes - at))], at) catch return false;
            if (n == 0) return false;
            h.update(buf[0..n]);
            at += n;
        }
        var d: [32]u8 = undefined;
        h.final(&d);
        return std.mem.eql(u8, &d, &s.sha256);
    }
};

fn hashShards(gpa: std.mem.Allocator, io: Io, shards: []const Shard, files: []const Io.File, o: Options) !void {
    if (shards.len == 0) return;
    var job: HashJob = .{ .io = io, .shards = shards, .files = files };
    const n = @max(1, @min(o.threads, shards.len));
    var threads: [16]std.Thread = undefined;
    var spawned: usize = 0;
    while (spawned + 1 < @min(n, threads.len)) : (spawned += 1)
        threads[spawned] = std.Thread.spawn(.{}, HashJob.work, .{ &job, gpa }) catch break;
    job.work(gpa);
    for (threads[0..spawned]) |t| t.join();
    if (job.oom.load(.monotonic)) return error.OutOfMemory;
    const bad = job.failed.load(.monotonic);
    if (bad != std.math.maxInt(usize)) {
        Why.set(o.why, "{s}", .{shards[bad].file});
        return error.ShardHashMismatch;
    }
}

fn str(v: ?std.json.Value) ?[]const u8 {
    const x = v orelse return null;
    return if (x == .string) x.string else null;
}

fn array(v: ?std.json.Value) ?[]const std.json.Value {
    const x = v orelse return null;
    return if (x == .array) x.array.items else null;
}

fn obj(v: ?std.json.Value, key: []const u8) ?std.json.Value {
    const x = v orelse return null;
    return if (x == .object) x.object.get(key) else null;
}

fn intOf(v: ?std.json.Value) ?u64 {
    const x = v orelse return null;
    return if (x == .integer and x.integer >= 0) @intCast(x.integer) else null;
}

fn intEq(v: ?std.json.Value, want: u64) bool {
    return (intOf(v) orelse return false) == want;
}

fn inList(list: []const []const u8, item: []const u8) bool {
    for (list) |x| if (std.mem.eql(u8, x, item)) return true;
    return false;
}
