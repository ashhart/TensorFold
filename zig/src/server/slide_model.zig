//! The served model as the Living Weights view draws it: layers, pockets and shards, named as its family names them.
const std = @import("std");
const json = @import("json");
const families = @import("living").families;
const Family = @import("living").Family;
const Allocator = std.mem.Allocator;
const Io = std.Io;

/// A shard file: its size and identity on disk, the layers whose shared expert it holds, and those tensors' bytes.
pub const Shard = struct { name: []const u8, bytes: u64, mtime: i96, inode: u64, layers: []const u16, expert_bytes: u64 };

/// The model's layers, its pockets and its shards, as the folder holds them.
pub const Model = struct {
    kinds: []const u8, // one letter a layer: M mamba, E MoE, * attention, - MLP, ? other
    pocket: usize = 0, // the pocket's width in the shards (0: none written yet)
    moe: []const u16 = &.{}, // the MoE layers
    used: []const u16 = &.{}, // the neurons each of them holds in the shards
    shards: []const Shard = &.{},
};

/// The family a config names by its model_type; Nemotron's for one that names none (as configs before families did).
fn familyOf(o: std.json.ObjectMap) ?Family {
    const v = o.get("model_type") orelse return families.nemotron_h;
    return if (v == .string) families.byModelType(v.string) else null;
}

/// The folder's config, shard headers and stats; the pocket's rows read where the shards hold one.
pub fn scan(a: Allocator, io: Io, dir: []const u8) !Model {
    const config = try readJson(a, io, try std.fs.path.join(a, &.{ dir, "config.json" }));
    if (config != .object) return error.BadConfig;
    const f = familyOf(config.object) orelse return error.UnknownFamily;
    const keys = families.scoped(f, config.object) orelse return error.BadConfig; // text_config, or the top level
    const kinds = try families.kinds(a, f, keys);
    const pocket = intField(keys, f.keys.pocket) orelse 0;
    var moe: std.ArrayList(u16) = .empty;
    for (kinds, 0..) |k, l| if (k == 'E') try moe.append(a, @intCast(l));
    const used = try a.alloc(u16, moe.items.len);
    @memset(used, 0);
    var shards: std.ArrayList(Shard) = .empty;
    var d = try Io.Dir.cwd().openDir(io, dir, .{ .iterate = true });
    defer d.close(io);
    var it = d.iterate();
    while (try it.next(io)) |e| {
        if (!std.mem.endsWith(u8, e.name, ".safetensors")) continue;
        const name = try a.dupe(u8, e.name);
        const path = try std.fs.path.join(a, &.{ dir, name });
        const stat = try Io.Dir.cwd().statFile(io, path, .{});
        var shard: Shard = .{ .name = name, .bytes = stat.size, .mtime = stat.mtime.toNanoseconds(), .inode = @intCast(stat.inode), .layers = &.{}, .expert_bytes = 0 };
        readShard(a, io, path, f, &shard, pocket, moe.items, used) catch {};
        try shards.append(a, shard);
    }
    std.mem.sort(Shard, shards.items, {}, byName);
    return .{ .kinds = kinds, .pocket = pocket, .moe = moe.items, .used = used, .shards = shards.items };
}

fn byName(_: void, x: Shard, y: Shard) bool {
    return std.mem.lessThan(u8, x.name, y.name);
}

fn intField(o: std.json.ObjectMap, key: []const u8) ?usize {
    const v = o.get(key) orelse return null;
    return if (v == .integer and v.integer >= 0) @intCast(v.integer) else null;
}

/// The shard's header: the layers whose shared expert it holds and their bytes, and the pocket's neurons in use there.
fn readShard(a: Allocator, io: Io, path: []const u8, f: Family, shard: *Shard, pocket: usize, moe: []const u16, used: []u16) !void {
    var file = try Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    var len: [8]u8 = undefined;
    if (try file.readPositionalAll(io, &len, 0) != 8) return error.BadSafetensors;
    const n = std.mem.readInt(u64, &len, .little);
    if (shard.bytes < 8 or n > shard.bytes - 8 or n > 1 << 27) return error.BadSafetensors;
    const text = try a.alloc(u8, @intCast(n));
    if (try file.readPositionalAll(io, text, 8) != text.len) return error.BadSafetensors;
    const header = try std.json.parseFromSliceLeaky(std.json.Value, a, text, .{});
    if (header != .object) return error.BadSafetensors;
    var layers: std.ArrayList(u16) = .empty;
    var it = header.object.iterator();
    while (it.next()) |kv| {
        const name = kv.key_ptr.*;
        const expert = f.names.expert;
        const at = std.mem.indexOf(u8, name, expert) orelse continue; // an MTP head's expert's bytes count too
        const range = offsets(kv.value_ptr.*) orelse continue;
        shard.expert_bytes += range[1] - range[0];
        const layer: u16 = @intCast(f.expertLayer(name) orelse continue);
        if (std.mem.indexOfScalar(u16, layers.items, layer) == null) try layers.append(a, layer);
        if (pocket == 0) continue;
        // a row a neuron: the up projection's scales or biases, zero once a neuron is free (a SwiGLU gate row may stay)
        const leaf = name[at + expert.len ..];
        const proj = f.upInput();
        if (!std.mem.startsWith(u8, leaf, proj) or leaf.len <= proj.len or leaf[proj.len] != '.') continue;
        const part = leaf[proj.len + 1 ..];
        if (!std.mem.eql(u8, part, "scales") and !std.mem.eql(u8, part, "biases")) continue;
        const slot = std.mem.indexOfScalar(u16, moe, layer) orelse continue;
        used[slot] = @max(used[slot], try inUse(a, io, file, 8 + n + range[0], kv.value_ptr.*, pocket));
    }
    std.mem.sort(u16, layers.items, {}, std.sort.asc(u16));
    shard.layers = layers.items;
}

fn offsets(entry: std.json.Value) ?[2]u64 {
    if (entry != .object) return null;
    const o = entry.object.get("data_offsets") orelse return null;
    if (o != .array or o.array.items.len != 2) return null;
    const b = o.array.items[0];
    const e = o.array.items[1];
    if (b != .integer or e != .integer or b.integer < 0 or e.integer < b.integer) return null;
    return .{ @intCast(b.integer), @intCast(e.integer) };
}

/// Neurons in use among a per-row tensor's last `pocket` rows (bf16 scales or biases): up to the last not all zero.
fn inUse(a: Allocator, io: Io, file: Io.File, at: u64, entry: std.json.Value, pocket: usize) !u16 {
    const shape = entry.object.get("shape") orelse return 0;
    if (shape != .array or shape.array.items.len != 2) return 0;
    const rows = shape.array.items[0];
    const cols = shape.array.items[1];
    if (rows != .integer or cols != .integer or cols.integer < 1 or rows.integer < @as(i64, @intCast(pocket))) return 0;
    const row: usize = @as(usize, @intCast(cols.integer)) * 2;
    const own: u64 = @intCast(rows.integer - @as(i64, @intCast(pocket)));
    const bytes = try a.alloc(u8, pocket * row);
    if (try file.readPositionalAll(io, bytes, at + own * row) != bytes.len) return 0;
    var last: usize = 0;
    for (0..pocket) |r| if (!std.mem.allEqual(u8, bytes[r * row ..][0..row], 0)) {
        last = r + 1;
    };
    return @intCast(last);
}

fn readJson(a: Allocator, io: Io, path: []const u8) !std.json.Value {
    const bytes = try Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 24));
    return std.json.parseFromSliceLeaky(std.json.Value, a, bytes, .{});
}

/// The shards a save changed (size, inode or time moved, or new) and their shared experts' bytes.
pub fn written(a: Allocator, before: []const Shard, after: []const Shard) !struct { files: []const []const u8, bytes: u64 } {
    var files: std.ArrayList([]const u8) = .empty;
    var bytes: u64 = 0;
    for (after) |s| {
        const old = for (before) |b| {
            if (std.mem.eql(u8, b.name, s.name)) break b;
        } else null;
        if (old) |b| if (b.bytes == s.bytes and b.mtime == s.mtime and b.inode == s.inode) continue;
        try files.append(a, s.name);
        bytes += s.expert_bytes;
    }
    return .{ .files = files.items, .bytes = bytes };
}

/// {"layers", "kinds", "pocket": {"neurons", "layers", "used"}, "shards": [{"name", "bytes", "pocket", "layers"}]}.
pub fn value(a: Allocator, m: Model) !json.Value {
    const o = try json.newObject(a);
    try o.put(a, "layers", try json.intValue(a, m.kinds.len));
    try o.put(a, "kinds", .{ .string = m.kinds });
    const p = try json.newObject(a);
    try p.put(a, "neurons", try json.intValue(a, m.pocket));
    try p.put(a, "layers", try ints(a, m.moe));
    try p.put(a, "used", try ints(a, m.used));
    try o.put(a, "pocket", .{ .object = p });
    const shards = try a.alloc(json.Value, m.shards.len);
    for (m.shards, shards) |s, *out| {
        const x = try json.newObject(a);
        try x.put(a, "name", .{ .string = s.name });
        try x.put(a, "bytes", try json.intValue(a, s.bytes));
        try x.put(a, "pocket", .{ .bool = m.pocket > 0 and s.expert_bytes > 0 });
        try x.put(a, "layers", try ints(a, s.layers));
        out.* = .{ .object = x };
    }
    try o.put(a, "shards", .{ .array = shards });
    return .{ .object = o };
}

pub fn ints(a: Allocator, xs: anytype) !json.Value {
    const out = try a.alloc(json.Value, xs.len);
    for (xs, out) |x, *v| v.* = try json.intValue(a, x);
    return .{ .array = out };
}

/// A safetensors file of the named tensors, each its dtype, shape and bytes, in order.
fn writeShard(io: Io, dir: Io.Dir, name: []const u8, tensors: []const struct { []const u8, []const u8, [2]usize, []const u8 }) !void {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var head: std.ArrayList(u8) = .empty;
    var data: std.ArrayList(u8) = .empty;
    try head.append(a, '{');
    for (tensors, 0..) |t, i| {
        try head.print(a, "{s}\"{s}\": {{\"dtype\": \"{s}\", \"shape\": [{d}, {d}], \"data_offsets\": [{d}, {d}]}}", .{ if (i > 0) ", " else "", t[0], t[1], t[2][0], t[2][1], data.items.len, data.items.len + t[3].len });
        try data.appendSlice(a, t[3]);
    }
    try head.append(a, '}');
    var len: [8]u8 = undefined;
    std.mem.writeInt(u64, &len, head.items.len, .little);
    try dir.writeFile(io, .{ .sub_path = name, .data = try std.mem.concat(a, u8, &.{ &len, head.items, data.items }) });
}

test "a folder's layers, pockets in use and shards; a save's files and bytes" {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = "{\"layers_block_type\": [\"mamba\", \"moe\", \"attention\", \"moe\"], \"tensorfold_learned_neurons\": 64}" });
    // 64 own rows and 64 pocket rows of one bf16 scale: pocket rows 0 and 2 in use, so three neurons count
    var scales: [128 * 2]u8 = @splat(0);
    scales[(64 + 0) * 2 + 1] = 0x3f;
    scales[(64 + 2) * 2 + 1] = 0x3f;
    const zeros: [128 * 2]u8 = @splat(0);
    try writeShard(io, tmp.dir, "model-00001-of-00002.safetensors", &.{
        .{ "backbone.layers.1.mixer.shared_experts.up_proj.scales", "BF16", .{ 128, 1 }, &scales },
        .{ "backbone.layers.1.mixer.shared_experts.up_proj.biases", "BF16", .{ 128, 1 }, &zeros },
        .{ "backbone.layers.0.mixer.in_proj.scales", "BF16", .{ 128, 1 }, &zeros },
    });
    try writeShard(io, tmp.dir, "model-00002-of-00002.safetensors", &.{
        .{ "backbone.layers.3.mixer.shared_experts.up_proj.scales", "BF16", .{ 128, 1 }, &zeros },
    });
    const dir = try std.fs.path.join(a, &.{ ".zig-cache", "tmp", &tmp.sub_path });
    const m = try scan(a, io, dir);
    try std.testing.expectEqualStrings("ME*E", m.kinds);
    try std.testing.expectEqual(@as(usize, 64), m.pocket);
    try std.testing.expectEqualSlices(u16, &.{ 1, 3 }, m.moe);
    try std.testing.expectEqualSlices(u16, &.{ 3, 0 }, m.used);
    try std.testing.expectEqual(@as(usize, 2), m.shards.len);
    try std.testing.expectEqualSlices(u16, &.{1}, m.shards[0].layers);
    try std.testing.expectEqual(@as(u64, 512), m.shards[0].expert_bytes);
    // the second shard rewritten: it alone is a save's, with its shared expert's bytes
    try writeShard(io, tmp.dir, "model-00002-of-00002.safetensors", &.{
        .{ "backbone.layers.3.mixer.shared_experts.up_proj.scales", "BF16", .{ 128, 1 }, &scales },
        .{ "backbone.layers.3.mixer.shared_experts.up_proj.biases", "BF16", .{ 128, 1 }, &zeros },
    });
    const again = try scan(a, io, dir);
    try std.testing.expectEqualSlices(u16, &.{ 3, 3 }, again.used);
    const w = try written(a, m.shards, again.shards);
    try std.testing.expectEqual(@as(usize, 1), w.files.len);
    try std.testing.expectEqualStrings("model-00002-of-00002.safetensors", w.files[0]);
    try std.testing.expectEqual(@as(u64, 512), w.bytes);
    const text = try json.stringify(a, try value(a, again), .{ .compact = true });
    try std.testing.expect(std.mem.indexOf(u8, text, "\"pocket\":{\"neurons\":64,\"layers\":[1,3],\"used\":[3,3]}") != null);
}

test "a SwiGLU pocket counts its neurons by their up rows: one retired with its gate row left is free" {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const f: Family = .{
        .model_type = "swiglu_scan_test",
        .names = .{ .layers = "language_model.model.layers.", .expert = "mlp.shared_expert.", .inputs = &.{ "gate_proj", "up_proj" } },
        .neuron = .{ .act = .swiglu, .margin = 0.5, .sharp = 16 },
        .quant = .{ .bits = 4, .group = 32 },
        .keys = .{ .width = "shared_expert_intermediate_size" },
        .experts = .every,
    };
    // 32 own rows and 32 pocket rows: gate rows 0 to 2 set, up rows 0 and 1 only (neuron 2 retired, its gate kept)
    var gate: [64 * 2]u8 = @splat(0);
    var up: [64 * 2]u8 = @splat(0);
    for ([_]usize{ 0, 1, 2 }) |r| gate[(32 + r) * 2 + 1] = 0x3f;
    for ([_]usize{ 0, 1 }) |r| up[(32 + r) * 2 + 1] = 0x3f;
    try writeShard(io, tmp.dir, "model.safetensors", &.{
        .{ "language_model.model.layers.0.mlp.shared_expert.gate_proj.scales", "BF16", .{ 64, 1 }, &gate },
        .{ "language_model.model.layers.0.mlp.shared_expert.up_proj.scales", "BF16", .{ 64, 1 }, &up },
    });
    const path = try std.fs.path.join(a, &.{ ".zig-cache", "tmp", &tmp.sub_path, "model.safetensors" });
    var shard: Shard = .{ .name = "model.safetensors", .bytes = (try Io.Dir.cwd().statFile(io, path, .{})).size, .mtime = 0, .inode = 0, .layers = &.{}, .expert_bytes = 0 };
    var used = [_]u16{0};
    try readShard(a, io, path, f, &shard, 32, &.{0}, &used);
    try std.testing.expectEqual(@as(u16, 2), used[0]);
    try std.testing.expectEqualStrings("up_proj", f.upInput());
}
