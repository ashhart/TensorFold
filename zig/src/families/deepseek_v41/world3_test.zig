//! Rank files on a synthetic pack: the even worlds pinned to main, three ranks checked against the stored bytes.

const std = @import("std");
const testing = std.testing;
const admission = @import("block_admission.zig");
const block = @import("block.zig");
const config = @import("config.zig");
const Config = config.Config;
const exl3 = @import("core").exl3_layout;
const named = @import("named.zig");
const Pack = @import("pack.zig").Pack;
const partition = @import("partition.zig");
const plan = @import("plan.zig");
const prepared = @import("prepared.zig");
const Io = std.Io;
const Sha256 = std.crypto.hash.sha2.Sha256;

/// tiny() with the default's eight output groups and whole blocks for every world here.
fn eightGroups() Config {
    var c = config.tiny();
    c.vocab = 640;
    c.heads = 8;
    c.head_dim = 128;
    c.o_groups = 8;
    c.expert_width = 768;
    return c;
}

/// A pack for `c` on disk; bit 14 of each 16-bit word stays clear, so every element is finite in F32, BF16 and F16.
fn writePack(a: std.mem.Allocator, io: Io, dir: Io.Dir, c: *const Config) !void {
    var n: u64 = 0;
    const json = try @import("host_test.zig").tinyPackLen(a, c, "", "", &n);
    defer a.free(json);
    const payload = try a.alloc(u8, @intCast(n));
    defer a.free(payload);
    var word: u32 = 0x9e3779b9;
    for (0..payload.len / 2) |i| {
        word = (word ^ @as(u32, @truncate(i))) *% 0x01000193;
        std.mem.writeInt(u16, payload[2 * i ..][0..2], @truncate((word >> 16) & 0xbfff), .little);
    }
    var len: [8]u8 = undefined;
    std.mem.writeInt(u64, &len, json.len, .little);
    var f = try dir.createFile(io, "model.safetensors", .{});
    defer f.close(io);
    try f.writePositionalAll(io, &len, 0);
    try f.writePositionalAll(io, json, 8);
    try f.writePositionalAll(io, payload, 8 + json.len);
}

/// Adds the named file's tag and bytes to `h`.
fn hashFile(a: std.mem.Allocator, io: Io, root: []const u8, tag: []const u8, h: *Sha256) !void {
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const file = try Io.Dir.cwd().openFile(io, try std.fmt.bufPrint(&buf, "{s}/{s}.bin", .{ root, tag }), .{});
    defer file.close(io);
    const size: usize = @intCast((try file.stat(io)).size);
    const bytes = try a.alloc(u8, size);
    defer a.free(bytes);
    try testing.expectEqual(size, try file.readPositionalAll(io, bytes, 0));
    h.update(tag);
    h.update(bytes);
}

fn keyOf(c: *const Config, rank: u32, world: u32) prepared.Key {
    return .{ .rank = rank, .world = world, .o_groups = c.o_groups, .blocks = &.{}, .dspark = true };
}

/// Every rank's files as the loader writes them, a block at a time and then the model-level tensors, as one digest.
fn rankFiles(a: std.mem.Allocator, io: Io, root: []const u8, pack: *const Pack, c: *const Config, world: u32) ![64]u8 {
    var h = Sha256.init(.{});
    for (0..world) |r| {
        const rank: u32 = @intCast(r);
        var pl = try plan.build(a, c, pack, .{ .rank = rank, .world = world });
        defer pl.deinit();
        var cache = try prepared.Cache.open(a, io, root, @splat(0), keyOf(c, rank, world), true, false);
        defer cache.deinit();
        for (pl.layers, 0..) |*l, i| {
            var b: named.Builder = .{ .gpa = a, .io = io, .pack = pack };
            defer b.deinit();
            var tag_buf: [16]u8 = undefined;
            const tag = try std.fmt.bufPrint(&tag_buf, "L{d}", .{i});
            try b.layer(l, c.o_groups, rank, world);
            try cache.put(&b, tag);
            try hashFile(a, io, cache.root, tag, &h);
        }
        var b: named.Builder = .{ .gpa = a, .io = io, .pack = pack };
        defer b.deinit();
        try b.top(&pl);
        try b.dspark(&pl);
        try cache.put(&b, "top");
        try hashFile(a, io, cache.root, "top", &h);
    }
    var digest: [32]u8 = undefined;
    h.final(&digest);
    return std.fmt.bytesToHex(digest, .lower);
}

/// A written pack and a folder for its prepared files.
const Fixture = struct {
    tmp: testing.TmpDir,
    dir: [:0]u8,
    root: []u8,
    pack: Pack,
    c: Config,

    fn init(f: *Fixture, a: std.mem.Allocator, io: Io) !void {
        f.c = eightGroups();
        f.tmp = testing.tmpDir(.{});
        errdefer f.tmp.cleanup();
        try writePack(a, io, f.tmp.dir, &f.c);
        f.dir = try f.tmp.dir.realPathFileAlloc(io, ".", a);
        errdefer a.free(f.dir);
        f.root = try std.fmt.allocPrint(a, "{s}/prepared", .{f.dir});
        errdefer a.free(f.root);
        f.pack = try Pack.open(a, io, f.dir);
    }

    fn deinit(f: *Fixture, a: std.mem.Allocator) void {
        f.pack.deinit();
        a.free(f.root);
        a.free(f.dir);
        f.tmp.cleanup();
    }
};

test "worlds one, two and four: rank files keep their main bytes" {
    const a = testing.allocator;
    const io = testing.io;
    var f: Fixture = undefined;
    try f.init(a, io);
    defer f.deinit(a);
    const pins = [_]struct { world: u32, hex: []const u8 }{
        .{ .world = 1, .hex = "93c42adbc0adb4e50036c743613c41cc7be72d17b38f1061e2912ec787ac85c6" },
        .{ .world = 2, .hex = "54a3f8703840a210c409b673e96f4acfb665abba13c344aa62cb197b98376442" },
        .{ .world = 4, .hex = "272a15e624d30215336f243f6789a62b6fdb2029680e78f701b63d9cbde74bcb" },
    };
    for (pins) |pin| {
        const hex = try rankFiles(a, io, f.root, &f.pack, &f.c, pin.world);
        try testing.expectEqualStrings(pin.hex, &hex);
    }
}

/// A pack of the default configuration's headers, no payload.
fn defaultPack(a: std.mem.Allocator, c: *const Config) !Pack {
    const header = try @import("host_test.zig").tinyPack(a, c, "", "");
    defer a.free(header);
    var p = Pack.init(a);
    errdefer p.deinit();
    try p.addHeader("model.safetensors", header, std.math.maxInt(u64));
    return p;
}

test "world three: whole output groups set the head split, and other widths keep whole blocks" {
    const c: Config = .{};
    const first = [_]u32{ 0, 3, 6 };
    const groups = [_]u32{ 3, 3, 2 };
    const heads = [_]u32{ 24, 24, 16 };
    const vocab = [_]u32{ 337 * 128, 337 * 128, 336 * 128 };
    for (0..3) |rank| {
        const g = try partition.outputGroups(c.o_groups, rank, 3);
        try testing.expectEqual(first[rank], g.lo);
        try testing.expectEqual(groups[rank], g.len);
        try testing.expectEqual(heads[rank], (try partition.outputShare(c.heads, c.o_groups, rank, 3)).len);
        try testing.expectEqual(heads[rank] * c.head_dim, (try partition.outputShare(c.heads * c.head_dim, c.o_groups, rank, 3)).len);
        try testing.expectEqual(groups[rank] * c.o_lora, (try partition.outputShare(c.o_groups * c.o_lora, c.o_groups, rank, 3)).len);
        try testing.expectEqual(@as(u32, 768), (try partition.share(c.expert_width, rank, 3)).len);
        try testing.expectEqual(vocab[rank], (try partition.share(c.vocab, rank, 3)).len);
    }
    // the even worlds' block shares and output-group shares are the same split
    for ([_]usize{ 1, 2, 4 }) |world| for (0..world) |rank| {
        for ([_]u32{ c.heads * c.head_dim, c.o_groups * c.o_lora }) |width| {
            try testing.expectEqualDeep(try partition.share(width, rank, world), try partition.outputShare(width, c.o_groups, rank, world));
        }
    };
    try testing.expectError(error.SplitNotOnGroups, partition.outputGroups(2, 0, 3));
    try testing.expectError(error.SplitNotOnGroups, partition.outputShare(9, 8, 0, 3));
}

test "world three: the default configuration plans every rank, each with its own output groups" {
    const a = testing.allocator;
    const c: Config = .{};
    var pack = try defaultPack(a, &c);
    defer pack.deinit();
    for (0..3) |r| {
        const rank: u32 = @intCast(r);
        var p = try plan.build(a, &c, &pack, .{ .rank = rank, .world = 3 });
        defer p.deinit();
        try testing.expectEqual(@as(usize, c.allLayers()), p.layers.len);
        try testing.expectEqual(@as(usize, 0), p.leftovers.len);
        const own = try partition.outputGroups(c.o_groups, rank, 3);
        const per_group = c.heads / c.o_groups;
        var seen: u32 = 0;
        for (p.layers[2].groups) |g| switch (g.proj) {
            .wo_a => {
                try testing.expectEqual(own.lo + seen, g.slice);
                seen += 1;
            },
            .wq_b => try testing.expectEqual(own.len * per_group * c.head_dim, g.part.group.n()),
            .wo_b => try testing.expectEqual(own.len * c.o_lora, g.part.group.k()),
            else => {},
        };
        try testing.expectEqual(own.len, seen);
        var sink: ?u64 = null;
        for (p.layers[2].natives) |v| {
            if (v.slice == .heads) sink = v.bytes;
        }
        try testing.expectEqual(@as(?u64, 4 * own.len * per_group), sink);
    }
}

test "world three: other uneven worlds and every serving path are still refused" {
    const a = testing.allocator;
    const c: Config = .{};
    var pack = try defaultPack(a, &c);
    defer pack.deinit();
    for ([_]u32{ 5, 6, 7 }) |world| try testing.expectError(error.BadWorld, plan.build(a, &c, &pack, .{ .rank = 0, .world = world }));
    try testing.expectError(error.BadWorld, plan.build(a, &c, &pack, .{ .rank = 3, .world = 3 }));
    const two_groups = config.tiny();
    try testing.expectError(error.BadWorld, plan.build(a, &two_groups, &pack, .{ .rank = 0, .world = 3 }));
    for ([_]u32{ 1, 2, 4 }) |world| try admission.serving(&c, world);
    try testing.expectError(error.BadWorld, admission.serving(&c, 3));
    try testing.expectError(error.BadWorld, admission.nativePrefill(&c, 3));
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const b = arena.allocator();
    const widths: block.Widths = .{};
    for ([_]u32{ 5, 6, 7 }) |world| try testing.expectError(error.InvalidBlock, block.emit(b, &c, &widths, .{ .world = world }, &.{0}, 1, 0, true));
    try testing.expectError(error.InvalidBlock, block.emit(b, &c, &widths, .{ .world = 3, .rank = 3 }, &.{0}, 1, 0, true));
    // experts wide enough for three ranks, so only the rank's group check refuses two output groups
    var narrow: Config = .{};
    narrow.o_groups = 2;
    try testing.expectError(error.InvalidBlock, block.emit(b, &narrow, &widths, .{ .world = 3 }, &.{0}, 1, 0, true));
}

const Seen = std.StringHashMapUnmanaged([]u8);

/// Counts each stored byte of `name` the span reads.
fn mark(a: std.mem.Allocator, seen: *Seen, pack: *const Pack, prefix: []const u8, part: []const u8, span: exl3.Span) !void {
    const name = try std.fmt.allocPrint(a, "{s}.{s}", .{ prefix, part });
    const info = try pack.need(name);
    const gop = try seen.getOrPut(a, name);
    if (!gop.found_existing) {
        gop.value_ptr.* = try a.alloc(u8, @intCast(info.nbytes));
        @memset(gop.value_ptr.*, 0);
    }
    for (0..@intCast(span.count)) |i| {
        const at: usize = @intCast(span.offset + i * span.stride);
        try testing.expect(at + span.len <= info.nbytes);
        for (gop.value_ptr.*[at..][0..@intCast(span.len)]) |*n| n.* += 1;
    }
}

fn markPart(a: std.mem.Allocator, seen: *Seen, pack: *const Pack, prefix: []const u8, p: exl3.Part) !void {
    try mark(a, seen, pack, prefix, "trellis", p.trellis);
    try mark(a, seen, pack, prefix, "suh", p.suh);
    try mark(a, seen, pack, prefix, "svh", p.svh);
}

/// Across the three ranks every stored byte is read once when split, or by all three when each holds it whole.
fn partsTileTheCheckpoint(gpa: std.mem.Allocator, f: *const Fixture) !void {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var seen: Seen = .empty;
    for (0..3) |r| {
        var p = try plan.build(gpa, &f.c, &f.pack, .{ .rank = @intCast(r), .world = 3 });
        defer p.deinit();
        for (p.layers) |*l| {
            for (l.groups) |g| try markPart(a, &seen, &f.pack, g.prefix, g.part);
            for (&l.experts) |*x| for (x.parts, x.prefixes) |part, prefix| try markPart(a, &seen, &f.pack, prefix, part);
        }
        if (p.head) |h| try markPart(a, &seen, &f.pack, h.prefix, h.part);
        if (p.dspark) |d| try markPart(a, &seen, &f.pack, d.main_proj.prefix, d.main_proj.part);
    }
    var it = seen.iterator();
    while (it.next()) |e| {
        const n = e.value_ptr.*[0];
        try testing.expect(n == 1 or n == 3);
        for (e.value_ptr.*) |x| try testing.expectEqual(n, x);
    }
    const expected = [_]struct { name: []const u8, reads: u8 }{
        .{ .name = "layers.2.attn.wq_b.trellis", .reads = 1 },
        .{ .name = "layers.2.attn.wq_b.svh", .reads = 1 },
        .{ .name = "layers.2.attn.wq_b.suh", .reads = 3 },
        .{ .name = "layers.2.attn.wo_b.trellis", .reads = 1 },
        .{ .name = "layers.2.attn.wo_b.suh", .reads = 1 },
        .{ .name = "layers.2.attn.wo_b.svh", .reads = 3 },
        .{ .name = "layers.2.attn.wo_a.slice.7.trellis", .reads = 1 },
        .{ .name = "layers.2.attn.wq_a.trellis", .reads = 3 },
        .{ .name = "layers.2.ffn.experts.0.w1.trellis", .reads = 1 },
        .{ .name = "head.trellis", .reads = 1 },
    };
    for (expected) |want| try testing.expectEqual(want.reads, seen.get(want.name).?[0]);
}

/// A wo_a entry's output group and the text after it, when `name` is one.
fn outputGroupOf(name: []const u8) ?struct { head: []const u8, group: u32, tail: []const u8 } {
    const mark_text = ".attn.wo_a.";
    const at = std.mem.indexOf(u8, name, mark_text) orelse return null;
    const start = at + mark_text.len;
    const dot = std.mem.indexOfScalarPos(u8, name, start, '.') orelse return null;
    const group = std.fmt.parseInt(u32, name[start..dot], 10) catch return null;
    return .{ .head = name[0..start], .group = group, .tail = name[dot..] };
}

fn product(shape: []const usize) usize {
    var n: usize = 1;
    for (shape) |d| n *= d;
    return n;
}

/// `piece` is the span of `whole` starting at `from` along the one axis where their shapes differ.
fn expectSlice(whole: *const named.Named, piece: *const named.Named, axis: usize, from: usize) !void {
    for (whole.shape, piece.shape, 0..) |w, p, i| if (i != axis) try testing.expectEqual(w, p);
    const element = whole.bytes.len / product(&whole.shape);
    const inner = product(whole.shape[axis + 1 ..]) * element;
    const take = piece.shape[axis] * inner;
    for (0..product(whole.shape[0..axis])) |outer| {
        const at = outer * whole.shape[axis] * inner + from * inner;
        try testing.expectEqualSlices(u8, whole.bytes[at..][0..take], piece.bytes[outer * take ..][0..take]);
    }
}

/// Reassembling the three ranks' entries gives world one's, whose bytes are pinned to main above.
fn imagesReassemble(a: std.mem.Allocator, io: Io, f: *const Fixture) !void {
    var whole_plan = try plan.build(a, &f.c, &f.pack, .{ .rank = 0, .world = 1 });
    defer whole_plan.deinit();
    var whole: named.Builder = .{ .gpa = a, .io = io, .pack = &f.pack };
    defer whole.deinit();
    try whole.all(&whole_plan, f.c.o_groups);
    var plans: [3]plan.Plan = undefined;
    var images: [3]named.Builder = undefined;
    var built: usize = 0;
    defer for (plans[0..built], images[0..built]) |*p, *b| {
        b.deinit();
        p.deinit();
    };
    for (&plans, &images, 0..) |*p, *b, r| {
        p.* = try plan.build(a, &f.c, &f.pack, .{ .rank = @intCast(r), .world = 3 });
        b.* = .{ .gpa = a, .io = io, .pack = &f.pack };
        built += 1;
        try b.all(p, f.c.o_groups);
    }
    var used = [_]usize{ 0, 0, 0 };
    var nb: [160]u8 = undefined;
    for (whole.out.items) |*want| {
        if (outputGroupOf(want.name)) |wo| {
            // exactly one rank holds this output group, under its own local index
            var r: usize = 0;
            while (r < 3) : (r += 1) {
                const own = try partition.outputGroups(f.c.o_groups, r, 3);
                if (wo.group < own.lo or wo.group >= own.lo + own.len) continue;
                const local = try std.fmt.bufPrint(&nb, "{s}{d}{s}", .{ wo.head, wo.group - own.lo, wo.tail });
                const got = images[r].find(local) orelse return error.TestMissingEntry;
                try testing.expectEqualSlices(usize, &want.shape, &got.shape);
                try testing.expectEqualSlices(u8, want.bytes, got.bytes);
                used[r] += 1;
                break;
            }
            try testing.expect(r < 3);
            continue;
        }
        var pieces: [3]*const named.Named = undefined;
        for (&pieces, &images, &used) |*piece, *image, *n| {
            piece.* = image.find(want.name) orelse return error.TestMissingEntry;
            try testing.expectEqual(want.kind, piece.*.kind);
            n.* += 1;
        }
        if (std.mem.endsWith(u8, want.name, ".data")) {
            // a ragged layout packs experts of different widths; raggedMatchesStored checks these bytes
            try testing.expectEqual(want.bytes.len, pieces[0].bytes.len + pieces[1].bytes.len + pieces[2].bytes.len);
            continue;
        }
        const axis = for (want.shape, pieces[0].shape, 0..) |w, p, i| {
            if (w != p) break i;
        } else null;
        if (axis) |split_axis| {
            var from: usize = 0;
            for (pieces) |piece| {
                try expectSlice(want, piece, split_axis, from);
                from += piece.shape[split_axis];
            }
            try testing.expectEqual(want.shape[split_axis], from);
        } else for (pieces) |piece| {
            try testing.expectEqualSlices(usize, &want.shape, &piece.shape);
            try testing.expectEqualSlices(u8, want.bytes, piece.bytes);
        }
    }
    for (used, &images) |n, *image| try testing.expectEqual(image.out.items.len, n);
}

/// A ragged projection's packed bytes are each expert's stored slice, back to back, in expert order.
fn raggedMatchesStored(a: std.mem.Allocator, io: Io, f: *const Fixture) !void {
    var checked: usize = 0;
    var nb: [160]u8 = undefined;
    for (0..3) |r| {
        var p = try plan.build(a, &f.c, &f.pack, .{ .rank = @intCast(r), .world = 3 });
        defer p.deinit();
        var b: named.Builder = .{ .gpa = a, .io = io, .pack = &f.pack };
        defer b.deinit();
        try b.all(&p, f.c.o_groups);
        for (p.layers) |*l| for (&l.experts, [_][]const u8{ "w1", "w2", "w3" }) |*x, w| {
            if (x.layout.kind == .stack) continue;
            const got = b.find(try std.fmt.bufPrint(&nb, "L{d}.moe.{s}.data", .{ l.index, w })) orelse return error.TestMissingEntry;
            var at: usize = 0;
            for (x.parts, x.prefixes) |part, prefix| {
                const bytes = try a.alloc(u8, @intCast(part.trellis.bytes()));
                defer a.free(bytes);
                try f.pack.read(io, try f.pack.need(try std.fmt.bufPrint(&nb, "{s}.trellis", .{prefix})), part.trellis, bytes);
                try testing.expectEqualSlices(u8, bytes, got.bytes[at..][0..bytes.len]);
                at += bytes.len;
            }
            try testing.expectEqual(got.bytes.len, at);
            checked += 1;
        };
    }
    try testing.expect(checked > 0);
}

/// The files the loader writes for three ranks read back byte for byte, against the schema it would expect.
fn filesReadBack(a: std.mem.Allocator, io: Io, f: *const Fixture) !void {
    _ = try rankFiles(a, io, f.root, &f.pack, &f.c, 3);
    for (0..3) |r| {
        const rank: u32 = @intCast(r);
        var p = try plan.build(a, &f.c, &f.pack, .{ .rank = rank, .world = 3 });
        defer p.deinit();
        var cache = try prepared.Cache.open(a, io, f.root, @splat(0), keyOf(&f.c, rank, 3), false, true);
        defer cache.deinit();
        for (p.layers, 0..) |*l, i| {
            var fresh: named.Builder = .{ .gpa = a, .io = io, .pack = &f.pack };
            defer fresh.deinit();
            try fresh.layer(l, f.c.o_groups, rank, 3);
            var schema: named.Builder = .{ .gpa = a, .io = io, .pack = &f.pack, .metadata_only = true };
            defer schema.deinit();
            try schema.layer(l, f.c.o_groups, rank, 3);
            var back: named.Builder = .{ .gpa = a, .io = io, .pack = &f.pack };
            defer back.deinit();
            var tag_buf: [16]u8 = undefined;
            try testing.expect(try cache.get(&back, try std.fmt.bufPrint(&tag_buf, "L{d}", .{i}), schema.out.items));
            try expectSameEntries(&fresh, &back);
        }
        var fresh: named.Builder = .{ .gpa = a, .io = io, .pack = &f.pack };
        defer fresh.deinit();
        try fresh.top(&p);
        try fresh.dspark(&p);
        try testing.expect(fresh.out.items.len > 0);
        var schema: named.Builder = .{ .gpa = a, .io = io, .pack = &f.pack, .metadata_only = true };
        defer schema.deinit();
        try schema.top(&p);
        try schema.dspark(&p);
        var back: named.Builder = .{ .gpa = a, .io = io, .pack = &f.pack };
        defer back.deinit();
        try testing.expect(try cache.get(&back, "top", schema.out.items));
        try expectSameEntries(&fresh, &back);
    }
}

fn expectSameEntries(fresh: *const named.Builder, back: *const named.Builder) !void {
    try testing.expectEqual(fresh.out.items.len, back.out.items.len);
    for (fresh.out.items, back.out.items) |x, y| {
        try testing.expectEqualStrings(x.name, y.name);
        try testing.expectEqualSlices(usize, &x.shape, &y.shape);
        try testing.expectEqualSlices(u8, x.bytes, y.bytes);
    }
}

test "world three: rank files tile the checkpoint, rebuild world one's entries and read back" {
    const a = testing.allocator;
    const io = testing.io;
    var f: Fixture = undefined;
    try f.init(a, io);
    defer f.deinit(a);
    try partsTileTheCheckpoint(a, &f);
    try imagesReassemble(a, io, &f);
    try raggedMatchesStored(a, io, &f);
    try filesReadBack(a, io, &f);
}
