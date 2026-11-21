//! checkpoint_coded's tests: a folder of two DFloat11 groups and a plain tensor, read through a Checkpoint.

const std = @import("std");
const testing = std.testing;
const Io = std.Io;
const Checkpoint = @import("checkpoint.zig").Checkpoint;
const coded = @import("checkpoint_coded.zig");
const layout = @import("dfloat11_config.zig");
const fixture = @import("checkpoint_fixture.zig");
const groups = @import("dfloat11_fixtures.zig").groups;

const layer = groups[1]; // four members of 60, 1, 150 and 29 elements, blocks of 4 threads
const head = groups[2]; // one weight of 1,500 elements, codes of two tables, blocks of 4 threads
const members = [_][]const u8{ "attn.q", "attn.k", "mlp.up", "mlp.down" };
const ends = [_]usize{ 0, 60, 61, 211, 240 };
const config_text =
    \\{"dfloat11_config":{"version":"0.2.0","bytes_per_thread":8,"threads_per_block":[4],"pattern_dict":
    \\{"lm_head":[],"model.layers.\\d+":["attn.q","attn.k","mlp.up","mlp.down"]}}}
;
const plain = [_]u8{ 0x80, 0x3f, 0x00, 0x40 };

/// What a family's loader would know: the two names these tests take with `get`.
fn shapeOf(_: *const anyopaque, name: []const u8, buf: *[5]usize) ?[]const usize {
    const known = [_]struct { []const u8, [2]usize }{ .{ "lm_head.weight", .{ 30, 50 } }, .{ "model.layers.0.attn.q.weight", .{ 6, 10 } } };
    for (known) |k| if (std.mem.eql(u8, name, k[0])) {
        buf[0..2].* = k[1];
        return buf[0..2];
    };
    return null;
}
const shapes: coded.Shapes = .{ .ctx = &plain, .of = shapeOf };

/// The six tensors of `g` under `prefix`, with `gaps` in place of its own when given.
fn groupEntries(a: std.mem.Allocator, out: *std.ArrayList(fixture.Entry), prefix: []const u8, g: anytype, gaps: ?[]const u8) !void {
    const split = try a.alloc(u8, 8 * g.split_positions.len);
    for (g.split_positions, 0..) |v, i| std.mem.writeInt(i64, split[8 * i ..][0..8], v, .little);
    const stored = [_]struct { []const u8, []const u8, []const u8, []const usize }{
        .{ "luts", "U8", g.luts, &.{ g.luts.len / 256, 256 } },
        .{ "encoded_exponent", "U8", g.encoded_exponent, &.{g.encoded_exponent.len} },
        .{ "sign_mantissa", "U8", g.sign_mantissa, &.{g.sign_mantissa.len} },
        .{ "output_positions", "U8", g.output_positions, &.{g.output_positions.len} },
        .{ "gaps", "U8", gaps orelse g.gaps, &.{g.gaps.len} },
        .{ "split_positions", "I64", split, &.{g.split_positions.len} },
    };
    for (stored) |s| try out.append(a, .{
        .name = try std.fmt.allocPrint(a, "{s}.{s}", .{ prefix, s[0] }),
        .dtype = s[1],
        .shape = try a.dupe(usize, s[3]),
        .bytes = s[2],
    });
}

const Folder = struct {
    tmp: testing.TmpDir,
    arena: std.heap.ArenaAllocator,
    entries: std.ArrayList(fixture.Entry) = .empty,
    path: [64]u8 = undefined,

    /// The layer's group, the head's and a plain norm, not yet written.
    fn init() !Folder {
        var f: Folder = .{ .tmp = testing.tmpDir(.{}), .arena = .init(testing.allocator) };
        errdefer f.deinit();
        const a = f.arena.allocator();
        try groupEntries(a, &f.entries, "model.layers.0", layer, null);
        try groupEntries(a, &f.entries, "lm_head", head, null);
        try f.entries.append(a, .{ .name = "model.norm.weight", .dtype = "BF16", .shape = &.{2}, .bytes = &plain });
        return f;
    }

    fn deinit(f: *Folder) void {
        f.arena.deinit();
        f.tmp.cleanup();
    }

    /// Writes the entries (less the first one named `without`) as one file and gives the folder's path.
    fn write(f: *Folder, without: ?[]const u8) ![]const u8 {
        const a = f.arena.allocator();
        var kept: std.ArrayList(fixture.Entry) = .empty;
        var left = without;
        for (f.entries.items) |e| {
            if (left != null and std.mem.eql(u8, e.name, left.?)) left = null else try kept.append(a, e);
        }
        try fixture.write(a, testing.io, f.tmp.dir, "model.safetensors", kept.items);
        return std.fmt.bufPrint(&f.path, ".zig-cache/tmp/{s}", .{f.tmp.sub_path});
    }

    fn config(f: *Folder, text: []const u8) !layout.Config {
        return (try layout.parse(f.arena.allocator(), text)).?;
    }
};

test "a coded member answers under its own weight name with the bytes it was coded from, in the shape asked" {
    var f = try Folder.init();
    defer f.deinit();
    var ck = try Checkpoint.openModel(testing.allocator, testing.io, try f.write(null));
    defer ck.close();
    // unattached, the names are no header's: missing, as before
    try testing.expect(!ck.has("lm_head.weight"));
    try coded.attach(&ck, try f.config(config_text), shapes);
    var name: [64]u8 = undefined;
    for (members, 0..) |m, j| {
        const full = try std.fmt.bufPrint(&name, "model.layers.0.{s}.weight", .{m});
        try testing.expect(ck.has(full));
        const t = try ck.expect(full, .bf16, &.{ends[j + 1] - ends[j]});
        try testing.expectEqualSlices(u8, layer.want[2 * ends[j] .. 2 * ends[j + 1]], t.bytes);
        try testing.expectEqual(@as(usize, 0), @intFromPtr(t.bytes.ptr) % 2);
    }
    // `get` has no shape of its own: the family's
    const w = try ck.get("lm_head.weight");
    try testing.expect(w.is(.bf16, &.{ 30, 50 }));
    try testing.expectEqualSlices(u8, head.want, w.bytes);
    try testing.expectEqualSlices(u8, &plain, (try ck.get("model.norm.weight")).bytes);
    try testing.expect(!ck.has("model.layers.0.attn.q") and !ck.has("model.layers.1.attn.q.weight"));
    try testing.expectEqual(@as(usize, 0), ck.unused());
}

test "a group is decoded once: later gets share its buffer until a release, and after one it decodes again" {
    var f = try Folder.init();
    defer f.deinit();
    var ck = try Checkpoint.openModel(testing.allocator, testing.io, try f.write(null));
    defer ck.close();
    try coded.attach(&ck, try f.config(config_text), shapes);
    const c = ck.coded.?;
    for (c.groups.items) |g| try testing.expect(g.decoded == null);
    const q = try ck.get("model.layers.0.attn.q.weight");
    const again = try ck.get("model.layers.0.attn.q.weight");
    try testing.expectEqual(q.bytes.ptr, again.bytes.ptr);
    // another member of the group is a slice of the same buffer
    const up = try ck.expect("model.layers.0.mlp.up.weight", .bf16, &.{ 15, 10 });
    try testing.expectEqual(q.bytes.ptr + 2 * 61, up.bytes.ptr);
    var held: usize = 0;
    for (c.groups.items) |g| held += @intFromBool(g.decoded != null);
    try testing.expectEqual(@as(usize, 1), held); // the head's group was not asked for
    ck.release();
    for (c.groups.items) |g| try testing.expect(g.decoded == null);
    const back = try ck.get("model.layers.0.attn.q.weight");
    try testing.expectEqualSlices(u8, layer.want[0..120], back.bytes);
}

test "a checkpoint opened under a prefix and read under a scope finds the groups under it alone" {
    var f = try Folder.init();
    defer f.deinit();
    var ck = try Checkpoint.openModelPrefix(testing.allocator, testing.io, try f.write(null), "model.");
    defer ck.close();
    ck.scope = "model.";
    try coded.attach(&ck, try f.config(config_text), shapes);
    try testing.expectEqual(@as(usize, 1), ck.coded.?.groups.items.len);
    try testing.expect(ck.has("layers.0.mlp.down.weight"));
    const q = try ck.get("layers.0.attn.q.weight");
    try testing.expect(q.is(.bf16, &.{ 6, 10 }));
    try testing.expectEqualSlices(u8, layer.want[0..120], q.bytes);
    const down = try ck.expect("layers.0.mlp.down.weight", .bf16, &.{29});
    try testing.expectEqualSlices(u8, layer.want[2 * 211 ..], down.bytes);
}

test "a coded name with no shape, or a shape of another count, is refused by name and nothing is decoded" {
    var f = try Folder.init();
    defer f.deinit();
    var ck = try Checkpoint.openModel(testing.allocator, testing.io, try f.write(null));
    defer ck.close();
    try coded.attach(&ck, try f.config(config_text), shapes);
    // the family names shapes for two names; this is neither
    try testing.expectError(error.CodedShapeUnknown, ck.get("model.layers.0.mlp.up.weight"));
    try testing.expectError(error.UnexpectedTensor, ck.expect("model.layers.0.mlp.up.weight", .bf16, &.{ 10, 16 }));
    try testing.expectError(error.UnexpectedTensor, ck.expect("lm_head.weight", .bf16, &.{ 1 << 40, 1 << 40 }));
    for (ck.coded.?.groups.items) |g| try testing.expect(g.decoded == null);
}

test "a damaged group gives the decoder's refusal and holds nothing; the other group still reads" {
    var f = try Folder.init();
    defer f.deinit();
    const a = f.arena.allocator();
    const gaps = try a.dupe(u8, layer.gaps);
    gaps[0] ^= 0x08; // the first thread's gap, 0, becomes 1
    f.entries.clearRetainingCapacity();
    try groupEntries(a, &f.entries, "model.layers.0", layer, gaps);
    try groupEntries(a, &f.entries, "lm_head", head, null);
    var ck = try Checkpoint.openModel(testing.allocator, testing.io, try f.write(null));
    defer ck.close();
    try coded.attach(&ck, try f.config(config_text), shapes);
    try testing.expectError(error.GapMismatch, ck.get("model.layers.0.attn.q.weight"));
    for (ck.coded.?.groups.items) |g| try testing.expect(g.decoded == null);
    try testing.expectEqualSlices(u8, head.want, (try ck.get("lm_head.weight")).bytes);
}

test "attach refuses a group it cannot answer for: unlisted, partial, miscounted, or a name stored twice" {
    const head_only = "{\"dfloat11_config\":{\"bytes_per_thread\":8,\"threads_per_block\":[4],\"pattern_dict\":{\"lm_head\":[]}}}";
    const three = "{\"dfloat11_config\":{\"bytes_per_thread\":8,\"threads_per_block\":[4],\"pattern_dict\":" ++
        "{\"lm_head\":[],\"model.layers.\\\\d+\":[\"attn.q\",\"attn.k\",\"mlp.up\"]}}}";
    const cases = [_]struct { config: []const u8, without: ?[]const u8, extra: ?[]const u8, want: anyerror }{
        .{ .config = head_only, .without = null, .extra = null, .want = error.CodedGroupUnlisted },
        .{ .config = config_text, .without = "model.layers.0.gaps", .extra = null, .want = error.CodedGroupPartial },
        .{ .config = config_text, .without = "lm_head.split_positions", .extra = null, .want = error.CodedGroupPartial },
        .{ .config = config_text, .without = "lm_head.gaps", .extra = "lm_head.gaps", .want = error.CodedGroupPartial },
        .{ .config = three, .without = null, .extra = null, .want = error.CodedMembers },
        .{ .config = config_text, .without = null, .extra = "model.layers.0.mlp.up.weight", .want = error.CodedNameStored },
        .{ .config = config_text, .without = null, .extra = "lm_head.weight", .want = error.CodedNameStored },
    };
    for (cases) |case| {
        var f = try Folder.init();
        defer f.deinit();
        // a BF16 tensor of that name: a weight stored plain as well, or a group's own tensor of another dtype
        if (case.extra) |name| try f.entries.append(f.arena.allocator(), .{ .name = name, .dtype = "BF16", .shape = &.{2}, .bytes = &plain });
        var ck = try Checkpoint.openModel(testing.allocator, testing.io, try f.write(case.without));
        defer ck.close();
        try testing.expectError(case.want, coded.attach(&ck, try f.config(case.config), null));
        // refused whole: the checkpoint answers as an unattached one
        try testing.expectEqual(@as(?*coded.Coded, null), ck.coded);
        try testing.expectEqualSlices(u8, &plain, (try ck.get("model.norm.weight")).bytes);
    }
}

fn roundTrip(gpa: std.mem.Allocator, io: Io, dir: []const u8) !void {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const config = (try layout.parse(arena.allocator(), config_text)).?;
    var ck = try Checkpoint.openModel(gpa, io, dir);
    defer ck.close();
    try coded.attach(&ck, config, shapes);
    const q = try ck.get("model.layers.0.attn.q.weight");
    try testing.expectEqualSlices(u8, layer.want[0..120], q.bytes);
    ck.release();
    const w = try ck.expect("lm_head.weight", .bf16, &.{1500});
    try testing.expectEqualSlices(u8, head.want, w.bytes);
}

test "the content digest reads a coded folder: a group's empty tensor starts where the next tensor does" {
    var f = try Folder.init();
    defer f.deinit();
    // the head's group has no split positions: its last tensor is empty, at the offset the norm's bytes begin
    try testing.expectEqual(@as(usize, 0), head.split_positions.len);
    const Content = @import("checkpoint_identity.zig").Content;
    const dir = try f.write(null);
    const digest = try (try Content.startWithCache(testing.allocator, testing.io, dir, false, null)).finish();
    try testing.expectEqual(digest, try (try Content.startWithCache(testing.allocator, testing.io, dir, false, null)).finish());
}

test "a failed allocation while groups attach, decode, drop and close leaves nothing behind" {
    var f = try Folder.init();
    defer f.deinit();
    const dir = try f.write(null);
    var i: usize = 0;
    while (true) : (i += 1) {
        var failing = testing.FailingAllocator.init(testing.allocator, .{ .fail_index = i, .resize_fail_index = 0 });
        roundTrip(failing.allocator(), testing.io, dir) catch |err| if (!failing.has_induced_failure) return err;
        try testing.expectEqual(failing.allocated_bytes, failing.freed_bytes);
        if (!failing.has_induced_failure) break;
    }
    try testing.expect(i > 8); // the walk reached past the open into the attach and the decodes
}
