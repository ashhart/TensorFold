//! The snapshot --slide keeps before a folder's first lesson, the loader passing it by, and the rollback that uses it.
const std = @import("std");
const originals = @import("originals.zig");
const Allocator = std.mem.Allocator;
const Io = std.Io;
const Sha256 = std.crypto.hash.sha2.Sha256;

const shard1 = "model-00001-of-00002.safetensors";
const shard2 = "model-00002-of-00002.safetensors";
/// The files a lesson may write, as the snapshot lists them (name order), then one it never writes.
const files = [_][]const u8{
    "config.json", shard1, shard2, "model.safetensors.index.json", "mtp-4bit.safetensors", "tokenizer.json",
};
const kept_files = files.len - 1;
const up = [_]f32{ 1, 2, 3, 4, 5, 6 };
const index_pulled = "{\"metadata\": {\"total_size\": 48}, \"weight_map\": {\"layers.0.up.weight\": \"" ++ shard1 ++
    "\", \"layers.1.down.weight\": \"" ++ shard2 ++ "\"}}";
const index_taught = "{\"metadata\": {\"total_size\": 60}, \"weight_map\": {\"layers.0.up.weight\": \"" ++ shard1 ++
    "\", \"layers.1.down.weight\": \"" ++ shard2 ++ "\"}}";

fn path(a: Allocator, dir: []const u8, name: []const u8) ![:0]const u8 {
    return std.fmt.allocPrintSentinel(a, "{s}/{s}", .{ dir, name }, 0);
}

/// A safetensors file of one f32 tensor, as a shard holds it.
fn tensor(a: Allocator, name: []const u8, values: []const f32) ![]const u8 {
    const fmt = "{{\"{s}\":{{\"dtype\":\"F32\",\"shape\":[{d}],\"data_offsets\":[0,{d}]}}}}";
    const head = try std.fmt.allocPrint(a, fmt, .{ name, values.len, values.len * 4 });
    const padded = std.mem.alignForward(usize, head.len, 8);
    const out = try a.alloc(u8, 8 + padded + values.len * 4);
    std.mem.writeInt(u64, out[0..8], padded, .little);
    @memcpy(out[8..][0..head.len], head);
    @memset(out[8 + head.len ..][0 .. padded - head.len], ' ');
    @memcpy(out[8 + padded ..], std.mem.sliceAsBytes(values));
    return out;
}

fn put(a: Allocator, io: Io, dir: []const u8, name: []const u8, bytes: []const u8) !void {
    try Io.Dir.cwd().writeFile(io, .{ .sub_path = try path(a, dir, name), .data = bytes });
}

/// `bytes` written beside `name` and renamed over it, as the learner rewrites a shard, the index or config.json.
fn replace(a: Allocator, io: Io, dir: []const u8, name: []const u8, bytes: []const u8) !void {
    const tmp = try std.fmt.allocPrint(a, "{s}.tmp", .{name});
    try put(a, io, dir, tmp, bytes);
    try Io.Dir.cwd().rename(try path(a, dir, tmp), Io.Dir.cwd(), try path(a, dir, name), io);
}

/// A model folder as a pull leaves it: two indexed shards, an MTP head, config, tokenizer and a download's .cache.
fn model(a: Allocator, io: Io, tmp: std.testing.TmpDir) ![]const u8 {
    const dir = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}/m", .{tmp.sub_path});
    try Io.Dir.cwd().createDirPath(io, try path(a, dir, ".cache/huggingface"));
    try put(a, io, dir, shard1, try tensor(a, "layers.0.up.weight", &up));
    try put(a, io, dir, shard2, try tensor(a, "layers.1.down.weight", &.{ 7, 8, 9, 10 }));
    try put(a, io, dir, "mtp-4bit.safetensors", try tensor(a, "layers.1.mixer.weight", &.{ 11, 12 }));
    try put(a, io, dir, "model.safetensors.index.json", index_pulled);
    const config = "{\"model_type\": \"nemotron_h\", \"moe_shared_expert_intermediate_size\": 3712}";
    try put(a, io, dir, "config.json", config);
    try put(a, io, dir, "tokenizer.json", "{\"model\": {}}");
    try put(a, io, dir, ".cache/huggingface/x.metadata", "etag");
    return dir;
}

/// A lesson's writes as the learner makes them: a shard rewritten, one written in place, the index, config.json, more.
fn teach(a: Allocator, io: Io, dir: []const u8) !void {
    try replace(a, io, dir, shard1, try tensor(a, "layers.0.up.weight", &.{ 1, 2, 3, 4, 5, 6, 0.5, 0, 0 }));
    const fd = std.c.open(try path(a, dir, shard2), .{ .ACCMODE = .WRONLY }, @as(std.c.mode_t, 0));
    try std.testing.expect(fd >= 0);
    defer _ = std.c.close(fd);
    const moved = [_]f32{99};
    const end = (try read(a, io, try path(a, dir, shard2))).len - 4;
    try std.testing.expectEqual(@as(isize, 4), std.c.pwrite(fd, std.mem.asBytes(&moved), 4, @intCast(end)));
    try replace(a, io, dir, "model.safetensors.index.json", index_taught);
    const config = "{\"model_type\": \"nemotron_h\", \"moe_shared_expert_intermediate_size\": 3776, " ++
        "\"tensorfold_learned_neurons\": 64}";
    try replace(a, io, dir, "config.json", config);
    try put(a, io, dir, "living-weights-00001.safetensors", try tensor(a, "layers.0.a", &.{0.5}));
    try put(a, io, dir, shard2 ++ ".tmp", "a write cut short");
}

fn read(a: Allocator, io: Io, p: []const u8) ![]const u8 {
    return Io.Dir.cwd().readFileAlloc(io, p, a, .limited(1 << 20));
}

/// Every file of `files` in `dir`, read now.
fn bytesOf(a: Allocator, io: Io, dir: []const u8) ![files.len][]const u8 {
    var out: [files.len][]const u8 = undefined;
    for (files, &out) |name, *b| b.* = try read(a, io, try path(a, dir, name));
    return out;
}

fn sameBytes(x: [files.len][]const u8, y: [files.len][]const u8) bool {
    for (x, y) |p, q| if (!std.mem.eql(u8, p, q)) return false;
    return true;
}

fn exists(io: Io, p: []const u8) bool {
    _ = Io.Dir.cwd().statFile(io, p, .{ .follow_symlinks = false }) catch return false;
    return true;
}

fn has(text: []const u8, part: []const u8) bool {
    return std.mem.indexOf(u8, text, part) != null;
}

/// The first byte of `p` flipped, in place.
fn flip(p: [:0]const u8) void {
    const fd = std.c.open(p, .{ .ACCMODE = .RDWR }, @as(std.c.mode_t, 0));
    if (fd < 0) return;
    defer _ = std.c.close(fd);
    var b: [1]u8 = undefined;
    if (std.c.pread(fd, &b, 1, 0) != 1) return;
    b[0] ^= 1;
    _ = std.c.pwrite(fd, &b, 1, 0);
}

fn short(_: [:0]const u8) ?u64 {
    return 1;
}

fn plenty(_: [:0]const u8) ?u64 {
    return 1 << 50;
}

test "a snapshot is made once, before the first lesson, and later lessons reuse it as it was" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try model(a, io, tmp);
    const before = try bytesOf(a, io, dir);
    var why: []const u8 = "";
    const first = try originals.keep(io, dir, null, .{}, a, &why);
    try std.testing.expect(first.made);
    try std.testing.expectEqual(@as(usize, kept_files), first.files);
    try std.testing.expect(first.cloned == 0 or first.cloned == first.files); // one file system: all clones or none
    try std.testing.expect(!first.taught);
    const snap = try path(a, dir, originals.folder_name);
    const sums = try read(a, io, try path(a, snap, "SHA256SUMS"));
    for (files[0..kept_files], before[0..kept_files]) |name, bytes| {
        var d: [32]u8 = undefined;
        Sha256.hash(bytes, &d, .{});
        const hex = std.fmt.bytesToHex(d, .lower);
        try std.testing.expect(has(sums, try std.fmt.allocPrint(a, "{s}  {s}\n", .{ &hex, name })));
        try std.testing.expectEqualSlices(u8, bytes, try read(a, io, try path(a, snap, name)));
    }
    try std.testing.expect(!exists(io, try path(a, snap, "tokenizer.json"))); // a lesson never writes it
    try teach(a, io, dir);
    try std.testing.expect(!sameBytes(before, try bytesOf(a, io, dir)));
    const again = try originals.keep(io, dir, null, .{}, a, &why);
    try std.testing.expect(!again.made);
    try std.testing.expectEqualStrings(sums, try read(a, io, try path(a, snap, "SHA256SUMS")));
    for (files[0..kept_files], before[0..kept_files]) |name, bytes| {
        try std.testing.expectEqualSlices(u8, bytes, try read(a, io, try path(a, snap, name)));
    }
}

test "the snapshot is a hidden folder, and the folder's own files keep their bytes, inodes and single links" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try model(a, io, tmp);
    const before = try bytesOf(a, io, dir);
    var stats: [files.len]Io.File.Stat = undefined;
    for (files, &stats) |name, *st| {
        st.* = try Io.Dir.cwd().statFile(io, try path(a, dir, name), .{ .follow_symlinks = false });
    }
    var why: []const u8 = "";
    _ = try originals.keep(io, dir, null, .{}, a, &why);
    try std.testing.expect(originals.folder_name[0] == '.'); // loaders read a folder's own files, not hidden ones
    try std.testing.expect(sameBytes(before, try bytesOf(a, io, dir)));
    for (files, stats) |name, was| {
        const now = try Io.Dir.cwd().statFile(io, try path(a, dir, name), .{ .follow_symlinks = false });
        // never a link, which --slide refuses
        try std.testing.expect(now.inode == was.inode and now.nlink == 1 and now.kind == .file);
    }
    var d = try Io.Dir.cwd().openDir(io, dir, .{ .iterate = true });
    defer d.close(io);
    var it = d.iterate();
    var seen: usize = 0;
    while (try it.next(io)) |_| seen += 1;
    try std.testing.expectEqual(files.len + 2, seen); // its files, .cache and the snapshot: nothing else at the top
}

test "a kept file that does not match its original removes the snapshot and refuses the lesson" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try model(a, io, tmp);
    const before = try bytesOf(a, io, dir);
    for ([_]bool{ true, false }) |clones| {
        var why: []const u8 = "";
        const o: originals.Options = .{ .clones = clones, .free = plenty, .tamper = flip };
        try std.testing.expectError(error.Mismatch, originals.keep(io, dir, null, o, a, &why));
        try std.testing.expect(has(why, "does not match the original") and has(why, "nothing was taught"));
        try std.testing.expect(!exists(io, try path(a, dir, originals.folder_name)));
        try std.testing.expect(!exists(io, try path(a, dir, originals.folder_name ++ ".partial")));
        try std.testing.expect(sameBytes(before, try bytesOf(a, io, dir)));
    }
    var why: []const u8 = "";
    try std.testing.expect((try originals.keep(io, dir, null, .{}, a, &why)).made);
}

test "without clones the copy is refused by name when the disk is short, and checked like a clone when it fits" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try model(a, io, tmp);
    const before = try bytesOf(a, io, dir);
    var why: []const u8 = "";
    const tight: originals.Options = .{ .clones = false, .free = short };
    try std.testing.expectError(error.NoSpace, originals.keep(io, dir, null, tight, a, &why));
    try std.testing.expect(has(why, dir) and has(why, "GB free"));
    try std.testing.expect(!exists(io, try path(a, dir, originals.folder_name)));
    try std.testing.expect(!exists(io, try path(a, dir, originals.folder_name ++ ".partial")));
    const k = try originals.keep(io, dir, null, .{ .clones = false, .free = plenty }, a, &why);
    try std.testing.expect(k.made and k.cloned == 0 and k.files == kept_files);
    const snap = try path(a, dir, originals.folder_name);
    for (files[0..kept_files], before[0..kept_files]) |name, bytes| {
        try std.testing.expectEqualSlices(u8, bytes, try read(a, io, try path(a, snap, name)));
    }
}

test "a rollback puts back every kept file byte for byte, removes what lessons added and keeps the snapshot" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    for ([_]bool{ true, false }) |clones| {
        var tmp = std.testing.tmpDir(.{});
        defer tmp.cleanup();
        const dir = try model(a, io, tmp);
        const before = try bytesOf(a, io, dir);
        // relative here, kept absolute
        const graph = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}/facts.json", .{tmp.sub_path});
        var why: []const u8 = "";
        const o: originals.Options = .{ .clones = clones, .free = plenty };
        _ = try originals.keep(io, dir, graph, o, a, &why);
        const facts = "{\"facts\": [{\"id\": 1, \"text\": \"I like blue.\"}], \"next\": 2}";
        try Io.Dir.cwd().writeFile(io, .{ .sub_path = graph, .data = facts });
        try teach(a, io, dir);
        try std.testing.expect(!sameBytes(before, try bytesOf(a, io, dir)));
        const r = try originals.rollback(io, dir, o, a, &why);
        try std.testing.expectEqual(@as(usize, kept_files), r.files);
        if (!clones) try std.testing.expectEqual(@as(usize, 0), r.cloned);
        try std.testing.expect(sameBytes(before, try bytesOf(a, io, dir)));
        try std.testing.expectEqual(@as(usize, 2), r.removed.len);
        for (r.removed) |name| {
            const lesson = std.mem.eql(u8, name, "living-weights-00001.safetensors");
            try std.testing.expect(lesson or std.mem.eql(u8, name, shard2 ++ ".tmp"));
        }
        try std.testing.expect(!exists(io, try path(a, dir, "living-weights-00001.safetensors")));
        try std.testing.expect(!exists(io, try path(a, dir, shard2 ++ ".tmp")));
        try std.testing.expectEqual(@as(usize, 1), r.graphs.len);
        try std.testing.expect(std.fs.path.isAbsolute(r.graphs[0]) and !exists(io, graph));
        try std.testing.expect(exists(io, try path(a, dir, ".cache/huggingface/x.metadata")));
        try std.testing.expect(!(try originals.keep(io, dir, null, o, a, &why)).made); // the next lesson reuses it
    }
}

test "a rollback removes only what TensorFold writes there: files the user added after the snapshot stay" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try model(a, io, tmp);
    const before = try bytesOf(a, io, dir);
    var why: []const u8 = "";
    _ = try originals.keep(io, dir, null, .{}, a, &why);
    try teach(a, io, dir);
    const theirs = [_][]const u8{
        "user-extra.safetensors", "other.tensorfold-rollback", "living-weights-mine.safetensors",
        "notes.tmp",              "tokenizer.json.tmp",
    };
    for (theirs) |name| try put(a, io, dir, name, "the user's");
    const ours = [_][]const u8{
        "living-weights-00001.safetensors", shard2 ++ ".tmp",
        "living-weights.safetensors",       "living-weights-00002.safetensors.tmp",
    };
    for (ours[2..]) |name| try put(a, io, dir, name, "a lesson's");
    const r = try originals.rollback(io, dir, .{}, a, &why);
    try std.testing.expect(sameBytes(before, try bytesOf(a, io, dir)));
    for (theirs) |name| try std.testing.expectEqualStrings("the user's", try read(a, io, try path(a, dir, name)));
    for (ours) |name| try std.testing.expect(!exists(io, try path(a, dir, name)));
    try std.testing.expectEqual(ours.len, r.removed.len);
}

fn configUncloned(src: [:0]const u8) bool {
    return std.mem.endsWith(u8, src, "/config.json");
}

/// The put-back temps left in `dir`.
fn temps(io: Io, dir: []const u8) !usize {
    var d = try Io.Dir.cwd().openDir(io, dir, .{ .iterate = true });
    defer d.close(io);
    var n: usize = 0;
    var it = d.iterate();
    while (try it.next(io)) |e| n += @intFromBool(std.mem.endsWith(u8, e.name, ".tensorfold-rollback"));
    return n;
}

test "a rollback refused for space leaves the folder as it was, though other files were cloned before the check" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try model(a, io, tmp);
    const before = try bytesOf(a, io, dir);
    var why: []const u8 = "";
    _ = try originals.keep(io, dir, null, .{}, a, &why);
    try teach(a, io, dir);
    const taught = try bytesOf(a, io, dir);
    const short_mixed: originals.Options = .{ .free = short, .uncloned = configUncloned }; // config.json is copied
    try std.testing.expectError(error.NoSpace, originals.rollback(io, dir, short_mixed, a, &why));
    try std.testing.expect(has(why, "GB free"));
    try std.testing.expect(sameBytes(taught, try bytesOf(a, io, dir)));
    try std.testing.expectEqual(@as(usize, 0), try temps(io, dir));
    _ = try originals.rollback(io, dir, .{ .free = plenty, .uncloned = configUncloned }, a, &why);
    try std.testing.expect(sameBytes(before, try bytesOf(a, io, dir)));
}

test "a rollback whose put-back files do not match keeps what lessons added, and the next one finishes" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    for ([_]bool{ true, false }) |clones| {
        var tmp = std.testing.tmpDir(.{});
        defer tmp.cleanup();
        const dir = try model(a, io, tmp);
        const before = try bytesOf(a, io, dir);
        const graph = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}/facts.json", .{tmp.sub_path});
        var why: []const u8 = "";
        _ = try originals.keep(io, dir, graph, .{ .clones = clones, .free = plenty }, a, &why);
        try Io.Dir.cwd().writeFile(io, .{ .sub_path = graph, .data = "{\"facts\": [], \"next\": 1}" });
        try teach(a, io, dir);
        const off: originals.Options = .{ .clones = clones, .free = plenty, .tamper = flip };
        try std.testing.expectError(error.NotRestored, originals.rollback(io, dir, off, a, &why));
        try std.testing.expect(has(why, "roll back again"));
        const lesson = try path(a, dir, "living-weights-00001.safetensors");
        const cut = try path(a, dir, shard2 ++ ".tmp");
        try std.testing.expect(exists(io, lesson) and exists(io, cut) and exists(io, graph));
        _ = try originals.rollback(io, dir, .{ .clones = clones, .free = plenty }, a, &why);
        try std.testing.expect(sameBytes(before, try bytesOf(a, io, dir)));
        try std.testing.expect(!exists(io, lesson) and !exists(io, graph));
    }
}

test "a rollback is refused while any server holds the folder, and a --slide server holds it alone" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try model(a, io, tmp);
    const before = try bytesOf(a, io, dir);
    var why: []const u8 = "";
    _ = try originals.keep(io, dir, null, .{}, a, &why);
    try teach(a, io, dir);
    const taught = try bytesOf(a, io, dir);
    const plain = try originals.hold(a, dir, false);
    const other = try originals.hold(a, dir, false); // two plain servers share a folder
    other.release();
    try std.testing.expectError(error.Busy, originals.hold(a, dir, true)); // a --slide server refuses a folder in use
    try std.testing.expectError(error.Busy, originals.rollback(io, dir, .{}, a, &why));
    try std.testing.expect(has(why, "open: stop it, then roll back"));
    plain.release();
    const slide = try originals.hold(a, dir, true);
    try std.testing.expectError(error.Busy, originals.hold(a, dir, false));
    why = "";
    try std.testing.expectError(error.Busy, originals.rollback(io, dir, .{}, a, &why));
    try std.testing.expect(sameBytes(taught, try bytesOf(a, io, dir)));
    slide.release();
    _ = try originals.rollback(io, dir, .{}, a, &why);
    try std.testing.expect(sameBytes(before, try bytesOf(a, io, dir)));
}

test "a server is refused beside a --slide server or a rollback, and --slide refuses a folder that cannot lock" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try model(a, io, tmp);
    var why: []const u8 = "";
    const rewriting = try originals.hold(a, dir, true); // a --slide server's hold, or a rollback's
    try std.testing.expectError(error.Busy, originals.serving(a, dir, false, &why));
    try std.testing.expect(has(why, "--slide server") and has(why, "rollback"));
    try std.testing.expectError(error.Busy, originals.serving(a, dir, true, &why));
    try std.testing.expect(has(why, "teach a copy"));
    rewriting.release();
    const one = (try originals.serving(a, dir, false, &why)) orelse return error.TestExpectedHold;
    const two = (try originals.serving(a, dir, false, &why)) orelse return error.TestExpectedHold; // servers share it
    try std.testing.expectError(error.Busy, originals.serving(a, dir, true, &why));
    two.release();
    one.release();
    const unlockable = try path(a, dir, "no-such-folder"); // a folder flock cannot hold
    try std.testing.expectError(error.Unlockable, originals.serving(a, unlockable, true, &why));
    try std.testing.expect(has(why, "disk that locks"));
    try std.testing.expect(try originals.serving(a, unlockable, false, &why) == null); // served unheld, and told
    try std.testing.expect(has(why, "cannot be locked"));
}

test "a damaged snapshot or none is refused by rollback and by the next lesson, and nothing changes" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try model(a, io, tmp);
    var why: []const u8 = "";
    try std.testing.expectError(error.NoSnapshot, originals.rollback(io, dir, .{}, a, &why));
    try std.testing.expect(has(why, "nothing to roll back to"));
    _ = try originals.keep(io, dir, null, .{}, a, &why);
    try teach(a, io, dir);
    const taught = try bytesOf(a, io, dir);
    const snap = try path(a, dir, originals.folder_name);
    flip(try path(a, snap, shard1));
    why = "";
    try std.testing.expectError(error.Damaged, originals.rollback(io, dir, .{}, a, &why));
    try std.testing.expect(has(why, shard1) and has(why, "left as it is"));
    try std.testing.expect(sameBytes(taught, try bytesOf(a, io, dir)));
    flip(try path(a, snap, shard1)); // whole again: the next check is a missing file alone
    try Io.Dir.cwd().deleteFile(io, try path(a, snap, shard2));
    why = "";
    try std.testing.expectError(error.Damaged, originals.keep(io, dir, null, .{}, a, &why));
    try std.testing.expect(has(why, shard2));
    try Io.Dir.cwd().deleteFile(io, try path(a, snap, "SHA256SUMS"));
    why = "";
    try std.testing.expectError(error.Damaged, originals.keep(io, dir, null, .{}, a, &why));
    try std.testing.expect(has(why, "SHA256SUMS"));
    try std.testing.expect(sameBytes(taught, try bytesOf(a, io, dir)));
}

test "a later lesson hashes the snapshot it finds: a damaged one refuses the lesson and the folder stays as taught" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try model(a, io, tmp);
    var why: []const u8 = "";
    _ = try originals.keep(io, dir, null, .{}, a, &why);
    try teach(a, io, dir);
    const taught = try bytesOf(a, io, dir);
    flip(try path(a, try path(a, dir, originals.folder_name), shard1));
    why = "";
    try std.testing.expectError(error.Damaged, originals.keep(io, dir, null, .{}, a, &why));
    try std.testing.expect(has(why, shard1) and has(why, "the way back is broken"));
    try std.testing.expect(sameBytes(taught, try bytesOf(a, io, dir)));
}

test "the process that made or hashed the snapshot does not hash it again at its later lessons" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try model(a, io, tmp);
    var why: []const u8 = "";
    const first = try originals.keep(io, dir, null, .{}, a, &why);
    try std.testing.expect(first.made and !first.checked);
    const found = try originals.keep(io, dir, null, .{}, a, &why); // another process's first lesson
    try std.testing.expect(!found.made and found.checked and found.files == kept_files and found.bytes == first.bytes);
    flip(try path(a, try path(a, dir, originals.folder_name), shard1));
    const later = try originals.keep(io, dir, null, .{ .checked = true }, a, &why);
    try std.testing.expect(!later.made and !later.checked);
}

test "a folder an earlier release taught is kept as it is now, and the snapshot says so" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try model(a, io, tmp);
    try teach(a, io, dir);
    const taught = try bytesOf(a, io, dir);
    var why: []const u8 = "";
    const k = try originals.keep(io, dir, null, .{}, a, &why);
    try std.testing.expect(k.made and k.taught and k.files == kept_files + 1); // its lesson file is kept as well
    try std.testing.expect(sameBytes(taught, try bytesOf(a, io, dir)));
}
