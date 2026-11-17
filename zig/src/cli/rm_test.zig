//! `rm` in a temporary cache: whole repos, one snapshot, shared blobs, refs, dry runs, refusals and links out.
const std = @import("std");
const builtin = @import("builtin");
const platform = @import("sys");
const rm = @import("rm.zig");
const Allocator = std.mem.Allocator;
const Io = std.Io;
const Dir = std.Io.Dir;

const rev_a = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const rev_b = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
const shared = "1111111111111111111111111111111111111111111111111111111111111111";
const only_a = "2222222222222222222222222222222222222222222222222222222222222222";
const only_b = "3333333333333333333333333333333333333333333333333333333333333333";
const shared_body: [100]u8 = @splat('s');
const a_body: [1000]u8 = @splat('a');
const b_body: [2000]u8 = @splat('b');
const kept_body: [300]u8 = @splat('k');

/// How snapshots hold their files: links into blobs/, as huggingface_hub writes them, or hard links and copies.
const Layout = enum { links, hard_links };

/// What one `tensorfold rm` printed and returned.
const Run = struct { code: u8, out: []const u8, err: []const u8 };

/// A temporary cache with Org/Name, whose snapshots a and b share a blob, and Org/Other beside it.
const Cache = struct {
    tmp: std.testing.TmpDir,
    a: Allocator,
    io: Io,
    base: []const u8,
    hub: []const u8,
    folder: []const u8,

    fn init(a: Allocator, io: Io, layout: Layout) !Cache {
        var tmp = std.testing.tmpDir(.{ .iterate = true });
        errdefer tmp.cleanup();
        const base = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
        const hub = try std.fs.path.join(a, &.{ base, "hub" });
        var c: Cache = .{ .tmp = tmp, .a = a, .io = io, .base = base, .hub = hub, .folder = try std.fs.path.join(a, &.{ hub, "models--Org--Name" }) };
        try c.write("blobs/" ++ shared, &shared_body);
        try c.write("blobs/" ++ only_a, &a_body);
        // TensorFold's own file inside a snapshot (--slide's copy of the originals) goes with the snapshot.
        try c.write("snapshots/" ++ rev_a ++ "/.tensorfold-original/model.safetensors", &kept_body);
        try c.write(".no_exist/" ++ rev_a ++ "/generation_config.json", "");
        switch (layout) {
            .links => {
                try c.write("blobs/" ++ only_b, &b_body);
                try c.link("snapshots/" ++ rev_a ++ "/config.json", &.{ "..", "..", "blobs", shared });
                try c.link("snapshots/" ++ rev_a ++ "/model.safetensors", &.{ "..", "..", "blobs", only_a });
                try c.link("snapshots/" ++ rev_b ++ "/config.json", &.{ "..", "..", "blobs", shared });
                try c.link("snapshots/" ++ rev_b ++ "/deep/model.safetensors", &.{ "..", "..", "..", "blobs", only_b });
            },
            .hard_links => {
                try c.hardLink("blobs/" ++ shared, "snapshots/" ++ rev_a ++ "/config.json");
                try c.hardLink("blobs/" ++ only_a, "snapshots/" ++ rev_a ++ "/model.safetensors");
                try c.hardLink("blobs/" ++ shared, "snapshots/" ++ rev_b ++ "/config.json");
                try c.write("snapshots/" ++ rev_b ++ "/deep/model.safetensors", &b_body); // a copy, kept without links
            },
        }
        for ([_][2][]const u8{ .{ "main", rev_a }, .{ "v1", rev_a }, .{ "pr/1", rev_a }, .{ "dev", rev_b } }) |ref| {
            try c.write(try std.fmt.allocPrint(a, "refs/{s}", .{ref[0]}), ref[1]);
        }
        try c.tmp.dir.createDirPath(io, "hub/models--Org--Other/snapshots/" ++ rev_a);
        try c.tmp.dir.writeFile(io, .{ .sub_path = "hub/models--Org--Other/snapshots/" ++ rev_a ++ "/config.json", .data = "{}" });
        return c;
    }

    fn deinit(c: *Cache) void {
        c.tmp.cleanup();
    }

    fn path(c: *Cache, rel: []const u8) ![]const u8 {
        return std.fs.path.join(c.a, &.{ c.folder, rel });
    }

    fn write(c: *Cache, rel: []const u8, data: []const u8) !void {
        const at = try c.path(rel);
        try Dir.cwd().createDirPath(c.io, std.fs.path.dirname(at).?);
        try Dir.cwd().writeFile(c.io, .{ .sub_path = at, .data = data });
    }

    /// A link at `rel` to `target`'s parts, joined with this platform's separators.
    fn link(c: *Cache, rel: []const u8, target: []const []const u8) !void {
        const at = try c.path(rel);
        try Dir.cwd().createDirPath(c.io, std.fs.path.dirname(at).?);
        try symLink(c.io, try std.fs.path.join(c.a, target), at, false);
    }

    /// A hard link at `rel` to the file `from`, as `pull` makes one where Windows grants no symlinks.
    fn hardLink(c: *Cache, from: []const u8, rel: []const u8) !void {
        const at = try c.path(rel);
        try Dir.cwd().createDirPath(c.io, std.fs.path.dirname(at).?);
        if (platform.c.link(try std.fs.path.joinZ(c.a, &.{ c.folder, from }), try c.a.dupeSentinel(u8, at, 0)) != 0) return error.HardLinkFailed;
    }

    fn has(c: *Cache, rel: []const u8) bool {
        _ = Dir.cwd().statFile(c.io, c.path(rel) catch return false, .{ .follow_symlinks = false }) catch return false;
        return true;
    }

    fn read(c: *Cache, rel: []const u8) ![]u8 {
        return Dir.cwd().readFileAlloc(c.io, try std.fs.path.join(c.a, &.{ c.base, rel }), c.a, .limited(1 << 20));
    }

    /// Org/Other's config, which no run against Org/Name may touch.
    fn otherStays(c: *Cache) !void {
        try std.testing.expectEqualStrings("{}", try c.read("hub/models--Org--Other/snapshots/" ++ rev_a ++ "/config.json"));
    }

    /// Every entry under the temporary folder as "path kind size", sorted: what a run that removes nothing leaves.
    fn listing(c: *Cache) ![]const u8 {
        var lines: std.ArrayList([]const u8) = .empty;
        var walker = try c.tmp.dir.walk(c.a);
        defer walker.deinit();
        while (try walker.next(c.io)) |e| {
            if (std.mem.eql(u8, e.path, "hub/.tensorfold-lock") or std.mem.eql(u8, e.path, "hub\\.tensorfold-lock")) continue;
            const st = try e.dir.statFile(c.io, e.basename, .{ .follow_symlinks = false });
            try lines.append(c.a, try std.fmt.allocPrint(c.a, "{s} {t} {d}", .{ e.path, st.kind, st.size }));
        }
        std.mem.sort([]const u8, lines.items, {}, struct {
            fn lessThan(_: void, l: []const u8, r: []const u8) bool {
                return std.mem.lessThan(u8, l, r);
            }
        }.lessThan);
        return std.mem.join(c.a, "\n", lines.items);
    }

    /// `tensorfold rm ARGS` on this cache; `answer` is what a terminal types, and null means stdin is not one.
    fn run(c: *Cache, args: []const []const u8, answer: ?[]const u8) !Run {
        var out: std.Io.Writer.Allocating = .init(c.a);
        var err: std.Io.Writer.Allocating = .init(c.a);
        var input = std.Io.Reader.fixed(answer orelse "");
        const parsed = rm.parseArgs(args) orelse return error.TestBadArguments;
        const code = try rm.run(c.a, c.io, &out.writer, &err.writer, null, c.hub, parsed, .{ .interactive = answer != null, .input = &input });
        return .{ .code = code, .out = out.written(), .err = err.written() };
    }
};

/// A symlink, or a skip where this Windows account may not make one.
fn symLink(io: Io, target: []const u8, at: []const u8, is_directory: bool) (Dir.SymLinkError || error{SkipZigTest})!void {
    Dir.cwd().symLink(io, target, at, .{ .is_directory = is_directory }) catch |e| switch (e) {
        error.AccessDenied, error.PermissionDenied => return if (builtin.os.tag == .windows) error.SkipZigTest else e,
        else => return e,
    };
}

/// The cache for `layout`, or null where this platform cannot make it.
fn cacheFor(a: Allocator, layout: Layout) !?Cache {
    return Cache.init(a, std.testing.io, layout) catch |e| switch (e) {
        error.SkipZigTest => return null,
        else => return e,
    };
}

fn expectRun(r: Run, code: u8) !void {
    if (r.code == code) return;
    std.debug.print("exit {d}, wanted {d}\nout:\n{s}\nerr:\n{s}\n", .{ r.code, code, r.out, r.err });
    return error.TestUnexpectedExit;
}

fn expectHas(text: []const u8, needle: []const u8) !void {
    if (std.mem.indexOf(u8, text, needle) != null) return;
    std.debug.print("missing \"{s}\" in:\n{s}\n", .{ needle, text });
    return error.TestExpectedText;
}

test "rm REPO removes every snapshot, blob and ref with the folder, and frees a blob two snapshots share once" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    for ([_]Layout{ .links, .hard_links }) |layout| {
        var c = (try cacheFor(arena.allocator(), layout)) orelse continue;
        defer c.deinit();
        const r = try c.run(&.{ "Org/Name", "--yes" }, null);
        try expectRun(r, 0);
        try expectHas(r.out, "  " ++ rev_a ++ " (main, pr/1, v1)\n  " ++ rev_b ++ " (dev)\n");
        try expectHas(r.out, "goes: 2 snapshots, ");
        try expectHas(r.out, ", 4 refs and the folder, 0.00 GiB (3560 bytes) freed\n");
        try expectHas(r.out, "removed Org/Name: 0.00 GiB (3560 bytes) freed\n");
        try std.testing.expect(!c.has(""));
        try c.otherStays();
    }
}

test "rm REPO@COMMIT removes one snapshot and only the blobs no other snapshot links to" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    for ([_]Layout{ .links, .hard_links }) |layout| {
        var c = (try cacheFor(arena.allocator(), layout)) orelse continue;
        defer c.deinit();
        const r = try c.run(&.{ "Org/Name@" ++ rev_a, "--yes" }, null);
        try expectRun(r, 0);
        try expectHas(r.out, "goes: 1 snapshot, 1 blob and 3 refs, 0.00 GiB (1420 bytes) freed\nstays: 1 snapshot and 1 shared blob\n");
        for ([_][]const u8{ "snapshots/" ++ rev_a, "blobs/" ++ only_a, ".no_exist" }) |gone| try std.testing.expect(!c.has(gone));
        try std.testing.expect(c.has("blobs/" ++ shared));
        try std.testing.expectEqualStrings(&shared_body, try c.read("hub/models--Org--Name/snapshots/" ++ rev_b ++ "/config.json"));
        try std.testing.expectEqualStrings(&b_body, try c.read("hub/models--Org--Name/snapshots/" ++ rev_b ++ "/deep/model.safetensors"));
        try c.otherStays();
    }
}

test "rm REPO@REF removes the refs naming its snapshot, and the folder once nothing is left" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    for ([_]Layout{ .links, .hard_links }) |layout| {
        var c = (try cacheFor(arena.allocator(), layout)) orelse continue;
        defer c.deinit();
        const first = try c.run(&.{ "Org/Name@main", "--yes" }, null);
        try expectRun(first, 0);
        for ([_][]const u8{ "refs/main", "refs/v1", "refs/pr/1", "refs/pr", "snapshots/" ++ rev_a }) |gone| try std.testing.expect(!c.has(gone));
        try std.testing.expect(c.has("refs/dev") and c.has("snapshots/" ++ rev_b));
        const last = try c.run(&.{ "Org/Name@dev", "--yes" }, null);
        try expectRun(last, 0);
        try expectHas(last.out, " and 1 ref, 0.00 GiB (2140 bytes) freed\n");
        try expectHas(last.out, "nothing of Org/Name is left, so its folder went too\n");
        try std.testing.expect(!c.has(""));
        try c.otherStays();
    }
}

test "rm --dry-run lists what would go and leaves every file as it was" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    for ([_]Layout{ .links, .hard_links }) |layout| {
        var c = (try cacheFor(arena.allocator(), layout)) orelse continue;
        defer c.deinit();
        const before = try c.listing();
        for ([_][2][]const u8{ .{ "Org/Name", "(3560 bytes)" }, .{ "Org/Name@" ++ rev_a, "(1420 bytes)" } }) |case| {
            const r = try c.run(&.{ case[0], "--dry-run" }, null);
            try expectRun(r, 0);
            try expectHas(r.out, case[1]);
            try expectHas(r.out, "dry run: nothing was removed\n");
            try std.testing.expectEqualStrings(before, try c.listing());
        }
    }
}

test "rm refuses a path, a parent folder, an absolute path and a repo the cache does not hold, by name" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var c = (try cacheFor(arena.allocator(), .hard_links)).?;
    defer c.deinit();
    const before = try c.listing();
    const paths = [_][]const u8{ "./Org/Name", "../Org/Name", "Org/..", "..", "/models/Org/Name", "C:\\models\\Name", "Org\\Name", "Org/Name/x", "~/Org/Name", "plain" };
    for (paths) |spec| {
        const r = try c.run(&.{ spec, "--yes" }, null);
        try expectRun(r, 2);
        try expectHas(r.err, try std.fmt.allocPrint(c.a, "tensorfold rm: {s} is ", .{spec}));
        try expectHas(r.err, "; rm takes a cached repo id (owner/name), and tensorfold models lists what is cached\n");
    }
    const missing = try c.run(&.{ "Org/Missing", "--yes" }, null);
    try expectRun(missing, 1);
    try expectHas(missing.err, "tensorfold rm: Org/Missing is not in the Hugging Face cache at ");
    try expectHas(missing.err, "; tensorfold models lists what is cached\n");
    const unknown = try c.run(&.{ "Org/Name@v9", "--yes" }, null);
    try expectRun(unknown, 1);
    try expectHas(unknown.err, "Org/Name@v9 is not cached; the snapshots of Org/Name are " ++ rev_a ++ " (main, pr/1, v1), " ++ rev_b ++ " (dev)\n");
    const bad = try c.run(&.{ "Org/Name@../main", "--yes" }, null);
    try expectRun(bad, 2);
    try expectHas(bad.err, "Org/Name@../main names no revision");
    try std.testing.expectEqualStrings(before, try c.listing());
}

test "rm asks on a terminal, and without one refuses by name unless --yes" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var c = (try cacheFor(arena.allocator(), .hard_links)).?;
    defer c.deinit();
    const before = try c.listing();
    const piped = try c.run(&.{"Org/Name"}, null);
    try expectRun(piped, 2);
    try expectHas(piped.err, "tensorfold rm: stdin is not a terminal to ask on; pass --yes to remove Org/Name\n");
    for ([_][]const u8{ "n\n", "", "maybe\n" }) |answer| {
        const declined = try c.run(&.{"Org/Name"}, answer);
        try expectRun(declined, 1);
        try std.testing.expectEqualStrings("Remove? [y/N] ", declined.err);
        try expectHas(declined.out, "goes: 2 snapshots, 2 blobs, 4 refs and the folder, 0.00 GiB (3560 bytes) freed\nnothing was removed\n");
    }
    try std.testing.expectEqualStrings(before, try c.listing());
    const one = try c.run(&.{"Org/Name@" ++ rev_a}, "y\n");
    try expectRun(one, 0);
    try std.testing.expect(!c.has("snapshots/" ++ rev_a) and c.has("snapshots/" ++ rev_b));
    const rest = try c.run(&.{"Org/Name"}, "YES\r\n");
    try expectRun(rest, 0);
    try std.testing.expect(!c.has(""));
}

test "security: rm never follows a link out of the repo folder" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var c = (try cacheFor(arena.allocator(), .links)) orelse return error.SkipZigTest;
    defer c.deinit();
    try c.tmp.dir.createDirPath(c.io, "outside/inner");
    try c.tmp.dir.writeFile(c.io, .{ .sub_path = "outside/keep.bin", .data = "keep me" });
    try c.tmp.dir.writeFile(c.io, .{ .sub_path = "outside/inner/x.bin", .data = "and me" });
    const outside = try Dir.cwd().realPathFileAlloc(c.io, try std.fs.path.join(c.a, &.{ c.base, "outside" }), c.a);
    try c.link("snapshots/" ++ rev_a ++ "/escape.bin", &.{ "..", "..", "..", "..", "outside", "keep.bin" });
    try symLink(c.io, outside, try c.path("snapshots/" ++ rev_a ++ "/escape"), true);
    try c.link("snapshots/" ++ rev_b ++ "/far.bin", &.{ outside, "inner", "x.bin" });
    const one = try c.run(&.{ "Org/Name@" ++ rev_a, "--yes" }, null);
    try expectRun(one, 0);
    try expectHas(one.out, "  snapshots/" ++ rev_a ++ "/escape points outside the folder: only the link goes\n");
    try expectHas(one.out, "  snapshots/" ++ rev_a ++ "/escape.bin points outside the folder: only the link goes\n");
    try expectHas(one.out, "(1420 bytes) freed");
    const all = try c.run(&.{ "Org/Name", "--yes" }, null);
    try expectRun(all, 0);
    try expectHas(all.out, "  snapshots/" ++ rev_b ++ "/far.bin points outside the folder: only the link goes\n");
    try expectHas(all.out, "(2140 bytes) freed");
    try std.testing.expect(!c.has(""));
    try std.testing.expectEqualStrings("keep me", try c.read("outside/keep.bin"));
    try std.testing.expectEqualStrings("and me", try c.read("outside/inner/x.bin"));
}

test "security: a repo folder that is itself a link is refused, and what it points to stays" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var c = (try cacheFor(arena.allocator(), .hard_links)).?;
    defer c.deinit();
    const kept = "elsewhere/models--Org--Linked/snapshots/" ++ rev_a ++ "/config.json";
    try c.tmp.dir.createDirPath(c.io, std.fs.path.dirname(kept).?);
    try c.tmp.dir.writeFile(c.io, .{ .sub_path = kept, .data = "{}" });
    const target = try Dir.cwd().realPathFileAlloc(c.io, try std.fs.path.join(c.a, &.{ c.base, "elsewhere", "models--Org--Linked" }), c.a);
    try symLink(c.io, target, try std.fs.path.join(c.a, &.{ c.hub, "models--Org--Linked" }), true);
    const r = try c.run(&.{ "Org/Linked", "--yes" }, null);
    try expectRun(r, 1);
    try expectHas(r.err, "models--Org--Linked is a link, and rm deletes only inside a repo's own cache folder; remove it by hand\n");
    try std.testing.expectEqualStrings("{}", try c.read(kept));
}

/// Publishes a second snapshot with pull's real cached-blob linker while the confirmation reader is waiting.
const Publish = struct {
    cache: *Cache,
    reader: Io.Reader = .{ .vtable = &.{ .stream = stream }, .buffer = &.{}, .seek = 0, .end = 0 },
    buffer: [64]u8 = undefined,
    done: bool = false,
    failure: ?anyerror = null,

    fn stream(r: *Io.Reader, w: *Io.Writer, limit: Io.Limit) Io.Reader.StreamError!usize {
        const self: *@This() = @fieldParentPtr("reader", r);
        if (self.done) return error.EndOfStream;
        self.done = true;
        self.publish() catch |e| {
            self.failure = e;
            return error.ReadFailed;
        };
        const reply = "y\n";
        const n = limit.minInt(reply.len);
        try w.writeAll(reply[0..n]);
        return n;
    }

    fn publish(self: *@This()) !void {
        const c = self.cache;
        const dir = try c.path("snapshots/cccccccccccccccccccccccccccccccccccccccc");
        try Dir.cwd().createDirPath(c.io, dir);
        try @import("pull_file.zig").linkIntoSnapshot(c.a, c.io, dir, "model.safetensors", only_a);
        try c.write("refs/new", "cccccccccccccccccccccccccccccccccccccccc");
    }
};

test "rm refuses a changed plan after a pull links a new snapshot during confirmation" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var c = try Cache.init(arena.allocator(), std.testing.io, .links);
    defer c.deinit();
    var input: Publish = .{ .cache = &c };
    input.reader.buffer = &input.buffer;
    var out: Io.Writer.Allocating = .init(c.a);
    var err: Io.Writer.Allocating = .init(c.a);
    const code = try rm.run(c.a, c.io, &out.writer, &err.writer, null, c.hub, .{ .spec = "Org/Name@" ++ rev_a }, .{ .interactive = true, .input = &input.reader });
    try std.testing.expect(input.done and input.failure == null);
    try std.testing.expectEqual(@as(u8, 1), code);
    try std.testing.expect(std.mem.indexOf(u8, err.written(), "changed") != null);
    try std.testing.expect(c.has("snapshots/" ++ rev_a));
    try std.testing.expectEqualStrings(&a_body, try c.read("hub/models--Org--Name/snapshots/cccccccccccccccccccccccccccccccccccccccc/model.safetensors"));
    const retry = try c.run(&.{ "Org/Name@" ++ rev_a, "--yes" }, null);
    try std.testing.expectEqual(@as(u8, 0), retry.code);
    try std.testing.expect(c.has("blobs/" ++ only_a));
    try std.testing.expectEqualStrings(&a_body, try c.read("hub/models--Org--Name/snapshots/cccccccccccccccccccccccccccccccccccccccc/model.safetensors"));
}

test "rm refuses a snapshot held by a server lease before removing refs or files" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var c = try Cache.init(arena.allocator(), std.testing.io, .hard_links);
    defer c.deinit();
    const path = try c.a.dupeSentinel(u8, try c.path("snapshots/" ++ rev_a), 0);
    const fd = platform.c.open(path, .{ .ACCMODE = .RDONLY, .DIRECTORY = true, .CLOEXEC = true }, @as(platform.c.mode_t, 0));
    try std.testing.expect(fd >= 0);
    defer _ = platform.c.close(fd);
    try std.testing.expectEqual(@as(c_int, 0), platform.c.flock(fd, platform.c.LOCK.SH | platform.c.LOCK.NB));
    defer _ = platform.c.flock(fd, platform.c.LOCK.UN);
    const before = try c.listing();
    for ([_][]const u8{ "Org/Name@main", "Org/Name" }) |spec| {
        const got = try c.run(&.{ spec, "--yes" }, null);
        try std.testing.expectEqual(@as(u8, 1), got.code);
        try std.testing.expect(std.mem.indexOf(u8, got.err, rev_a) != null);
        try std.testing.expect(std.mem.indexOf(u8, got.err, "held") != null);
        try std.testing.expectEqualStrings(before, try c.listing());
    }
}

test "rm refuses a cache held by a pull without removing data" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var c = try Cache.init(arena.allocator(), std.testing.io, .hard_links);
    defer c.deinit();
    var held: ?platform.folder_lock.Lease = try platform.folder_lock.hold(c.a, c.hub, true);
    defer if (held) |l| l.release();
    const before = try c.listing();
    const refused = try c.run(&.{ "Org/Name", "--yes" }, null);
    try std.testing.expectEqual(@as(u8, 1), refused.code);
    try std.testing.expect(std.mem.indexOf(u8, refused.err, "Busy") != null);
    try std.testing.expectEqualStrings(before, try c.listing());
    held.?.release();
    held = null;
    try std.testing.expectEqual(@as(u8, 0), (try c.run(&.{ "Org/Name", "--yes" }, null)).code);
}

test "rm keeps a served snapshot and its blobs when removing another snapshot" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var c = try Cache.init(arena.allocator(), std.testing.io, .hard_links);
    defer c.deinit();
    const lease = try platform.folder_lock.hold(c.a, try c.path("snapshots/" ++ rev_b), false);
    defer lease.release();
    const got = try c.run(&.{ "Org/Name@" ++ rev_a, "--yes" }, null);
    try std.testing.expectEqual(@as(u8, 0), got.code);
    try std.testing.expect(c.has("blobs/" ++ shared));
    try std.testing.expectEqualStrings(&shared_body, try c.read("hub/models--Org--Name/snapshots/" ++ rev_b ++ "/config.json"));
}
