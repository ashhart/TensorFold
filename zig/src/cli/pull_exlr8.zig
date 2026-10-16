//! `pull REPO --exlr8 PROFILE --node R/N --dir DIR`: one node's share of an EXLR8 release, every byte checked.
const std = @import("std");
const Allocator = std.mem.Allocator;
const exlr8 = @import("exlr8");
const pull = @import("pull.zig");
const ranged = @import("pull_parts.zig");
const single = @import("pull_file.zig");
const runs = @import("pull_runs.zig");
const plan = @import("pull_exlr8_plan.zig");

pub const Options = struct { profile: []const u8, node: exlr8.node.Node, dir: []const u8 };

/// Whole-file pieces (the index and dense files): fewer, larger requests than the expert runs.
const whole_piece: u64 = 16 << 20;

pub fn run(a: Allocator, io: std.Io, out: *std.Io.Writer, err_out: *std.Io.Writer, env: ?*const std.process.Environ.Map, spec: []const u8, o: Options) !u8 {
    moreFiles();
    const at = std.mem.indexOfScalar(u8, spec, '@');
    const repo = if (at) |i| spec[0..i] else spec;
    const revision = if (at) |i| spec[i + 1 ..] else "main";
    const endpoint = pull.envValue(env, "HF_ENDPOINT") orelse "https://huggingface.co";
    var client: std.http.Client = .{ .allocator = a, .io = io };
    defer client.deinit();
    const sha = pull.resolveRevision(a, &client, out, endpoint, repo, revision) catch |e| {
        try err_out.print("{s}@{s}: the hub did not name a commit ({t})\n", .{ repo, revision, e });
        return 1;
    };
    const entries = pull.listTree(a, &client, endpoint, repo, sha) catch |e| {
        try err_out.print("{s}@{s}: the hub's file tree is unreachable ({t})\n", .{ repo, revision, e });
        return 1;
    };
    const manifest_rel = try std.fmt.allocPrint(a, "manifests/exlr8-manifest-{s}.json", .{o.profile});
    var whole: std.ArrayList(single.Entry) = .empty;
    var widths: std.StringHashMapUnmanaged(u64) = .empty;
    var has_manifest = false;
    for (entries) |e| {
        if (std.mem.eql(u8, e.path, manifest_rel)) has_manifest = true;
        if (plan.isWidthFile(e.path)) {
            try widths.put(a, e.path, e.size);
            continue;
        }
        if (std.mem.startsWith(u8, e.path, "index-slices/") or std.mem.startsWith(u8, e.path, "assets/")) continue;
        try whole.append(a, e);
    }
    if (!has_manifest) {
        try err_out.print("{s}@{s}: no {s}; a node pull needs an EXLR8 release and one of its profiles\n", .{ repo, sha, manifest_rel });
        return 1;
    }
    try out.print("[tensorfold] EXLR8 {s}, profile {s}, node {d}/{d} (halves {d}-{d} of every expert) into {s}\n", .{ repo, o.profile, o.node.rank, o.node.ranks, o.node.first(), o.node.end() - 1, o.dir });
    try out.flush();
    const files = fetchWhole(a, io, &client, out, err_out, endpoint, repo, sha, o.dir, whole.items) catch |e| {
        try err_out.print("  a whole file failed ({t}); pull again to resume\n", .{e});
        return 1;
    };
    const set_id = plan.checkPublished(a, io, o.dir, entries, manifest_rel) catch |e| {
        try err_out.print("  the release's hashes do not hold together ({t}); refusing\n", .{e});
        return 1;
    };
    try out.print("  {d} whole files equal their sha256; set_id {s}\n", .{ files, set_id });
    try out.flush();
    var p = plan.Plan.build(a, io, o.dir, manifest_rel, o.node, widths) catch |e| {
        try err_out.print("  planning the expert halves failed ({t})\n", .{e});
        return 1;
    };
    defer p.deinit(io);
    const targets = try a.alloc(runs.Target, p.targets.items.len);
    for (targets, p.targets.items) |*t, pt| t.* = .{
        .url = try std.fmt.allocPrint(a, "{s}/{s}/resolve/{s}/{s}", .{ endpoint, repo, sha, pt.rel }),
        .path = try std.fs.path.join(a, &.{ o.dir, pt.rel }),
        .size = pt.size,
    };
    const todo = try p.missing(a, io, targets);
    try out.print("  {d} width files, {d} halves ({d:.2} GiB); {d} of {d} runs to fetch\n", .{ targets.len, p.units.items.len, gib(p.expert_bytes), todo.len, p.runs.items.len });
    try out.flush();
    runs.fetch(a, io, targets, todo, p.check(), out) catch |e| {
        try err_out.print("  the expert halves failed ({t}); pull again to fetch only what is missing\n", .{e});
        return 1;
    };
    const left = try p.missing(a, io, targets);
    if (left.len != 0) {
        try err_out.print("  {d} runs still fail their checks on disk; refusing\n", .{left.len});
        return 1;
    }
    try (exlr8.node.Record{ .repo = repo, .revision = sha, .set_id = set_id, .profile = o.profile, .node = o.node, .expert_bytes = p.expert_bytes, .whole_files = files }).write(io, a, o.dir);
    try out.print("stored node {d}/{d} of {s}@{s} ({s}): every half equals its xxhash64, {d:.2} GiB of experts\n", .{ o.node.rank, o.node.ranks, repo, sha, o.profile, gib(p.expert_bytes) });
    return 0;
}

/// Index maps, connections and scans pass the default 256 descriptors: raise this process's own limit.
fn moreFiles() void {
    var l = std.posix.getrlimit(.NOFILE) catch return;
    const want: std.posix.rlim_t = if (l.max == std.posix.RLIM.INFINITY) 4096 else @min(l.max, 4096);
    if (l.cur >= want) return;
    l.cur = want;
    std.posix.setrlimit(.NOFILE, l) catch {};
}

fn gib(n: u64) f64 {
    return @as(f64, @floatFromInt(n)) / (1 << 30);
}

/// Every file the node needs whole, each equal to the hub's sha256 (LFS) or git blob sha1; returns how many.
pub fn fetchWhole(a: Allocator, io: std.Io, client: *std.http.Client, out: *std.Io.Writer, err_out: *std.Io.Writer, endpoint: []const u8, repo: []const u8, sha: []const u8, dir: []const u8, whole: []const single.Entry) !u32 {
    const w = std.Io.Dir.cwd();
    const tmp = try std.fs.path.join(a, &.{ dir, ".pull" });
    try w.createDirPath(io, tmp);
    var big: std.ArrayList(ranged.File) = .empty;
    var big_entries: std.ArrayList(single.Entry) = .empty;
    for (whole) |e| {
        const final = try std.fs.path.join(a, &.{ dir, e.path });
        if (std.fs.path.dirname(final)) |parent| try w.createDirPath(io, parent);
        if (try present(a, io, final, e)) continue;
        const url = try std.fmt.allocPrint(a, "{s}/{s}/resolve/{s}/{s}", .{ endpoint, repo, sha, e.path });
        if (e.sha256 != null) {
            // the storage link, not the hub's: pieces then skip a redirect each
            const link = try runs.resolve(a, client, url);
            try big.append(a, .{ .url = link, .path = try std.fmt.allocPrint(a, "{s}.part", .{final}), .size = e.size });
            try big_entries.append(a, e);
            continue;
        }
        const got = try single.downloadRetrying(a, io, client, out, url, tmp, e);
        try w.rename(try std.fs.path.join(a, &.{ tmp, got }), w, final, io);
    }
    if (big.items.len > 0) {
        try out.print("  {d} whole files in {d} MiB pieces\n", .{ big.items.len, whole_piece >> 20 });
        try out.flush();
        try ranged.fetch(a, io, big.items, whole_piece, out);
        for (big.items, big_entries.items) |f, e| {
            const d = try ranged.digest(a, io, f.path, f.size);
            if (!std.mem.eql(u8, &d, &e.sha256.?)) {
                ranged.discard(a, io, f, whole_piece);
                try err_out.print("  {s}: the bytes do not match the hub's sha256\n", .{e.path});
                return error.ShaMismatch;
            }
            try w.rename(f.path, w, try std.fs.path.join(a, &.{ dir, e.path }), io);
        }
    }
    w.deleteTree(io, tmp) catch {};
    return @intCast(whole.len);
}

/// Whether `path` already holds the entry's bytes (size, then its sha256 or git blob sha1).
fn present(a: Allocator, io: std.Io, path: []const u8, e: single.Entry) !bool {
    const size = single.fileStat(io, path) catch return false;
    if (size != e.size) return false;
    if (e.sha256) |want| return std.mem.eql(u8, &(try ranged.digest(a, io, path, size)), &want);
    const want = e.git_sha1 orelse return false;
    const file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    var git = std.crypto.hash.Sha1.init(.{});
    var head: [32]u8 = undefined;
    git.update(try std.fmt.bufPrint(&head, "blob {d}\x00", .{size}));
    var buf: [64 << 10]u8 = undefined;
    var off: u64 = 0;
    while (off < size) {
        const n = try file.readPositionalAll(io, buf[0..@intCast(@min(buf.len, size - off))], off);
        if (n == 0) return false;
        git.update(buf[0..n]);
        off += n;
    }
    return std.mem.eql(u8, &git.finalResult(), &want);
}

/// `--exlr8 PROFILE --node R/N --dir DIR` after the repo; null when none of them is there.
pub fn parseFlags(args: []const []const u8) !?Options {
    var o: Options = .{ .profile = "", .node = .{}, .dir = "" };
    var i: usize = 0;
    var any = false;
    while (i < args.len) : (i += 2) {
        if (i + 1 >= args.len) return error.BadFlags;
        const v = args[i + 1];
        any = true;
        if (std.mem.eql(u8, args[i], "--exlr8")) {
            o.profile = v;
        } else if (std.mem.eql(u8, args[i], "--node")) {
            o.node = try exlr8.node.Node.parse(v);
        } else if (std.mem.eql(u8, args[i], "--dir")) {
            o.dir = v;
        } else return error.BadFlags;
    }
    if (!any) return null;
    if (o.profile.len == 0 or o.dir.len == 0 or !plan.profileOk(o.profile)) return error.BadFlags;
    return o;
}

test "node flags parse; a missing profile or directory, or a stray flag, is refused" {
    const o = (try parseFlags(&.{ "--exlr8", "home", "--node", "1/2", "--dir", "/tmp/x" })).?;
    try std.testing.expectEqualStrings("home", o.profile);
    try std.testing.expectEqual(@as(u8, 1), o.node.rank);
    try std.testing.expect((try parseFlags(&.{})) == null);
    try std.testing.expectError(error.BadFlags, parseFlags(&.{ "--exlr8", "home" }));
    try std.testing.expectError(error.BadFlags, parseFlags(&.{ "--dir", "/tmp/x", "--exlr8", "../x" }));
    try std.testing.expectError(error.BadFlags, parseFlags(&.{ "--exlr8", "home", "--dir", "/x", "--fast", "1" }));
    try std.testing.expectError(error.BadExlr8Node, parseFlags(&.{ "--exlr8", "home", "--dir", "/x", "--node", "2/3" }));
}
