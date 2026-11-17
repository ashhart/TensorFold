//! The `rm` command: a checkpoint `pull` cached, or one of its snapshots, out of the Hugging Face cache.
const std = @import("std");
const Allocator = std.mem.Allocator;
const hub = @import("hub.zig");
const cache = @import("rm_plan.zig");
const remove = @import("rm_remove.zig");
const locks = @import("sys").folder_lock;

/// The repo id with an optional @revision, and the two flags.
pub const Args = struct { spec: []const u8, yes: bool = false, dry_run: bool = false };

/// Where the question is asked: `interactive` when stdin is a terminal, and `input` reads its answer.
pub const Ask = struct { interactive: bool, input: *std.Io.Reader };

/// `REPO[@REVISION]` with `--yes` and `--dry-run` in any order; null for anything else.
pub fn parseArgs(args: []const []const u8) ?Args {
    var parsed: Args = .{ .spec = "" };
    for (args) |arg| {
        if (std.mem.eql(u8, arg, "--yes")) {
            parsed.yes = true;
        } else if (std.mem.eql(u8, arg, "--dry-run")) {
            parsed.dry_run = true;
        } else if (arg.len == 0 or arg[0] == '-' or parsed.spec.len > 0) {
            return null;
        } else parsed.spec = arg;
    }
    return if (parsed.spec.len > 0) parsed else null;
}

/// Lists what `args.spec` takes from the cache, asks unless --yes or --dry-run, then removes it; returns the exit code.
pub fn run(a: Allocator, io: std.Io, out: *std.Io.Writer, err_out: *std.Io.Writer, env: ?*const std.process.Environ.Map, override: ?[]const u8, args: Args, ask: Ask) !u8 {
    const at = std.mem.indexOfScalar(u8, args.spec, '@');
    const repo = if (at) |i| args.spec[0..i] else args.spec;
    if (refusal(repo)) |why| {
        try err_out.print("tensorfold rm: {s} is {s}; rm takes a cached repo id (owner/name), and tensorfold models lists what is cached\n", .{ args.spec, why });
        return 2;
    }
    const revision = if (at) |i| args.spec[i + 1 ..] else null;
    if (revision) |r| if (!hub.revisionOk(r)) {
        try err_out.print("tensorfold rm: {s} names no revision: after @ comes a ref such as main, or a 40-hex commit\n", .{args.spec});
        return 2;
    };
    const root = try hub.cacheDir(a, env, override);
    const name = try hub.repoDirName(a, repo);
    const folder = try std.fs.path.join(a, &.{ root, name });
    const kind = (std.Io.Dir.cwd().statFile(io, folder, .{ .follow_symlinks = false }) catch |e| switch (e) {
        error.FileNotFound, error.NotDir => {
            try err_out.print("tensorfold rm: {s} is not in the Hugging Face cache at {s}; tensorfold models lists what is cached\n", .{ repo, root });
            return 1;
        },
        else => return unreadable(err_out, folder, e),
    }).kind;
    if (kind != .directory) {
        const what = if (kind == .sym_link) "a link" else "not a folder";
        try err_out.print("tensorfold rm: {s} is {s}, and rm deletes only inside a repo's own cache folder; remove it by hand\n", .{ folder, what });
        return 1;
    }
    var mutation: ?locks.Lease = null;
    defer if (mutation) |l| l.release();
    if (!args.dry_run) mutation = locks.hold(a, root, true) catch |e| return lockFailed(err_out, root, e);
    const found = cache.scan(a, io, folder) catch |e| switch (e) {
        error.OutOfMemory => return e,
        else => return unreadable(err_out, folder, e),
    };
    const only = if (revision) |r| found.find(r) orelse return uncached(a, err_out, args.spec, repo, found) else null;
    const p = try cache.plan(a, found, only);
    try show(out, args.spec, folder, p);
    if (args.dry_run) {
        try out.writeAll("dry run: nothing was removed\n");
        return 0;
    }
    if (!args.yes) {
        mutation.?.release();
        mutation = null;
        if (!ask.interactive) {
            try out.flush();
            try err_out.print("tensorfold rm: stdin is not a terminal to ask on; pass --yes to remove {s}\n", .{args.spec});
            return 2;
        }
        try out.flush();
        try err_out.writeAll("Remove? [y/N] "); // stderr, so a redirected stdout cannot hide the question
        try err_out.flush();
        const answer = std.mem.trim(u8, (ask.input.takeDelimiter('\n') catch null) orelse "", " \t\r");
        if (!std.ascii.eqlIgnoreCase(answer, "y") and !std.ascii.eqlIgnoreCase(answer, "yes")) {
            try out.writeAll("nothing was removed\n");
            return 1;
        }
    }
    if (mutation == null) mutation = locks.hold(a, root, true) catch |e| return lockFailed(err_out, root, e);
    const current = cache.scan(a, io, folder) catch |e| return unreadable(err_out, folder, e);
    const current_only = if (revision) |r| current.find(r) orelse return uncached(a, err_out, args.spec, repo, current) else null;
    const fresh = try cache.plan(a, current, current_only);
    if (!std.mem.eql(u8, &found.fingerprint, &current.fingerprint)) {
        try err_out.print("tensorfold rm: {s} changed while awaiting confirmation; nothing was removed; run rm again to review the new plan\n", .{args.spec});
        return 1;
    }
    var held: std.ArrayList(locks.Lease) = .empty;
    defer {
        for (held.items) |l| l.release();
        held.deinit(a);
    }
    for (fresh.snapshots) |snapshot| {
        const path = try std.fs.path.join(a, &.{ folder, "snapshots", snapshot.name });
        const lease = locks.noFollow(a, io, path) catch |e| {
            try err_out.print("tensorfold rm: snapshot {s}/{s} is held by a server or cannot be locked ({t}); nothing was removed; stop its server and retry\n", .{ repo, snapshot.name, e });
            return 1;
        };
        held.append(a, lease) catch |e| {
            lease.release();
            return e;
        };
    }
    const folder_gone = remove.apply(a, io, root, name, fresh) catch |e| {
        try out.flush();
        try err_out.print("tensorfold rm: removing from {s} stopped ({t}); run it again to finish once that is fixed\n", .{ folder, e });
        return 1;
    };
    try out.print("removed {s}: {d:.2} GiB ({d} bytes) freed\n", .{ args.spec, gib(p.bytes), p.bytes });
    if (!p.whole and folder_gone) try out.print("nothing of {s} is left, so its folder went too\n", .{repo});
    if (!p.whole and !folder_gone and p.kept_snapshots == 0) {
        try out.print("{s} has no snapshots left; tensorfold rm {s} removes the rest of its folder\n", .{ repo, repo });
    }
    return 0;
}

/// Why `repo` cannot be a cached repo id (a path, a parent folder, an absolute path), or null when it can.
fn refusal(repo: []const u8) ?[]const u8 {
    const drive = repo.len >= 2 and std.ascii.isAlphabetic(repo[0]) and repo[1] == ':';
    if (std.fs.path.isAbsolutePosix(repo) or std.fs.path.isAbsoluteWindows(repo) or drive) return "an absolute path";
    var parts = std.mem.splitAny(u8, repo, "/\\");
    while (parts.next()) |part| if (std.mem.eql(u8, part, "..")) return "a path into a parent folder";
    parts.reset();
    while (parts.next()) |part| if (std.mem.eql(u8, part, ".")) return "a path";
    if (std.mem.indexOfAny(u8, repo, "\\~") != null or std.mem.countScalar(u8, repo, '/') > 1) return "a path";
    return if (hub.isRepoIdLike(repo)) null else "not a Hugging Face repo id (owner/name)";
}

/// What goes: each snapshot with its refs, the links out, then the counts and the bytes freed.
fn show(out: *std.Io.Writer, spec: []const u8, folder: []const u8, p: cache.Plan) !void {
    try out.print("{s} in {s}:\n", .{ spec, folder });
    for (p.snapshots) |s| {
        try out.print("  {s}", .{s.name});
        try refList(out, s.refs);
        try out.writeByte('\n');
    }
    for (p.outside) |link| try out.print("  {s} points outside the folder: only the link goes\n", .{link});
    try out.writeAll("goes: ");
    try counted(out, p.snapshots.len, "snapshot");
    try out.writeAll(", ");
    try counted(out, p.blobs.len, "blob");
    try out.writeAll(if (p.whole) ", " else " and ");
    try counted(out, p.refs.len, "ref");
    if (p.whole) try out.writeAll(" and the folder");
    try out.print(", {d:.2} GiB ({d} bytes) freed\n", .{ gib(p.bytes), p.bytes });
    if (p.whole or p.kept_snapshots == 0) return;
    try out.writeAll("stays: ");
    try counted(out, p.kept_snapshots, "snapshot");
    if (p.kept_blobs > 0) {
        try out.writeAll(" and ");
        try counted(out, p.kept_blobs, "shared blob");
    }
    try out.writeByte('\n');
}

/// The refusal of a revision this cache holds no snapshot of, naming the snapshots it does hold.
fn uncached(a: Allocator, err_out: *std.Io.Writer, spec: []const u8, repo: []const u8, found: cache.Repo) !u8 {
    if (found.snapshots.len == 0) {
        try err_out.print("tensorfold rm: {s} is not cached, and {s} has no snapshots; tensorfold rm {s} removes its folder\n", .{ spec, repo, repo });
        return 1;
    }
    try err_out.print("tensorfold rm: {s} is not cached; the snapshots of {s} are", .{ spec, repo });
    for (found.snapshots, 0..) |s, i| {
        try err_out.print("{s} {s}", .{ if (i == 0) "" else ",", s.name });
        try refList(err_out, try found.refsOf(a, s.name));
    }
    try err_out.writeByte('\n');
    return 1;
}

fn lockFailed(err_out: *std.Io.Writer, root: []const u8, e: anyerror) !u8 {
    try err_out.print("tensorfold rm: cache {s} is busy or cannot be locked ({t}); nothing was removed; retry after other pulls or removals finish\n", .{ root, e });
    return 1;
}

fn unreadable(err_out: *std.Io.Writer, folder: []const u8, e: anyerror) !u8 {
    try err_out.print("tensorfold rm: cannot read {s} ({t})\n", .{ folder, e });
    return 1;
}

fn refList(out: *std.Io.Writer, refs: []const []const u8) !void {
    for (refs, 0..) |r, i| try out.print("{s}{s}", .{ if (i == 0) " (" else ", ", r });
    if (refs.len > 0) try out.writeByte(')');
}

fn counted(out: *std.Io.Writer, n: usize, word: []const u8) !void {
    try out.print("{d} {s}{s}", .{ n, word, if (n == 1) "" else "s" });
}

fn gib(bytes: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / hub.GiB;
}

test "rm takes one repo id and its two flags in any order, and nothing else" {
    const both = parseArgs(&.{ "--dry-run", "Org/Name@main", "--yes" }).?;
    try std.testing.expectEqualStrings("Org/Name@main", both.spec);
    try std.testing.expect(both.yes and both.dry_run);
    const plain = parseArgs(&.{"Org/Name"}).?;
    try std.testing.expect(!plain.yes and !plain.dry_run);
    for ([_][]const []const u8{ &.{}, &.{"--yes"}, &.{ "Org/A", "Org/B" }, &.{ "Org/Name", "--force" }, &.{ "Org/Name", "-y" }, &.{""} }) |args| {
        try std.testing.expect(parseArgs(args) == null);
    }
}
