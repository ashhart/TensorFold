const std = @import("std");

const upstream_url = "git@github.com:ashhart/TensorFold.git";

const usage =
    \\Usage: .zig-toolchain/zig run tools/sync_upstream.zig -- [--check | --continue] [--remote NAME]
    \\Fetch TensorFold main and zig over SSH. --check reports drift without merging.
    \\Sync requires a clean PR branch containing the latest zig history. It prepares
    \\an uncommitted main merge, aligns dependencies and runs correctness checks.
    \\Review and commit the result, then open a PR against zig. Never pushes.
    \\--continue verifies the pending merge after resolving conflicts or drift.
    \\The remote is discovered by its SSH URL; --remote selects one explicitly.
    \\
;

const Options = struct {
    check: bool = false,
    continuing: bool = false,
    help: bool = false,
    remote: ?[]const u8 = null,

    fn parse(args: []const []const u8) !Options {
        var options: Options = .{};
        var i: usize = 0;
        while (i < args.len) : (i += 1) {
            if (std.mem.eql(u8, args[i], "--check")) {
                options.check = true;
            } else if (std.mem.eql(u8, args[i], "--continue")) {
                options.continuing = true;
            } else if (std.mem.eql(u8, args[i], "--help")) {
                options.help = true;
            } else if (std.mem.eql(u8, args[i], "--remote")) {
                i += 1;
                if (i == args.len) return error.MissingRemote;
                if (args[i].len == 0 or args[i][0] == '-') return error.InvalidRemote;
                if (options.remote != null) return error.DuplicateRemote;
                options.remote = args[i];
            } else return error.UnknownOption;
        }
        if (options.check and options.continuing) return error.ConflictingOptions;
        return options;
    }
};

pub const Git = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    archive_sources: bool = false,
    cwd: std.process.Child.Cwd = .inherit,

    fn run(g: Git, args: []const []const u8) !std.process.RunResult {
        const argv = try g.allocator.alloc([]const u8, args.len + 1);
        defer g.allocator.free(argv);
        argv[0] = "git";
        @memcpy(argv[1..], args);
        return std.process.run(g.allocator, g.io, .{ .argv = argv, .cwd = g.cwd });
    }

    fn output(g: Git, args: []const []const u8) ![]const u8 {
        const result = try g.run(args);
        defer g.allocator.free(result.stderr);
        if (!result.term.success()) {
            if (result.stdout.len > 0) std.debug.print("{s}", .{result.stdout});
            g.allocator.free(result.stdout);
            var lines = std.mem.splitScalar(u8, result.stderr, '\n');
            while (lines.next()) |line| {
                if (std.mem.startsWith(u8, line, "sign_and_send_pubkey:")) continue;
                if (line.len > 0) std.debug.print("{s}\n", .{line});
            }
            return error.GitCommandFailed;
        }
        // The caller uses the process arena, including the untrimmed allocation.
        return std.mem.trim(u8, result.stdout, " \r\n");
    }

    fn ancestor(g: Git, older: []const u8, newer: []const u8) !bool {
        const result = try g.run(&.{ "merge-base", "--is-ancestor", older, newer });
        defer g.allocator.free(result.stdout);
        defer g.allocator.free(result.stderr);
        return switch (result.term) {
            .exited => |code| switch (code) {
                0 => true,
                1 => false,
                else => error.InvalidGitHistory,
            },
            else => error.GitCommandFailed,
        };
    }
};

fn validateRemote(actual: []const u8, expected: []const u8) !void {
    if (!std.mem.eql(u8, actual, expected)) return error.UnexpectedRemote;
}

fn validateWorktree(branch: []const u8, status: []const u8) !void {
    if (branch.len == 0) return error.DetachedHead;
    if (std.mem.eql(u8, branch, "main") or std.mem.eql(u8, branch, "zig")) return error.ProtectedBranch;
    if (status.len != 0) return error.UncommittedChanges;
}

fn upstreamRemote(git: Git, requested: ?[]const u8) ![]const u8 {
    if (requested) |name| {
        try validateRemote(try git.output(&.{ "remote", "get-url", "--all", name }), upstream_url);
        return name;
    }
    var names = std.mem.splitScalar(u8, try git.output(&.{"remote"}), '\n');
    var found: ?[]const u8 = null;
    while (names.next()) |name| {
        if (name.len == 0) continue;
        const url = try git.output(&.{ "remote", "get-url", "--all", name });
        if (!std.mem.eql(u8, url, upstream_url)) continue;
        if (found != null) return error.AmbiguousUpstreamRemote;
        found = name;
    }
    return found orelse error.UpstreamRemoteMissing;
}

fn validateIdle(git: Git, allow_merge: bool) !void {
    for ([_][]const u8{ "MERGE_HEAD", "CHERRY_PICK_HEAD", "REVERT_HEAD", "rebase-apply", "rebase-merge", "sequencer", "BISECT_LOG" }) |marker| {
        if (allow_merge and std.mem.eql(u8, marker, "MERGE_HEAD")) continue;
        const path = try git.output(&.{ "rev-parse", "--path-format=absolute", "--git-path", marker });
        if (std.Io.Dir.cwd().access(git.io, path, .{})) return error.GitOperationInProgress else |err| if (err != error.FileNotFound) return err;
    }
}

fn prepareMerge(git: Git, zig_ref: []const u8, tip: []const u8) !bool {
    try validateWorktree(try git.output(&.{ "branch", "--show-current" }), try git.output(&.{ "status", "--porcelain", "--untracked-files=all" }));
    try validateIdle(git, false);
    if (!try git.ancestor(zig_ref, "HEAD")) {
        std.debug.print("Start a PR branch from the latest TensorFold zig history before syncing.\n", .{});
        return error.ZigBaseOutdated;
    }
    if (try git.ancestor(tip, "HEAD")) return false;
    _ = git.output(&.{ "merge", "--no-ff", "--no-commit", tip }) catch |err| {
        std.debug.print("Merge stopped. Inspect git status; resolve conflicts and review dependency/kernel changes before committing, or run git merge --abort. No refs were pushed.\n", .{});
        return err;
    };
    return true;
}

fn pendingMerge(git: Git, zig_ref: []const u8, main_ref: []const u8) ![]const u8 {
    try validateWorktree(try git.output(&.{ "branch", "--show-current" }), "");
    try validateIdle(git, true);
    const tip = try git.output(&.{ "rev-parse", "--verify", "MERGE_HEAD" });
    if ((try git.output(&.{ "diff", "--name-only", "--diff-filter=U" })).len != 0) return error.UnresolvedConflicts;
    if (!try git.ancestor(zig_ref, "HEAD")) return error.ZigBaseOutdated;
    if (!try git.ancestor(tip, main_ref)) return error.UnexpectedMergeTarget;
    return tip;
}

pub fn command(io: std.Io, argv: []const []const u8) !void {
    var child = try std.process.spawn(io, .{ .argv = argv });
    if (!(try child.wait(io)).success()) return error.CommandFailed;
}

fn source(git: Git, dir: []const u8, url: []const u8, revision: []const u8) ![]const u8 {
    std.debug.print("Preparing {s} at {s}\n", .{ dir, revision });
    const exists = if (std.Io.Dir.cwd().access(git.io, dir, .{})) true else |err| switch (err) {
        error.FileNotFound => false,
        else => return err,
    };
    if (git.archive_sources) {
        const marker = try std.fmt.allocPrint(git.allocator, "{s}/.git", .{dir});
        if (std.Io.Dir.cwd().access(git.io, marker, .{})) return error.ExistingGitCheckout else |err| if (err != error.FileNotFound) return err;
        const identity = try std.fmt.allocPrint(git.allocator, "{s}#{s}", .{ url, revision });
        if (exists) {
            try @import("native_install.zig").verify(git.allocator, git.io, dir, .source, identity, null);
            return revision;
        }
        if (!std.mem.startsWith(u8, url, "git@github.com:") or !std.mem.endsWith(u8, url, ".git")) return error.UnexpectedRemote;
        const archive_url = try std.fmt.allocPrint(git.allocator, "https://codeload.github.com/{s}/tar.gz/{s}", .{ url[15 .. url.len - 4], revision });
        const archive = try std.fmt.allocPrint(git.allocator, "{s}.tar.gz", .{dir});
        try command(git.io, &.{ "curl", "--fail", "--location", "--retry", "3", archive_url, "--output", archive });
        try std.Io.Dir.cwd().createDirPath(git.io, dir);
        try command(git.io, &.{ "tar", "-xzf", archive, "--strip-components=1", "-C", dir });
        try @import("native_install.zig").record(git.allocator, git.io, dir, .source, identity, null);
        return revision;
    }
    if (!exists) _ = try git.output(&.{ "clone", "--no-checkout", url, dir });
    try validateRemote(try git.output(&.{ "-C", dir, "remote", "get-url", "origin" }), url);
    if (exists and (try git.output(&.{ "-C", dir, "status", "--porcelain" })).len != 0) return error.DependencySourceDirty;
    _ = try git.output(&.{ "-C", dir, "fetch", "--depth=1", "origin", revision });
    _ = try git.output(&.{ "-C", dir, "checkout", "--detach", "FETCH_HEAD" });
    return git.output(&.{ "-C", dir, "rev-parse", "HEAD" });
}

pub fn buildDependencies(git: Git, record: *std.json.Value, jpeg: bool, mlx: bool, jobs: []const u8) !void {
    const deps = if (git.archive_sources) "build/deps-archives" else "build/deps";
    const jpeg_source = try std.fmt.allocPrint(git.allocator, "{s}/libjpeg-turbo", .{deps});
    const mlx_source = try std.fmt.allocPrint(git.allocator, "{s}/mlx", .{deps});
    const bridge_source = try std.fmt.allocPrint(git.allocator, "{s}/mlx-c", .{deps});
    const fmt_source = try std.fmt.allocPrint(git.allocator, "{s}/fmt", .{deps});
    const jpeg_build = if (git.archive_sources) "build/jpeg-archive-build" else "build/jpeg-build";
    const mlx_build = if (git.archive_sources) "build/mlx-archive-build" else "build/mlx-build";
    const bridge_build = if (git.archive_sources) "build/mlxc-archive-build" else "build/mlxc-build";
    if (jpeg) {
        try std.Io.Dir.cwd().createDirPath(git.io, deps);
        _ = try source(git, jpeg_source, "git@github.com:libjpeg-turbo/libjpeg-turbo.git", record.object.get("jpeg_version").?.string);
        const root = try std.process.currentPathAlloc(git.io, git.allocator);
        const prefix = try std.fmt.allocPrint(git.allocator, "-DCMAKE_INSTALL_PREFIX={s}/build/jpeg", .{root});
        try command(git.io, &.{ "cmake", "-S", jpeg_source, "-B", jpeg_build, "-DCMAKE_BUILD_TYPE=Release", "-DENABLE_SHARED=OFF", "-DENABLE_STATIC=ON", "-DWITH_TOOLS=OFF", "-DWITH_TESTS=OFF", prefix });
        try command(git.io, &.{ "cmake", "--build", jpeg_build, "--parallel", jobs });
        try command(git.io, &.{ "cmake", "--install", jpeg_build });
        try @import("native_install.zig").record(git.allocator, git.io, "build/jpeg", .jpeg, record.object.get("jpeg_version").?.string, null);
    }
    if (mlx) {
        try std.Io.Dir.cwd().createDirPath(git.io, deps);
        const revision = try source(git, mlx_source, "git@github.com:ml-explore/mlx.git", record.object.get("mlx_revision").?.string);
        const bridge_revision = try source(git, bridge_source, "git@github.com:ml-explore/mlx-c.git", record.object.get("mlx_c_revision").?.string);
        _ = try source(git, fmt_source, "git@github.com:fmtlib/fmt.git", "12.1.0");
        const root = try std.process.currentPathAlloc(git.io, git.allocator);
        const prefix = try std.fmt.allocPrint(git.allocator, "-DCMAKE_INSTALL_PREFIX={s}/build/mlx", .{root});
        const mlx_prefix = try std.fmt.allocPrint(git.allocator, "-DCMAKE_PREFIX_PATH={s}/build/mlx", .{root});
        const fmt = try std.fmt.allocPrint(git.allocator, "-DFETCHCONTENT_SOURCE_DIR_FMT={s}/{s}", .{ root, fmt_source });
        try command(git.io, &.{ "cmake", "-S", mlx_source, "-B", mlx_build, "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_OSX_DEPLOYMENT_TARGET=26.2", "-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON", "-DCMAKE_INSTALL_RPATH=@loader_path", "-DBUILD_SHARED_LIBS=ON", "-DMLX_BUILD_TESTS=OFF", "-DMLX_BUILD_EXAMPLES=OFF", prefix, fmt });
        try command(git.io, &.{ "cmake", "--build", mlx_build, "--parallel", jobs });
        try command(git.io, &.{ "cmake", "--install", mlx_build });
        // MLX 0.32.3 adds an optional global scale; retain the C ABI's unscaled behavior.
        const patch = "native/patches/mlx-c-0.32.3.patch";
        const patch_dir = try std.fmt.allocPrint(git.allocator, "--directory={s}", .{bridge_source});
        const compatibility = std.mem.eql(u8, record.object.get("python").?.object.get("mlx").?.string, "0.32.3");
        if (compatibility) {
            _ = try git.output(&.{ "apply", "--check", patch_dir, patch });
            _ = try git.output(&.{ "apply", patch_dir, patch });
        }
        var restored = false;
        defer if (!restored) {
            if (compatibility) _ = git.output(&.{ "apply", "--reverse", patch_dir, patch }) catch |err| failed: {
                std.debug.print("Could not remove MLX-C compatibility patch: {s}\n", .{@errorName(err)});
                break :failed "";
            };
        };
        try command(git.io, &.{ "cmake", "-S", bridge_source, "-B", bridge_build, "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_OSX_DEPLOYMENT_TARGET=26.2", "-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON", "-DCMAKE_INSTALL_RPATH=@loader_path", "-DBUILD_SHARED_LIBS=ON", "-DMLX_C_USE_SYSTEM_MLX=ON", "-DMLX_C_BUILD_EXAMPLES=OFF", prefix, mlx_prefix });
        try command(git.io, &.{ "cmake", "--build", bridge_build, "--parallel", jobs });
        try command(git.io, &.{ "cmake", "--install", bridge_build });
        if (compatibility) _ = try git.output(&.{ "apply", "--reverse", patch_dir, patch });
        restored = true;
        try record.object.put(git.allocator, "mlx_revision", .{ .string = revision });
        try record.object.put(git.allocator, "mlx_c_revision", .{ .string = bridge_revision });
        try @import("native_install.zig").record(git.allocator, git.io, "build/mlx", .mlx, revision, bridge_revision);
    }
}

fn alignDependencies(git: Git, tip: []const u8) !void {
    try command(git.io, &.{ ".venv/bin/python", "tools/native_runtime.py", "--upstream-ref", tip, "--resolve" });
    const bytes = try std.Io.Dir.cwd().readFileAlloc(git.io, "build/native-dependencies-resolved.json", git.allocator, .limited(16384));
    const parsed = try std.json.parseFromSlice(std.json.Value, git.allocator, bytes, .{});
    var record = parsed.value;
    try buildDependencies(git, &record, record.object.get("rebuild_jpeg").?.bool, record.object.get("rebuild_mlx").?.bool, "4");
    _ = record.object.swapRemove("rebuild_jpeg");
    _ = record.object.swapRemove("rebuild_mlx");
    const content = try std.json.Stringify.valueAlloc(git.allocator, record, .{ .whitespace = .indent_2 });
    const file = try std.Io.Dir.cwd().createFile(git.io, "native/dependencies.json", .{});
    defer file.close(git.io);
    try file.writeStreamingAll(git.io, content);
    try file.writeStreamingAll(git.io, "\n");
    try command(git.io, &.{ ".venv/bin/python", "tools/native_runtime.py" });
    try command(git.io, &.{ ".venv/bin/python", "tools/export_native_kernels.py" });
    try command(git.io, &.{ ".zig-toolchain/zig", "build", "test", "test-prefill", "test-variants", "test-metal", "test-models", "test-vision", "-Doptimize=safe", "-j1" });
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.arena.allocator();
    const args = try init.minimal.args.toSlice(allocator);
    const options = Options.parse(args[1..]) catch |err| {
        std.debug.print("{s}\n{s}", .{ @errorName(err), usage });
        return err;
    };
    if (options.help) {
        std.debug.print("{s}", .{usage});
        return;
    }
    const git = Git{ .allocator = allocator, .io = init.io };
    if (!options.check and !options.continuing) {
        try validateWorktree(try git.output(&.{ "branch", "--show-current" }), try git.output(&.{ "status", "--porcelain", "--untracked-files=all" }));
        try validateIdle(git, false);
    }
    const remote = try upstreamRemote(git, options.remote);
    const main_ref = try std.fmt.allocPrint(allocator, "refs/remotes/{s}/main", .{remote});
    const zig_ref = try std.fmt.allocPrint(allocator, "refs/remotes/{s}/zig", .{remote});
    const main_fetch = try std.fmt.allocPrint(allocator, "refs/heads/main:{s}", .{main_ref});
    const zig_fetch = try std.fmt.allocPrint(allocator, "refs/heads/zig:{s}", .{zig_ref});
    if (!options.continuing) _ = try git.output(&.{ "fetch", "--no-tags", "--no-recurse-submodules", remote, main_fetch, zig_fetch });
    const tip = if (options.continuing) try pendingMerge(git, zig_ref, main_ref) else try git.output(&.{ "rev-parse", main_ref });
    const missing = try git.output(&.{ "rev-list", "--count", try std.fmt.allocPrint(allocator, "HEAD..{s}", .{main_ref}) });
    const zig_missing = try git.output(&.{ "rev-list", "--count", try std.fmt.allocPrint(allocator, "{s}..{s}", .{ zig_ref, main_ref }) });
    std.debug.print("TensorFold main merge target: {s}\nCommits missing from zig: {s}; current branch: {s}\n", .{ tip, zig_missing, missing });
    const dependency_diff = try git.output(&.{ "diff", "HEAD", tip, "--", "pyproject.toml", "uv.lock", "requirements*.txt", "poetry.lock", "setup.cfg", "setup.py" });
    if (dependency_diff.len > 0) std.debug.print("Upstream dependency changes:\n{s}\n", .{dependency_diff});
    if (options.check) return command(init.io, &.{ ".venv/bin/python", "tools/native_runtime.py", "--upstream-ref", tip });
    if (!options.continuing and !try prepareMerge(git, zig_ref, tip)) {
        std.debug.print("Current branch already contains TensorFold main; checking dependency parity.\n", .{});
        return command(init.io, &.{ ".venv/bin/python", "tools/native_runtime.py", "--upstream-ref", tip });
    }
    errdefer std.debug.print("Sync verification stopped; the merge remains local and uncommitted. Inspect git status, resolve the failure and rerun with --continue, or run git merge --abort. No refs were pushed.\n", .{});
    try command(init.io, &.{ ".zig-toolchain/zig", "build", "check-upstream-coverage", "-j1" });
    try alignDependencies(git, tip);
    std.debug.print("Local main merge and dependency checks passed. Review staged and unstaged changes, commit with a GitHub noreply identity, and open a PR against TensorFold zig. No refs were pushed.\n", .{});
}

test "sync rejects uncommitted work and detached HEAD" {
    try validateWorktree("feat/zig", "");
    try std.testing.expectError(error.DetachedHead, validateWorktree("", ""));
    try std.testing.expectError(error.ProtectedBranch, validateWorktree("main", ""));
    try std.testing.expectError(error.ProtectedBranch, validateWorktree("zig", ""));
    for ([_][]const u8{ " M native/lanes.zig", "M  build.zig", "?? new.zig", "UU native/main.zig" }) |status| {
        try std.testing.expectError(error.UncommittedChanges, validateWorktree("feat/zig", status));
    }
}

test "sync options select a remote and resume only the pending merge" {
    const options = try Options.parse(&.{ "--remote", "tensorfold", "--continue" });
    try std.testing.expectEqualStrings("tensorfold", options.remote.?);
    try std.testing.expect(options.continuing);
    try std.testing.expectError(error.MissingRemote, Options.parse(&.{"--remote"}));
    try std.testing.expectError(error.InvalidRemote, Options.parse(&.{ "--remote", "--check" }));
    try std.testing.expectError(error.DuplicateRemote, Options.parse(&.{ "--remote", "a", "--remote", "b" }));
    try std.testing.expectError(error.ConflictingOptions, Options.parse(&.{ "--check", "--continue" }));
    try std.testing.expectError(error.UnknownOption, Options.parse(&.{"--push"}));
}

fn testRepository(allocator: std.mem.Allocator, dir: std.Io.Dir) !Git {
    const git = Git{ .allocator = allocator, .io = std.testing.io, .cwd = .{ .dir = dir } };
    _ = try git.output(&.{ "init", "--quiet", "--initial-branch=main" });
    _ = try git.output(&.{ "config", "user.name", "Sync Test" });
    _ = try git.output(&.{ "config", "user.email", "sync-test@users.noreply.github.com" });
    _ = try git.output(&.{ "config", "commit.gpgSign", "false" });
    _ = try git.output(&.{ "config", "core.hooksPath", "/dev/null" });
    try dir.writeFile(git.io, .{ .sub_path = "shared", .data = "base\n" });
    _ = try git.output(&.{ "add", "shared" });
    _ = try git.output(&.{ "commit", "--quiet", "-m", "base" });
    _ = try git.output(&.{ "branch", "zig" });
    _ = try git.output(&.{ "remote", "add", "tensorfold", upstream_url });
    return git;
}

test "upstream discovery works in the main repository and contributor checkouts" {
    var dir = std.testing.tmpDir(.{});
    defer dir.cleanup();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const git = try testRepository(arena.allocator(), dir.dir);
    try std.testing.expectEqualStrings("tensorfold", try upstreamRemote(git, null));
    _ = try git.output(&.{ "remote", "rename", "tensorfold", "origin" });
    try std.testing.expectEqualStrings("origin", try upstreamRemote(git, null));
    _ = try git.output(&.{ "remote", "rename", "origin", "upstream" });
    _ = try git.output(&.{ "remote", "add", "origin", "git@github.com:contributor/TensorFold.git" });
    try std.testing.expectEqualStrings("upstream", try upstreamRemote(git, null));
    try std.testing.expectError(error.UnexpectedRemote, upstreamRemote(git, "origin"));
    _ = try git.output(&.{ "remote", "add", "duplicate", upstream_url });
    try std.testing.expectError(error.AmbiguousUpstreamRemote, upstreamRemote(git, null));
    try std.testing.expectEqualStrings("upstream", try upstreamRemote(git, "upstream"));
    _ = try git.output(&.{ "config", "--add", "remote.upstream.url", "git@github.com:other/TensorFold.git" });
    try std.testing.expectError(error.UnexpectedRemote, upstreamRemote(git, "upstream"));
    _ = try git.output(&.{ "remote", "remove", "upstream" });
    _ = try git.output(&.{ "remote", "remove", "duplicate" });
    try std.testing.expectError(error.UpstreamRemoteMissing, upstreamRemote(git, null));
}

test "main integration keeps HEAD and protected refs unchanged until review" {
    var dir = std.testing.tmpDir(.{});
    defer dir.cleanup();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const git = try testRepository(arena.allocator(), dir.dir);
    try dir.dir.writeFile(git.io, .{ .sub_path = "main-only", .data = "upstream\n" });
    _ = try git.output(&.{ "add", "main-only" });
    _ = try git.output(&.{ "commit", "--quiet", "-m", "main update" });
    const main_tip = try git.output(&.{ "rev-parse", "main" });
    _ = try git.output(&.{ "update-ref", "refs/remotes/tensorfold/main", main_tip });
    _ = try git.output(&.{ "switch", "zig" });
    try dir.dir.writeFile(git.io, .{ .sub_path = "zig-only", .data = "native\n" });
    _ = try git.output(&.{ "add", "zig-only" });
    _ = try git.output(&.{ "commit", "--quiet", "-m", "zig implementation" });
    const zig_tip = try git.output(&.{ "rev-parse", "zig" });
    _ = try git.output(&.{ "update-ref", "refs/remotes/tensorfold/zig", zig_tip });
    _ = try git.output(&.{ "switch", "-c", "review-sync" });
    try std.testing.expect(try prepareMerge(git, "refs/remotes/tensorfold/zig", main_tip));
    try std.testing.expectEqualStrings(zig_tip, try git.output(&.{ "rev-parse", "HEAD" }));
    try std.testing.expectEqualStrings(main_tip, try git.output(&.{ "rev-parse", "main" }));
    try std.testing.expectEqualStrings(zig_tip, try git.output(&.{ "rev-parse", "zig" }));
    try std.testing.expectEqualStrings(main_tip, try pendingMerge(git, "refs/remotes/tensorfold/zig", "refs/remotes/tensorfold/main"));
    try std.testing.expectError(error.GitOperationInProgress, validateIdle(git, false));
    try std.testing.expectError(error.UncommittedChanges, prepareMerge(git, "refs/remotes/tensorfold/zig", main_tip));
    _ = try git.output(&.{ "commit", "--quiet", "-m", "reviewed main integration" });
    try std.testing.expect(!try prepareMerge(git, "refs/remotes/tensorfold/zig", main_tip));
    try std.testing.expect(try git.ancestor(main_tip, "HEAD"));
    try std.testing.expectEqualStrings(main_tip, try git.output(&.{ "rev-parse", "main" }));
    try std.testing.expectEqualStrings(zig_tip, try git.output(&.{ "rev-parse", "zig" }));
}

test "conflicts can resume at their original target or abort without losing history" {
    var dir = std.testing.tmpDir(.{});
    defer dir.cleanup();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const git = try testRepository(arena.allocator(), dir.dir);
    try dir.dir.writeFile(git.io, .{ .sub_path = "shared", .data = "main\n" });
    _ = try git.output(&.{ "commit", "--quiet", "-am", "main change" });
    const main_tip = try git.output(&.{ "rev-parse", "main" });
    _ = try git.output(&.{ "update-ref", "refs/remotes/tensorfold/main", main_tip });
    _ = try git.output(&.{ "switch", "zig" });
    try dir.dir.writeFile(git.io, .{ .sub_path = "shared", .data = "zig\n" });
    _ = try git.output(&.{ "commit", "--quiet", "-am", "zig change" });
    const zig_tip = try git.output(&.{ "rev-parse", "zig" });
    _ = try git.output(&.{ "update-ref", "refs/remotes/tensorfold/zig", zig_tip });
    _ = try git.output(&.{ "switch", "-c", "review-sync" });
    try std.testing.expectError(error.GitCommandFailed, prepareMerge(git, "refs/remotes/tensorfold/zig", main_tip));
    try std.testing.expectError(error.UnresolvedConflicts, pendingMerge(git, "refs/remotes/tensorfold/zig", "refs/remotes/tensorfold/main"));
    try std.testing.expectEqualStrings(zig_tip, try git.output(&.{ "rev-parse", "HEAD" }));
    try dir.dir.writeFile(git.io, .{ .sub_path = "shared", .data = "resolved\n" });
    _ = try git.output(&.{ "add", "shared" });
    // Main may advance during review; resume must still verify MERGE_HEAD.
    _ = try git.output(&.{ "update-ref", "refs/remotes/tensorfold/main", zig_tip });
    try std.testing.expectError(error.UnexpectedMergeTarget, pendingMerge(git, "refs/remotes/tensorfold/zig", "refs/remotes/tensorfold/main"));
    const main_tree = try git.output(&.{ "rev-parse", "main^{tree}" });
    const advanced_main = try git.output(&.{ "commit-tree", main_tree, "-p", main_tip, "-m", "main advance" });
    _ = try git.output(&.{ "update-ref", "refs/remotes/tensorfold/main", advanced_main });
    try std.testing.expectEqualStrings(main_tip, try pendingMerge(git, "refs/remotes/tensorfold/zig", "refs/remotes/tensorfold/main"));
    _ = try git.output(&.{ "merge", "--abort" });
    try std.testing.expectEqualStrings(zig_tip, try git.output(&.{ "rev-parse", "HEAD" }));
    try std.testing.expectEqualStrings("", try git.output(&.{ "status", "--porcelain" }));
}

test "sync refuses outdated bases and unfinished operations even with a clean index" {
    var dir = std.testing.tmpDir(.{});
    defer dir.cleanup();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const git = try testRepository(arena.allocator(), dir.dir);
    const base = try git.output(&.{ "rev-parse", "HEAD" });
    _ = try git.output(&.{ "branch", "review-sync" });
    _ = try git.output(&.{ "switch", "zig" });
    try dir.dir.writeFile(git.io, .{ .sub_path = "zig-only", .data = "new history\n" });
    _ = try git.output(&.{ "add", "zig-only" });
    _ = try git.output(&.{ "commit", "--quiet", "-m", "zig update" });
    _ = try git.output(&.{ "update-ref", "refs/remotes/tensorfold/zig", "HEAD" });
    _ = try git.output(&.{ "switch", "review-sync" });
    try std.testing.expectError(error.ZigBaseOutdated, prepareMerge(git, "refs/remotes/tensorfold/zig", base));
    try std.testing.expectEqualStrings(base, try git.output(&.{ "rev-parse", "HEAD" }));
    for ([_][]const u8{ "MERGE_HEAD", "CHERRY_PICK_HEAD", "REVERT_HEAD", "BISECT_LOG" }) |marker| {
        const path = try std.fmt.allocPrint(arena.allocator(), ".git/{s}", .{marker});
        try dir.dir.writeFile(git.io, .{ .sub_path = path, .data = base });
        try std.testing.expectError(error.GitOperationInProgress, validateIdle(git, false));
        try dir.dir.deleteFile(git.io, path);
    }
    for ([_][]const u8{ "rebase-apply", "rebase-merge", "sequencer" }) |marker| {
        const path = try std.fmt.allocPrint(arena.allocator(), ".git/{s}", .{marker});
        try dir.dir.createDir(git.io, path, .default_dir);
        try std.testing.expectError(error.GitOperationInProgress, validateIdle(git, false));
        try dir.dir.deleteDir(git.io, path);
    }
}

test "sync accepts only the intended SSH remotes" {
    try validateRemote(upstream_url, upstream_url);
    for ([_][]const u8{ "https://github.com/ashhart/TensorFold.git", "git@github.com:someone/TensorFold.git", upstream_url ++ "\n" ++ upstream_url }) |url| {
        try std.testing.expectError(error.UnexpectedRemote, validateRemote(url, upstream_url));
    }
}
