const std = @import("std");

const upstream_url = "git@github.com:ashhart/TensorFold.git";
const origin_url = "git@github.com:CerebralCoding/TensorFold.git";

const Git = struct {
    allocator: std.mem.Allocator,
    io: std.Io,

    fn run(g: Git, args: []const []const u8) !std.process.RunResult {
        const argv = try g.allocator.alloc([]const u8, args.len + 1);
        defer g.allocator.free(argv);
        argv[0] = "git";
        @memcpy(argv[1..], args);
        return std.process.run(g.allocator, g.io, .{ .argv = argv });
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
    if (status.len != 0) return error.UncommittedChanges;
}

fn command(io: std.Io, argv: []const []const u8) !void {
    var child = try std.process.spawn(io, .{ .argv = argv });
    if (!(try child.wait(io)).success()) return error.CommandFailed;
}

fn source(git: Git, dir: []const u8, url: []const u8, revision: []const u8) ![]const u8 {
    const exists = if (std.Io.Dir.cwd().access(git.io, dir, .{})) true else |err| switch (err) {
        error.FileNotFound => false,
        else => return err,
    };
    if (!exists) _ = try git.output(&.{ "clone", "--no-checkout", url, dir });
    try validateRemote(try git.output(&.{ "-C", dir, "remote", "get-url", "origin" }), url);
    if (exists and (try git.output(&.{ "-C", dir, "status", "--porcelain" })).len != 0) return error.DependencySourceDirty;
    _ = try git.output(&.{ "-C", dir, "fetch", "--depth=1", "origin", revision });
    _ = try git.output(&.{ "-C", dir, "checkout", "--detach", "FETCH_HEAD" });
    return git.output(&.{ "-C", dir, "rev-parse", "HEAD" });
}

fn alignDependencies(git: Git, tip: []const u8) !void {
    try command(git.io, &.{ ".venv/bin/python", "tools/native_runtime.py", "--upstream-ref", tip, "--resolve" });
    const bytes = try std.Io.Dir.cwd().readFileAlloc(git.io, "build/native-dependencies-resolved.json", git.allocator, .limited(16384));
    const parsed = try std.json.parseFromSlice(std.json.Value, git.allocator, bytes, .{});
    var record = parsed.value;
    if (record.object.get("rebuild_jpeg").?.bool) {
        try std.Io.Dir.cwd().createDirPath(git.io, "build/deps");
        _ = try source(git, "build/deps/libjpeg-turbo", "git@github.com:libjpeg-turbo/libjpeg-turbo.git", record.object.get("jpeg_version").?.string);
        const root = try std.process.currentPathAlloc(git.io, git.allocator);
        const prefix = try std.fmt.allocPrint(git.allocator, "-DCMAKE_INSTALL_PREFIX={s}/build/jpeg", .{root});
        try command(git.io, &.{ "cmake", "-S", "build/deps/libjpeg-turbo", "-B", "build/jpeg-build", "-DCMAKE_BUILD_TYPE=Release", "-DENABLE_SHARED=OFF", "-DENABLE_STATIC=ON", "-DWITH_TOOLS=OFF", "-DWITH_TESTS=OFF", prefix });
        try command(git.io, &.{ "cmake", "--build", "build/jpeg-build", "--parallel", "4" });
        try command(git.io, &.{ "cmake", "--install", "build/jpeg-build" });
    }
    _ = record.object.swapRemove("rebuild_jpeg");
    if (record.object.get("rebuild_mlx").?.bool) {
        try std.Io.Dir.cwd().createDirPath(git.io, "build/deps");
        const revision = try source(git, "build/deps/mlx", "git@github.com:ml-explore/mlx.git", record.object.get("mlx_revision").?.string);
        _ = try source(git, "build/deps/mlx-c", "git@github.com:ml-explore/mlx-c.git", record.object.get("mlx_c_revision").?.string);
        _ = try source(git, "build/deps/fmt", "git@github.com:fmtlib/fmt.git", "12.1.0");
        const root = try std.process.currentPathAlloc(git.io, git.allocator);
        const prefix = try std.fmt.allocPrint(git.allocator, "-DCMAKE_INSTALL_PREFIX={s}/build/mlx", .{root});
        const mlx_prefix = try std.fmt.allocPrint(git.allocator, "-DCMAKE_PREFIX_PATH={s}/build/mlx", .{root});
        const fmt = try std.fmt.allocPrint(git.allocator, "-DFETCHCONTENT_SOURCE_DIR_FMT={s}/build/deps/fmt", .{root});
        try command(git.io, &.{ "cmake", "-S", "build/deps/mlx", "-B", "build/mlx-build", "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_OSX_DEPLOYMENT_TARGET=26.2", "-DBUILD_SHARED_LIBS=ON", "-DMLX_BUILD_TESTS=OFF", "-DMLX_BUILD_EXAMPLES=OFF", prefix, fmt });
        try command(git.io, &.{ "cmake", "--build", "build/mlx-build", "--parallel", "4" });
        try command(git.io, &.{ "cmake", "--install", "build/mlx-build" });
        try command(git.io, &.{ "cmake", "-S", "build/deps/mlx-c", "-B", "build/mlxc-build", "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_OSX_DEPLOYMENT_TARGET=26.2", "-DBUILD_SHARED_LIBS=ON", "-DMLX_C_USE_SYSTEM_MLX=ON", "-DMLX_C_BUILD_EXAMPLES=OFF", prefix, mlx_prefix });
        try command(git.io, &.{ "cmake", "--build", "build/mlxc-build", "--parallel", "4" });
        try command(git.io, &.{ "cmake", "--install", "build/mlxc-build" });
        try record.object.put(git.allocator, "mlx_revision", .{ .string = revision });
    }
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
    const check = args.len == 2 and std.mem.eql(u8, args[1], "--check");
    if (!check and args.len != 1) return error.InvalidArguments;
    const git = Git{ .allocator = allocator, .io = init.io };
    try validateRemote(try git.output(&.{ "remote", "get-url", "origin" }), origin_url);
    try validateRemote(try git.output(&.{ "remote", "get-url", "--push", "--all", "origin" }), origin_url);
    const branch = try git.output(&.{ "branch", "--show-current" });
    if (!check) {
        validateWorktree(branch, try git.output(&.{ "status", "--porcelain", "--untracked-files=all" })) catch |err| {
            std.debug.print("Commit or stash your work and finish any Git operation before syncing.\n", .{});
            return err;
        };
    }
    const remotes = try git.output(&.{"remote"});
    var names = std.mem.splitScalar(u8, remotes, '\n');
    var registered = false;
    while (names.next()) |name| {
        if (std.mem.eql(u8, name, "upstream")) registered = true;
    }
    if (!registered) _ = try git.output(&.{ "remote", "add", "upstream", upstream_url });
    try validateRemote(try git.output(&.{ "remote", "get-url", "upstream" }), upstream_url);
    _ = try git.output(&.{ "fetch", "--no-tags", "upstream", "main" });
    _ = try git.output(&.{ "fetch", "--no-tags", "origin", "main" });
    const tip = try git.output(&.{ "rev-parse", "upstream/main" });
    const missing = try git.output(&.{ "rev-list", "--count", "HEAD..upstream/main" });
    const fork_missing = try git.output(&.{ "rev-list", "--count", "origin/main..upstream/main" });
    std.debug.print("Upstream main: {s}\nCommits missing from fork main: {s}; current branch: {s}\n", .{ tip, fork_missing, missing });
    const dependency_diff = try git.output(&.{ "diff", "HEAD", tip, "--", "pyproject.toml", "uv.lock", "requirements*.txt", "poetry.lock", "setup.cfg", "setup.py" });
    if (dependency_diff.len > 0) std.debug.print("Upstream dependency changes:\n{s}\n", .{dependency_diff});
    if (check) return command(init.io, &.{ ".venv/bin/python", "tools/native_runtime.py", "--upstream-ref", tip });
    if (!try git.ancestor("origin/main", tip)) return error.ForkMainDiverged;
    if (!try git.ancestor("main", tip)) return error.LocalMainDiverged;
    if (std.mem.eql(u8, branch, "main")) {
        _ = try git.output(&.{ "merge", "--ff-only", tip });
    } else {
        // Git refuses to move main if it is checked out in another worktree.
        _ = try git.output(&.{ "branch", "-f", "main", tip });
        _ = git.output(&.{ "rebase", tip }) catch |err| {
            std.debug.print("Resolve the rebase with git rebase --continue, or restore it with git rebase --abort. Then run sync-upstream again.\n", .{});
            return err;
        };
    }
    try command(init.io, &.{ ".zig-toolchain/zig", "build", "check-upstream-coverage", "-j1" });
    try alignDependencies(git, tip);
    const refspec = try std.fmt.allocPrint(allocator, "{s}:refs/heads/main", .{tip});
    // A concurrent or divergent update is rejected by this ordinary push.
    _ = try git.output(&.{ "push", "origin", refspec });
    std.debug.print("Fork main synced; {s} includes upstream main and dependency parity checks passed. Review any dependency/kernel changes and push the feature branch explicitly when ready.\n", .{branch});
}

test "sync rejects uncommitted work and detached HEAD" {
    try validateWorktree("feat/zig", "");
    try std.testing.expectError(error.DetachedHead, validateWorktree("", ""));
    for ([_][]const u8{ " M native/lanes.zig", "M  build.zig", "?? new.zig", "UU native/main.zig" }) |status| {
        try std.testing.expectError(error.UncommittedChanges, validateWorktree("feat/zig", status));
    }
}

test "sync accepts only the intended SSH remotes" {
    try validateRemote(origin_url, origin_url);
    try validateRemote(upstream_url, upstream_url);
    for ([_][]const u8{ "https://github.com/CerebralCoding/TensorFold.git", upstream_url, "git@github.com:someone/TensorFold.git", origin_url ++ "\n" ++ upstream_url }) |url| {
        try std.testing.expectError(error.UnexpectedRemote, validateRemote(url, origin_url));
    }
}
