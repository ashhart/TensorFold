const std = @import("std");
const builtin = @import("builtin");
const sync = @import("sync_upstream.zig");

const usage =
    \\Usage: .zig-toolchain/zig run tools/setup_native.zig -- [options]
    \\Run from the TensorFold repository root on Apple Silicon macOS.
    \\Builds pinned MLX/MLX-C and JPEG locally, installs Python test dependencies,
    \\then builds the native executable and runs host checks. No model downloads.
    \\  --python PATH  Python 3.11+ used when creating .venv (default: python3)
    \\  --jobs N       Parallel CMake jobs (default: 4; Zig builds use -j1)
    \\  --check        Check prerequisites and installed dependencies without installing
    \\  --dry-run      Print the setup plan without running commands
    \\  --help         Show this help
    \\Existing dependency checkouts must be clean and use the expected SSH remote.
    \\Re-running setup incrementally rebuilds dependencies at the checked-in pins.
    \\
;

const Options = struct {
    python: []const u8 = "python3",
    jobs: []const u8 = "4",
    check: bool = false,
    dry_run: bool = false,
    help: bool = false,

    fn parse(args: []const []const u8) !Options {
        var result: Options = .{};
        var i: usize = 0;
        while (i < args.len) : (i += 1) {
            const arg = args[i];
            if (std.mem.eql(u8, arg, "--help")) {
                result.help = true;
            } else if (std.mem.eql(u8, arg, "--check")) {
                result.check = true;
            } else if (std.mem.eql(u8, arg, "--dry-run")) {
                result.dry_run = true;
            } else if (std.mem.eql(u8, arg, "--python") or std.mem.eql(u8, arg, "--jobs")) {
                i += 1;
                if (i == args.len or args[i].len == 0) return error.MissingOptionValue;
                if (std.mem.eql(u8, arg, "--python")) {
                    if (args[i][0] == '-') return error.InvalidPython;
                    result.python = args[i];
                } else {
                    const n = std.fmt.parseInt(u16, args[i], 10) catch return error.InvalidJobs;
                    if (n == 0) return error.InvalidJobs;
                    result.jobs = args[i];
                }
            } else return error.UnknownOption;
        }
        if (result.check and result.dry_run) return error.ConflictingOptions;
        return result;
    }
};

fn output(init: std.process.Init, argv: []const []const u8) ![]const u8 {
    const result = std.process.run(init.arena.allocator(), init.io, .{ .argv = argv }) catch |err| {
        std.debug.print("Cannot run {s}: {s}. See native/README.md prerequisites.\n", .{ argv[0], @errorName(err) });
        return err;
    };
    if (!result.term.success()) {
        std.debug.print("Prerequisite command failed: {s}\n{s}{s}\nSee native/README.md prerequisites.\n", .{ argv[0], result.stdout, result.stderr });
        return error.PrerequisiteFailed;
    }
    return std.mem.trim(u8, result.stdout, " \r\n");
}

fn atLeast(text: []const u8, major: u32, minor: u32) bool {
    var parts = std.mem.tokenizeAny(u8, text, ". \r\n");
    const actual_major = std.fmt.parseInt(u32, parts.next() orelse return false, 10) catch return false;
    const actual_minor = std.fmt.parseInt(u32, parts.next() orelse return false, 10) catch return false;
    return actual_major > major or (actual_major == major and actual_minor >= minor);
}

fn prerequisites(init: std.process.Init, options: Options) !void {
    if (builtin.os.tag != .macos or builtin.cpu.arch != .aarch64) return error.AppleSiliconMacRequired;
    const cwd = std.Io.Dir.cwd();
    const expected = std.mem.trim(u8, try cwd.readFileAlloc(init.io, ".zig-version", init.arena.allocator(), .limited(256)), " \r\n");
    if (!std.mem.eql(u8, expected, builtin.zig_version_string) or
        !std.mem.eql(u8, expected, try output(init, &.{ ".zig-toolchain/zig", "version" })))
    {
        std.debug.print("Use the exact nightly in .zig-version: bash scripts/fetch-zig.sh\n", .{});
        return error.ZigVersionMismatch;
    }
    if (!atLeast(try output(init, &.{ "sw_vers", "-productVersion" }), 26, 2)) return error.MacOS26_2Required;
    if (!atLeast(try output(init, &.{ "xcrun", "--sdk", "macosx", "--show-sdk-version" }), 26, 2)) return error.MacOS26_2SDKRequired;
    _ = try output(init, &.{ "xcrun", "--sdk", "macosx", "metal", "--version" });
    _ = try output(init, &.{ "git", "--version" });
    const cmake = try output(init, &.{ "cmake", "--version" });
    if (!std.mem.startsWith(u8, cmake, "cmake version ") or !atLeast(cmake[14..], 3, 25)) return error.CMake3_25Required;
    const python = if (cwd.access(init.io, ".venv/bin/python", .{})) ".venv/bin/python" else |_| options.python;
    const version = try output(init, &.{ python, "--version" });
    if (!std.mem.startsWith(u8, version, "Python ") or !atLeast(version[7..], 3, 11)) return error.Python3_11Required;
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
    const cwd = std.Io.Dir.cwd();
    const bytes = cwd.readFileAlloc(init.io, "native/dependencies.json", allocator, .limited(16384)) catch |err| {
        std.debug.print("Run setup from the TensorFold repository root.\n", .{});
        return err;
    };
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, bytes, .{});
    var record = parsed.value;
    const pins = record.object.get("python").?.object;
    if (options.dry_run) {
        std.debug.print("Check Apple Silicon, macOS/SDK >=26.2, Metal compiler, CMake >=3.25, Git, Python >=3.11 and pinned Zig.\nCreate .venv using {s} if absent; install editable .[test,vision] with:\n", .{options.python});
        var entries = pins.iterator();
        while (entries.next()) |entry| std.debug.print("  {s}=={s}\n", .{ entry.key_ptr.*, entry.value_ptr.string });
        std.debug.print("Build via SSH checkouts, CMake Release, {s} jobs:\n  MLX {s}\n  MLX-C {s}\n  fmt 12.1.0\n  libjpeg-turbo {s}\nInstall into build/mlx and build/jpeg.\nCheck dependency and exported kernel parity.\nBuild ReleaseSafe (-j1), then run host, checkpoint-file and setup/sync guard tests.\nNo model downloads, Git branch changes, pushes or system installs.\n", .{ options.jobs, record.object.get("mlx_revision").?.string, record.object.get("mlx_c_revision").?.string, record.object.get("jpeg_version").?.string });
        return;
    }
    try prerequisites(init, options);
    if (options.check) {
        try sync.command(init.io, &.{ ".venv/bin/python", "tools/native_runtime.py" });
        std.debug.print("Native build prerequisites and installed dependency versions passed.\n", .{});
        return;
    }
    if (cwd.access(init.io, ".venv", .{})) {} else |err| switch (err) {
        error.FileNotFound => try sync.command(init.io, &.{ options.python, "-m", "venv", ".venv" }),
        else => return err,
    }
    var pip: std.array_list.Managed([]const u8) = .init(allocator);
    try pip.appendSlice(&.{ ".venv/bin/python", "-m", "pip", "install", "--editable", ".[test,vision]" });
    var entries = pins.iterator();
    while (entries.next()) |entry| try pip.append(try std.fmt.allocPrint(allocator, "{s}=={s}", .{ entry.key_ptr.*, entry.value_ptr.string }));
    try sync.command(init.io, pip.items);
    try sync.buildDependencies(.{ .allocator = allocator, .io = init.io }, &record, true, true, options.jobs);
    try sync.command(init.io, &.{ ".venv/bin/python", "tools/native_runtime.py" });
    try sync.command(init.io, &.{ ".venv/bin/python", "tools/export_native_kernels.py", "--check" });
    try sync.command(init.io, &.{ ".zig-toolchain/zig", "build", "-Doptimize=safe", "-j1" });
    try sync.command(init.io, &.{ ".zig-toolchain/zig", "build", "test", "test-checkpoint-files", "test-setup", "test-sync-upstream", "-Doptimize=safe", "-j1" });
    std.debug.print("Setup passed. Native CLI: zig-out/bin/tensorfold\nNext: native/README.md for Metal tests and existing model paths.\n", .{});
}

test "setup accepts explicit resource controls and rejects ambiguous options" {
    const options = try Options.parse(&.{ "--python", "/path with spaces/python3", "--jobs", "2", "--dry-run" });
    try std.testing.expectEqualStrings("/path with spaces/python3", options.python);
    try std.testing.expectEqualStrings("2", options.jobs);
    try std.testing.expect(options.dry_run);
    try std.testing.expectError(error.InvalidJobs, Options.parse(&.{ "--jobs", "0" }));
    try std.testing.expectError(error.InvalidJobs, Options.parse(&.{ "--jobs", "-1" }));
    try std.testing.expectError(error.MissingOptionValue, Options.parse(&.{"--python"}));
    try std.testing.expectError(error.ConflictingOptions, Options.parse(&.{ "--dry-run", "--check" }));
    try std.testing.expectError(error.UnknownOption, Options.parse(&.{"--download-models"}));
}

test "prerequisite version comparison accepts patch releases and future majors" {
    try std.testing.expect(atLeast("26.2", 26, 2));
    try std.testing.expect(atLeast("26.3.1\n", 26, 2));
    try std.testing.expect(atLeast("4.0.0", 3, 25));
    try std.testing.expect(!atLeast("3.9.10", 3, 11));
    try std.testing.expect(!atLeast("unknown", 3, 11));
}
