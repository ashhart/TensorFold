//! Discovery must retain constructor callbacks and destructors until the helper process exits.
const std = @import("std");
const platform = @import("sys");
const libs = platform.optional_libraries;
const fixture = @import("discovery_fixture");
var selected: libs.Id = .verbs;
const Probe = struct {
    pub fn getenv(_: [*:0]const u8) ?[:0]const u8 {
        return null;
    }
    pub const DynLib = struct {
        pub fn open(path: []const u8) !platform.DynLib {
            if (!std.mem.eql(u8, path, libs.spec(selected).names[0])) return error.FileNotFound;
            return platform.DynLib.open(fixture.library);
        }
    };
};
fn child(init: std.process.Init, kind: []const u8, marker: []const u8) !void {
    selected = if (std.mem.eql(u8, kind, "verbs")) .verbs else if (std.mem.eql(u8, kind, "grammar")) .grammar else return error.BadKind;
    if (!libs.found(libs.spec(selected), init.arena.allocator(), Probe, null)) return error.DiscoveryMissedFixture;
    var own = try platform.DynLib.open(fixture.library);
    defer own.close();
    const mark = own.lookup(*const fn ([*:0]const u8) callconv(.c) c_int, "tf_probe_ready") orelse return error.MissingFixtureExport;
    if (mark(try init.arena.allocator().dupeSentinel(u8, marker, 0)) != 0) return error.BadMarkerPath;
}
pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    if (args.len == 4 and std.mem.eql(u8, args[1], "--child")) return child(init, args[2], args[3]);
    if (args.len != 1) return error.BadArguments;
    const exe = try std.process.executablePathAlloc(init.io, a);
    const root = try std.fmt.allocPrint(a, ".tf-probe-exit-{d}", .{std.Io.Clock.awake.now(init.io).toNanoseconds()});
    try std.Io.Dir.cwd().createDir(init.io, root, .default_dir);
    defer std.Io.Dir.cwd().deleteTree(init.io, root) catch {};
    var env: std.process.Environ.Map = .init(init.gpa);
    defer env.deinit();
    if (init.environ_map.get("SystemRoot")) |value| try env.put("SystemRoot", value);
    const absolute = try std.Io.Dir.cwd().realPathFileAlloc(init.io, root, a);
    var failed = false;
    for ([_][]const u8{ "verbs", "grammar" }) |kind| {
        const marker = try std.fs.path.join(a, &.{ absolute, kind });
        const result = try std.process.run(init.gpa, init.io, .{ .argv = &.{ exe, "--child", kind, marker }, .environ_map = &env, .stdout_limit = .limited(1024), .stderr_limit = .limited(4096) });
        defer init.gpa.free(result.stdout);
        defer init.gpa.free(result.stderr);
        if (result.term != .exited or result.term.exited != 0) {
            std.debug.print("discovery {s}: child termination {any}\n", .{ kind, result.term });
            failed = true;
            continue;
        }
        const receipt = try std.Io.Dir.cwd().readFileAlloc(init.io, marker, a, .limited(128));
        if (std.mem.count(u8, receipt, "atexit\n") != 1 or std.mem.count(u8, receipt, "destructor\n") != 1 or std.mem.count(u8, receipt, "thread_local\n") != 1) return error.ExitCallbacksMissing;
        std.debug.print("discovery {s}: retained through TLS, atexit and destructor; child exit 0\n", .{kind});
    }
    if (failed) return error.DiscoveryUnloadedBeforeExit;
}
