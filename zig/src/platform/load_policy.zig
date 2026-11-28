//! The --confidential loader policy: once confined, libraries load from the engine's folder, system folders or a list.
const std = @import("std");
const builtin = @import("builtin");

/// The fixed system folders a confined macOS process loads from; Linux has none, only `runtime_names`.
pub const system_folders: []const []const u8 = if (builtin.os.tag.isDarwin()) &.{ "/usr/lib/", "/System/Library/" } else &.{};

/// The bare names a confined Linux process may hand the loader: libc, and the CUDA driver, NVML, NCCL and cuBLAS.
pub const runtime_names = [_][]const u8{
    "libc.so.6",         "libcuda.so.1", "libnvidia-ml.so.1", "libnccl.so.2", "libcublas.so.13",
    "libcublasLt.so.13", "libcublas.so", "libcublasLt.so",
};

var confined: std.atomic.Value(bool) = .init(false);
/// How many loads the policy has refused, so a test sees a load it never reached the loader for.
pub var refusals: std.atomic.Value(u32) = .init(0);
var folder_bytes: [2][std.fs.max_path_bytes]u8 = undefined;
var folders: [2][]const u8 = .{ "", "" };

/// From here on every load goes through `allows`: the engine's folder is the executable's, and ../lib beside it.
pub fn confine() void {
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    if (executableDir(&buf)) |dir| {
        folders[0] = copy(0, dir);
        var lib: [std.fs.max_path_bytes]u8 = undefined;
        if (resolve(&lib, &.{ dir, "..", "lib" })) |path| folders[1] = copy(1, path);
    }
    confined.store(true, .release);
}

fn copy(i: usize, path: []const u8) []const u8 {
    @memcpy(folder_bytes[i][0..path.len], path);
    return folder_bytes[i][0..path.len];
}

/// Tests only: back to unconfined, so one test's confinement doesn't refuse the next test's loads.
pub fn release() void {
    confined.store(false, .release);
}

pub fn isConfined() bool {
    return confined.load(.acquire);
}

/// Whether `path` may load now: always outside confidential mode, else as `allowedIn` the engine's folders says.
pub fn allows(path: []const u8) bool {
    if (!isConfined()) return true;
    if (allowedIn(&folders, path)) return true;
    _ = refusals.fetchAdd(1, .monotonic);
    std.log.warn("--confidential loads no library from outside the engine's folder: refused {s}", .{path});
    return false;
}

/// A bare name only from `runtime_names`; a path, after `..` is resolved, only inside `engine` or `system_folders`.
pub fn allowedIn(engine: []const []const u8, path: []const u8) bool {
    if (std.mem.indexOfScalar(u8, path, '/') == null) {
        for (runtime_names) |name| if (std.mem.eql(u8, path, name)) return true;
        return false;
    }
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const at_exe = "@executable_path/";
    const full = if (std.mem.startsWith(u8, path, at_exe))
        resolve(&buf, &.{ engine[0], path[at_exe.len..] })
    else if (path[0] == '/') resolve(&buf, &.{path}) else null; // a relative path would be read from the working dir
    const lexical = full orelse return false;
    var real: [std.fs.max_path_bytes]u8 = undefined;
    const p = target(lexical, &real) orelse lexical; // a symlink is judged by the file it names, not by its own folder
    for (engine) |folder| if (folder.len > 1 and inside(folder, p)) return true;
    for (system_folders) |folder| if (std.mem.startsWith(u8, p, folder)) return true;
    return false;
}

/// The file `path` names after every symlink is followed; null: no such file, which the loader cannot open either.
fn target(path: []const u8, out: *[std.fs.max_path_bytes]u8) ?[]const u8 {
    if (comptime builtin.os.tag == .windows) return null;
    var z: [std.fs.max_path_bytes + 1]u8 = undefined;
    if (path.len >= z.len) return null;
    @memcpy(z[0..path.len], path);
    z[path.len] = 0;
    const r = realpath(@ptrCast(&z), out) orelse return null;
    return std.mem.span(r);
}

fn inside(folder: []const u8, path: []const u8) bool {
    return path.len > folder.len and std.mem.startsWith(u8, path, folder) and path[folder.len] == '/';
}

/// The parts joined with `..` and `.` resolved in text, as one absolute path in `buf`; null: none fits.
fn resolve(buf: []u8, parts: []const []const u8) ?[]const u8 {
    if (parts.len == 0 or parts[0].len == 0 or parts[0][0] != '/') return null;
    var fba: std.heap.FixedBufferAllocator = .init(buf);
    return std.fs.path.resolvePosix(fba.allocator(), parts) catch null;
}

extern "c" fn realpath(noalias name: [*:0]const u8, noalias resolved: [*]u8) ?[*:0]u8;

fn executableDir(buf: []u8) ?[]const u8 {
    if (comptime builtin.os.tag == .linux) {
        const n = std.c.readlink("/proc/self/exe", buf.ptr, buf.len);
        if (n < 0 or n >= buf.len) return null;
        return std.fs.path.dirname(buf[0..@intCast(n)]);
    } else if (comptime builtin.os.tag.isDarwin()) {
        var raw: [std.fs.max_path_bytes:0]u8 = undefined;
        var size: u32 = raw.len;
        if (std.c._NSGetExecutablePath(&raw, &size) != 0) return null;
        const real = realpath(&raw, buf.ptr) orelse return null;
        return std.fs.path.dirname(std.mem.span(real));
    }
    return null;
}
