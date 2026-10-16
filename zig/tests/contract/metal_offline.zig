//! M1 and M2 thread limits without those chips: native code built for T threads equal to the plain build fits T.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");

const limits = contract.limits;

/// The translator's M1 and M2 targets (phones left out); a family's limit is its lowest.
const m1_archs = [_][]const u8{ "applegpu_g13g", "applegpu_g13s", "applegpu_g13d" };
const m2_archs = [_][]const u8{ "applegpu_g14g", "applegpu_g14s", "applegpu_g14d" };

/// MLX's language version for this macOS, as build.zig's metallib step picks it.
fn std_flag() []const u8 {
    const major = mtl.objc.osVersion().major;
    return if (major >= 27) "-std=metal4.1" else if (major >= 26) "-std=metal4.0" else if (major >= 15) "-std=metal3.2" else "-std=metal3.1";
}

const Ctx = struct { a: std.mem.Allocator, io: std.Io, dir: []const u8 };

fn exec(x: Ctx, argv: []const []const u8) !void {
    const r = try std.process.run(x.a, x.io, .{ .argv = argv });
    defer x.a.free(r.stdout);
    defer x.a.free(r.stderr);
    if (!r.term.success()) {
        std.debug.print("{s} failed: {s}\n", .{ argv[2], r.stderr });
        return error.ToolFailed;
    }
}

/// The source compiled as MLX compiles it at run time, linked into a metallib named by its hash.
fn library(x: Ctx, source: []const u8) ![]const u8 {
    var h = std.hash.Wyhash.init(0);
    h.update(source);
    const stem = try std.fmt.allocPrint(x.a, "{s}/src_{x}", .{ x.dir, h.final() });
    const lib = try std.fmt.allocPrint(x.a, "{s}.metallib", .{stem});
    if (std.Io.Dir.cwd().access(x.io, lib, .{})) |_| return lib else |_| {}
    const src = try std.fmt.allocPrint(x.a, "{s}.metal", .{stem});
    const air = try std.fmt.allocPrint(x.a, "{s}.air", .{stem});
    try std.Io.Dir.cwd().writeFile(x.io, .{ .sub_path = src, .data = source });
    try exec(x, &.{ "xcrun", "-sdk", "macosx", "metal", std_flag(), "-fmetal-math-mode=safe", "-fmetal-math-fp32-functions=fast", "-c", src, "-o", air });
    try exec(x, &.{ "xcrun", "-sdk", "macosx", "metallib", air, "-o", lib });
    return lib;
}

/// A section's bytes in a Mach-O image at `base`, found by section name.
fn section(buf: []const u8, base: usize, name: []const u8) ![]const u8 {
    if (buf.len < base + 32 or std.mem.readInt(u32, buf[base..][0..4], .little) != 0xfeedfacf) return error.NotMachO;
    const ncmds = std.mem.readInt(u32, buf[base + 16 ..][0..4], .little);
    var at = base + 32;
    for (0..ncmds) |_| {
        const cmd = std.mem.readInt(u32, buf[at..][0..4], .little);
        const size = std.mem.readInt(u32, buf[at + 4 ..][0..4], .little);
        if (cmd == 0x19) for (0..std.mem.readInt(u32, buf[at + 64 ..][0..4], .little)) |i| {
            const s = buf[at + 72 + 80 * i ..][0..80];
            if (std.mem.eql(u8, std.mem.sliceTo(s[0..16], 0), name)) {
                const len: usize = @intCast(std.mem.readInt(u64, s[40..48], .little));
                const off: usize = std.mem.readInt(u32, s[48..52], .little);
                return buf[base + off ..][0..len];
            }
        };
        at += size;
    }
    return error.NoSection;
}

/// The native code metal-tt builds for one architecture, asking for `threads` a group (0: no request).
fn native(x: Ctx, lib: []const u8, fun: []const u8, arch: []const u8, threads: u32) ![]const u8 {
    const out = try std.fmt.allocPrint(x.a, "{s}.{s}.{s}.{d}.bin", .{ lib, fun, arch, threads });
    if (std.Io.Dir.cwd().access(x.io, out, .{})) |_| {} else |_| {
        var script: std.ArrayList(u8) = .empty;
        try script.print(x.a, "{{\"libraries\": {{\"paths\": [{{\"path\": \"{s}\", \"label\": \"lib\"}}]}}, \"pipelines\": {{\"compute_pipelines\": [{{\"compute_function\": \"alias:lib#{s}\"", .{ lib, fun });
        if (threads > 0) try script.print(x.a, ", \"max_total_threads_per_threadgroup\": {d}", .{threads});
        try script.appendSlice(x.a, "}]}}");
        const path = try std.fmt.allocPrint(x.a, "{s}.mtlp-json", .{out});
        try std.Io.Dir.cwd().writeFile(x.io, .{ .sub_path = path, .data = script.items });
        var min: [16]u8 = undefined;
        const ver = try std.fmt.bufPrint(&min, "{d}.0", .{mtl.objc.osVersion().major}); // the AIR version std_flag() made
        try exec(x, &.{ "xcrun", "metal-tt", "-arch", arch, "-macos-version-min", ver, "-o", out, lib, path });
    }
    const bytes = try std.Io.Dir.cwd().readFileAlloc(x.io, out, x.a, .limited(1 << 26));
    const outer = try section(bytes, 0, "__compute");
    const inner_at = @intFromPtr(outer.ptr) - @intFromPtr(bytes.ptr);
    return section(bytes, inner_at, "__text");
}

/// The most threads (a multiple of 32) whose build equals the plain one: what the plain pipeline runs at.
fn limitOn(x: Ctx, lib: []const u8, fun: []const u8, arch: []const u8) !u32 {
    const plain = try native(x, lib, fun, arch, 0);
    if (std.mem.eql(u8, plain, try native(x, lib, fun, arch, 1024))) return 1024;
    var lo: u32 = 0;
    var hi: u32 = 33;
    while (lo + 1 < hi) {
        const mid = (lo + hi) / 2;
        if (std.mem.eql(u8, plain, try native(x, lib, fun, arch, 32 * mid))) lo = mid else hi = mid;
    }
    return 32 * lo;
}

fn familyLimit(x: Ctx, lib: []const u8, fun: []const u8, archs: []const []const u8) !u32 {
    var least: u32 = 1024;
    for (archs) |arch| least = @min(least, try limitOn(x, lib, fun, arch));
    return least;
}

/// Each launch's M1 and M2 limits compiled here: every one must fit, and no record may exceed what compiled here.
pub fn run(a: std.mem.Allocator, io: std.Io, dir: []const u8, list: []const limits.MetalLaunch, comptime launchOf: anytype, device: mtl.Device) !usize {
    try std.Io.Dir.cwd().createDirPath(io, dir);
    const x = Ctx{ .a = a, .io = io, .dir = dir };
    var n: usize = 0;
    for (list) |d| {
        const l = try launchOf(a, device, d);
        const control = std.mem.startsWith(u8, d.name, "control");
        const safe = d.adapts or d.reserved >= d.threads or (!d.offline and d.threads <= 256);
        const got = blk: {
            const lib = library(x, l.pre_m5) catch |err| break :blk err;
            const m1 = familyLimit(x, lib, l.function, &m1_archs) catch |err| break :blk err;
            const m2 = familyLimit(x, lib, l.function, &m2_archs) catch |err| break :blk err;
            break :blk [2]u32{ m1, m2 };
        };
        const m = got catch |err| {
            const word = if (control) "RED " else if (safe) "N/A " else "FAIL";
            if (!control and !safe) n += 1;
            std.debug.print("{s} offline {s}: no M1/M2 build ({t}); {d} threads a group\n", .{ word, d.name, err, d.threads });
            continue;
        };
        const fits = (m[0] >= d.threads and m[1] >= d.threads) or d.reserved >= d.threads or d.adapts;
        const current = (d.m1 <= m[0] and d.m2 <= m[1] and d.m1 > 0) or d.adapts or d.reserved >= d.threads; // a record: the least any toolchain compiled
        if (!control and (!fits or !current)) n += 1;
        const word = if (control) (if (fits) "MISSED" else "RED ") else if (fits and current) "PASS" else "FAIL";
        std.debug.print("{s} offline {s}: {d} threads; M1 pipelines take {d}, M2 {d} (recorded {d}, {d})\n", .{ word, d.name, d.threads, m[0], m[1], d.m1, d.m2 });
    }
    return n;
}
