//! The saved profile: a round trip through the file, the rescan rules, and what it never records.
const std = @import("std");
const profile = @import("profile.zig");
const scan = @import("scan.zig");
const support = @import("support.zig");
const machine = @import("machine.zig");
const parse = @import("parse.zig");
const fx = @import("test_fixtures.zig");
const t = std.testing;

const cuda: support.Offer = .{ .version = "1.2.8", .backends = &.{"cuda"}, .families = &.{.{ .model_type = "nemotron_h", .formats = &.{"mlx-q4g64"} }}, .kernel_sms = &support.release_sms };

var driver_calls: usize = 0;
fn gb10Driver(a: std.mem.Allocator) ?[]const machine.Gpu {
    driver_calls += 1;
    return a.dupe(machine.Gpu, &.{.{ .vendor = .nvidia, .name = "NVIDIA GB10", .sm = 121, .unified = true, .source = .driver }}) catch null;
}
fn silentDriver(_: std.mem.Allocator) ?[]const machine.Gpu {
    driver_calls += 1;
    return null;
}

/// A fresh folder name under the working directory, deleted by the caller.
fn folder(a: std.mem.Allocator) ![]const u8 {
    return std.fmt.allocPrint(a, ".tf-device-test-{d}", .{std.Io.Clock.awake.now(t.io).toNanoseconds()});
}

test "device profile: a scan saves, reads back whole, and stays until the hardware or the binary changes" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const root = try folder(a);
    defer std.Io.Dir.cwd().deleteTree(t.io, root) catch {};
    const file = try std.fs.path.join(a, &.{ root, ".tensorfold", "device.json" });
    const quick = try parse.linux(a, "aarch64", fx.gb10);
    const bin: scan.Binary = .{ .version = "1.2.8", .offer = cuda, .driver = gb10Driver };
    driver_calls = 0;
    const first = try scan.current(a, t.io, file, quick, bin, "2026-11-13T01:00:00Z", false, false);
    try t.expectEqual(@as(?profile.Due, .missing), first.scanned);
    try t.expect(first.saved);
    try t.expectEqual(profile.Scan.quick, first.doc.scan);
    try t.expectEqual(@as(usize, 0), driver_calls); // a summary never asks the driver
    const back = profile.load(a, t.io, file).?;
    try t.expectEqualStrings("2026-11-13T01:00:00Z", back.scanned_at);
    try t.expectEqualStrings("1.2.8", back.binary_version);
    try t.expectEqualStrings(&machine.fingerprint(quick), back.fingerprint);
    try t.expectEqualStrings("NVIDIA GB10", back.machine.gpus[0].name);
    try t.expectEqual(@as(u32, 121), back.machine.gpus[0].sm);
    try t.expectEqual(machine.Source.table, back.machine.gpus[0].source);
    try t.expectEqualStrings("cuda", back.support.backend.?);
    try t.expectEqual(quick.ram, back.machine.ram);

    const again = try scan.current(a, t.io, file, quick, bin, "2026-11-13T02:00:00Z", false, false);
    try t.expectEqual(@as(?profile.Due, null), again.scanned);
    try t.expectEqualStrings("2026-11-13T01:00:00Z", again.doc.scanned_at);
    try t.expect(scan.fresh(a, t.io, file, quick, "1.2.8") != null);

    // `tensorfold device` upgrades a quick profile by asking the driver, once
    const device = try scan.current(a, t.io, file, quick, bin, "2026-11-13T03:00:00Z", false, true);
    try t.expectEqual(@as(?profile.Due, .quick), device.scanned);
    try t.expectEqual(profile.Scan.full, device.doc.scan);
    try t.expectEqual(machine.Source.driver, device.doc.machine.gpus[0].source);
    try t.expectEqual(@as(usize, 1), driver_calls);
    const settled = try scan.current(a, t.io, file, quick, bin, "2026-11-13T04:00:00Z", false, true);
    try t.expectEqual(@as(?profile.Due, null), settled.scanned);
    try t.expectEqual(@as(usize, 1), driver_calls);
    try t.expect(scan.fresh(a, t.io, file, quick, "1.2.8") != null); // the driver's facts keep the quick fingerprint

    try t.expectEqual(@as(?profile.Due, .asked), (try scan.current(a, t.io, file, quick, bin, "x", true, true)).scanned);
    const newer: scan.Binary = .{ .version = "1.2.9", .offer = cuda, .driver = gb10Driver };
    try t.expect(scan.fresh(a, t.io, file, quick, "1.2.9") == null);
    try t.expectEqual(@as(?profile.Due, .binary), (try scan.current(a, t.io, file, quick, newer, "x", false, false)).scanned);
    var more = quick;
    more.ram *= 2;
    try t.expect(scan.fresh(a, t.io, file, more, "1.2.9") == null);
    try t.expectEqual(@as(?profile.Due, .hardware), (try scan.current(a, t.io, file, more, newer, "x", false, false)).scanned);
}

test "device profile: a driver that does not answer keeps the OS's listing and a quick scan" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const quick = try parse.linux(a, "x86_64", fx.rtx3090);
    const got = try scan.current(a, t.io, null, quick, .{ .version = "1.2.8", .offer = cuda, .driver = silentDriver }, "x", false, true);
    try t.expectEqual(profile.Scan.quick, got.doc.scan);
    try t.expect(!got.saved); // no home: nothing written
    try t.expectEqualStrings("NVIDIA GeForce RTX 3090", got.doc.machine.gpus[0].name);
    const mac = try scan.current(a, t.io, null, try parse.macos(a, "aarch64", fx.m3_ultra), .{ .version = "1.2.8", .offer = cuda }, "x", false, true);
    try t.expectEqual(profile.Scan.full, mac.doc.scan); // sysctl is a Mac's whole story
}

test "device profile: the driver's GPUs replace the OS's NVIDIA entries in ordinal order and keep the rest" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const listed = [_]machine.Gpu{ .{ .vendor = .intel, .name = "Intel GPU (8086:a780)" }, .{ .vendor = .nvidia, .name = "NVIDIA GeForce RTX 3090", .sm = 86, .source = .table } };
    const driver = [_]machine.Gpu{ .{ .vendor = .nvidia, .name = "NVIDIA GeForce RTX 5090", .sm = 120, .source = .driver }, .{ .vendor = .nvidia, .name = "NVIDIA GeForce RTX 3090", .sm = 86, .source = .driver } };
    const merged = try scan.merge(a, &listed, &driver);
    try t.expectEqual(@as(usize, 3), merged.len);
    try t.expectEqualStrings("NVIDIA GeForce RTX 5090", merged[0].name);
    try t.expectEqual(machine.Vendor.intel, merged[2].vendor);
}

test "device profile: the file holds no host or user name, serial or path, and refuses another schema" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const got = try scan.current(a, t.io, null, try parse.linux(a, "aarch64", fx.gb10), .{ .version = "1.2.8", .offer = cuda }, "2026-11-13T01:00:00Z", false, false);
    var out: std.Io.Writer.Allocating = .init(a);
    try profile.write(&out.writer, got.doc);
    const text = out.written();
    for ([_][]const u8{ "/proc", "/sys", "/Users", "/home", "GPU-", "UUID", "000f:01:00.0" }) |bad| try t.expect(std.mem.indexOf(u8, text, bad) == null);
    try t.expect(std.mem.indexOf(u8, text, "\"schema\": 1") != null);
    try t.expect(profile.parse(a, text) != null);
    const other = try std.mem.replaceOwned(u8, a, text, "\"schema\": 1", "\"schema\": 2");
    try t.expect(profile.parse(a, other) == null);
    try t.expect(profile.parse(a, "{not json") == null);
    var env: std.process.Environ.Map = .init(a);
    try t.expect(try profile.path(a, &env) == null);
    try env.put("USERPROFILE", "D:\\tf");
    try t.expect(std.mem.startsWith(u8, (try profile.path(a, &env)).?, "D:\\tf"));
    try env.put("HOME", "/srv/tf");
    const home = (try profile.path(a, &env)).?;
    try t.expect(std.mem.startsWith(u8, home, "/srv/tf") and std.mem.endsWith(u8, home, "device.json"));
    try t.expect(std.mem.indexOf(u8, home, ".tensorfold") != null);
    var buf: [20]u8 = undefined;
    try t.expectEqualStrings("2026-11-13T01:22:01Z", profile.stamp(&buf, 1794532921));
}
