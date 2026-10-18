//! Identify the OS-supplied runtime compiler deployment without invoking a compiler command.
const std = @import("std");
const mtl = @import("metal");
const f = @import("tensorfold").flashnext_vision_contract;
const accuracy = f.rotary_accuracy;
const objc = mtl.objc;
pub const service = "/System/Library/Frameworks/Metal.framework/Versions/A/XPCServices/MTLCompilerService.xpc";
pub const framework = "/System/Library/PrivateFrameworks/MTLCompiler.framework";

fn version(a: std.mem.Allocator, path: []const u8) ![]const u8 {
    const name = objc.string(path);
    defer objc.release(name);
    const bundle = objc.msg(?objc.Id, objc.class("NSBundle"), "bundleWithPath:", .{name}) orelse return error.UnknownMetalCompiler;
    const key = objc.string("CFBundleVersion");
    defer objc.release(key);
    const value = objc.msg(?objc.Id, bundle, "objectForInfoDictionaryKey:", .{key}) orelse return error.UnknownMetalCompiler;
    const text = std.mem.span(objc.utf8(value));
    if (text.len == 0) return error.UnknownMetalCompiler;
    return a.dupe(u8, text);
}
pub fn current(a: std.mem.Allocator, io: std.Io, device: mtl.Device) !accuracy.Identity {
    const osv = objc.osVersion();
    var build: [128]u8 = undefined;
    var length: usize = build.len;
    if (@import("sys").c.sysctlbyname("kern.osversion", &build, &length, null, 0) != 0) return error.UnknownMetalCompiler;
    var identity: accuracy.Identity = .{
        .compiler_service = try version(a, service),
        .compiler_framework = try version(a, framework),
        .compiler_deployment = undefined,
        .os_version = try std.fmt.allocPrint(a, "{d}.{d}.{d}", .{ osv.major, osv.minor, osv.patch }),
        .os_build = try a.dupe(u8, std.mem.sliceTo(build[0..length], 0)),
        .device_name = try a.dupe(u8, std.mem.span(device.name())),
        .device_generation = device.generation(),
        .language = mtl.CompileOptions.mlx().language,
        .math_mode = @intCast(@backingInt(mtl.CompileOptions.mlx().math_mode)),
        .math_functions = @intCast(@backingInt(mtl.CompileOptions.mlx().math_functions)),
        .sources = undefined,
        .oracle_sources = undefined,
    };
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    for ([_][]const u8{ service ++ "/Contents/Info.plist", service ++ "/Contents/version.plist", framework ++ "/Resources/Info.plist", framework ++ "/Resources/version.plist" }) |path| {
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 20));
        defer a.free(bytes);
        hash.update(path);
        hash.update(bytes);
    }
    var digest: [32]u8 = undefined;
    hash.final(&digest);
    identity.compiler_deployment = std.fmt.bytesToHex(digest, .lower);
    var oracle = std.crypto.hash.sha2.Sha256.init(.{});
    oracle.update(&accuracy.oracleSources());
    for ([_][]const u8{ @embedFile("rotary_deployment.zig"), @embedFile("rotary_sweep_cpu.zig"), @embedFile("rotary_sweep_gpu.zig"), @embedFile("rotary_sweep_norm.zig"), @embedFile("rotary_sweep_main.zig"), @embedFile("metal_flashnext.zig"), @embedFile("metal_flashnext_pool.zig"), @embedFile("rig.zig"), @import("kernel_sources").flashnext_rotary_sweep }) |bytes| oracle.update(bytes);
    oracle.final(&digest);
    identity.oracle_sources = std.fmt.bytesToHex(digest, .lower);
    for ([_]f.Which{ .mrope, .target, .pooled, .absolute }, &identity.sources) |which, *pin| {
        std.crypto.hash.sha2.Sha256.hash(f.source(which), &digest, .{});
        pin.* = std.fmt.bytesToHex(digest, .lower);
    }
    return identity;
}
pub fn load(a: std.mem.Allocator, io: std.Io, path: ?[]const u8, identity: accuracy.Identity) !std.json.Parsed(accuracy.Receipt) {
    const name = path orelse return error.UnrecordedRotaryCompiler;
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, name, a, .limited(8 << 20));
    defer a.free(bytes);
    const receipt = try std.json.parseFromSlice(accuracy.Receipt, a, bytes, .{ .allocate = .alloc_always });
    errdefer receipt.deinit();
    try receipt.value.validate(identity);
    return receipt;
}
