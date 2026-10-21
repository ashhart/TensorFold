//! Local CPU hashing measurements use generated files and explicit cache directories, without a model or GPU.
const std = @import("std");
const sha = @import("runtime_sha256.zig");
const Content = @import("checkpoint_identity.zig").Content;
const fixture = @import("digest_fixture.zig");

fn output(text: []const u8) !void {
    if (std.c.write(1, text.ptr, text.len) != text.len) return error.Output;
}

fn mode(name: []const u8) !sha.Mode {
    if (std.mem.eql(u8, name, "software")) return .software;
    if (std.mem.eql(u8, name, "auto")) return .auto;
    if (std.mem.eql(u8, name, "hardware")) return .hardware;
    return error.Arguments;
}

pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    if (args.len < 3) return error.Arguments;
    var line: [1024]u8 = undefined;
    if (std.mem.eql(u8, args[1], "sha") and args.len == 3) {
        const bytes = try a.alloc(u8, 4 << 20);
        for (bytes, 0..) |*b, i| b.* = @truncate(i *% 17 + 11);
        const standard = std.mem.eql(u8, args[2], "standard");
        const backend = if (standard) sha.Backend.software else try sha.select(try mode(args[2]));
        var fast = sha.Hash.init(backend);
        var old = std.crypto.hash.sha2.Sha256.init(.{});
        const started = std.Io.Clock.awake.now(init.io);
        for (0..32) |_| {
            std.mem.doNotOptimizeAway(bytes.ptr);
            if (standard) old.update(bytes) else fast.update(bytes);
        }
        const digest = if (standard) old.finalResult() else fast.finalResult();
        const ns = started.durationTo(std.Io.Clock.awake.now(init.io)).toNanoseconds();
        const total = bytes.len * 32;
        try output(try std.fmt.bufPrint(&line, "{{\"kind\":\"sha\",\"mode\":\"{s}\",\"backend\":\"{s}\",\"bytes\":{d},\"ns\":{d},\"MB_per_s\":{d:.3},\"digest\":\"{s}\"}}\n", .{ args[2], @tagName(backend), total, ns, @as(f64, @floatFromInt(total)) * 1000 / @as(f64, @floatFromInt(ns)), &std.fmt.bytesToHex(digest, .lower) }));
    } else if (std.mem.eql(u8, args[1], "make") and args.len == 3) {
        var dir = try std.Io.Dir.cwd().createDirPathOpen(init.io, args[2], .{});
        defer dir.close(init.io);
        try fixture.write(init.gpa, init.io, dir, 8, 16 << 20);
    } else if (std.mem.eql(u8, args[1], "files") and args.len == 6) {
        const chosen = try mode(args[4]);
        const cap = try std.fmt.parseInt(usize, args[5], 10);
        if (cap < 1 or cap > Content.max_workers) return error.Arguments;
        const workers = Content.workerCount(std.Thread.getCpuCount() catch 1, 2, 8, cap);
        const started = std.Io.Clock.awake.now(init.io);
        const result = try (try Content.startWithOptions(init.gpa, init.io, args[2], false, args[3], .{ .storage_workers = cap, .hash_mode = chosen })).finishWithStats();
        const ns = started.durationTo(std.Io.Clock.awake.now(init.io)).toNanoseconds();
        try output(try std.fmt.bufPrint(&line, "{{\"kind\":\"files\",\"mode\":\"{s}\",\"backend\":\"{s}\",\"workers\":{d},\"bytes\":{d},\"cache_hits\":{d},\"ns\":{d},\"MB_per_s\":{d:.3},\"digest\":\"{s}\"}}\n", .{ args[4], @tagName(try sha.select(chosen)), workers, result.bytes_read, result.cache_hits, ns, @as(f64, @floatFromInt(result.bytes_read)) * 1000 / @as(f64, @floatFromInt(ns)), &std.fmt.bytesToHex(result.digest, .lower) }));
    } else return error.Arguments;
}
