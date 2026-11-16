//! The connection flags: the open-connection limit, one address's share under pressure, the trusted proxies.
const std = @import("std");
const cli = @import("cli.zig");
const admission = @import("http_admission.zig");

pub const flags = [_]cli.Flag{
    .{ .name = "--max-connections", .native = true },
    .{ .name = "--max-connections-per-address", .native = true },
    .{ .name = "--trusted-proxy", .kind = .append, .native = true },
};

/// At most this many --trusted-proxy addresses, each compared on every accept.
pub const max_trusted = 64;

/// --max-connections and the per-address rule, as the serve parser reads them.
pub const Args = struct { max: u32 = admission.default_limit, peering: admission.Peering = .{} };

/// True when `name` is a connection flag and `value` valid for it; false for any other flag.
pub fn apply(a: std.mem.Allocator, out: *Args, name: []const u8, value: []const u8, usage: *cli.Usage) !bool {
    const limit = std.mem.eql(u8, name, "--max-connections");
    const share = std.mem.eql(u8, name, "--max-connections-per-address");
    if (limit or share) {
        const least: u32 = if (limit) 1 else 0;
        const parsed = std.fmt.parseInt(u32, value, 10) catch admission.Peers.capacity + 1;
        if (parsed < least or parsed > admission.Peers.capacity) {
            usage.message = try std.fmt.allocPrint(a, "argument {s}: expected {d} through {d}: '{s}'", .{ name, least, admission.Peers.capacity, value });
            return error.Usage;
        }
        if (limit) out.max = parsed else out.peering.per_address = parsed;
        return true;
    }
    if (!std.mem.eql(u8, name, "--trusted-proxy")) return false;
    const ip = std.Io.net.IpAddress.parse(value, 0) catch {
        usage.message = try std.fmt.allocPrint(a, "argument --trusted-proxy: expected an IPv4 or IPv6 address without a port: '{s}'", .{value});
        return error.Usage;
    };
    if (out.peering.trusted.len >= max_trusted) {
        usage.message = try std.fmt.allocPrint(a, "at most {d} --trusted-proxy entries are allowed", .{max_trusted});
        return error.Usage;
    }
    out.peering.trusted = try std.mem.concat(a, std.Io.net.IpAddress, &.{ out.peering.trusted, &.{ip} });
    return true;
}

test "connection CLI: the limit, the per-address share and trusted proxies parse, bounded, and are advertised" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var usage: cli.Usage = .{};
    const defaults = try cli.parse(a, &.{"model"}, &usage);
    try std.testing.expectEqual(@as(u32, 128), defaults.connections.max);
    try std.testing.expectEqual(@as(?u32, null), defaults.connections.peering.per_address);
    try std.testing.expectEqual(@as(usize, 0), defaults.connections.peering.trusted.len);
    const args = try cli.parse(a, &.{ "model", "--max-connections=512", "--max-connections-per-address", "0", "--trusted-proxy", "172.17.0.1", "--trusted-proxy=fd00::1" }, &usage);
    try std.testing.expectEqual(@as(u32, 512), args.connections.max);
    try std.testing.expectEqual(@as(?u32, 0), args.connections.peering.per_address);
    try std.testing.expectEqual(@as(usize, 2), args.connections.peering.trusted.len);
    try std.testing.expect(args.connections.peering.trusts(.{ .ip4 = .{ .bytes = .{ 172, 17, 0, 1 }, .port = 51000 } }));
    for ([_][]const u8{ "0", "1025", "-1", "nan", "" }) |v| try std.testing.expectError(error.Usage, cli.parse(a, &.{ "model", "--max-connections", v }, &usage));
    for ([_][]const u8{ "1025", "-1", "1.5" }) |v| try std.testing.expectError(error.Usage, cli.parse(a, &.{ "model", "--max-connections-per-address", v }, &usage));
    for ([_][]const u8{ "proxy.local", "10.0.0.1:80", "" }) |v| try std.testing.expectError(error.Usage, cli.parse(a, &.{ "model", "--trusted-proxy", v }, &usage));
    for (flags) |flag| {
        var present = false;
        for (cli.flags) |known| if (std.mem.eql(u8, known.name, flag.name)) {
            present = known.native;
        };
        try std.testing.expect(present);
    }
}
