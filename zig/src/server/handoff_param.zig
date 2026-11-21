//! kv_transfer_params.mcdma_handoff on a completion request (MCDMA's kv-handoff.md): export this prompt's state.
const std = @import("std");
const json = @import("json");
const Handoff = @import("engine_api").LaneHost.Export.Handoff;

/// The request's handoff, or null without one; a malformed one is refused with `why`.
pub fn parse(f: json.Value, why: *[]const u8) error{BadHandoff}!?Handoff {
    const params = f.field("kv_transfer_params") orelse return null;
    const h = params.field("mcdma_handoff") orelse return null;
    if (h != .object) return bad(why, "kv_transfer_params.mcdma_handoff: expected an object with id and export_from");
    const text = (h.field("id") orelse return bad(why, "kv_transfer_params.mcdma_handoff.id: missing")).str() orelse
        return bad(why, "kv_transfer_params.mcdma_handoff.id: expected 32 hex digits");
    var id: [16]u8 = undefined;
    if (text.len != 32) return bad(why, "kv_transfer_params.mcdma_handoff.id: expected 32 hex digits");
    _ = std.fmt.hexToBytes(&id, text) catch return bad(why, "kv_transfer_params.mcdma_handoff.id: expected 32 hex digits");
    var from: u64 = 0;
    if (h.field("export_from")) |v| {
        if (v != .int) return bad(why, "kv_transfer_params.mcdma_handoff.export_from: expected an integer");
        from = std.fmt.parseInt(u64, v.int, 10) catch return bad(why, "kv_transfer_params.mcdma_handoff.export_from: expected an integer");
    }
    if (from != 0) return bad(why, "kv_transfer_params.mcdma_handoff.export_from: this server exports from token 0 only");
    return .{ .id = id, .export_from = from };
}

fn bad(why: *[]const u8, text: []const u8) error{BadHandoff} {
    why.* = text;
    return error.BadHandoff;
}

/// The route a decoder reads and releases a held export on: `/v1/handoff/<32 hex digits>`.
pub const route_prefix = "/v1/handoff/";

/// The id a hand-off route names, or null for any other path: exactly the prefix and 32 hex digits, nothing after.
pub fn routeId(route: []const u8) ?[16]u8 {
    if (!std.mem.startsWith(u8, route, route_prefix)) return null;
    const hex = route[route_prefix.len..];
    if (hex.len != 32) return null;
    var id: [16]u8 = undefined;
    _ = std.fmt.hexToBytes(&id, hex) catch return null;
    return id;
}

test "a hand-off route names exactly 32 hex digits and nothing else" {
    const id = routeId("/v1/handoff/00112233445566778899aabbccddeeff").?;
    try std.testing.expectEqual(@as(u8, 0xff), id[15]);
    for ([_][]const u8{ "/v1/handoff/", "/v1/handoff", "/v1/handoff/0011", "/v1/handoff/00112233445566778899aabbccddeeff0", "/v1/handoff/00112233445566778899aabbccddeefg", "/v1/handoff/../etc/passwd", "/v1/handoff/00112233445566778899aabbccddeeff/x", "/handoff/00112233445566778899aabbccddeeff", "/v1/handoff/00112233445566778899aabbccddee%66" }) |bad_route|
        try std.testing.expectEqual(@as(?[16]u8, null), routeId(bad_route));
}

test "a handoff is read from kv_transfer_params, and a malformed one is refused by its field" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var why: []const u8 = "";
    const body = try json.newObject(a);
    try std.testing.expectEqual(@as(?Handoff, null), try parse(.{ .object = body }, &why));
    const params = try json.newObject(a);
    const handoff = try json.newObject(a);
    try body.put(a, "kv_transfer_params", .{ .object = params });
    try params.put(a, "mcdma_handoff", .{ .object = handoff });
    try handoff.put(a, "id", .{ .string = "00112233445566778899aabbccddeeff" });
    try handoff.put(a, "export_from", .{ .int = "0" });
    const got = (try parse(.{ .object = body }, &why)).?;
    try std.testing.expectEqual(@as(u8, 0xee), got.id[14]);
    try handoff.put(a, "id", .{ .string = "0011" });
    try std.testing.expectError(error.BadHandoff, parse(.{ .object = body }, &why));
    try std.testing.expectEqualStrings("kv_transfer_params.mcdma_handoff.id: expected 32 hex digits", why);
    try handoff.put(a, "id", .{ .string = "00112233445566778899aabbccddeeff" });
    try handoff.put(a, "export_from", .{ .int = "512" });
    try std.testing.expectError(error.BadHandoff, parse(.{ .object = body }, &why));
}

test "a handoff that is not an object, has no id, or has a non-integer export_from is refused by that field" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var why: []const u8 = "";
    const body = try json.newObject(a);
    const params = try json.newObject(a);
    try body.put(a, "kv_transfer_params", .{ .object = params });
    try params.put(a, "mcdma_handoff", .{ .string = "00112233445566778899aabbccddeeff" });
    try std.testing.expectError(error.BadHandoff, parse(.{ .object = body }, &why));
    try std.testing.expectEqualStrings("kv_transfer_params.mcdma_handoff: expected an object with id and export_from", why);
    const handoff = try json.newObject(a);
    try params.put(a, "mcdma_handoff", .{ .object = handoff });
    try std.testing.expectError(error.BadHandoff, parse(.{ .object = body }, &why));
    try std.testing.expectEqualStrings("kv_transfer_params.mcdma_handoff.id: missing", why);
    try handoff.put(a, "id", .{ .int = "7" });
    try std.testing.expectError(error.BadHandoff, parse(.{ .object = body }, &why));
    try std.testing.expectEqualStrings("kv_transfer_params.mcdma_handoff.id: expected 32 hex digits", why);
    try handoff.put(a, "id", .{ .string = "00112233445566778899aabbccddeeff" });
    for ([_]json.Value{ .{ .string = "0" }, .{ .float = 0.5 }, .{ .bool = false } }) |v| {
        try handoff.put(a, "export_from", v);
        try std.testing.expectError(error.BadHandoff, parse(.{ .object = body }, &why));
        try std.testing.expectEqualStrings("kv_transfer_params.mcdma_handoff.export_from: expected an integer", why);
    }
}
