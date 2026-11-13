//! A configured exact browser origin may preflight; credentials are still checked on its actual request.
const std = @import("std");
const http = @import("http_conn.zig");
const Allocator = std.mem.Allocator;
const methods = "GET, POST, PUT, DELETE, OPTIONS";
const headers = "authorization, x-api-key, content-type, anthropic-version, anthropic-beta, anthropic-dangerous-direct-browser-access";
const sdk_prefix = "x-stainless-"; // the OpenAI and Anthropic SDKs' platform and retry headers

fn member(list: []const u8, value: []const u8) bool {
    var it = std.mem.splitScalar(u8, list, ',');
    while (it.next()) |item| if (std.ascii.eqlIgnoreCase(std.mem.trim(u8, item, " \t"), value)) return true;
    return false;
}

/// An SDK header past the fixed list: x-stainless- and a name of letters, digits, '-' and '_' (echoed lowercased).
fn sdkHeader(name: []const u8) bool {
    if (name.len <= sdk_prefix.len or !std.ascii.startsWithIgnoreCase(name, sdk_prefix)) return false;
    for (name) |ch| if (!std.ascii.isAlphanumeric(ch) and ch != '-' and ch != '_') return false;
    return true;
}

pub fn valid(c: *const http.Conn) bool {
    if (c.cors_origin == null or c.headerCount("Access-Control-Request-Method") != 1 or
        c.headerCount("Access-Control-Request-Headers") > 1) return false;
    if (!member(methods, c.header("Access-Control-Request-Method").?)) return false;
    if (c.header("Access-Control-Request-Headers")) |asked| {
        var it = std.mem.splitScalar(u8, asked, ',');
        while (it.next()) |item| {
            const name = std.mem.trim(u8, item, " \t");
            if (!member(headers, name) and !sdkHeader(name)) return false;
        }
    }
    return true;
}

pub fn preflight(c: *http.Conn, a: Allocator) void {
    c.close = true; // preflights read no body; the next authenticated request uses its own connection
    if (!valid(c)) {
        c.sendError(a, 403, "CORS preflight is not allowed for this origin, method or headers", null) catch {};
        return;
    }
    var allowed: std.ArrayList(u8) = .empty;
    defer allowed.deinit(a);
    allowed.appendSlice(a, headers) catch return;
    if (c.header("Access-Control-Request-Headers")) |asked| {
        var it = std.mem.splitScalar(u8, asked, ',');
        while (it.next()) |item| {
            const name = std.mem.trim(u8, item, " \t");
            if (member(headers, name)) continue;
            allowed.appendSlice(a, ", ") catch return;
            for (name) |ch| allowed.append(a, std.ascii.toLower(ch)) catch return;
        }
    }
    c.startResponse(204, null) catch return;
    c.addHeader("Access-Control-Allow-Methods", methods) catch return;
    c.addHeader("Access-Control-Allow-Headers", allowed.items) catch return;
    c.addHeader("Access-Control-Max-Age", "600") catch return;
    c.addHeader("Connection", "close") catch return; // a 204 carries no Content-Length
    c.finish("") catch {};
}
