//! Opt-in key-file scopes: a restricted key reaches exact routes only; a key without scopes keeps every route.
const std = @import("std");
const json = @import("json");
const routePath = @import("auth.zig").routePath;

pub const Scopes = packed struct(u8) {
    inference: bool = false,
    read: bool = false,
    admin: bool = false,
    padding: u5 = 0,

    pub fn eql(a: Scopes, b: Scopes) bool {
        return @as(u8, @bitCast(a)) == @as(u8, @bitCast(b));
    }
};

/// Every key without a scope list, from a flag, the environment or a plain file line.
pub const full: Scopes = .{ .admin = true };
pub const Label = struct { name: []const u8, scopes: Scopes };
pub const ParseError = error{ InvalidLabel, InvalidScopes };

/// A file label: `name`, or `name[inference]`, `name[read]`, `name[inference,read]`, `name[admin]`.
pub fn parseLabel(raw: []const u8) ParseError!Label {
    const bracket = std.mem.indexOfScalar(u8, raw, '[') orelse {
        if (!validLabel(raw)) return error.InvalidLabel;
        return .{ .name = raw, .scopes = full };
    };
    const name = raw[0..bracket];
    if (!validLabel(name)) return error.InvalidLabel;
    if (raw.len < bracket + 3 or raw[raw.len - 1] != ']') return error.InvalidScopes;
    var scopes: Scopes = .{};
    var members = std.mem.splitScalar(u8, raw[bracket + 1 .. raw.len - 1], ',');
    while (members.next()) |member| {
        const text = std.mem.trim(u8, member, " \t");
        const bit: Scopes = if (std.mem.eql(u8, text, "inference"))
            .{ .inference = true }
        else if (std.mem.eql(u8, text, "read"))
            .{ .read = true }
        else if (std.mem.eql(u8, text, "admin"))
            full
        else
            return error.InvalidScopes;
        const before: u8 = @bitCast(scopes);
        const added: u8 = @bitCast(bit);
        if (before & added != 0) return error.InvalidScopes;
        scopes = @bitCast(before | added);
    }
    if (scopes.admin and (scopes.inference or scopes.read)) return error.InvalidScopes;
    return .{ .name = name, .scopes = scopes };
}

/// 1 to 64 letters, digits, dots, underscores or hyphens.
fn validLabel(label: []const u8) bool {
    if (label.len == 0 or label.len > 64) return false;
    for (label) |ch| if (!(std.ascii.isAlphanumeric(ch) or ch == '_' or ch == '.' or ch == '-')) return false;
    return true;
}

const listing = [_][]const u8{ "/models", "/v1/models", "/v1/voices" };
const reading = [_][]const u8{ "/memory", "/v1/memory", "/metrics", "/v1/metrics", "/stats", "/dashboard", "/v1/benchmark/manifest" };
const inferring = [_][]const u8{
    "/chat/completions", "/v1/chat/completions", "/completions",           "/v1/completions",
    "/messages",         "/v1/messages",         "/messages/count_tokens", "/v1/messages/count_tokens",
    "/responses",        "/v1/responses",        "/tokenize",              "/v1/tokenize",
    "/detokenize",       "/v1/detokenize",       "/v1/audio/speech",       "/v1/embeddings",
    "/decisions",        "/v1/decisions",        "/systemone",             "/v1/systemone",
};

/// A gated request a key may make: the route as the dispatcher reads it, matched exactly, never by suffix.
pub fn allows(scopes: Scopes, method: []const u8, raw_path: []const u8) bool {
    if (scopes.admin) return true;
    const route = routePath(raw_path);
    if (std.mem.eql(u8, method, "GET")) {
        if ((scopes.inference or scopes.read) and oneOf(route, &listing)) return true;
        return scopes.read and oneOf(route, &reading);
    }
    return scopes.inference and std.mem.eql(u8, method, "POST") and oneOf(route, &inferring);
}

fn oneOf(route: []const u8, names: []const []const u8) bool {
    for (names) |name| if (std.mem.eql(u8, route, name)) return true;
    return false;
}

pub const Denial = enum { route, response_history, response_storage };

/// The stored Responses are the server's, so a restricted key's request must be stateless: store false, no parent.
pub fn responseDenial(scopes: Scopes, body: json.Value) ?Denial {
    if (scopes.admin or body != .object) return null;
    if (body.get("previous_response_id")) |parent| if (parent != .null) return .response_history;
    const store = body.get("store") orelse return .response_storage;
    if (store != .bool or store.bool) return .response_storage;
    return null;
}
