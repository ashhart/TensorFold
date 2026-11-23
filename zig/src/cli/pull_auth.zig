//! The Hugging Face token for gated and private repos: where pull reads it, the one origin it goes to, the refusals.
const std = @import("std");
const Allocator = std.mem.Allocator;
const http = std.http;
const hub = @import("hub.zig");
const environment = @import("sys").environment;

/// The hub pull talks to when HF_ENDPOINT names none.
pub const default_endpoint = "https://huggingface.co";
/// The most bytes a token may have.
const max_token = 4096;

/// A token and where it came from: an environment variable's name or a file's path, never the token itself.
pub const Token = struct { value: []const u8, source: []const u8 };

/// The hub endpoint pull talks to, and the token that only requests to that endpoint's own origin carry.
pub const Auth = struct {
    /// What request URLs are built on: HF_ENDPOINT, else the public hub.
    endpoint: []const u8 = default_endpoint,
    /// "Bearer TOKEN", or null when no token is set.
    bearer: ?[]const u8 = null,
    /// The token's source, which a refusal names in the token's place.
    source: []const u8 = "",

    /// The hub from HF_ENDPOINT and the first token set; null once `err_out` names a token pull cannot send.
    pub fn init(a: Allocator, io: std.Io, env: ?*const std.process.Environ.Map, err_out: *std.Io.Writer) !?Auth {
        var auth: Auth = .{ .endpoint = hub.envValue(env, "HF_ENDPOINT") orelse default_endpoint };
        var problem: []const u8 = "";
        const found = token(a, io, env, &problem) catch |e| switch (e) {
            error.BadToken => {
                try err_out.print("{s}\n", .{problem});
                return null;
            },
            else => return e,
        };
        if (found) |t| {
            auth.bearer = try std.fmt.allocPrint(a, "Bearer {s}", .{t.value});
            auth.source = t.source;
        }
        return auth;
    }

    /// Whether `uri` is on the endpoint's own origin: the same scheme, host and port.
    pub fn owns(auth: Auth, uri: std.Uri) bool {
        var mine: [255]u8 = undefined;
        var theirs: [255]u8 = undefined;
        const own = origin(std.Uri.parse(auth.endpoint) catch return false, &mine) orelse return false;
        const other = origin(uri, &theirs) orelse return false;
        return std.ascii.eqlIgnoreCase(own.scheme, other.scheme) and std.ascii.eqlIgnoreCase(own.host, other.host) and own.port == other.port;
    }

    /// The Authorization a request to `uri` sends: the token on the endpoint's origin, the client's default elsewhere.
    pub fn header(auth: Auth, uri: std.Uri) http.Client.Request.Headers.Value {
        const bearer = auth.bearer orelse return .default;
        return if (auth.owns(uri)) .{ .override = bearer } else .default;
    }

    /// Whether server-controlled text contains this hub's bearer, directly or percent-encoded.
    pub fn reflects(auth: Auth, text: []const u8) bool {
        const bearer = auth.bearer orelse return false;
        const secret = if (std.mem.startsWith(u8, bearer, "Bearer ")) bearer[7..] else bearer;
        return containsCredential(text, secret);
    }

    /// One bodiless request into `req`, redirects followed here so each hop decides the token anew; see `header`.
    pub fn send(auth: Auth, client: *http.Client, req: *http.Client.Request, method: http.Method, uri: std.Uri, options: http.Client.RequestOptions, buf: []u8) !http.Client.Response {
        var o = options;
        o.redirect_behavior = .unhandled;
        var left = options.redirect_behavior;
        var at = uri;
        var rest = buf;
        while (true) {
            try @import("pull_route.zig").note(client.io, at);
            o.headers.authorization = auth.header(at);
            req.* = try client.request(method, at, o);
            errdefer req.deinit();
            try req.sendBodiless();
            const response = try req.receiveHead(&.{});
            const status = response.head.status;
            if ((status == .unauthorized or status == .forbidden) and auth.owns(at)) return error.HubDenied;
            if (status.class() != .redirect or left == .unhandled) return response;
            if (left == .not_allowed) return error.TooManyHttpRedirects;
            const location = response.head.location orelse return error.HttpRedirectLocationMissing;
            if (location.len > rest.len) return error.HttpRedirectLocationOversize;
            @memcpy(rest[0..location.len], location); // the head's bytes go when the connection does
            if (auth.reflects(location)) return error.HubCredentialRedirect;
            at = at.resolveInPlace(location.len, &rest) catch return error.HttpRedirectLocationInvalid;
            left.subtractOne();
            req.deinit();
        }
    }

    /// The refusal of a repo the hub answered 401 or 403; it names the token's source, never the token. Returns 1.
    pub fn refuse(auth: Auth, err_out: *std.Io.Writer, repo: []const u8) !u8 {
        var buf: [255]u8 = undefined;
        const parsed = std.Uri.parse(auth.endpoint) catch null;
        const own = if (parsed) |u| origin(u, &buf) else null;
        const host = if (own) |o| o.host else auth.endpoint;
        if (auth.bearer == null) {
            try err_out.print("{s} was refused: it is gated or private, or no repo has that name. Check the name; for a gated or private repo, accept its terms on {s}, then set HF_TOKEN or log in with `hf auth login`\n", .{ repo, host });
        } else {
            try err_out.print("{s} is gated or private, and the token from {s} lacks access: accept its terms on {s} with that account, or use a token that can read it\n", .{ repo, auth.source, host });
        }
        return 1;
    }
};

const Origin = struct { scheme: []const u8, host: []const u8, port: u16 };

/// `uri`'s scheme, host (decoded into `buf`) and port, the scheme's default port when it names none.
fn origin(uri: std.Uri, buf: []u8) ?Origin {
    const host = (uri.host orelse return null).toRaw(buf) catch return null;
    if (host.len == 0) return null;
    const port: u16 = uri.port orelse if (std.ascii.eqlIgnoreCase(uri.scheme, "https")) 443 else if (std.ascii.eqlIgnoreCase(uri.scheme, "http")) 80 else return null;
    return .{ .scheme = uri.scheme, .host = host, .port = port };
}

/// Match a credential directly or through percent encoding, without decoding an untrusted whole URL.
fn containsCredential(text: []const u8, secret: []const u8) bool {
    if (secret.len == 0) return false;
    for (0..text.len) |start| {
        var at = start;
        var matched: usize = 0;
        while (matched < secret.len and at < text.len) {
            var byte = text[at];
            if (byte == '%' and at + 2 < text.len) {
                const hi = hex(text[at + 1]) orelse break;
                const lo = hex(text[at + 2]) orelse break;
                byte = hi * 16 + lo;
                at += 2;
            }
            if (byte != secret[matched]) break;
            matched += 1;
            at += 1;
        }
        if (matched == secret.len) return true;
    }
    return false;
}

fn hex(c: u8) ?u8 {
    if (c >= '0' and c <= '9') return c - '0';
    if (c >= 'a' and c <= 'f') return c - 'a' + 10;
    if (c >= 'A' and c <= 'F') return c - 'A' + 10;
    return null;
}

/// The first non-empty of HF_TOKEN and HUGGING_FACE_HUB_TOKEN, else the one token file the `hf` tools read.
pub fn token(a: Allocator, io: std.Io, env: ?*const std.process.Environ.Map, problem: *[]const u8) !?Token {
    for ([_][:0]const u8{ "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN" }) |name| {
        const text = (environment.path(name, hub.envValue(env, name)) catch return bad(a, problem, unsendable, .{name})) orelse "";
        if (text.len > 0) return try usable(a, .{ .value = text, .source = name }, problem);
    }
    const path = try tokenPath(a, env, problem) orelse return null;
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(max_token + 1)) catch |e| switch (e) {
        error.FileNotFound => return null,
        error.StreamTooLong => return bad(a, problem, "the Hugging Face token file {s} is larger than a token; refusing to send it", .{path}),
        else => return bad(a, problem, "the Hugging Face token file {s} cannot be read ({t}); fix it, or set HF_TOKEN", .{ path, e }),
    };
    const text = std.mem.trim(u8, bytes, &std.ascii.whitespace);
    return if (text.len > 0) try usable(a, .{ .value = text, .source = path }, problem) else null;
}

/// HF_TOKEN_PATH, else $HF_HOME/token, else the home cache's; an empty value is unset, so no `token` where pull runs.
fn tokenPath(a: Allocator, env: ?*const std.process.Environ.Map, problem: *[]const u8) !?[]const u8 {
    const named = environment.path("HF_TOKEN_PATH", hub.envValue(env, "HF_TOKEN_PATH")) catch return bad(a, problem, "HF_TOKEN_PATH has a control character in it; fix it, or set HF_TOKEN", .{});
    if (named) |p| return p;
    if (nonEmpty(hub.envValue(env, "HF_HOME"))) |h| return try std.fs.path.join(a, &.{ h, "token" });
    if (nonEmpty(hub.homeDir(env))) |h| return try std.fs.path.join(a, &.{ h, ".cache", "huggingface", "token" });
    return null;
}

fn nonEmpty(value: ?[]const u8) ?[]const u8 {
    const v = value orelse return null;
    return if (v.len > 0) v else null;
}

const unsendable = "the Hugging Face token in {s} has a space or control character in it, or passes 4096 bytes; refusing to send it";

/// `t` when a header can carry it, visible ASCII of at most 4096 bytes; else error.BadToken naming its source.
fn usable(a: Allocator, t: Token, problem: *[]const u8) !Token {
    var ok = t.value.len <= max_token;
    for (t.value) |ch| ok = ok and ch > ' ' and ch < 0x7f;
    if (ok) return t;
    return bad(a, problem, unsendable, .{t.source});
}

fn bad(a: Allocator, problem: *[]const u8, comptime fmt: []const u8, args: anytype) error{ BadToken, OutOfMemory } {
    problem.* = try std.fmt.allocPrint(a, fmt, args);
    return error.BadToken;
}

test {
    _ = @import("pull_auth_test.zig");
}
