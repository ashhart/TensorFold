//! The Hugging Face token for gated and private repos: where pull reads it, the one origin it goes to, the refusals.
const std = @import("std");
const Allocator = std.mem.Allocator;
const http = std.http;
const hub = @import("hub.zig");
const credential = @import("sys").credential;
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

    /// Complete credentials and decoded prefixes at ``credential.min_prefix`` match at any encoding depth.
    pub fn reflects(auth: Auth, text: []const u8) bool {
        const bearer = auth.bearer orelse return false;
        const secret = if (std.mem.startsWith(u8, bearer, "Bearer ")) bearer[7..] else bearer;
        return credential.contains(std.heap.page_allocator, text, &.{secret}) catch true;
    }

    /// Refuse complete raw or encoded credentials; ambiguous suffixes stay buffered until later bytes decide.
    pub const Body = struct {
        auth: Auth,
        pending: std.ArrayList(u8) = .empty,

        pub fn init(auth: Auth) Body {
            return .{ .auth = auth };
        }

        pub fn deinit(b: *Body, a: Allocator) void {
            b.pending.deinit(a);
        }

        pub fn buffered(b: *const Body) usize {
            return b.pending.items.len;
        }

        /// The prefix now proven safe; the caller writes it once, then commits that exact length.
        pub fn feed(b: *Body, a: Allocator, bytes: []const u8) ![]const u8 {
            try b.pending.appendSlice(a, bytes);
            const bearer = b.auth.bearer orelse return b.pending.items;
            const secret = if (std.mem.startsWith(u8, bearer, "Bearer ")) bearer[7..] else bearer;
            if (secret.len == 0) return b.pending.items;
            const secrets = [_][]const u8{secret};
            var matcher = credential.Matcher.init(a, b.pending.items);
            defer matcher.deinit();
            var i: usize = 0;
            while (i < b.pending.items.len) : (i += 1) {
                const found = try matcher.secretAt(i, &secrets);
                if (found.consumed > 0) return error.HubCredentialBody;
                if (found.hold and found.prefix_len > 0) return b.pending.items[0..i];
            }
            return b.pending.items;
        }

        pub fn commit(b: *Body, sent: usize) void {
            const left = b.pending.items.len - sent;
            std.mem.copyForwards(u8, b.pending.items[0..left], b.pending.items[sent..]);
            b.pending.shrinkRetainingCapacity(left);
        }

        /// The real end of a contiguous body: an incomplete encoding is data, a full key is not.
        pub fn finish(b: *Body, a: Allocator) ![]const u8 {
            const bearer = b.auth.bearer orelse return b.pending.items;
            const secret = if (std.mem.startsWith(u8, bearer, "Bearer ")) bearer[7..] else bearer;
            if (secret.len == 0) return b.pending.items;
            var matcher = credential.Matcher.init(a, b.pending.items);
            defer matcher.deinit();
            if (try matcher.spanFrom(0, secret) != null) return error.HubCredentialBody;
            return b.pending.items;
        }
    };

    /// A cached file with a complete raw or nested-encoded key, checked without whole-file allocation.
    pub fn fileContains(auth: Auth, a: Allocator, io: std.Io, path: []const u8) !bool {
        if (auth.bearer == null) return false;
        const file = try std.Io.Dir.cwd().openFile(io, path, .{});
        defer file.close(io);
        var body: Body = .init(auth);
        defer body.deinit(a);
        var buf: [64 << 10]u8 = undefined;
        var offset: u64 = 0;
        while (true) {
            const n = try file.readPositionalAll(io, &buf, offset);
            if (n == 0) break;
            const safe = body.feed(a, buf[0..n]) catch |err| {
                if (err == error.HubCredentialBody) return true;
                return err;
            };
            body.commit(safe.len);
            offset += n;
        }
        _ = body.finish(a) catch |err| {
            if (err == error.HubCredentialBody) return true;
            return err;
        };
        return false;
    }

    /// Server-controlled text with each plain or repeatedly percent-encoded credential occurrence replaced.
    pub fn redact(auth: Auth, a: Allocator, text: []const u8) ![]const u8 {
        const bearer = auth.bearer orelse return text;
        const secret = if (std.mem.startsWith(u8, bearer, "Bearer ")) bearer[7..] else bearer;
        if (secret.len == 0) return text;
        var matcher = credential.Matcher.init(a, text);
        defer matcher.deinit();
        var out: std.ArrayList(u8) = .empty;
        var at: usize = 0;
        var replaced = false;
        while (try matcher.spanFrom(at, secret)) |span| {
            try out.appendSlice(a, text[at..span.start]);
            try out.appendSlice(a, "[redacted]");
            at = span.end;
            replaced = true;
        }
        if (!replaced) return text;
        try out.appendSlice(a, text[at..]);
        return out.items;
    }

    /// One bodiless request into `req`, redirects followed here so each hop decides the token anew; see `header`.
    pub fn send(auth: Auth, client: *http.Client, req: *http.Client.Request, method: http.Method, uri: std.Uri, options: http.Client.RequestOptions, buf: []u8) !http.Client.Response {
        var o = options;
        o.redirect_behavior = .unhandled;
        var left = options.redirect_behavior;
        var at = uri;
        var rest = buf;
        while (true) {
            const exact = std.fmt.allocPrint(client.allocator, "{f}", .{at}) catch return error.HubCredentialRedirect;
            const reflected = auth.reflects(exact);
            client.allocator.free(exact);
            if (reflected) return error.HubCredentialRedirect;
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
