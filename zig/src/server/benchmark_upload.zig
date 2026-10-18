//! Opt-in uploads: login, stored credentials, the public-checkpoint check and the submission.
const std = @import("std");
const json = @import("json");
const receipt = @import("benchmark_receipt.zig");
const check = @import("benchmark_check.zig");
const stream = @import("benchmark_stream.zig");
const Allocator = std.mem.Allocator;
const Value = json.Value;

pub const default_api = "https://tensorfold.dev/api/benchmarks/v1";
pub const hf_models = "https://huggingface.co/api/models/";

/// Enabled only in a release coordinated with the receiver's v2 validation and separate rankings.
pub const v2_submissions_enabled = false;
pub const v2_disabled = "community-v2 submissions are disabled until the client and receiver both support v2; the receipt remains local";

pub fn publicationAllowed(v: Value, problem: *[]const u8) bool {
    const suite = v.get("suite") orelse return true;
    if (!v2_submissions_enabled and std.mem.eql(u8, json.strOr(suite.get("id")), "community-v2")) {
        problem.* = v2_disabled;
        return false;
    }
    return true;
}

/// ``api`` is an ``apiUrl`` result; ``hf_base`` is a test seam; ``home`` holds ``.tensorfold/benchmarks``.
pub const Options = struct { api: []const u8, hf_base: []const u8 = hf_models, environ: *const std.process.Environ.Map, home: []const u8, now_ms: i64 };

pub const Ack = struct { id: []const u8, status: []const u8, results_url: ?[]const u8 };

/// The receiver's API URL: no credentials, query or fragment; HTTPS, or HTTP to a loopback host; no trailing slash.
pub fn apiUrl(text: []const u8, problem: *[]const u8) ?[]const u8 {
    const uri = std.Uri.parse(text) catch null;
    const shape_ok = if (uri) |u| u.user == null and u.password == null and (u.query == null or u.query.?.isEmpty()) and (u.fragment == null or u.fragment.?.isEmpty()) and u.host != null and !u.host.?.isEmpty() else false;
    if (!shape_ok) {
        problem.* = "The upload API URL must not contain credentials, a query or a fragment";
        return null;
    }
    const u = uri.?;
    const https = std.mem.eql(u8, u.scheme, "https");
    if (!https and !(std.mem.eql(u8, u.scheme, "http") and loopback(u.host.?.percent_encoded))) {
        problem.* = "Benchmark uploads require HTTPS, except for loopback development servers";
        return null;
    }
    return std.mem.trimEnd(u8, text, "/");
}

/// ``localhost`` or a loopback IP literal.
pub fn loopback(host: []const u8) bool {
    if (std.ascii.eqlIgnoreCase(host, "localhost")) return true;
    const ip = std.Io.net.IpAddress.parse(std.mem.trim(u8, host, "[]"), 0) catch return false;
    return switch (ip) {
        .ip4 => |v| v.bytes[0] == 127,
        .ip6 => |v| std.mem.eql(u8, &v.bytes, &(@as([15]u8, @splat(0)) ++ [1]u8{1})),
    };
}

/// The receiver's own token rule: 32 to 256 of ``[A-Za-z0-9_-]``.
pub fn tokenOk(text: []const u8) bool {
    if (text.len < 32 or text.len > 256) return false;
    for (text) |ch| if (!std.ascii.isAlphanumeric(ch) and ch != '_' and ch != '-') return false;
    return true;
}

fn authPath(a: Allocator, home: []const u8) ![]const u8 {
    return std.fs.path.join(a, &.{ home, ".tensorfold", "benchmarks", "auth.json" });
}

/// The upload token: ``TENSORFOLD_BENCHMARK_TOKEN``, else the private auth file bound to this API.
pub fn credential(a: Allocator, io: std.Io, o: Options, problem: *[]const u8) !?[]const u8 {
    if (o.environ.get("TENSORFOLD_BENCHMARK_TOKEN")) |t| if (t.len > 0) {
        if (tokenOk(t)) return t;
        problem.* = "Invalid benchmark upload token";
        return null;
    };
    const path = try authPath(a, o.home);
    const st = std.Io.Dir.cwd().statFile(io, path, .{}) catch {
        problem.* = "An upload token is required; run tensorfold benchmark login";
        return null;
    };
    if (st.permissions.toMode() & 0o077 != 0) {
        problem.* = "Upload credentials have broad file permissions; run benchmark login again";
        return null;
    }
    problem.* = "Invalid benchmark credentials";
    if (st.size > 4096) return null;
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(4097)) catch return null;
    const v = switch (try json.parse(a, bytes)) {
        .ok => |x| x,
        .err => {
            problem.* = "An upload token is required; run tensorfold benchmark login";
            return null;
        },
    };
    if (v != .object or v.object.count() != 2 or !v.has("api") or !v.has("token")) return null;
    if (!std.mem.eql(u8, json.strOr(v.get("api")), o.api) or v.get("api").? != .string) {
        problem.* = "Upload credentials belong to a different API URL; log in to this receiver first";
        return null;
    }
    const token = v.get("token").?;
    if (token != .string or !tokenOk(token.string)) {
        problem.* = "Invalid benchmark upload token";
        return null;
    }
    return token.string;
}

/// Checks ``token`` with the receiver, then stores it 0600 beside the API it belongs to; the display name.
pub fn login(a: Allocator, io: std.Io, o: Options, token: []const u8, problem: *[]const u8) !?[]const u8 {
    if (!tokenOk(token)) {
        problem.* = "Invalid benchmark upload token";
        return null;
    }
    const profile = try request(a, io, o, .GET, "/me", token, null, problem) orelse return null;
    const path = try authPath(a, o.home);
    const dir = std.fs.path.dirname(path).?;
    const cwd = std.Io.Dir.cwd();
    var rand: [8]u8 = undefined;
    io.random(&rand);
    const temp = try std.fmt.allocPrint(a, "{s}/.auth-{s}", .{ dir, std.fmt.bytesToHex(rand, .lower) });
    problem.* = "Could not store the upload credentials; check the TensorFold directory permissions";
    _ = cwd.createDirPathStatus(io, dir, .fromMode(0o700)) catch return null;
    const stored = try json.newObject(a);
    try stored.put(a, "api", .{ .string = o.api });
    try stored.put(a, "token", .{ .string = token });
    const text = try json.stringify(a, .{ .object = stored }, .{});
    var file = cwd.createFile(io, temp, .{ .exclusive = true, .permissions = .fromMode(0o600) }) catch return null;
    const written = if (file.writeStreamingAll(io, text)) true else |_| false;
    file.close(io);
    if (!written or (if (cwd.rename(temp, cwd, path, io)) false else |_| true)) {
        cwd.deleteFile(io, temp) catch {};
        return null;
    }
    problem.* = "";
    const name = profile.get("display_name") orelse return "contributor";
    return if (name == .string) name.string else "contributor";
}

pub fn logout(a: Allocator, io: std.Io, home: []const u8) !void {
    std.Io.Dir.cwd().deleteFile(io, try authPath(a, home)) catch |e| if (e != error.FileNotFound) return e;
}

/// A client that connects directly: proxy variables are not read, so HTTPS always reaches the host itself.
fn newClient(io: std.Io) std.http.Client {
    var client: std.http.Client = .{ .allocator = std.heap.smp_allocator, .io = io };
    client.connection_pool.free_size = 0;
    return client;
}

/// One receiver exchange; only fixed texts are reported, never the receiver's body.
fn request(a: Allocator, io: std.Io, o: Options, method: std.http.Method, route: []const u8, token: []const u8, body: ?[]const u8, problem: *[]const u8) !?Value {
    const url = try std.mem.concat(a, u8, &.{ o.api, route });
    var client = newClient(io);
    defer client.deinit();
    const got = stream.fetch(a, io, &client, .{ .method = method, .url = url, .body = body, .token = token, .limit = receipt.max_bytes, .timeout_s = 30 });
    switch (got.failed) {
        .network, .deadline => {
            problem.* = "Could not reach the benchmark receiver; your local receipt is saved";
            return null;
        },
        .stopped => {
            problem.* = "";
            return null;
        },
        else => {},
    }
    if (got.status < 200 or got.status > 299) {
        problem.* = switch (got.status) {
            401 => "Upload token was refused; run tensorfold benchmark login again",
            403 => "This contributor cannot upload results",
            404 => "The benchmark receiver is not available at that API URL",
            409 => "A different receipt already uses this run identifier",
            413 => "The benchmark receipt is too large",
            429 => "Upload quota reached; retry later",
            503 => "The benchmark receiver is not configured yet",
            else => try std.fmt.allocPrint(a, "Benchmark receiver refused the request, HTTP {d}", .{got.status}),
        };
        return null;
    }
    problem.* = if (!got.json_type) "The benchmark receiver returned a non-JSON response" else if (got.failed == .too_large) "Benchmark receiver response is too large" else "Benchmark receiver returned invalid JSON";
    if (!got.json_type or got.failed == .too_large) return null;
    const v = switch (try json.parse(a, got.body)) {
        .ok => |x| x,
        .err => return null,
    };
    if (v != .object) {
        problem.* = "Benchmark receiver returned an invalid response";
        return null;
    }
    return v;
}

/// The board's rules, then an unauthenticated Hugging Face lookup proving the model (and drafter) public.
pub fn verifyPublic(a: Allocator, io: std.Io, o: Options, v: Value, problem: *[]const u8) !bool {
    if (!check.receipt(v, .publish, o.now_ms, problem)) return false;
    if (!publicationAllowed(v, problem)) return false;
    const m = v.get("model").?;
    for ([_]?Value{ m.get("repo_id"), m.get("drafter_repo_id") }) |r| {
        const repo = (r orelse continue).str() orelse continue;
        if (repo.len == 0) continue;
        const url = try std.mem.concat(a, u8, &.{ o.hf_base, repo });
        var client = newClient(io);
        defer client.deinit();
        const got = stream.fetch(a, io, &client, .{ .url = url, .limit = 256 << 10, .timeout_s = 15 });
        problem.* = "Could not reach huggingface.co to verify public checkpoint metadata; the receipt remains local";
        switch (got.failed) {
            .network, .deadline => return false,
            .stopped => {
                problem.* = "";
                return false;
            },
            else => {},
        }
        problem.* = "Only publicly accessible Hugging Face checkpoint IDs can be published";
        if (got.status < 200 or got.status > 299) return false;
        problem.* = "Public checkpoint metadata could not be verified";
        if (!got.json_type or got.failed == .too_large) return false;
        const info = switch (try json.parse(a, got.body)) {
            .ok => |x| x,
            .err => {
                problem.* = "Could not verify public checkpoint metadata; the receipt remains local";
                return false;
            },
        };
        const private = info.get("private");
        problem.* = "Only publicly accessible Hugging Face checkpoint IDs can be published";
        if (private == null or private.? != .bool or private.?.bool) return false;
    }
    problem.* = "";
    return true;
}

/// Verifies again, then posts the canonical receipt with the contributor token; the receiver's acknowledgement.
pub fn upload(a: Allocator, io: std.Io, o: Options, v: Value, problem: *[]const u8) !?Ack {
    if (!try verifyPublic(a, io, o, v, problem)) return null;
    const token = try credential(a, io, o, problem) orelse return null;
    const bytes = try receipt.canonical(a, v);
    if (bytes.len > receipt.max_bytes) {
        problem.* = "Upload exceeds 1 MiB";
        return null;
    }
    const result = try request(a, io, o, .POST, "/submissions", token, bytes, problem) orelse return null;
    const id = result.get("id");
    const status = json.strOr(result.get("status"));
    if (id == null or id.? != .string or !(std.mem.eql(u8, status, "pending") or std.mem.eql(u8, status, "approved"))) {
        problem.* = "Benchmark receiver did not acknowledge a valid submission";
        return null;
    }
    const link = result.get("results_url");
    return .{ .id = id.?.string, .status = status, .results_url = if (link != null and link.? == .string and link.?.string.len > 0) link.?.string else null };
}

test "upload URLs need HTTPS unless loopback, and no credentials, query or fragment" {
    var problem: []const u8 = "";
    try std.testing.expectEqualStrings("https://tensorfold.dev/api/benchmarks/v1", apiUrl("https://tensorfold.dev/api/benchmarks/v1/", &problem).?);
    for ([_][]const u8{ "http://127.0.0.1:9/api", "http://localhost:9", "http://[::1]:9/x" }) |ok| try std.testing.expect(apiUrl(ok, &problem) != null);
    for ([_][]const u8{ "http://example.test/api", "http://192.0.2.1/api" }) |bad| {
        try std.testing.expect(apiUrl(bad, &problem) == null);
        try std.testing.expectEqualStrings("Benchmark uploads require HTTPS, except for loopback development servers", problem);
    }
    for ([_][]const u8{ "https://u:p@example.test", "https://example.test/?k=v", "https://example.test/#f", "not a url" }) |bad| try std.testing.expect(apiUrl(bad, &problem) == null);
}

test "tokens follow the receiver's rule" {
    try std.testing.expect(tokenOk("synthetic-contributor-token-0001") and tokenOk("tfp_0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"));
    try std.testing.expect(!tokenOk("short") and !tokenOk("synthetic.contributor.token.0001") and !tokenOk("synthetic contributor token 0001"));
}
