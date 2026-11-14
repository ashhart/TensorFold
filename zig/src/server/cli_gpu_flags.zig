//! The CUDA build's --device family and the GPU engine's --tp, --mtp, --checkpoint-slots and --policy flags.
const std = @import("std");
const api = @import("engine_api");
const cli = @import("cli.zig");
const sys = @import("sys");
const auth = sys.rank_auth;

const Args = cli.Args;
const Usage = cli.Usage;
const fail = cli.fail;
const int = cli.int;
const float = cli.float;
const gib = cli.gib;

/// The CUDA build's --device, --segments and --precision; false for any other flag.
pub fn cudaFlag(a: Allocator, out: *Args, name: []const u8, v: []const u8, u: *Usage) error{ Usage, OutOfMemory }!bool {
    if (std.mem.eql(u8, name, "--precision")) {
        out.precision = std.meta.stringToEnum(api.Precision, v) orelse return fail(u, a, "argument --precision: checkpoint or full: '{s}'", .{v});
    } else if (std.mem.eql(u8, name, "--device")) {
        out.device = std.math.cast(u32, try int(u, a, name, v)) orelse return fail(u, a, "argument --device: a GPU ordinal from 0: '{s}'", .{v});
    } else if (std.mem.eql(u8, name, "--rope-yarn")) out.rope_yarn = try float(u, a, name, v) else if (std.mem.eql(u8, name, "--loops")) out.loops = std.math.cast(u32, try int(u, a, name, v)) orelse return fail(u, a, "argument --loops: a count from 1: '{s}'", .{v}) else if (std.mem.eql(u8, name, "--segments")) {
        const n = try int(u, a, name, v);
        out.segments = if (n >= 1) @intCast(@min(n, std.math.maxInt(u32))) else return fail(u, a, "argument --segments: a count from 1: '{s}'", .{v});
    } else return false;
    return true;
}

/// A rank names one of the `tp` processes.
pub fn ranks(u: *Usage, a: Allocator, out: Args) error{ Usage, OutOfMemory }!void {
    if (out.rank >= out.tp) return fail(u, a, "argument --rank: {d} is not below --tp {d}", .{ out.rank, out.tp });
}

/// The GPU engine's tensor-parallel, MTP, kept-entry and policy flags; false for any other flag.
pub fn gpuFlag(a: Allocator, out: *Args, name: []const u8, v: []const u8, u: *Usage) error{ Usage, OutOfMemory }!bool {
    const is = struct {
        fn f(x: []const u8, y: []const u8) bool {
            return std.mem.eql(u8, x, y);
        }
    }.f;
    if (is(name, "--tp")) out.tp = @intCast(try int(u, a, name, v)) else if (is(name, "--rank")) {
        out.rank = std.math.cast(u32, try int(u, a, name, v)) orelse return fail(u, a, "argument --rank: a rank from 0: '{s}'", .{v});
    } else if (is(name, "--master")) out.master = v else if (is(name, "--master-port")) {
        const p = try int(u, a, name, v);
        if (p < 0 or p > 65535) return fail(u, a, "argument --master-port: invalid port: '{s}'", .{v});
        out.master_port = @intCast(p);
    } else if (is(name, "--rank-secret")) {
        // every rank link reads its key from TF_RANK_SECRET_FILE (sys.rank_auth): the flag names that file
        const path = try a.dupeSentinel(u8, v, 0);
        _ = auth.Secret.fromFile(path) catch |e| return fail(u, a, "argument --rank-secret: {s}: '{s}'", .{ auth.explain(e) orelse @errorName(e), v });
        if (sys.env.setenv(auth.env_name, path, 1) != 0) return fail(u, a, "argument --rank-secret: the environment could not be set", .{});
    } else if (is(name, "--mtp-drafts")) {
        const n = try int(u, a, name, v);
        if (n < 0 or n > 64) return fail(u, a, "argument --mtp-drafts: a count from 0 to 64: '{s}'", .{v});
        out.mtp_drafts = @intCast(n);
    } else if (is(name, "--mtp-confidence")) {
        const c = try float(u, a, name, v);
        if (!(c >= 0 and c <= 1)) return fail(u, a, "argument --mtp-confidence: a probability in [0, 1]: '{s}'", .{v});
        out.mtp_confidence = c;
    } else if (is(name, "--checkpoint-slots")) {
        const n = try int(u, a, name, v);
        if (n < 0) return fail(u, a, "argument --checkpoint-slots: a count from 0: '{s}'", .{v});
        out.checkpoint_slots = n;
    } else if (is(name, "--spill-gib")) out.spill_gib = try gib(u, a, name, v) else if (is(name, "--policy")) {
        if (std.mem.indexOfScalar(u8, v, '=') == null) return fail(u, a, "argument --policy: expected key=value: '{s}'", .{v});
        out.policy = if (out.policy.len == 0) v else try std.fmt.allocPrint(a, "{s},{s}", .{ out.policy, v });
    } else return false;
    return true;
}

const Allocator = std.mem.Allocator;

test "security: --rank-secret takes only a key file its owner alone can read, and names it for every rank link" {
    if (sys.windows) return error.SkipZigTest; // refusal by mode bits
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const c = std.c;
    const path = try std.fmt.allocPrintSentinel(a, "/tmp/tf-rank-secret-flag-{d}", .{c.getpid()}, 0);
    defer _ = c.unlink(path);
    defer _ = sys.env.unsetenv(auth.env_name);
    _ = sys.env.unsetenv(auth.env_name);
    var args: Args = .{};
    var u: Usage = .{};
    for ([_]c.mode_t{ 0o644, 0o600 }) |mode| {
        _ = c.unlink(path);
        const fd = c.open(path, .{ .ACCMODE = .WRONLY, .CREAT = true, .EXCL = true }, mode);
        try std.testing.expect(fd >= 0);
        _ = c.fchmod(fd, mode);
        _ = c.write(fd, "0123456789abcdef0123456789abcdef0123\n", 37);
        _ = c.close(fd);
        if (mode == 0o600) {
            try std.testing.expect(try gpuFlag(a, &args, "--rank-secret", path, &u));
            try std.testing.expectEqualStrings(path, sys.getenv(auth.env_name).?);
            try std.testing.expect((try auth.Secret.fromEnv()).keyed);
        } else {
            try std.testing.expectError(error.Usage, gpuFlag(a, &args, "--rank-secret", path, &u));
            try std.testing.expect(std.mem.indexOf(u8, u.message, "chmod 600") != null);
            try std.testing.expect(sys.getenv(auth.env_name) == null);
        }
    }
    try std.testing.expectError(error.Usage, gpuFlag(a, &args, "--rank-secret", "/nonexistent/rank-secret", &u));
}
