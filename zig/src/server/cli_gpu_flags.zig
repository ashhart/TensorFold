//! The CUDA build's --device family and the GPU engine's --tp, --mtp, --checkpoint-slots and --policy flags.
const std = @import("std");
const api = @import("engine_api");
const cli = @import("cli.zig");

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

/// A rank names one of the `tp` processes; without --tp a CUDA family's world alias may set `tp`, checked there.
pub fn ranks(u: *Usage, a: Allocator, out: Args) error{ Usage, OutOfMemory }!void {
    if (out.tp_given and out.rank >= out.tp) return fail(u, a, "argument --rank: {d} is not below --tp {d}", .{ out.rank, out.tp });
}

/// The GPU engine's tensor-parallel, MTP, kept-entry and policy flags; false for any other flag.
pub fn gpuFlag(a: Allocator, out: *Args, name: []const u8, v: []const u8, u: *Usage) error{ Usage, OutOfMemory }!bool {
    const is = struct {
        fn f(x: []const u8, y: []const u8) bool {
            return std.mem.eql(u8, x, y);
        }
    }.f;
    if (is(name, "--tp")) {
        const n = try int(u, a, name, v);
        if (n < 1 or n > 64) return fail(u, a, "argument --tp: a world from 1 to 64: '{s}'", .{v});
        out.tp = @intCast(n);
        out.tp_given = true;
    } else if (is(name, "--pp")) {
        const n = try int(u, a, name, v);
        if (n < 1 or n > 64) return fail(u, a, "argument --pp: a stage count from 1 to 64: '{s}'", .{v});
        if (n > 1) return fail(u, a, "argument --pp: pipeline stages are not served yet (#2005); serve with --pp 1", .{});
    } else if (is(name, "--rank")) {
        out.rank = std.math.cast(u32, try int(u, a, name, v)) orelse return fail(u, a, "argument --rank: a rank from 0: '{s}'", .{v});
        out.rank_given = true;
    } else if (is(name, "--master")) out.master = v else if (is(name, "--master-port")) {
        const p = try int(u, a, name, v);
        if (p < 0 or p > 65535) return fail(u, a, "argument --master-port: invalid port: '{s}'", .{v});
        out.master_port = @intCast(p);
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

test "--tp takes 1 to 64 and marks itself given; --pp takes 1 until stages serve; --rank marks itself given" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var u: Usage = .{};
    var args: Args = .{};
    try std.testing.expect(!args.tp_given and !args.rank_given);
    try std.testing.expect(try gpuFlag(a, &args, "--tp", "3", &u));
    try std.testing.expectEqual(@as(u32, 3), args.tp);
    try std.testing.expect(args.tp_given and !args.rank_given);
    try std.testing.expect(try gpuFlag(a, &args, "--rank", "2", &u));
    try std.testing.expect(args.rank_given);
    try ranks(&u, a, args);
    // --rank without --tp waits for the family's world (TF_GLM_WORLD=3 TF_GLM_RANK=2 with --rank 2)
    try ranks(&u, a, .{ .rank = 2, .rank_given = true });
    try std.testing.expectError(error.Usage, ranks(&u, a, .{ .tp = 2, .tp_given = true, .rank = 2, .rank_given = true }));
    try std.testing.expect(try gpuFlag(a, &args, "--tp", "64", &u));
    try std.testing.expect(try gpuFlag(a, &args, "--pp", "1", &u));
    for ([_][2][]const u8{ .{ "--tp", "0" }, .{ "--tp", "65" }, .{ "--pp", "0" }, .{ "--pp", "2" }, .{ "--pp", "65" } }) |g|
        try std.testing.expectError(error.Usage, gpuFlag(a, &args, g[0], g[1], &u));
    const o = try cli.request(a, "d", "t", args);
    try std.testing.expect(o.given.tp and o.given.rank);
    try std.testing.expect(!(try cli.request(a, "d", "t", .{})).given.tp);
}
