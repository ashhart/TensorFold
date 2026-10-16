//! Emit full BF16 score rows at fixed histories for comparison with an independent model implementation.
const std = @import("std");
const cuda = @import("cuda");
const flash = @import("flashnext");
const check = @import("check.zig");
const context = 8192;
const chunk = 32;

const Case = struct { id: []const u8, prompt: []const u32, continuation: []const u32 };

fn validate(cases: []const Case) !void {
    if (cases.len == 0 or cases.len > 64) return error.BadFixture;
    for (cases, 0..) |case, i| {
        if (case.id.len == 0 or case.id.len > 64 or case.prompt.len == 0 or case.continuation.len > 64 or case.prompt.len + case.continuation.len > context) return error.BadFixture;
        for (case.id) |c| if (!std.ascii.isAlphanumeric(c) and c != '-') return error.BadFixture;
        for (cases[0..i]) |prior| if (std.mem.eql(u8, prior.id, case.id)) return error.BadFixture;
        for (case.prompt) |id| if (id >= flash.prompt.vocabulary) return error.InvalidToken;
        for (case.continuation) |id| if (id >= flash.prompt.vocabulary) return error.InvalidToken;
    }
}

pub fn run(gpu: check.Gpu, model: []const u8, rank: usize, port: u16, fixture: []const u8, output: []const u8) !void {
    const text = try std.Io.Dir.cwd().readFileAlloc(gpu.io, fixture, gpu.gpa, .limited(1 << 22));
    defer gpu.gpa.free(text);
    const parsed = try std.json.parseFromSlice([]const Case, gpu.gpa, text, .{});
    defer parsed.deinit();
    try validate(parsed.value);
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();
    var glue = try flash.Glue.load(gpu.d);
    defer glue.deinit();
    const tri: flash.Tri = .{ .s = stream, .own = &glue };
    const identity = try flash.rank_engine.identity(gpu.gpa, gpu.io, model, context, chunk);
    const session = try flash.rank_engine.Session.init(gpu.gpa, gpu.io, "127.0.0.1", port, rank, identity);
    const e = flash.prompt.Engine.initRank(gpu.gpa, gpu.io, @constCast(gpu.d), &stream, model, 2, rank) catch |err| {
        session.poison();
        session.deinit();
        return err;
    };
    session.engine = e;
    e.collective = &session.collective;
    defer {
        session.deinit();
        e.deinit();
    }
    if (e.chunk != chunk) return error.RankIdentity;
    var logits = try cuda.DeviceBuffer.alloc(gpu.d, flash.prompt.vocabulary * 4);
    defer logits.free();
    if (rank == 1) {
        try session.follow(flash.Tri, tri, logits.ptr);
        check.pass("fidelity follower completed every fixed-history case", .{});
        return;
    }
    for (parsed.value) |case| {
        var seq = try session.create(context);
        defer session.drop(&seq);
        const row_bytes = flash.prompt.vocabulary * 2;
        const rows = try gpu.gpa.alloc(u8, (case.continuation.len + 1) * row_bytes);
        defer gpu.gpa.free(rows);
        try session.forward(flash.Tri, tri, &seq, case.prompt, .{});
        @memcpy(rows[0..row_bytes], seq.logits);
        for (case.continuation, 0..) |token, i| {
            try session.forward(flash.Tri, tri, &seq, &.{token}, .{});
            @memcpy(rows[(i + 1) * row_bytes ..][0..row_bytes], seq.logits);
        }
        const name = try std.fmt.allocPrint(gpu.gpa, "{s}.bf16", .{case.id});
        defer gpu.gpa.free(name);
        const path = try std.fs.path.join(gpu.gpa, &.{ output, name });
        defer gpu.gpa.free(path);
        try std.Io.Dir.cwd().writeFile(gpu.io, .{ .sub_path = path, .data = rows });
        var digest: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(rows, &digest, .{});
        const hex = std.fmt.bytesToHex(digest, .lower);
        check.pass("fixed history {s}: prompt={d} score-rows={d} sha256={s}", .{ case.id, case.prompt.len, case.continuation.len + 1, &hex });
    }
}

test "fixed histories reject unsafe output identities and invalid token or context bounds" {
    const good: Case = .{ .id = "short-1", .prompt = &.{1}, .continuation = &.{2} };
    try validate(&.{good});
    var bad = good;
    bad.id = "../bad";
    try std.testing.expectError(error.BadFixture, validate(&.{bad}));
    try std.testing.expectError(error.BadFixture, validate(&.{ good, good }));
    bad = good;
    bad.prompt = &.{flash.prompt.vocabulary};
    try std.testing.expectError(error.InvalidToken, validate(&.{bad}));
}
