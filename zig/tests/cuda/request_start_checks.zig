//! Prompt-cache device commands use real forwards, snapshots and device downloads.
const std = @import("std");
const check = @import("check.zig");
const resume_tests = @import("resume_tests.zig");
const cuts = @import("prompt_cuts.zig");
pub const Sha = std.crypto.hash.sha2.Sha256;
pub const Digest = struct { state: [32]u8, head: [32]u8 = @splat(0), logits: [32]u8, hidden: [32]u8, token: u32 };

pub fn dispatch(gpu: check.Gpu, cmd: []const u8, rest: []const [:0]const u8) !bool {
    if (std.mem.eql(u8, cmd, "resume")) {
        try resume_tests.run(gpu, try arg(rest, 0), try arg(rest, 1));
    } else if (std.mem.eql(u8, cmd, "marks")) {
        try resume_tests.insideChunks(gpu, try arg(rest, 0), try arg(rest, 1), try arg(rest, 2), try arg(rest, 3));
    } else if (std.mem.eql(u8, cmd, "prompt-cuts")) {
        try cuts.run(gpu, try arg(rest, 0), try arg(rest, 1));
    } else if (std.mem.eql(u8, cmd, "prompt-segments")) {
        try cuts.segments(gpu, try arg(rest, 0), try arg(rest, 1));
    } else if (std.mem.eql(u8, cmd, "nemotron-request-starts")) {
        try @import("request_starts_nemotron.zig").run(gpu, try arg(rest, 0), try arg(rest, 1), try arg(rest, 2), try arg(rest, 3), if (rest.len > 4) rest[4] else "1");
    } else if (std.mem.eql(u8, cmd, "lfm2-moe-request-starts")) {
        try @import("request_starts_lfm2_moe.zig").run(gpu, try arg(rest, 0), try arg(rest, 1), try arg(rest, 2), try arg(rest, 3));
    } else return false;
    return true;
}

fn arg(rest: []const [:0]const u8, i: usize) ![]const u8 {
    return if (i < rest.len) rest[i] else error.MissingArgument;
}

pub fn readIds(gpu: check.Gpu, path: []const u8, n_text: []const u8) ![]u32 {
    const text = try std.Io.Dir.cwd().readFileAlloc(gpu.io, path, gpu.gpa, .limited(64 << 20));
    defer gpu.gpa.free(text);
    const n = try std.fmt.parseInt(usize, n_text, 10);
    if (n < 2 or n > 16000) return error.BadPromptLength;
    const ids = try gpu.gpa.alloc(u32, n);
    errdefer gpu.gpa.free(ids);
    var it = std.mem.tokenizeAny(u8, text, ", \n\r\t");
    for (ids) |*id| id.* = try std.fmt.parseInt(u32, it.next() orelse return error.PromptTooShort, 10);
    return ids;
}

pub fn points(text: []const u8, len: usize, out: *[32]u32) ![]const u32 {
    var it = std.mem.tokenizeScalar(u8, text, ',');
    var n: usize = 0;
    while (it.next()) |t| {
        if (n == out.len) return error.TooManyStarts;
        const at = try std.fmt.parseInt(u32, t, 10);
        if (at == 0 or at >= len or (n > 0 and at <= out[n - 1])) return error.BadStart;
        out[n] = at;
        n += 1;
    }
    if (n == 0) return error.NoStarts;
    return out[0..n];
}

pub fn add(gpu: check.Gpu, hash: *Sha, ptr: u64, len: usize) !void {
    const host = try gpu.gpa.alloc(u8, @min(len, 16 << 20));
    defer gpu.gpa.free(host);
    var at: usize = 0;
    while (at < len) {
        const n = @min(host.len, len - at);
        try gpu.d.check(gpu.d.api.cuMemcpyDtoH_v2(host.ptr, ptr + at, n), "cuMemcpyDtoH");
        hash.update(host[0..n]);
        at += n;
    }
}

pub fn show(label: []const u8, got: Digest) void {
    std.debug.print("RESULT {s} state={x} head={x} logits={x} hidden={x} token={d}\n", .{ label, got.state, got.head, got.logits, got.hidden, got.token });
}

pub fn equal(label: []const u8, got: Digest, want: Digest) !void {
    show(label, got);
    try check.expect(std.meta.eql(got, want), "{s}: native state, head, logits, hidden or first token differs", .{label});
}

pub fn validateIds(ids: []const u32, vocab: usize) !void {
    for (ids) |id| if (id >= vocab) return error.TokenPastVocab;
}

pub fn poison(ids: []u32, vocab: usize) void {
    for (ids) |*id| id.* = if (id.* + 1 == vocab) 0 else id.* + 1;
}
