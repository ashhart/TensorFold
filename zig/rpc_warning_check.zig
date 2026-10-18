//! Check the real loader's warnings in an executable whose log handler counts them.
const std = @import("std");
const words = @import("src/fabric/words.zig");
const fixtures = @import("rpc_fixtures");
var warnings: usize = 0;
var refusals: usize = 0;
pub const std_options: std.Options = .{ .logFn = log };
fn log(comptime level: std.log.Level, comptime scope: @EnumLiteral(), comptime fmt: []const u8, args: anytype) void {
    _ = scope;
    _ = fmt;
    _ = args;
    if (level == .warn) warnings += 1;
    if (level == .err) refusals += 1;
}
pub fn main() !void {
    var lib = try words.Library.openCandidates(&.{ "/__tf_missing_runtime__/rpc", (fixtures.complete ++ "") });
    lib.close();
    if (warnings != 0) return error.SuccessfulFallbackWarned;
    warnings = 0;
    if (words.Library.openCandidates(&.{ "/__tf_missing_runtime__/one", "/__tf_missing_runtime__/two" })) |opened| {
        var unexpected = opened;
        unexpected.close();
        return error.UnexpectedLibrary;
    } else |err| if (err != error.McdmaRpcUnavailable) return err;
    if (warnings != 1) return error.ExhaustedSearchMustWarnOnce;
    warnings = 0;
    {
        const report = @import("sys").optional_libraries.atBoundary();
        defer report.deinit();
        const nested = @import("sys").optional_libraries.atBoundary();
        nested.deinit();
        try missingVision();
        if (warnings != 0) return error.NativeBoundaryMustRenderTheRefusal;
        const text = @import("sys").optional_libraries.explain(error.CublasLtUnavailable) orelse return error.LostRefusal;
        if (std.mem.indexOf(u8, text, @import("sys").optional_libraries.spec(.cublaslt).names[0]) == null or std.mem.indexOf(u8, text, if (@import("builtin").os.tag == .windows) "PATH" else "LD_LIBRARY_PATH") == null) return error.IncompleteRefusal;
    }
    try missingVision();
    if (warnings != 1) return error.DirectCallerWarningMustBeRestored;
    try hipFollowerWiring();
    warnings = 0;
    missingFollower() catch |err| {
        const text = @import("sys").optional_libraries.explain(err) orelse return error.LostFollowerRefusal;
        std.log.err("tensorfold: {s}", .{text});
    };
    if (warnings != 0 or refusals != 1) return error.FollowerMustRenderExactlyOneRefusal;
    if (@import("sys").optional_libraries.unavailable(.rccl) != error.RcclUnavailable) return error.WrongRcclFailure;
    if (warnings != 1) return error.FailedFollowerMustRestoreDirectWarnings;
}

fn missingVision() !void {
    if (@import("src/cuda/cublaslt.zig").Library.openPath("/__tf_missing_runtime__/libcublasLt.so.13")) |loaded| {
        var lib = loaded;
        lib.close();
        return error.UnexpectedLibrary;
    } else |err| if (err != error.CublasLtUnavailable) return err;
}

// The host control exercises a missing RCCL refusal without opening a GPU or a system RCCL image.
fn missingFollower() !void {
    const report = @import("sys").optional_libraries.atBoundary();
    defer report.deinit();
    return @import("sys").optional_libraries.unavailable(.rccl);
}

// Couple the no-GPU control to the real follower entry, including restoration on its early return.
fn hipFollowerWiring() !void {
    const source = @embedFile("src/native/hip.zig");
    const at = std.mem.indexOf(u8, source, "pub fn follow(") orelse return error.MissingFollower;
    const brace = std.mem.indexOfScalarPos(u8, source, at, '{') orelse return error.MissingFollower;
    const body = std.mem.trimStart(u8, source[brace + 1 ..], " \t\r\n");
    const line = std.mem.indexOfScalar(u8, body, '\n') orelse return error.MissingFollowerScope;
    if (std.mem.indexOf(u8, body[0..line], "optional_libraries.atBoundary()") == null)
        return error.MissingFollowerScope;
    if (!std.mem.startsWith(u8, std.mem.trimStart(u8, body[line + 1 ..], " \t"), "defer report.deinit();"))
        return error.MissingFollowerRestoration;
}
