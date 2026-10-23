//! Observe the real optional-library warning with a recording opener and no installed runtime library.
const std = @import("std");
const libs = @import("sys").optional_libraries;
const pin = @import("grammar_pin");
var warnings: usize = 0;
var text: [2048]u8 = undefined;
var written: []const u8 = "";
pub const std_options: std.Options = .{ .logFn = log };

fn log(comptime level: std.log.Level, comptime scope: @EnumLiteral(), comptime fmt: []const u8, args: anytype) void {
    _ = scope;
    if (level != .warn) return;
    warnings += 1;
    written = std.fmt.bufPrint(&text, fmt, args) catch "overflow";
}

const Fake = struct {
    var present = false;
    var attempts: usize = 0;
    pub fn getenv(_: [*:0]const u8) ?[:0]const u8 {
        return null;
    }
    pub const DynLib = struct {
        pub fn open(path: []const u8) !@This() {
            attempts += 1;
            if (present and std.mem.eql(u8, path, libs.spec(.grammar).names[0])) return .{};
            return error.FileNotFound;
        }
    };
};

pub fn main(init: std.process.Init) !void {
    libs.warnMissing(.grammar, init.gpa, init.io, Fake);
    if (warnings != 1) return error.MissingGrammarMustWarnOnce;
    if (Fake.attempts != 3) return error.GrammarCandidatesChanged;
    if (!std.mem.eql(u8, written, libs.spec(.grammar).message)) return error.CatalogWarningChanged;
    for ([_][]const u8{ libs.spec(.grammar).names[0], "structured output", "required tool calls", "HTTP 400", "release archive's lib/", "zig build grammar -Dxgrammar=DIR", pin.clone }) |part| {
        if (std.mem.indexOf(u8, written, part) == null) return error.IncompleteGrammarRemedy;
    }
    Fake.present = true;
    libs.warnMissing(.grammar, init.gpa, init.io, Fake);
    if (warnings != 1) return error.AvailableGrammarMustNotWarn;
    const source = @embedFile("src/server/serve.zig");
    const call = "platform.optional_libraries.warnMissing(.grammar, a, io, platform);";
    if (std.mem.count(u8, source, call) != 1) return error.StartupMustWarnOnce;
    const at = std.mem.indexOf(u8, source, call).?;
    if (std.mem.indexOf(u8, source[0..at], "const port = lis.port();") == null) return error.WarningBeforeListen;
    if (std.mem.indexOf(u8, source[at..], "if (s.on_listen)") == null) return error.WarningAfterStartup;
}
