//! Missing runtime files must identify the dependency before model loading can hide the cause.
const std = @import("std");
const lt = @import("src/cuda/cublaslt.zig");
const nccl = @import("src/cuda/nccl.zig");
const jpeg = @import("src/core/picture/jpeg.zig");
const webp = @import("src/core/picture/webp.zig");

test "runtime library: missing CUDA vision dependency retains its identity" {
    try std.testing.expectError(error.CublasLtUnavailable, lt.Library.openPath("/__tf_missing_runtime__/libcublasLt.so.13"));
}

test "runtime library: missing CUDA rank dependency retains its identity" {
    try std.testing.expectError(error.NcclUnavailable, nccl.Library.openPath("/__tf_missing_runtime__/libnccl.so.2"));
}

test "runtime library: missing media refusals name the actual loader soname" {
    const mac = @import("builtin").os.tag == .macos;
    const windows = @import("builtin").os.tag == .windows;
    try std.testing.expect(std.mem.containsAtLeast(u8, jpeg.missing, 1, if (mac) "libturbojpeg.0.dylib" else if (windows) "turbojpeg.dll" else "libturbojpeg.so.0"));
    try std.testing.expect(std.mem.containsAtLeast(u8, webp.missing, 1, if (mac) "libwebpdecoder.3.dylib" else "libwebpdecoder.so.3"));
}

test "runtime library: missing RoCE dependency retains its identity" {
    try std.testing.expectError(error.VerbsUnavailable, @import("src/fabric/verbs.zig").Verbs.open("/__tf_missing_runtime__/libibverbs.so.1"));
}

test "runtime library: catalogue messages identify soname, feature and installation" {
    const libs = @import("sys").optional_libraries;
    for (libs.all) |s| {
        const message = libs.explain(s.failure).?;
        try std.testing.expect(std.mem.containsAtLeast(u8, message, 1, s.names[0]));
        try std.testing.expect(std.mem.containsAtLeast(u8, message, 1, s.feature));
        try std.testing.expect(std.mem.containsAtLeast(u8, message, 1, s.install));
    }
    try std.testing.expect(libs.explain(error.OutOfMemory) == null);
}

test "runtime library: discovery uses overrides and retains found libraries without invoking a device" {
    const libs = @import("sys").optional_libraries;
    const Fake = struct {
        var override: bool = false;
        var empty_override: bool = false;
        var attempts: usize = 0;
        var closes: usize = 0;
        pub fn getenv(_: [*:0]const u8) ?[:0]const u8 {
            if (empty_override) return "";
            return if (override) "/missing-override" else null;
        }
        pub const DynLib = struct {
            pub fn open(path: []const u8) !@This() {
                attempts += 1;
                if (std.mem.eql(u8, path, "libnccl.so.2") or std.mem.eql(u8, path, libs.spec(.jpeg).names[0])) return .{};
                return error.FileNotFound;
            }
            pub fn close(_: *@This()) void {
                closes += 1;
            }
        };
    };
    Fake.override = false;
    Fake.attempts = 0;
    Fake.closes = 0;
    try std.testing.expect(libs.found(libs.spec(.nccl), std.testing.allocator, Fake, null));
    try std.testing.expect(libs.found(libs.spec(.jpeg), std.testing.allocator, Fake, null));
    try std.testing.expect(!libs.found(libs.spec(.cublaslt), std.testing.allocator, Fake, null));
    try std.testing.expectEqual(@as(usize, 0), Fake.closes);
    const rows = try libs.probe(std.testing.allocator, std.testing.io, Fake);
    defer std.testing.allocator.free(rows);
    try std.testing.expectEqual(libs.all.len, rows.len);
    var found: usize = 0;
    for (rows) |row| if (row.found) {
        found += 1;
    };
    try std.testing.expectEqual(@as(usize, 2), found);
    try std.testing.expectEqual(@as(usize, 0), Fake.closes);
    Fake.override = true;
    const start = Fake.attempts;
    try std.testing.expect(!libs.found(libs.spec(.jpeg), std.testing.allocator, Fake, null));
    try std.testing.expectEqual(start + 1, Fake.attempts);
    try std.testing.expectEqual(@as(usize, 0), Fake.closes);
    Fake.empty_override = true;
    try std.testing.expect(!libs.found(libs.spec(.jpeg), std.testing.allocator, Fake, null));
    Fake.empty_override = false;
}

const words = @import("src/fabric/words.zig");
const fixtures = @import("rpc_fixtures");
test "runtime library: a loadable RPC helper missing each required export keeps the dependency identity" {
    for ([_][:0]const u8{ (fixtures.abi ++ ""), (fixtures.wait ++ ""), (fixtures.store ++ "") }) |path| {
        try std.testing.expectError(error.McdmaRpcUnavailable, words.Library.openPath(path));
    }
    try std.testing.expectError(error.WrongAbi, words.Library.openPath((fixtures.wrong ++ "")));
    var lib = try words.Library.openPath((fixtures.complete ++ ""));
    defer lib.close();
    var word: u64 = 0;
    lib.words().store(&word, 42);
    try std.testing.expectEqual(@as(u64, 42), lib.words().wait(&word, 1, true, 0, 1));
}

test "runtime library: RPC discovery refuses bare and extra default paths the runtime never uses" {
    const libs = @import("sys").optional_libraries;
    const Fake = struct {
        var only: []const u8 = "";
        pub fn getenv(_: [*:0]const u8) ?[:0]const u8 {
            return null;
        }
        pub const DynLib = struct {
            pub fn open(path: []const u8) !@This() {
                if (std.mem.eql(u8, only, path)) return .{};
                return error.FileNotFound;
            }
            pub fn close(_: *@This()) void {}
        };
    };
    Fake.only = libs.spec(.mcdma_rpc).names[0];
    try std.testing.expect(!libs.found(libs.spec(.mcdma_rpc), std.testing.allocator, Fake, null));
    if (@import("builtin").os.tag == .macos) {
        Fake.only = "/usr/lib/libmcdma-rpc.dylib";
        try std.testing.expect(!libs.found(libs.spec(.mcdma_rpc), std.testing.allocator, Fake, null));
    }
    Fake.only = if (@import("builtin").os.tag == .macos) "/usr/local/lib/libmcdma-rpc.dylib" else "/usr/lib/libmcdma-rpc.so";
    try std.testing.expect(libs.found(libs.spec(.mcdma_rpc), std.testing.allocator, Fake, null));
}

test "runtime library: RPC defaults cover both OS layouts and overrides keep their precedence" {
    const libs = @import("sys").optional_libraries;
    const linux = libs.rpcPaths(.linux);
    try std.testing.expectEqual(@as(usize, 2), linux.len);
    try std.testing.expectEqualStrings("/usr/local/lib/libmcdma-rpc.so", linux[0]);
    try std.testing.expectEqualStrings("/usr/lib/libmcdma-rpc.so", linux[1]);
    const mac = libs.rpcPaths(.macos);
    try std.testing.expectEqual(@as(usize, 1), mac.len);
    try std.testing.expectEqualStrings("/usr/local/lib/libmcdma-rpc.dylib", mac[0]);
    const Fake = struct {
        var override: ?[:0]const u8 = null;
        var only: []const u8 = "";
        var attempts: usize = 0;
        var closes: usize = 0;
        pub fn getenv(_: [*:0]const u8) ?[:0]const u8 {
            return override;
        }
        pub const DynLib = struct {
            pub fn open(path: []const u8) !@This() {
                attempts += 1;
                if (std.mem.eql(u8, path, only)) return .{};
                return error.FileNotFound;
            }
            pub fn close(_: *@This()) void {
                closes += 1;
            }
        };
    };
    Fake.override = "/chosen-rpc";
    Fake.only = "/chosen-rpc";
    try std.testing.expect(libs.found(libs.spec(.mcdma_rpc), std.testing.allocator, Fake, null));
    try std.testing.expectEqual(@as(usize, 1), Fake.attempts);
    try std.testing.expectEqual(@as(usize, 0), Fake.closes);
    Fake.override = "/missing-rpc";
    Fake.only = libs.rpcPaths(@import("builtin").os.tag)[0];
    try std.testing.expect(!libs.found(libs.spec(.mcdma_rpc), std.testing.allocator, Fake, null));
    try std.testing.expectEqual(@as(usize, 2), Fake.attempts);
    Fake.override = "";
    try std.testing.expect(libs.found(libs.spec(.mcdma_rpc), std.testing.allocator, Fake, null));
    try std.testing.expectEqual(@as(usize, 0), Fake.closes);
}

test "runtime library: installation copy names the platform loader environment" {
    const libs = @import("sys").optional_libraries;
    const name = if (@import("builtin").os.tag == .windows) "PATH" else "LD_LIBRARY_PATH";
    for ([_]libs.Id{ .cublaslt, .cublas, .nccl, .rccl }) |id| {
        const text = libs.spec(id).install;
        try std.testing.expect(std.mem.indexOf(u8, text, name) != null);
        try std.testing.expect(std.mem.indexOf(u8, text, "expose it to the library loader") == null);
    }
}

test "runtime library: native entry points defer the catalog warning to their refusal" {
    for ([_][]const u8{ @embedFile("src/native/cuda.zig"), @embedFile("src/native/hip.zig"), @embedFile("src/native/metal.zig") }) |source| {
        const at = std.mem.indexOf(u8, source, "pub fn open(").?;
        const brace = std.mem.indexOfScalarPos(u8, source, at, '{').?;
        const body = std.mem.trimStart(u8, source[brace + 1 ..], " \t\r\n");
        const line = std.mem.indexOfScalar(u8, body, '\n').?;
        try std.testing.expect(std.mem.indexOf(u8, body[0..line], "optional_libraries.atBoundary()") != null);
        try std.testing.expect(std.mem.startsWith(u8, std.mem.trimStart(u8, body[line + 1 ..], " \t"), "defer report.deinit();"));
    }
}
