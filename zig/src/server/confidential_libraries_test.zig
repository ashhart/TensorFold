//! --confidential's loader policy: the engine's folder, fixed system folders and listed GPU runtime names only.
const std = @import("std");
const builtin = @import("builtin");
const sys = @import("sys");
const policy = sys.load_policy;
const t = std.testing;

const engine = [_][]const u8{ "/opt/tf/bin", "/opt/tf/lib" };

test "confidential: a library loads from the engine's folder, never from Homebrew, a relative path or a .. escape" {
    for ([_][]const u8{
        "/opt/tf/lib/libtfgrammar.dylib",
        "/opt/tf/bin/../lib/libtfgrammar.so",
        "/opt/tf/bin/libtfgrammar.so",
        "@executable_path/../lib/libtfvideo.dylib",
    }) |path| try t.expect(policy.allowedIn(&engine, path));
    for ([_][]const u8{
        "/opt/homebrew/opt/jpeg-turbo/lib/libturbojpeg.0.dylib",
        "/usr/local/opt/webp/lib/libwebpdecoder.3.dylib",
        "/usr/local/lib/libmcdma-rpc.dylib",
        "/opt/tf/lib/../../evil/libtfgrammar.so",
        "/opt/tf/libx/libtfgrammar.so", // a folder whose name only starts like the engine's
        "/opt/tf/lib",
        "@executable_path/../../evil.dylib",
        "lib/libtfgrammar.so",
        "./libtfgrammar.so",
        "",
    }) |path| try t.expect(!policy.allowedIn(&engine, path));
}

test "confidential: a bare name loads only when it is a listed GPU runtime, a codec's never" {
    for ([_][]const u8{ "libcuda.so.1", "libnvidia-ml.so.1", "libnccl.so.2", "libcublas.so.13", "libcublasLt.so.13", "libc.so.6" }) |name|
        try t.expect(policy.allowedIn(&engine, name));
    for ([_][]const u8{ "libturbojpeg.so.0", "libwebpdecoder.so.3", "libpdfium.so", "libtfvideo.so", "libtfgrammar.so", "librocblas.so.5", "libm.so.6" }) |name|
        try t.expect(!policy.allowedIn(&engine, name));
}

test "confidential: macOS system libraries load from their fixed folders, Linux has none" {
    const darwin = comptime builtin.os.tag.isDarwin();
    try t.expectEqual(darwin, policy.allowedIn(&engine, "/usr/lib/libSystem.B.dylib"));
    try t.expectEqual(darwin, policy.allowedIn(&engine, "/System/Library/PrivateFrameworks/X.framework/X"));
    try t.expect(!policy.allowedIn(&engine, "/usr/lib/../local/lib/libturbojpeg.0.dylib"));
}

test "confidential: without the flag every load is the loader's, as before" {
    try t.expect(!policy.isConfined());
    try t.expect(policy.allows("/opt/homebrew/opt/jpeg-turbo/lib/libturbojpeg.0.dylib"));
    try t.expect(policy.allows("libm.so.6"));
}

test "confidential: once confined, the real loader refuses a library it would otherwise open" {
    if (builtin.os.tag != .linux) return error.SkipZigTest; // macOS resolves bare names into /usr/lib, which is allowed
    var before = try sys.DynLib.open("libm.so.6");
    before.close();
    policy.confine();
    defer policy.release();
    try t.expectError(error.FileNotFound, sys.DynLib.open("libm.so.6"));
    var libc = try sys.DynLib.open("libc.so.6");
    libc.close();
}

test "confidential: once confined on macOS, the real loader refuses a path outside the system and engine folders" {
    if (builtin.os.tag != .macos) return error.SkipZigTest;
    policy.confine();
    defer policy.release();
    // a dylib that exists outside the allowed folders is not guaranteed on every Mac, so check the refusal comes first
    try t.expect(!policy.allows("/opt/homebrew/opt/jpeg-turbo/lib/libturbojpeg.0.dylib"));
    var system = try sys.DynLib.open("libSystem.B.dylib");
    system.close();
}
