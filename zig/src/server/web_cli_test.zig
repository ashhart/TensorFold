const std = @import("std");
const web = @import("web_cli.zig");
const t = std.testing;
const io = t.io;
const release: web.Version = .{ .version = "9.8.7", .is_release = true };
const spec = "tensorfold-web @ git+https://github.com/ashhart/TensorFold@v9.8.7#subdirectory=web";
const prompt = "Install it now with uv? [y/N]";
const app_script =
    \\#!/bin/sh
    \\printf '%s\n' "$TENSORFOLD_BIN" "$@" > "$TF_WEB_TEST_LAUNCH"
    \\exit "${TF_WEB_TEST_STATUS:-0}"
    \\
;
const uv_script =
    \\#!/bin/sh
    \\printf '%s\n' "$@" >> "$TF_WEB_TEST_UV_LOG"
    \\if [ "$1 $2" = 'tool install' ]; then
    \\  [ "${TF_WEB_TEST_FAIL:-0}" = 0 ] || exit "$TF_WEB_TEST_FAIL"
    \\  [ "${TF_WEB_TEST_MISSING:-0}" = 0 ] || exit 0
    \\  /bin/cp "$TF_WEB_TEST_SOURCE" "$TF_WEB_TEST_BIN/tensorfold-web"
    \\elif [ "$1 $2 $3" = 'tool dir --bin' ]; then
    \\  printf '%s\n' "$TF_WEB_TEST_BIN"
    \\else exit 99
    \\fi
    \\
;

const Case = struct {
    installed: bool = false,
    uv: bool = true,
    interactive: bool = true,
    no_prompt: bool = false,
    version: web.Version = release,
    answer: []const u8 = "y\n",
    remote_bin: bool = false,
    install_fail: bool = false,
    missing_after_install: bool = false,
    web_status: u8 = 0,
    expected: u8 = 1,
    asks: bool = false,
    installs: bool = false,
    launches: bool = false,
};

fn check(c: Case) !void {
    if (@import("builtin").os.tag == .windows) return error.SkipZigTest;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const root = try tmp.dir.realPathFileAlloc(io, ".", a);
    try tmp.dir.createDir(io, "bin", .default_dir);
    try tmp.dir.createDir(io, "other bin", .default_dir);
    const bin = try std.fs.path.join(a, &.{ root, "bin" });
    const target_bin = if (c.remote_bin) try std.fs.path.join(a, &.{ root, "other bin" }) else bin;
    const source = try std.fs.path.join(a, &.{ root, "app-source" });
    try tmp.dir.writeFile(io, .{ .sub_path = "app-source", .data = app_script, .flags = .{ .permissions = .fromMode(0o700) } });
    if (c.installed) try tmp.dir.writeFile(io, .{ .sub_path = "bin/tensorfold-web", .data = app_script, .flags = .{ .permissions = .fromMode(0o700) } });
    if (c.uv) try tmp.dir.writeFile(io, .{ .sub_path = "bin/uv", .data = uv_script, .flags = .{ .permissions = .fromMode(0o700) } });
    var env: std.process.Environ.Map = .init(a);
    defer env.deinit();
    try env.put("PATH", bin);
    try env.put("TF_WEB_TEST_BIN", target_bin);
    try env.put("TF_WEB_TEST_SOURCE", source);
    try env.put("TF_WEB_TEST_UV_LOG", try std.fs.path.join(a, &.{ root, "uv.log" }));
    try env.put("TF_WEB_TEST_LAUNCH", try std.fs.path.join(a, &.{ root, "launch.log" }));
    try env.put("TF_WEB_TEST_STATUS", try std.fmt.allocPrint(a, "{d}", .{c.web_status}));
    try env.put("TF_WEB_TEST_FAIL", if (c.install_fail) "23" else "0");
    try env.put("TF_WEB_TEST_MISSING", if (c.missing_after_install) "1" else "0");
    try env.put("TENSORFOLD_BIN", "wrong-binary");
    if (c.no_prompt) try env.put("TENSORFOLD_WEB_NO_PROMPT", "1");
    var out: std.Io.Writer.Allocating = .init(a);
    defer out.deinit();
    var input: std.Io.Reader = .fixed(c.answer);
    const status = try web.command(.{
        .a = a,
        .io = io,
        .environ = &env,
        .out = &out.writer,
        .input = &input,
        .interactive = c.interactive,
        .self_exe = "/example path/tensorfold",
        .launch = web.spawnAndWait,
    }, &.{ "--port", "9090", "two words", "$(untouched);*", "" }, c.version);
    try t.expectEqual(c.expected, status);
    try t.expectEqual(c.asks, std.mem.indexOf(u8, out.written(), prompt) != null);
    if (!c.installed) {
        try t.expectEqual(@as(usize, 1), std.mem.count(u8, out.written(), "TensorFold Web,"));
        if (!c.launches) try t.expect(std.mem.indexOf(u8, out.written(), "uv tool install \"" ++ spec ++ "\"") != null);
    } else try t.expectEqualStrings("", out.written());
    const uv_log = tmp.dir.readFileAlloc(io, "uv.log", a, .limited(4096)) catch "";
    const expected_uv = if (!c.installs) "" else if (c.remote_bin or c.missing_after_install) "tool\ninstall\n" ++ spec ++ "\ntool\ndir\n--bin\n" else "tool\ninstall\n" ++ spec ++ "\n";
    try t.expectEqualStrings(expected_uv, uv_log);
    const launched = tmp.dir.readFileAlloc(io, "launch.log", a, .limited(4096)) catch "";
    try t.expectEqualStrings(if (c.launches) "/example path/tensorfold\n--port\n9090\ntwo words\n$(untouched);*\n\n" else "", launched);
    try t.expectEqualStrings("wrong-binary", env.get("TENSORFOLD_BIN").?);
    if (!c.uv and !c.installed) try t.expect(std.mem.indexOf(u8, out.written(), "python3 -m pip install --user uv") != null);
    if (!c.version.is_release and !c.installed) try t.expect(std.mem.indexOf(u8, out.written(), "Development build without a release tag") != null);
}

test "web: existing PATH app gets exact arguments and binary even without uv or prompting" {
    try check(.{ .installed = true, .uv = false, .interactive = false, .no_prompt = true, .expected = 0, .launches = true });
}

test "web: launched app status is retained" {
    try check(.{ .installed = true, .web_status = 37, .expected = 37, .launches = true });
}

test "web: missing uv prints exact commands and never prompts" {
    try check(.{ .uv = false });
}

test "web: no terminal never installs even with yes on input" {
    try check(.{ .interactive = false });
}

test "web: no-prompt environment prevents installation" {
    try check(.{ .no_prompt = true });
}

test "web: development build never installs" {
    try check(.{ .version = .{ .version = "9.8.7", .is_release = false } });
}

test "web: no, blank, EOF and other replies decline" {
    for ([_][]const u8{ "n\n", "\n", "", "sure\n", "yes please\n" }) |answer| try check(.{ .answer = answer, .asks = true });
}

test "web: affirmative reply pins the binary release then launches" {
    for ([_][]const u8{ "y\n", "YES\n", " Yes \r\n" }) |answer| try check(.{ .answer = answer, .asks = true, .installs = true, .launches = true, .expected = 0 });
}

test "web: successful install outside PATH uses uv tool dir --bin" {
    try check(.{ .remote_bin = true, .asks = true, .installs = true, .launches = true, .expected = 0 });
}

test "web: failed install never launches" {
    try check(.{ .install_fail = true, .asks = true, .installs = true });
}

test "web: missing executable after successful install does not retry" {
    try check(.{ .missing_after_install = true, .asks = true, .installs = true });
}
