//! A production-mode CLI root with the --tp capability both rank-capable backends publish.
const std = @import("std");
const cli = @import("cli.zig");
const config = @import("cli_config.zig");

test {
    _ = @import("cli_capabilities.zig");
}

pub const serves: []const []const u8 = &.{"--tp"};

pub fn main(init: std.process.Init) !void {
    var arena: std.heap.ArenaAllocator = .init(init.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    if (!cli.find("--rank-secret").?.native) return error.RankFlagNotAdvertised;
    var usage: cli.Usage = .{};
    const io = init.io;
    const cwd = std.Io.Dir.cwd();
    const folder = ".rank-secret-config-fixture";
    const key = ".rank-secret-config-key";
    try cwd.createDir(io, folder, .default_dir);
    defer cwd.deleteTree(io, folder) catch {};
    const permissions: std.Io.File.Permissions = if (@import("builtin").os.tag == .windows) .default_file else .fromMode(0o600);
    const before = try cwd.createFile(io, key, .{ .exclusive = true, .permissions = permissions });
    before.close(io);
    defer cwd.deleteFile(io, key) catch {};
    const sep = std.fs.path.sep_str;
    const conf = folder ++ sep ++ "conf";
    try cwd.createDirPath(io, conf);
    const after_path = conf ++ sep ++ key;
    const after = try cwd.createFile(io, after_path, .{ .exclusive = true, .permissions = permissions });
    after.close(io);
    const bytes = "a fictional group key of32 bytes!!";
    try cwd.writeFile(io, .{ .sub_path = key, .data = bytes });
    try cwd.writeFile(io, .{ .sub_path = after_path, .data = bytes });
    const argv: []const []const u8 = &.{ "--config", conf ++ sep ++ "f.toml" };
    const text =
        \\rank-secret = ".rank-secret-config-key"
    ;
    const out = try config.merge(a, argv, (try config.locate(a, argv, &usage)).?, text, &usage);
    const want = "--rank-secret=" ++ conf ++ sep ++ key;
    if (out.argv.len != 1) return error.ConfigArgumentCount;
    if (!std.mem.eql(u8, want, out.argv[0])) {
        std.debug.print("rank secret config: expected {s}; actual {s}\n", .{ want, out.argv[0] });
        return error.ConfigRelativeKeyPath;
    }
}
