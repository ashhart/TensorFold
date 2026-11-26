//! ``fake_serve serve FIXTURES [flags]``: the scripted engine; TF_FAKE_CONTEXT, _TOKENIZER, _SCRIPTS and _LOAD_FD.
const std = @import("std");
const builtin = @import("builtin");
const server = @import("server");
const FakeText = @import("fake_text.zig").FakeText;
const ScriptEngine = @import("fake_engine.zig").ScriptEngine;

fn announce(port: u16) void {
    var buf: [32]u8 = undefined;
    const line = std.fmt.bufPrint(&buf, "PORT {d}\n", .{port}) catch return;
    server.log.raw(line);
}

pub fn main(init: std.process.Init) !u8 {
    const gpa = init.gpa;
    const io = init.io;
    const a = init.arena.allocator();
    const argv = try init.minimal.args.toSlice(a);
    server.log.init(io, false);
    if (argv.len < 3 or !std.mem.eql(u8, argv[1], "serve")) {
        std.debug.print("usage: fake_serve serve FIXTURES [serve flags]\n", .{});
        return 2;
    }
    var u: server.cli.Usage = .{};
    const args = server.cli.parse(a, argv[2..], &u) catch {
        std.debug.print("tensorfold serve: error: {s}\n", .{u.message});
        return 2;
    };
    // as main does: --listen-fd is taken before the load, and the load runs until TF_FAKE_LOAD_FD says it ends
    const inherited = if (args.listen_fd) |fd| server.listen_fd.open(fd) orelse return 2 else null;
    // the load hold is a POSIX pipe: a Windows fd_t is a HANDLE, and the --listen-fd checks run on POSIX only
    if (builtin.os.tag != .windows) if (init.environ_map.get("TF_FAKE_LOAD_FD")) |text| holdLoad(try std.fmt.parseInt(std.c.fd_t, text, 10));
    const dir = args.model;
    const vocab_bytes = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ dir, "vocab.json" }), a, .limited(1 << 20));
    const vocab = try std.json.parseFromSliceLeaky(std.json.Value, a, vocab_bytes, .{});
    var pieces: std.ArrayList([]const u8) = .empty;
    for (vocab.object.get("pieces").?.array.items) |p| try pieces.append(a, p.string);
    var fake = FakeText.init(pieces.items);
    var problem: []const u8 = "";
    const real = if (init.environ_map.get("TF_FAKE_TOKENIZER")) |path| server.hf_text.HfText.load(gpa, io, path, args.chat_template, a, &problem) catch {
        std.debug.print("tensorfold: {s}\n", .{problem});
        return 1;
    } else null;
    defer if (real) |r| r.deinit();
    const text = if (real) |r| r.text() else fake.text();
    const scripts_path = init.environ_map.get("TF_FAKE_SCRIPTS") orelse try std.fs.path.join(a, &.{ dir, "scripts.json" });
    const scripts_bytes = try std.Io.Dir.cwd().readFileAlloc(io, scripts_path, a, .limited(4 << 20));
    const scripts = try std.json.parseFromSliceLeaky(std.json.Value, a, scripts_bytes, .{});
    const context: u32 = if (init.environ_map.get("TF_FAKE_CONTEXT")) |c| try std.fmt.parseInt(u32, c, 10) else 0;
    var engine: ScriptEngine = .{ .gpa = gpa, .io = io, .text = text, .scripts = scripts, .info_ = .{ .lanes = server.cli.parallel(args.parallel) orelse 8, .context_window = context, .call_gates = true } };
    defer engine.cancelled.deinit(gpa);
    return server.serve.run(gpa, io, args, .{
        .engine = engine.engine(),
        .text = text,
        .served = server.hub.servedName(args.name, args.model, dir),
        .environ = init.environ_map,
        .on_listen = announce,
        .listener = inherited,
    });
}

/// TF_FAKE_LOAD_FD's socket: "t" once --listen-fd is taken, then the load lasts until one byte comes back.
fn holdLoad(fd: std.c.fd_t) void {
    var byte: [1]u8 = undefined;
    _ = std.c.write(fd, "t", 1);
    _ = std.c.read(fd, &byte, 1);
    _ = std.c.close(fd);
}
