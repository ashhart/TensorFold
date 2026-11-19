//! The shared compile path takes no lock of its own: one caller's slow compile never holds up another's.
const std = @import("std");
const grammar = @import("grammar.zig");
const fake = @import("grammar_fake.zig");
const t = std.testing;

const Backend = struct {
    entered: std.atomic.Value(bool) = .init(false),
    release: std.atomic.Value(bool) = .init(false),
    fn compile(ctx: *anyopaque, _: c_int, text: [*]const u8, len: usize, _: [*]u8, _: usize) callconv(.c) ?*anyopaque {
        const b: *Backend = @ptrCast(@alignCast(ctx));
        if (std.mem.eql(u8, text[0..len], "slow")) {
            b.entered.store(true, .release);
            while (!b.release.load(.acquire)) std.Io.sleep(t.io, .fromMilliseconds(1), .awake) catch {};
        }
        return @ptrFromInt(17);
    }
    fn free(_: *anyopaque) callconv(.c) void {}
};

const Caller = struct {
    compiler: *grammar.Compiler,
    text: []const u8,
    done: std.atomic.Value(bool) = .init(false),
    result: ?grammar.Compiled = null,
    fn run(w: *Caller) void {
        var err: [512]u8 = undefined;
        w.result = w.compiler.compile(t.io, .json_schema, w.text, &err) catch null;
        w.done.store(true, .release);
    }
};

fn within(flag: *std.atomic.Value(bool), ms: usize) bool {
    for (0..ms) |_| {
        if (flag.load(.acquire)) return true;
        std.Io.sleep(t.io, .fromMilliseconds(1), .awake) catch {};
    }
    return flag.load(.acquire);
}

test "grammar compile: a small request finishes while another caller is still compiling" {
    var backend: Backend = .{};
    var compiler = fake.compiler();
    compiler.h = &backend;
    compiler.f.compile = Backend.compile;
    compiler.f.free = Backend.free;
    var large: Caller = .{ .compiler = &compiler, .text = "slow" };
    const first = try std.Thread.spawn(.{}, Caller.run, .{&large});
    defer first.join();
    defer backend.release.store(true, .release);
    try t.expect(within(&backend.entered, 2000));
    var small: Caller = .{ .compiler = &compiler, .text = "small" };
    const second = try std.Thread.spawn(.{}, Caller.run, .{&small});
    const passed = within(&small.done, 2000);
    const running = !large.done.load(.acquire);
    backend.release.store(true, .release);
    second.join();
    try t.expect(passed and small.result != null and running);
}
