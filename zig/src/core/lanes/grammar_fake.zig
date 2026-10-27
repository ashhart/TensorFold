//! A stand-in libtfgrammar for tests: each grammar allows its own token pattern, and compiles and frees are counted.
const std = @import("std");
const grammar = @import("grammar.zig");

pub var compiles: usize = 0;
pub var frees: usize = 0;
pub var arms: usize = 0; // grammar_budget: matchers armed, and masks it was asked to narrow (it narrows none)
pub var narrows: usize = 0;

/// Grammar `g` at step `k` allows token t when (t + g + k) % 3 == 0 (so masks tell grammars and steps apart).
const State = struct { g: usize, k: usize };

fn compile(_: *anyopaque, kind: c_int, text: [*]const u8, len: usize, err: [*]u8, _: usize) callconv(.c) ?*anyopaque {
    if (std.mem.eql(u8, text[0..len], "bad")) {
        err[0] = 0;
        return null;
    }
    compiles += 1;
    return @ptrFromInt(1 + @as(usize, @intCast(kind)) * 1000 + len);
}

fn free(_: *anyopaque) callconv(.c) void {
    frees += 1;
}

fn matcher(g: *anyopaque) callconv(.c) ?*anyopaque {
    const m = std.heap.page_allocator.create(State) catch return null;
    m.* = .{ .g = @intFromPtr(g), .k = 0 };
    return m;
}

fn matcherFree(m: *anyopaque) callconv(.c) void {
    std.heap.page_allocator.destroy(@as(*State, @ptrCast(@alignCast(m))));
}

fn accept(p: *anyopaque, token: i32) callconv(.c) c_int {
    const m: *State = @ptrCast(@alignCast(p));
    if ((@as(usize, @intCast(token)) + m.g + m.k) % 3 != 0) return 0;
    m.k += 1;
    return 1;
}

fn rollback(p: *anyopaque, n: c_int) callconv(.c) c_int {
    const m: *State = @ptrCast(@alignCast(p));
    m.k -= @intCast(n);
    return 0;
}

fn terminated(_: *anyopaque) callconv(.c) c_int {
    return 0;
}

fn fill(p: *anyopaque, words: [*]i32, n: c_int) callconv(.c) c_int {
    const m: *State = @ptrCast(@alignCast(p));
    const out: [*]u32 = @ptrCast(words);
    for (0..@intCast(n)) |w| {
        var bits: u32 = 0;
        for (0..32) |b| if ((w * 32 + b + m.g + m.k) % 3 == 0) {
            bits |= @as(u32, 1) << @intCast(b);
        };
        out[w] = bits;
    }
    return 0;
}

fn drafts(_: *anyopaque, _: [*]i32, _: c_int) callconv(.c) c_int {
    return 0;
}

fn budget(_: *anyopaque, _: [*]i32, n: c_int, _: c_int) callconv(.c) c_int {
    if (n == 0) arms += 1 else narrows += 1;
    return 0;
}

fn budgetReady(_: *anyopaque) callconv(.c) c_int {
    return 1;
}

fn unused(_: *anyopaque) callconv(.c) void {}

fn open(_: [*]const u8, _: usize, _: c_int, _: [*]const i32, _: c_int, _: [*]u8, _: usize) callconv(.c) ?*anyopaque {
    return null;
}

fn wordCount(_: *anyopaque) callconv(.c) c_int {
    return 2;
}

const library: grammar.Library = .{
    .open = open,
    .close = unused,
    .words = wordCount,
    .compile = compile,
    .free = free,
    .matcher = matcher,
    .matcher_free = matcherFree,
    .accept = accept,
    .rollback = rollback,
    .terminated = terminated,
    .fill = fill,
    .drafts = drafts,
    .budget = budget,
    .budget_ready = budgetReady,
};

/// The stand-in installed (give the returned library back to grammar.swapLibrary after) and a two-word compiler.
pub fn install() ?grammar.Library {
    compiles = 0;
    frees = 0;
    arms = 0;
    narrows = 0;
    return grammar.swapLibrary(library);
}

pub fn compiler() grammar.Compiler {
    return .{ .f = library, .h = @ptrFromInt(8), .words = 2 };
}
