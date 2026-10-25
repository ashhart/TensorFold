//! Owned test states model all per-stream recurrent, convolutional, lexical and indexed-attention history.
const std = @import("std");
const c = @import("../../src/families/qwen4/config.zig");
pub const Cache = struct {
    conv: []u16,
    recurrent: []f32,
    history: []u32,
    ple_conv: []u16,
    keys: std.ArrayList([]u16) = .empty,
    values: std.ArrayList([]u16) = .empty,
    index: std.ArrayList([]u16) = .empty,
};
pub const State = struct {
    a: std.mem.Allocator,
    c: *const c.Config,
    cache: []Cache,
    pos: usize = 0,
    poisoned: bool = false,
    pub fn init(a: std.mem.Allocator, config: *const c.Config) !State {
        const caches = try a.alloc(Cache, config.layers);
        for (caches) |*cache| {
            cache.* = .{ .conv = try a.alloc(u16, (config.conv - 1) * config.convWidth()), .recurrent = try a.alloc(f32, config.value_heads * config.value_dim * config.key_dim), .history = try a.alloc(u32, config.ngram - 1), .ple_conv = try a.alloc(u16, (config.ple_conv - 1) * config.ngram * config.wide()) };
            @memset(cache.conv, 0);
            @memset(cache.recurrent, 0);
            @memset(cache.ple_conv, 0);
            @memset(cache.history, config.ple_eos);
        }
        return .{ .a = a, .c = config, .cache = caches };
    }
    pub fn layers(s: *State) usize {
        return s.cache.len;
    }
    pub fn admit(s: *State, n: usize) !void {
        if (s.poisoned) return error.Poisoned;
        if (n > s.c.context - s.pos) return error.ContextFull;
    }
    pub fn advanced(s: *State, tokens: []const u32) !void {
        s.pos += tokens.len;
    }
    pub fn poison(s: *State) void {
        s.poisoned = true;
    }
    pub fn clone(s: *const State, a: std.mem.Allocator) !State {
        var out = try State.init(a, s.c);
        out.pos = s.pos;
        out.poisoned = s.poisoned;
        for (out.cache, s.cache) |*to, from| {
            @memcpy(to.conv, from.conv);
            @memcpy(to.recurrent, from.recurrent);
            @memcpy(to.history, from.history);
            @memcpy(to.ple_conv, from.ple_conv);
            for (from.keys.items) |v| try to.keys.append(a, try a.dupe(u16, v));
            for (from.values.items) |v| try to.values.append(a, try a.dupe(u16, v));
            for (from.index.items) |v| try to.index.append(a, try a.dupe(u16, v));
        }
        return out;
    }
};
