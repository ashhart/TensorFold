//! Dory's sublayer order: input, the recurrent block `loops` times, output; per_loop KV caches, one an attention step.
const std = @import("std");
const config = @import("config.zig");

const Config = config.Config;
const Kind = config.Kind;

pub const max_steps = config.max_sublayers * config.max_loops;

/// One virtual sublayer: its weights (`sub`), loop, cache (attention only) and the loop's entry and exit norms.
pub const Step = struct {
    sub: u16,
    kind: Kind,
    loop: u8 = 0, // 0 outside the recurrent block
    cache: ?u16 = null,
    /// h = l2(state + input_h) before the sublayer; at loop 0 input_h = state = h first.
    enter: bool = false,
    /// state = l2(h) after the sublayer: the loop's last.
    leave: bool = false,
};

pub const Schedule = struct {
    steps: [max_steps]Step = undefined,
    len: usize = 0,
    caches: usize = 0,
    cache_kind: [max_steps]Kind = undefined, // sliding or full, by cache

    pub fn slice(s: *const Schedule) []const Step {
        return s.steps[0..s.len];
    }

    /// Caches of `kind` (sliding or full).
    pub fn cacheCount(s: *const Schedule, kind: Kind) usize {
        var n: usize = 0;
        for (s.cache_kind[0..s.caches]) |k| n += @intFromBool(k == kind);
        return n;
    }

    /// vLLM #60149's attention layer index of a step: loop * sublayers + sub (its per_loop virtual layers).
    pub fn vllmLayer(c: Config, st: Step) usize {
        return @as(usize, st.loop) * c.sublayers + st.sub;
    }
};

/// The steps of `c` at `c.loops` loops: I + R * loops + O, caches numbered in the order the steps run.
pub fn build(c: Config) Schedule {
    var s: Schedule = .{};
    const add = struct {
        fn one(sch: *Schedule, cc: Config, sub: usize, loop: usize) void {
            var st: Step = .{ .sub = @intCast(sub), .kind = cc.kinds[sub], .loop = @intCast(loop) };
            if (st.kind != .mlp) {
                st.cache = @intCast(sch.caches);
                sch.cache_kind[sch.caches] = st.kind;
                sch.caches += 1;
            }
            sch.steps[sch.len] = st;
            sch.len += 1;
        }
    }.one;
    for (0..c.input) |i| add(&s, c, i, 0);
    for (0..c.loops) |l| {
        for (c.input..c.input + c.recurrent) |i| add(&s, c, i, l);
        s.steps[s.len - c.recurrent].enter = true;
        s.steps[s.len - 1].leave = true;
    }
    for (c.input + c.recurrent..c.sublayers) |i| add(&s, c, i, 0);
    return s;
}

const testing = std.testing;

fn pinnedConfig() !Config {
    const text = try config.pinnedText(testing.allocator);
    defer testing.allocator.free(text);
    var why: config.Why = .{};
    return config.parse(testing.allocator, text, &why);
}

/// llama.cpp #30306's `physical` and per_loop `cache_layer` (llama_model_dory::load_arch_hparams at 3834b4ae).
fn llamaCpp(c: Config, physical: []u32, cache_layer: []u32) usize {
    var n: usize = 0;
    for (0..c.input) |i| {
        physical[n] = @intCast(i);
        n += 1;
    }
    for (0..c.loops) |_| for (0..c.recurrent) |i| {
        physical[n] = @intCast(c.input + i);
        n += 1;
    };
    for (0..c.output) |i| {
        physical[n] = @intCast(c.input + c.recurrent + i);
        n += 1;
    }
    for (0..n) |il| cache_layer[il] = @intCast(il);
    return n;
}

test "the schedule at 1 to 4 loops: llama.cpp's physical sublayers, a cache an attention step, 8 + 20 L + 8 caches" {
    const base = try pinnedConfig();
    for (1..5) |loops| {
        const c = try base.withLoops(loops);
        const s = build(c);
        try testing.expectEqual(16 + 40 * loops + 16, s.len);
        try testing.expectEqual(8 + 20 * loops + 8, s.caches);
        try testing.expectEqual(4 + 5 * loops, s.cacheCount(.full));
        try testing.expectEqual(12 + 15 * loops, s.cacheCount(.sliding));
        var physical: [max_steps]u32 = undefined;
        var layer: [max_steps]u32 = undefined;
        try testing.expectEqual(s.len, llamaCpp(c, &physical, &layer));
        // llama.cpp numbers its caches by virtual sublayer and skips the MLP ones: the same order as ours
        var seen: usize = 0;
        for (s.slice(), 0..) |st, il| {
            try testing.expectEqual(physical[il], st.sub);
            if (st.kind == .mlp) {
                try testing.expectEqual(@as(?u16, null), st.cache);
                continue;
            }
            try testing.expectEqual(@as(?u16, @intCast(seen)), st.cache);
            try testing.expect(seen == 0 or layer[il] > layer[il - 1]);
            seen += 1;
        }
        var enters: usize = 0;
        var leaves: usize = 0;
        for (s.slice(), 0..) |st, i| {
            if (st.enter) {
                enters += 1;
                try testing.expectEqual(@as(u16, 16), st.sub);
                try testing.expectEqual(16 + 40 * (enters - 1), i);
            }
            if (st.leave) {
                leaves += 1;
                try testing.expectEqual(@as(u16, 55), st.sub);
                try testing.expectEqual(st.loop, @as(u8, @intCast(leaves - 1)));
            }
        }
        try testing.expectEqual(loops, enters);
        try testing.expectEqual(loops, leaves);
    }
}

test "vLLM's per_loop virtual layers are unique per (loop, sublayer) and keep the physical index at loop 0" {
    const c = try (try pinnedConfig()).withLoops(4);
    const s = build(c);
    var used: [config.max_sublayers * config.max_loops]bool = @splat(false);
    for (s.slice()) |st| {
        if (st.kind == .mlp) continue;
        const v = Schedule.vllmLayer(c, st);
        try testing.expect(!used[v]);
        used[v] = true;
        if (st.loop == 0) try testing.expectEqual(@as(usize, st.sub), v);
        if (st.loop > 0) try testing.expect(st.sub >= 16 and st.sub < 56);
    }
    try testing.expectEqual(@as(usize, 22 + 72), Schedule.vllmLayer(c, .{ .sub = 22, .kind = .full, .loop = 1 }));
}
