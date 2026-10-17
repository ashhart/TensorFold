//! The forward's later-layer entry points (prefill, Engram gate, round graph, greedy, images, branches), set at open.
const std = @import("std");
const cuda = @import("cuda");
const calls = @import("block_calls.zig");
const buffers = @import("buffers.zig");
const run = @import("run.zig");
const eh = @import("engram_host.zig");
const ft = @import("forward_types.zig");
const Forward = @import("forward.zig").Forward;

/// forward_prefill.zig: the prompt's segments, the CED decoder replay and the prefill's state on the forward.
pub const Prefill = struct {
    plan: *const fn (f: *Forward, p: *buffers.Plan, rows: u32) anyerror!void,
    prompt: *const fn (f: *Forward, ids: []const u32) anyerror!void,
    promptMulti: *const fn (f: *Forward, segs: []const ft.MultiSeg) anyerror!void,
    finish: *const fn (f: *Forward) anyerror!void,
    finishMulti: *const fn (f: *Forward, slots: []const u32, out: []ft.Replayed) anyerror!void,
    ahead: *const fn (f: *Forward) ?*std.ArrayList(u32), // the read-ahead ids' list; null: TF_DSV41_PREFETCH_AHEAD off
    replaying: *const fn (f: *const Forward) bool, // TF_DSV41_PREFILL=replay
    deinit: *const fn (f: *Forward) void,
};

/// engram_gate_fwd.zig: TF_DSV41_ENGRAM_GATE's gate (`Forward.gate`) opened with the table.
pub const Gate = struct {
    open: *const fn (f: *Forward, h: *const eh.Host) anyerror!void,
    source: *const fn (g: *anyopaque) eh.Source,
    eligible: *const fn (g: *anyopaque, rows: usize) bool,
    arm: *const fn (g: *anyopaque, items: []const ft.Item, fill: u32) anyerror!void,
    check: *const fn (g: *anyopaque) anyerror!void,
    copy: *const fn (f: *Forward, r: *run.Runner, L: u32, dst: u64, n: usize) anyerror!void,
    warm: *const fn (g: *anyopaque, items: []const ft.Item) void,
    close: *const fn (g: *anyopaque) void,
};

/// round_graph.zig: TF_DSV41_ROUND_GRAPH's settings and a window's split point.
pub const RoundGraph = struct {
    current: *const fn () anyerror!ft.RoundGraph,
    splitAt: *const fn (cs: []const calls.Call, head: u32) ?usize,
};

/// greedy.zig: the host's best column of a row.
pub const Greedy = struct { argmax: *const fn (vals: []const f32) usize };

/// vision_rows.zig: rank 0's image rows (`Forward.vision`), the no-image window check and the prefill's image glue.
pub const Vision = struct {
    fill: *const fn (v: *anyopaque, s: cuda.Stream, ids: []const u32, rows: u64, D: usize) anyerror!void,
    checkWindow: *const fn (ids: []const u32) anyerror!void,
    keep: *const fn (f: *Forward, r: *run.Runner, c: *const calls.Call) anyerror!void, // glue image_keep
    split: *const fn (f: *Forward, r: *run.Runner, c: *const calls.Call, merge: bool) anyerror!void, // or image_merge
};

/// branches.zig: the runner's side streams (`Forward.branches`) released.
pub const Branches = struct { close: *const fn (f: *Forward) void };

pub const Hooks = struct {
    prefill: ?*const Prefill = null,
    gate: ?*const Gate = null,
    round_graph: ?*const RoundGraph = null,
    greedy: ?*const Greedy = null,
    vision: ?*const Vision = null,
    branches: ?*const Branches = null,
};

/// The table every forward uses; an object a later layer made (gate, images, branches) means its part is set.
pub var set: Hooks = .{};

/// Engine open (engine_hooks.zig) sets every part before the first forward runs.
pub fn install(h: Hooks) void {
    set = h;
}

fn missing(comptime name: []const u8) error{ForwardHookUnset} {
    std.log.scoped(.dsv41).err("forward hook `" ++ name ++ "` unset: engine open calls engine_hooks.open first", .{});
    return error.ForwardHookUnset;
}

pub fn prefill() !*const Prefill {
    return set.prefill orelse missing("prefill");
}

pub fn gate() !*const Gate {
    return set.gate orelse missing("gate");
}

pub fn roundGraph() !*const RoundGraph {
    return set.round_graph orelse missing("round_graph");
}

pub fn greedy() !*const Greedy {
    return set.greedy orelse missing("greedy");
}

pub fn vision() !*const Vision {
    return set.vision orelse missing("vision");
}
