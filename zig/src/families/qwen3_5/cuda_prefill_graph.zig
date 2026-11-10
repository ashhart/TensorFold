//! Opt-in per-call prompt graph: no cache, no borrowed pointer survives the forward's completion.
const std = @import("std");
const cuda = @import("cuda");
pub const Body = cuda.graph_cache.Body;

pub fn selected(raw: ?[]const u8) !bool {
    const text = raw orelse return false;
    if (std.mem.eql(u8, text, "0")) return false;
    if (std.mem.eql(u8, text, "1")) return true;
    return error.BadPrefillGraph;
}

pub fn requested() !bool {
    return selected(try @import("sys").environment.given("TF_QWEN_PREFILL_GRAPH"));
}

pub const Facts = struct {
    dense27: bool = true,
    one_segment: bool = true,
    prompt: bool = true,
    prepared: bool = true,
    learning: bool = false,
    pockets: bool = false,
    images: bool = false,
    int8: bool = false,
    vendor: bool = false,
    profiling: bool = false,
    hooks: bool = false,

    pub fn admits(f: Facts) bool {
        return f.dense27 and f.one_segment and f.prompt and f.prepared and !f.learning and !f.pockets and
            !f.images and !f.int8 and !f.vendor and !f.profiling and !f.hooks;
    }
};

fn prepared(e: anytype, rows: usize) bool {
    if (e.arena.ptr == 0 or e.tables.ptr == 0) return false;
    if (e.mo) |m| {
        if (m.native_requested and m.native == null) return false;
        if (m.native) |n| {
            if (n.scratch.ptr == 0 or rows > n.rows) return false;
            if (n.tma != null and (n.workspace == null or n.prepared[0] == null or n.prepared[1] == null)) return false;
        }
        if (m.fast8) |n| if (n.scratch.ptr == 0 or rows > n.max_rows) return false;
    }
    return true;
}

/// Setup and uploads precede this call; the returned executable must live through the caller's completion wait.
pub fn run(e: anytype, shape: anytype) !?cuda.graph.Exec {
    if (!try requested()) {
        try e.body(shape);
        return null;
    }
    const profiling = try @import("sys").environment.read(bool, "TF_QWEN_PREFILL_PROFILE", false);
    const facts: Facts = .{
        .dense27 = @import("cuda_geometry.zig").dense27(e.g),
        .one_segment = shape.nseg == 1,
        .prompt = shape.prompt and !shape.record and !shape.hidden,
        .prepared = prepared(e, shape.total),
        .learning = e.learning != null,
        .pockets = e.pockets.any(),
        .images = e.features != 0,
        .int8 = e.int8.on,
        .vendor = if (e.mo) |m| m.vendor != null else false,
        .profiling = profiling,
        .hooks = e.mlp != null,
    };
    const Call = struct {
        engine: @TypeOf(e),
        shape: @TypeOf(shape),
        fn body(p: *anyopaque, _: cuda.Stream) !void {
            const call: *@This() = @ptrCast(@alignCast(p));
            try call.engine.body(call.shape);
        }
    };
    var call: Call = .{ .engine = e, .shape = shape };
    return execute(true, facts, e.stream, .{ .ctx = &call, .run = Call.body });
}

/// A capture or launch error is never retried eagerly: a failed launch may already have updated recurrence state.
pub fn execute(on: bool, facts: Facts, stream: cuda.Stream, body: Body) !?cuda.graph.Exec {
    if (!on or !facts.admits()) {
        try body.run(body.ctx, stream);
        return null;
    }
    var open = false;
    var owned: ?cuda.graph.Exec = null;
    errdefer {
        if (open) abort(stream);
        cuda.completion.drain(stream.d, stream, "failed prompt graph did not drain before releasing its executable");
        if (owned) |*exec| exec.deinit();
    }
    try cuda.graph.beginCapture(stream, .thread_local);
    open = true;
    try body.run(body.ctx, stream);
    var captured = try cuda.graph.endCapture(stream);
    open = false;
    defer captured.deinit();
    owned = try captured.instantiate();
    try owned.?.upload(stream);
    try owned.?.launchOn(stream);
    return owned;
}

fn abort(stream: cuda.Stream) void {
    if ((cuda.graph.captureStatus(stream) catch .active) == .none) return;
    var broken = cuda.graph.endCapture(stream) catch return;
    broken.deinit();
}

test "prompt graph: active native paths require their preallocated scratch and TMA weight buffers" {
    const B = struct { ptr: u64 = 1 };
    const Native = struct { scratch: B = .{}, rows: usize = 8192, tma: ?u8 = null, workspace: ?u8 = null, prepared: [2]?u8 = .{ null, null } };
    const Fast = struct { scratch: B = .{}, max_rows: usize = 8192 };
    const Model = struct { native_requested: bool = true, native: ?Native = .{}, fast8: ?Fast = .{} };
    const Engine = struct { arena: B = .{}, tables: B = .{}, mo: ?Model = .{} };
    var e: Engine = .{};
    try std.testing.expect(prepared(e, 7680));
    e.mo.?.native.?.tma = 1;
    try std.testing.expect(!prepared(e, 7680));
    e.mo.?.native.?.workspace = 1;
    e.mo.?.native.?.prepared[0] = 1;
    try std.testing.expect(!prepared(e, 7680));
    e.mo.?.native.?.prepared[1] = 1;
    try std.testing.expect(prepared(e, 7680));
    e.mo.?.fast8.?.scratch.ptr = 0;
    try std.testing.expect(!prepared(e, 7680));
    e.mo.?.fast8 = null;
    try std.testing.expect(!prepared(e, 8193));
    e.mo.?.native = null;
    try std.testing.expect(!prepared(e, 7680));
}

test {
    _ = @import("cuda_prefill_graph_test.zig");
}
