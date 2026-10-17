//! TF_DSV41_ENGRAM_GATE on the forward: the gate opened with the Engram table, behind forward_hooks.Gate.
const std = @import("std");
const quiet = @import("builtin").is_test; // a test's stderr is the build runner's channel
const egate = @import("engram_gate.zig");
const eh = @import("engram_host.zig");
const run = @import("run.zig");
const round_graph = @import("round_graph.zig");
const hooks = @import("forward_hooks.zig");
const ft = @import("forward_types.zig");
const Forward = @import("forward.zig").Forward;
const Gate = egate.Gate;

pub const table: hooks.Gate = .{ .open = open, .source = source, .eligible = eligible, .arm = arm, .check = check, .copy = gateCopy, .warm = warm, .close = close };

/// The forward's gate (`Forward.gate`), null when off.
pub fn of(f: *const Forward) ?*Gate {
    return @ptrCast(@alignCast(f.gate orelse return null));
}

fn self(g: *anyopaque) *Gate {
    return @ptrCast(@alignCast(g));
}

/// TF_DSV41_ENGRAM_GATE on: the gate over the packed table, its arm's wait picked by TF_DSV41_ROUND_GRAPH.
fn open(f: *Forward, h: *const eh.Host) anyerror!void {
    const s = try egate.settings();
    if (s.on) {
        if (f.engram_rows) |x| {
            var la = std.heap.ArenaAllocator.init(f.gpa);
            defer la.deinit();
            var layers: std.ArrayList(u32) = .empty;
            for (try f.backbone(la.allocator())) |L| if (std.mem.indexOfScalar(u32, h.layer_ids, L) != null) try layers.append(la.allocator(), L);
            const g = try egate.Gate.open(std.heap.c_allocator, f.runner.d, h, x.source(), layers.items, f.comm.rank(), f.comm.world(), f.cfg.engram_head_dim, s);
            g.sync = if ((try round_graph.current()).gate) .{ .ctx = f, .run = syncGatedEnd } else .{ .ctx = f.runner, .run = syncRunner };
            f.gate = g;
        } else if (!quiet) std.log.scoped(.dsv41).warn("engram gate off: no packed table (TF_DSV41_ENGRAM_DIR)", .{});
    }
}

fn source(g: *anyopaque) eh.Source {
    return self(g).source();
}

fn eligible(g: *anyopaque, rows: usize) bool {
    return self(g).eligible(rows);
}

fn arm(g: *anyopaque, items: []const ft.Item, fill: u32) anyerror!void {
    return self(g).arm(items, fill);
}

fn check(g: *anyopaque) anyerror!void {
    return self(g).check();
}

fn warm(g: *anyopaque, items: []const ft.Item) void {
    self(g).warm(items);
}

fn close(g: *anyopaque) void {
    self(g).close();
}

pub fn syncRunner(ctx: *anyopaque) anyerror!void {
    const r: *run.Runner = @ptrCast(@alignCast(ctx));
    try r.stream.synchronize();
}

/// TF_DSV41_ROUND_GRAPH's `gate`: the gate's arm waits for the last gated window's end, not all queued work.
pub fn syncGatedEnd(ctx: *anyopaque) anyerror!void {
    const f: *Forward = @ptrCast(@alignCast(ctx));
    if (f.gated_end) |e| return e.synchronize();
    try f.runner.stream.synchronize();
}

/// The gate kernel of Engram layer L: `n` rows of the armed round's staging into `dst` once published.
fn gateCopy(f: *Forward, r: *run.Runner, L: u32, dst: u64, n: usize) anyerror!void {
    const g = of(f).?;
    const k = g.slotOf(L) orelse return error.NotEngram;
    return r.kernels.others(r.stream).engramGate(g.mem.ctl_dev, @intCast(egate.ctl_ready + k), egate.ctl_err, g.layerBase(k), g.parityStride(), dst, @intCast(2 * n * g.cols), g.timeout_ns, true);
}
