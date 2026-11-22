//! Real start/control kernels inside a conditional while, with synthetic probabilities and actual DepthRule as oracle.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const nemo = @import("nemotron");
const rig_ = @import("contract_rig.zig");
const c = core.contract.spec;
const g = core.contract.gates;
const policy = nemo.mtp_graph_policy;
fn noSums(_: c.Shape) usize {
    return 0;
}
pub const Controller = struct {
    rig: rig_.Rig,
    start: cuda.Function,
    step: cuda.Function,
    feed: cuda.Function,
    corrupt: bool = false,
    exec: ?cuda.graph.Exec = null,
    output: u64 = 0,
    want: policy.State = undefined,
    pub fn deinit(self: *Controller) void {
        if (self.exec) |*x| x.deinit();
        self.exec = null;
        self.rig.deinit();
    }
    pub fn kernel(self: *Controller) g.Kernel {
        return .{ .spec = .{ .name = if (self.corrupt) "control: Nemotron MTP controller count byte" else "cuda Nemotron MTP device DepthRule while", .accum = .none, .depth = noSums, .out = .raw, .nan = .{ .stated = "same kept/executed counts, metadata and probability classes as host DepthRule; nonfinite confidence follows comparisons" }, .served = &.{ .{ .rows = 15, .n = 1, .k = 1 }, .{ .rows = 31, .n = 2, .k = 1 }, .{ .rows = 1, .n = 1, .k = 1 } } }, .ctx = self, .vt = &vt, .control = if (self.corrupt) .shapes else null };
    }
    const vt = g.hooks(Controller);
    fn go(self: *Controller, f: cuda.Function, values: []const u64) !void {
        var args: cuda.Args = .{};
        for (values) |x| args.add(x);
        try cuda.launch.launch(f, .{ .grid = .{ .x = 1 }, .block = .{ .x = 1 } }, self.rig.s, &args);
    }
    pub fn prepare(self: *Controller, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows < 1 or shape.rows > 31 or shape.n < 1 or shape.n > 2 or shape.k != 1) return false;
        if (self.exec) |*x| x.deinit();
        self.exec = null;
        self.rig.reset();
        var costs: core.draft_depth.Costs = .{ .rows = 32, .level = 0.71 };
        for (costs.verify[1..], 1..) |*v, rows| v.* = 10 + 0.1 * @as(f64, @floatFromInt(rows));
        const rule = core.draft_depth.DepthRule.init(costs, shape.rows, if (shape.n == 2) 0.4 else null, 2);
        const inputs: policy.State = .{ .policy = policy.Policy.init(rule).? };
        const data = try self.rig.scratch(@sizeOf(policy.State));
        const state_buf: cuda.DeviceBuffer = .{ .d = self.rig.gpu.d, .ptr = data, .len = @sizeOf(policy.State) };
        try state_buf.upload(0, std.mem.asBytes(&inputs));
        const meta = [_]u32{ 17, 23 };
        const meta_dev = try self.rig.input(std.mem.asBytes(&meta));
        var probabilities: [31]f32 = undefined;
        var prng = std.Random.DefaultPrng.init(seed);
        for (&probabilities) |*p| p.* = 0.85 + 0.149 * prng.random().float(f32);
        if (fill == .nonfinite) {
            probabilities[0] = std.math.nan(f32);
            probabilities[1] = std.math.inf(f32);
            probabilities[2] = -std.math.inf(f32);
        }
        const probs = try self.rig.input(std.mem.asBytes(&probabilities));
        const ids_out = try self.rig.scratch(32 * 4);
        const probs_out = try self.rig.scratch(31 * 4);
        self.output = try self.rig.output(@sizeOf(policy.State));
        self.want = inputs;
        self.want.meta = .{ 22, 23 };
        self.want.run = 1;
        for (1..shape.rows + 1) |j| {
            self.want.token = @intCast(j * 17);
            self.want.probability = probabilities[j - 1];
            self.want.levels = @intCast(j);
            self.want.ids[j - 1] = self.want.token;
            self.want.probs[j - 1] = self.want.probability;
            self.want.run *= self.want.probability;
            if (!rule.keep(j, self.want.run)) break;
            self.want.n = @intCast(j);
            if (!rule.more(j, self.want.run)) break;
            self.want.j += 1;
            self.want.meta[0] += 1;
            self.want.meta[1] += 1;
        }
        var library = self.rig.gpu.d.lib;
        const api = cuda.conditional.Api.resolve(&library) orelse return error.GraphNotPermitted;
        var graph = try cuda.graph.Graph.init(self.rig.gpu.d);
        defer graph.deinit();
        const condition = try api.handle(graph, self.rig.gpu.ctx.handle);
        // The first turn and body use the same nodes; start writes j=1 before the first controller.
        try api.begin(self.rig.s, graph, &.{});
        errdefer if ((cuda.graph.captureStatus(self.rig.s) catch .none) != .none) {
            _ = cuda.graph.endCapture(self.rig.s) catch {};
        };
        try self.go(self.start, &.{ data, meta_dev });
        try self.go(self.feed, &.{ data, probs });
        try self.go(self.step, &.{ data, ids_out, probs_out, condition });
        _ = try cuda.graph.endCapture(self.rig.s);
        const node_count = try graph.nodeCount();
        const nodes = try self.rig.gpu.gpa.alloc(cuda.graph.Node, node_count);
        defer self.rig.gpu.gpa.free(nodes);
        const loop = try api.loop(graph, try graph.nodes(nodes), self.rig.gpu.ctx.handle, condition);
        try api.begin(self.rig.s, loop.body, &.{});
        try self.go(self.feed, &.{ data, probs });
        try self.go(self.step, &.{ data, ids_out, probs_out, condition });
        _ = try cuda.graph.endCapture(self.rig.s);
        try api.begin(self.rig.s, graph, &.{loop.node});
        const out: cuda.DeviceBuffer = .{ .d = self.rig.gpu.d, .ptr = self.output, .len = @sizeOf(policy.State) };
        try out.copyFrom(0, data, @sizeOf(policy.State), self.rig.s.handle);
        _ = try cuda.graph.endCapture(self.rig.s);
        self.exec = try cuda.conditional.Api.instantiate(graph);
        return true;
    }
    pub fn run(self: *Controller) !void {
        try self.exec.?.launchOn(self.rig.s);
        if (self.corrupt) {
            const at = self.output + @offsetOf(policy.State, "n");
            const bytes = try self.rig.bytes(at, 4);
            bytes[0] ^= 1;
            const b: cuda.DeviceBuffer = .{ .d = self.rig.gpu.d, .ptr = at, .len = 4 };
            try b.upload(0, bytes);
        }
    }
    pub fn count(_: *Controller) usize {
        return @sizeOf(policy.State);
    }
    pub fn reference(_: *Controller, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
    }
    pub fn check(self: *Controller, _: g.Gate) !?[]const u8 {
        const bytes = try self.rig.outputBytes();
        var actual: policy.State = undefined;
        @memcpy(std.mem.asBytes(&actual), bytes);
        const got = &actual;
        const want = &self.want;
        if (got.n != want.n or got.levels != want.levels or got.j != want.j or !std.mem.eql(u32, &got.meta, &want.meta)) return "device count/level/position decision differs from host DepthRule";
        for (0..want.levels) |j| if (got.ids[j] != want.ids[j] or @as(u32, @bitCast(got.probs[j])) != @as(u32, @bitCast(want.probs[j]))) return "device controller changed a draft or its probability";
        if (!(std.math.isNan(got.run) and std.math.isNan(want.run)) and @as(u64, @bitCast(got.run)) != @as(u64, @bitCast(want.run))) return "cumulative confidence differs from the host";
        return null;
    }
};
