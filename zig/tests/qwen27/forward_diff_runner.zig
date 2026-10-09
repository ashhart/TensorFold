const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const q = tf.qwen27;
const Ref = q.projection.Ref;
const Frame = q.gpu_frame.Frame;
const Weights = q.gpu_weights;
const Round = q.round_plan.Round;

const Hooks = struct {
    runner: *q.decode_round.Runner,
    fn gdn(ptr: *anyopaque, e: mtl.ComputeEncoder, block: usize, weights: Weights.Gdn, frame: *Frame, rows: u32) !void {
        const h: *Hooks = @ptrCast(@alignCast(ptr));
        const r = h.runner;
        var hook = q.gdn_hook.Hook{ .pool = &r.gdn, .quant = &r.model.kernels, .glue = r.model.glue, .plan = r.current_gdn.?, .linear_index = r.linear_index };
        try q.gdn_hook.Hook.forward(&hook, e, block, weights, frame, rows);
    }
    fn attention(ptr: *anyopaque, e: mtl.ComputeEncoder, block: usize, weights: Weights.Attention, f: *Frame, rows: u32) !void {
        const h: *Hooks = @ptrCast(@alignCast(ptr));
        const r = h.runner;
        const plan = r.current_attention.?.*;
        const layer = r.attention_index[block];
        const cache = r.caches[layer];
        try r.bindings[layer].upload(plan, &.{cache});
        try r.model.kernels.quant(e, weights.q, f.get(.input), q.forward.sums(f, .sums, rows), f.get(.dims), f.get(.q_gate), rows);
        try r.model.kernels.quant(e, weights.kv, f.get(.input), q.forward.sums(f, .sums, rows), f.get(.dims), f.get(.mixed), rows);
        e.barrier();
        try r.model.glue.unstack(e, f.get(.mixed), f.get(.key), .{ .rows = rows, .width = 1024, .stride = 2048, .offset = 0 });
        try r.model.glue.unstack(e, f.get(.mixed), f.get(.value), .{ .rows = rows, .width = 1024, .stride = 2048, .offset = 1024 });
        e.barrier();
        try r.attention.preprocess(e, .{ .qg = f.get(.q_gate), .key = f.get(.key), .q_gain = Weights.ref(weights.q_norm), .k_gain = Weights.ref(weights.k_norm), .norm_q = f.get(.norm_query), .norm_k = f.get(.norm_key), .query = f.get(.query), .rotated_key = f.get(.rotated_key), .positions = f.get(.positions), .rows = rows, .eps = r.model.config.eps, .theta = r.model.config.rope_theta });
        try r.attention.encode(e, &r.scratch, &r.bindings[layer], plan, f.get(.query), f.get(.rotated_key), f.get(.value), f.get(.mixer_out), &.{cache}, f.get(.q_gate), false);
    }
    fn tap(ptr: *anyopaque, e: mtl.ComputeEncoder, layer: usize, hidden: Ref, rows: u32) !void {
        const h: *Hooks = @ptrCast(@alignCast(ptr));
        try h.runner.taps.record(h.runner.model.glue, e, layer, hidden, rows);
    }
};

fn Wrapped(comptime S: type) type {
    return struct {
        inner: S,
        abort_at: ?usize,
        call: usize = 0,
        perturb: bool,
        fn tick(s: *@This()) !void {
            if (s.abort_at == s.call) return error.ControlCanceled;
            s.call += 1;
        }
        pub fn begin(s: *@This(), lane: *tf.segments.Lane) !void {
            try s.tick();
            try s.inner.begin(lane);
        }
        pub fn wait(s: *@This(), i: usize) tf.segments.Wait {
            return s.inner.wait(i);
        }
        pub fn handoff(s: *@This(), lane: *tf.segments.Lane, i: usize) !void {
            try s.inner.handoff(lane, i);
        }
        pub fn pre(s: *@This(), lane: *tf.segments.Lane, i: usize) !void {
            try s.tick();
            try s.inner.pre(lane, i);
        }
        pub fn mixer(s: *@This(), lane: *tf.segments.Lane, i: usize) !void {
            try s.tick();
            try s.inner.mixer(lane, i);
        }
        pub fn post(s: *@This(), lane: *tf.segments.Lane, i: usize) !void {
            try s.tick();
            try s.inner.post(lane, i);
            if (s.perturb) s.inner.pending = s.inner.model.frame.get(.input);
        }
        pub fn finish(s: *@This(), lane: *tf.segments.Lane) !void {
            try s.tick();
            try s.inner.finish(lane);
        }
    };
}

pub fn verify(comptime S: type, r: *q.decode_round.Runner, round: *const Round, head: q.forward.Head, abort_at: ?usize, perturb: bool) !void {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const params = try r.gdn.budget.shape.params(round.ids.len, r.slots, .{ .conv = .bf16, .a_log = .bf16, .dt = .bf16, .norm = .bf16 });
    var recurrent = try round.recurrent(params, &.{&.{}});
    defer recurrent.deinit();
    var attention = try q.attention.Plan.init(r.allocator, &.{.{ .parents = round.windows[0].parents, .start = round.windows[0].start, .capacity = r.capacity }}, .{ .shared_prefix = r.model.device.tensorUnits() });
    defer attention.deinit();
    try r.model.frame.setRows(round.ids, round.feed_positions);
    try r.scratch.upload(attention);
    r.current_round = round;
    r.current_gdn = &recurrent;
    r.current_attention = &attention;
    defer {
        r.current_round = null;
        r.current_gdn = null;
        r.current_attention = null;
    }
    r.taps.begin();
    var hooks = Hooks{ .runner = r };
    var stages = Wrapped(S){ .inner = .{ .model = r.model, .rows = @intCast(round.ids.len), .hooks = .{ .ptr = &hooks, .gdn = Hooks.gdn, .attention = Hooks.attention, .tap = Hooks.tap }, .head = @fromBackingInt(@intCast(@backingInt(head))) }, .abort_at = abort_at, .perturb = perturb };
    _ = try tf.segments.run(r.model.device, &.{r.model.queue}, r.model.config.layers, .serial, &stages);
    try r.taps.complete();
    r.active = fingerprint(round);
}

fn fingerprint(round: *const Round) [32]u8 {
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    hash.update(std.mem.sliceAsBytes(round.ids));
    hash.update(std.mem.sliceAsBytes(round.slots));
    hash.update(std.mem.sliceAsBytes(round.feed_positions));
    hash.update(std.mem.sliceAsBytes(round.draw_positions));
    hash.update(std.mem.sliceAsBytes(round.firsts));
    for (round.windows) |window| {
        hash.update(std.mem.asBytes(&window.start));
        hash.update(std.mem.asBytes(&window.taps));
        hash.update(std.mem.sliceAsBytes(window.parents));
        hash.update(std.mem.sliceAsBytes(window.conv));
    }
    var out: [32]u8 = undefined;
    hash.final(&out);
    return out;
}
