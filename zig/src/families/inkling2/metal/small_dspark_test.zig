//! End-to-end Metal GPU tests for Inkling 2 Small DSpark: drafted == plain, prompt-cache restore and concurrency.
const std = @import("std");
const mtl = @import("metal");
const bf = @import("bf16_round");
const lanes = @import("lanes");
const api = @import("engine_api");
const ck = @import("../../../core/checkpoint_metal.zig");
const cfg = @import("../config.zig");
const st = @import("small_state.zig");
const backend = @import("small_backend.zig");
const synth = @import("small_forward_test.zig");
const dspark = @import("small_dspark.zig");
const dw = @import("small_dspark_weights.zig");
const ds = @import("small_dspark_state.zig");
const Model = @import("small_model.zig").Model;

const t = std.testing;
const gpa = t.allocator;
const io = t.io;

fn allocBf16(device: mtl.Device, bufs: *std.ArrayList(mtl.Buffer), rng: std.Random, dims: []const usize, scale: f32) !ck.Tensor {
    var elems: usize = 1;
    for (dims) |d| elems *= d;
    const buf = try device.buffer(elems * 2, mtl.ResourceOptions.shared);
    errdefer buf.deinit();
    for (buf.slice(u16, elems)) |*v| v.* = bf.bits((rng.float(f32) * 2.0 - 1.0) * scale);
    try bufs.append(gpa, buf);
    var out: ck.Tensor = .{ .buffer = buf, .offset = 0, .bytes = elems * 2, .dtype = .bf16, .rank = @intCast(dims.len) };
    for (dims, 0..) |d, i| out.shape[i] = d;
    return out;
}

fn viewBf16(buf: mtl.Buffer, dims: []const usize) ck.Tensor {
    var elems: usize = 1;
    for (dims) |d| elems *= d;
    var out: ck.Tensor = .{ .buffer = buf, .offset = 0, .bytes = elems * 2, .dtype = .bf16, .rank = @intCast(dims.len) };
    for (dims, 0..) |d, i| out.shape[i] = d;
    return out;
}

fn buildSyntheticDspark(device: mtl.Device, target: *const Model, seed: u64) !dw.Loaded {
    try dw.admit(&target.config);
    const vocab_n = target.config.text.vocab;
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    var bufs: std.ArrayList(mtl.Buffer) = .empty;
    errdefer {
        for (bufs.items) |b| b.deinit();
        bufs.deinit(gpa);
    }

    const h = dw.hidden;
    const q = dw.query;
    const kv = dw.kv;
    const inter = dw.intermediate;
    const rank = dw.rank;
    const fuse = try allocBf16(device, &bufs, rng, &.{ dw.fusions_count, dw.taps_count }, 0.3);
    const mask = try allocBf16(device, &bufs, rng, &.{h}, 0.1);
    const norm = try allocBf16(device, &bufs, rng, &.{h}, 0.5);
    const markov_norm = try allocBf16(device, &bufs, rng, &.{h}, 0.5);
    const hidden_out = try allocBf16(device, &bufs, rng, &.{ h, rank }, 0.02);
    const hidden_bias = try allocBf16(device, &bufs, rng, &.{h}, 0.02);
    const markov_w1 = try allocBf16(device, &bufs, rng, &.{ vocab_n, rank }, 0.05);
    const rnn = try allocBf16(device, &bufs, rng, &.{ 3 * rank, 2 * h + 2 * rank }, 0.01);
    const rnn_bias = try allocBf16(device, &bufs, rng, &.{3 * rank}, 0.02);

    // Share compact random layer matrices across the 5 Small DSpark layers to keep test RAM ~250 MiB.
    const dflare_w = try allocBf16(device, &bufs, rng, &.{h}, 0.5);
    const in_norm = try allocBf16(device, &bufs, rng, &.{h}, 0.5);
    const post_norm = try allocBf16(device, &bufs, rng, &.{h}, 0.5);
    const q_norm = try allocBf16(device, &bufs, rng, &.{128}, 0.5);
    const k_norm = try allocBf16(device, &bufs, rng, &.{128}, 0.5);
    const k_norm_ctx = try allocBf16(device, &bufs, rng, &.{128}, 0.5);
    const wq = try allocBf16(device, &bufs, rng, &.{ q, h }, 0.02);
    const wk = try allocBf16(device, &bufs, rng, &.{ kv, h }, 0.02);
    const wmlp = try allocBf16(device, &bufs, rng, &.{ inter, h }, 0.01);

    const layer: dw.Layer = .{
        .dflare = dflare_w,
        .input_norm = in_norm,
        .post_norm = post_norm,
        .q_norm = q_norm,
        .k_norm = k_norm,
        .k_norm_ctx = k_norm_ctx,
        .q = wq,
        .k = wk,
        .v = wk,
        .k_ctx = wk,
        .v_ctx = wk,
        .o = viewBf16(wq.buffer, &.{ h, q }),
        .gate = wmlp,
        .up = wmlp,
        .down = viewBf16(wmlp.buffer, &.{ h, inter }),
    };
    const weights: dw.Weights = .{
        .fuse = fuse,
        .mask = mask,
        .norm = norm,
        .markov_norm = markov_norm,
        .hidden_out = hidden_out,
        .hidden_bias = hidden_bias,
        .markov_w1 = markov_w1,
        .rnn = rnn,
        .rnn_bias = rnn_bias,
        .layers = @splat(layer),
    };

    const dflare = try device.buffer(dw.fusions_count * h * 2, mtl.ResourceOptions.shared);
    errdefer dflare.deinit();
    const dflare_packed = dflare.slice(u16, dw.fusions_count * h);
    for (weights.layers, 0..) |l, i| @memcpy(dflare_packed[i * h ..][0..h], l.dflare.host(u16));
    @memcpy(dflare_packed[dw.layers_count * h ..][0..h], weights.markov_norm.host(u16));

    const phi = try device.buffer(vocab_n * rank * 2, mtl.ResourceOptions.shared);
    errdefer phi.deinit();
    const beta = try device.buffer(vocab_n * 2, mtl.ResourceOptions.shared);
    errdefer beta.deinit();
    try dw.fold(gpa, target.weights.unembed.host(u16), hidden_out.host(u16), hidden_bias.host(u16), phi.slice(u16, vocab_n * rank), beta.slice(u16, vocab_n));

    const rnn_kappa = try device.buffer(2 * h * 3 * rank * 2, mtl.ResourceOptions.shared);
    errdefer rnn_kappa.deinit();
    const rnn_rec = try device.buffer(2 * rank * 3 * rank * 2, mtl.ResourceOptions.shared);
    errdefer rnn_rec.deinit();
    dw.packRnn(rnn.host(u16), rnn_rec.slice(u16, 2 * rank * 3 * rank), rnn_kappa.slice(u16, 2 * h * 3 * rank));

    return .{
        .gpa = gpa,
        .checkpoint = ck.Checkpoint.init(gpa),
        .weights = weights,
        .vocab = vocab_n,
        .dflare = dflare,
        .phi = phi,
        .beta = beta,
        .rnn_kappa = rnn_kappa,
        .rnn_rec = rnn_rec,
        .owned_bufs = try bufs.toOwnedSlice(gpa),
    };
}

const Reply = struct {
    mutex: std.Io.Mutex = .init,
    tokens: std.ArrayList(u32) = .empty,
    rows: std.ArrayList(lanes.LogprobRow) = .empty,
    cached: ?u32 = null,
    done: ?api.Reason = null,

    fn event(ptr: *anyopaque, _: api.Id, item: *const api.Event) void {
        const self: *Reply = @ptrCast(@alignCast(ptr));
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        switch (item.*) {
            .tokens => |ids| self.tokens.appendSlice(gpa, ids) catch {},
            .logprobs => |rows| self.rows.appendSlice(gpa, rows) catch {},
            .prefilled => |n| self.cached = n,
            .finished => |f| self.done = f.reason,
            else => {},
        }
    }

    fn wait(self: *Reply) !api.Reason {
        for (0..60_000) |_| {
            self.mutex.lockUncancelable(io);
            const done = self.done;
            self.mutex.unlock(io);
            if (done) |reason| return reason;
            std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
        }
        return error.Timeout;
    }

    fn deinit(self: *Reply) void {
        self.tokens.deinit(gpa);
        self.rows.deinit(gpa);
    }
};

fn serialDecode(b: lanes.backend.Backend, ids: []const u32, max_new: usize, out: []u32) !lanes.LogprobRow {
    var s = try lanes.Stream.init(gpa, .{ .id = "serial", .prompt = ids, .max_new = @intCast(max_new), .drafts = false, .logprobs = 2 });
    defer s.deinit(gpa);
    try b.prefill(&s);
    defer b.release(&s);
    var next = try b.read(try b.first(&s, ids.len));
    const first = try b.firstRow(&s);
    out[0] = next;
    for (1..max_new) |i| {
        next = try b.read(try b.queue(&s, .{ .value = next }, ids.len + i));
        out[i] = next;
    }
    return first;
}

test "inkling2 small: DSpark drafted serving (solo, prompt-cache restore, and 2-stream concurrent) is bit-exact vs plain" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    const queue = device.queue() catch {
        device.deinit();
        return error.SkipZigTest;
    };

    const model = try synth.buildSyntheticModel(gpa, device, queue);
    defer model.deinit();
    model.config.mtp = .{
        .layers = 5,
        .q_heads = 48,
        .kv_heads = 8,
        .head_dim = 128,
        .inter = 18432,
        .window = 1024,
        .block = 8,
        .rank = 256,
        .rope_theta = 1e6,
        .taps_n = 8,
        .taps = .{ 5, 11, 17, 23, 29, 35, 41, 48, 0, 0, 0, 0, 0, 0, 0, 0 },
        .local = .{ true, true, false, true, true, false, false, false },
    };

    const budget: u64 = 2 << 30;
    const metal = try backend.Metal.initFor(gpa, model, .{ .capacity = 64, .streams = 4, .budget = budget }, 32, 16);
    defer metal.deinit();
    try metal.warm();
    try metal.measure();

    const p0 = [_]u32{ 3, 7, 11, 15, 19, 23 };
    const p1 = [_]u32{ 5, 9, 13, 17, 21, 25 };
    var want0: [6]u32 = undefined;
    var want1: [6]u32 = undefined;
    const row0 = try serialDecode(metal.backend(), &p0, want0.len, &want0);
    const row1 = try serialDecode(metal.backend(), &p1, want1.len, &want1);

    const loaded = try buildSyntheticDspark(device, model, 47);
    const drafter = try dspark.Drafter.initLoaded(gpa, device, model, loaded);
    defer drafter.deinit();
    try drafter.setContext(64);
    try drafter.warm(2);
    metal.drafter = drafter;

    var drafted = try lanes.drafted.Drafted.init(gpa, metal.backend(), drafter.drafter());
    defer drafted.deinit();
    drafted.first_rows = true;

    var config = try lanes.Config.init(gpa, drafted.facts(metal.facts()), st.window_rows, st.window_rows - 1);
    defer config.deinit(gpa);
    var clock: lanes.backend.WallClock = .{ .io = io };
    var core = lanes.Engine.init(gpa, &config, drafted.backend(), clock.clock());
    defer core.deinit();
    var host = api.LaneHost.init(gpa, io, &core, .{ .lanes = 2, .context_window = 64, .prefill_step = 4, .structures = true, .logprobs = true });
    var cache = api.prompt_cache.Store.init(gpa, .{
        .ptr = metal,
        .vtable = &.{ .bytes = backend.Metal.snapBytes, .save = backend.Metal.snapSave, .restore = backend.Metal.snapRestore, .drop = backend.Metal.snapDrop },
    }, .{ .lookahead = 0, .planned = true, .grid = 4, .min_prompt = 0 }, budget / 2);
    defer cache.deinit();
    host.cache = &cache;
    try host.start();
    defer host.stop();
    const eng = host.engine();

    // 1. Fresh solo drafted request (seeds prompt cache at history_len = 4).
    var r_fresh: Reply = .{};
    defer r_fresh.deinit();
    const req_fresh: api.Request = .{ .prompt = &p0, .history_len = 4, .max_tokens = 6, .logprobs = 2 };
    try eng.submit(1, &req_fresh, .{ .ctx = &r_fresh, .event = Reply.event });
    try t.expectEqual(api.Reason.length, try r_fresh.wait());
    try t.expectEqualSlices(u32, &want0, r_fresh.tokens.items);
    try t.expectEqual(@as(u32, @bitCast(row0.logprob)), @as(u32, @bitCast(r_fresh.rows.items[0].logprob)));

    // 2. Resumed solo drafted request hitting the DSpark + target prompt cache (cached == 4).
    var r_resumed: Reply = .{};
    defer r_resumed.deinit();
    const req_resumed: api.Request = .{ .prompt = &p0, .history_len = 4, .max_tokens = 6, .logprobs = 2 };
    try eng.submit(2, &req_resumed, .{ .ctx = &r_resumed, .event = Reply.event });
    try t.expectEqual(api.Reason.length, try r_resumed.wait());
    try t.expectEqual(@as(?u32, 4), r_resumed.cached);
    try t.expectEqualSlices(u32, &want0, r_resumed.tokens.items);
    try t.expectEqual(@as(u32, @bitCast(row0.logprob)), @as(u32, @bitCast(r_resumed.rows.items[0].logprob)));

    // 3. Two concurrent drafted streams match their serial plain targets bit-for-bit.
    var replies = [_]Reply{ .{}, .{} };
    defer for (&replies) |*r| r.deinit();
    var reqs = [_]api.Request{
        .{ .prompt = &p0, .max_tokens = 6, .logprobs = 2 },
        .{ .prompt = &p1, .max_tokens = 6, .logprobs = 2 },
    };
    for (&reqs, &replies, 0..) |*req, *rep, idx| try eng.submit(@intCast(3 + idx), req, .{ .ctx = rep, .event = Reply.event });
    for (&replies) |*rep| try t.expectEqual(api.Reason.length, try rep.wait());
    try t.expectEqualSlices(u32, &want0, replies[0].tokens.items);
    try t.expectEqualSlices(u32, &want1, replies[1].tokens.items);
    try t.expectEqual(@as(u32, @bitCast(row1.logprob)), @as(u32, @bitCast(replies[1].rows.items[0].logprob)));

    // Check DSpark accounting stays positive and finite.
    try t.expect(drafter.fixedBytes() > 0);
    try t.expect(ds.laneBytes(64) > ds.laneFixedBytes());
}
