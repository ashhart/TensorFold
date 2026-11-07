// Admission precedes weights, state allocates last, and post-write failures poison until reset; no rollback.
const std = @import("std");
const vk = @import("vulkan");
const pinned = @import("../../config.zig");
const weights = @import("weights.zig");
const state = @import("state.zig");
const fwd = @import("forward.zig");

const Allocator = std.mem.Allocator;
const View = vk.tensor.View;

pub const max_prefill_rows = fwd.max_prefill_rows;

// Chunk cut = min(done+32,total), pure, host-tested; totals 32/33/63/64/65 cut at 32 / 32,33 / 32,63 / 32,64 / 32,64,65
pub fn chunkEnd(done: usize, total: usize) usize {
    return @min(done +| max_prefill_rows, total);
}

// The staged-transfer buffer: 8 MiB so the load's ~27k tensors stream in large chunks instead of one 64 KiB fence each.
pub const staging_bytes: u64 = 8 << 20;
// The 8 MiB scratch arena holds persistent windows and worst-layer prefill transients (~3.4 MiB), with headroom.
pub const arena_bytes: u64 = 8 << 20;

// Step errors include dispatch.Error; Timeout exits 124 and InFlight exits 1 inside `vk.orExit`, before teardown.
pub const Error = vk.dispatch.Error ||
    error{ ContextTooLong, NonFinite, UnsupportedPrompt, Poisoned };
// Engine.init has no explicit error set: inferred platform I/O/config errors pass through; steps do no filesystem I/O.

// Build-supplied SPIR-V variants (norm/swiglu/rope use ASH_QWEN35 Metal oracle); engine checks and pipelines them.
pub const ShaderBundle = struct {
    ops: vk.ops.Shaders,
    norm: vk.norm_mlp.Shaders,
    attention: vk.attention.Shaders,
    gdn: vk.gdn.Shaders,
    greedy: vk.greedy.Shaders,
};

// The plan is checked before GPU work: `steps` draws need `rows + steps - 1` positions; checked sums refuse overflow.
pub fn planFor(capacity: usize, visible: u64, rows: usize, steps: usize) Error!void {
    if (rows == 0 or steps == 0) return error.UnsupportedPrompt;
    if (capacity > state.max_capacity) return error.ContextTooLong;
    const need = std.math.add(u64, visible, rows) catch return error.ContextTooLong;
    const total = std.math.add(u64, need, steps) catch return error.ContextTooLong;
    if (total - 1 > capacity) return error.ContextTooLong;
}

// Shared logits readiness: poison or zero visible prefix refuses draw/readback before GPU work; pure and host-tested.
pub fn logitsReady(poisoned: bool, visible: u64) Error!void {
    if (poisoned) return error.Poisoned;
    if (visible == 0) return error.Invalid;
}

// The arena stores four hidden-row buffers, ids, and full logits; it also fits the worst transient layer.
pub fn persistentBytes() u64 {
    const rows: u64 = max_prefill_rows;
    return 4 * rows * pinned.hidden * 2 + rows * 4 + pinned.vocab * 2;
}

pub const Engine = struct {
    gpa: Allocator,
    timeout_ns: u64,
    rt: *vk.Runtime,
    dev: *vk.Device,
    cmds: *vk.Commands,
    staging: vk.Buffer,
    ops: vk.ops.Ops,
    norm: vk.norm_mlp.NormMlp,
    attn: vk.Attention,
    gdn: vk.Gdn,
    greedy: vk.Greedy,
    arena: vk.arena.Arena,
    model: weights.Model,
    state: state.State,
    // The residual ring: h[0]/h[1] alternate as the add's input and output, so no add ever writes the buffer it reads.
    h: [2]View,
    r: View,
    x: View,
    ids: View,
    logits: View,
    base_mark: u64,
    poisoned: bool,

    // Pinned-model admission runs inside the weights load before GPU work; SDPA arm limit is 7168; probe uses 256.
    pub fn init(
        gpa: Allocator,
        io: std.Io,
        bundle: ShaderBundle,
        model_dir: []const u8,
        context: usize,
        timeout_ns: u64,
    ) !*Engine {
        const cap = try state.capacityFor(context);
        const e = try gpa.create(Engine);
        errdefer gpa.destroy(e);
        e.gpa = gpa;
        e.timeout_ns = timeout_ns;
        e.poisoned = false;

        e.rt = try gpa.create(vk.Runtime);
        errdefer gpa.destroy(e.rt);
        e.rt.* = try vk.Runtime.open();
        errdefer e.rt.close();
        e.dev = try gpa.create(vk.Device);
        errdefer gpa.destroy(e.dev);
        e.dev.* = try vk.Device.open(e.rt, .{ .features = .{ .storage16 = true, .int16 = true } });
        errdefer e.dev.close();
        e.cmds = try gpa.create(vk.Commands);
        errdefer gpa.destroy(e.cmds);
        e.cmds.* = try vk.Commands.init(e.dev);
        errdefer e.cmds.deinit();
        e.staging = try vk.Buffer.init(e.dev, .staging, staging_bytes);
        errdefer e.staging.deinit();

        // device gates first: every operator refuses a lacking device before a single weight byte is read
        e.ops = try vk.ops.Ops.init(e.dev, bundle.ops);
        errdefer e.ops.deinit();
        e.norm = try vk.norm_mlp.NormMlp.init(e.dev, bundle.norm);
        errdefer e.norm.deinit();
        e.attn = try vk.Attention.init(e.dev, bundle.attention, &e.ops);
        errdefer e.attn.deinit();
        e.gdn = try vk.Gdn.init(e.dev, bundle.gdn);
        errdefer e.gdn.deinit();
        e.greedy = try vk.Greedy.init(e.dev, bundle.greedy, pinned.vocab);
        errdefer e.greedy.deinit();

        e.arena = try vk.arena.Arena.init(e.dev, arena_bytes);
        errdefer e.arena.deinit();
        const rows: u64 = max_prefill_rows;
        e.h[0] = try e.arena.alloc(.bf16, rows * pinned.hidden);
        e.h[1] = try e.arena.alloc(.bf16, rows * pinned.hidden);
        e.r = try e.arena.alloc(.bf16, rows * pinned.hidden);
        e.x = try e.arena.alloc(.bf16, rows * pinned.hidden);
        e.ids = try e.arena.alloc(.u32, rows);
        e.logits = try e.arena.alloc(.bf16, pinned.vocab);
        e.base_mark = e.arena.mark();

        e.model = try vk.orExit(weights.Model.load(gpa, e.dev, e.cmds, io, model_dir, e.staging, timeout_ns));
        errdefer e.model.deinit();
        e.state = try vk.orExit(state.State.init(gpa, e.dev, e.cmds, cap, e.staging, timeout_ns));
        return e;
    }

    // Reverse teardown is legal only with no pending submission; `orExit` ensures Timeout/InFlight never return here.
    pub fn deinit(e: *Engine) void {
        e.state.deinit();
        e.model.deinit();
        e.arena.deinit();
        e.greedy.deinit();
        e.gdn.deinit();
        e.attn.deinit();
        e.norm.deinit();
        e.ops.deinit();
        e.staging.deinit();
        e.cmds.deinit();
        e.dev.close();
        e.rt.close();
        const gpa = e.gpa;
        gpa.destroy(e.cmds);
        gpa.destroy(e.dev);
        gpa.destroy(e.rt);
        gpa.destroy(e);
    }

    // Reset zeroes all state on GPU; commits length/parities after fences; clears poison and rewinds scratch to base.
    pub fn reset(e: *Engine) Error!void {
        try vk.orExit(e.state.reset(e.cmds, e.staging, e.timeout_ns));
        e.arena.reset(e.base_mark) catch {};
        e.poisoned = false;
    }

    // Check ids/bounds, commit 32-row chunks. Failure poisons (partial RNN state never reused); refusals touch nothing.
    pub fn prefill(e: *Engine, tokens: []const u32) Error!void {
        if (e.poisoned) return error.Poisoned;
        if (tokens.len == 0) return error.UnsupportedPrompt;
        try vk.ops_q4.checkIds(tokens, e.model.embed.n);
        try planFor(@intCast(e.state.capacity), e.state.visible(), tokens.len, 1);
        var done: usize = 0;
        while (done < tokens.len) {
            const end = chunkEnd(done, tokens.len);
            try e.prefillChunk(tokens[done..end]);
            done = end;
        }
    }

    // One forward chunk has 1..32 checked rows; its logits give the next-token distribution after the chunk.
    pub fn prefillChunk(e: *Engine, tokens: []const u32) Error!void {
        if (e.poisoned) return error.Poisoned;
        if (tokens.len == 0 or tokens.len > max_prefill_rows) return error.UnsupportedPrompt;
        try vk.ops_q4.checkIds(tokens, e.model.embed.n);
        try planFor(@intCast(e.state.capacity), e.state.visible(), tokens.len, 1);
        e.poisoned = true;
        try vk.orExit(vk.upload(e.cmds, e.ids.buffer.*, e.ids.byte_offset, e.staging, std.mem.sliceAsBytes(tokens), e.timeout_ns));
        try vk.orExit(embed(e, try e.h[0].slice(0, tokens.len * pinned.hidden), tokens.len));
        try runLayers(e, tokens.len);
        try vk.orExit(e.state.append(tokens.len));
        e.poisoned = false;
    }

    // Decode embeds the fed token, runs all 24 layers and final norm/head; resulting logits cover the next token.
    pub fn decode(e: *Engine, token: u32) Error!void {
        if (e.poisoned) return error.Poisoned;
        try vk.ops_q4.checkIds(&.{token}, e.model.embed.n);
        try planFor(@intCast(e.state.capacity), e.state.visible(), 1, 1);
        e.poisoned = true;
        try vk.orExit(vk.upload(e.cmds, e.ids.buffer.*, e.ids.byte_offset, e.staging, std.mem.sliceAsBytes(&[1]u32{token}), e.timeout_ns));
        try vk.orExit(embed(e, try e.h[0].slice(0, pinned.hidden), 1));
        try runLayers(e, 1);
        try vk.orExit(e.state.append(1));
        e.poisoned = false;
    }

    // Ash draw picks the largest finite logit; ties (+0/-0) go lowest index. Any NaN/Inf refuses and poisons.
    pub fn draw(e: *Engine) Error!u32 {
        try logitsReady(e.poisoned, e.state.visible());
        return vk.orExit(e.greedy.select(e.cmds, e.staging, e.logits, e.timeout_ns)) catch |err| {
            if (err == error.NonFinite) e.poisoned = true;
            return err;
        };
    }

    // Logits are bytes for receipt readback, never CPU argmax; wrong `out.len` is refused without poisoning.
    pub fn logitsBytes(e: *Engine, out: []u8) Error!void {
        try logitsReady(e.poisoned, e.state.visible());
        if (out.len != e.logits.byteLen()) return error.Invalid;
        try vk.orExit(vk.download(e.cmds, e.logits.buffer.*, e.logits.byte_offset, e.staging, out, e.timeout_ns));
    }

    pub fn visible(e: *Engine) u64 {
        return e.state.visible();
    }

    /// The logits row's byte length, for sizing caller buffers.
    pub fn logitsLen(e: *Engine) u64 {
        return e.logits.byteLen();
    }

    /// The capacity actually allocated (the context request, SDPA-capped).
    pub fn capacity(e: *Engine) usize {
        return @intCast(e.state.capacity);
    }

    /// config.json's max_position_embeddings, as the weights admission read it.
    pub fn modelContext(e: *Engine) usize {
        return e.model.context;
    }

    /// The chosen device's Vulkan vendorID, for the probe's optional guard.
    pub fn vendorId(e: *Engine) u32 {
        return e.dev.properties.vendorID;
    }

    fn embed(e: *Engine, out: View, rows: usize) !void {
        const cx = e.core();
        const mark = e.arena.mark();
        errdefer e.arena.reset(mark) catch {};
        try vk.orExit(vk.ops_q4.embedding(cx.acc.ops, cx.cmds, out, try e.ids.slice(0, rows), e.model.embed.w, e.model.embed.scales, e.model.embed.biases, e.model.embed.k, &e.arena, e.timeout_ns));
        e.arena.reset(mark) catch {};
    }

    fn core(e: *Engine) fwd.Core {
        return .{
            .cmds = e.cmds,
            .staging = e.staging,
            .acc = .{ .ops = &e.ops, .norm = &e.norm, .attn = &e.attn, .gdn = &e.gdn },
            .scratch = &e.arena,
            .timeout_ns = e.timeout_ns,
        };
    }

    // 24 layers: norm (plain at layer 0, else residual add), mixer, post norm, MLP; ring keeps add buffers disjoint.
    fn runLayers(e: *Engine, rows: usize) Error!void {
        const cx = e.core();
        const span = rows * pinned.hidden;
        const base = e.state.visible();
        var ring: usize = 0;
        const x = try e.x.slice(0, span);
        const r = try e.r.slice(0, span);
        for (e.model.layers, 0..) |layer, i| {
            const input_norm, const post_norm, const mlp_q = switch (layer) {
                .full => |fl| .{ fl.input_norm, fl.post_norm, fl.mlp },
                .linear => |ll| .{ ll.input_norm, ll.post_norm, ll.mlp },
            };
            if (i == 0) {
                try cx.plainNorm(x, try e.h[0].slice(0, span), input_norm, rows);
            } else {
                try cx.addNorm(x, try e.h[ring ^ 1].slice(0, span), try e.h[ring].slice(0, span), r, input_norm, rows);
                ring ^= 1;
            }
            switch (layer) {
                .full => |fl| try cx.fullLayer(x, fl, &e.state, state.fullSlot(i).?, rows, r, base),
                .linear => |ll| try cx.linearLayer(x, ll, &e.state, state.linearSlot(i).?, rows, r),
            }
            try cx.addNorm(x, try e.h[ring ^ 1].slice(0, span), try e.h[ring].slice(0, span), r, post_norm, rows);
            ring ^= 1;
            try cx.mlp(x, rows, mlp_q, r);
        }
        try cx.finish(try e.h[ring ^ 1].slice(0, span), try e.h[ring].slice(0, span), r, try e.x.slice(0, span), e.model.final_norm, e.model.outputHead(), e.logits, rows);
    }
};

const testing = std.testing;

test "logitsReady admits only a clean engine with a visible prefix" {
    try logitsReady(false, 1);
    try logitsReady(false, 256);
    try testing.expectError(error.Invalid, logitsReady(false, 0));
    try testing.expectError(error.Poisoned, logitsReady(true, 7));
    try testing.expectError(error.Poisoned, logitsReady(true, 0)); // poison wins
}

test "the plan refuses empty, stepless and overflowing sequences before any GPU work" {
    const cap = 256;
    try planFor(cap, 0, 16, 16);
    try planFor(cap, 240, 16, 1); // exactly fills: 240 + 16 + 1 - 1 = 256
    try testing.expectError(error.UnsupportedPrompt, planFor(cap, 0, 0, 1));
    try testing.expectError(error.UnsupportedPrompt, planFor(cap, 0, 1, 0));
    try testing.expectError(error.ContextTooLong, planFor(cap, 241, 16, 1));
    try testing.expectError(error.ContextTooLong, planFor(0, 0, 1, 1));
    // checked sums: extreme caller inputs refuse instead of wrapping
    try testing.expectError(error.ContextTooLong, planFor(cap, std.math.maxInt(u64), 1, 1));
    try testing.expectError(error.ContextTooLong, planFor(cap, 0, 1, std.math.maxInt(usize)));
    try testing.expectError(error.ContextTooLong, planFor(cap, std.math.maxInt(u64) - 1, 1, 3));
}

test "the canonical cut ends every whole-request length at 32-row boundaries" {
    for ([_]struct { len: usize, cuts: []const usize }{
        .{ .len = 1, .cuts = &.{1} },
        .{ .len = 31, .cuts = &.{31} },
        .{ .len = 32, .cuts = &.{32} },
        .{ .len = 33, .cuts = &.{ 32, 33 } },
        .{ .len = 63, .cuts = &.{ 32, 63 } },
        .{ .len = 64, .cuts = &.{ 32, 64 } },
        .{ .len = 65, .cuts = &.{ 32, 64, 65 } },
    }) |case| {
        var done: usize = 0;
        var at: usize = 0;
        while (done < case.len) : (at += 1) {
            const end = chunkEnd(done, case.len);
            try testing.expect(at < case.cuts.len and end == case.cuts[at]); // exactly the expected schedule
            done = end;
        }
        try testing.expectEqual(case.cuts.len, at);
    }
}

test "the plan admits whole requests up to the capacity ceiling and refuses one past" {
    const cap: usize = 7168;
    for ([_]usize{ 1, 31, 32, 33, 63, 64, 65, 7105, 7168 }) |len| {
        try planFor(cap, 0, len, 1); // a prompt-only request always fits at or under the ceiling
    }
    try testing.expectError(error.ContextTooLong, planFor(cap, 0, 7169, 1)); // one past the ceiling
    try planFor(cap, 0, 7168, 1); // exact fit prompt + one draw reads a written row
    try testing.expectError(error.ContextTooLong, planFor(cap, 0, 7168, 2)); // one past with a decode
    try planFor(cap, 0, 7105, 64); // the frozen long case ends at exactly 7168 writes
    try testing.expectError(error.ContextTooLong, planFor(cap, 0, 7106, 64));
    try testing.expectError(error.ContextTooLong, planFor(cap + 1, 0, 1, 1));
    try planFor(cap, 0, 7167, 2); // 7167 + 2 - 1 = 7168: exact fit
}

test "the arena holds the persistent windows plus the worst transient layer" {
    const rows: u64 = max_prefill_rows;
    const conv: u64 = pinned.conv_dim;
    const dim: u64 = pinned.linear_heads * pinned.linear_dim;
    // one linear layer at the cap, term by term in bytes: qkv, z, a, b, soa, g (f32), beta, conv_out, conv_in, y, mix
    const linear = 2 * rows * (conv + 4 * dim + 3 * conv + dim + dim) +
        7 * rows * pinned.linear_heads + 2 * (rows + state.conv_rows) * conv;
    // one full-attention layer at the cap: qraw, q, qn, k, kn, v, queries, y, mix
    const full = 2 * rows * (2 * 8 * 256 + 5 * 8 * 256 + 3 * 2 * 256);
    // the MLP: gate + up + act
    const mlp = 6 * pinned.intermediate * rows;
    const worst = @max(linear, @max(full, mlp));
    try testing.expect(worst > 0);
    try testing.expect(arena_bytes >= persistentBytes() + worst + (1 << 20));
}
