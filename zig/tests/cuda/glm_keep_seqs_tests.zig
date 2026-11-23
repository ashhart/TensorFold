//! `glm-keep-seqs-check CKPT ID_PATH`: TF_GLM_KEEP_SEQS's kept sequences against fresh ones, byte for byte.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const openComm = @import("glm_lanes_tests.zig").openComm;

/// Greedy steps a run records, and the MTP head's drafts at each.
const steps = 24;
const depth = 2;
/// The longer prompt a slot holds before the compared one, and the compared prompts: dense only, then past the limit.
const long_prompt = 6000;
const prompts = [_]usize{ 600, 3000 };

/// One run's record: each step's drawn token, its logits row (bf16) and the MTP head's drafts after it.
const Trace = struct {
    tokens: [steps]u32 = undefined,
    drafts: [steps * depth]u32 = undefined,
    logits: []u16,

    fn init(a: std.mem.Allocator, v: usize) !Trace {
        return .{ .logits = try a.alloc(u16, steps * v) };
    }

    /// The first step where `t` differs from `ref` (token, logits row or drafts), null when none does.
    fn differs(t: *const Trace, ref: *const Trace, v: usize) ?usize {
        for (0..steps) |s| {
            if (t.tokens[s] != ref.tokens[s]) return s;
            if (!std.mem.eql(u16, t.logits[s * v ..][0..v], ref.logits[s * v ..][0..v])) return s;
            if (!std.mem.eql(u32, t.drafts[s * depth ..][0..depth], ref.drafts[s * depth ..][0..depth])) return s;
        }
        return null;
    }
};

/// `n` prompt ids from `seed`, inside the vocabulary.
fn ids(a: std.mem.Allocator, n: usize, seed: u64, vocab: usize) ![]u32 {
    const out = try a.alloc(u32, n);
    var r = std.Random.DefaultPrng.init(seed);
    for (out) |*u| u.* = r.random().uintLessThan(u32, @intCast(vocab));
    return out;
}

const Run = struct {
    gpu: check.Gpu,
    e: *@import("glm_cuda").engine.Engine,
    x: @import("glm_cuda").follow.Leader,
    v: usize,

    /// One request on `slot`: its prompt, then lone greedy steps; `plant` writes a finite pattern over the KDA states.
    fn request(r: Run, slot: u32, prompt: []const u32, out: ?*Trace, plant: bool) !void {
        try r.x.bind(slot);
        try r.x.begin(null);
        if (plant) {
            const st = &r.e.cur.st;
            const rec: cuda.DeviceBuffer = .{ .d = r.gpu.d, .ptr = st.rec, .len = 2 * st.kda_layers * st.rec_layer, .borrowed = true };
            try rec.fill8(0x3C, r.e.stream.handle); // after State.reset's zeros, on the same stream
        }
        var t = try r.x.prefill(prompt, null, null, null);
        var held: [@import("glm_cuda").engine.max_drafts]u32 = undefined;
        try r.x.draft(true, &.{t}, depth, &held, null);
        for (0..steps) |s| {
            var o: [1]u32 = undefined;
            try r.x.verify(&.{t}, &o, 0);
            if (out) |tr| try cuda.memory.download(r.gpu.d, r.e.bd.logits, std.mem.sliceAsBytes(tr.logits[s * r.v ..][0..r.v]));
            try r.x.keep(1);
            try r.x.draft(false, &o, depth, &held, null);
            if (out) |tr| {
                tr.tokens[s] = o[0];
                @memcpy(tr.drafts[s * depth ..][0..depth], held[0..depth]);
            }
            t = o[0];
        }
        try r.x.release(slot);
    }
};

const Case = struct { name: []const u8, slot: u32, poison: bool, plant: bool };

pub fn keepSeqsCheck(gpu: check.Gpu, checkpoint: []const u8, id_path: []const u8, rank: c_int, world: c_int) !void {
    const glm = @import("glm_cuda");
    var arena = std.heap.ArenaAllocator.init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var lib = try cuda.nccl.Library.open();
    defer lib.close();
    const comm = try openComm(gpu, &lib, rank, world, id_path);
    defer _ = lib.api.ncclCommDestroy(comm);
    var problem: []const u8 = "";
    var set = glm.ranks.Settings.read(&glm.ranks.environment, &problem) catch |err| {
        std.debug.print("REFUSE {s}\n", .{problem});
        return err;
    };
    var failed = false;
    for ([_]bool{ false, true }) |kv8| {
        set.kv8 = kv8;
        const e = try glm.engine.Engine.init(gpu.gpa, gpu.io, gpu.d, gpu.ctx, checkpoint, &lib, comm, @intCast(rank), @intCast(world), .{ .capacity = 8192, .mtp = true, .settings = set });
        try check.expect(e.keepsSequence(1), "glm-keep-seqs-check needs TF_GLM_KEEP_SEQS=1", .{});
        defer e.deinit();
        var link = try glm.follow.Link.init(gpu.gpa, gpu.d, &lib, comm, e.stream, 8192);
        defer link.deinit();
        const r: Run = .{ .gpu = gpu, .e = e, .x = .{ .e = e, .link = &link }, .v = e.share.vocab.len() };
        const long = try ids(a, long_prompt, 1, e.c.vocab);
        for (prompts, 0..) |n, pi| {
            const p = try ids(a, n, 2 + pi, e.c.vocab);
            const base: u32 = @intCast(10 * pi);
            var fresh = try Trace.init(a, r.v);
            try r.request(base + 9, p, &fresh, false); // a slot made for it: a zeroed arena
            const cases = [_]Case{
                .{ .name = "kept after the long prompt", .slot = base + 1, .poison = false, .plant = false },
                .{ .name = "kept, every byte poisoned (0xFF)", .slot = base + 2, .poison = true, .plant = false },
                .{ .name = "planted control: the KDA states written after the reset", .slot = base + 3, .poison = false, .plant = true },
            };
            for (cases) |c| {
                try r.request(c.slot, long, null, false);
                const kept = e.seqs.items[c.slot] orelse return check.expect(false, "slot {d} was freed with the knob on", .{c.slot});
                if (c.poison) try kept.st.arena.buf.fill8(0xFF, e.stream.handle); // NaN in bf16, f32 and E4M3 alike
                var got = try Trace.init(a, r.v);
                try r.request(c.slot, p, &got, c.plant);
                try check.expect(e.seqs.items[c.slot].? == kept, "slot {d}'s sequence was made again", .{c.slot});
                const at = got.differs(&fresh, r.v);
                if (c.plant) {
                    if (at) |s| {
                        std.debug.print("RESULT control kv8={}, {d}-token prompt: differs from the fresh run at step {d}, as planted\n", .{ kv8, n, s });
                    } else {
                        std.debug.print("FAIL kv8={}, {d}-token prompt, {s}: equal to the fresh run, so the check cannot fail\n", .{ kv8, n, c.name });
                        failed = true;
                    }
                } else if (at) |s| {
                    std.debug.print("FAIL kv8={}, {d}-token prompt, {s}: differs from the fresh run at step {d}\n", .{ kv8, n, c.name, s });
                    failed = true;
                }
            }
            var own = try Trace.init(a, r.v); // slot 0 keeps its sequence without the knob too
            try r.request(0, long, null, false);
            try r.request(0, p, &own, false);
            if (own.differs(&fresh, r.v)) |s| {
                std.debug.print("FAIL kv8={}, {d}-token prompt, slot 0 after the long prompt: differs at step {d}\n", .{ kv8, n, s });
                failed = true;
            }
        }
        try r.x.end();
    }
    if (failed) return error.TestFailed;
    check.pass("kept sequences, as left and with every byte poisoned, equal fresh ones: {d} greedy steps' tokens, bf16 logits rows and MTP drafts after prompts of {any} tokens, fp8 and bf16 caches", .{ steps, prompts });
}
