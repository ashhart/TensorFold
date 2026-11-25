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
const resume_at = 2048;

/// The prompt state saved at `resume_at` on the first pass, ready for every rank's resumed prefill.
const Save = struct {
    e: *@import("glm_cuda").engine.Engine,
    id: ?u64 = null,
    spent: bool = false,
    failed: bool = false,

    /// Keeps the prompt state ONCE, on the pass that owns this callback and only on rank 0. Every rank reaches the
    /// mark, and a keep on each would take the same id on all of them, so the followers are left to the fence: rank 0
    /// announces the id and `gate` has every other rank keep it.
    fn call(ctx: *anyopaque, at: u32) void {
        const s: *Save = @ptrCast(@alignCast(ctx));
        if (s.spent) return;
        s.spent = true;
        if (at != resume_at) {
            s.failed = true;
            return;
        }
        if (s.e.share.rank != 0) return; // the fence keeps it here, from rank 0's announce
        s.id = s.e.keepState() catch {
            s.failed = true;
            return;
        };
    }
};

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

    const Options = struct {
        out: ?*Trace = null,
        plant: bool = false,
        from: ?@import("glm_cuda").engine.Resume = null,
        save: ?*Save = null,
    };

    /// One request on `slot`: its prompt, then lone greedy steps; `plant` writes a finite pattern over the KDA states.
    fn request(r: Run, slot: u32, prompt: []const u32, options: Options) !void {
        try r.x.bind(slot);
        try r.x.begin(null);
        if (options.plant) {
            const st = &r.e.cur.st;
            const rec: cuda.DeviceBuffer = .{ .d = r.gpu.d, .ptr = st.rec, .len = 2 * st.kda_layers * st.rec_layer, .borrowed = true };
            try rec.fill8(0x3C, r.e.stream.handle); // after State.reset's zeros, on the same stream
        }
        const marks: ?@import("glm_cuda").engine.Marks = if (options.save) |s| .{ .at = &.{resume_at}, .ctx = s, .call = Save.call } else null;
        var t = try r.x.prefill(prompt, options.from, null, marks);
        var held: [@import("glm_cuda").engine.max_drafts]u32 = undefined;
        try r.x.draft(true, &.{t}, depth, &held, null);
        for (0..steps) |s| {
            var o: [1]u32 = undefined;
            try r.x.verify(&.{t}, &o, 0);
            if (options.out) |tr| try cuda.memory.download(r.gpu.d, r.e.bd.logits, std.mem.sliceAsBytes(tr.logits[s * r.v ..][0..r.v]));
            try r.x.keep(1);
            try r.x.draft(false, &o, depth, &held, null);
            if (options.out) |tr| {
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
        try r.x.snapshots(e.stateBytes(resume_at));
        const long = try ids(a, long_prompt, 1, e.c.vocab);
        for (prompts, 0..) |n, pi| {
            const p = try ids(a, n, 2 + pi, e.c.vocab);
            const base: u32 = @intCast(10 * pi);
            var fresh = try Trace.init(a, r.v);
            try r.request(base + 9, p, .{ .out = &fresh }); // a slot made for it: a zeroed arena
            const cases = [_]Case{
                .{ .name = "kept after the long prompt", .slot = base + 1, .poison = false, .plant = false },
                .{ .name = "kept, every byte poisoned (0xFF)", .slot = base + 2, .poison = true, .plant = false },
                .{ .name = "planted control: the KDA states written after the reset", .slot = base + 3, .poison = false, .plant = true },
            };
            for (cases) |c| {
                try r.request(c.slot, long, .{});
                const kept = e.seqs.items[c.slot] orelse return check.expect(false, "slot {d} was freed with the knob on", .{c.slot});
                if (c.poison) try kept.st.arena.buf.fill8(0xFF, e.stream.handle); // NaN in bf16, f32 and E4M3 alike
                var got = try Trace.init(a, r.v);
                try r.request(c.slot, p, .{ .out = &got, .plant = c.plant });
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
            try r.request(0, long, .{});
            try r.request(0, p, .{ .out = &own });
            if (own.differs(&fresh, r.v)) |s| {
                std.debug.print("FAIL kv8={}, {d}-token prompt, slot 0 after the long prompt: differs at step {d}\n", .{ kv8, n, s });
                failed = true;
            }
            if (n > resume_at) {
                const slot = base + 4;
                var save: Save = .{ .e = e };
                var initial = try Trace.init(a, r.v);
                try r.request(slot, p, .{ .out = &initial, .save = &save });
                try check.expect(!save.failed, "the prompt state at {d} was not kept", .{resume_at});
                // Rank 0 takes the id and announces it; the fence has every other rank keep the same one, so only
                // rank 0 learns it from the callback and the followers read it back from their own map.
                const id = if (save.id) |i| i else id: {
                    var found: ?u64 = null;
                    var it = e.snaps.keyIterator();
                    while (it.next()) |k| found = k.*;
                    break :id found orelse return check.expect(false, "the prompt state at {d} was not kept on any rank", .{resume_at});
                };
                const kept = e.seqs.items[slot] orelse return check.expect(false, "resume slot {d} was freed with the knob on", .{slot});
                try check.expect(e.keepsSequence(slot), "resume slot {d} does not keep its sequence", .{slot});

                var resumed = try Trace.init(a, r.v);
                try r.request(slot, p, .{ .out = &resumed, .from = .{ .id = id, .at = resume_at } });
                try check.expect(e.seqs.items[slot].? == kept, "resume slot {d}'s sequence was made again", .{slot});
                const initial_at = initial.differs(&fresh, r.v);
                if (initial_at) |s| {
                    std.debug.print("FAIL kv8={}, {d}-token prompt, first pass keeping resume state at {d}: differs from fresh at step {d}\n", .{ kv8, n, resume_at, s });
                    failed = true;
                }
                const resumed_at = resumed.differs(&fresh, r.v);
                if (resumed_at) |s| {
                    std.debug.print("FAIL kv8={}, {d}-token prompt, resumed at {d} onto kept slot {d}: differs from fresh at step {d}\n", .{ kv8, n, resume_at, slot, s });
                    failed = true;
                }
                if (initial_at == null and resumed_at == null)
                    check.pass("prompt-cache resume at {d} onto kept slot {d} equals a fresh {d}-token prompt: {d} greedy steps' tokens, bf16 logits rows and MTP drafts, kv8={}", .{ resume_at, slot, n, steps, kv8 });

                try check.expect(e.holds(id), "the prompt state at {d} was dropped before its control", .{resume_at});
                const stored = e.snap_pages.?.arena.?;
                const bytes: cuda.DeviceBuffer = .{ .d = gpu.d, .ptr = stored.ptr(), .len = stored.len(), .borrowed = true };
                try bytes.fill8(0xFF, e.stream.handle); // every prompt-cache byte poisoned, NaN in every cache format
                var control = try Trace.init(a, r.v);
                try r.request(slot, p, .{ .out = &control, .from = .{ .id = id, .at = resume_at } });
                try check.expect(e.seqs.items[slot].? == kept, "control resume slot {d}'s sequence was made again", .{slot});
                if (control.differs(&fresh, r.v)) |s| {
                    std.debug.print("RESULT resume control kv8={}, {d}-token prompt at {d} onto a kept slot: poisoned prompt state differs from fresh at step {d}, as planted\n", .{ kv8, n, resume_at, s });
                } else {
                    std.debug.print("FAIL kv8={}, {d}-token prompt, poisoned prompt-cache resume control: equal to fresh, so the check cannot fail\n", .{ kv8, n });
                    failed = true;
                }
            }
        }
        try r.x.end();
    }
    if (failed) return error.TestFailed;
    check.pass("kept sequences, as left and with every byte poisoned, equal fresh ones: {d} greedy steps' tokens, bf16 logits rows and MTP drafts after prompts of {any} tokens, fp8 and bf16 caches", .{ steps, prompts });
}
