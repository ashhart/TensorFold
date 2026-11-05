//! Qwen3 dense checkpoints for the native HIP server: the head the folder declares, sized to the memory budget.
const std = @import("std");
const hip = @import("hip");
const lanes = @import("lanes");
const api = @import("engine_api");
const serve = @import("hip_serve");
const engine = @import("engine.zig");
const prompt = @import("prompt.zig");
const score = @import("score.zig");
const folder = @import("head.zig");

const Engine = engine.Engine;
const Allocator = std.mem.Allocator;

/// The registry entry: `native/hip.zig` opens `qwen3` checkpoints through it.
pub const model_type = "qwen3";
pub const formats: []const []const u8 = &.{"bf16"};
/// The window when neither --context nor the checkpoint's config names one (the Qwen3 8B serving default).
pub const default_context: i64 = 32768;
/// The engine takes a stream's whole prompt in one step.
pub const prefill_step: u32 = 0;
pub const slide_refusal = "--slide: Qwen3 dense checkpoints do not learn on HIP";
pub const confidence_refusal = "--mtp-confidence: Qwen3 dense checkpoints have no draft head";
/// An embedding engine writes no reply text, so no reasoning block (Info.reasoning).
pub const reasoning = false;
/// Every product is a BF16 dense one, which gfx12's WMMA tile runs: matrix=on is admitted on gfx12.
pub const matrix_wmma12 = true;
pub const tp_refusal = "--tp: Qwen3 dense checkpoints run on one HIP GPU; serve with --tp 1";
/// Checked by native/hip.zig before any rank joins a group (hip_families.refused).
pub fn refusal(o: api.Open) ?[]const u8 {
    return if (o.tp != 1) tp_refusal else null;
}
pub const open = openEngine;
pub const follow = followRank;

/// What the server drives, as the other HIP families' `Loaded`; its offer says what this engine serves.
pub const Loaded = struct {
    backend: lanes.backend.Backend,
    facts: lanes.Model,
    rows: u32,
    ctx: *anyopaque,
    deinit: *const fn (*anyopaque) void,
    streams: usize,
    startup: []const u8,
    /// No text generation; embeddings or a reranker's labels; the window the budget sized (openEngine).
    offer: serve.Offer,
};

pub const Refused = error{Refused};

fn refuse(a: Allocator, problem: *[]const u8, comptime fmt: []const u8, args: anytype) Refused {
    problem.* = std.fmt.allocPrint(a, fmt, args) catch fmt;
    return error.Refused;
}

pub const Head = folder.Head;
pub const headOf = folder.headOf;
pub const fitWindow = folder.fitWindow;

fn gibs(bytes: usize) f64 {
    return @as(f64, @floatFromInt(bytes)) / (1 << 30);
}

fn scratchOf(e: *Engine, window: usize) anyerror!usize {
    return e.scratchBytes(window);
}

const Owned = struct {
    gpa: Allocator,
    e: *Engine,
    p: prompt.Prompt,
    startup: []const u8,
    /// The checkpoint folder, kept for the reranker's head rows.
    dir: []const u8,
};

fn release(ptr: *anyopaque) void {
    const own: *Owned = @ptrCast(@alignCast(ptr));
    own.p.deinit();
    own.e.deinit();
    own.gpa.free(own.startup);
    own.gpa.free(own.dir);
    own.gpa.destroy(own);
}

/// The engine for `dir` on `dev`, its head from the folder and its window from the memory budget.
fn openEngine(a: Allocator, gpa: Allocator, io: std.Io, dev: hip.Device, dir: []const u8, o: anytype, problem: *[]const u8) anyerror!Loaded {
    if (o.world > 1) return refuse(a, problem, "--tp {d}: Qwen3 dense checkpoints run on one HIP GPU", .{o.world});
    // the BF16 weights run on the BF16 dot2 GEMM tile: gfx11 and gfx12, not an FP16 build (RDNA2)
    if (dev.caps.act != .bf16) return refuse(a, problem, "Qwen3 dense checkpoints on HIP need a GPU with BF16 dot2 (gfx11 or gfx12); {s} runs FP16 activations", .{hip.kernels.arch});
    const head = try headOf(a, io, dir, problem);
    const e = Engine.load(gpa, io, dir, dev.index, dev.policy.choices()) catch |err|
        return refuse(a, problem, "the native HIP engine cannot load {s} ({s})", .{ dir, @errorName(err) });
    errdefer e.deinit();
    const pool = hip.admission.budget(gpa, io, e.ctx) catch |err|
        return refuse(a, problem, "the native HIP engine cannot read the GPU's memory ({s})", .{@errorName(err)});
    const room = pool.room(e.weights);
    const window = (fitWindow(o.window, room, engine.least_window, e, scratchOf) catch |err|
        return refuse(a, problem, "the native HIP engine cannot count its scratch ({s})", .{@errorName(err)})) orelse {
        const least = e.scratchBytes(@min(o.window, engine.least_window)) catch 0;
        return refuse(a, problem, "the HIP memory budget fits no {d}-token pass: it takes {d:.2} GiB, and {d:.2} GiB is left after the model's {d:.2} GiB and the {d:.1} GiB reserve; free device memory or lower --context", .{ @min(o.window, engine.least_window), gibs(least), gibs(room), gibs(e.weights), gibs(pool.reserve) });
    };
    e.size(window) catch |err|
        return refuse(a, problem, "the native HIP engine cannot allocate its scratch ({s})", .{@errorName(err)});
    const own = try gpa.create(Owned);
    errdefer gpa.destroy(own);
    const normalize = switch (head) {
        .embedding => |h| h.normalize,
        .rerank => false,
    };
    const labels: ?api.Rerank = switch (head) {
        .embedding => null,
        .rerank => |l| l,
    };
    if (labels) |l| if (l.yes >= e.m.spec.vocab or l.no >= e.m.spec.vocab)
        return refuse(a, problem, "1_LogitScore/config.json: tokens {d} and {d} are not both in this {d}-token vocabulary", .{ l.yes, l.no, e.m.spec.vocab });
    const kept = try gpa.dupe(u8, dir);
    errdefer gpa.free(kept);
    own.* = .{ .gpa = gpa, .e = e, .p = prompt.Prompt.init(gpa, e, normalize), .startup = undefined, .dir = kept };
    errdefer own.p.deinit();
    if (labels) |l| {
        own.p.heads = score.Heads.of(&e.m, io, own.dir);
        own.p.table.score = prompt.Prompt.scoreLabels;
        own.p.table.decide = prompt.Prompt.decideLabels;
        // both label rows read at load: a bad head fails startup, and the first request reads nothing
        for ([_]u32{ l.yes, l.no }) |id| _ = own.p.heads.?.row(gpa, id) catch |err|
            return refuse(a, problem, "the native HIP engine cannot read lm_head row {d} ({s})", .{ id, @errorName(err) });
    }
    const shrunk: []const u8 = if (window < o.window) " (shrunk from the asked window to fit the budget)" else "";
    own.startup = try std.fmt.allocPrint(gpa, "HIP {s} device {d}: qwen3 {s}, model {d:.2} GiB; {d}-token passes{s}, {d:.2} GiB scratch, of {d:.1} GiB left after a {d:.1} GiB reserve; {s}", .{
        hip.kernels.arch, dev.index, if (head == .rerank) "reranker" else "embedding", gibs(e.weights), window, shrunk, gibs(e.scratchBytes(window) catch 0), gibs(room), gibs(pool.reserve),
        switch (head) {
            .embedding => |h| if (h.normalize) "vectors unit-normed" else "vectors as pooled",
            .rerank => "label scores from lm_head rows read at load, on the host",
        },
    });
    const spec = e.m.spec;
    return .{
        .backend = own.p.backend(),
        .facts = .{},
        .rows = 1,
        .ctx = own,
        .deinit = release,
        .streams = 1,
        .startup = own.startup,
        .offer = .{
            .text_generation = false,
            .embeddings = if (head == .rerank) null else .{ .dimensions = @intCast(spec.hidden), .vocabulary = @intCast(spec.vocab), .max_batch_tokens = @intCast(window), .max_inputs = @intCast(engine.max_inputs), .mrl_min = @min(32, @as(u32, @intCast(spec.hidden))) },
            .window = window,
            .decides = head == .rerank,
            .rerank = labels,
        },
    };
}

/// Ranks above 0 have nothing to hold: the engine runs on one GPU.
fn followRank(a: Allocator, _: Allocator, _: std.Io, _: hip.Device, _: []const u8, o: anytype, problem: *[]const u8) anyerror!void {
    return refuse(a, problem, "--tp {d}: Qwen3 dense checkpoints run on one HIP GPU", .{o.world});
}
