//! `tensorfold run` for a Dory checkpoint: one stream through the lane core at `--loops` loops, its tokens and speed.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const wts = @import("cuda_weights.zig");
const fwd = @import("cuda_model.zig");
const be = @import("cuda_backend.zig");

pub const Options = struct {
    tokens: []const u32,
    max_tokens: u32 = 256,
    loops: ?usize = null, // null: the checkpoint's n_recurrent_loops
    context: usize = 0, // 0: the prompt, the reply and the verify reserve
    drafts: bool = true, // copy drafts from the prompt's own text
    eager: bool = false, // every decode forward without CUDA graphs
    ignore_eos: bool = false,
    sampling: ?lanes.Sampling = null,
};

pub const Result = struct { tokens: []u32, loops: usize, prompt_ms: f64, decode_ms: f64, rounds: u64, accepted: u64 };

fn ms(io: std.Io, since: std.Io.Timestamp) f64 {
    return @as(f64, @floatFromInt(since.durationTo(std.Io.Timestamp.now(io, .awake)).nanoseconds)) / 1e6;
}

/// The checkpoint at `o.loops`, the prompt prefilled, then decoded in the lane core's rounds; tokens owned by `gpa`.
pub fn run(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, o: Options) !Result {
    if (o.tokens.len == 0) return error.EmptyPrompt;
    var w = try wts.load(gpa, io, ctx.d, dir, o.loops);
    defer w.deinit();
    var k = try fwd.Kernels.load(ctx);
    defer k.deinit();
    if (!k.served) return error.BuiltWithoutKernels;
    var s = try cuda.Stream.init(ctx.d, false);
    defer s.deinit();
    const context = try be.capacity(if (o.context != 0) o.context else o.tokens.len + o.max_tokens);
    var m = try fwd.Model.init(gpa, ctx.d, s, &w, &k, context, 1);
    defer m.deinit();
    m.graphed = !o.eager;
    var db = try be.Dory.init(gpa, ctx.d, &m, &k, s);
    defer db.deinit();
    var cfg = try lanes.Config.init(gpa, db.facts(), be.max_window, be.max_window - 1);
    defer cfg.deinit(gpa);
    var clock: lanes.backend.WallClock = .{ .io = io };
    var engine = lanes.Engine.init(gpa, &cfg, db.backend(), clock.clock());
    defer engine.deinit();
    var lookup = try lanes.SuffixLookup.init(gpa, .{ .min_match = 4 });
    defer lookup.deinit();
    const eos = w.config.eos[0..w.config.eos_count];
    var st = try lanes.Stream.init(gpa, .{ .id = "run", .prompt = o.tokens, .max_new = o.max_tokens, .eos = if (o.ignore_eos) &.{} else eos, .sampling = o.sampling, .drafts = o.drafts, .proposer = lookup.proposer() });
    defer st.deinit(gpa);
    const t0 = std.Io.Timestamp.now(io, .awake);
    try engine.addStream(&st);
    var prompt_ms: ?f64 = null; // until the first token is out: the prompt pass and its draw
    var t1 = t0;
    while (engine.activeCount() > 0) {
        try engine.step();
        if (prompt_ms == null and st.emitted().len > 0) {
            prompt_ms = ms(io, t0);
            t1 = std.Io.Timestamp.now(io, .awake);
        }
    }
    return .{ .tokens = try gpa.dupe(u32, st.emitted()), .loops = w.config.loops, .prompt_ms = prompt_ms orelse ms(io, t0), .decode_ms = ms(io, t1), .rounds = st.rounds, .accepted = st.accepted };
}

test "the run driver type-checks against the lane core" {
    std.testing.refAllDecls(@This());
}
