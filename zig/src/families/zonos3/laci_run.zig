//! A request generated under LACI (synth._laci): attempts on one lane, each rolled back to where its reading failed.
const std = @import("std");
const cfg = @import("config.zig");
const generation = @import("generation.zig");
const laci = @import("laci.zig");
const probe_mod = @import("laci_probe.zig");
const sampler = @import("sampler.zig");
const short_text = @import("short_text.zig");
const Steer = @import("steering.zig").Steer;

const C = cfg.codebooks;
const W = cfg.frame_width;

/// What a request under LACI needs beside its prompt: the language's heads and, for a short text, its cut.
pub const Setup = struct {
    heads: laci.Heads,
    /// The prompt's BOS..EOS text rows.
    text_start: usize,
    text_len: usize,
    /// The silence rows that end the prompt: the stream's first rows.
    prefill: usize,
    /// A whole decode: no row goes out before the result is final.
    hold: bool,
    short: ?short_text.Cut = null,
};

/// The row the request's lane runs next, with its probe (the queries out, the heads to mask) and its deltas.
pub const Turn = struct { row: [W]i32, probe: *probe_mod.Probed, steer: []const Steer };

pub const Step = struct {
    /// The generated rows that are final so far, from the first.
    rows: []const [W]i32,
    /// With `final`: the aligned end-of-audio frame, or null when the budget ran out.
    eos_frame: ?usize = null,
    final: bool = false,
    /// The forward that gives the next row's logits; null when the request is final or an attempt just restarted.
    turn: ?Turn = null,
    /// The row this step sampled, and the reading it took before (none on an attempt's first step).
    row: [W]i32,
    reading: ?probe_mod.Reading = null,
    /// The masks' centre that reading left; null: no head is masked.
    center: ?usize = null,
};

/// `Replace`: the engine's replaced-row type; the engine has reset(), run(), lastLogits() and keys() on the lane.
pub fn Run(comptime Replace: type) type {
    return struct {
        const Self = @This();

        gpa: std.mem.Allocator,
        session: laci.Session,
        setup: Setup,
        probe: laci.Probe,
        /// Decode steps of the current attempt.
        steps: usize = 0,
        run: generation.Run,
        /// The prompt without its prefill rows, and the rows a speaker or condition vector replaces.
        prompt: []const [W]i32,
        replaced: []const Replace,
        sampling: sampler.Sampling,
        /// A rolled-back attempt's prompt: `prompt`, then the kept stream rows.
        again: std.ArrayList([W]i32) = .empty,
        /// Kept generated rows before the current attempt: its frames count on from here.
        history: usize = 0,
        keys: [probe_mod.layers][]f32,
        probed: probe_mod.Probed,
        masks: [cfg.heads * probe_mod.layers]probe_mod.Mask = undefined,
        scratch: []f32,
        steer: ?generation.Run.Steering = null,
        /// A regeneration's prompt failed on the engine: there is no run to free.
        dead: bool = false,

        /// Runs the prompt on the lane in use and reads the text rows' keys. `rows` is the whole prompt.
        pub fn begin(gpa: std.mem.Allocator, engine: anytype, rows: []const [W]i32, replaced: []const Replace, setup: Setup, sampling: sampler.Sampling, max_frames: usize) !Self {
            if (setup.text_len == 0 or setup.text_start + setup.text_len + setup.prefill > rows.len) return error.BadTextSpan;
            var run = try generation.Run.begin(gpa, engine, rows, replaced, &.{}, replaced, 1, sampling, max_frames);
            errdefer run.deinit();
            var keys: [probe_mod.layers][]f32 = @splat(&.{});
            errdefer for (keys) |k| gpa.free(k);
            for (&keys, 0..) |*k, s| {
                k.* = try gpa.alloc(f32, setup.text_len * probe_mod.key_width);
                try engine.keys((s + 1) * cfg.gdn_period - 1, setup.text_start, k.*);
            }
            const scratch = try gpa.alloc(f32, 3 * setup.text_len);
            errdefer gpa.free(scratch);
            const session = try laci.Session.init(gpa, setup.heads, .{}, rows[rows.len - setup.prefill ..], sampling.seed, max_frames, setup.hold, setup.short);
            return .{ .gpa = gpa, .session = session, .setup = setup, .probe = .{ .text_len = setup.text_len }, .run = run, .prompt = rows[0 .. rows.len - setup.prefill], .replaced = replaced, .sampling = sampling, .keys = keys, .probed = .{ .text_start = setup.text_start, .text_len = setup.text_len }, .scratch = scratch };
        }

        pub fn deinit(r: *Self) void {
            if (!r.dead) r.run.deinit();
            r.session.deinit();
            r.again.deinit(r.gpa);
            for (r.keys) |k| r.gpa.free(k);
            r.gpa.free(r.scratch);
        }

        fn final(r: *const Self) []const [W]i32 {
            return r.session.stream.items[r.setup.prefill..][0..r.session.emitted];
        }

        /// One turn: the last forward's reading, the next row, then its forward or a regeneration's prompt.
        pub fn step(r: *Self, engine: anytype) !Step {
            const s = &r.session;
            const bias = s.stopBoost();
            var reading: ?probe_mod.Reading = null;
            if (r.steps > 0 and !r.probe.stop) {
                reading = probe_mod.readout(s.heads, r.probed.masks, &r.probed.q, .{ r.keys[0], r.keys[1] }, r.setup.text_len, r.scratch);
                try s.onStep(&r.probe, r.steps, reading.?.pos, reading.?.m);
            }
            const center = r.probe.center;
            r.run.sampling.stop_bias = bias;
            // a probe stop spends the budget: this row is the attempt's last
            if (r.probe.stop) r.run.max_frames = r.run.rows.items.len + 1;
            const finished = try r.run.sample();
            const row = r.run.lastRow();
            try s.append(row, r.run.eos_frame);
            if (!finished) {
                _ = s.drain(false);
                r.steps += 1;
                r.probed.masks = probe_mod.masks(s.heads, r.probe.center, r.setup.text_len, &r.masks);
                const steer: []const Steer = if (r.steer) |st| st.at(st.ctx, r.history + r.run.rows.items.len) else &.{};
                return .{ .rows = r.final(), .turn = .{ .row = row, .probe = &r.probed, .steer = steer }, .row = row, .reading = reading, .center = center };
            }
            const cont = try s.finishAttempt() orelse {
                _ = s.drain(true);
                return .{ .rows = r.final(), .eos_frame = s.eosFrame(), .final = true, .row = row, .reading = reading, .center = center };
            };
            _ = s.drain(false);
            // the regeneration: the prompt and the kept rows prefilled again, under the attempt's seed
            r.again.clearRetainingCapacity();
            try r.again.appendSlice(r.gpa, r.prompt);
            try r.again.appendSlice(r.gpa, s.stream.items[0..cont.rows]);
            var sampling = r.sampling;
            sampling.seed = cont.seed.?;
            r.run.deinit();
            r.dead = true;
            try engine.reset();
            r.run = try generation.Run.begin(r.gpa, engine, r.again.items, r.replaced, &.{}, r.replaced, 1, sampling, cont.max_tokens);
            r.dead = false;
            try r.run.keep(s.stream.items[r.setup.prefill..cont.rows]);
            r.history = cont.history_rows;
            r.steps = 0;
            r.probe = .{ .text_len = r.setup.text_len, .center = s.probeCenter() };
            r.probed.masks = &.{};
            return .{ .rows = r.final(), .row = row, .reading = reading, .center = center };
        }
    };
}
