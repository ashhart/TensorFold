//! Exact CPU Ling forward over a weight source, with persistent KDA/MLA state and F32 arithmetic; no GPU.
const std = @import("std");
const config = @import("config.zig");
const op = @import("cpu_rows.zig");
const kda = @import("kda.zig");
const mla = @import("cpu_mla.zig");
const route = @import("route.zig");
const State = @import("cpu_state.zig").State;

pub const Result = struct {
    arena: std.heap.ArenaAllocator,
    logits: []f32,
    picks: []u32,
    pub fn deinit(self: *Result) void {
        self.arena.deinit();
    }
};

/// Borrowed output slices owned by a Session and valid until the next step, reset, or deinit.
pub const StepView = struct {
    logits: []f32,
    picks: []u32,
};

/// A source gives values, project, embed and optionally projectExperts; KDA rates are stored as exp(A_log).
pub fn Forward(comptime Source: type) type {
    return struct {
        const Run = struct {
            a: std.mem.Allocator,
            source: *Source,
            c: *const config.Config,
            threads: usize,
            count: usize,
            state: ?*State = null,

            fn name(self: Run, p: []const u8, n: []const u8) ![]const u8 {
                return std.mem.concat(self.a, u8, &.{ p, n });
            }

            fn alloc(self: Run, width: usize) ![]f32 {
                return self.a.alloc(f32, self.count * width);
            }

            fn values(self: Run, p: []const u8, n: []const u8, shape: []const usize) ![]f32 {
                return self.source.values(self.a, try self.name(p, n), shape);
            }

            fn linearInto(self: Run, p: []const u8, n: []const u8, x: []const f32, y: []f32, width: usize, outputs: usize, rounded: bool) !void {
                try self.source.project(try self.name(p, n), x, y, self.count, width, outputs, rounded, self.threads);
            }

            fn linear(self: Run, p: []const u8, n: []const u8, x: []const f32, width: usize, outputs: usize, rounded: bool) ![]f32 {
                const y = try self.alloc(outputs);
                try self.linearInto(p, n, x, y, width, outputs, rounded);
                return y;
            }

            fn norm(self: Run, x: []const f32, p: []const u8, n: []const u8, width: usize) ![]f32 {
                const w = try self.values(p, n, &.{width});
                const y = try self.alloc(width);
                try op.norm(x, w, y, self.count, width, self.c.eps);
                return y;
            }

            fn mlp(self: Run, x: []const f32, p: []const u8, inter: usize, limit: f32) ![]f32 {
                const hidden = self.c.hidden;
                const g = try self.linear(p, "gate_proj.weight", x, hidden, inter, true);
                const u = try self.linear(p, "up_proj.weight", x, hidden, inter, true);
                for (g, u) |*v, *up| {
                    var silu = op.round(v.* * op.sigmoid(v.*));
                    if (limit > 0) {
                        silu = @min(silu, limit);
                        up.* = std.math.clamp(up.*, -limit, limit);
                    }
                    v.* = op.round(silu * up.*);
                }
                return self.linear(p, "down_proj.weight", g, inter, hidden, true);
            }

            fn kdaForward(self: Run, x: []const f32, p: []const u8, layer: usize) ![]f32 {
                const c = self.c;
                const W = c.kda_heads * 128;
                const q = try self.linear(p, "q_proj.weight", x, c.hidden, W, true);
                const k = try self.linear(p, "k_proj.weight", x, c.hidden, W, true);
                const v = try self.linear(p, "v_proj.weight", x, c.hidden, W, true);
                const proj = try self.alloc(3 * W);
                for (0..self.count) |t| {
                    @memcpy(proj[t * 3 * W ..][0..W], q[t * W ..][0..W]);
                    @memcpy(proj[t * 3 * W + W ..][0..W], k[t * W ..][0..W]);
                    @memcpy(proj[t * 3 * W + 2 * W ..][0..W], v[t * W ..][0..W]);
                }
                const f = try self.linear(p, "f_proj.weight", x, c.hidden, W, true);
                const b = try self.linear(p, "b_proj.weight", x, c.hidden, c.kda_heads, true);
                const g = try self.linear(p, "g_proj.weight", x, c.hidden, W, true);
                const cq = try self.values(p, "q_conv1d.weight", &.{ W, 1, 4 });
                const ck = try self.values(p, "k_conv1d.weight", &.{ W, 1, 4 });
                const cv = try self.values(p, "v_conv1d.weight", &.{ W, 1, 4 });
                const rate = try self.values(p, "A_log", &.{c.kda_heads});
                const dt = try self.values(p, "dt_bias", &.{W});
                const on = try self.values(p, "o_norm.weight", &.{128});
                const state = if (self.state) |s| s.layers[layer].kda.matrix else try self.a.alloc(f32, c.kda_heads * 128 * 128);
                const history = if (self.state) |s| s.layers[layer].kda.history else try self.a.alloc([3]f32, 3 * W);
                if (self.state == null) {
                    @memset(state, 0);
                    @memset(history, .{ 0, 0, 0 });
                }
                const y = try self.alloc(W);
                const shape = kda.Shape{ .heads = c.kda_heads, .lower_bound = c.lower_bound, .stored_rate = true, .eps = c.eps };
                if (comptime @hasDecl(Source, "workers")) {
                    if (self.threads > 1) try kda.mixStateOnPool(try self.source.workers(self.threads), y, state, proj, f, b, g, cq, ck, cv, rate, dt, on, shape, @intCast(self.count), history) else try kda.mixState(y, state, proj, f, b, g, cq, ck, cv, rate, dt, on, shape, @intCast(self.count), history);
                } else try kda.mixState(y, state, proj, f, b, g, cq, ck, cv, rate, dt, on, shape, @intCast(self.count), history);
                return self.linear(p, "o_proj.weight", y, W, c.hidden, true);
            }

            fn mlaForward(self: Run, x: []const f32, p: []const u8, layer: usize) ![]f32 {
                const c = self.c;
                const qa = try self.linear(p, "q_a_proj.weight", x, c.hidden, c.q_lora, true);
                const qn = try self.norm(qa, p, "q_a_layernorm.weight", c.q_lora);
                const q = try self.linear(p, "q_b_proj.weight", qn, c.q_lora, c.mla_heads * 192, true);
                const kv = try self.linear(p, "kv_a_proj_with_mqa.weight", x, c.hidden, 576, true);
                const g = try self.linear(p, "g_proj.weight", x, c.hidden, c.mla_heads, true);
                const w = try self.values(p, "kv_a_layernorm.weight", &.{512});
                const b = try self.values(p, "kv_b_proj.weight", &.{ c.mla_heads * 256, 512 });
                const y = try self.alloc(c.mla_heads * 128);
                const shape = mla.Shape{ .heads = c.mla_heads, .eps = c.eps, .theta = c.rope_theta };
                if (self.state) |state| {
                    try mla.append(self.a, y, q, kv, w, b, g, shape, @intCast(self.count), &state.layers[layer].mla);
                } else try mla.mix(self.a, y, q, kv, w, b, g, shape, @intCast(self.count), 0);
                return self.linear(p, "dense.weight", y, c.mla_heads * 128, c.hidden, true);
            }

            fn moe(self: Run, x: []const f32, p: []const u8, layer: usize, picks: []u32) ![]f32 {
                const c = self.c;
                const scores = try self.linear(p, "gate.weight", x, c.hidden, c.experts, false);
                const bias = try self.values(p, "gate.expert_bias", &.{c.experts});
                const shape = route.Shape{ .experts = c.experts, .topk = c.topk, .n_group = c.n_group, .topk_group = c.topk_group, .scale = c.routed_scale };
                try route.checkBias(bias, shape);
                const out = try self.alloc(c.hidden);
                @memset(out, 0);
                for (0..self.count) |t| {
                    var selected: [route.max_topk]route.Pick = undefined;
                    const n = route.selectF32(scores[t * c.experts ..][0..c.experts], bias, selected[0..c.topk], shape);
                    if (n != c.topk) return error.CpuRoute;
                    for (selected[0..n], 0..) |pick, j| picks[t * c.topk + j] = pick.id;
                    if (comptime @hasDecl(Source, "projectExperts")) if (self.threads > 1) {
                        var ids: [route.max_topk]u32 = undefined;
                        for (selected[0..n], 0..) |pick, j| ids[j] = pick.id;
                        const ys = try self.a.alloc(f32, n * c.hidden);
                        try self.source.projectExperts(p, ids[0..n], x[t * c.hidden ..][0..c.hidden], ys, c.hidden, c.moe_inter, c.expert_limit[layer], self.threads);
                        for (selected[0..n], 0..) |pick, j| {
                            for (out[t * c.hidden ..][0..c.hidden], ys[j * c.hidden ..][0..c.hidden]) |*o, v| o.* += v * pick.wt;
                        }
                        continue;
                    };
                    const one = Run{ .a = self.a, .source = self.source, .c = c, .threads = self.threads, .count = 1 };
                    for (selected[0..n]) |pick| {
                        const prefix = try std.fmt.allocPrint(self.a, "{s}experts.{d}.", .{ p, pick.id });
                        const y = try one.mlp(x[t * c.hidden ..][0..c.hidden], prefix, c.moe_inter, c.expert_limit[layer]);
                        for (out[t * c.hidden ..][0..c.hidden], y) |*o, v| o.* += v * pick.wt;
                    }
                }
                const shared = try self.mlp(x, try self.name(p, "shared_experts."), c.shared_inter, c.shared_limit[layer]);
                for (out, shared) |*o, v| o.* = op.round(op.round(o.*) + v);
                return out;
            }
        };

        const LogitsMode = enum { all, last, none };

        fn runInto(work_a: std.mem.Allocator, source: *Source, c: *const config.Config, ids: []const u32, threads: usize, state: ?*State, logits_mode: LogitsMode, logits: []f32, picks: []u32) !void {
            const r = Run{ .a = work_a, .source = source, .c = c, .threads = threads, .count = ids.len, .state = state };
            const x = try r.alloc(c.hidden);
            try source.embed(ids, x);
            for (0..c.layers) |i| {
                const p = try std.fmt.allocPrint(r.a, "model.layers.{d}.", .{i});
                const h = try r.norm(x, p, "input_layernorm.weight", c.hidden);
                const att = try r.name(p, "attention.");
                const d = if (c.kind(@intCast(i)) == .kda) try r.kdaForward(h, att, i) else try r.mlaForward(h, att, i);
                for (x, d) |*v, z| v.* = op.round(v.* + z);
                const n = try r.norm(x, p, "post_attention_layernorm.weight", c.hidden);
                const m = try r.name(p, "mlp.");
                const y = if (i < c.dense_layers) try r.mlp(n, m, c.dense_inter, 0) else try r.moe(n, m, i, picks[(i - c.dense_layers) * ids.len * c.topk ..][0 .. ids.len * c.topk]);
                for (x, y) |*v, z| v.* = op.round(v.* + z);
            }
            switch (logits_mode) {
                .none => {},
                .all => {
                    const final = try r.norm(x, "model.", "norm.weight", c.hidden);
                    try r.linearInto("", "lm_head.weight", final, logits, c.hidden, c.vocab, false);
                    for (logits) |v| if (!std.math.isFinite(v)) return error.NonfiniteCpuLogits;
                },
                .last => {
                    const last_x = x[(ids.len - 1) * c.hidden ..][0..c.hidden];
                    const one = Run{ .a = r.a, .source = r.source, .c = r.c, .threads = r.threads, .count = 1, .state = r.state };
                    const final = try one.norm(last_x, "model.", "norm.weight", c.hidden);
                    try one.linearInto("", "lm_head.weight", final, logits, c.hidden, c.vocab, false);
                    for (logits) |v| if (!std.math.isFinite(v)) return error.NonfiniteCpuLogits;
                },
            }
        }

        /// One to eight rows through every layer; without a state each call starts from position 0.
        pub fn run(a: std.mem.Allocator, source: *Source, c: *const config.Config, ids: []const u32, threads: usize, state: ?*State, scratch: ?*std.heap.ArenaAllocator) !Result {
            if (ids.len == 0 or ids.len > 8) return error.CpuRows;
            if (threads == 0 or threads > 8) return error.CpuThreadCount;
            var arena = std.heap.ArenaAllocator.init(a);
            errdefer arena.deinit();
            const out_a = arena.allocator();
            const work_a = if (scratch) |s| s.allocator() else out_a;
            const picks = try out_a.alloc(u32, (c.layers - c.dense_layers) * ids.len * c.topk);
            const logits = try out_a.alloc(f32, ids.len * c.vocab);
            try runInto(work_a, source, c, ids, threads, state, .all, logits, picks);
            return .{ .arena = arena, .logits = logits, .picks = picks };
        }

        /// Serialized stream; a failed layer poisons its partial state until reset rather than reusing it silently.
        pub const Session = struct {
            a: std.mem.Allocator,
            source: *Source,
            config: config.Config,
            state: State,
            scratch: std.heap.ArenaAllocator,
            logits_buf: []f32 = &.{},
            picks_buf: []u32 = &.{},
            threads: usize,

            pub fn init(a: std.mem.Allocator, source: *Source, c: config.Config, capacity: u32, threads: usize) !Session {
                if (threads == 0 or threads > 8) return error.CpuThreadCount;
                return .{ .a = a, .source = source, .config = c, .state = try State.init(a, c, capacity), .scratch = std.heap.ArenaAllocator.init(a), .threads = threads };
            }

            pub fn run(self: *Session, a: std.mem.Allocator, ids: []const u32) !Result {
                if (self.state.poisoned) return error.CpuPoisonedState;
                if (ids.len == 0 or ids.len > 8) return error.CpuRows;
                if (ids.len > self.state.capacity - self.state.position) return error.CpuContextCapacity;
                for (ids) |id| if (id >= self.config.vocab) return error.InvalidToken;
                _ = self.scratch.reset(.retain_capacity);
                errdefer self.state.poisoned = true;
                const result = try Forward(Source).run(a, self.source, &self.config, ids, self.threads, &self.state, &self.scratch);
                self.state.position += @intCast(ids.len);
                return result;
            }

            fn stepMode(self: *Session, ids: []const u32, mode: LogitsMode) !StepView {
                if (self.state.poisoned) return error.CpuPoisonedState;
                if (ids.len == 0 or ids.len > 8) return error.CpuRows;
                if (ids.len > self.state.capacity - self.state.position) return error.CpuContextCapacity;
                for (ids) |id| if (id >= self.config.vocab) return error.InvalidToken;
                const need_logits: usize = switch (mode) {
                    .none => 0,
                    .last => self.config.vocab,
                    .all => ids.len * self.config.vocab,
                };
                if (self.logits_buf.len < need_logits) {
                    const next_logits = try self.a.alloc(f32, need_logits);
                    if (self.logits_buf.len > 0) self.a.free(self.logits_buf);
                    self.logits_buf = next_logits;
                }
                const need_picks = (self.config.layers - self.config.dense_layers) * ids.len * self.config.topk;
                if (self.picks_buf.len < need_picks) {
                    const next_picks = try self.a.alloc(u32, need_picks);
                    if (self.picks_buf.len > 0) self.a.free(self.picks_buf);
                    self.picks_buf = next_picks;
                }
                _ = self.scratch.reset(.retain_capacity);
                errdefer self.state.poisoned = true;
                const logits: []f32 = if (need_logits > 0) self.logits_buf[0..need_logits] else &.{};
                const picks = self.picks_buf[0..need_picks];
                try runInto(self.scratch.allocator(), self.source, &self.config, ids, self.threads, &self.state, mode, logits, picks);
                self.state.position += @intCast(ids.len);
                return .{ .logits = logits, .picks = picks };
            }

            /// Runs a step into session-owned output buffers reused across tokens without per-step allocation.
            pub fn step(self: *Session, ids: []const u32) !StepView {
                return self.stepMode(ids, .all);
            }

            /// Advances state over a prompt chunk, skipping discarded lm_head projections on non-final prompt rows.
            pub fn prefillChunk(self: *Session, ids: []const u32, want_last_logits: bool) !StepView {
                return self.stepMode(ids, if (want_last_logits) .last else .none);
            }

            pub fn reset(self: *Session) void {
                self.state.reset();
                _ = self.scratch.reset(.retain_capacity);
            }

            pub fn deinit(self: *Session) void {
                if (self.logits_buf.len > 0) self.a.free(self.logits_buf);
                if (self.picks_buf.len > 0) self.a.free(self.picks_buf);
                self.scratch.deinit();
                self.state.deinit();
            }
        };
    };
}
