//! Nemotron-H: Mamba2, NoPE attention, routed/shared ReLU² experts, and MTP.
const std = @import("std");
const mx = @import("mlx.zig");
const cp = @import("checkpoint.zig");
const src = @import("kernel_sources.zig");
const A = mx.Array;
const ti = mx.ti;
const Cache = @import("model.zig").Cache;
const kv = @import("kv_buffer.zig");
pub const Pass = struct {
    scope: mx.Scope = .{},
    logits: A = mx.empty,
    hidden: A = mx.empty,
    prefilled: bool = false,
    records: [52]Cache = @splat(.{}),
    pub fn deinit(p: *Pass) void {
        p.scope.deinit();
    }
};
pub const Model = struct {
    pub const SerialPass = Pass;
    pub const DraftCache = Cache;
    weights: cp.Store,
    kernels: mx.Kernels,
    cache: [52]Cache = @splat(.{}),
    kinds: [52]u8 = undefined,
    position: i32 = 0,
    mtp: bool = false,
    prefill_ops: @import("prefill_ops.zig").Ops = .{},
    prefill_route: @import("nemotron_prefill.zig").Route = .{},
    pub const vocab = 131072;
    pub fn eos(id: i32) bool {
        return id == 2 or id == 11;
    }
    pub fn init(io: std.Io, dir: []const u8, drafts: bool) !Model {
        var m = Model{ .weights = cp.Store.init(64), .kernels = mx.Kernels.init() };
        errdefer m.deinit();
        var buf: [4096]u8 = undefined;
        const bytes = try @import("weights.zig").readFile(io, try std.fmt.bufPrint(&buf, "{s}/config.json", .{dir}));
        defer mx.allocator.free(bytes);
        const cfg = try std.json.parseFromSlice(std.json.Value, mx.allocator, bytes, .{});
        defer cfg.deinit();
        try @import("config.zig").nemotron(cfg.value);
        for (cfg.value.object.get("layers_block_type").?.array.items, 0..) |v, i| {
            m.kinds[i] = if (std.mem.eql(u8, v.string, "mamba")) 'M' else if (std.mem.eql(u8, v.string, "moe")) 'E' else '*';
        }
        try m.weights.load(io, dir, "");
        if (drafts) {
            m.weights.loadFile(io, try std.fmt.bufPrint(&buf, "{s}/mtp-4bit.safetensors", .{dir}), "mtp.", "") catch |err| return if (err == error.FileNotFound) error.MissingDraftHead else err;
            m.mtp = true;
        }
        try @import("schema.zig").validate(.nemotron, &m.weights.arrays, drafts);
        // Small constants are prepared once; expert tables remain in their packed format.
        for (m.kinds, 0..) |kind, i| if (kind == 'M') {
            var s = mx.Scope{};
            defer s.deinit();
            const key = try std.fmt.bufPrint(&buf, "backbone.layers.{d}.mixer.conv1d.weight", .{i});
            const v = try m.weights.get(key);
            const cw = try s.cast(try s.transpose(try s.reshape(v, &.{ 6144, 4 }), &.{ 1, 0 }), mx.f32t);
            try mx.eval(cw);
            try m.weights.put(key, cw);
        };
        return m;
    }
    pub fn reset(m: *Model) void {
        for (&m.cache) |*c| c.deinit();
        m.position = 0;
    }
    pub fn deinit(m: *Model) void {
        m.reset();
        m.weights.deinit();
        m.kernels.deinit();
        m.prefill_ops.deinit();
        m.prefill_route.deinit();
    }
    fn norm(m: *Model, s: *mx.Scope, x: A, name: []const u8) !A {
        return cp.norm(s, x, try m.weights.field(name, "weight"), 1e-5);
    }
    fn lin(m: *Model, s: *mx.Scope, name: []const u8, x: A) !A {
        return m.weights.linear(&m.kernels, s, name, x, true);
    }
    fn f(m: *Model, base: []const u8, suffix: []const u8) !A {
        return m.weights.field(base, suffix);
    }
    fn project(m: *Model, s: *mx.Scope, base: []const u8, suffix: []const u8, x: A) !A {
        var buf: [256]u8 = undefined;
        return m.lin(s, try std.fmt.bufPrint(&buf, "{s}.{s}", .{ base, suffix }), x);
    }
    pub fn forward(m: *Model, tokens: []const i32) !Pass {
        var p = try m.forwardQueued(tokens);
        errdefer p.deinit();
        try mx.eval(p.logits);
        try observeBuffers(&p);
        return p;
    }
    pub fn prefill(m: *Model, tokens: []const i32) !Pass {
        if (tokens.len <= 16) return m.forward(tokens);
        return @import("nemotron_prefill.zig").forward(m, tokens);
    }
    pub fn observeBuffers(p: *Pass) !void {
        if (!kv.track_reuse) return;
        for (p.records) |rec| {
            try kv.observe(rec.key_write);
            try kv.observe(rec.value_write);
        }
    }
    pub fn forwardQueued(m: *Model, tokens: []const i32) !Pass {
        if (tokens.len == 0 or tokens.len > 16) return error.InvalidLaneWidth;
        var s = mx.Scope{};
        defer s.deinit();
        return m.forwardArray(try s.ints(tokens));
    }
    /// Internal GPU-token entry point; IDs come from the validated prompt or sampler.
    pub fn forwardArray(m: *Model, tokens: A) !Pass {
        if (tokens.ctx == null) return error.InvalidToken;
        if (mx.c.mlx_array_ndim(tokens) != 1 or mx.dim(tokens, 0) < 1 or mx.dim(tokens, 0) > 16) return error.InvalidLaneWidth;
        const dtype = mx.c.mlx_array_dtype(tokens);
        if (dtype != mx.c.MLX_INT32 and dtype != mx.c.MLX_UINT32) return error.InvalidToken;
        var p = Pass{};
        errdefer p.deinit();
        const s = &p.scope;
        var h = try m.weights.embedArray(s, "backbone.embeddings", tokens);
        var x = try m.norm(s, h, "backbone.layers.0.norm");
        var buf: [256]u8 = undefined;
        var nb: [256]u8 = undefined;
        for (m.kinds, 0..) |kind, i| {
            const base = try std.fmt.bufPrint(&buf, "backbone.layers.{d}.mixer", .{i});
            const next = if (i + 1 == 52) "backbone.norm_f" else try std.fmt.bufPrint(&nb, "backbone.layers.{d}.norm", .{i + 1});
            const nw = try m.weights.field(next, "weight");
            if (kind == 'M') {
                const delta = try m.mamba(s, base, x, m.cache[i], &p.records[i]);
                const both = try m.addNorm(s, h, delta, nw);
                h = both[0];
                x = both[1];
            } else if (kind == '*') {
                const delta = try m.attention(s, base, x, &m.cache[i], &p.records[i]);
                const both = try m.addNorm(s, h, delta, nw);
                h = both[0];
                x = both[1];
            } else {
                const both = try m.moe(s, base, h, x, nw);
                h = both[0];
                x = both[1];
            }
            if ((i + 1) % 8 == 0) try mx.evalMany(&.{ h, x }, true);
        }
        p.hidden = x;
        p.logits = try m.lin(s, "lm_head", x);
        return p;
    }
    pub fn forwardSerialArray(m: *Model, tokens: A) !Pass {
        if (tokens.ctx == null or mx.c.mlx_array_size(tokens) != 1) return error.InvalidToken;
        return m.forwardArray(tokens);
    }
    fn addNorm(m: *Model, s: *mx.Scope, h: A, delta: A, nw: A) ![5]A {
        const r = mx.dim(h, 0);
        return m.kernels.run(s, src.nemotron_add_norm_plain, &.{ h, delta, nw, try s.scalar(1e-5) }, &.{ ti("D", 2688), ti("T", 896) }, .{ 896 * r, 1, 1 }, .{ 896, 1, 1 }, &.{ .{ .shape = &.{ r, 2688 } }, .{ .shape = &.{ r, 2688 } } });
    }
    fn mamba(m: *Model, s: *mx.Scope, base: []const u8, x: A, cache: Cache, record: *Cache) !A {
        const r = mx.dim(x, 0);
        const p = try m.project(s, base, "in_proj", x);
        const cs = if (cache.a.ctx != null) cache.a else try s.zeros(&.{ 3, 6144 }, mx.bf16);
        const st = if (cache.b.ctx != null) cache.b else try s.zeros(&.{ 64, 64, 128 }, mx.f32t);
        const limits = [_]f32{ 0, std.math.inf(f32) };
        const out = try m.kernels.run(s, src.nemotron_mamba_step, &.{ p, cs, st, try m.f(base, "conv1d.weight"), try s.cast(try m.f(base, "conv1d.bias"), mx.f32t), try s.cast(try m.f(base, "A_log"), mx.f32t), try s.cast(try m.f(base, "D"), mx.f32t), try s.cast(try m.f(base, "dt_bias"), mx.f32t), try s.data(&limits, &.{2}, mx.f32t), try s.ints(&.{r}) }, &.{ ti("H", 64), ti("DH", 64), ti("NG", 8), ti("DS", 128), ti("XD", 4096), ti("KC", 4), ti("PROJ", 10304), ti("XOFF", 4096), ti("DTOFF", 10240), ti("MAXR", 16), ti("TGY", 8), ti("SSZ", 524288) }, .{ 32, 64, 64 }, .{ 32, 8, 1 }, &.{ .{ .shape = &.{ r, 4096 } }, .{ .shape = &.{ r, 3, 6144 } }, .{ .shape = &.{ r, 64, 64, 128 }, .dtype = mx.f32t } });
        record.* = .{ .a = out[1], .b = out[2] };
        const normed = (try m.kernels.run(s, src.nemotron_group_norm, &.{ out[0], try m.f(base, "norm.weight"), try s.scalar(1e-5) }, &.{ ti("XD", 4096), ti("GS", 512) }, .{ 1024, r, 1 }, .{ 128, 1, 1 }, &.{.{ .shape = &.{ r, 4096 } }}))[0];
        return m.project(s, base, "out_proj", normed);
    }
    fn attention(m: *Model, s: *mx.Scope, base: []const u8, x: A, cache: *Cache, record: *Cache) !A {
        const r = mx.dim(x, 0);
        const q = try s.transpose(try s.reshape(try m.project(s, base, "q_proj", x), &.{ 1, r, 32, 128 }), &.{ 0, 2, 1, 3 });
        var keys = try s.transpose(try s.reshape(try m.project(s, base, "k_proj", x), &.{ 1, r, 2, 128 }), &.{ 0, 2, 1, 3 });
        var values = try s.transpose(try s.reshape(try m.project(s, base, "v_proj", x), &.{ 1, r, 2, 128 }), &.{ 0, 2, 1, 3 });
        var kw = kv.Write{};
        var vw = kv.Write{};
        if (kv.enabled) {
            kw = try cache.keys.append(s, cache.a, keys, 2);
            vw = try cache.values.append(s, cache.b, values, 2);
            keys = kw.view;
            values = vw.view;
        } else if (cache.a.ctx != null) {
            keys = try s.cat(&.{ cache.a, keys }, 2);
            values = try s.cat(&.{ cache.b, values }, 2);
        }
        record.* = .{ .a = keys, .b = values, .key_write = kw, .value_write = vw };
        // Each query uses the serial kernel and exactly its own visible key length.
        var rows: [16]A = undefined;
        const start = mx.dim(keys, 2) - r;
        for (0..@intCast(r)) |i| {
            const j: i32 = @intCast(i);
            if (mx.tensor_units and start + j + 1 >= 10000) {
                rows[i] = try @import("lanes.zig").sdpa(&m.kernels, s, try s.slice(q, 2, j, j + 1), try s.slice(keys, 2, 0, start + j + 1), try s.slice(values, 2, 0, start + j + 1), 0.08838834764831845);
                continue;
            }
            var out = mx.c.mlx_array_new();
            const rc = mx.c.mlx_fast_scaled_dot_product_attention(&out, try s.slice(q, 2, j, j + 1), try s.slice(keys, 2, 0, start + j + 1), try s.slice(values, 2, 0, start + j + 1), 0.08838834764831845, "", mx.empty, mx.empty, false, mx.stream);
            rows[i] = try s.result(rc, out);
        }
        const out = try s.reshape(try s.transpose(try s.cat(rows[0..@intCast(r)], 2), &.{ 0, 2, 1, 3 }), &.{ r, 4096 });
        return m.project(s, base, "o_proj", out);
    }
    fn moe(m: *Model, s: *mx.Scope, base: []const u8, h: A, x: A, nw: A) ![5]A {
        const r = mx.dim(x, 0);
        const logits = (try m.kernels.run(s, src.nemotron_router, &.{ x, try m.f(base, "gate.weight"), try s.ints(&.{r}) }, &.{ ti("D", 2688), ti("NE", 128), ti("SG", 8), ti("MAXR", 16) }, .{ 256, 128, 1 }, .{ 256, 1, 1 }, &.{.{ .shape = &.{ r, 128 } }}))[0];
        const route = try m.kernels.run(s, src.nemotron_route, &.{ logits, try s.cast(try m.f(base, "gate.e_score_correction_bias"), mx.f32t), try s.scalar(2.5) }, &.{ ti("NE", 128), ti("K", 6), ti("OK", 6) }, .{ 32 * r, 1, 1 }, .{ 32, 1, 1 }, &.{ .{ .shape = &.{ r, 6 }, .dtype = mx.c.MLX_UINT32 }, .{ .shape = &.{ r, 6 }, .dtype = mx.f32t } });
        var buf: [256]u8 = undefined;
        const up = try m.weights.triple(try std.fmt.bufPrint(&buf, "{s}.switch_mlp.fc1", .{base}));
        const down = try m.weights.triple(try std.fmt.bufPrint(&buf, "{s}.switch_mlp.fc2", .{base}));
        const ids = try s.reshape(route[0], &.{r * 6});
        const act = (try m.kernels.run(s, src.nemotron_rows_expert_up, &.{ x, ids, up[0], up[1], up[2] }, &.{ ti("K", 2688), ti("N", 1856), ti("GS", 64), ti("RPS", 4), ti("SG", 2), ti("TOPK", 6) }, .{ 64, 232, r * 6 }, .{ 64, 1, 1 }, &.{.{ .shape = &.{ r * 6, 1856 } }}))[0];
        const routed = (try m.kernels.run(s, src.nemotron_rows_expert_down, &.{ act, ids, down[0], down[1], down[2] }, &.{ ti("K", 1856), ti("N", 2688), ti("GS", 64), ti("RPS", 4), ti("SG", 2) }, .{ 64, 336, r * 6 }, .{ 64, 1, 1 }, &.{.{ .shape = &.{ r, 6, 2688 } }}))[0];
        const shared_up = try s.binary(mx.c.mlx_maximum, try m.project(s, base, "shared_experts.up_proj", x), try s.cast(try s.scalar(0), mx.bf16));
        const shared = try m.project(s, base, "shared_experts.down_proj", try s.binary(mx.c.mlx_multiply, shared_up, shared_up));
        return m.kernels.run(s, src.nemotron_add_norm_moe, &.{ h, routed, route[1], shared, nw, try s.scalar(1e-5) }, &.{ ti("D", 2688), ti("T", 896), ti("E", 6) }, .{ 896 * r, 1, 1 }, .{ 896, 1, 1 }, &.{ .{ .shape = &.{ r, 2688 } }, .{ .shape = &.{ r, 2688 } } });
    }
    pub fn commit(m: *Model, p: *Pass, keep: usize) !void {
        return m.commitImpl(p, keep, true);
    }
    pub fn commitSerialQueued(m: *Model, p: *Pass) !void {
        if (mx.dim(p.hidden, 0) != 1) return error.InvalidCommit;
        return m.commitImpl(p, 1, false);
    }
    fn commitImpl(m: *Model, p: *Pass, keep: usize, evaluate: bool) !void {
        if (keep == 0 or keep > @as(usize, @intCast(mx.dim(p.hidden, 0)))) return error.InvalidCommit;
        if (p.prefilled and keep != mx.dim(p.hidden, 0)) return error.InvalidCommit;
        var next: [52]Cache = @splat(.{});
        errdefer for (&next) |*c| c.deinit();
        const n: i32 = @intCast(keep);
        for (m.kinds, 0..) |kind, i| {
            const rec = p.records[i];
            if (kind == 'M') {
                next[i].a = try mx.retain(if (p.prefilled) rec.a else try p.scope.slice(rec.a, 0, n - 1, n));
                next[i].b = try mx.retain(if (p.prefilled) rec.b else try p.scope.slice(rec.b, 0, n - 1, n));
            }
            if (kind == '*') {
                next[i].a = try mx.retain(try p.scope.slice(rec.a, 2, 0, m.position + n));
                next[i].b = try mx.retain(try p.scope.slice(rec.b, 2, 0, m.position + n));
                next[i].keys = try m.cache[i].keys.finish(&p.scope, rec.key_write, n);
                next[i].values = try m.cache[i].values.finish(&p.scope, rec.value_write, n);
            }
            if (kind != 'E' and evaluate) try mx.evalMany(&.{ next[i].a, next[i].b }, false);
        }
        for (&m.cache) |*c| c.deinit();
        m.cache = next;
        m.position += n;
    }
    pub fn draftStep(m: *Model, s: *mx.Scope, hidden: A, token: i32, cache: *Cache) !A {
        return m.draftStepArray(s, hidden, try s.ints(&.{token}), cache, false);
    }
    pub fn draftStepArray(m: *Model, s: *mx.Scope, hidden: A, token: A, cache: *Cache, queued: bool) !A {
        const rows = mx.dim(hidden, 0);
        if (rows < 1 or rows > 16 or mx.c.mlx_array_size(token) != @as(usize, @intCast(rows))) return error.InvalidDraftRows;
        const e = try m.norm(s, try m.weights.embedArray(s, "backbone.embeddings", token), "mtp.layers.0.enorm");
        const h = try m.norm(s, hidden, "mtp.layers.0.hnorm");
        var x = try m.lin(s, "mtp.layers.0.eh_proj", try s.cat(&.{ e, h }, -1));
        var record = Cache{};
        const delta = try m.attention(s, "mtp.layers.0.mixer", try m.norm(s, x, "mtp.layers.0.norm"), cache, &record);
        x = try s.binary(mx.c.mlx_add, x, delta);
        const out = try m.moe(s, "mtp.layers.1.mixer", x, try m.norm(s, x, "mtp.layers.1.norm"), try m.weights.get("mtp.layers.1.final_layernorm.weight"));
        if (!queued) try mx.evalMany(&.{ out[1], record.a, record.b }, false);
        var next = try record.clone();
        errdefer next.deinit();
        next.keys = try cache.keys.finish(s, record.key_write, rows);
        next.values = try cache.values.finish(s, record.value_write, rows);
        cache.deinit();
        cache.* = next;
        return out[1];
    }
    pub fn draftPrefix(s: *mx.Scope, cache: Cache, rows: usize, keep: usize) !Cache {
        if (rows == 0 or keep > rows or rows > @as(usize, @intCast(mx.dim(cache.a, 2)))) return error.InvalidCommit;
        const end = mx.dim(cache.a, 2) - @as(i32, @intCast(rows - keep));
        if (end == 0) return .{};
        return (Cache{ .a = try s.slice(cache.a, 2, 0, end), .b = try s.slice(cache.b, 2, 0, end), .keys = try cache.keys.prefix(s, end), .values = try cache.values.prefix(s, end) }).clone();
    }
    pub fn head(m: *Model, s: *mx.Scope, h: A) !A {
        return m.lin(s, "lm_head", h);
    }
    pub fn draftHead(m: *Model, s: *mx.Scope, h: A) !A {
        return m.lin(s, if (m.weights.has("draft_ids")) "draft_lm_head" else "lm_head", h);
    }
    pub const draft_vocabulary = @import("draft_vocab.zig").data.nemotron;
    pub const draft_prior = &@import("draft_depth.zig").nemotron_prior;
};
