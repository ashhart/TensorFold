const std = @import("std");
const mx = @import("mlx.zig");
const lanes = @import("lanes.zig");
const src = @import("kernel_sources.zig");
const A = mx.Array;
const ti = mx.ti;
const Weights = @import("weights.zig").Weights;
const kv = @import("kv_buffer.zig");
pub const Cache = struct {
    a: A = mx.empty,
    b: A = mx.empty,
    keys: kv.Buffer = .{},
    values: kv.Buffer = .{},
    // Forward records borrow these from their Pass scope; persistent clones omit them.
    key_write: kv.Write = .{},
    value_write: kv.Write = .{},
    pub fn clone(c: Cache) !Cache {
        var out = Cache{};
        errdefer out.deinit();
        if (c.a.ctx != null) out.a = try mx.retain(c.a);
        if (c.b.ctx != null) out.b = try mx.retain(c.b);
        out.keys = try c.keys.clone();
        out.values = try c.values.clone();
        return out;
    }
    pub fn deinit(c: *Cache) void {
        mx.free(c.a);
        mx.free(c.b);
        c.keys.deinit();
        c.values.deinit();
        c.* = .{};
    }
};
pub const Record = struct { values: [8]A = @splat(mx.empty), key_write: kv.Write = .{}, value_write: kv.Write = .{} };
pub const Pass = struct {
    scope: mx.Scope = .{},
    logits: A = mx.empty,
    records: [64]Record = @splat(.{}),
    taps: [5]A = @splat(mx.empty),
    parents: [2048]i32 = undefined,
    count: usize = 0,
    start: i32 = 0,
    pub fn deinit(p: *Pass) void {
        p.scope.deinit();
    }
};
pub const Model = struct {
    pub const SerialPass = Pass;
    weights: Weights,
    kernels: mx.Kernels,
    cache: [64]Cache = @splat(.{}),
    position: i32 = 0,
    prefill_ops: @import("prefill_ops.zig").Ops = .{},
    trace_dir: ?[]const u8 = null,
    pub fn init(io: std.Io, dir: []const u8) !Model {
        var m = Model{ .weights = Weights.init(), .kernels = mx.Kernels.init() };
        errdefer m.deinit();
        try m.weights.load(io, dir);
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
    }
    pub fn weight(m: *Model, index: usize, suffix: []const u8) !A {
        var buf: [192]u8 = undefined;
        return m.weights.get(try std.fmt.bufPrint(&buf, "model.layers.{d}.{s}", .{ index, suffix }));
    }
    fn project(m: *Model, s: *mx.Scope, index: usize, suffix: []const u8, x: lanes.Act) !A {
        var buf: [192]u8 = undefined;
        const l = try m.weights.linear(try std.fmt.bufPrint(&buf, "model.layers.{d}.{s}", .{ index, suffix }));
        // Python's SIMD decoder stacks [qkv,z,b,a] and [q, k, v]. The combined
        // output widths select eight reduction chunks, including the small heads.
        if (!mx.tensor_units) {
            for ([_][]const u8{ "linear_attn.in_proj_z", "linear_attn.in_proj_b", "linear_attn.in_proj_a", "self_attn.k_proj", "self_attn.v_proj" }) |stacked|
                if (std.mem.eql(u8, suffix, stacked)) return l.applyWithReduction(&m.kernels, s, x, 8);
        }
        return l.apply(&m.kernels, s, x);
    }
    pub fn prefillProject(m: *Model, s: *mx.Scope, index: usize, suffix: []const u8, x: A) !A {
        var buf: [192]u8 = undefined;
        const l = try m.weights.linear(try std.fmt.bufPrint(&buf, "model.layers.{d}.{s}", .{ index, suffix }));
        return l.prefill(&m.kernels, s, x);
    }
    pub fn prefill(m: *Model, tokens: []const i32) !Pass {
        return @import("qwen_prefill.zig").forward(m, tokens);
    }
    pub fn trace(m: *Model, s: *mx.Scope, position: i32, layer: usize, label: []const u8, value: A) !void {
        const dir = m.trace_dir orelse return;
        var buf: [4096]u8 = undefined;
        const path = try std.fmt.bufPrintSentinel(&buf, "{s}/{d}-{d}-{s}.npy", .{ dir, position, layer, label }, 0);
        const out = try s.cast(value, mx.f32t);
        try mx.eval(out);
        try mx.check(mx.c.mlx_save(path, out));
    }
    pub fn forward(m: *Model, tokens: []const i32, parents: []const i32) !Pass {
        if (tokens.len != parents.len) return error.InvalidTree;
        var s = mx.Scope{};
        defer s.deinit();
        var p = try m.forwardTokens(try s.ints(tokens), parents);
        errdefer p.deinit();
        try mx.eval(p.logits);
        try observeBuffers(&p);
        return p;
    }
    pub fn observeBuffers(p: *Pass) !void {
        if (!kv.track_reuse) return;
        for (p.records) |rec| {
            try kv.observe(rec.key_write);
            try kv.observe(rec.value_write);
        }
    }
    pub fn forwardSerialArray(m: *Model, tokens: A) !Pass {
        if (tokens.ctx == null or mx.c.mlx_array_size(tokens) != 1) return error.InvalidToken;
        return m.forwardTokens(tokens, &.{-1});
    }
    fn forwardTokens(m: *Model, tokens: A, parents: []const i32) !Pass {
        if (mx.c.mlx_array_ndim(tokens) != 1 or mx.c.mlx_array_size(tokens) != parents.len) return error.InvalidTree;
        if (mx.dtype(tokens) != mx.i32t and mx.dtype(tokens) != mx.c.MLX_UINT32) return error.InvalidToken;
        const tree = try lanes.Tree.init(parents);
        var p = Pass{ .count = parents.len, .start = m.position };
        @memcpy(p.parents[0..parents.len], parents);
        errdefer p.deinit();
        const s = &p.scope;
        const kernels = &m.kernels;
        var h = try m.weights.embedArray(s, tokens);
        var pending: ?A = null;
        var positions: [128]i32 = undefined;
        for (0..parents.len) |i| positions[i] = m.position + tree.depths[i];
        const pos = try s.ints(positions[0..parents.len]);
        for (0..64) |i| {
            const inorm = try lanes.norm(kernels, s, h, pending, try m.weight(i, "input_layernorm.weight"));
            h = inorm.h;
            const r = if (i % 4 == 3) try m.attn(s, i, inorm.x, &tree, pos, &p.records[i]) else try m.gdn(s, i, inorm.x, &tree, &p.records[i]);
            const post = try lanes.norm(kernels, s, h, r, try m.weight(i, "post_attention_layernorm.weight"));
            h = post.h;
            const act = try lanes.mlp(kernels, s, try m.project(s, i, "mlp.gate_proj", post.x), try m.project(s, i, "mlp.up_proj", post.x));
            pending = try m.project(s, i, "mlp.down_proj", act);
            // DFlash taps are the post-residual layer outputs, before the next norm.
            for ([_]usize{ 5, 19, 33, 47, 61 }, 0..) |layer, j| if (i == layer) {
                p.taps[j] = try s.binary(mx.c.mlx_add, h, pending.?);
            };
            if (i == 0 or (i + 1) % 4 == 0) try mx.evalMany(&.{ h, pending.? }, true);
        }
        const normed = try lanes.norm(kernels, s, h, pending, try m.weights.get("model.norm.weight"));
        p.logits = try (try m.weights.linear("lm_head")).apply(kernels, s, normed.x);
        try m.trace(s, m.position, 64, "decode-logits", p.logits);
        return p;
    }
    fn attn(m: *Model, s: *mx.Scope, i: usize, x: lanes.Act, t: *const lanes.Tree, pos: A, rec: *Record) !A {
        const w: i32 = @intCast(t.parents.len);
        const qg = try s.reshape(try m.project(s, i, "self_attn.q_proj", x), &.{ 1, w, 24, 512 });
        var q = try s.rms(try s.slice(qg, 3, 0, 256), try m.weight(i, "self_attn.q_norm.weight"));
        const gate = try s.reshape(try s.slice(qg, 3, 256, 512), &.{ 1, w, 6144 });
        var key = try s.rms(try s.reshape(try m.project(s, i, "self_attn.k_proj", x), &.{ 1, w, 4, 256 }), try m.weight(i, "self_attn.k_norm.weight"));
        var value = try s.transpose(try s.reshape(try m.project(s, i, "self_attn.v_proj", x), &.{ 1, w, 4, 256 }), &.{ 0, 2, 1, 3 });
        q = try s.transpose(try s.rope(try s.transpose(q, &.{ 1, 2, 0, 3 }), pos, 64), &.{ 2, 1, 0, 3 });
        key = try s.transpose(try s.rope(try s.transpose(key, &.{ 1, 2, 0, 3 }), pos, 64), &.{ 2, 1, 0, 3 });
        rec.values[0] = key;
        rec.values[1] = value;
        if (kv.enabled) {
            rec.key_write = try m.cache[i].keys.append(s, m.cache[i].a, key, 2);
            rec.value_write = try m.cache[i].values.append(s, m.cache[i].b, value, 2);
            key = rec.key_write.capacity;
            value = rec.value_write.capacity;
        } else if (m.cache[i].a.ctx != null) {
            key = try s.cat(&.{ m.cache[i].a, key }, 2);
            value = try s.cat(&.{ m.cache[i].b, value }, 2);
        }
        const out = try s.reshape(try s.transpose(try lanes.attentionCapacity(&m.kernels, s, q, key, value, t, m.position + w), &.{ 0, 2, 1, 3 }), &.{ 1, w, 6144 });
        return m.project(s, i, "self_attn.o_proj", .{ .x = try s.binary(mx.c.mlx_multiply, out, try s.unary(mx.c.mlx_sigmoid, gate)) });
    }
    fn gdn(m: *Model, s: *mx.Scope, i: usize, x: lanes.Act, t: *const lanes.Tree, rec: *Record) !A {
        const w: i32 = @intCast(t.parents.len);
        const mp = @divTrunc(w + 15, 16) * 16;
        const qkv = try m.project(s, i, "linear_attn.in_proj_qkv", x);
        const z = try m.project(s, i, "linear_attn.in_proj_z", x);
        const b = try m.project(s, i, "linear_attn.in_proj_b", x);
        const a = try m.project(s, i, "linear_attn.in_proj_a", x);
        const cs = if (m.cache[i].a.ctx != null) m.cache[i].a else try s.zeros(&.{ 1, 3, 10240 }, mx.bf16);
        const st = if (m.cache[i].b.ctx != null) m.cache[i].b else try s.zeros(&.{ 1, 48, 128, 128 }, mx.f32t);
        const cw = try s.reshape(try m.weight(i, "linear_attn.conv1d.weight"), &.{ 10240, 4 });
        const vals = try m.kernels.run(s, src.lane_glue_gdn_pre, &.{ qkv, cs, cw, try s.ints(t.windows[0 .. t.parents.len * 4]), a, b, try m.weight(i, "linear_attn.A_log"), try m.weight(i, "linear_attn.dt_bias") }, &.{ ti("NK", 16), ti("NV", 48), ti("DK", 128), ti("DV", 128), ti("TAPS", 4) }, .{ 32, 80, w }, .{ 32, 1, 1 }, &.{ .{ .shape = &.{ 1, w, 16, 128 } }, .{ .shape = &.{ 1, w, 16, 128 } }, .{ .shape = &.{ 1, w, 48, 128 } }, .{ .shape = &.{ 1, w, 48 }, .dtype = mx.f32t }, .{ .shape = &.{ 1, w, 48 } } });
        const y = (try m.kernels.run(s, src.lane_tree_tree, &.{ vals[0], vals[1], vals[2], vals[3], vals[4], st, try s.ints(t.parents), try s.ints(&.{w}) }, &.{ mx.td("InT", mx.bf16), ti("Dk", 128), ti("Dv", 128), ti("Hk", 16), ti("Hv", 48), ti("MAXW", if (t.chain) 1 else if (w <= 16) 16 else 32), mx.tb("CHAIN", t.chain) }, .{ 32, 128, 48 }, .{ 32, 4, 1 }, &.{.{ .shape = &.{ 1, w, 48, 128 } }}))[0];
        @memcpy(rec.values[0..5], vals[0..5]);
        // A pass must own its replay base even if the caller restores/replaces the
        // live cache before committing a different accepted path from this pass.
        rec.values[5] = try s.own(try mx.retain(st));
        rec.values[6] = try s.cat(&.{ cs, qkv }, 1);
        const post = try m.kernels.run(s, src.lane_glue_gdn_post, &.{ y, z, try m.weight(i, "linear_attn.norm.weight"), try s.scalar(1e-6), try s.ints(&.{ w, mp }) }, &.{ ti("NV", 48), ti("DV", 128) }, .{ 32, 48, mp }, .{ 32, 1, 1 }, &.{ .{ .shape = &.{ 1, w, 6144 } }, .{ .shape = &.{ 96, mp }, .dtype = mx.f32t } });
        return m.project(s, i, "linear_attn.out_proj", .{ .x = post[0], .sums = post[1] });
    }
    pub fn commit(m: *Model, p: *Pass, rows: []const i32) !void {
        return m.commitImpl(p, rows, true);
    }
    pub fn commitSerialQueued(m: *Model, p: *Pass) !void {
        if (p.count != 1) return error.InvalidCommit;
        return m.commitImpl(p, &.{0}, false);
    }
    fn commitImpl(m: *Model, p: *Pass, rows: []const i32, evaluate: bool) !void {
        if (rows.len == 0) return error.EmptyCommit;
        if (m.position != p.start or rows.len > p.count or rows[0] != 0) return error.InvalidCommit;
        for (rows, 0..) |row, i| {
            if (row < 0 or row >= p.count or p.parents[@intCast(row)] != (if (i == 0) @as(i32, -1) else rows[i - 1])) return error.InvalidCommit;
        }
        var commit_scope = mx.Scope{};
        defer commit_scope.deinit();
        const s = &commit_scope;
        const ids = try s.ints(rows);
        const count = try s.ints(&.{@intCast(rows.len)});
        // Construct all replacement handles before publishing the new cache.
        // Serial pipelining evaluates the graphs with the next draw; regular
        // commits also wait for GPU execution before publishing.
        var next: [64]Cache = @splat(.{});
        errdefer for (&next) |*c| c.deinit();
        for (&p.records, 0..) |*rec, i| {
            const v = rec.values;
            if (i % 4 == 3) {
                var consecutive = true;
                for (rows, 0..) |row, j| if (row != j) {
                    consecutive = false;
                };
                if (kv.enabled and consecutive and rec.key_write.capacity.ctx != null) {
                    const end = m.position + @as(i32, @intCast(rows.len));
                    next[i].a = try mx.retain(try s.slice(rec.key_write.capacity, 2, 0, end));
                    next[i].b = try mx.retain(try s.slice(rec.value_write.capacity, 2, 0, end));
                    next[i].keys = try m.cache[i].keys.finish(s, rec.key_write, @intCast(rows.len));
                    next[i].values = try m.cache[i].values.finish(s, rec.value_write, @intCast(rows.len));
                    continue;
                }
                var keys = try s.take(v[0], ids, 2);
                var vals = try s.take(v[1], ids, 2);
                if (kv.enabled) {
                    // A branched path needs compaction. Verification reused a spare;
                    // commit writes the gathered path into a protected replacement.
                    const kw = try m.cache[i].keys.append(s, m.cache[i].a, keys, 2);
                    const vw = try m.cache[i].values.append(s, m.cache[i].b, vals, 2);
                    keys = kw.view;
                    vals = vw.view;
                    next[i].keys = try m.cache[i].keys.finish(s, kw, @intCast(rows.len));
                    next[i].values = try m.cache[i].values.finish(s, vw, @intCast(rows.len));
                } else if (m.cache[i].a.ctx != null) {
                    keys = try s.cat(&.{ m.cache[i].a, keys }, 2);
                    vals = try s.cat(&.{ m.cache[i].b, vals }, 2);
                }
                next[i].a = try mx.retain(keys);
                next[i].b = try mx.retain(vals);
            } else {
                const state = (try m.kernels.run(s, src.lane_tree_replay, &.{ v[0], v[1], v[2], v[3], v[4], v[5], ids, count }, &.{ ti("Dk", 128), ti("Dv", 128), ti("Hk", 16), ti("Hv", 48), mx.td("StT", mx.f32t) }, .{ 32, 128, 48 }, .{ 32, 4, 1 }, &.{.{ .shape = &.{ 1, 48, 128, 128 }, .dtype = mx.f32t }}))[0];
                var tail: [3]i32 = undefined;
                for (0..3) |j| {
                    const n = @as(i32, @intCast(rows.len)) + @as(i32, @intCast(j));
                    tail[j] = if (n < 3) n else 3 + rows[@intCast(n - 3)];
                }
                next[i].a = try mx.retain(try s.contiguous(try s.take(v[6], try s.ints(&tail), 1)));
                next[i].b = try mx.retain(state);
            }
        }
        var arrays: [128]A = undefined;
        for (next, 0..) |c, i| {
            arrays[2 * i] = c.a;
            arrays[2 * i + 1] = c.b;
        }
        if (evaluate) try mx.evalMany(&arrays, false);
        for (&m.cache) |*c| c.deinit();
        m.cache = next;
        m.position += @intCast(rows.len);
    }
};
