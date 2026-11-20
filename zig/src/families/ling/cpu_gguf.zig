//! Checked borrowed GGUF source views for Ling-3.0-tiny on CPU; not a serving admission path.
const std = @import("std");
const g = @import("core").gguf;
pub const reference = @import("core").gguf_source_reference;
pub const views = @import("cpu_gguf_views.zig");
pub const names = @import("gguf_names.zig");

fn integer(d: *const g.Document, key: []const u8, expected: u64) !void {
    const v = d.get(key) orelse return error.MissingLingGgufMetadata;
    const n: u64 = switch (v) {
        .uint32 => |x| x,
        .uint64 => |x| x,
        else => return error.BadLingGgufMetadata,
    };
    if (n != expected) return error.UnsupportedLingGgufGeometry;
}
fn metadata(d: *const g.Document) !void {
    const arch = d.get("general.architecture") orelse return error.MissingLingGgufMetadata;
    if (arch != .string or !std.mem.eql(u8, arch.string, "bailingmoe3")) return error.UnsupportedLingGgufArchitecture;
    const checks = .{ .{ "block_count", 24 }, .{ "embedding_length", 1536 }, .{ "feed_forward_length", 4608 }, .{ "attention.head_count", 16 }, .{ "expert_count", 128 }, .{ "expert_used_count", 8 }, .{ "expert_group_count", 8 }, .{ "expert_group_used_count", 4 }, .{ "expert_gating_func", 2 }, .{ "vocab_size", 157184 }, .{ "ssm.conv_kernel", 4 }, .{ "kda.head_dim", 128 }, .{ "attention.q_lora_rank", 256 }, .{ "attention.kv_lora_rank", 512 }, .{ "rope.dimension_count", 64 }, .{ "expert_feed_forward_length", 512 }, .{ "expert_shared_feed_forward_length", 512 }, .{ "leading_dense_block_count", 1 } };
    inline for (checks) |check| try integer(d, "bailingmoe3." ++ check[0], check[1]);
    try integer(d, "bailingmoe3.context_length", 131072);
    try integer(d, "bailingmoe3.attention.key_length_mla", 192);
    try integer(d, "bailingmoe3.attention.value_length_mla", 128);
    try integer(d, "bailingmoe3.expert_shared_count", 1);
    inline for (.{ .{ "rope.freq_base", 6000000.0 }, .{ "attention.layer_norm_rms_epsilon", 0.000001 }, .{ "kda.gate_lower_bound", -5.0 }, .{ "expert_weights_scale", 2.5 } }) |check| {
        const value = d.get("bailingmoe3." ++ check[0]) orelse return error.MissingLingGgufMetadata;
        const actual = try value.float();
        if (!std.math.isFinite(actual) or @as(f32, @floatCast(actual)) != @as(f32, check[1])) return error.UnsupportedLingGgufGeometry;
    }
    inline for (.{ "kda.safe_gate", "expert_weights_norm" }) |key| {
        const value = d.get("bailingmoe3." ++ key) orelse return error.MissingLingGgufMetadata;
        if (value != .boolean or !value.boolean) return error.UnsupportedLingGgufGeometry;
    }
    const kv = d.get("bailingmoe3.attention.head_count_kv") orelse return error.MissingLingGgufMetadata;
    if (kv != .array or kv.array.len != 24) return error.UnsupportedLingGgufGeometry;
    var it = kv.array.iterator();
    for (0..24) |layer| {
        const value = (try it.next()) orelse return error.UnsupportedLingGgufGeometry;
        if (try value.unsigned() != @as(u64, if (layer % 4 == 3) 1 else 0)) return error.UnsupportedLingGgufGeometry;
    }
    const tokens = d.get("tokenizer.ggml.tokens") orelse return error.MissingLingGgufMetadata;
    if (tokens != .array or tokens.array.kind != .string or tokens.array.len != 157184) return error.UnsupportedLingGgufGeometry;
    if (d.get("bailingmoe3.swiglu_clamp_exp") != null or d.get("bailingmoe3.swiglu_clamp_shexp") != null) return error.UnsupportedLingGgufClamp;
}
pub fn checkTensor(t: g.Tensor, n: names.Name) !void {
    if (t.rank != n.rank or !std.mem.eql(u64, t.shape[0..t.rank], n.dims[0..n.rank])) return error.BadLingGgufTensorShape;
    switch (n.role) {
        .positive_exp_decay => {
            if (t.kind != .f32 or t.bytes.len != 64) return error.UnsupportedLingGgufSmallType;
            for (0..16) |i| {
                const rate: f32 = @bitCast(std.mem.readInt(u32, t.bytes[i * 4 ..][0..4], .little));
                if (!std.math.isFinite(rate) or rate <= 0) return error.BadLingGgufDecay;
            }
        },
        .vector, .convolution => if (t.kind != .f32) return error.UnsupportedLingGgufSmallType,
        else => switch (t.kind) {
            .f32, .q4_k, .q6_k, .q5_0 => {},
            else => return error.UnsupportedLingGgufQuant,
        },
    }
}
pub const File = struct {
    const gguf_sdot = @import("cpu_gguf_q4_sdot.zig");
    const gguf_repack8 = @import("cpu_gguf_repack8.zig");
    const gguf_repack_cache = @import("cpu_gguf_repack_cache.zig");
    const CachedValue = struct { data: []f32, rank: u8, shape: [4]usize };
    source: g.File,
    path: []const u8,
    arena: std.heap.ArenaAllocator,
    plan: []names.Name,
    joined_mla: [24]?[]f32 = @splat(null),
    cached_values: std.StringHashMapUnmanaged(CachedValue) = .empty,
    validated: std.StringHashMapUnmanaged(void) = .empty,
    repack_cache: ?gguf_repack_cache.Cache = null,
    fast_mode: bool = false,
    io: std.Io,
    pool: ?*Pool = null,
    pub const Pool = @import("cpu_gguf_pool.zig").Pool;
    pub fn open(a: std.mem.Allocator, io: std.Io, path: []const u8) !File {
        var source = try g.File.open(a, io, path, .{});
        errdefer source.close(io);
        try metadata(&source.document);
        var arena = std.heap.ArenaAllocator.init(a);
        errdefer arena.deinit();
        const dup_path = try arena.allocator().dupe(u8, path);
        const plan = try names.tiny(arena.allocator());
        if (source.document.tensors.len != plan.len) return error.UnexpectedLingGgufTensor;
        for (plan) |name| try checkTensor(source.document.getTensor(name.source) orelse return error.MissingLingGgufTensor, name);
        return .{ .source = source, .path = dup_path, .arena = arena, .plan = plan, .io = io };
    }
    /// Opt-in fast integer SDOT and 8x4 repacked weight cache; exact mode remains the default.
    pub fn enableFastMode(f: *File) !void {
        try gguf_sdot.checkTarget();
        if (f.repack_cache == null) {
            f.repack_cache = try gguf_repack_cache.Cache.openOrBuild(f.arena.child_allocator, f.io, f.path, &f.source);
        }
        f.fast_mode = true;
    }
    pub fn config(f: *const File) !@import("config.zig").Config {
        return @import("cpu_gguf_config.zig").tiny(&f.source.document);
    }
    pub fn view(f: *const File, key: []const u8) !views.View {
        return views.resolve(&f.source.document, f.plan, key);
    }
    pub fn values(f: *File, _: std.mem.Allocator, key: []const u8, shape: []const usize) ![]f32 {
        if (f.cached_values.get(key)) |entry| {
            if (shape.len != entry.rank or !std.mem.eql(usize, shape, entry.shape[0..entry.rank])) return error.BadLingGgufTensorShape;
            return entry.data;
        }
        const v = try f.view(key);
        const data: []f32 = if (v.name.role != .mla_key_transposed)
            try @import("cpu_gguf_values.zig").small(f.arena.allocator(), v, shape)
        else blk: {
            if (!std.mem.eql(usize, shape, &.{ 4096, 512 })) return error.BadLingGgufTensorShape;
            const prefix = "model.layers.";
            if (!std.mem.startsWith(u8, key, prefix)) return error.MissingLingGgufTensor;
            const tail = key[prefix.len..];
            const end = std.mem.indexOfScalar(u8, tail, '.') orelse return error.MissingLingGgufTensor;
            const layer = try std.fmt.parseInt(usize, tail[0..end], 10);
            if (layer >= 24 or layer % 4 != 3) return error.BadLingGgufTensorShape;
            if (f.joined_mla[layer] == null) {
                var buf: [64]u8 = undefined;
                const name = try std.fmt.bufPrint(&buf, "blk.{d}.attn_v_b.weight", .{layer});
                const value = f.source.document.getTensor(name) orelse return error.MissingLingGgufTensor;
                f.joined_mla[layer] = try @import("cpu_gguf_values.zig").joinMla(f.arena.child_allocator, v.tensor, value);
            }
            break :blk f.joined_mla[layer].?;
        };
        if (shape.len <= 4) {
            var stored_shape: [4]usize = @splat(0);
            @memcpy(stored_shape[0..shape.len], shape);
            const dup = try f.arena.allocator().dupe(u8, key);
            try f.cached_values.put(f.arena.allocator(), dup, .{
                .data = data,
                .rank = @intCast(shape.len),
                .shape = stored_shape,
            });
        }
        return data;
    }
    fn validateOnce(f: *File, key: []const u8, t: g.Tensor) !void {
        if (t.kind != .f32 and !f.validated.contains(key)) {
            try g.affine.validate(t.kind, t.bytes);
            const dup = try f.arena.allocator().dupe(u8, key);
            try f.validated.put(f.arena.allocator(), dup, {});
        }
    }
    /// A matrix view whose quantized scales have been validated once.
    fn matrix(f: *File, key: []const u8, width: usize, n: usize) !g.Tensor {
        const v = try f.view(key);
        if (v.tensor.rank != 2 or v.tensor.shape[0] != width or v.tensor.shape[1] != n) return error.BadLingGgufMatrix;
        try f.validateOnce(key, v.tensor);
        return v.tensor;
    }
    /// Fast-mode matrix lookup: returns immediately on repack cache hit without faulting cold raw GGUF pages.
    fn fastMatrix(f: *File, key: []const u8, width: usize, n: usize) !struct { tensor: g.Tensor, rep: ?gguf_repack8.Repacked } {
        const v = try f.view(key);
        if (v.tensor.rank != 2 or v.tensor.shape[0] != width or v.tensor.shape[1] != n) return error.BadLingGgufMatrix;
        if (f.lookupRepacked(v.tensor)) |rep| return .{ .tensor = v.tensor, .rep = rep };
        try f.validateOnce(key, v.tensor);
        return .{ .tensor = v.tensor, .rep = null };
    }
    /// The persistent workers for this thread count, made on first use and replaced when the count changes.
    pub fn workers(f: *File, threads: usize) !*Pool {
        if (f.pool) |p| {
            if (p.threads == threads) return p;
            p.destroy();
            f.pool = null;
        }
        f.pool = try Pool.create(f.arena.child_allocator, f.io, threads);
        return f.pool.?;
    }
    fn lookupRepacked(f: *const File, t: g.Tensor) ?gguf_repack8.Repacked {
        const cache = f.repack_cache orelse return null;
        if (t.kind != .q4_k and t.kind != .q6_k) return null;
        const src_off = @intFromPtr(t.bytes.ptr) - @intFromPtr(f.source.map.memory.ptr);
        return cache.lookup(src_off, t.bytes.len, t.kind);
    }
    pub fn project(f: *File, key: []const u8, input: []const f32, output: []f32, count: usize, width: usize, n: usize, rounded: bool, threads: usize) !void {
        if (f.fast_mode) {
            const m = try f.fastMatrix(key, width, n);
            if (m.rep) |rep| {
                if (threads > 1) return (try f.workers(threads)).projectSdotRepacked(f.arena.child_allocator, rep, m.tensor, input, output, count, rounded);
                return gguf_repack8.projectBatch(f.arena.child_allocator, rep, m.tensor, input, output, count, rounded);
            }
            if (m.tensor.kind == .q4_k or m.tensor.kind == .q6_k or m.tensor.kind == .q5_0) {
                if (threads > 1) return (try f.workers(threads)).projectSdot(f.arena.child_allocator, m.tensor, input, output, count, rounded);
                return gguf_sdot.projectBatch(f.arena.child_allocator, m.tensor, input, output, count, rounded);
            }
            if (threads > 1) return (try f.workers(threads)).project(m.tensor, input, output, count, rounded);
            return @import("cpu_gguf_rows.zig").projectValidatedOn(m.tensor, input, output, count, rounded, 1);
        }
        const t = try f.matrix(key, width, n);
        if (threads > 1) return (try f.workers(threads)).project(t, input, output, count, rounded);
        try @import("cpu_gguf_rows.zig").projectValidatedOn(t, input, output, count, rounded, 1);
    }
    /// One row's routed experts on the workers; `out` holds each result, `hidden` values apiece, in `ids` order.
    pub fn projectExperts(f: *File, prefix: []const u8, ids: []const u32, x: []const f32, out: []f32, hidden: usize, inter: usize, limit: f32, threads: usize) !void {
        if (f.fast_mode) {
            var list: [16]@import("cpu_gguf_pool.zig").FastExpert = undefined;
            if (ids.len > list.len or out.len != ids.len * hidden) return error.BadLingGgufMatrix;
            var acts_buf: [16]gguf_sdot.Activation256 = undefined;
            const hidden_blocks = hidden / 256;
            const can_sdot_in = hidden % 256 == 0 and hidden_blocks <= acts_buf.len;
            const can_sdot_down = inter % 256 == 0 and inter / 256 <= 8;
            var any_sdot = false;
            var buf: [128]u8 = undefined;
            for (ids, 0..) |id, j| {
                const gm = try f.fastMatrix(try std.fmt.bufPrint(&buf, "{s}experts.{d}.gate_proj.weight", .{ prefix, id }), hidden, inter);
                const um = try f.fastMatrix(try std.fmt.bufPrint(&buf, "{s}experts.{d}.up_proj.weight", .{ prefix, id }), hidden, inter);
                const dm = try f.fastMatrix(try std.fmt.bufPrint(&buf, "{s}experts.{d}.down_proj.weight", .{ prefix, id }), inter, hidden);
                const use_g = can_sdot_in and (gm.tensor.kind == .q4_k or gm.tensor.kind == .q6_k);
                const use_u = can_sdot_in and (um.tensor.kind == .q4_k or um.tensor.kind == .q6_k);
                const use_d = can_sdot_down and (dm.tensor.kind == .q4_k or dm.tensor.kind == .q6_k);
                if (use_g or use_u) any_sdot = true;
                list[j] = .{
                    .gate = gm.tensor,
                    .up = um.tensor,
                    .down = dm.tensor,
                    .gate_rep = if (use_g) gm.rep else null,
                    .up_rep = if (use_u) um.rep else null,
                    .down_rep = if (use_d) dm.rep else null,
                    .use_sdot_gate = use_g,
                    .use_sdot_up = use_u,
                    .use_sdot_down = use_d,
                    .out = out[j * hidden ..].ptr,
                };
            }
            if (any_sdot) try gguf_sdot.quantize256(x, acts_buf[0..hidden_blocks], true);
            try (try f.workers(threads)).expertsFast(list[0..ids.len], x, &acts_buf, hidden, inter, limit);
            return;
        }
        var list: [16]@import("cpu_gguf_pool.zig").Expert = undefined;
        if (ids.len > list.len or out.len != ids.len * hidden) return error.BadLingGgufMatrix;
        var buf: [128]u8 = undefined;
        for (ids, 0..) |id, j| list[j] = .{
            .gate = try f.matrix(try std.fmt.bufPrint(&buf, "{s}experts.{d}.gate_proj.weight", .{ prefix, id }), hidden, inter),
            .up = try f.matrix(try std.fmt.bufPrint(&buf, "{s}experts.{d}.up_proj.weight", .{ prefix, id }), hidden, inter),
            .down = try f.matrix(try std.fmt.bufPrint(&buf, "{s}experts.{d}.down_proj.weight", .{ prefix, id }), inter, hidden),
            .out = out[j * hidden ..].ptr,
        };
        try (try f.workers(threads)).experts(list[0..ids.len], x, hidden, inter, limit);
    }
    /// Multi-row fast-mode routed experts grouped by expert ID across a prefill chunk (1..8 rows).
    pub fn projectExpertsBatch(f: *File, prefix: []const u8, ids: []const u32, count: usize, topk: usize, x: []const f32, out: []f32, hidden: usize, inter: usize, limit: f32, threads: usize) !void {
        if (!f.fast_mode or count == 0 or count > 8 or topk == 0 or topk > 16 or ids.len != count * topk or x.len != count * hidden or out.len != count * topk * hidden or hidden > 2048 or inter > 1024)
            return error.BadLingGgufMatrix;
        const hidden_blocks = hidden / 256;
        const can_sdot_in = hidden % 256 == 0 and count * hidden_blocks <= 64;
        const can_sdot_down = inter % 256 == 0 and count * (inter / 256) <= 64;
        var groups: [128]@import("cpu_gguf_pool.zig").FastExpertGroup = undefined;
        var group_ids: [128]u32 = undefined;
        var n_groups: usize = 0;
        var any_sdot = false;
        var buf: [128]u8 = undefined;
        for (0..count) |t| {
            for (0..topk) |j| {
                const eid = ids[t * topk + j];
                const slot_out = out[(t * topk + j) * hidden ..].ptr;
                var found: ?usize = null;
                for (0..n_groups) |gidx| if (group_ids[gidx] == eid and groups[gidx].n_rows < 8) {
                    found = gidx;
                    break;
                };
                if (found) |gidx| {
                    const idx = groups[gidx].n_rows;
                    groups[gidx].rows[idx] = @intCast(t);
                    groups[gidx].outs[idx] = slot_out;
                    groups[gidx].n_rows += 1;
                } else {
                    const gm = try f.fastMatrix(try std.fmt.bufPrint(&buf, "{s}experts.{d}.gate_proj.weight", .{ prefix, eid }), hidden, inter);
                    const um = try f.fastMatrix(try std.fmt.bufPrint(&buf, "{s}experts.{d}.up_proj.weight", .{ prefix, eid }), hidden, inter);
                    const dm = try f.fastMatrix(try std.fmt.bufPrint(&buf, "{s}experts.{d}.down_proj.weight", .{ prefix, eid }), inter, hidden);
                    const use_g = can_sdot_in and (gm.tensor.kind == .q4_k or gm.tensor.kind == .q6_k);
                    const use_u = can_sdot_in and (um.tensor.kind == .q4_k or um.tensor.kind == .q6_k);
                    const use_d = can_sdot_down and (dm.tensor.kind == .q4_k or dm.tensor.kind == .q6_k);
                    if (use_g or use_u) any_sdot = true;
                    group_ids[n_groups] = eid;
                    groups[n_groups] = .{
                        .expert = .{
                            .gate = gm.tensor,
                            .up = um.tensor,
                            .down = dm.tensor,
                            .gate_rep = if (use_g) gm.rep else null,
                            .up_rep = if (use_u) um.rep else null,
                            .down_rep = if (use_d) dm.rep else null,
                            .use_sdot_gate = use_g,
                            .use_sdot_up = use_u,
                            .use_sdot_down = use_d,
                            .out = slot_out,
                        },
                        .n_rows = 1,
                        .rows = .{ @intCast(t), 0, 0, 0, 0, 0, 0, 0 },
                        .outs = .{ slot_out, undefined, undefined, undefined, undefined, undefined, undefined, undefined },
                    };
                    n_groups += 1;
                }
            }
        }
        var acts_buf: [64]gguf_sdot.Activation256 = undefined;
        if (any_sdot) {
            for (0..count) |t| try gguf_sdot.quantize256(x[t * hidden ..][0..hidden], acts_buf[t * hidden_blocks ..][0..hidden_blocks], true);
        }
        try (try f.workers(threads)).expertsFastBatch(groups[0..n_groups], x, &acts_buf, count, hidden, inter, limit);
    }
    pub fn embed(f: *const File, ids: []const u32, output: []f32) !void {
        try @import("cpu_gguf_values.zig").embedding(&f.source.document, ids, output);
    }
    pub fn close(f: *File, io: std.Io) void {
        if (f.pool) |p| p.destroy();
        if (f.repack_cache) |*c| c.close(io);
        for (f.joined_mla) |entry| if (entry) |data| f.arena.child_allocator.free(data);
        f.arena.deinit();
        f.source.close(io);
        f.* = undefined;
    }
};

test "Ling GGUF source views refuse wrong dimensions and quantized small tensors" {
    const n = names.Name{ .source = "matrix", .native = "matrix", .role = .matrix, .rank = 2, .dims = .{ 1536, 512, 1 } };
    var t = g.Tensor{ .name = "matrix", .kind = .q4_k, .rank = 2, .shape = .{ 1536, 512, 1, 1 }, .offset = 0, .len = 0 };
    try checkTensor(t, n);
    t.shape[1] = 511;
    try std.testing.expectError(error.BadLingGgufTensorShape, checkTensor(t, n));
    const gain = names.Name{ .source = "gain", .native = "gain", .role = .vector, .rank = 1, .dims = .{ 1536, 1, 1 } };
    t = .{ .name = "gain", .kind = .q4_k, .rank = 1, .shape = .{ 1536, 1, 1, 1 }, .offset = 0, .len = 0 };
    try std.testing.expectError(error.UnsupportedLingGgufSmallType, checkTensor(t, gain));
    t.kind = .f32;
    try checkTensor(t, gain);
}
