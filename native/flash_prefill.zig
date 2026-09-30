const std = @import("std");
const mx = @import("mlx.zig");
const flash = @import("flash.zig");
const hc = @import("flash_prefill_ops.zig");
const gdn = @import("flash_prefill_gdn.zig");
const attn = @import("flash_prefill_attention.zig");
const moe = @import("flash_prefill_moe.zig");
const ple = @import("flash_prefill_ple.zig");
const A = mx.Array;

fn weight(m: *flash.Model, base: []const u8, name: []const u8) !@import("flash_ops.zig").Weight {
    var key: [256]u8 = undefined;
    return m.weights.affine(try std.fmt.bufPrint(&key, "{s}.{s}", .{ base, name }));
}
fn hyper(m: *flash.Model, s: *mx.Scope, base: []const u8, h: A, pending: ?hc.Pending, inject: bool) !hc.Hyper {
    const w = try m.hcWeights(s, base, inject);
    return hc.hyper(&m.kernels, s, h, pending, w.down, w.up, w.scale, try s.scalar(1e-6), 4, 320);
}
fn recurrent(m: *flash.Model, s: *mx.Scope, base: []const u8, x: A, previous: flash.Cache, record: *flash.Cache) !A {
    var w = gdn.Weights{ .qkv = try weight(m, base, "in_proj_qkv"), .z = try weight(m, base, "in_proj_z"), .b = try weight(m, base, "in_proj_b"), .a = try weight(m, base, "in_proj_a"), .out = try weight(m, base, "out_proj"), .conv = try s.reshape(try m.f(base, "conv1d.weight"), &.{ 10240, 4, 1 }), .a_log = try m.f(base, "A_log"), .dt_bias = try m.f(base, "dt_bias"), .norm = try m.f(base, "norm.weight") };
    var buf: [256]u8 = undefined;
    const key = try std.fmt.bufPrint(&buf, "{s}.native_prefill_stack", .{base});
    if (m.weights.has(key)) {
        w.stacked = try m.weights.affine(key);
    } else if (try gdn.stack(s, w)) |stacked| {
        try mx.evalMany(&stacked.arrays, false);
        try m.weights.putAffine(key, stacked);
        try m.weights.put(key, stacked.arrays[0]);
        w.stacked = stacked;
    }
    const out = try gdn.forward(&m.kernels, &m.prefill_ops, s, try s.reshape(x, &.{ 1, mx.dim(x, 0), 2560 }), w, .{ .key_heads = 16, .value_heads = 48, .key_dims = 128, .value_dims = 128 }, .{ .conv = if (previous.a.ctx != null) try s.reshape(previous.a, &.{ 1, 3, 10240 }) else mx.empty, .state = if (previous.b.ctx != null) try s.reshape(previous.b, &.{ 1, 48, 128, 128 }) else mx.empty });
    record.a = try s.reshape(out.cache.conv, &.{ 3, 10240 });
    record.b = try s.reshape(out.cache.state, &.{ 48, 128, 128 });
    return s.reshape(out.output, &.{ mx.dim(x, 0), 2560 });
}
fn attention(m: *flash.Model, s: *mx.Scope, base: []const u8, x: A, previous: *flash.Cache, record: *flash.Cache) !A {
    var buf: [256]u8 = undefined;
    var w: attn.Weights = undefined;
    inline for (.{ .{ "q", "q_proj" }, .{ "k", "k_proj" }, .{ "v", "v_proj" }, .{ "out", "o_proj" }, .{ "index", "indexer.index_qk_proj" } }) |entry| @field(w, entry[0]) = try weight(m, base, entry[1]);
    inline for (.{ .{ "q_scale", "q_norm" }, .{ "k_scale", "k_norm" }, .{ "iq_scale", "indexer.q_layernorm" }, .{ "ik_scale", "indexer.k_layernorm" } }) |entry| @field(w, entry[0]) = try m.scale(s, try std.fmt.bufPrint(&buf, "{s}.{s}", .{ base, entry[1] }), "weight");
    const out = try attn.forward(&m.kernels, &m.prefill_ops, s, try s.reshape(x, &.{ 1, mx.dim(x, 0), 2560 }), w, .{ .heads = 24, .kv_heads = 2, .dims = 256, .rotary_dims = 64, .index_heads = 4, .index_dims = 128, .top = 512 }, .{ .keys = previous.a, .values = previous.b, .raw = if (previous.raw.ctx != null) try s.reshape(previous.raw, &.{ 1, previous.offset, 128 }) else mx.empty, .pooled = if (previous.pooled.ctx != null) try s.reshape(previous.pooled, &.{ 1, mx.dim(previous.pooled, 0), 128 }) else mx.empty, .offset = previous.offset });
    record.a = out.cache.keys;
    record.b = out.cache.values;
    record.raw = try s.reshape(out.cache.raw, &.{ out.cache.offset, 128 });
    record.pooled = if (out.cache.pooled.ctx != null) try s.reshape(out.cache.pooled, &.{ mx.dim(out.cache.pooled, 1), 128 }) else mx.empty;
    record.offset = out.cache.offset;
    if (@import("kv_buffer.zig").enabled) {
        record.key_write = try previous.keys.append(s, previous.a, try s.slice(record.a, 2, previous.offset, record.offset), 2);
        record.value_write = try previous.values.append(s, previous.b, try s.slice(record.b, 2, previous.offset, record.offset), 2);
        record.index_write = try previous.index_keys.append(s, previous.raw, try s.slice(record.raw, 0, previous.offset, record.offset), 0);
        record.a = record.key_write.view;
        record.b = record.value_write.view;
        record.raw = record.index_write.view;
    }
    return s.reshape(out.output, &.{ mx.dim(x, 0), 2560 });
}
fn experts(m: *flash.Model, s: *mx.Scope, base: []const u8, x: A) !A {
    var w: moe.Weights = undefined;
    w.router = try m.f(base, "gate.weight");
    inline for (.{ .{ "gate", "switch_mlp.gate_proj" }, .{ "up", "switch_mlp.up_proj" }, .{ "down", "switch_mlp.down_proj" }, .{ "shared_gate", "shared_expert.gate_proj" }, .{ "shared_up", "shared_expert.up_proj" }, .{ "shared_down", "shared_expert.down_proj" }, .{ "shared_route", "shared_expert_gate" } }) |entry| @field(w, entry[0]) = try weight(m, base, entry[1]);
    const out = try moe.forward(&m.kernels, &m.prefill_ops, s, try s.reshape(x, &.{ 1, mx.dim(x, 0), 2560 }), w, 10);
    return s.reshape(out.output, &.{ mx.dim(x, 0), 2560 });
}
fn embedding(m: *flash.Model, s: *mx.Scope, h: A, tokens: []const i32, previous: flash.Cache, record: *flash.Cache) !A {
    const rows: i32 = @intCast(tokens.len);
    var hist = previous.history;
    var emb: A = undefined;
    if (m.gpuTokensEnabled()) {
        const before = if (previous.token_history.ctx != null) previous.token_history else try s.ints(&.{ 248044, 248044 });
        record.token_history = try s.cat(&.{ before, try s.ints(tokens) }, 0);
        emb = try m.ple_tables.?.resident.?.gather(&m.kernels, s, try m.ngram.idsArray(s, record.token_history));
    } else {
        const ids = try mx.allocator.alloc(i64, tokens.len * 16);
        defer mx.allocator.free(ids);
        for (tokens, 0..) |token, row| {
            @memcpy(ids[row * 16 ..][0..16], &m.ngram.ids(hist, token));
            hist = .{ hist[1], token };
        }
        emb = try m.ple_tables.?.gather(s, ids);
    }
    const base = "model.layers.1.ple";
    const w = ple.Weights{ .key = try weight(m, base, "key_proj"), .value = try weight(m, base, "value_proj"), .key_scale = try m.scale(s, base ++ ".norm_key", "weight"), .query_scale = try m.scale(s, base ++ ".norm_query", "weight"), .conv_scale = try m.scale(s, base ++ ".norm_conv", "weight"), .conv = try s.reshape(try m.f(base, "conv1d.weight"), &.{ 10240, 4, 1 }) };
    const out = try ple.forward(&m.prefill_ops, s, try s.reshape(h, &.{ 1, rows, 10240 }), try s.reshape(emb, &.{ 1, rows, 2560 }), w, if (previous.ple.ctx != null) try s.reshape(previous.ple, &.{ 1, 9, 10240 }) else mx.empty, 4, 3);
    record.ple = try s.reshape(out.tail, &.{ 9, 10240 });
    record.history = hist;
    return s.binary(mx.c.mlx_add, h, try s.reshape(out.branch, &.{ rows, 10240 }));
}

pub fn forward(m: *flash.Model, tokens: []const i32) !flash.Pass {
    if (tokens.len < 17 or tokens.len > 2048 or tokens.len > 262144 - m.position) return error.ContextLimitExceeded;
    for (tokens) |token| if (token < 0 or token >= flash.Model.vocab) return error.InvalidToken;
    var pass = flash.Pass{ .prefilled = true, .count = tokens.len, .start = m.position };
    errdefer pass.deinit();
    @memcpy(pass.tokens[0..tokens.len], tokens);
    var carry = mx.Scope{};
    defer carry.deinit();
    const e = try m.weights.embed(&carry, "model.embed_tokens", tokens);
    var h = try carry.cat(&.{ e, e, e, e }, -1);
    var pending: ?hc.Pending = null;
    var buf: [256]u8 = undefined;
    for (0..48) |i| {
        var scratch = mx.Scope{};
        defer scratch.deinit();
        const s = &scratch;
        if (i == 1) {
            if (pending != null) h = (try hc.writeBack(&m.kernels, s, h, pending, 4))[0];
            pending = null;
            h = try embedding(m, s, h, tokens, m.cache[i], &pass.records[i]);
        }
        const ah = try hyper(m, s, try std.fmt.bufPrint(&buf, "model.layers.{d}.attn_hyper_connection", .{i}), h, pending, true);
        const branch = if (i % 4 != 3) try recurrent(m, s, try std.fmt.bufPrint(&buf, "model.layers.{d}.linear_attn", .{i}), ah.mixed, m.cache[i], &pass.records[i]) else try attention(m, s, try std.fmt.bufPrint(&buf, "model.layers.{d}.self_attn", .{i}), ah.mixed, &m.cache[i], &pass.records[i]);
        const mh = try hyper(m, s, try std.fmt.bufPrint(&buf, "model.layers.{d}.mlp_hyper_connection", .{i}), ah.residual, .{ .branch = branch, .inject = ah.inject.? }, true);
        h = mh.residual;
        pending = .{ .branch = try experts(m, s, try std.fmt.bufPrint(&buf, "model.layers.{d}.mlp", .{i}), mh.mixed), .inject = mh.inject.? };
        try mx.evalMany(&.{ h, pending.?.branch, pending.?.inject, pass.records[i].a, pass.records[i].b }, false);
        const record = &pass.records[i];
        inline for (.{ "a", "b", "raw", "pooled", "ple", "token_history" }) |field| {
            const value = @field(record, field);
            if (value.ctx != null) {
                try mx.eval(value);
                @field(record, field) = try pass.scope.own(try mx.retain(value));
            }
        }
        inline for (.{ "key_write", "value_write", "index_write" }) |field| {
            const write = &@field(record, field);
            inline for (.{ "capacity", "added", "view" }) |part| {
                const value = @field(write, part);
                if (value.ctx != null) @field(write, part) = try pass.scope.own(try mx.retain(value));
            }
        }
        carry.deinit();
        carry = .{};
        h = try carry.own(try mx.retain(h));
        pending = .{ .branch = try carry.own(try mx.retain(pending.?.branch)), .inject = try carry.own(try mx.retain(pending.?.inject)) };
    }
    const s = &pass.scope;
    const mixed = try hyper(m, s, "model.hyper_connection_mixer", h, pending, false);
    pass.hidden = mixed.residual;
    pass.logits = try m.weights.linear(&m.kernels, s, "lm_head", try s.slice(mixed.mixed, 0, @intCast(tokens.len - 1), @intCast(tokens.len)), true);
    try mx.eval(pass.logits);
    try flash.Model.observeBuffers(&pass);
    return pass;
}

pub fn check(io: std.Io, dir: []const u8, output: []const u8, custom_tiles: bool) !void {
    try mx.init();
    defer mx.shutdown();
    var m = try flash.Model.init(io, dir, false);
    defer m.deinit();
    if (custom_tiles) m.kernels.flash_prefill.decision = true;
    try std.Io.Dir.cwd().createDirPath(io, output);
    var buf: [256]u8 = undefined;
    for ([_]usize{ 17, 63, 64, 2048, 2048, 17, 1 }, 0..) |count, step| {
        var tokens: [2048]i32 = undefined;
        for (tokens[0..count], 0..) |*id, j| id.* = 1000 + @mod(m.position + @as(i32, @intCast(j)), 37);
        var pass = try m.prefill(tokens[0..count]);
        defer pass.deinit();
        try save(&pass.scope, output, try std.fmt.bufPrint(&buf, "hidden-{d}", .{step}), pass.hidden);
        try save(&pass.scope, output, try std.fmt.bufPrint(&buf, "logits-{d}", .{step}), try pass.scope.slice(pass.logits, 0, mx.dim(pass.logits, 0) - 1, mx.dim(pass.logits, 0)));
        if (pass.prefilled) try std.testing.expectError(error.InvalidCommit, m.commit(&pass, count - 1));
        try m.commit(&pass, count);
        for (m.cache, 0..) |cache, i| {
            try save(&pass.scope, output, try std.fmt.bufPrint(&buf, "cache-{d}-{d}-0", .{ step, i }), cache.a);
            try save(&pass.scope, output, try std.fmt.bufPrint(&buf, "cache-{d}-{d}-1", .{ step, i }), cache.b);
            if (i % 4 == 3) {
                try save(&pass.scope, output, try std.fmt.bufPrint(&buf, "raw-{d}-{d}", .{ step, i }), cache.raw);
                if (cache.pooled.ctx != null) try save(&pass.scope, output, try std.fmt.bufPrint(&buf, "pooled-{d}-{d}", .{ step, i }), cache.pooled);
            }
            if (i == 1) {
                try save(&pass.scope, output, try std.fmt.bufPrint(&buf, "ple-{d}", .{step}), cache.ple);
                try save(&pass.scope, output, try std.fmt.bufPrint(&buf, "history-{d}", .{step}), try pass.scope.ints(&cache.history));
            }
        }
        std.debug.print("Flash prefill cache commit at {d} tokens.\n", .{m.position});
    }
    for (0..4) |step| {
        var pass = try m.forward(&.{@as(i32, @intCast(step)) + 2000});
        defer pass.deinit();
        try save(&pass.scope, output, try std.fmt.bufPrint(&buf, "continuation-{d}", .{step}), pass.logits);
        try m.commit(&pass, 1);
    }
}
fn save(s: *mx.Scope, dir: []const u8, name: []const u8, value: A) !void {
    const path = try std.fmt.allocPrintSentinel(mx.allocator, "{s}/{s}.npy", .{ dir, name }, 0);
    defer mx.allocator.free(path);
    const out = try s.cast(value, mx.f32t);
    try mx.eval(out);
    try mx.check(mx.c.mlx_save(path, out));
}
