//! Ling tiny's 24-layer bf16 backbone over the existing Metal row, KDA, MLA and route operations.
const std = @import("std");
const mtl = @import("metal");
const ck = @import("ling_checkpoint");
const Rows = @import("ling_rows").Rows;
const Kda = @import("ling_kda").Kda;
const Mla = @import("ling_mla").Mla;
const Moe = @import("ling_moe").Moe;
const Attention = union(enum) { kda: Kda, mla: Mla };

pub const Forward = struct {
    rows: *Rows,
    model: *ck.Checkpoint,
    attention: [24]Attention,
    moe: Moe,
    x: mtl.Buffer,
    normed: mtl.Buffer,
    delta: mtl.Buffer,
    gate: mtl.Buffer,
    up: mtl.Buffer,
    activation: mtl.Buffer,
    logits: mtl.Buffer,
    picks: mtl.Buffer,
    position: u32 = 0,

    pub fn init(rows: *Rows, model: *ck.Checkpoint, capacity: u32) !Forward {
        const buffer_mark = rows.owned.items.len;
        errdefer rows.rollbackBuffers(buffer_mark);
        const c = model.config orelse return error.MissingConfig;
        if (c.quant != .bf16 or c.layers != 24 or c.hidden != 1536 or c.vocab != 157184 or c.group != 4 or
            c.kda_heads != 16 or c.kda_dim != 128 or c.mla_heads != 16 or c.q_lora != 256 or c.kv_lora != 512 or
            c.nope != 128 or c.rope_dim != 64 or c.v_dim != 128 or c.dense_layers != 1 or c.dense_inter != 4608 or
            c.experts != 128 or c.topk != 8 or c.n_group != 8 or c.topk_group != 4 or c.moe_inter != 512 or
            c.shared_inter != 512 or c.routed_scale != 2.5 or c.eps != 1e-6 or c.rope_theta != 6000000 or
            c.lower_bound != -5 or c.mtp != 0 or !c.head_gate or c.tie or c.rope_style != .gptj or !c.norm_topk or !c.expert_bias) return error.UnsupportedLingGeometry;
        for (0..24) |i| if (c.expert_limit[i] != 0 or c.shared_limit[i] != 0) return error.UnsupportedLingGeometry;
        if (capacity == 0 or rows.max_rows == 0) return error.ContextCapacity;
        var attention: [24]Attention = undefined;
        var initialized: usize = 0;
        errdefer for (attention[0..initialized]) |*a| switch (a.*) {
            .mla => |*m| m.deinit(),
            .kda => {},
        };
        for (&attention, 0..) |*a, i| {
            const prefix = try std.fmt.allocPrint(rows.allocator, "model.layers.{d}.attention.", .{i});
            defer rows.allocator.free(prefix);
            a.* = if (i % 4 == 3) .{ .mla = try Mla.init(rows, 16, capacity) } else .{ .kda = try Kda.init(rows, model, prefix, 16) };
            initialized += 1;
        }
        const r: usize = rows.max_rows;
        var moe = try Moe.init(rows, 1536, 512);
        errdefer moe.deinit();
        const x = try rows.buffer(r * 1536 * 2);
        const normed = try rows.buffer(r * 1536 * 2);
        const delta = try rows.buffer(r * 1536 * 2);
        const gate = try rows.buffer(r * 4608 * 2);
        const up = try rows.buffer(r * 4608 * 2);
        const activation = try rows.buffer(r * 4608 * 2);
        const logits = try rows.buffer(r * 157184 * 4);
        const picks = try rows.buffer(23 * r * 8 * 4);
        return .{
            .rows = rows,
            .model = model,
            .attention = attention,
            .moe = moe,
            .x = x,
            .normed = normed,
            .delta = delta,
            .gate = gate,
            .up = up,
            .activation = activation,
            .logits = logits,
            .picks = picks,
        };
    }

    pub fn run(self: *Forward, ids: []const u32) ![]const f32 {
        const r = self.rows;
        if (ids.len == 0 or ids.len > r.max_rows) return error.RowCapacity;
        const n: u32 = @intCast(ids.len);
        for (&self.attention) |*a| if (a.* == .mla and (self.position > a.mla.capacity or n > a.mla.capacity - self.position)) return error.ContextCapacity;

        const embedding = try self.model.get("model.word_embeddings.weight", .bf16, &.{ 157184, 1536 });
        for (ids, 0..) |id, i| {
            if (id >= 157184) return error.InvalidToken;
            const offset = embedding.offset + @as(usize, id) * 1536 * 2;
            @memcpy(self.x.contents()[i * 1536 * 2 ..][0 .. 1536 * 2], embedding.buffer.contents()[offset..][0 .. 1536 * 2]);
        }
        for (&self.attention, 0..) |*attention, i| {
            const pre = try std.fmt.allocPrint(r.allocator, "model.layers.{d}.", .{i});
            defer r.allocator.free(pre);
            const norm = try std.mem.concat(r.allocator, u8, &.{ pre, "input_layernorm.weight" });
            defer r.allocator.free(norm);
            try r.norm(self.model, norm, .{ .buf = self.x }, .{ .buf = self.normed }, n, 1536, 1e-6);
            const a = try std.mem.concat(r.allocator, u8, &.{ pre, "attention." });
            defer r.allocator.free(a);
            switch (attention.*) {
                .kda => |*k| try k.forward(r, self.model, a, .{ .buf = self.normed }, .{ .buf = self.delta }, n, 1536, -5, 1e-6),
                .mla => |*m| try m.forward(r, self.model, a, .{ .buf = self.normed }, .{ .buf = self.delta }, n, 1536, 1e-6, 6000000),
            }
            try r.add(.{ .buf = self.x }, .{ .buf = self.delta }, .{ .buf = self.x }, n * 1536);
            const post = try std.mem.concat(r.allocator, u8, &.{ pre, "post_attention_layernorm.weight" });
            defer r.allocator.free(post);
            try r.norm(self.model, post, .{ .buf = self.x }, .{ .buf = self.normed }, n, 1536, 1e-6);
            const mlp = try std.mem.concat(r.allocator, u8, &.{ pre, "mlp." });
            defer r.allocator.free(mlp);
            if (i == 0) {
                try r.mlp(self.model, mlp, .{ .buf = self.normed }, .{ .buf = self.delta }, .{ .buf = self.gate }, .{ .buf = self.up }, .{ .buf = self.activation }, n, 1536, 4608);
            } else {
                try self.moe.forward(r, self.model, mlp, .{ .buf = self.normed }, .{ .buf = self.delta }, n);
                @memcpy(self.picks.contents()[(i - 1) * n * 8 * 4 ..][0 .. n * 8 * 4], self.moe.ids.contents()[0 .. n * 8 * 4]);
            }
            try r.add(.{ .buf = self.x }, .{ .buf = self.delta }, .{ .buf = self.x }, n * 1536);
        }
        try r.norm(self.model, "model.norm.weight", .{ .buf = self.x }, .{ .buf = self.normed }, n, 1536, 1e-6);
        try r.linear(self.model, "lm_head.weight", .{ .buf = self.normed }, .{ .buf = self.logits }, n, 1536, 157184, true);
        self.position += n;
        return self.logits.slice(f32, n * 157184);
    }

    fn stateBytes(position: u32) usize {
        return 12 + 18 * (16 * 128 * 128 * 4 + 9 * 2048 * 2) + 6 * @as(usize, position) * 576 * 2;
    }

    pub fn snapshot(self: *const Forward, allocator: std.mem.Allocator) ![]u8 {
        const data = try allocator.alloc(u8, stateBytes(self.position));
        @memcpy(data[0..8], "TFLINGS1");
        std.mem.writeInt(u32, data[8..12], self.position, .little);
        var at: usize = 12;
        for (&self.attention) |*attention| switch (attention.*) {
            .kda => |*k| {
                const state = k.state.contents()[0 .. 16 * 128 * 128 * 4];
                const conv = k.conv.contents()[0 .. 9 * 2048 * 2];
                @memcpy(data[at..][0..state.len], state);
                at += state.len;
                @memcpy(data[at..][0..conv.len], conv);
                at += conv.len;
            },
            .mla => |*m| {
                const cache = m.cache.contents()[0 .. @as(usize, self.position) * 576 * 2];
                @memcpy(data[at..][0..cache.len], cache);
                at += cache.len;
            },
        };
        return data;
    }

    pub fn restore(self: *Forward, data: []const u8) !void {
        if (data.len < 12 or !std.mem.eql(u8, data[0..8], "TFLINGS1")) return error.BadLingState;
        const position = std.mem.readInt(u32, data[8..12], .little);
        if (data.len != stateBytes(position)) return error.BadLingState;
        for (&self.attention) |*a| if (a.* == .mla and position > a.mla.capacity) return error.ContextCapacity;
        var at: usize = 12;
        for (&self.attention) |*attention| switch (attention.*) {
            .kda => |*k| {
                const state = k.state.contents()[0 .. 16 * 128 * 128 * 4];
                const conv = k.conv.contents()[0 .. 9 * 2048 * 2];
                @memcpy(state, data[at..][0..state.len]);
                at += state.len;
                @memcpy(conv, data[at..][0..conv.len]);
                at += conv.len;
            },
            .mla => |*m| {
                const cache = m.cache.contents()[0 .. @as(usize, position) * 576 * 2];
                @memcpy(cache, data[at..][0..cache.len]);
                at += cache.len;
                m.position = position;
            },
        };
        self.position = position;
    }

    pub fn deinit(self: *Forward) void {
        for (&self.attention) |*a| switch (a.*) {
            .mla => |*m| m.deinit(),
            .kda => {},
        };
        self.moe.deinit();
    }
};
