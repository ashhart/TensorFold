//! A bounded training tape copies native outputs before scratch reuse and refuses every missing consumed row.
const std = @import("std");
const cfg = @import("config.zig");
const trace = @import("node_trace.zig");
pub fn Tape(comptime B: type) type {
    const Ref = @import("train_tensor.zig").Ref(B);
    return struct {
        const Self = @This();
        pub const Level = enum { keys, loss, back };
        const Entry = struct { ref: Ref, width: usize, kind: trace.Kind, seen: []bool, key: bool, head: bool };
        a: std.mem.Allocator,
        b: *B,
        rows: usize,
        entries: std.StringArrayHashMapUnmanaged(Entry) = .empty,
        owned: usize = 0,
        bytes: u64 = 0,
        first: usize = 0,
        level: Level = .keys,
        pub fn describe(a: std.mem.Allocator, b: *B, c: *const cfg.Config, rows: usize) !Self {
            var t: Self = .{ .a = a, .b = b, .rows = rows };
            errdefer t.deinitDrained();
            try t.add("embed", c.wide(), .bf16, true, false);
            try t.add("head", c.vocab, .bf16, false, true);
            try t.hc(null, "model.hyper_connection_mixer", c, false);
            for (0..c.layers) |l| {
                try t.hc(l, "attn_hyper_connection", c, true);
                try t.hc(l, "mlp_hyper_connection", c, true);
                try t.layer(l, "mlp.gate", c.intermediate, .bf16);
                try t.layer(l, "mlp.up", c.intermediate, .bf16);
                try t.layer(l, "mlp.act", c.intermediate, .bf16);
                try t.layer(l, "mlp.pocket_output", c.hidden, .bf16);
                switch (c.kinds[l]) {
                    .linear_attention => {
                        try t.layer(l, "linear_attn.output", c.hidden, .bf16);
                        try t.layer(l, "linear_attn.gated", c.valueWidth(), .bf16);
                        try t.layer(l, "linear_attn.projections", c.convWidth() + c.valueWidth() + 2 * c.value_heads, .bf16);
                        try t.layer(l, "linear_attn.recurrent", c.value_heads * 128 * 128, .f32);
                    },
                    .indexed_attention => {
                        var problem: cfg.Problem = .{};
                        const g = try @import("cuda_attention.zig").Geometry.of(c, &problem);
                        try t.layer(l, "self_attn.output", c.hidden, .bf16);
                        try t.layer(l, "self_attn.gated", c.heads * c.head_dim, .bf16);
                        try t.layer(l, "self_attn.projections", g.projectionWidth(), .bf16);
                        try t.layer(l, "self_attn.query", c.heads * c.head_dim, .bf16);
                        try t.layer(l, "self_attn.key", c.kv_heads * c.head_dim, .bf16);
                        try t.layer(l, "self_attn.value", c.kv_heads * c.head_dim, .bf16);
                        try t.layer(l, "self_attn.attention", c.heads * c.head_dim, .bf16);
                        try t.layer(l, "self_attn.selected_ids", try g.ids(), .i32);
                        try t.layer(l, "self_attn.selection_count", 1, .i32);
                        try t.layer(l, "self_attn.selection_sparse", 1, .i32);
                    },
                }
            }
            return t;
        }
        pub fn allocate(t: *Self, budget: u64) !void {
            if (t.owned != 0 or t.bytes > budget) return error.Qwen4LearningBudget;
            for (t.entries.values()) |*entry| {
                entry.ref.buffer = try t.b.allocate(entry.ref.len);
                t.owned += 1;
            }
        }
        pub fn deinitDrained(t: *Self) void {
            for (t.entries.values()[0..t.owned]) |entry| t.b.release(entry.ref.buffer);
            for (t.entries.keys(), t.entries.values()) |name, entry| {
                t.a.free(name);
                t.a.free(entry.seen);
            }
            t.entries.deinit(t.a);
            t.* = undefined;
        }
        fn add(t: *Self, name: []const u8, width: usize, kind: trace.Kind, key: bool, head: bool) !void {
            const len = try std.math.mul(usize, try std.math.mul(usize, t.rows, width), if (kind == .bf16) 2 else 4);
            const charge = try std.math.add(u64, try B.allocationBytes(len), t.rows);
            t.bytes = try std.math.add(u64, t.bytes, charge);
            const label = try t.a.dupe(u8, name);
            errdefer t.a.free(label);
            const seen = try t.a.alloc(bool, t.rows);
            errdefer t.a.free(seen);
            @memset(seen, false);
            try t.entries.put(t.a, label, .{ .ref = .{ .buffer = undefined, .len = len }, .width = width, .kind = kind, .seen = seen, .key = key, .head = head });
        }
        fn layer(t: *Self, l: usize, suffix: []const u8, width: usize, kind: trace.Kind) !void {
            var buffer: [192]u8 = undefined;
            try t.add(try std.fmt.bufPrint(&buffer, "model.layers.{d}.{s}", .{ l, suffix }), width, kind, false, false);
        }
        fn hc(t: *Self, l: ?usize, scope: []const u8, c: *const cfg.Config, inject: bool) !void {
            const specs = [_]struct { []const u8, usize }{ .{ "norm", c.wide() }, .{ "down", c.lowrank }, .{ "up", c.wide() }, .{ "mix", c.hidden } };
            for (specs) |spec| {
                var buffer: [192]u8 = undefined;
                const name = try trace.label(&buffer, l, scope, spec[0]);
                try t.add(name, spec[1], .bf16, l != null and std.mem.eql(u8, scope, "mlp_hyper_connection") and std.mem.eql(u8, spec[0], "mix"), false);
            }
            if (inject) {
                var buffer: [192]u8 = undefined;
                try t.add(try trace.label(&buffer, l, scope, "raw_inject"), c.streams, .bf16, true, false);
                try t.add(try trace.label(&buffer, l, scope, "injection"), c.streams, .bf16, true, false);
                try t.add(try trace.label(&buffer, l, scope, "residual"), c.wide(), .bf16, true, false);
            }
        }
        pub fn reset(t: *Self, level: Level) void {
            t.first = 0;
            t.level = level;
            for (t.entries.values()) |entry| @memset(entry.seen, false);
        }
        pub fn sink(t: *Self) trace.Sink(B) {
            return .{ .ctx = t, .emit = emit, .metadata = metadata, .selection = t.level == .back };
        }
        fn wanted(t: *Self, e: Entry) bool {
            return e.key or (e.head and t.level != .keys) or t.level == .back;
        }
        fn accept(t: *Self, name: []const u8, rows: usize, width: usize, kind: trace.Kind, first: usize) !?struct { *Entry, usize } {
            const e = t.entries.getPtr(name) orelse return null;
            if (!t.wanted(e.*)) return null;
            const at = try std.math.add(usize, t.first, first);
            if (rows == 0 or at > t.rows or rows > t.rows - at or width != e.width or kind != e.kind) return error.Qwen4TrainingTapeShape;
            for (e.seen[at..][0..rows]) |seen| if (seen) return error.Qwen4TrainingTapeDuplicate;
            @memset(e.seen[at..][0..rows], true);
            return .{ e, at };
        }
        fn emit(ctx: *anyopaque, name: []const u8, buffer: B.Buffer, offset: usize, rows: usize, width: usize, kind: trace.Kind, first: usize) !void {
            const t: *Self = @ptrCast(@alignCast(ctx));
            const dest = try t.accept(name, rows, width, kind, first) orelse return;
            const stride = width * @as(usize, if (kind == .bf16) 2 else 4);
            try t.b.copy(dest[0].ref.buffer, dest[1] * stride, buffer, offset, rows * stride);
        }
        fn metadata(ctx: *anyopaque, name: []const u8, values: []const i32, first: usize) !void {
            const t: *Self = @ptrCast(@alignCast(ctx));
            const dest = try t.accept(name, values.len, 1, .i32, first) orelse return;
            try t.b.write(dest[0].ref.buffer, dest[1] * 4, std.mem.sliceAsBytes(values));
        }
        pub fn get(t: *Self, name: []const u8, rows: usize) !Ref {
            const e = t.entries.getPtr(name) orelse return error.Qwen4TrainingTapeMissing;
            if (rows > t.rows) return error.Qwen4TrainingTapeShape;
            for (e.seen[0..rows]) |seen| if (!seen) return error.Qwen4TrainingTapeMissing;
            return e.ref.plus(0, rows * e.width * @as(usize, if (e.kind == .bf16) 2 else 4));
        }
        pub fn indexed(t: *Self, l: usize, suffix: []const u8, rows: usize) !Ref {
            var name: [192]u8 = undefined;
            return t.get(try std.fmt.bufPrint(&name, "model.layers.{d}.{s}", .{ l, suffix }), rows);
        }
        pub fn raw(t: *Self, l: usize, suffix: []const u8, rows: usize) !Ref {
            var name: [192]u8 = undefined;
            const entry = t.entries.getPtr(try std.fmt.bufPrint(&name, "model.layers.{d}.{s}", .{ l, suffix })) orelse return error.Qwen4TrainingTapeMissing;
            return entry.ref.plus(0, rows * entry.width * @as(usize, if (entry.kind == .bf16) 2 else 4));
        }
    };
}
