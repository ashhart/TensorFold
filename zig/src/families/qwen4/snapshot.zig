//! Prompt states pack only the kept prefix and complete QSA blocks; all copies stay on the owning inference queue.
const std = @import("std");
const cfg = @import("config.zig");
const state_ = @import("state.zig");
pub fn Snapshot(comptime B: type) type {
    return struct {
        const Self = @This();
        const State = state_.State(B);
        a: std.mem.Allocator,
        backend: *B,
        c: *const cfg.Config,
        key: [32]u8,
        buffer: B.Buffer,
        tokens: []u32,
        at: usize,
        bytes: usize,
        payload_bytes: usize,
        charged: u64,
        ready: bool = false,
        last_valid: bool = false,
        retained_next: ?*Self = null,
        pub fn deviceBytes(s: *const State) !usize {
            return std.math.add(usize, (try s.layout.savedBytes(s.c, s.pos)) - s.layout.history_bytes, try std.math.mul(usize, s.c.vocab, 2));
        }
        pub fn charge(s: *const State) !u64 {
            return std.math.add(u64, try B.allocationBytes(try deviceBytes(s)), try std.math.add(u64, @sizeOf(Self), try std.math.mul(u64, s.pos, 4)));
        }
        /// Open queues no work. A failed capture leaves this owner available for close or a successful drain retry.
        pub fn open(a: std.mem.Allocator, s: *const State, key: [32]u8, budget: u64) !Self {
            if (s.phase != .ready) return error.Qwen4SnapshotPhase;
            const charged = try charge(s);
            if (charged > budget) return error.Qwen4SnapshotBudget;
            const tokens = try a.dupe(u32, s.history());
            errdefer a.free(tokens);
            const n = try deviceBytes(s);
            return .{ .a = a, .backend = s.backend, .c = s.c, .key = key, .buffer = try s.backend.allocate(n), .tokens = tokens, .at = s.pos, .bytes = n, .payload_bytes = n - s.c.vocab * 2, .charged = charged };
        }
        pub fn close(s: *Self) !void {
            try s.backend.drain();
            s.deinitDrained();
        }
        pub fn deinitDrained(s: *Self) void {
            s.backend.release(s.buffer);
            s.a.free(s.tokens);
            s.* = undefined;
        }
        fn valid(s: *const Self, state: *const State, key: [32]u8) !void {
            if (s.backend != state.backend or s.c != state.c or !std.mem.eql(u8, &s.key, &key)) return error.Qwen4SnapshotIdentity;
            if (s.at > state.layout.capacity) return error.Qwen4SnapshotCapacity;
        }
        fn piece(s: *Self, state: *State, offset: usize, n: usize, cursor: *usize, save: bool) !void {
            if (n == 0) return;
            if (cursor.* > s.bytes or n > s.bytes - cursor.*) return error.Qwen4SnapshotLayout;
            if (save) try s.backend.copy(s.buffer, cursor.*, state.buffer, offset, n) else try s.backend.copy(state.buffer, offset, s.buffer, cursor.*, n);
            cursor.* += n;
        }
        fn move(s: *Self, state: *State, save: bool) !void {
            var cursor: usize = 0;
            try s.piece(state, state.layout.kv_bytes, state.layout.state_bytes, &cursor, save);
            for (state.layout.layers[0..state.layout.count]) |layer| {
                if (layer.key.bytes == 0) continue;
                const kv = s.at * s.c.kv_heads * s.c.head_dim * 2;
                try s.piece(state, layer.key.offset, kv, &cursor, save);
                try s.piece(state, layer.value.offset, kv, &cursor, save);
                try s.piece(state, layer.index.offset, s.at * s.c.index_dim * 2, &cursor, save);
                try s.piece(state, layer.pooled.offset, s.at / s.c.index_ratio * s.c.index_dim * 2, &cursor, save);
            }
            if (cursor != s.payload_bytes) return error.Qwen4SnapshotLayout;
        }
        pub fn capture(s: *Self, state: *State, key: [32]u8) !void {
            try s.valid(state, key);
            if (state.phase != .ready or state.pos != s.at or !std.mem.eql(u32, s.tokens, state.history())) return error.Qwen4SnapshotPhase;
            s.ready = false;
            s.last_valid = false;
            try s.move(state, true);
            s.ready = true;
        }
        /// Keeping the final logits allows a fully cached prompt to draw its first token without replaying a row.
        pub fn captureWithLogits(s: *Self, state: *State, key: [32]u8, last: anytype) !void {
            if (last.kind != .bf16 or last.rows != 1 or last.width != s.c.vocab) return error.Qwen4SnapshotLogits;
            try s.capture(state, key);
            s.ready = false;
            try s.backend.copy(s.buffer, s.payload_bytes, last.buffer, last.offset, s.c.vocab * 2);
            s.last_valid = true;
            s.ready = true;
        }
        pub fn admitRestore(s: *const Self, state: *const State, key: [32]u8) !void {
            try s.valid(state, key);
            if (!s.ready or (state.phase != .ready and state.phase != .empty)) return error.Qwen4SnapshotPhase;
        }
        pub fn restore(s: *Self, state: *State, key: [32]u8) !void {
            try s.admitRestore(state, key);
            state.poison();
            try s.move(state, false);
            @memcpy(state.tokens[0..s.at], s.tokens);
            state.pos = s.at;
            state.base_pos = s.at;
            state.rows = 0;
            state.base_valid = false;
            state.phase = .ready;
        }
        pub fn restoreWithLogits(s: *Self, state: *State, key: [32]u8, last: anytype) !void {
            try s.admitRestore(state, key);
            if (!s.last_valid or last.kind != .bf16 or last.rows != 1 or last.width != s.c.vocab) return error.Qwen4SnapshotLogits;
            errdefer state.poison();
            try s.restore(state, key);
            try s.backend.copy(last.buffer, last.offset, s.buffer, s.payload_bytes, s.c.vocab * 2);
        }
    };
}
