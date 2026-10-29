//! One stream owns its KV prefix and recurrent frames; queue-ordered copies commit exactly the accepted row.
const std = @import("std");
const cfg = @import("config.zig");
const layout_ = @import("state_layout.zig");
pub fn State(comptime B: type) type {
    return struct {
        const Self = @This();
        pub const View = struct { buffer: B.Buffer, offset: usize, len: usize };
        const Phase = enum { empty, ready, working, verified, poisoned };
        a: std.mem.Allocator,
        backend: *B,
        c: *const cfg.Config,
        layout: layout_.Layout,
        buffer: B.Buffer,
        tokens: []u32,
        window: usize,
        bytes: usize,
        pos: usize = 0,
        base_pos: usize = 0,
        rows: usize = 0,
        phase: Phase = .empty,
        base_valid: bool = false,
        recorded: [cfg.max_layers]bool = @splat(false),
        /// Allocation alone queues no writes; reset initializes the frame after ownership has transferred.
        pub fn open(a: std.mem.Allocator, backend: *B, c: *const cfg.Config, capacity: usize, window: usize, budget: u64) !Self {
            const layout = try layout_.Layout.of(c, capacity);
            const n = try layout.deviceBytes(window);
            if (try B.allocationBytes(n) > budget) return error.Qwen4StateBudget;
            const tokens = try a.alloc(u32, capacity);
            errdefer a.free(tokens);
            return .{ .a = a, .backend = backend, .c = c, .layout = layout, .buffer = try backend.allocate(n), .tokens = tokens, .window = window, .bytes = n };
        }
        pub fn close(s: *Self) !void {
            try s.backend.drain();
            s.deinitDrained();
        }
        /// The engine may drain all its slots once; a failed drain must retain every slot instead.
        pub fn deinitDrained(s: *Self) void {
            s.backend.release(s.buffer);
            s.a.free(s.tokens);
            s.* = undefined;
        }
        /// A failed queued write poisons this stream, while retaining every resource until a successful drain.
        pub fn poison(s: *Self) void {
            s.phase = .poisoned;
        }
        pub fn reset(s: *Self) !void {
            s.poison();
            try s.backend.zero(s.buffer, 0, s.bytes);
            s.pos = 0;
            s.base_pos = 0;
            s.rows = 0;
            s.base_valid = false;
            s.phase = .ready;
        }
        pub fn layers(s: *const Self) usize {
            return s.layout.count;
        }
        pub fn history(s: *const Self) []const u32 {
            return s.tokens[0..s.pos];
        }
        fn frame(s: *const Self, slot: usize, range: layout_.Range) !View {
            if (slot > s.window + 1 or range.offset > s.layout.state_bytes or range.bytes > s.layout.state_bytes - range.offset) return error.Qwen4StateRange;
            return .{ .buffer = s.buffer, .offset = s.layout.kv_bytes + slot * s.layout.state_bytes + range.offset, .len = range.bytes };
        }
        pub fn current(s: *const Self, range: layout_.Range) !View {
            return s.frame(0, range);
        }
        pub fn base(s: *const Self, range: layout_.Range) !View {
            return s.frame(1, range);
        }
        pub fn row(s: *const Self, index: usize, range: layout_.Range) !View {
            if ((s.phase != .working and s.phase != .verified) or index >= s.rows) return error.Qwen4StateRow;
            return s.frame(index + 2, range);
        }
        /// One layer's outputs occupy the same range in each complete kept frame, separated by the frame stride.
        pub fn rowSpan(s: *const Self, range: layout_.Range) !View {
            var out = try s.row(0, range);
            out.len = try std.math.add(usize, try std.math.mul(usize, s.rows - 1, s.layout.state_bytes), range.bytes);
            if (out.offset > s.bytes or out.len > s.bytes - out.offset) return error.Qwen4StateRange;
            return out;
        }
        pub fn kv(s: *const Self, range: layout_.Range) !View {
            if (range.offset > s.layout.kv_bytes or range.bytes > s.layout.kv_bytes - range.offset) return error.Qwen4StateRange;
            return .{ .buffer = s.buffer, .offset = range.offset, .len = range.bytes };
        }
        pub fn canBegin(s: *const Self, rows: usize) !void {
            if (s.phase != .ready) return error.Qwen4StatePhase;
            if (rows == 0 or rows > s.window or rows > s.layout.capacity - s.pos) return error.Qwen4StateCapacity;
        }
        pub fn begin(s: *Self, rows: usize) !void {
            try s.canBegin(rows);
            s.base_valid = false;
            s.poison();
            try s.backend.copy(s.buffer, s.layout.kv_bytes + s.layout.state_bytes, s.buffer, s.layout.kv_bytes, s.layout.state_bytes);
            s.base_pos = s.pos;
            s.rows = rows;
            s.base_valid = true;
            s.recorded = @splat(false);
            s.phase = .working;
        }
        pub fn admit(s: *const Self, rows: usize) !void {
            if (s.phase != .working or rows != s.rows or rows > s.layout.capacity - s.base_pos) return error.Qwen4StatePhase;
        }
        /// The forward calls this only after its layer's convolution, recurrence and lexical row writes were queued.
        pub fn captured(s: *Self, layer: usize) !void {
            if (s.phase != .working or layer >= s.layout.count or s.recorded[layer]) return error.Qwen4StatePhase;
            s.recorded[layer] = true;
        }
        pub fn advanced(s: *Self, tokens: []const u32) !void {
            try s.admit(tokens.len);
            for (s.layout.layers[0..s.layout.count], 0..) |layer, i| {
                if ((layer.conv.bytes != 0 or layer.ple.bytes != 0) and !s.recorded[i]) return error.Qwen4StateMissingRow;
            }
            for (tokens) |token| if (token >= s.c.vocab) return error.TokenOutOfRange;
            @memcpy(s.tokens[s.base_pos..][0..tokens.len], tokens);
            s.pos = s.base_pos + tokens.len;
            s.phase = .verified;
        }
        /// The shared lane verifier passes a prefix count; no discarded row's recurrent state is replayed or reused.
        pub fn keep(s: *Self, count: usize) !void {
            if (s.phase != .verified or count > s.rows) return error.Qwen4StatePhase;
            const source = if (count == 0) @as(usize, 1) else count + 1;
            s.poison();
            try s.backend.copy(s.buffer, s.layout.kv_bytes, s.buffer, s.layout.kv_bytes + source * s.layout.state_bytes, s.layout.state_bytes);
            s.pos = s.base_pos + count;
            s.rows = 0;
            s.base_valid = false;
            s.phase = .ready;
        }
        /// An interrupted forward rolls back to its untouched base; a failed base copy requires reset instead.
        pub fn abort(s: *Self) !void {
            if (!s.base_valid) return error.Qwen4StateMissingBase;
            s.poison();
            try s.backend.copy(s.buffer, s.layout.kv_bytes, s.buffer, s.layout.kv_bytes + s.layout.state_bytes, s.layout.state_bytes);
            s.pos = s.base_pos;
            s.rows = 0;
            s.base_valid = false;
            s.phase = .ready;
        }
    };
}
