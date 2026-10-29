//! The shared lane core schedules native forwards; this adapter owns only stable stream slots and draw handles.
const std = @import("std");
const cfg = @import("config.zig");
pub fn Adapter(comptime E: type, comptime D: type, comptime L: type) type {
    const be = L.backend;
    const B = E.Storage;
    return struct {
        const Self = @This();
        pub const Stream = L.Stream;
        pub const Snap = @import("snapshot.zig").Snapshot(B);
        pub const Lane = struct { stream: ?*L.Stream = null, slot: ?*E.Slot = null, filled: usize = 0, handle: u64 = 0, value: ?u32 = null };
        a: std.mem.Allocator,
        e: *E,
        d: *D,
        lanes: []Lane,
        next_handle: u64 = 1,
        key: [32]u8,
        pub fn init(a: std.mem.Allocator, e: *E, d: *D) !Self {
            const lanes = try a.alloc(Lane, e.o.streams);
            @memset(lanes, .{});
            var h: std.crypto.hash.sha2.Sha256 = .init(.{});
            h.update("qwen4-bf16-hf-v1\x00");
            h.update(e.config_text);
            return .{ .a = a, .e = e, .d = d, .lanes = lanes, .key = h.finalResult() };
        }
        pub fn deinitDrained(l: *Self) void {
            l.a.free(l.lanes);
            l.* = undefined;
        }
        pub fn facts(l: *const Self) L.Model {
            return .{ .exact_width = @intCast(l.e.o.window), .hidden_rows = true, .batch_rows = @intCast(l.e.plan.rows), .max_streams = @intCast(l.lanes.len), .streams_exact = true, .copy_drafts = true };
        }
        pub fn backend(l: *Self) be.Backend {
            return .{ .ptr = l, .vtable = &.{ .prefill = prefill, .prefill_step = step, .cancel_prefill = cancelPrefill, .first = first, .first_masked = firstMasked, .queue = queue, .read = read, .verify = verify, .keep = keep, .draft = draft, .release = release, .reports_fill = true } };
        }
        fn self(p: *anyopaque) *Self {
            return @ptrCast(@alignCast(p));
        }
        pub fn find(l: *Self, s: *L.Stream) !*Lane {
            for (l.lanes) |*lane| if (lane.stream == s) return lane;
            return error.Qwen4ForeignStream;
        }
        pub fn last(l: *Self, lane: *Lane) E.Tensor {
            return .{ .buffer = lane.slot.?.last, .offset = 0, .rows = 1, .width = l.e.c.vocab, .kind = .bf16 };
        }
        pub fn acquire(l: *Self, s: *L.Stream) !*Lane {
            if (!l.e.healthy()) return error.Qwen4EngineFailed;
            if (s.prompt_len == 0 or s.prompt_len > l.e.o.capacity or s.max_new > l.e.o.capacity - s.prompt_len) return error.Qwen4Context;
            if (s.images.len > 0 or s.embedding != null) return error.Qwen4TextOnly;
            if (s.logprobs != null or s.prompt_logprobs != null) return error.Qwen4LogprobsUnsupported;
            for (s.prompt()) |token| if (token >= l.e.c.vocab) return error.TokenOutOfRange;
            for (l.lanes) |*lane| if (lane.stream == null) {
                const slot = try l.e.lease();
                lane.* = .{ .stream = s, .slot = slot };
                if (s.reuse.saved) |ptr| {
                    const snap: *Snap = @ptrCast(@alignCast(ptr));
                    l.restore(lane, snap) catch |err| {
                        if (!l.e.healthy()) return err;
                        s.reuse_failed = true;
                        s.reuse_error = err;
                        try slot.state.reset();
                        slot.has_logits = false;
                    };
                }
                lane.filled = slot.state.pos;
                s.cached = @intCast(lane.filled);
                s.fill_at.store(@intCast(lane.filled), .release);
                return lane;
            };
            return error.Qwen4NoStreamSlot;
        }
        pub fn restore(l: *Self, lane: *Lane, snap: *Snap) !void {
            const slot = lane.slot.?;
            const s = lane.stream.?;
            try snap.admitRestore(&slot.state, l.key);
            if (!snap.last_valid or snap.at != s.reuse.at or snap.at > s.prompt_len or !std.mem.eql(u32, snap.tokens, s.prompt()[0..snap.at])) return error.Qwen4SnapshotPrefix;
            try l.e.gpu.begin();
            errdefer {
                l.e.fail();
                slot.state.poison();
                l.e.gpu.discard();
            }
            try snap.restoreWithLogits(&slot.state, l.key, l.last(lane));
            try l.e.gpu.submit();
            slot.has_logits = true;
        }
        fn prefill(p: *anyopaque, s: *L.Stream) !void {
            while (!try step(p, s)) {}
        }
        fn step(p: *anyopaque, s: *L.Stream) !bool {
            const l = self(p);
            const lane = l.find(s) catch try l.acquire(s);
            if (s.isCancelled()) {
                s.keepInterruptedPrefill(@intCast(lane.filled));
                return error.Cancelled;
            }
            if (lane.filled == s.prompt_len) return true;
            var end = @min(s.prompt_len, lane.filled + l.e.o.window);
            for (s.chunks) |cut| if (cut > lane.filled) {
                end = @min(end, cut);
                break;
            };
            for (s.reuse.marks) |cut| if (cut > lane.filled) {
                end = @min(end, cut);
                break;
            };
            var problem: cfg.Problem = .{};
            try l.e.forward(&.{.{ .state = &lane.slot.?.state, .tokens = s.prompt()[lane.filled..end] }}, E.NoDraw{}, &problem);
            try l.e.keep(lane.slot.?, end - lane.filled);
            l.e.backend.drain() catch |err| {
                l.e.fail();
                return err;
            };
            lane.filled = end;
            s.fill_at.store(@intCast(end), .release);
            for (s.reuse.marks) |cut| if (cut == end) {
                if (s.reuse.hook) |hook| hook.at(hook.ptr, s, @intCast(end));
                break;
            };
            return end == s.prompt_len;
        }
        fn cancelPrefill(p: *anyopaque, s: *L.Stream) !void {
            const l = self(p);
            const lane = try l.find(s);
            s.keepInterruptedPrefill(@intCast(lane.filled));
        }
        fn allocateHandle(l: *Self, lane: *Lane) !u64 {
            if (l.next_handle == std.math.maxInt(u64)) return error.Qwen4DrawHandleExhausted;
            lane.handle = l.next_handle;
            l.next_handle += 1;
            lane.value = null;
            return lane.handle;
        }
        fn outputOffset(l: *Self, lane: *Lane) usize {
            return (l.d.rows + (@intFromPtr(lane) - @intFromPtr(l.lanes.ptr)) / @sizeOf(Lane)) * 4;
        }
        fn first(p: *anyopaque, s: *L.Stream, pos: u64) !u64 {
            return firstMasked(p, s, pos, &.{});
        }
        fn firstMasked(p: *anyopaque, s: *L.Stream, pos: u64, bits: []const u32) !u64 {
            const l = self(p);
            const lane = try l.find(s);
            const slot = lane.slot.?;
            if (!l.e.healthy()) return error.Qwen4EngineFailed;
            if (!slot.has_logits or slot.state.pos != s.prompt_len or pos != s.prompt_len) return error.Qwen4FirstPosition;
            try @import("draw_inputs.zig").check(l.e.c.vocab, &.{pos}, s.sampling, bits);
            const handle = try l.allocateHandle(lane);
            try l.e.gpu.begin();
            errdefer {
                l.e.fail();
                l.e.gpu.discard();
            }
            try l.e.scratch.rewind(0);
            const x = try l.e.scratch.take(1, l.e.c.vocab, .bf16);
            try l.e.backend.copy(x.buffer, x.offset, slot.last, 0, try x.bytes());
            try l.d.draw(x, l.outputOffset(lane), &.{pos}, s.sampling, bits);
            try l.e.gpu.submit();
            return handle;
        }
        fn queue(p: *anyopaque, s: *L.Stream, feed: be.Feed, pos: u64) !u64 {
            const l = self(p);
            const lane = try l.find(s);
            const value = switch (feed) {
                .value => |v| v,
                .handle => |h| try read(p, h),
            };
            const tokens = [_]u32{value};
            var problem: cfg.Problem = .{};
            if (pos != lane.slot.?.state.pos + 1) return error.Qwen4DrawPosition;
            const handle = try l.allocateHandle(lane);
            const Draw = struct {
                l: *Self,
                lane: *Lane,
                position: u64,
                pub fn encode(c: @This(), _: *E.Gpu, x: E.Tensor, _: []const E.Segment) !void {
                    try c.l.d.draw(x, c.l.outputOffset(c.lane), &.{c.position}, c.lane.stream.?.sampling, &.{});
                }
            };
            try l.e.forward(&.{.{ .state = &lane.slot.?.state, .tokens = &tokens }}, Draw{ .l = l, .lane = lane, .position = pos }, &problem);
            try l.e.keep(lane.slot.?, 1);
            return handle;
        }
        fn read(p: *anyopaque, h: u64) !u32 {
            const l = self(p);
            if (!l.e.healthy()) return error.Qwen4EngineFailed;
            for (l.lanes) |*lane| if (lane.stream != null and lane.handle == h and h != 0) {
                if (lane.value) |v| return v;
                var out: [1]u32 = undefined;
                l.d.read(l.outputOffset(lane), &out) catch |err| {
                    l.e.fail();
                    return err;
                };
                lane.value = out[0];
                return out[0];
            };
            return error.Qwen4StaleDrawHandle;
        }
        fn verify(p: *anyopaque, windows: []const be.Window, out: []be.Verified) !void {
            const l = self(p);
            if (windows.len == 0 or windows.len != out.len or windows.len > l.lanes.len) return error.Qwen4VerifyShape;
            var segments: [128]E.Segment = undefined;
            var tokens: [128]u32 = undefined;
            var count: usize = 0;
            for (windows, out, 0..) |w, result, i| {
                const lane = try l.find(w.stream);
                const n = w.rows();
                for (windows[0..i]) |prior| if (prior.stream == w.stream) return error.Qwen4DuplicateStream;
                const state = &lane.slot.?.state;
                if (state.phase != .ready and state.phase != .verified) return error.Qwen4StatePhase;
                if (n > state.layout.capacity - state.pos) return error.Qwen4StateCapacity;
                if (w.pending >= l.e.c.vocab) return error.TokenOutOfRange;
                for (w.tokens) |token| if (token >= l.e.c.vocab) return error.TokenOutOfRange;
                if (w.held != 0 or w.early) return error.Qwen4NoDraftHead;
                if (n > l.e.o.window or n > tokens.len - count or w.positions.len != n or result.sampled.len != n or result.drafts.len != w.tokens.len or result.rows.len != 0) return error.Qwen4VerifyShape;
                if (w.parents) |parents| {
                    if (parents.len != n) return error.Qwen4VerifyShape;
                    for (parents, 0..) |parent, r| if (parent != @as(i32, @intCast(r)) - 1) return error.Qwen4TreeUnsupported;
                }
                for (w.positions, 0..) |pos, r| if (pos != lane.slot.?.state.pos + r + 1) return error.Qwen4DrawPosition;
                try @import("draw_inputs.zig").check(l.e.c.vocab, w.positions, w.stream.sampling, w.masks);
                tokens[count] = w.pending;
                @memcpy(tokens[count + 1 ..][0..w.tokens.len], w.tokens);
                segments[i] = .{ .state = &lane.slot.?.state, .tokens = tokens[count..][0..n] };
                count += n;
            }
            const Draw = struct {
                l: *Self,
                windows: []const be.Window,
                pub fn encode(c: @This(), _: *E.Gpu, x: E.Tensor, _: []const E.Segment) !void {
                    var off: usize = 0;
                    for (c.windows) |w| {
                        try c.l.d.draw(try x.subRows(off, w.rows()), off * 4, w.positions, w.stream.sampling, w.masks);
                        off += w.rows();
                    }
                }
            };
            // A lone fully accepted window needs no keep callback; settle it before reusing its saved row frames.
            for (segments[0..windows.len]) |seg| if (seg.state.phase == .verified) try l.e.keep((try l.e.slotOf(seg.state)), seg.state.rows);
            var problem: cfg.Problem = .{};
            try l.e.forward(segments[0..windows.len], Draw{ .l = l, .windows = windows }, &problem);
            var sampled: [128]u32 = undefined;
            l.d.read(0, sampled[0..count]) catch |err| {
                l.e.fail();
                return err;
            };
            var off: usize = 0;
            for (windows, out) |w, result| {
                @memcpy(result.sampled, sampled[off..][0..w.rows()]);
                @memcpy(result.drafts, w.tokens);
                off += w.rows();
            }
        }
        fn keep(p: *anyopaque, windows: []const be.Window, paths: []const []const u32) !void {
            const l = self(p);
            if (windows.len != paths.len) return error.Qwen4VerifyShape;
            for (windows, paths) |w, path| {
                if (path.len > w.rows()) return error.Qwen4KeepPath;
                for (path, 0..) |row, i| if (row != i) return error.Qwen4KeepPath;
                _ = try l.find(w.stream);
            }
            for (windows, paths) |w, path| try l.e.keep((try l.find(w.stream)).slot.?, path.len);
        }
        fn draft(_: *anyopaque, requests: []const be.DraftRequest) !void {
            for (requests) |r| if (r.depth != 0 or r.lanes != null or r.early) return error.Qwen4NoDraftHead;
        }
        fn release(p: *anyopaque, s: *L.Stream) void {
            const l = self(p);
            const lane = l.find(s) catch return;
            l.e.release(lane.slot.?) catch {
                l.e.fail();
            };
            lane.* = .{};
        }
    };
}
