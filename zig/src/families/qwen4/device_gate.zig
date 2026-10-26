//! The real engine, GPU drawer and queued logit capture share one gate on Metal, CUDA and the host instrument.
const std = @import("std");
const fixture = @import("device_fixture.zig");
const cfg = @import("config.zig");
pub fn Gate(comptime G: type, comptime D: type, comptime L: type, comptime read: anytype) type {
    const E = @import("engine.zig").Engine(G);
    const B = G.Storage;
    return struct {
        const Self = @This();
        pub const Output = struct { logits: []u16, picks: []u32 };
        a: std.mem.Allocator,
        e: *E,
        d: *D,
        buffer: B.Buffer,
        const Capture = struct {
            gate: *Self,
            pub fn encode(c: @This(), _: *G, x: E.Tensor, items: []const E.Segment) !void {
                const s = c.gate;
                if (x.rows > 32 or x.kind != .bf16) return error.Qwen4DeviceCapture;
                try s.e.backend.copy(s.buffer, 0, x.buffer, x.offset, try x.bytes());
                var positions: [32]u64 = undefined;
                var at: usize = 0;
                for (items) |item| for (0..item.tokens.len) |r| {
                    positions[at] = item.state.pos + r;
                    at += 1;
                };
                if (at != x.rows) return error.Qwen4DeviceCapture;
                try s.d.draw(x, 0, positions[0..at], null, &.{});
            }
        };
        fn run(s: *Self, items: []const E.Segment) !Output {
            var count: usize = 0;
            for (items) |item| count += item.tokens.len;
            const logits = try s.a.alloc(u16, count * s.e.c.vocab);
            const picks = try s.a.alloc(u32, count);
            var p: cfg.Problem = .{};
            try s.e.forward(items, Capture{ .gate = s }, &p);
            try s.e.backend.drain();
            try read(s.e.backend, s.buffer, std.mem.sliceAsBytes(logits));
            try s.d.read(0, picks);
            return .{ .logits = logits, .picks = picks };
        }
        fn one(s: *Self, slot: *E.Slot, tokens: []const u32) !Output {
            return s.run(&.{.{ .state = &slot.state, .tokens = tokens }});
        }
        fn equal(name: []const u8, want: Output, got: Output) !void {
            try fixture.same(u16, name, want.logits, got.logits);
            try fixture.same(u32, name, want.picks, got.picks);
        }
        fn slice(s: *const Self, out: Output, start: usize, end: usize) Output {
            return .{ .logits = out.logits[start * s.e.c.vocab .. end * s.e.c.vocab], .picks = out.picks[start..end] };
        }
        pub fn runAll(a: std.mem.Allocator, io: std.Io, b: *B, loader: anytype, d: *D) !void {
            var arena: std.heap.ArenaAllocator = .init(a);
            defer arena.deinit();
            const mem = arena.allocator();
            const file = try fixture.Fixture.open(a, io);
            defer file.close();
            var p: cfg.Problem = .{};
            const e = @import("engine.zig").Engine(G).open(a, io, file.path, b, .{ .capacity = 32, .window = 16, .streams = 2, .budget = std.math.maxInt(u64) }, &p, loader) catch |err| {
                std.debug.print("FAIL qwen4 engine open: {s}: {t}\n", .{ p.name(), err });
                return err;
            };
            var closed = false;
            defer if (!closed) e.close() catch {};
            const buffer = try b.allocate(32 * e.c.vocab * 2);
            var released = false;
            defer if (!released) {
                if (b.drain()) |_| b.release(buffer) else |_| {}
            };
            var s: Self = .{ .a = mem, .e = e, .d = d, .buffer = buffer };
            const doc = try std.json.parseFromSlice(fixture.Oracle, mem, if (@hasDecl(@TypeOf(loader), "reference")) @TypeOf(loader).reference else fixture.reference, .{});
            const ids = doc.value.tokens;
            const left = try e.lease();
            const right = try e.lease();
            const baseline = try s.one(left, ids);
            try fixture.oracle(doc.value, baseline.logits, baseline.picks);
            try e.keep(left, ids.len);
            std.debug.print("PASS qwen4 binary64: {d} logits and {d} GPU token picks\n", .{ baseline.logits.len, baseline.picks.len });
            for (0..5) |_| {
                try left.state.reset();
                try equal("repeat", baseline, try s.one(left, ids));
                try e.keep(left, ids.len);
            }
            for (1..ids.len) |cut| {
                try left.state.reset();
                const first = try s.one(left, ids[0..cut]);
                try e.keep(left, cut);
                try equal("prompt cut prefix", s.slice(baseline, 0, cut), first);
                try equal("prompt cut suffix", s.slice(baseline, cut, ids.len), try s.one(left, ids[cut..]));
                try e.keep(left, ids.len - cut);
                try left.state.reset();
                _ = try s.one(left, ids[0..cut]);
                try e.keep(left, cut);
                var saved = try @import("snapshot.zig").Snapshot(B).open(a, &left.state, @splat(17), std.math.maxInt(u64));
                defer saved.close() catch {};
                try saved.capture(&left.state, @splat(17));
                try right.state.reset();
                try saved.restore(&right.state, @splat(17));
                try equal("resumed suffix", s.slice(baseline, cut, ids.len), try s.one(right, ids[cut..]));
                try e.keep(right, ids.len - cut);
            }
            std.debug.print("PASS qwen4 prompt cuts and resumed == fresh: {d} cuts, every logit and token\n", .{ids.len - 1});
            const other = [_]u32{ 19, 1, 4, 17, 9, 6, 12, 24, 8, 3, 5, 11 };
            try right.state.reset();
            const other_baseline = try s.one(right, &other);
            try e.keep(right, other.len);
            try left.state.reset();
            try right.state.reset();
            for (0..ids.len) |r| {
                const shared = try s.run(&.{ .{ .state = &left.state, .tokens = ids[r..][0..1] }, .{ .state = &right.state, .tokens = other[r..][0..1] } });
                try equal("concurrent left", s.slice(baseline, r, r + 1), s.slice(shared, 0, 1));
                try equal("concurrent right", s.slice(other_baseline, r, r + 1), s.slice(shared, 1, 2));
                try e.keep(left, 1);
                try e.keep(right, 1);
            }
            try left.state.reset();
            try right.state.reset();
            const paired = try s.run(&.{ .{ .state = &left.state, .tokens = ids }, .{ .state = &right.state, .tokens = &other } });
            try equal("concurrent packed left", baseline, s.slice(paired, 0, ids.len));
            try equal("concurrent packed right", other_baseline, s.slice(paired, ids.len, ids.len + other.len));
            try e.keep(left, ids.len);
            try e.keep(right, other.len);
            std.debug.print("PASS qwen4 concurrent == solo: unequal histories, single rows and packed prompts\n", .{});
            for (0..ids.len) |accepted| {
                try left.state.reset();
                _ = try s.one(left, ids);
                try e.keep(left, accepted);
                try equal("partial acceptance", s.slice(baseline, accepted, ids.len), try s.one(left, ids[accepted..]));
                try e.keep(left, ids.len - accepted);
            }
            try left.state.reset();
            _ = try s.one(left, ids[0..7]);
            try left.state.abort();
            try equal("abort retry", baseline, try s.one(left, ids));
            try e.keep(left, ids.len);
            try e.release(left);
            try e.release(right);
            try @import("device_lanes.zig").run(E, D, L, a, io, e, d, ids);
            try b.drain();
            b.release(buffer);
            released = true;
            try e.close();
            closed = true;
            std.debug.print("N/A qwen4 learned MTP: no declared head; positive MTP configs refuse admission\n", .{});
            std.debug.print("PASS qwen4 device: binary64, tokens, repeats, cuts, resume, concurrency, rollback and copied drafts\n", .{});
        }
    };
}
