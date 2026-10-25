//! The native forward owner borrows one backend; the shared lane core decides scheduling and accepted prefixes.
const std = @import("std");
const cfg = @import("config.zig");
const admission = @import("native_admission.zig");
const graph = @import("graph.zig");
pub fn Engine(comptime G: type) type {
    const B = G.Storage;
    const Placed = @import("device_weights.zig").Placed(B);
    const Model = @import("model.zig").Model(B);
    const Scratch = @import("arena.zig").Arena(B);
    const State = @import("state.zig").State(B);
    const Batch = @import("batch_state.zig").Batch(State);
    return struct {
        const Self = @This();
        pub const Storage = B;
        pub const Gpu = G;
        pub const Segment = Batch.Segment;
        pub const Tensor = Scratch.Tensor;
        pub const Slot = struct { state: State, last: B.Buffer, leased: bool = false, has_logits: bool = false };
        const Stage = enum { empty, text, config, reader, weights, model, gpu, scratch, slots };
        a: std.mem.Allocator,
        backend: *B,
        o: admission.Options,
        stage: Stage = .empty,
        config_text: []u8 = undefined,
        c: cfg.Config = undefined,
        reader: @import("reader.zig").Reader = undefined,
        weights: Placed = undefined,
        model: Model = undefined,
        gpu: G = undefined,
        scratch: Scratch = undefined,
        plan: admission.Plan = undefined,
        slots: []Slot = undefined,
        created: usize = 0,
        failed: std.atomic.Value(bool) = .init(false),
        pub fn open(a: std.mem.Allocator, io: std.Io, dir: []const u8, b: *B, o: admission.Options, p: *cfg.Problem, loader: anytype) !*Self {
            // A supplied backend may not transfer pending work into a new engine's cold teardown path.
            try b.drain();
            const e = try a.create(Self);
            e.* = .{ .a = a, .backend = b, .o = o };
            errdefer {
                e.drop();
                a.destroy(e);
            }
            const path = try std.fs.path.join(a, &.{ dir, "config.json" });
            defer a.free(path);
            e.config_text = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 24));
            e.stage = .text;
            if (o.config_digest) |expected| {
                var actual: [32]u8 = undefined;
                std.crypto.hash.sha2.Sha256.hash(e.config_text, &actual, .{});
                if (!std.mem.eql(u8, &expected, &actual)) return p.reject("config.json changed during admission");
            }
            e.c = try cfg.parse(a, e.config_text, p);
            e.stage = .config;
            _ = try admission.check(&e.c, G.kind, p);
            e.reader = try @import("reader.zig").Reader.open(a, io, dir, &e.c, p);
            e.stage = .reader;
            const bytes = try Placed.measure(a, &e.reader, &e.c, p);
            e.plan = try admission.Plan.make(&e.c, G.kind, bytes, o, p);
            e.weights = try Placed.open(a, b, &e.reader, &e.c, e.plan.weights, p);
            e.stage = .weights;
            e.model = try Model.init(a, &e.weights, p);
            e.stage = .model;
            e.gpu = try loader.load(a, b, &e.c, p);
            e.stage = .gpu;
            e.scratch = try Scratch.open(b, e.plan.scratch, e.plan.scratch);
            e.stage = .scratch;
            e.slots = try a.alloc(Slot, o.streams);
            e.stage = .slots;
            for (e.slots) |*slot| {
                var state = try State.open(a, b, &e.c, o.capacity, o.window, e.plan.state);
                const last = b.allocate(e.plan.last_logits) catch |err| {
                    state.deinitDrained();
                    return err;
                };
                slot.* = .{ .state = state, .last = last };
                e.created += 1;
            }
            return e;
        }
        fn atLeast(e: *const Self, stage: Stage) bool {
            return @backingInt(e.stage) >= @backingInt(stage);
        }
        /// Cold construction queues no inference; normal teardown proves completion before entering here.
        fn drop(e: *Self) void {
            if (e.stage == .slots) {
                for (e.slots[0..e.created]) |*slot| {
                    e.backend.release(slot.last);
                    slot.state.deinitDrained();
                }
                e.a.free(e.slots);
            }
            if (e.atLeast(.scratch)) e.backend.release(e.scratch.buffer);
            if (e.atLeast(.gpu)) e.gpu.deinitDrained();
            if (e.atLeast(.model)) e.model.deinit();
            if (e.atLeast(.weights)) e.weights.deinitDrained();
            if (e.atLeast(.reader)) e.reader.close();
            if (e.atLeast(.config)) e.c.deinit();
            if (e.atLeast(.text)) e.a.free(e.config_text);
        }
        pub fn healthy(e: *const Self) bool {
            return !e.failed.load(.acquire);
        }
        pub fn fail(e: *Self) void {
            e.failed.store(true, .release);
        }
        pub fn close(e: *Self) !void {
            try e.backend.drain();
            const a = e.a;
            e.drop();
            a.destroy(e);
        }
        pub fn lease(e: *Self) !*Slot {
            if (!e.healthy()) return error.Qwen4EngineFailed;
            for (e.slots) |*slot| if (!slot.leased) {
                slot.leased = true;
                slot.has_logits = false;
                slot.state.reset() catch |err| {
                    e.fail();
                    return err;
                };
                return slot;
            };
            return error.Qwen4NoStreamSlot;
        }
        pub const slotOf = own;
        fn own(e: *Self, state: *State) !*Slot {
            for (e.slots) |*slot| if (&slot.state == state and slot.leased) return slot;
            return error.Qwen4ForeignStream;
        }
        /// The callback queues draws of all verification rows before scratch can serve another shared round.
        pub fn forward(e: *Self, items: []const Segment, after: anytype, p: *cfg.Problem) !void {
            if (!e.healthy()) return error.Qwen4EngineFailed;
            var batch: Batch = .{ .items = items };
            const count = try batch.count();
            if (count > e.plan.rows) return error.Qwen4BatchRows;
            for (items) |item| {
                _ = try e.own(item.state);
                try item.state.canBegin(item.tokens.len);
                for (item.tokens) |token| if (token >= e.c.vocab) return error.TokenOutOfRange;
            }
            const tokens = try e.a.alloc(u32, count);
            defer e.a.free(tokens);
            var off: usize = 0;
            for (items) |item| {
                @memcpy(tokens[off..][0..item.tokens.len], item.tokens);
                off += item.tokens.len;
            }
            var ls: [cfg.max_layers]graph.Layer = undefined;
            const geometry = try graph.geometry(&e.c, try graph.layers(&e.c, &ls));
            e.gpu.begin() catch |err| {
                e.fail();
                return err;
            };
            errdefer {
                e.fail();
                batch.poison();
                for (items) |item| item.state.base_valid = false;
                e.gpu.discard();
            }
            try e.scratch.rewind(0);
            try batch.begin();
            var ops: @import("gpu_ops.zig").Ops(G) = .{ .gpu = &e.gpu, .model = &e.model, .scratch = &e.scratch, .host = e.a, .problem = p };
            const logits = try graph.forward(&ops, geometry, &batch, tokens);
            off = 0;
            for (items) |item| {
                const slot = try e.own(item.state);
                const row = off + item.tokens.len - 1;
                try e.backend.copy(slot.last, 0, logits.buffer, logits.offset + row * e.c.vocab * 2, e.plan.last_logits);
                off += item.tokens.len;
            }
            try after.encode(&e.gpu, logits, items);
            try e.gpu.submit();
            for (items) |item| (try e.own(item.state)).has_logits = true;
        }
        pub fn keep(e: *Self, slot: *Slot, count: usize) !void {
            if (!e.healthy()) return error.Qwen4EngineFailed;
            _ = try e.own(&slot.state);
            const all = count == slot.state.rows;
            slot.state.keep(count) catch |err| {
                e.fail();
                return err;
            };
            if (!all) slot.has_logits = false;
        }
        /// Releasing a stream retains its stable allocation for another request; teardown alone frees device storage.
        pub fn release(e: *Self, slot: *Slot) !void {
            _ = try e.own(&slot.state);
            if (slot.state.rows != 0) try e.keep(slot, 0);
            slot.leased = false;
            slot.has_logits = false;
        }
        pub const NoDraw = struct {
            pub fn encode(_: @This(), _: *G, _: Tensor, _: []const Segment) !void {}
        };
    };
}
