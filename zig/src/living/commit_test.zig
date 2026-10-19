//! Rejection restores every pocket component, including changes to old neurons during a repair.
const std = @import("std");
const t = std.testing;

fn Cases(comptime family: @import("family.zig").Family) type {
    return struct {
        const Trainer = struct {};
        const P = @import("pocket.zig").Pocket(Trainer, family);
        const Host = struct {
            pub fn pocketSwiglu(_: @This(), _: usize, _: []const f32, _: []const f32, _: []const f32, _: usize) !void {}
        };
        test "Learning persistence: rejected repair restores gate up down used and lesson metadata exactly" {
            var p: P = .{ .gpa = t.allocator, .mode = .reply, .hidden = family.quant.group };
            defer p.deinit();
            const n = P.neurons * p.hidden;
            try p.moe.append(t.allocator, 3);
            inline for (.{ "up", "down", "gate" }) |f| try @field(p, f).appendNTimes(t.allocator, 0.125, n);
            try p.scratch.appendNTimes(t.allocator, 0, n);
            try p.used.append(t.allocator, 1);
            try p.learned.append(t.allocator, .{ .slot = 0, .neuron = 0, .key = .reply, .start = 0, .count = 1, .subject = 7, .lesson = 1 });
            try p.pos.append(t.allocator, 0x3f80);
            try p.mark();
            p.gate.items[0] = 7;
            p.up.items[0] = 8;
            p.down.items[0] = 9;
            p.used.items[0] = 2;
            p.learned.items[0].lesson = 77;
            p.pos.items[0] = 0;
            try p.retired_lessons.append(t.allocator, 1);
            var trainer: Trainer = .{};
            try p.undo(&trainer, Host{});
            try t.expectEqual(@as(f32, 0.125), p.gate.items[0]);
            try t.expectEqual(@as(f32, 0.125), p.up.items[0]);
            try t.expectEqual(@as(f32, 0.125), p.down.items[0]);
            try t.expectEqual(@as(usize, 1), p.used.items[0]);
            try t.expectEqual(@as(u32, 1), p.learned.items[0].lesson);
            try t.expectEqual(@as(u16, 0x3f80), p.pos.items[0]);
            try t.expectEqual(@as(usize, 0), p.retired_lessons.items.len);
        }
        const ReadHost = struct {
            fail: bool,
            pub fn layout(_: @This()) @import("family.zig").Layout {
                var has: @import("family.zig").Layers = .empty;
                has.set(0);
                has.set(1);
                return .fromExperts(family.quant.group, 64, 3, has);
            }
            pub fn pocketReadSwiglu(h: @This(), _: usize, gate: []f32, up: []f32, down: []f32) !usize {
                if (h.fail) return error.ReadFailed;
                @memset(gate, 0);
                @memset(up, 0);
                @memset(down, 0);
                gate[0] = 16;
                up[0] = 1;
                down[0] = 1;
                return 1;
            }
        };
        test "Learning persistence: failed initial read publishes no count or lesson and a successful retry rereads" {
            var p: P = .{ .gpa = t.allocator, .mode = .reply };
            defer p.deinit();
            try t.expectError(error.ReadFailed, @import("pocket_load.zig").load(&p, ReadHost{ .fail = true }));
            try t.expectEqual(@as(usize, 0), p.used.items.len);
            try t.expectEqual(@as(usize, 0), p.moe.items.len);
            try t.expectEqual(@as(usize, 0), p.learned.items.len);
            try t.expect(!p.snap.valid);
            try @import("pocket_load.zig").load(&p, ReadHost{ .fail = false });
            try t.expectEqualSlices(usize, &.{ 1, 1 }, p.used.items);
        }
        const DeleteHost = struct {
            fails: *bool,
            pub fn layout(_: @This()) @import("family.zig").Layout {
                return (ReadHost{ .fail = false }).layout();
            }
            pub fn pocketReadSwiglu(_: @This(), _: usize, _: []f32, _: []f32, _: []f32) !usize {
                return error.UnexpectedRead;
            }
            pub fn pocketSwiglu(h: @This(), _: usize, _: []const f32, _: []const f32, _: []const f32, _: usize) !void {
                if (h.fails.*) return error.UploadFailed;
            }
        };
        test "Learning persistence: SwiGLU delete retains gates and exact metadata until a stale put-back lands" {
            var p: P = .{ .gpa = t.allocator, .mode = .reply };
            defer p.deinit();
            try @import("pocket_load.zig").load(&p, ReadHost{ .fail = false });
            try p.learned.append(t.allocator, .{ .slot = 0, .neuron = 0, .key = .reply, .start = 0, .count = 0, .subject = 7, .lesson = 1 });
            const before = @import("pocket_mark.zig").hash(&p);
            var fails = false;
            const host: DeleteHost = .{ .fails = &fails };
            var trainer: Trainer = .{};
            const f = @import("pocket_forget.zig");
            try t.expectEqual(@as(usize, 1), try f.forget(&p, &trainer, host, &.{.{ 0, 0 }}));
            try t.expect(f.pending(&p));
            try t.expectEqual(@as(usize, 0), p.used.items[0]);
            fails = true;
            try t.expectError(error.PutBackNotOnDevice, f.unforget(&p, &trainer, host));
            try t.expect(f.stale(&p));
            try t.expectEqual(before, @import("pocket_mark.zig").hash(&p));
            try t.expectEqual(@as(u32, 1), p.learned.items[0].lesson);
            fails = false;
            _ = try f.unforget(&p, &trainer, host);
            try t.expect(!f.pending(&p) and !f.stale(&p));
            try t.expectEqual(before, @import("pocket_mark.zig").hash(&p));
        }
        const ChangedSave = struct {
            pub fn pocketSaveSwiglu(_: @This(), _: []const usize, _: usize, gate: []f32, _: []f32, _: []f32) !usize {
                gate[0] += 1;
                return 1;
            }
        };
        test "Learning persistence: a writer that changes the judged form marks the learner stale" {
            var p: P = .{ .gpa = t.allocator, .mode = .reply };
            defer p.deinit();
            try @import("pocket_load.zig").load(&p, ReadHost{ .fail = false });
            var trainer: Trainer = .{};
            try t.expectError(error.PocketSavedDifferently, p.save(&trainer, ChangedSave{}));
            try t.expect(p.saved_mismatch);
        }
        test "Learning persistence: failed replacement stays pending until every earlier neuron is put back" {
            var p: P = .{ .gpa = t.allocator, .mode = .reply };
            defer p.deinit();
            try @import("pocket_load.zig").load(&p, ReadHost{ .fail = false });
            const before = @import("pocket_mark.zig").hash(&p);
            var fails = true;
            const host: DeleteHost = .{ .fails = &fails };
            var trainer: Trainer = .{};
            try t.expectError(error.UploadFailed, p.retire(&trainer, host, &.{ .{ 0, 0 }, .{ 0, 0 }, .{ 1, 0 } }));
            try t.expect(p.retire_pending);
            try t.expectError(error.PocketUnresolved, p.mark());
            fails = false;
            try p.unretire(&trainer, host);
            try t.expect(!p.retire_pending);
            try t.expectEqual(before, @import("pocket_mark.zig").hash(&p));
        }
    };
}
test {
    _ = Cases(@import("families.zig").qwen4_exp);
    _ = Cases(@import("families.zig").qwen3_5);
}
