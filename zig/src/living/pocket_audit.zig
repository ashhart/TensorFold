//! Read a private live-pocket snapshot against forward-only rows, without opening or altering a lesson.
const std = @import("std");
const pockets = @import("pocket.zig");
const loading = @import("pocket_load.zig");
const whole = @import("whole.zig");

pub fn Audit(comptime H: type) type {
    return struct {
        const Self = @This();
        const P = pockets.Pocket(H.Trainer, H.family);
        pocket: P,

        pub fn init(l: anytype) !Self {
            try l.auditTrainer();
            var p = P.init(l.gpa);
            errdefer p.deinit();
            try loading.load(&p, whole.Hooks(H){ .b = l.b, .gpa = l.gpa, .io = l.io });
            return .{ .pocket = p };
        }

        pub fn deinit(s: *Self) void {
            s.pocket.deinit();
        }

        /// Every prompt row, including its last: the dummy target is never forwarded by a keys-only pass.
        pub fn read(s: *Self, l: anytype, ids: []const u32, neurons: anytype, out: anytype) !void {
            const p = &s.pocket;
            const t = l.trainer.?;
            if (ids.len == 0 or ids.len > H.max_rows or ids.len > t.pack) return error.AuditPromptTooLong;
            if (out.len != neurons.len) return error.BadAudit;
            for (neurons) |n| {
                const slot = std.mem.indexOfScalar(usize, p.moe.items, n.layer) orelse return error.AuditMissingNeuron;
                if (n.neuron >= p.used.items[slot]) return error.AuditMissingNeuron;
            }
            const input = try l.gpa.alloc(u32, ids.len + 1);
            defer l.gpa.free(input);
            @memcpy(input[0..ids.len], ids);
            input[ids.len] = 0;
            const count = if (@hasField(H.Trainer, "count")) t.count else {};
            defer if (@hasField(H.Trainer, "count")) {
                t.count = count;
            };
            _ = try t.step(input, ids.len, .keys, .{ .decide = @intCast(ids.len - 1), .whole = true });
            for (out) |*o| o.* = .{};
            for (neurons, out) |n, *o| {
                const slot = std.mem.indexOfScalar(usize, p.moe.items, n.layer).?;
                for (0..ids.len) |row| {
                    const x = p.act(slot, n.neuron, t.normed(n.layer, row));
                    if (!std.math.isFinite(x)) return error.AuditNonfinite;
                    if (@abs(x) > @abs(o.activation)) o.* = .{ .activation = x, .row = @intCast(row) };
                }
            }
        }
    };
}
