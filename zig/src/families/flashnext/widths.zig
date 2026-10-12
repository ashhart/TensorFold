//! Per-module MLX widths through core/quant, which indexes overrides by module; `false` and `{}` mark plain modules.
const std = @import("std");
const mlx = @import("../../core/quant/mlx.zig");
const st = @import("../../core/safetensors.zig");
const affine = @import("affine.zig");
const config = @import("config.zig");

pub const Width = mlx.Width;

/// The widths the device kernels read: MLX affine 2, 3, 4, 5, 6 or 8 bits in groups of 32, 64 or 128.
pub fn launched(w: Width) bool {
    const bits_ok = w.bits >= 2 and w.bits <= 8 and w.bits != 7;
    return bits_ok and (w.group == 32 or w.group == 64 or w.group == 128);
}

pub const Widths = struct {
    arena: std.heap.ArenaAllocator,
    /// Overrides by config key and by canonical module (`modules`); null is a width core/quant refuses (no guessing).
    q: mlx.Config,
    /// Modules the config declares unquantized (`false` or `{}`).
    plain: std.StringHashMapUnmanaged(void) = .empty,

    pub fn init(gpa: std.mem.Allocator, c: *const config.Config) !Widths {
        var w: Widths = .{ .arena = .init(gpa), .q = undefined };
        errdefer w.arena.deinit();
        const a = w.arena.allocator();
        // core/quant indexes overrides by module and refuses spellings that disagree
        w.q = (mlx.detect(a, .{ .root = c.parsed.value.object, .quantization = c.quant }) catch |err| return switch (err) {
            error.UnsupportedQuantization => error.ConflictingAffineAlias, // the only table core/quant refuses
            else => err,
        }) orelse return error.UnsupportedAffineMode;
        var it = c.quant.iterator();
        while (it.next()) |item| {
            const plain = switch (item.value_ptr.*) {
                .bool => |on| !on,
                .object => |o| o.count() == 0,
                else => continue,
            };
            const key = config.canonical(item.key_ptr.*);
            if (!plain) {
                if (w.plain.contains(key)) return error.ConflictingAffineAlias;
                continue;
            }
            if (w.q.modules.get(key)) |declared| if (declared != null) return error.ConflictingAffineAlias;
            try w.plain.put(a, key, {});
        }
        return w;
    }

    pub fn deinit(w: *Widths) void {
        w.arena.deinit();
        w.* = undefined;
    }

    /// No override at all: every quantized matrix has the global width.
    pub fn uniform(w: *const Widths) bool {
        return w.q.overrides.count() == 0 and w.plain.count() == 0;
    }

    /// The module's width under any spelling of its key, or null when the config declares it unquantized.
    pub fn of(w: *const Widths, path: []const u8) !?Width {
        const key = config.canonical(path);
        if (w.plain.contains(key)) return null;
        return try w.q.width(key);
    }

    /// The module's stored matrix checked against its own width, never one inferred from the shapes.
    pub fn shaped(w: *const Widths, path: []const u8, weight: st.Entry, scales: st.Entry, biases: st.Entry) !affine.Shape {
        const width = (try w.of(path)) orelse return error.UnquantizedAffineTensor;
        return affine.matrix(weight, scales, biases, try affine.Spec.init(width.bits, width.group));
    }
};

test {
    _ = @import("widths_test.zig");
}
