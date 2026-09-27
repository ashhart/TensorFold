//! Original Python packed row selection, checked before any float conversion.
const std = @import("std");
const mx = @import("mlx.zig");
const vocab = @import("draft_vocab.zig");
pub fn check(io: std.Io, dir: []const u8) !void {
    try mx.init();
    defer mx.shutdown();
    inline for (.{ "nemotron", "flash" }) |name| {
        const size: usize = if (std.mem.eql(u8, name, "nemotron")) 131072 else 248320;
        var store = @import("checkpoint.zig").Store.init(if (size == 131072) 64 else 32);
        defer store.deinit();
        var path: [4096]u8 = undefined;
        try store.loadFile(io, try std.fmt.bufPrint(&path, "{s}/{s}.safetensors", .{ dir, name }), "", "");
        try vocab.install(&store, @field(vocab.data, name), size, 8);
        var s = mx.Scope{};
        defer s.deinit();
        inline for (.{ "ids", "weight", "scales", "biases" }) |field| {
            const a = try store.get(if (std.mem.eql(u8, field, "ids")) "draft_ids" else "draft_lm_head." ++ field);
            const b = try store.get("expected." ++ field);
            if (!std.mem.eql(c_int, mx.shape(a), mx.shape(b)) or mx.dtype(a) != mx.dtype(b)) return error.DraftVocabularyMismatch;
            if (mx.dtype(a) == mx.c.MLX_UINT32) {
                try mx.evalMany(&.{ a, b }, false);
                const count = mx.c.mlx_array_size(a);
                if (!std.mem.eql(u32, mx.c.mlx_array_data_uint32(a)[0..count], mx.c.mlx_array_data_uint32(b)[0..count])) return error.DraftVocabularyMismatch;
            } else try @import("sampling_checks.zig").equal(&s, a, b);
        }
        std.debug.print("PASS: {s} draft mapping and all packed head rows match original Python\n", .{name});
    }
}
