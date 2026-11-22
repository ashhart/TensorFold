//! Checkpoint ownership and admission for the native Hy4-flash model family.
const std = @import("std");
const core = @import("core");
const config = @import("config.zig");
const weights = @import("weights.zig");

pub const Attention = enum { target, draft };

pub const Loaded = struct {
    config: std.json.Parsed(config.Model),
    checkpoint: core.Checkpoint,
    bits: u8,

    pub fn open(a: std.mem.Allocator, io: std.Io, directory: []const u8) !Loaded {
        const config_path = try std.fs.path.join(a, &.{ directory, "config.json" });
        defer a.free(config_path);
        const text = try std.Io.Dir.cwd().readFileAlloc(io, config_path, a, .limited(1 << 20));
        defer a.free(text);
        const parsed = try config.parse(a, text);
        errdefer parsed.deinit();
        const q = parsed.value.quantization orelse return error.UnsupportedHy4Affine;
        if (q.group_size != 64 or !std.mem.eql(u8, q.mode, "affine") or (q.bits != 4 and q.bits != 8))
            return error.UnsupportedHy4Affine;
        var checkpoint = try core.Checkpoint.openModel(a, io, directory);
        errdefer checkpoint.close();
        var validation: Validation = .{ .checkpoint = &checkpoint };
        try weights.check(&validation, parsed.value, q.bits);
        return .{ .config = parsed, .checkpoint = checkpoint, .bits = q.bits };
    }

    pub fn close(self: *Loaded) void {
        self.checkpoint.close();
        self.config.deinit();
        self.* = undefined;
    }

    pub fn attentionTargetLayer(self: *const Loaded, kind: Attention, layer: u32) !u32 {
        const model = self.config.value;
        return switch (kind) {
            .target => if (layer < model.text_config.num_hidden_layers) layer else error.BadAttentionLayer,
            .draft => if (layer < model.dfly_target_kv_layers.len) model.dfly_target_kv_layers[layer] else return error.BadAttentionLayer,
        };
    }

    /// Return this layer's learned sink, even when its draft attention shares the target's KV.
    pub fn attentionSink(self: *Loaded, kind: Attention, layer: u32, output: []f32) ![]f32 {
        const layer_config = try self.attentionTargetLayer(kind, layer);
        const heads = self.config.value.text_config.headShape(layer_config).heads;
        if (output.len < heads) return error.ShortAttentionSinkBuffer;
        var buffer: [96]u8 = undefined;
        const prefix = switch (kind) {
            .target => "model.layers",
            .draft => "model.dfly_layers",
        };
        const name = try std.fmt.bufPrint(&buffer, "{s}.{d}.self_attn.learnable_sink_param", .{ prefix, layer });
        const result = output[0..heads];
        switch (kind) {
            .target => {
                const tensor = try self.checkpoint.expect(name, .f32, &.{heads});
                for (result, 0..) |*value, i| value.* = @bitCast(std.mem.readInt(u32, tensor.bytes[4 * i ..][0..4], .little));
            },
            .draft => {
                const tensor = try self.checkpoint.expect(name, .bf16, &.{heads});
                for (result, 0..) |*value, i| {
                    const raw = std.mem.readInt(u16, tensor.bytes[2 * i ..][0..2], .little);
                    value.* = @bitCast(@as(u32, raw) << 16);
                }
            },
        }
        return result;
    }
};

const Validation = struct {
    checkpoint: *core.Checkpoint,

    pub fn expect(self: *Validation, name: []const u8, format: weights.DType, shape: []const usize) !void {
        const dtype: core.checkpoint_host.DType = switch (format) {
            inline else => |tag| @field(core.checkpoint_host.DType, @tagName(tag)),
        };
        _ = try self.checkpoint.expect(name, dtype, shape);
    }
};

/// `ck` names `get` never took, refused with up to five named.
pub fn refuseLeftovers(ck: *const core.Checkpoint, comptime err: anyerror) !void {
    var n: usize = 0;
    for (ck.files.items) |*f| for (f.names.keys()) |k| {
        if (ck.used.contains(k)) continue;
        if (n < 5 and !@import("builtin").is_test) std.log.err("unused checkpoint tensor {s}", .{k});
        n += 1;
    };
    if (n != 0) return err;
}
