//! One aggregate row frame owns activation, group sums and metadata independently of stream caches.
const std = @import("std");
const mtl = @import("metal");
const Config = @import("config.zig").Config;
const Ref = @import("projection.zig").Ref;

pub const Field = enum { ids, dims, positions, hidden, residual, input, sums, mixed, mixer_out, output, z, a, b, query, key, value, norm_query, norm_key, q_gate, gate, up, activated, act_sums, logits, drawn, eps, parents, windows, row_slots, segments, keeps, kept_rows, rotated_key };

pub const Frame = struct {
    buffers: [std.enums.values(Field).len]mtl.Buffer,
    capacity: u32,
    config: Config,

    pub fn init(device: mtl.Device, config: Config, capacity: u32) !Frame {
        if (capacity == 0 or capacity > 128) return error.BadFrameRows;
        const mp = 16 * ((capacity + 15) / 16);
        const widths = [_]usize{
            capacity * 4,                                                                       8 * 4,                                                                                                                                  capacity * 4,
            capacity * config.hidden * 2,                                                       capacity * config.hidden * 2,                                                                                                           capacity * config.hidden * 2,
            mp * @max(config.hidden, @max(config.intermediate, config.gdnValueDim())) / 64 * 4, capacity * @max(config.gdnQkvDim(), @max(config.qDim(), @max(2 * config.intermediate, config.gdnValueDim() + 2 * config.v_heads))) * 2, capacity * config.gdnValueDim() * 2,
            capacity * config.hidden * 2,                                                       capacity * config.gdnValueDim() * 2,                                                                                                    capacity * config.v_heads * 2,
            capacity * config.v_heads * 2,                                                      capacity * config.heads * config.head_dim * 2,                                                                                          capacity * config.kvDim() * 2,
            capacity * config.kvDim() * 2,                                                      capacity * config.heads * config.head_dim * 2,                                                                                          capacity * config.kvDim() * 2,
            capacity * config.qDim() * 2,                                                       capacity * config.intermediate * 2,                                                                                                     capacity * config.intermediate * 2,
            capacity * config.intermediate * 2,                                                 mp * config.intermediate / 64 * 4,                                                                                                      capacity * config.vocab * 2,
            capacity * 4,                                                                       4,                                                                                                                                      capacity * 4,
            capacity * config.conv_kernel * 4,                                                  capacity * 4,                                                                                                                           64 * 16,
            64 * 16,                                                                            capacity * 4,                                                                                                                           capacity * config.kvDim() * 2,
        };
        comptime std.debug.assert(widths.len == std.enums.values(Field).len);
        var buffers: [widths.len]mtl.Buffer = undefined;
        var count: usize = 0;
        errdefer for (buffers[0..count]) |b| b.deinit();
        for (widths, &buffers) |bytes, *b| {
            b.* = try device.buffer(@max(bytes, 16), mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
            count += 1;
        }
        buffers[@backingInt(Field.eps)].slice(f32, 1)[0] = config.eps;
        return .{ .buffers = buffers, .capacity = capacity, .config = config };
    }

    pub fn deinit(frame: *Frame) void {
        for (frame.buffers) |b| b.deinit();
    }

    pub fn get(frame: *const Frame, field: Field) Ref {
        return .{ .buffer = frame.buffers[@backingInt(field)] };
    }

    pub fn setRows(frame: *Frame, ids: []const u32, positions: []const i32) !void {
        if (ids.len == 0 or ids.len > frame.capacity or positions.len != ids.len) return error.BadFrameRows;
        @memcpy(frame.get(.ids).buffer.slice(u32, ids.len), ids);
        @memcpy(frame.get(.positions).buffer.slice(i32, positions.len), positions);
        @memset(frame.get(.dims).buffer.slice(i32, 8), 0);
        frame.get(.dims).buffer.slice(i32, 8)[0] = @intCast(ids.len);
        frame.get(.dims).buffer.slice(i32, 8)[1] = @intCast(16 * ((ids.len + 15) / 16));
    }
};
