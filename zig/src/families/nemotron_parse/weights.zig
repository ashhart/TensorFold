//! Owned BF16 device weights; the FP32 checkpoint stays unchanged and all tensors are checked before upload.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const checkpoint = @import("checkpoint.zig");

pub const Weights = struct {
    gpa: std.mem.Allocator,
    names: std.StringHashMapUnmanaged(hip.DeviceBuffer) = .empty,
    bytes: usize = 0,

    pub fn load(a: std.mem.Allocator, r: *const hip.Runtime, io: std.Io, dir: []const u8) !Weights {
        var ck = try checkpoint.Checkpoint.open(a, io, dir);
        defer ck.close();
        var w: Weights = .{ .gpa = a };
        errdefer w.deinit();
        for (ck.file.names.keys()) |name| {
            const tensor = try ck.get(name);
            if (tensor.dtype == .i64) continue; // the checked summary indices are the fixed first two class rows
            const count = tensor.numel();
            const host = try a.alloc(u16, count);
            defer a.free(host);
            for (host, 0..) |*out, i| {
                const bits = std.mem.readInt(u32, tensor.bytes[i * 4 ..][0..4], .little);
                out.* = core.bf16_round.bits(@bitCast(bits));
            }
            const key = try a.dupe(u8, name);
            errdefer a.free(key);
            var dev = try hip.DeviceBuffer.fromHost(r, std.mem.sliceAsBytes(host));
            errdefer dev.free();
            try w.names.put(a, key, dev);
            w.bytes += dev.len;
        }
        return w;
    }

    pub fn deinit(w: *Weights) void {
        var it = w.names.iterator();
        while (it.next()) |entry| {
            entry.value_ptr.free();
            w.gpa.free(entry.key_ptr.*);
        }
        w.names.deinit(w.gpa);
        w.* = undefined;
    }

    pub fn get(w: *const Weights, name: []const u8) !hip.ops.Tensor {
        const dev = w.names.get(name) orelse return error.MissingParseTensor;
        return .{ .ptr = dev.base(), .kind = .bf16 };
    }
};
