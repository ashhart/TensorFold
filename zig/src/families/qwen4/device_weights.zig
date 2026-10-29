//! Backend-neutral placement keeps lexical tables mapped and uploads each projection in its stored weight format.
const std = @import("std");
const cfg = @import("config.zig");
const schema = @import("weights.zig");
const Reader = @import("reader.zig").Reader;
const wo = @import("core").weight_only;
const st = @import("core").safetensors;
const quant = @import("core").quant;

pub fn centeredNorm(name: []const u8) bool {
    for ([_][]const u8{ ".hc_norm.weight", ".q_norm.weight", ".k_norm.weight", ".q_layernorm.weight", ".k_layernorm.weight", ".norm_key.weight", ".norm_query.weight", ".norm_conv.weight" }) |tail|
        if (std.mem.endsWith(u8, name, tail)) return true;
    return false;
}

/// B.upload finishes its host copy before returning. The owner drains all inference work before releasing buffers.
pub fn Placed(comptime B: type) type {
    return struct {
        const Self = @This();
        pub const Matrix = struct {
            weight: B.Buffer,
            scales: B.Buffer,
            biases: B.Buffer,
            params: wo.Params,
        };
        a: std.mem.Allocator,
        storage: std.heap.ArenaAllocator,
        backend: *B,
        reader: *Reader,
        c: *const cfg.Config,
        matrices: std.StringHashMapUnmanaged(Matrix) = .empty,
        vectors: std.StringHashMapUnmanaged(B.Buffer) = .empty,
        allocations: std.ArrayList(B.Buffer) = .empty,
        dummy: B.Buffer = undefined,
        bytes: u64 = 0,
        dry: bool = true,
        pub fn measure(a: std.mem.Allocator, reader: *Reader, c: *const cfg.Config, problem: *cfg.Problem) !u64 {
            var out: Self = .{ .a = a, .storage = .init(a), .backend = undefined, .reader = reader, .c = c };
            defer out.storage.deinit();
            try out.addBytes(4);
            try schema.walk(c, reader.namespace, &out, problem);
            return out.bytes;
        }
        pub fn open(a: std.mem.Allocator, backend: *B, reader: *Reader, c: *const cfg.Config, budget: u64, problem: *cfg.Problem) !Self {
            var out: Self = .{ .a = a, .storage = .init(a), .backend = backend, .reader = reader, .c = c };
            errdefer out.drop();
            // The whole path is counted before the first device allocation, including widened scalar vectors.
            try out.addBytes(4);
            try schema.walk(c, reader.namespace, &out, problem);
            if (out.bytes > budget) return error.Qwen4WeightsDoNotFit;
            const expected = out.bytes;
            out.bytes = 0;
            out.dry = false;
            out.dummy = try out.put(&.{ 0, 0, 0, 0 });
            try schema.walk(c, reader.namespace, &out, problem);
            if (out.bytes != expected) return error.Qwen4PlacementChanged;
            return out;
        }
        fn addBytes(self: *Self, len: usize) !void {
            self.bytes = try std.math.add(u64, self.bytes, try B.allocationBytes(len));
        }
        fn put(self: *Self, bytes: []const u8) !B.Buffer {
            try self.addBytes(bytes.len);
            const buffer = try self.backend.upload(bytes);
            errdefer self.backend.release(buffer);
            try self.allocations.append(self.a, buffer);
            return buffer;
        }
        fn tensor(self: *Self, name: []const u8, tail: []const u8) !st.Tensor {
            var buf: [320]u8 = undefined;
            return self.reader.get(try std.fmt.bufPrint(&buf, "{s}.{s}", .{ name, tail }));
        }
        fn raw(self: *Self, name: []const u8, tail: []const u8) !B.Buffer {
            const t = try self.tensor(name, tail);
            if (self.dry) {
                try self.addBytes(t.bytes.len);
                return undefined;
            }
            return self.put(t.bytes);
        }
        pub fn take(self: *Self, name: []const u8, want: schema.Want) !void {
            if (want.role == .integer_vector) return;
            if (want.role == .embedding and !(self.c.tie_embeddings and std.mem.endsWith(u8, name, "embed_tokens"))) return;
            const mem = self.storage.allocator();
            if (want.role != .matrix and want.role != .embedding) {
                const t = try self.reader.get(name);
                const count = t.bytes.len / t.dtype.size();
                const bytes = try std.math.mul(usize, count, 4);
                if (self.dry) return self.addBytes(bytes);
                const widened = try mem.alloc(f32, count);
                for (widened, 0..) |*v, i| v.* = switch (t.dtype) {
                    .bf16 => @bitCast(@as(u32, std.mem.readInt(u16, t.bytes[2 * i ..][0..2], .little)) << 16),
                    .f32 => @bitCast(std.mem.readInt(u32, t.bytes[4 * i ..][0..4], .little)),
                    else => return error.Qwen4VectorStorage,
                };
                // These HF names store gamma, while the shared native kernels read the prepared F32 multiplier.
                if (centeredNorm(name) and self.c.norm_weights == .hf_gamma) for (widened) |*v| {
                    v.* = 1 + v.*;
                };
                const key = try mem.dupe(u8, name);
                try self.vectors.put(mem, key, try self.put(std.mem.sliceAsBytes(widened)));
                return;
            }
            const recipe = want.recipe;
            const scale = switch (recipe.format) {
                .bf16 => self.dummy,
                .mlx4, .mlx8 => try self.raw(name, "scales"),
                .fp8 => self.dummy,
                .nvfp4 => try self.raw(name, "weight_scale"),
            };
            const biases = if (recipe.format == .mlx4 or recipe.format == .mlx8) try self.raw(name, "biases") else self.dummy;
            const weight = try self.raw(name, "weight");
            if (self.dry) return;
            const kind: wo.Scale = if (recipe.format == .mlx4 or recipe.format == .mlx8) switch ((try self.tensor(name, "scales")).dtype) {
                .bf16 => .bf16,
                .f16 => .f16,
                .f32 => .f32,
                else => return error.Qwen4ScaleStorage,
            } else .bf16;
            const global: f32 = if (recipe.format == .fp8 or recipe.format == .nvfp4) blk: {
                const t = try self.tensor(name, if (recipe.format == .fp8) "weight_scale" else "weight_scale_2");
                break :blk @bitCast(std.mem.readInt(u32, t.bytes[0..4], .little));
            } else 1;
            const p: wo.Params = .{
                .rows = 1,
                .n = @intCast(want.shape.dims[0]),
                .k = @intCast(want.shape.dims[1]),
                .format = @fromBackingInt(@intCast(@backingInt(recipe.format))),
                .group = if (recipe.format == .mlx4 or recipe.format == .mlx8) @intCast(recipe.group) else 64,
                .scale_kind = kind,
                .global_scale = global,
            };
            _ = try wo.validate(p);
            try self.matrices.put(mem, try mem.dupe(u8, name), .{ .weight = weight, .scales = scale, .biases = biases, .params = p });
        }
        /// The checkpoint's row is a lookup, not a projection: only the selected row is decoded and staged.
        pub fn lookup(self: *Self, name: []const u8, id: usize, out: []u16, problem: *cfg.Problem) !void {
            const w = try self.tensor(name, "weight");
            if (w.rank != 2) return error.Qwen4LookupShape;
            if (id >= w.shape[0]) return error.TokenOutOfRange;
            const recipe = try self.c.quant.of(name, problem);
            const width = out.len;
            const stored_width = switch (recipe.format) {
                .bf16, .fp8 => w.shape[1],
                .mlx4 => try std.math.mul(usize, w.shape[1], 8),
                .mlx8 => try std.math.mul(usize, w.shape[1], 4),
                .nvfp4 => try std.math.mul(usize, w.shape[1], 2),
            };
            if (width != stored_width) return error.Qwen4LookupShape;
            if (recipe.format == .bf16) {
                if (w.rank != 2 or w.shape[1] != width) return error.Qwen4LookupShape;
                for (out, 0..) |*v, i| v.* = std.mem.readInt(u16, w.bytes[2 * (id * width + i) ..][0..2], .little);
                return;
            }
            const bits: usize = if (recipe.format == .mlx4) 4 else 8;
            const s = if (recipe.format == .mlx4 or recipe.format == .mlx8) try self.tensor(name, "scales") else try self.tensor(name, "weight_scale");
            const b: ?st.Tensor = if (recipe.format == .mlx4 or recipe.format == .mlx8) try self.tensor(name, "biases") else null;
            const global: f32 = if (recipe.format == .fp8 or recipe.format == .nvfp4) blk: {
                const t = try self.tensor(name, if (recipe.format == .fp8) "weight_scale" else "weight_scale_2");
                break :blk @bitCast(std.mem.readInt(u32, t.bytes[0..4], .little));
            } else 1;
            for (out, 0..) |*value, i| {
                const at = id * width + i;
                const decoded: f32 = switch (recipe.format) {
                    .bf16 => unreachable,
                    .mlx4, .mlx8 => blk: {
                        const code = (w.bytes[at * bits / 8] >> @as(u3, @intCast((at * bits) % 8))) & (if (bits == 4) @as(u8, 15) else 255);
                        const g = id * (width / recipe.group) + i / recipe.group;
                        break :blk @mulAdd(f32, @floatFromInt(code), scalar(s, g), scalar(b.?, g));
                    },
                    .fp8 => @as(f32, @floatCast(quant.fp8.e4m3(w.bytes[at]))) * global,
                    .nvfp4 => blk: {
                        const code: u4 = @truncate(w.bytes[at / 2] >> @as(u3, @intCast((at % 2) * 4)));
                        break :blk (@as(f32, @floatCast(quant.nvfp4.e2m1(code))) * @as(f32, @floatCast(quant.fp8.e4m3(s.bytes[id * (width / 16) + i / 16])))) * global;
                    },
                };
                value.* = quant.convert.bf16(decoded);
            }
        }
        /// A failed drain retains every allocation and mapping owner for a retry or explicit engine teardown.
        pub fn close(self: *Self) !void {
            try self.backend.drain();
            self.drop();
        }
        /// Only an owner that has proved completion, or has queued no work, may release the borrowed views.
        pub fn deinitDrained(self: *Self) void {
            self.drop();
        }
        fn drop(self: *Self) void {
            for (self.allocations.items) |buffer| self.backend.release(buffer);
            self.allocations.deinit(self.a);
            self.storage.deinit();
            self.* = undefined;
        }
    };
}
fn scalar(t: st.Tensor, at: usize) f32 {
    return switch (t.dtype) {
        .bf16 => @bitCast(@as(u32, std.mem.readInt(u16, t.bytes[2 * at ..][0..2], .little)) << 16),
        .f16 => @as(f16, @bitCast(std.mem.readInt(u16, t.bytes[2 * at ..][0..2], .little))),
        .f32 => @bitCast(std.mem.readInt(u32, t.bytes[4 * at ..][0..4], .little)),
        else => unreachable,
    };
}
