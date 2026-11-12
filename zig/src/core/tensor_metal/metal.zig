const std = @import("std");
pub const mtl = @import("metal");
const sources = @import("kernel_sources");
const st = @import("safetensors");
const rounding = @import("bf16_round");
pub const ops = @import("tensor_ops.zig");
extern "c" fn getpagesize() c_int;

pub const DType = enum(u32) {
    bf16 = 0,
    f32 = 1,
    f16 = 2,

    pub fn size(self: DType) usize {
        return if (self == .f32) 4 else 2;
    }
};

pub const Tensor = struct {
    buffer: mtl.Buffer,
    count: usize,
    dtype: DType,
    // FP32 values expanded from FP16/BF16 have no operand bits for the tensor unit to discard.
    tensor_operand_exact: bool = false,
    guarded: bool = false,
    owned: bool = true,
    offset: usize = 0,
    host_data: ?[*]u8 = null,
    recycle: ?struct { owner: *Gpu, index: usize } = null,

    pub fn slice(self: Tensor, comptime T: type, count: usize) []T {
        const base = self.host_data orelse self.buffer.contents();
        return @as([*]T, @ptrCast(@alignCast(base + self.offset)))[0..count];
    }

    pub fn deinit(self: Tensor) void {
        if (!self.owned) return;
        if (self.guarded) {
            const offset = self.count * self.dtype.size();
            for (self.buffer.contents()[offset..][0..256]) |byte| if (byte != 0xa5) @panic("GPU tensor guard overwritten");
        }
        if (self.recycle) |lease| {
            const slot = &lease.owner.spares.items[lease.index];
            std.debug.assert(!slot.free);
            slot.free = true;
            slot.ready = lease.owner.submissions + @as(usize, @intFromBool(lease.owner.batch != null));
        } else self.buffer.deinit();
    }

    pub fn get(self: Tensor, index: usize) f32 {
        return switch (self.dtype) {
            .bf16 => @bitCast(@as(u32, self.slice(u16, self.count)[index]) << 16),
            .f32 => self.slice(f32, self.count)[index],
            .f16 => @floatCast(self.slice(f16, self.count)[index]),
        };
    }

    pub fn checkFinite(self: Tensor, name: []const u8) !void {
        const finite = @import("finite.zig");
        const bad = switch (self.dtype) {
            .f32 => finite.firstNonFinite(f32, self.slice(f32, self.count)),
            .bf16 => finite.firstNonFinite(u16, self.slice(u16, self.count)),
            .f16 => finite.firstNonFinite(f16, self.slice(f16, self.count)),
        };
        if (bad) |i| return nonFinite(name, i, self.count, self.get(i));
    }

    fn nonFinite(name: []const u8, index: usize, count: usize, value: f32) error{NonFiniteTensor} {
        std.log.err("nonfinite {s} at {d}/{d}: {d}", .{ name, index, count, value });
        return error.NonFiniteTensor;
    }
};

pub const Gpu = struct {
    pub const Pass = struct { cb: mtl.CommandBuffer, enc: mtl.ComputeEncoder };
    const WeightKey = struct { address: usize, count: usize, source: st.DType, target: DType };
    const Spare = struct { buffer: mtl.Buffer, bytes: usize, free: bool = false, ready: usize = 0 };
    allocator: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    library: mtl.Library,
    exact_library: ?mtl.Library = null,
    pipelines: std.StringHashMapUnmanaged(mtl.Pipeline) = .empty,
    gpu_seconds: f64 = 0,
    wait_seconds: f64 = 0,
    allocation_seconds: f64 = 0,
    allocations: usize = 0,
    submissions: usize = 0,
    buffer_cache_limit: usize = 0,
    buffer_cache_bytes: usize = 0,
    buffer_reuses: usize = 0,
    spares: std.ArrayList(Spare) = .empty,
    peak_bytes: u64 = 0,
    guards: bool = false,
    cache_weights: bool = false,
    map_weights: bool = false,
    weight_cache: std.AutoHashMapUnmanaged(WeightKey, Tensor) = .empty,
    batch: ?struct { pass: Pass, pool: mtl.objc.Pool } = null,

    pub fn init(allocator: std.mem.Allocator, extra_source: []const u8) !Gpu {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const device = try mtl.Device.init();
        errdefer device.deinit();
        if (!device.tensorUnits()) return error.RequiresM5TensorUnits;
        const queue = try device.queue();
        errdefer queue.deinit();
        const attention_source = try attentionSource(allocator);
        defer allocator.free(attention_source);
        const combined = try std.mem.concat(allocator, u8, &.{
            "#define TF_NAX_STRICT_FP32\n", sources.core_qwen_vision, sources.tensor_attention_pv, attention_source, sources.tensor_ops, sources.tensor_attention, extra_source,
        });
        defer allocator.free(combined);
        const source = try std.mem.replaceOwned(u8, allocator, combined, "#include \"../nax.h\"", "\n#ifndef TF_TENSOR_NAX_INCLUDED\n#define TF_TENSOR_NAX_INCLUDED\n" ++ sources.nax ++ "\n#endif\n");
        defer allocator.free(source);
        return .{ .allocator = allocator, .device = device, .queue = queue, .library = try mtl.Library.fromSource(device, source, mtl.CompileOptions.mlx()) };
    }

    pub fn deinit(self: *Gpu) void {
        if (self.batch != null) self.endBatch() catch {};
        self.clearWeights();
        for (self.spares.items) |slot| slot.buffer.deinit();
        self.spares.deinit(self.allocator);
        var it = self.pipelines.iterator();
        while (it.next()) |entry| {
            entry.value_ptr.deinit();
            self.allocator.free(entry.key_ptr.*);
        }
        self.pipelines.deinit(self.allocator);
        self.library.deinit();
        if (self.exact_library) |library| library.deinit();
        self.queue.deinit();
        self.device.deinit();
    }

    pub fn alloc(self: *Gpu, count: usize, dtype: DType) !Tensor {
        return self.allocBuffer(count, dtype, false);
    }

    // Only encoder writes may initialize this buffer until the current serial batch completes.
    pub fn allocDevice(self: *Gpu, count: usize, dtype: DType) !Tensor {
        return self.allocBuffer(count, dtype, true);
    }

    fn allocBuffer(self: *Gpu, count: usize, dtype: DType, device_only: bool) !Tensor {
        if (count == 0) return error.EmptyTensor;
        const bytes = try std.math.mul(usize, count, dtype.size());
        const capacity = bytes + @as(usize, if (self.guards) 256 else 0);
        const started = mtl.clock.seconds();
        if (self.buffer_cache_limit > 0) {
            var best: ?usize = null;
            for (self.spares.items, 0..) |slot, i| {
                const ready = self.submissions + @as(usize, @intFromBool(device_only and self.batch != null and !self.guards));
                if (!slot.free or slot.ready > ready or slot.bytes < capacity or (self.guards and slot.bytes != capacity)) continue;
                if (best == null or slot.bytes < self.spares.items[best.?].bytes) best = i;
                if (slot.bytes == capacity) break;
            }
            if (best) |i| {
                const slot = &self.spares.items[i];
                slot.free = false;
                self.buffer_reuses += 1;
                self.allocation_seconds += mtl.clock.seconds() - started;
                return .{ .buffer = slot.buffer, .count = count, .dtype = dtype, .guarded = self.guards, .host_data = slot.buffer.contents(), .recycle = .{ .owner = self, .index = i } };
            }
        }
        const buffer = try self.device.buffer(capacity, mtl.ResourceOptions.shared);
        errdefer buffer.deinit();
        self.allocation_seconds += mtl.clock.seconds() - started;
        self.allocations += 1;
        if (self.guards) @memset(buffer.contents()[bytes..][0..256], 0xa5);
        self.peak_bytes = @max(self.peak_bytes, self.device.allocated());
        var out = Tensor{ .buffer = buffer, .count = count, .dtype = dtype, .guarded = self.guards, .host_data = buffer.contents() };
        if (capacity <= self.buffer_cache_limit -| self.buffer_cache_bytes) {
            try self.spares.append(self.allocator, .{ .buffer = buffer, .bytes = capacity });
            self.buffer_cache_bytes += capacity;
            out.recycle = .{ .owner = self, .index = self.spares.items.len - 1 };
        }
        return out;
    }

    pub fn upload(self: *Gpu, data: []const f32, dtype: DType) !Tensor {
        const result = try self.alloc(data.len, dtype);
        switch (dtype) {
            .f32 => @memcpy(result.buffer.slice(f32, data.len), data),
            .bf16 => for (result.buffer.slice(u16, data.len), data) |*out, x| {
                out.* = rounding.bits(x);
            },
            .f16 => for (result.buffer.slice(f16, data.len), data) |*out, x| {
                out.* = @floatCast(x);
            },
        }
        return result;
    }

    pub fn weight(self: *Gpu, src: st.Tensor, dtype: DType) !Tensor {
        const cache_limit = self.buffer_cache_limit;
        self.buffer_cache_limit = 0;
        defer self.buffer_cache_limit = cache_limit;
        const key = WeightKey{ .address = @intFromPtr(src.bytes.ptr), .count = src.numel(), .source = src.dtype, .target = dtype };
        if (self.cache_weights) if (self.weight_cache.get(key)) |cached| {
            var borrowed = cached;
            borrowed.owned = false;
            return borrowed;
        };
        const same = (src.dtype == .bf16 and dtype == .bf16) or (src.dtype == .f32 and dtype == .f32) or (src.dtype == .f16 and dtype == .f16);
        const mapped = if (self.map_weights and @intFromPtr(src.bytes.ptr) % 4 == 0) try self.mapWeight(src) else null;
        defer if (!same) {
            if (mapped) |value| value.deinit();
        };
        var result = if (same and mapped != null) mapped.? else try self.alloc(src.numel(), dtype);
        errdefer result.deinit();
        if (mapped) |source| {
            if (!same) try self.dispatch("im_cast", &.{ source, result }, [3]u32{ @intCast(src.numel()), switch (src.dtype) {
                .bf16 => 0,
                .f32 => 1,
                .f16 => 2,
                else => return error.UnsupportedWeightType,
            }, @backingInt(dtype) }, mtl.Size.of(src.numel(), 1, 1), mtl.Size.of(256, 1, 1));
        } else if (same) {
            @memcpy(result.slice(u8, src.bytes.len), src.bytes);
        } else {
            for (0..result.count) |i| {
                const x = try readWeight(src, i);
                switch (dtype) {
                    .f32 => result.buffer.slice(f32, result.count)[i] = x,
                    .bf16 => result.buffer.slice(u16, result.count)[i] = rounding.bits(x),
                    .f16 => result.buffer.slice(f16, result.count)[i] = @floatCast(x),
                }
            }
        }
        result.tensor_operand_exact = dtype == .f32 and (src.dtype == .f16 or src.dtype == .bf16);
        if (self.cache_weights) {
            try self.weight_cache.put(self.allocator, key, result);
            var borrowed = result;
            borrowed.owned = false;
            return borrowed;
        }
        return result;
    }

    fn mapWeight(self: *Gpu, src: st.Tensor) !Tensor {
        if (src.dtype != .bf16 and src.dtype != .f32 and src.dtype != .f16) return error.UnsupportedWeightType;
        const page: usize = @intCast(getpagesize());
        const address = @intFromPtr(src.bytes.ptr);
        const base = std.mem.alignBackward(usize, address, page);
        const offset = address - base;
        const size = std.mem.alignForward(usize, offset + src.bytes.len, page);
        const buffer = try self.device.bufferNoCopy(@ptrFromInt(base), size, mtl.ResourceOptions.shared);
        self.peak_bytes = @max(self.peak_bytes, self.device.allocated());
        return .{ .buffer = buffer, .count = src.numel(), .dtype = switch (src.dtype) {
            .f32 => .f32,
            .f16 => .f16,
            .bf16 => .bf16,
            else => unreachable,
        }, .offset = offset, .host_data = buffer.contents() };
    }

    pub fn clearWeights(self: *Gpu) void {
        var it = self.weight_cache.valueIterator();
        while (it.next()) |value| value.deinit();
        self.weight_cache.deinit(self.allocator);
        self.weight_cache = .empty;
    }

    pub fn pipeline(self: *Gpu, name: []const u8) !mtl.Pipeline {
        if (self.pipelines.get(name)) |p| return p;
        const key = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(key);
        const library = if (std.mem.startsWith(u8, name, "qwv_gemm_f32_exact_")) blk: {
            if (self.exact_library == null) {
                const source = try std.mem.replaceOwned(u8, self.allocator, sources.tensor_exact_gemm, "#include \"../nax.h\"", sources.nax);
                defer self.allocator.free(source);
                self.exact_library = try mtl.Library.fromSource(self.device, source, mtl.CompileOptions.mlx());
            }
            break :blk self.exact_library.?;
        } else self.library;
        const p = try mtl.Pipeline.init(self.device, library, name, false);
        errdefer p.deinit();
        try self.pipelines.put(self.allocator, key, p);
        return p;
    }

    pub fn finish(self: *Gpu, cb: mtl.CommandBuffer, enc: mtl.ComputeEncoder) !void {
        if (self.batch != null) return;
        enc.end();
        const started = mtl.clock.seconds();
        cb.commit();
        cb.wait();
        self.wait_seconds += mtl.clock.seconds() - started;
        self.submissions += 1;
        if (cb.failure()) |msg| {
            std.log.err("Metal command failed: {s}", .{msg});
            return error.GpuFailure;
        }
        self.gpu_seconds += cb.gpuSeconds();
    }

    pub fn command(self: *Gpu) Pass {
        if (self.batch) |batch| return batch.pass;
        const cb = self.queue.commandBuffer();
        return .{ .cb = cb, .enc = cb.compute(.serial) };
    }

    pub fn beginBatch(self: *Gpu) !void {
        if (self.batch != null) return error.NestedGpuBatch;
        const pool = mtl.objc.Pool.push();
        const cb = self.queue.commandBuffer();
        const enc = cb.compute(.serial);
        self.batch = .{ .pass = .{ .cb = cb, .enc = enc }, .pool = pool };
    }

    pub fn endBatch(self: *Gpu) !void {
        const batch = self.batch orelse return error.NoGpuBatch;
        self.batch = null;
        defer batch.pool.pop();
        try self.finish(batch.pass.cb, batch.pass.enc);
    }

    pub fn dispatch(self: *Gpu, name: []const u8, buffers: []const Tensor, params: anytype, grid: mtl.Size, group: mtl.Size) !void {
        if (grid.width == 0 or grid.height == 0 or grid.depth == 0 or group.width == 0 or group.height == 0 or group.depth == 0) return error.EmptyGpuDispatch;
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const p = try self.pipeline(name);
        const pass = self.command();
        const cb = pass.cb;
        const enc = pass.enc;
        enc.setPipeline(p);
        for (buffers, 0..) |tensor, i| enc.setBuffer(tensor.buffer, tensor.offset, i);
        enc.setValue(params, buffers.len);
        enc.dispatchThreads(grid, group);
        try self.finish(cb, enc);
    }

    pub fn matmul(self: *Gpu, x: Tensor, w: Tensor, m: usize, n: usize, k: usize) !Tensor {
        const out = try self.allocDevice(try std.math.mul(usize, m, n), x.dtype);
        errdefer out.deinit();
        try self.matmulInto(x, w, out, m, n, k);
        return out;
    }

    pub fn matmulInto(self: *Gpu, x: Tensor, w: Tensor, out: Tensor, m: usize, n: usize, k: usize) !void {
        return self.matmulBatchedInto(x, w, out, m, n, k, 1);
    }

    pub fn matmulBatchedInto(self: *Gpu, x: Tensor, w: Tensor, out: Tensor, m: usize, n: usize, k: usize, batch: usize) !void {
        if (x.dtype == .f16) return error.UnsupportedGemmType;
        const xs = try std.math.mul(usize, m, k);
        const ws = try std.math.mul(usize, n, k);
        const ys = try std.math.mul(usize, m, n);
        if (batch == 0 or x.dtype != w.dtype or x.dtype != out.dtype or x.count != try std.math.mul(usize, xs, batch) or
            w.count != try std.math.mul(usize, ws, batch) or out.count != try std.math.mul(usize, ys, batch)) return error.MatrixShape;
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        var name: [64]u8 = undefined;
        const key = try std.fmt.bufPrint(&name, "qwv_gemm_{s}{s}_nt_{d}{d}", .{ @tagName(x.dtype), if (w.dtype == .f32 and w.tensor_operand_exact) "_exact" else "", @intFromBool(m % 64 == 0), @intFromBool(n % 128 == 0) });
        const p = try self.pipeline(key);
        const pass = self.command();
        const cb = pass.cb;
        const enc = pass.enc;
        enc.setPipeline(p);
        enc.setBuffer(x.buffer, x.offset, 0);
        enc.setBuffer(w.buffer, w.offset, 1);
        var args: [16]i32 = @splat(0);
        const tn = (n + 127) / 128;
        const tm = (m + 63) / 64;
        for ([_]usize{ m, n, k, k, k, n, tn, tm, 2, k / 256 }, 0..) |v, i| args[i] = std.math.cast(i32, v) orelse return error.MatrixTooLarge;
        for ([_]usize{ xs, ws, ys }, 10..) |v, i| args[i] = std.math.cast(i32, v) orelse return error.MatrixTooLarge;
        enc.setValue(args, 2);
        enc.setBuffer(out.buffer, out.offset, 3);
        enc.dispatchThreads(mtl.Size.of(tn * 4 * 32, (tm + 3) / 4 * 4, batch * 2), mtl.Size.of(32, 4, 2));
        try self.finish(cb, enc);
    }

    pub fn linear(self: *Gpu, x: Tensor, w: Tensor, bias: Tensor, m: usize, n: usize, k: usize) !Tensor {
        if (x.dtype != w.dtype or x.dtype != bias.dtype or x.count != m * k or w.count != n * k or bias.count != n)
            return error.MatrixShape;
        const product = try self.matmul(x, w, m, n, k);
        defer product.deinit();
        return @import("tensor_ops.zig").point(self, product, bias, 1, n);
    }
};

fn attentionSource(a: std.mem.Allocator) ![]u8 {
    const original = comptime found: {
        for (sources.prefill) |file| if (std.mem.eql(u8, file.name, "attention_nax")) break :found file.text;
        @compileError("TensorFold attention source is missing");
    };
    var source = try a.dupe(u8, original);
    errdefer a.free(source);
    const changes = .{
        .{ "attention_pv(acc[d], acc[d + 1], s[k], v0, v1, ushort(lane));", "if constexpr (DIM >= 128) tensor_attention_pv(acc[d], acc[d + 1], s[k], v0, v1, ushort(lane)); else attention_pv(acc[d], acc[d + 1], s[k], v0, v1, ushort(lane));", 1 },
        .{ "template <typename T, bool AQ, bool AK, bool CAUSAL>", "template <typename T, bool AQ, bool AK, bool CAUSAL, int DIM = 128, int WINDOW = 0, bool TILE_MASK = false>", 1 },
        .{ "constexpr int BQ = 64, BK = 32;", "constexpr int BQ = 64, BK = 32, VDIM = DIM > 128 ? 128 : DIM;\n  const int batch_index = int(tg.z) / (DIM / VDIM), value_part = int(tg.z) % (DIM / VDIM);", 1 },
        .{ "long(tg.z)", "long(batch_index)", 4 },
        .{ "V += long(batch_index) * P[18] + long(kv) * P[19];", "V += long(batch_index) * P[18] + long(kv) * P[19] + value_part * VDIM;", 1 },
        .{ "long(tg.x) * BQ * os;", "long(tg.x) * BQ * os + value_part * VDIM;", 1 },
        .{ "uint3 tg, uint sg, uint lane)", "uint3 tg, uint sg, uint lane, const device int* sizes = nullptr)", 1 },
        .{ "float top_new[2]", "if (TILE_MASK) { for (short f = 0; f < 2; f++) { for (short e = 0; e < 8; e++) { const int col = kb * BK + 16 * f + home.x + TF_COL(e); if (col % 64 >= sizes[col / 64]) s[f][e] = kMasked; } } }\n    float top_new[2]", 1 },
        .{ "if (CAUSAL && kb >= first_masked)", "if (CAUSAL && (WINDOW > 0 || kb >= first_masked))", 1 },
        .{ "(r0 + (e >> 2) * 8) < (c0 + 16 * f + TF_COL(e)) ?", "((r0 + (e >> 2) * 8) < (c0 + 16 * f + TF_COL(e)) || (WINDOW > 0 && (r0 + (e >> 2) * 8) >= (c0 + 16 * f + TF_COL(e)) + WINDOW)) ?", 1 },
        .{ "acc[8]", "acc[VDIM / 16]", 1 },
        .{ "short i = 0; i < 8;", "short i = 0; i < VDIM / 16;", 5 },
        .{ "short d = 0; d < 8; d++)", "short d = 0; d < DIM / 16; d++)", 1 },
        .{ "short d = 0; d < 8; d += 2)", "short d = 0; d < VDIM / 16; d += 2)", 1 },
    };
    inline for (changes) |change| {
        if (std.mem.count(u8, source, change[0]) != change[2]) return error.TensorFoldKernelChanged;
        const next = try std.mem.replaceOwned(u8, a, source, change[0], change[1]);
        a.free(source);
        source = next;
    }
    return source;
}

fn readWeight(tensor: st.Tensor, index: usize) !f32 {
    const bytes = tensor.bytes[index * tensor.dtype.size() ..];
    return switch (tensor.dtype) {
        .f32 => @bitCast(std.mem.readInt(u32, bytes[0..4], .little)),
        .f16 => @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bytes[0..2], .little)))),
        .bf16 => @bitCast(@as(u32, std.mem.readInt(u16, bytes[0..2], .little)) << 16),
        else => error.UnsupportedWeightType,
    };
}
