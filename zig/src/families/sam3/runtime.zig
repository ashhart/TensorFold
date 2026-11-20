const std = @import("std");
const gpu = @import("tensor_metal");
const st = @import("safetensors");
const Store = @import("checkpoint.zig").Store;
const Observer = @import("observe.zig").Observer;
const Config = @import("variant.zig").Config;

pub const Tensor = gpu.Tensor;
pub const DType = gpu.DType;
pub const Precision = enum { fp32, mixed };

pub const Routes = struct {
    precision: Precision = .mixed,
    split_attention: bool = false,
    device_projections: bool = false,
    device_attention: bool = false,
    compensated_norm: bool = false,
};

pub const FeatureMap = struct {
    tensor: Tensor,
    height: usize,
    width: usize,
    channels: usize,

    pub fn of(tensor: Tensor, height: usize, width: usize, channels: usize) FeatureMap {
        return .{ .tensor = tensor, .height = height, .width = width, .channels = channels };
    }
};

pub fn part(tensor: Tensor, first: usize, count: usize) Tensor {
    std.debug.assert(first + count <= tensor.count);
    return .{
        .buffer = tensor.buffer,
        .count = count,
        .dtype = tensor.dtype,
        .tensor_operand_exact = tensor.tensor_operand_exact,
        .owned = false,
        .offset = tensor.offset + first * tensor.dtype.size(),
        .host_data = tensor.host_data,
    };
}

pub const Mark = struct { held: usize };

fn same(a: Tensor, b: Tensor) bool {
    const left = a.host_data orelse a.buffer.contents();
    const right = b.host_data orelse b.buffer.contents();
    return left == right and a.offset == b.offset;
}

fn kept(keep: []const Tensor, tensor: Tensor) bool {
    for (keep) |candidate| if (same(candidate, tensor)) return true;
    return false;
}

fn seconds() f64 {
    return gpu.mtl.clock.seconds();
}

pub const Runtime = struct {
    gpu: *gpu.Gpu,
    store: *Store,
    scratch: std.mem.Allocator,
    config: Config,
    routes: Routes = .{},
    observer: ?Observer = null,
    trunk_warm: bool = false,
    held: std.ArrayList(Tensor) = .empty,
    constants: std.StringHashMapUnmanaged(Tensor) = .empty,

    pub fn deinit(self: *Runtime) void {
        std.debug.assert(self.gpu.batch == null);
        const allocator = self.gpu.allocator;
        self.rewind(.{ .held = 0 }, &.{});
        self.held.deinit(allocator);
        var entries = self.constants.iterator();
        while (entries.next()) |entry| {
            entry.value_ptr.deinit();
            allocator.free(entry.key_ptr.*);
        }
        self.constants.deinit(allocator);
        self.* = undefined;
    }

    pub fn mark(self: *const Runtime) Mark {
        return .{ .held = self.held.items.len };
    }

    pub fn rewind(self: *Runtime, to: Mark, keep: []const Tensor) void {
        std.debug.assert(to.held <= self.held.items.len);
        var next = to.held;
        for (self.held.items[to.held..]) |tensor| {
            if (kept(keep, tensor)) {
                self.held.items[next] = tensor;
                next += 1;
            } else tensor.deinit();
        }
        self.held.shrinkRetainingCapacity(next);
    }

    pub fn release(self: *Runtime, tensor: Tensor) void {
        for (self.held.items, 0..) |candidate, i| if (same(candidate, tensor)) {
            candidate.deinit();
            _ = self.held.orderedRemove(i);
            return;
        };
        @panic("released a SAM3 tensor the runtime does not hold");
    }

    pub fn hold(self: *Runtime, tensor: Tensor) !Tensor {
        self.held.append(self.gpu.allocator, tensor) catch |err| {
            tensor.deinit();
            return err;
        };
        return tensor;
    }

    pub fn hostTensor(self: *Runtime, count: usize) !Tensor {
        return self.hold(try self.gpu.alloc(count, .f32));
    }

    pub fn deviceTensor(self: *Runtime, count: usize, dtype: DType) !Tensor {
        return self.hold(try self.gpu.allocDevice(count, dtype));
    }

    pub fn dedicatedTensor(self: *Runtime, count: usize, dtype: DType) !Tensor {
        return self.hold(try self.gpu.allocDedicated(count, dtype));
    }

    pub fn upload(self: *Runtime, values: []const f32) !Tensor {
        return self.hold(try self.gpu.upload(values, .f32));
    }

    pub fn constant(self: *const Runtime, name: []const u8) ?Tensor {
        var found = self.constants.get(name) orelse return null;
        found.owned = false;
        return found;
    }

    pub fn keepConstant(self: *Runtime, name: []const u8, held_tensor: Tensor) !Tensor {
        std.debug.assert(self.gpu.batch == null);
        const fresh = try self.newConstant(held_tensor.count, held_tensor.dtype);
        const bytes = held_tensor.count * held_tensor.dtype.size();
        @memcpy(fresh.slice(u8, bytes), held_tensor.slice(u8, bytes));
        const borrowed = self.adoptConstant(name, fresh) catch |err| {
            fresh.deinit();
            return err;
        };
        self.release(held_tensor);
        return borrowed;
    }

    pub fn newConstant(self: *Runtime, count: usize, dtype: DType) !Tensor {
        const limit = self.gpu.buffer_cache_limit;
        self.gpu.buffer_cache_limit = 0;
        defer self.gpu.buffer_cache_limit = limit;
        return self.gpu.allocDevice(count, dtype);
    }

    pub fn adoptConstant(self: *Runtime, name: []const u8, tensor: Tensor) !Tensor {
        return self.register(name, tensor);
    }

    pub fn shareConstant(self: *Runtime, name: []const u8, existing: []const u8) !Tensor {
        var shared = self.constants.get(existing) orelse @panic("shared an unknown SAM3 constant");
        shared.owned = false;
        return self.register(name, shared);
    }

    fn register(self: *Runtime, name: []const u8, tensor: Tensor) !Tensor {
        const allocator = self.gpu.allocator;
        if (self.constants.contains(name)) @panic("registered a SAM3 constant name twice");
        const key = try allocator.dupe(u8, name);
        errdefer allocator.free(key);
        try self.constants.put(allocator, key, tensor);
        var borrowed = tensor;
        borrowed.owned = false;
        return borrowed;
    }

    pub fn findConstant(self: *const Runtime, suffix: []const u8, bytes: []const u8) ?[]const u8 {
        std.debug.assert(self.gpu.batch == null);
        var entries = self.constants.iterator();
        while (entries.next()) |entry| {
            const tensor = entry.value_ptr.*;
            if (!std.mem.endsWith(u8, entry.key_ptr.*, suffix)) continue;
            if (tensor.count * tensor.dtype.size() != bytes.len) continue;
            if (std.mem.eql(u8, tensor.slice(u8, bytes.len), bytes)) return entry.key_ptr.*;
        }
        return null;
    }

    pub fn print(self: *Runtime, comptime fmt: []const u8, args: anytype) ![]const u8 {
        return std.fmt.allocPrint(self.scratch, fmt, args);
    }

    pub fn param(self: *Runtime, name: []const u8) !Tensor {
        return self.paramAs(name, .f32);
    }

    pub fn paramAs(self: *Runtime, name: []const u8, dtype: DType) !Tensor {
        const started = seconds();
        defer self.store.parameter_prepare_seconds += seconds() - started;
        const source = self.store.get(name);
        self.store.checkpoint_read_seconds += seconds() - started;
        return self.hold(try self.gpu.weight(try source, dtype));
    }

    pub fn info(self: *Runtime, name: []const u8) !st.Tensor {
        return self.store.get(name);
    }

    pub fn has(self: *const Runtime, name: []const u8) bool {
        return self.store.has(name);
    }

    pub fn observe(self: *Runtime, label: []const u8, tensor: Tensor, shape: []const usize) !void {
        const observer = self.observer orelse return;
        const open = self.gpu.batch != null;
        if (open) try self.gpu.endBatch();
        try observer.write(observer.context, label, tensor, shape);
        if (open) try self.gpu.beginBatch();
    }
};

pub const Scope = struct {
    rt: *Runtime,
    path: []const u8,

    pub fn root(rt: *Runtime, path: []const u8) Scope {
        return .{ .rt = rt, .path = path };
    }

    pub fn sub(self: Scope, name: []const u8) !Scope {
        return .{ .rt = self.rt, .path = try self.rt.print("{s}.{s}", .{ self.path, name }) };
    }

    pub fn item(self: Scope, name: []const u8, index: usize) !Scope {
        return .{ .rt = self.rt, .path = try self.rt.print("{s}.{s}.{d}", .{ self.path, name, index }) };
    }

    pub fn key(self: Scope, leaf: []const u8) ![]const u8 {
        return self.rt.print("{s}.{s}", .{ self.path, leaf });
    }

    pub fn param(self: Scope, leaf: []const u8) !Tensor {
        return self.rt.param(try self.key(leaf));
    }

    pub fn info(self: Scope, leaf: []const u8) !st.Tensor {
        return self.rt.info(try self.key(leaf));
    }

    pub fn has(self: Scope, leaf: []const u8) bool {
        var buffer: [512]u8 = undefined;
        const name = std.fmt.bufPrint(&buffer, "{s}.{s}", .{ self.path, leaf }) catch return false;
        return self.rt.has(name);
    }
};

const testing = std.testing;

fn hostView(bytes: []u8, count: usize) Tensor {
    return .{ .buffer = undefined, .count = count, .dtype = .f32, .owned = false, .host_data = bytes.ptr };
}

fn testRuntime(device: *gpu.Gpu, arena: std.mem.Allocator) Runtime {
    device.allocator = testing.allocator;
    device.batch = null;
    return .{ .gpu = device, .store = undefined, .scratch = arena, .config = Config.of(.sam31) };
}

test "SAM scope keys join path segments with dots" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var device: gpu.Gpu = undefined;
    var rt = testRuntime(&device, arena.allocator());
    const scope = try (try Scope.root(&rt, "a").sub("b")).item("c", 3);
    try testing.expectEqualStrings("a.b.c.3", scope.path);
    try testing.expectEqualStrings("a.b.c.3.w", try scope.key("w"));
    const long: [600]u8 = @splat('x');
    try testing.expect(!scope.has(&long));
}

test "SAM part pieces keep buffer, dtype and flags at an element offset" {
    var bytes: [64]u8 align(4) = undefined;
    var whole = hostView(&bytes, 16);
    whole.offset = 8;
    whole.tensor_operand_exact = true;
    whole.dtype = .f16;
    const piece = part(whole, 3, 5);
    try testing.expectEqual(@as(usize, 14), piece.offset);
    try testing.expectEqual(@as(usize, 5), piece.count);
    try testing.expectEqual(DType.f16, piece.dtype);
    try testing.expect(piece.tensor_operand_exact);
    try testing.expect(!piece.owned and !piece.guarded and piece.recycle == null);
    try testing.expectEqual(whole.host_data, piece.host_data);
}

test "SAM rewind keeps listed tensors in order and release preserves order" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var device: gpu.Gpu = undefined;
    var rt = testRuntime(&device, arena.allocator());
    defer rt.held.deinit(testing.allocator);
    var bytes: [64]u8 align(4) = undefined;
    const base = hostView(&bytes, 16);
    var pieces: [5]Tensor = undefined;
    for (&pieces, 0..) |*piece, i| piece.* = try rt.hold(part(base, i, 1));
    try testing.expect(same(pieces[1], part(base, 1, 1)));
    try testing.expect(!same(pieces[1], pieces[2]));
    rt.release(pieces[1]);
    try testing.expectEqual(@as(usize, 4), rt.held.items.len);
    try testing.expect(same(rt.held.items[1], pieces[2]));
    const point = rt.mark();
    for (pieces[0..3]) |piece| _ = try rt.hold(piece);
    rt.rewind(point, &.{ pieces[2], pieces[0] });
    try testing.expectEqual(@as(usize, 6), rt.held.items.len);
    try testing.expect(same(rt.held.items[4], pieces[0]));
    try testing.expect(same(rt.held.items[5], pieces[2]));
    rt.rewind(.{ .held = 0 }, &.{});
    try testing.expectEqual(@as(usize, 0), rt.held.items.len);
}
