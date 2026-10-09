//! The Neural Engine through the private AppleNeuralEngine runtime: fp16 MIL programs over IOSurface tensors.
const std = @import("std");
const objc = @import("objc.zig");
const Device = @import("device.zig").Device;
const Buffer = @import("device.zig").Buffer;
const types = @import("types.zig");

const Id = objc.Id;

extern "c" fn dlopen(path: [*:0]const u8, mode: c_int) ?*anyopaque;
extern "c" fn objc_getClass(name: [*:0]const u8) ?objc.Class;
extern "c" fn IOSurfaceCreate(properties: Id) ?*anyopaque;
extern "c" fn IOSurfaceGetBaseAddress(surface: *anyopaque) ?*anyopaque;
extern "c" fn CFRelease(object: *anyopaque) void;
extern "c" const kIOSurfaceWidth: Id;
extern "c" const kIOSurfaceHeight: Id;
extern "c" const kIOSurfaceBytesPerElement: Id;
extern "c" const kIOSurfaceBytesPerRow: Id;
extern "c" const kIOSurfaceAllocSize: Id;
extern "c" const kIOSurfacePixelFormat: Id;

pub const Error = error{ NoNeuralEngine, Surface, Compile, Load, Request, Evaluate };

const qos: u32 = 21;

/// True when the private runtime loads and reports a Neural Engine (checked once).
pub fn available() bool {
    const Once = struct {
        var checked = false;
        var ok = false;
    };
    if (!Once.checked) {
        Once.checked = true;
        Once.ok = dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", 2) != null and
            objc_getClass("_ANEInMemoryModel") != null and objc.msg(bool, objc.class("_ANEDeviceInfo"), "hasANE", .{});
    }
    return Once.ok;
}

fn number(v: u64) Id {
    return objc.msg(Id, objc.class("NSNumber"), "numberWithUnsignedLongLong:", .{v});
}

fn dictionary(keys: []const Id, values: []const Id) Id {
    return objc.msg(Id, objc.class("NSDictionary"), "dictionaryWithObjects:forKeys:count:", .{ values.ptr, keys.ptr, keys.len });
}

fn array(items: []const Id) Id {
    return objc.msg(Id, objc.class("NSArray"), "arrayWithObjects:count:", .{ items.ptr, items.len });
}

/// An fp16 tensor in an IOSurface the Neural Engine reads or writes, and a Metal buffer over the same pages.
pub const Surface = struct {
    ref: *anyopaque,
    object: Id,
    buffer: Buffer,

    pub fn init(device: Device, bytes: usize) !Surface {
        const pool = objc.Pool.push();
        defer pool.pop();
        const len = std.mem.alignForward(usize, bytes, 16384);
        const keys = [_]Id{ kIOSurfaceWidth, kIOSurfaceHeight, kIOSurfaceBytesPerElement, kIOSurfaceBytesPerRow, kIOSurfaceAllocSize, kIOSurfacePixelFormat };
        const values = [_]Id{ number(len), number(1), number(1), number(len), number(len), number(0) };
        const ref = IOSurfaceCreate(dictionary(&keys, &values)) orelse return error.Surface;
        errdefer CFRelease(ref);
        const base = IOSurfaceGetBaseAddress(ref) orelse return error.Surface;
        const buffer = try device.bufferNoCopy(base, len, types.ResourceOptions.shared | types.ResourceOptions.untracked);
        const object = objc.retain(objc.msg(Id, objc.class("_ANEIOSurfaceObject"), "objectWithIOSurface:", .{ref}));
        return .{ .ref = ref, .object = object, .buffer = buffer };
    }

    pub fn deinit(s: Surface) void {
        objc.release(s.object);
        s.buffer.deinit();
        CFRelease(s.ref);
    }
};

/// MIL for y = W x as a 1x1 convolution: x [1, k, 1, s] fp16 planar, W [n, k] at blob offset 64.
pub fn conv(a: std.mem.Allocator, k: usize, n: usize, s: usize) ![]u8 {
    return std.fmt.allocPrint(a,
        \\program(1.3)
        \\[buildInfo = dict<string, string>({{{{"coremlc-component-MIL", "3510.2.1"}}, {{"coremlc-version", "3505.4.1"}}}})]
        \\{{
        \\  func main<ios18>(tensor<fp16, [1, {d}, 1, {d}]> x) {{
        \\    tensor<fp16, [{d}, {d}, 1, 1]> w = const()[name=string("w"), val=tensor<fp16, [{d}, {d}, 1, 1]>(BLOBFILE(path=string("@model_path/weights/weight.bin"), offset=uint64(64)))];
        \\    string pt = const()[name=string("pt"), val=string("valid")];
        \\    tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];
        \\    tensor<int32, [4]> pd = const()[name=string("pd"), val=tensor<int32, [4]>([0,0,0,0])];
        \\    tensor<int32, [2]> dl = const()[name=string("dl"), val=tensor<int32, [2]>([1,1])];
        \\    int32 gr = const()[name=string("gr"), val=int32(1)];
        \\    tensor<fp16, [1, {d}, 1, {d}]> y = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=w, x=x)[name=string("conv")];
        \\  }} -> (y);
        \\}}
        \\
    , .{ k, s, n, k, n, k, n, s });
}

/// An fp16 weight file: a 64-byte header, then per tensor a 64-byte header and its data, 64-aligned.
pub const Blob = struct {
    bytes: []align(16) u8,
    offsets: [4]usize = @splat(0),

    pub fn init(a: std.mem.Allocator, counts: []const usize) !Blob {
        if (counts.len == 0 or counts.len > 4) return error.Compile;
        var b = Blob{ .bytes = undefined };
        var at: usize = 64;
        for (counts, 0..) |n, i| {
            b.offsets[i] = at;
            at = std.mem.alignForward(usize, at + 64 + n * 2, 64);
        }
        b.bytes = try a.alignedAlloc(u8, .@"16", at);
        @memset(b.bytes, 0);
        std.mem.writeInt(u32, b.bytes[0..4], @intCast(counts.len), .little);
        std.mem.writeInt(u32, b.bytes[4..8], 2, .little);
        for (counts, 0..) |n, i| {
            const h = b.bytes[b.offsets[i]..][0..64];
            std.mem.writeInt(u32, h[0..4], 0xDEADBEEF, .little);
            std.mem.writeInt(u32, h[4..8], 1, .little);
            std.mem.writeInt(u64, h[8..16], n * 2, .little);
            std.mem.writeInt(u64, h[16..24], b.offsets[i] + 64, .little);
        }
        return b;
    }

    /// Tensor i's fp16 values, `n` of them.
    pub fn values(b: Blob, i: usize, n: usize) []f16 {
        return @alignCast(std.mem.bytesAsSlice(f16, b.bytes[b.offsets[i] + 64 ..][0 .. n * 2]));
    }

    pub fn deinit(b: Blob, a: std.mem.Allocator) void {
        a.free(b.bytes);
    }
};

/// MIL for one MLP column slice: gate/up = W1 x, silu(gate) * up, y = W2 h, all fp16 [1, h, 1, s].
pub fn mlpSlice(a: std.mem.Allocator, h: usize, c: usize, s: usize, w1: usize, w2: usize) ![]u8 {
    return std.fmt.allocPrint(a,
        \\program(1.3)
        \\[buildInfo = dict<string, string>({{{{"coremlc-component-MIL", "3510.2.1"}}, {{"coremlc-version", "3505.4.1"}}}})]
        \\{{
        \\  func main<ios18>(tensor<fp16, [1, {[h]d}, 1, {[s]d}]> x) {{
        \\    tensor<fp16, [{[c2]d}, {[h]d}, 1, 1]> w1 = const()[name=string("w1"), val=tensor<fp16, [{[c2]d}, {[h]d}, 1, 1]>(BLOBFILE(path=string("@model_path/weights/weight.bin"), offset=uint64({[w1]d})))];
        \\    tensor<fp16, [{[h]d}, {[c]d}, 1, 1]> w2 = const()[name=string("w2"), val=tensor<fp16, [{[h]d}, {[c]d}, 1, 1]>(BLOBFILE(path=string("@model_path/weights/weight.bin"), offset=uint64({[w2]d})))];
        \\    string pt = const()[name=string("pt"), val=string("valid")];
        \\    tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];
        \\    tensor<int32, [4]> pd = const()[name=string("pd"), val=tensor<int32, [4]>([0,0,0,0])];
        \\    tensor<int32, [2]> dl = const()[name=string("dl"), val=tensor<int32, [2]>([1,1])];
        \\    int32 gr = const()[name=string("gr"), val=int32(1)];
        \\    tensor<fp16, [1, {[c2]d}, 1, {[s]d}]> gu = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=w1, x=x)[name=string("gu")];
        \\    tensor<int32, [4]> b0 = const()[name=string("b0"), val=tensor<int32, [4]>([0,0,0,0])];
        \\    tensor<int32, [4]> e0 = const()[name=string("e0"), val=tensor<int32, [4]>([1,{[c]d},1,{[s]d}])];
        \\    tensor<int32, [4]> b1 = const()[name=string("b1"), val=tensor<int32, [4]>([0,{[c]d},0,0])];
        \\    tensor<int32, [4]> e1 = const()[name=string("e1"), val=tensor<int32, [4]>([1,{[c2]d},1,{[s]d}])];
        \\    tensor<fp16, [1, {[c]d}, 1, {[s]d}]> g = slice_by_index(x=gu, begin=b0, end=e0)[name=string("g")];
        \\    tensor<fp16, [1, {[c]d}, 1, {[s]d}]> u = slice_by_index(x=gu, begin=b1, end=e1)[name=string("u")];
        \\    tensor<fp16, [1, {[c]d}, 1, {[s]d}]> sg = silu(x=g)[name=string("sg")];
        \\    tensor<fp16, [1, {[c]d}, 1, {[s]d}]> hh = mul(x=sg, y=u)[name=string("hh")];
        \\    tensor<fp16, [1, {[h]d}, 1, {[s]d}]> y = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=w2, x=hh)[name=string("y")];
        \\  }} -> (y);
        \\}}
        \\
    , .{ .h = h, .s = s, .c = c, .c2 = 2 * c, .w1 = w1, .w2 = w2 });
}

/// A compiled, loaded program.
pub const Program = struct {
    model: Id,

    /// Loads `mil` with its weights, compiling unless the Neural Engine's cache holds it; the work dir goes after.
    pub fn init(mil: []const u8, weights: []const u8) !Program {
        if (!available()) return error.NoNeuralEngine;
        const pool = objc.Pool.push();
        defer pool.pop();
        const text = objc.msg(Id, objc.class("NSData"), "dataWithBytes:length:", .{ mil.ptr, mil.len });
        const data = objc.msg(Id, objc.class("NSData"), "dataWithBytes:length:", .{ weights.ptr, weights.len });
        const key_offset = objc.string("offset");
        defer objc.release(key_offset);
        const key_data = objc.string("data");
        defer objc.release(key_data);
        const path = objc.string("@model_path/weights/weight.bin");
        defer objc.release(path);
        const files = dictionary(&.{path}, &.{dictionary(&.{ key_offset, key_data }, &.{ number(0), data })});
        const desc = objc.msg(?Id, objc.class("_ANEInMemoryModelDescriptor"), "modelWithMILText:weights:optionsPlist:", .{ text, files, @as(?Id, null) }) orelse return error.Compile;
        const model = objc.msg(?Id, objc.class("_ANEInMemoryModel"), "inMemoryModelWithDescriptor:", .{desc}) orelse return error.Compile;
        const dir = objc.msg(Id, model, "localModelPath", .{});
        const fm = objc.msg(Id, objc.class("NSFileManager"), "defaultManager", .{});
        defer _ = objc.msg(bool, fm, "removeItemAtPath:error:", .{ dir, @as(?*?Id, null) });
        const none = dictionary(&.{}, &.{});
        var err: ?Id = null;
        if (!objc.msg(bool, model, "compiledModelExists", .{})) {
            const names = [_]Id{ objc.string("weights"), objc.string("model.mil"), objc.string("weights/weight.bin") };
            defer for (names) |n| objc.release(n);
            _ = objc.msg(bool, fm, "createDirectoryAtPath:withIntermediateDirectories:attributes:error:", .{ objc.msg(Id, dir, "stringByAppendingPathComponent:", .{names[0]}), true, @as(?Id, null), @as(?*?Id, null) });
            if (!objc.msg(bool, text, "writeToFile:atomically:", .{ objc.msg(Id, dir, "stringByAppendingPathComponent:", .{names[1]}), false })) return error.Compile;
            if (!objc.msg(bool, data, "writeToFile:atomically:", .{ objc.msg(Id, dir, "stringByAppendingPathComponent:", .{names[2]}), false })) return error.Compile;
            if (!objc.msg(bool, model, "compileWithQoS:options:error:", .{ qos, none, &err })) {
                std.log.err("ANE compile: {s}", .{objc.errorText(err)});
                return error.Compile;
            }
        }
        if (!objc.msg(bool, model, "loadWithQoS:options:error:", .{ qos, none, &err })) {
            std.log.err("ANE load: {s}", .{objc.errorText(err)});
            return error.Load;
        }
        return .{ .model = objc.retain(model) };
    }

    pub fn deinit(p: Program) void {
        var err: ?Id = null;
        _ = objc.msg(bool, p.model, "unloadWithQoS:error:", .{ qos, &err });
        objc.release(p.model);
    }

    /// A reusable request reading `input` and writing `output`.
    pub fn request(p: Program, input: Surface, output: Surface) !Request {
        const pool = objc.Pool.push();
        defer pool.pop();
        const zero = number(0);
        const req = objc.msg(?Id, objc.class("_ANERequest"), "requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:", .{ array(&.{input.object}), array(&.{zero}), array(&.{output.object}), array(&.{zero}), @as(?Id, null), @as(?Id, null), zero }) orelse return error.Request;
        return .{ .model = p.model, .id = objc.retain(req) };
    }
};

pub const Request = struct {
    model: Id,
    id: Id,

    /// Runs the program once and returns when the Neural Engine has written the output.
    pub fn run(r: Request) !void {
        const pool = objc.Pool.push();
        defer pool.pop();
        var err: ?Id = null;
        if (!objc.msg(bool, r.model, "evaluateWithQoS:options:request:error:", .{ qos, dictionary(&.{}, &.{}), r.id, &err })) {
            std.log.err("ANE evaluate: {s}", .{objc.errorText(err)});
            return error.Evaluate;
        }
    }

    pub fn deinit(r: Request) void {
        objc.release(r.id);
    }
};

test "a 1x1 convolution on the Neural Engine matches an fp32 sum of its fp16 inputs" {
    if (!available()) return error.SkipZigTest;
    const a = std.testing.allocator;
    const device = try Device.init();
    defer device.deinit();
    const k = 256;
    const n = 64;
    const s = 128;
    const weights = try Blob.init(a, &.{n * k});
    defer weights.deinit(a);
    const w = weights.values(0, n * k);
    for (w, 0..) |*v, i| v.* = @floatCast(@as(f32, @floatFromInt(@as(i32, @intCast(i * 7919 % 41)) - 20)) / 64.0);
    const mil = try conv(a, k, n, s);
    defer a.free(mil);
    const program = try Program.init(mil, weights.bytes);
    defer program.deinit();
    const x = try Surface.init(device, k * s * 2);
    defer x.deinit();
    const y = try Surface.init(device, n * s * 2);
    defer y.deinit();
    const xs = x.buffer.slice(f16, k * s);
    for (xs, 0..) |*v, i| v.* = @floatCast(@as(f32, @floatFromInt(@as(i32, @intCast(i * 104729 % 33)) - 16)) / 8.0);
    const req = try program.request(x, y);
    defer req.deinit();
    try req.run();
    const ys = y.buffer.slice(f16, n * s);
    for (0..n) |row| for (0..s) |col| {
        var want: f32 = 0;
        for (0..k) |i| want += @as(f32, w[row * k + i]) * @as(f32, xs[i * s + col]);
        try std.testing.expectApproxEqAbs(want, @as(f32, ys[row * s + col]), 0.05 + @abs(want) * 2e-3);
    };
}
