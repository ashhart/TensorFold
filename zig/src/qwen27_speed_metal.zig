//! Recording device facade; recurrent arithmetic is deliberately a host sentinel, never a GPU receipt.
const std = @import("std");
pub const ResourceOptions = struct {
    pub const shared = 0;
    pub const untracked = 0;
};
pub const Buffer = struct {
    id: usize,
    data: []align(16) u8,
    owner: ?*Allocations = null,
    pub fn length(b: Buffer) usize {
        return b.data.len;
    }
    pub fn contents(b: Buffer) [*]u8 {
        return b.data.ptr;
    }
    pub fn slice(b: Buffer, comptime T: type, n: usize) []T {
        std.debug.assert(n * @sizeOf(T) <= b.data.len);
        return @as([*]T, @ptrCast(@alignCast(b.data.ptr)))[0..n];
    }
    pub fn deinit(b: Buffer) void {
        if (b.owner) |a| {
            a.bytes -= b.data.len;
            a.live -= 1;
            std.testing.allocator.free(b.data);
        }
    }
};
pub const Size = struct { width: usize = 1, height: usize = 1, depth: usize = 1 };
pub const Pipeline = struct {
    id: usize,
    pub fn init(_: Device, _: Library, name: []const u8, _: bool) !Pipeline {
        const names = [_][]const u8{ "pre", "tree", "replay", "post", "conv_commit", "unused5", "unused6", "unused7", "publish_conv", "unused9", "tree_slots_1", "tree_slots_2", "tree_slots_4", "tree_slots_8", "tree_slots_16" };
        for (names, 0..) |suffix, i| if (std.mem.endsWith(u8, name, suffix)) return .{ .id = i };
        return .{ .id = 20 };
    }
    pub fn deinit(_: Pipeline) void {}
    pub fn simdWidth(_: Pipeline) usize {
        return 32;
    }
    pub fn maxThreads(_: Pipeline) usize {
        return 1024;
    }
};
pub const Allocations = struct { bytes: usize = 0, peak_bytes: usize = 0, live: usize = 0, calls: usize = 0, fail: usize = std.math.maxInt(usize) };
pub const Device = struct {
    allocations: *Allocations,
    pub fn buffer(d: Device, bytes: usize, _: usize) !Buffer {
        const a = d.allocations;
        const call = a.calls;
        a.calls += 1;
        if (call == a.fail) return error.NoBuffer;
        const data = try std.testing.allocator.alignedAlloc(u8, .@"16", bytes);
        a.bytes += bytes;
        a.peak_bytes = @max(a.peak_bytes, a.bytes);
        a.live += 1;
        return .{ .id = call + 2000, .data = data, .owner = a };
    }
};
pub const Library = struct {
    pub fn fromSource(_: Device, _: []const u8, _: CompileOptions) !Library {
        return .{};
    }
    pub fn deinit(_: Library) void {}
};
pub const CompileOptions = struct {
    pub fn mlx() CompileOptions {
        return .{};
    }
};
pub const Binding = struct { buffer: Buffer, offset: usize };
pub const Params = extern struct {
    rows: u32,
    nk: u32,
    nv: u32,
    dk: u32,
    dv: u32,
    taps: u32,
    slots: u32,
    flags: u32,
    eps: f32,
    zba_stride: u32,
    advance: u32,
};
pub const Recorder = struct {
    bindings: [16]?Binding = @splat(null),
    pipeline: usize = 0,
    params: Params = undefined,
    restores: usize = 0,
    segments: usize = 2,
    fn words(r: *Recorder, at: usize) []u32 {
        const b = r.bindings[at].?;
        return @as([*]u32, @ptrCast(@alignCast(b.buffer.data.ptr + b.offset)))[0 .. (b.buffer.data.len - b.offset) / 4];
    }
    fn run(r: *Recorder) void {
        const p = r.params;
        const stride = p.nv * p.dv * p.dk;
        if (r.pipeline >= 10 and r.pipeline <= 14) {
            const segments = r.words(7);
            for (0..r.segments) |i| {
                const slot = segments[i * 4 + 2];
                if (p.advance != 0 and segments[i * 4 + 1] == 1) {
                    const state = r.words(12)[slot * stride ..][0..stride];
                    if (r.bindings[13] != null) @memcpy(r.words(13)[slot * stride ..][0..stride], state);
                    for (state) |*word| word.* +%= @intCast(7 + slot);
                }
            }
        } else if (r.pipeline == 2) {
            const keeps = r.words(6);
            const segments = r.words(9);
            for (0..r.segments) |i| {
                const slot = keeps[i * 4 + 2];
                if (p.advance != 0 and segments[i * 4 + 1] == 1 and keeps[i * 4 + 1] == 0) {
                    @memcpy(r.words(8)[slot * stride ..][0..stride], r.words(10)[slot * stride ..][0..stride]);
                    r.restores += 1;
                }
            }
        } else if (r.pipeline == 4) {
            const keeps = r.words(3);
            const width = (2 * p.nk * p.dk + p.nv * p.dv) * (p.taps - 1) / 2;
            for (0..r.segments) |i| {
                const slot = keeps[i * 4 + 2];
                const out = r.words(6)[slot * width ..][0..width];
                if (keeps[i * 4 + 1] == 0) @memcpy(out, r.words(0)[slot * width ..][0..width]) else @memset(out, 0);
            }
        } else if (r.pipeline == 8) {
            const keeps = r.words(1);
            const width = (2 * p.nk * p.dk + p.nv * p.dv) * (p.taps - 1) / 2;
            for (0..r.segments) |i| {
                const slot = keeps[i * 4 + 3];
                @memcpy(r.words(3)[slot * width ..][0..width], r.words(0)[slot * width ..][0..width]);
            }
        }
    }
};
pub const ComputeEncoder = struct {
    record: *Recorder,
    pub fn setPipeline(e: ComputeEncoder, p: Pipeline) void {
        e.record.pipeline = p.id;
        e.record.bindings = @splat(null);
    }
    pub fn setBuffer(e: ComputeEncoder, b: Buffer, offset: usize, at: usize) void {
        e.record.bindings[at] = .{ .buffer = b, .offset = offset };
    }
    pub fn setValue(e: ComputeEncoder, value: anytype, _: usize) void {
        if (@sizeOf(@TypeOf(value)) == @sizeOf(Params)) @memcpy(std.mem.asBytes(&e.record.params), std.mem.asBytes(&value));
    }
    pub fn dispatchThreads(e: ComputeEncoder, _: Size, _: Size) void {
        e.record.run();
    }
    pub fn dispatchGroups(e: ComputeEncoder, _: Size, _: Size) void {
        e.record.run();
    }
    pub fn barrier(_: ComputeEncoder) void {}
    pub fn end(_: ComputeEncoder) void {}
};
pub const CommandBuffer = struct {
    pub fn compute(_: CommandBuffer, _: enum { serial }) ComputeEncoder {
        unreachable;
    }
    pub fn commit(_: CommandBuffer) void {}
    pub fn wait(_: CommandBuffer) void {}
    pub fn failure(_: CommandBuffer) ?[]const u8 {
        return null;
    }
    pub fn gpuSeconds(_: CommandBuffer) f64 {
        return 0;
    }
};
pub const Queue = struct {
    pub fn commandBuffer(_: Queue) CommandBuffer {
        return .{};
    }
};
