//! Records construction and dispatch only, without Objective-C, shader compilation or a device.
const real = @import("real_metal");
const Self = @This();
pub const Size = real.Size;
pub const ResourceOptions = real.ResourceOptions;
pub const objc = struct {
    pub const Pool = struct {
        pub fn push() @This() {
            return .{};
        }
        pub fn pop(_: @This()) void {}
    };
};
pub const ResidencySet = struct {
    pub const Follow = struct {
        ctx: *anyopaque,
        join: *const fn (*anyopaque, Buffer) void,
        leave: *const fn (*anyopaque, Buffer) void,
    };
};
pub const resident = struct {
    pub const fields = real.resident.Fields(Self);
};
pub const CompileOptions = struct {
    pub fn mlx() CompileOptions {
        return .{};
    }
};
pub const Record = struct {
    libraries: usize = 0,
    pipelines: usize = 0,
    released: usize = 0,
    live_bytes: usize = 0,
    fail_pipeline: bool = false,
    current: usize = 0,
    rows: u32 = 0,
    launches: [256]struct { pipeline: usize, rows: u32 } = undefined,
    count: usize = 0,
    pub fn device(r: *Record) Device {
        return .{ .r = r };
    }
    pub fn encoder(r: *Record) ComputeEncoder {
        return .{ .r = r };
    }
    pub fn virtual(r: *Record) Buffer {
        return .{ .r = r, .bytes = 1 << 40 };
    }
};
pub const Buffer = struct {
    r: *Record,
    bytes: usize,
    owned: bool = false,
    pub fn length(b: Buffer) usize {
        return b.bytes;
    }
    pub fn contents(_: Buffer) [*]u8 {
        @panic("recording buffer cannot be read");
    }
    pub fn slice(b: Buffer, comptime T: type, n: usize) []T {
        return @as([*]T, @ptrCast(@alignCast(b.contents())))[0..n];
    }
    pub fn deinit(b: Buffer) void {
        if (b.owned) b.r.live_bytes -= b.bytes;
    }
};
pub const Device = struct {
    r: *Record,
    pub fn tensorUnits(_: Device) bool {
        return true;
    }
    pub fn generation(_: Device) usize {
        return 17;
    }
    pub fn buffer(d: Device, bytes: usize, _: usize) !Buffer {
        d.r.live_bytes += bytes;
        return .{ .r = d.r, .bytes = bytes, .owned = true };
    }
    pub fn queue(_: Device) !Queue {
        return .{};
    }
};
pub const Library = struct {
    pub fn fromSource(d: Device, _: []const u8, _: CompileOptions) !Library {
        d.r.libraries += 1;
        return .{};
    }
    pub fn deinit(_: Library) void {}
};
pub const Pipeline = struct {
    r: *Record,
    id: usize,
    pub fn init(d: Device, _: Library, _: []const u8, _: bool) !Pipeline {
        if (d.r.fail_pipeline) return error.PipelineBuild;
        d.r.pipelines += 1;
        return .{ .r = d.r, .id = d.r.pipelines };
    }
    pub fn deinit(p: Pipeline) void {
        p.r.released += 1;
    }
    pub fn simdWidth(_: Pipeline) usize {
        return 32;
    }
    pub fn maxThreads(_: Pipeline) usize {
        return 1024;
    }
};
pub const ComputeEncoder = struct {
    r: *Record,
    pub fn setPipeline(e: ComputeEncoder, p: Pipeline) void {
        e.r.current = p.id;
    }
    pub fn setBuffer(_: ComputeEncoder, _: Buffer, _: usize, _: usize) void {}
    pub fn setValue(e: ComputeEncoder, v: anytype, at: usize) void {
        if (@TypeOf(v) == [2]i32 and at == 3) e.r.rows = @intCast(v[0]);
    }
    pub fn dispatchGroups(e: ComputeEncoder, _: Size, _: Size) void {
        e.r.launches[e.r.count] = .{ .pipeline = e.r.current, .rows = e.r.rows };
        e.r.count += 1;
    }
    pub fn dispatchThreads(_: ComputeEncoder, _: Size, _: Size) void {}
    pub fn barrier(_: ComputeEncoder) void {}
    pub fn end(_: ComputeEncoder) void {}
};
pub const Queue = struct {
    pub fn deinit(_: Queue) void {}
    pub fn commandBuffer(_: Queue) CommandBuffer {
        @panic("recording test cannot submit commands");
    }
};
pub const CommandBuffer = struct {
    pub fn compute(_: CommandBuffer, _: real.DispatchType) ComputeEncoder {
        unreachable;
    }
    pub fn commit(_: CommandBuffer) void {
        unreachable;
    }
    pub fn wait(_: CommandBuffer) void {
        unreachable;
    }
    pub fn failure(_: CommandBuffer) ?[]const u8 {
        unreachable;
    }
    pub fn gpuSeconds(_: CommandBuffer) f64 {
        unreachable;
    }
};
