//! Host recording boundary. Tokens below are width sentinels, not Metal arithmetic results.
const std = @import("std");
pub const flashnext_recording = true;
pub const DispatchType = @import("src/metal/types.zig").DispatchType;
pub const ResourceOptions = @import("src/metal/types.zig").ResourceOptions;
pub const Size = @import("src/metal/types.zig").Size;
pub var allocator: std.mem.Allocator = undefined;
pub var wrong: usize = 0;
pub var calls: [4]usize = @splat(0);
pub var expected: [4]usize = .{ 4, 64, 4, 64 };
pub var scale_calls: usize = 0;
pub var first_blit: ?struct { offset: usize, bytes: usize } = null;
pub var blits: usize = 0;
pub const Copy = struct { id: usize, off: usize };
pub var sources: [512]Copy = @splat(.{ .id = 0, .off = 0 }); // every copy's source buffer and offset, in order
pub var source_count: usize = 0;
pub var draws: usize = 0;
pub var rotary_calls: usize = 0;
pub var fsr_fence_updates: usize = 0;
pub var fsr_fence_waits: usize = 0;
pub const HandoffOp = struct {
    kind: enum { compute, end, signal, wait, dispatch },
    cb_id: usize = 0,
    event_id: usize = 0,
    seq: u64 = 0,
    mode: DispatchType = .serial,
    pipeline: []const u8 = "",
};
pub var record_handoffs: bool = false;
pub var handoff_ops: [1024]HandoffOp = undefined;
pub var handoff_op_count: usize = 0;
const SharedState = struct {
    value: std.atomic.Value(u64) = .init(0),
    sets: std.atomic.Value(usize) = .init(0),
};
var shared_states: [512]SharedState = @splat(.{});
var next_shared: usize = 1;

pub fn resetHandoffs() void {
    record_handoffs = false;
    handoff_op_count = 0;
    next_shared = 1;
    for (&shared_states) |*state| {
        state.value.store(0, .release);
        state.sets.store(0, .release);
    }
}

fn recordHandoff(op: HandoffOp) void {
    if (!record_handoffs) return;
    std.debug.assert(handoff_op_count < handoff_ops.len);
    handoff_ops[handoff_op_count] = op;
    handoff_op_count += 1;
}
pub const Buffer = struct {
    id: usize,
    data: []align(256) u8,
    pub fn contents(b: Buffer) [*]u8 {
        return b.data.ptr;
    }
    pub fn length(b: Buffer) usize {
        return b.data.len;
    }
    pub fn slice(b: Buffer, comptime T: type, n: usize) []T {
        return @as([*]T, @ptrCast(@alignCast(b.data.ptr)))[0..n];
    }
    pub fn deinit(_: Buffer) void {}
};
pub const Device = struct {
    id: usize = 0,
    tensor_units: bool = true,
    pub fn allocated(_: Device) u64 {
        return 0;
    }
    pub fn buffer(_: Device, n: usize, _: usize) !Buffer {
        const data = try allocator.alignedAlloc(u8, .fromByteUnits(256), n);
        @memset(data, 0);
        return .{ .id = @intFromPtr(data.ptr), .data = data };
    }
    pub fn event(_: Device) !Event {
        return .{};
    }
    pub fn sharedEvent(_: Device) !SharedEvent {
        if (next_shared == shared_states.len) return error.RecordingEventLimit;
        const id = next_shared;
        next_shared += 1;
        shared_states[id] = .{};
        return .{ .id = id };
    }
    pub fn tensorUnits(d: Device) bool {
        return d.tensor_units;
    }
    pub fn generation(_: Device) u32 {
        return 14;
    }
};
pub const CompileOptions = struct {
    pub fn mlx() @This() {
        return .{};
    }
};
pub const Library = struct {
    id: usize = 0,
    text: []const u8,
    pub fn fromSource(_: Device, text: []const u8, _: CompileOptions) !Library {
        return .{ .text = text };
    }
    pub fn deinit(_: Library) void {}
};
pub const Pipeline = struct {
    id: usize = 0,
    name: []const u8 = "noop",
    text: []const u8 = "",
    pub fn init(_: Device, lib: Library, name: []const u8, _: bool) !Pipeline {
        return .{ .name = name, .text = lib.text };
    }
    pub fn maxThreads(_: Pipeline) usize {
        return 1024;
    }
    pub fn simdWidth(_: Pipeline) usize {
        return 32;
    }
    pub fn deinit(_: Pipeline) void {}
};
pub const Bound = struct { b: Buffer, off: usize };
var pipe: Pipeline = .{};
var bound: [32]Bound = undefined;
pub const ComputeEncoder = struct {
    id: usize = 0,
    cb_id: usize = 0,
    mode: DispatchType = .serial,
    pub fn commandBuffer(e: ComputeEncoder) CommandBuffer {
        return .{ .id = e.cb_id };
    }
    pub fn dispatchType(e: ComputeEncoder) DispatchType {
        return e.mode;
    }
    pub fn setPipeline(_: ComputeEncoder, p: Pipeline) void {
        pipe = p;
    }
    pub fn setBuffer(_: ComputeEncoder, b: Buffer, off: usize, at: usize) void {
        bound[at] = .{ .b = b, .off = off };
    }
    pub fn setValue(_: ComputeEncoder, _: anytype, _: usize) void {}
    pub fn setBytes(_: ComputeEncoder, _: []const u8, _: usize) void {}
    fn value(text: []const u8, name: []const u8) usize {
        const at = (std.mem.indexOf(u8, text, name) orelse unreachable) + name.len;
        const stop = at + (std.mem.indexOfScalar(u8, text[at..], ';') orelse unreachable);
        return std.fmt.parseInt(usize, text[at..stop], 10) catch unreachable;
    }
    fn check(kind: usize, names: []const []const u8, wanted: []const usize) void {
        calls[kind] += 1;
        for (names, wanted) |name, want| if (value(pipe.text, name) != want) {
            wrong += 1;
        };
    }
    pub fn dispatchThreads(e: ComputeEncoder, grid: Size, _: Size) void {
        recordHandoff(.{ .kind = .dispatch, .cb_id = e.cb_id, .mode = e.mode, .pipeline = pipe.name });
        if (std.mem.eql(u8, pipe.name, "tf_rotary_prepare")) {
            rotary_calls += 1;
            return;
        }
        if (std.mem.eql(u8, pipe.name, "fz_xgu") or std.mem.eql(u8, pipe.name, "fz_xdown")) {
            calls[if (std.mem.eql(u8, pipe.name, "fz_xgu")) @as(usize, 2) else 3] += 1;
            for (expected, [_]usize{ 6, 32, 6, 32 }) |want, actual| if (want != actual) {
                wrong += 1;
            };
            return;
        }
        if (std.mem.eql(u8, pipe.name, "draw")) {
            draws += 1;
            return;
        }
        if (std.mem.indexOf(u8, pipe.name, "qa_embed_rows_") != null) check(0, &.{ "constexpr int BITS = ", "constexpr int GS = " }, expected[0..2]) else if (std.mem.indexOf(u8, pipe.name, "qa_ple_lookup_") != null) check(1, &.{ "constexpr int BITS = ", "constexpr int GS = " }, expected[0..2]) else if (std.mem.indexOf(u8, pipe.name, "qa_expert_gateup_") != null) check(2, &.{ "constexpr int WB = ", "constexpr int WG = ", "constexpr int SWB = ", "constexpr int SWG = " }, &expected) else if (std.mem.indexOf(u8, pipe.name, "qa_expert_down_y_") != null) check(3, &.{ "constexpr int WB = ", "constexpr int WG = ", "constexpr int SWB = ", "constexpr int SWG = " }, &expected) else if (std.mem.eql(u8, pipe.name, "scale")) {
            scale_calls += 1;
        } else if (std.mem.eql(u8, pipe.name, "argmax")) {
            const b = bound[1];
            const out = @as([*]u32, @ptrCast(@alignCast(b.b.contents() + b.off)));
            for (0..grid.width / 1024) |i| out[i] = if (wrong == 0) 7 else 9;
        } else if (std.mem.eql(u8, pipe.name, "fz_argmax_join")) { // the two-launch argmax: one group a row (dispatchGroups)
            const b = bound[1];
            const out = @as([*]u32, @ptrCast(@alignCast(b.b.contents() + b.off)));
            for (0..grid.width) |i| out[i] = if (wrong == 0) 7 else 9;
        }
    }
    pub fn dispatchGroups(e: ComputeEncoder, grid: Size, threads: Size) void {
        e.dispatchThreads(grid, threads);
    }
    pub fn dispatchIndirect(_: ComputeEncoder, _: Buffer, _: usize, _: Size) void {}
    pub fn barrier(_: ComputeEncoder) void {}
    pub fn end(e: ComputeEncoder) void {
        recordHandoff(.{ .kind = .end, .cb_id = e.cb_id, .mode = e.mode });
    }
};
pub const Queue = struct {
    pub fn deinit(_: @This()) void {}
    pub fn commandBuffer(_: Queue) CommandBuffer {
        return .{};
    }
};
pub const CommandBuffer = struct {
    id: usize = 0,
    pub fn retain(c: @This()) @This() {
        return c;
    }
    pub fn release(_: @This()) void {}
    pub fn compute(cb: CommandBuffer, mode: DispatchType) ComputeEncoder {
        recordHandoff(.{ .kind = .compute, .cb_id = cb.id, .mode = mode });
        return .{ .mode = mode, .cb_id = cb.id };
    }
    pub fn blit(_: CommandBuffer) BlitEncoder {
        return .{};
    }
    pub fn commit(_: CommandBuffer) void {}
    pub fn wait(_: CommandBuffer) void {}
    pub fn gpuSeconds(_: CommandBuffer) f64 {
        return 0;
    }
    pub fn gpuStart(_: CommandBuffer) f64 {
        return 0;
    }
    pub fn gpuEnd(_: CommandBuffer) f64 {
        return 0;
    }
    pub fn failure(_: CommandBuffer) ?[]const u8 {
        return null;
    }
    pub fn waitForEvent(_: CommandBuffer, _: Event, _: u64) void {}
    pub fn signalEvent(_: CommandBuffer, _: Event, _: u64) void {}
    pub fn waitFor(cb: CommandBuffer, event: SharedEvent, seq: u64) void {
        recordHandoff(.{ .kind = .wait, .cb_id = cb.id, .event_id = event.id, .seq = seq });
    }
    pub fn signal(cb: CommandBuffer, event: SharedEvent, seq: u64) void {
        recordHandoff(.{ .kind = .signal, .cb_id = cb.id, .event_id = event.id, .seq = seq });
    }
};
pub const BlitEncoder = struct {
    pub fn copy(_: BlitEncoder, a: Buffer, ao: usize, b: Buffer, bo: usize, n: usize) void {
        _ = b;
        _ = bo;
        if (first_blit == null) first_blit = .{ .offset = ao, .bytes = n };
        blits += 1;
        if (source_count < sources.len) {
            sources[source_count] = .{ .id = a.id, .off = ao };
            source_count += 1;
        }
    }
    pub fn end(_: BlitEncoder) void {}
};
/// Recorded calls run in order on the host, so a fence orders nothing here.
pub const counters = struct {
    pub const Fence = struct {
        pub fn init(_: Device) !Fence {
            return .{};
        }
        pub fn updateBlit(_: Fence, _: BlitEncoder) void {}
        pub fn waitCompute(_: Fence, _: ComputeEncoder) void {}
        pub fn deinit(_: Fence) void {}
    };
};
pub const ResidencySet = struct {
    pub const Follow = struct { ctx: *anyopaque, join: *const fn (*anyopaque, Buffer) void, leave: *const fn (*anyopaque, Buffer) void };
};
pub const SharedEvent = struct {
    id: usize = 0,
    pub fn value(event: @This()) u64 {
        return shared_states[event.id].value.load(.acquire);
    }
    pub fn wait(event: @This(), seq: u64, _: u64) bool {
        return event.value() >= seq;
    }
    pub fn set(event: @This(), seq: u64) void {
        shared_states[event.id].value.store(seq, .release);
        _ = shared_states[event.id].sets.fetchAdd(1, .acq_rel);
    }
    pub fn setCount(event: @This()) usize {
        return shared_states[event.id].sets.load(.acquire);
    }
    pub fn deinit(_: @This()) void {}
};
pub const Event = struct {
    id: usize = 0,
    pub fn deinit(_: @This()) void {}
};
pub const objc = struct {
    pub const Id = usize;
    pub fn msg(comptime T: type, _: Id, selector: []const u8, _: anytype) T {
        if (comptime T == ?Id) {
            std.debug.assert(std.mem.eql(u8, selector, "newFence"));
            return 1;
        } else if (comptime T == void) {
            if (std.mem.eql(u8, selector, "updateFence:")) fsr_fence_updates += 1 else if (std.mem.eql(u8, selector, "waitForFence:")) fsr_fence_waits += 1 else @panic("unexpected recording selector");
        } else @compileError("unexpected recording return type");
    }
    pub fn release(_: Id) void {}
    pub const Pool = struct {
        pub fn push() @This() {
            return .{};
        }
        pub fn pop(_: @This()) void {}
    };
};
pub const clock = struct {
    pub fn seconds() f64 {
        return 0;
    }
};

pub const MappedFile = struct {
    pub fn deinit(_: MappedFile) void {}
    bytes: [*]const u8,
    size: usize,
    pub fn open(_: []const u8) !MappedFile {
        return error.UnexpectedFile;
    }
};
