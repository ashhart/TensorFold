//! Host allocations, set membership and keepalive commands recorded without Objective-C or a GPU.
const std = @import("std");
const real = @import("src/metal/metal.zig");
const Self = @This();
pub const ResourceOptions = real.ResourceOptions;
pub const types = real.types;
pub const ResourceUsage = real.ResourceUsage;
pub const Size = real.Size;
pub const DispatchType = real.DispatchType;
pub const Pipeline = struct {
    id: real.objc.Id,
    pub fn init(_: Device, _: Library, _: []const u8, _: bool) !@This() {
        return .{ .id = @ptrFromInt(1) };
    }
    pub fn initOptional(_: Device, _: Library, _: []const u8, _: bool) !@This() {
        return .{ .id = @ptrFromInt(1) };
    }
    pub fn deinit(_: @This()) void {}
    pub fn simdWidth(_: @This()) usize {
        return 32;
    }
    pub fn init(_: Device, _: Library, _: []const u8, _: bool) !Pipeline {
        return .{ .id = @ptrFromInt(1) };
    }
    pub fn maxThreads(_: @This()) usize {
        return 1024;
    }
};
pub const Library = struct {
    id: real.objc.Id = @ptrFromInt(1),
    pub fn fromFile(_: Device, _: []const u8) !@This() {
        return .{};
    }
    pub fn fromBytes(_: Device, _: []const u8) !@This() {
        return .{};
    }
    pub fn fromSource(_: Device, _: []const u8, _: CompileOptions) !@This() {
        return .{};
    }
    pub fn deinit(_: @This()) void {}
};
pub const CompileOptions = struct {
    pub fn mlx() @This() {
        return .{};
    }
};
pub const Event = real.Event;
pub const SharedEvent = struct {
    pub fn value(_: @This()) u64 {
        return std.math.maxInt(u64);
    }
    pub fn wait(_: @This(), _: u64, _: u32) bool {
        return true;
    }
    pub fn set(_: @This(), _: u64) void {}
};
pub const ane = struct {
    pub const Program = real.ane.Program;
    pub const Request = real.ane.Request;
    pub const Surface = struct { buffer: Buffer };
};
pub const MappedFile = real.MappedFile;
pub const clock = struct {
    pub fn seconds() f64 {
        return 0;
    }
};
pub const residency_policy = real.residency_policy;
pub const keepalive = struct {
    pub const Target = real.keepalive.TargetFor(Self);
};
pub const resident = struct {
    pub const Resident = real.resident.Owner(Self);
    pub const fields = real.resident.Fields(Self);
    pub const streams = real.resident.Streams(Self);
};
pub const objc = struct {
    pub const Id = real.objc.Id;
    pub fn release(_: Id) void {}
    pub fn retain(id: Id) Id {
        return id;
    }
    pub fn msg(comptime R: type, _: Id, comptime _: [:0]const u8, _: anytype) R {
        if (R != void) @compileError("ownership fixture permits only inert fence messages");
    }
    pub const Pool = struct {
        pub fn push() @This() {
            return .{};
        }
        pub fn pop(_: @This()) void {}
    };
};

const Allocation = struct { record: *Record, data: []align(256) u8, bytes: usize, alive: bool = true, enrolled: bool = false };
pub const Record = struct {
    gpa: std.mem.Allocator,
    allocations: std.ArrayList(*Allocation) = .empty,
    active: bool = false,
    attached: bool = false,
    commits: usize = 0,
    waits: usize = 0,
    queue_releases: usize = 0,
    uses: usize = 0,
    copies: usize = 0,
    passes: [256]Pass = undefined,
    pass_count: usize = 0,
    current: Pass = .{},
    logits: usize = 0,
    bad: usize = 0,
    set_commits: usize = 0,
    initial_missing: usize = 0,
    fail_at: ?usize = null,
    refuse_set: bool = false,

    pub fn device(r: *Record) Device {
        return .{ .record = r };
    }
    pub fn queue(r: *Record) Queue {
        return .{ .record = r };
    }
    pub fn deinit(r: *Record) void {
        for (r.allocations.items) |p| {
            if (p.alive) r.gpa.free(p.data);
            r.gpa.destroy(p);
        }
        r.allocations.deinit(r.gpa);
    }
    pub fn check(r: *const Record) !void {
        try std.testing.expectEqual(@as(usize, 0), r.bad);
        for (r.allocations.items) |p| try std.testing.expectEqual(p.alive and r.active, p.enrolled);
    }
    pub fn allFreed(r: *const Record) !void {
        try std.testing.expectEqual(@as(usize, 0), r.bad);
        try std.testing.expect(!r.active and !r.attached);
        for (r.allocations.items) |p| try std.testing.expect(!p.alive);
    }
    pub fn checkInitial(r: *const Record) !void {
        try std.testing.expect(r.set_commits > 0);
        try std.testing.expectEqual(@as(usize, 0), r.initial_missing);
    }
};

pub const Buffer = struct {
    id: *Allocation,
    pub fn length(b: Buffer) usize {
        return b.id.bytes;
    }
    pub fn contents(b: Buffer) [*]u8 {
        return b.id.data.ptr;
    }
    pub fn slice(b: Buffer, comptime T: type, n: usize) []T {
        return @as([*]T, @ptrCast(@alignCast(b.contents())))[0..n];
    }
    pub fn deinit(b: Buffer) void {
        if (!b.id.alive or b.id.enrolled) b.id.record.bad += 1;
        if (b.id.alive) b.id.record.gpa.free(b.id.data);
        b.id.alive = false;
    }
};

pub const Bound = struct { id: usize = 0, off: usize = 0 };
pub const Pass = struct {
    pipe: usize = 0,
    buffers: [16]Bound = @splat(.{}),
    values: [16][64]u8 = @splat(@splat(0)),
    lengths: [16]usize = @splat(0),
    output: ?Buffer = null,
    grid: Size = .{ .width = 0 },
    threads: Size = .{ .width = 0 },
    direct: bool = false,
};

pub const Device = struct {
    record: *Record,
    pub fn buffer(d: Device, bytes: usize, _: usize) !Buffer {
        if (d.record.fail_at == d.record.allocations.items.len) return error.InjectedBufferFailure;
        const p = try d.record.gpa.create(Allocation);
        errdefer d.record.gpa.destroy(p);
        const data = try d.record.gpa.alignedAlloc(u8, .fromByteUnits(256), @max(1, bytes));
        errdefer d.record.gpa.free(data);
        @memset(data, 0);
        p.* = .{ .record = d.record, .data = data, .bytes = bytes };
        try d.record.allocations.append(d.record.gpa, p);
        return .{ .id = p };
    }
    pub fn tensorUnits(_: Device) bool {
        return true;
    }
    pub fn maxWorkingSet(_: Device) usize {
        return 256 * ResidencySet.GiB;
    }
    pub fn tensorUnits(_: Device) bool {
        return false;
    }
    pub fn allocated(d: Device) usize {
        var bytes: usize = 0;
        for (d.record.allocations.items) |p| if (p.alive) {
            bytes += p.bytes;
        };
        return bytes;
    }
    pub fn residencySet(d: Device, _: usize) !ResidencySet {
        if (d.record.refuse_set) return error.InjectedSetUnavailable;
        if (d.record.active) return error.SetAlreadyActive;
        d.record.active = true;
        return .{ .record = d.record };
    }
};

pub const ResidencySet = struct {
    record: *Record,
    pub const GiB = real.ResidencySet.GiB;
    pub const margin = real.ResidencySet.margin;
    pub const share_percent = real.ResidencySet.share_percent;
    pub const fitsUnder = real.ResidencySet.fitsUnder;
    pub const fits = real.ResidencySet.fits;
    pub const refusedUnder = real.ResidencySet.refusedUnder;
    pub const gibs = real.ResidencySet.gibs;
    pub fn totalRam() ?u64 {
        return 256 * GiB;
    }
    pub const Follow = struct { ctx: *anyopaque, join: *const fn (*anyopaque, Buffer) void, leave: *const fn (*anyopaque, Buffer) void };
    pub fn add(s: ResidencySet, b: Buffer) void {
        if (!s.record.active or !b.id.alive or b.id.enrolled) s.record.bad += 1;
        b.id.enrolled = true;
    }
    pub fn remove(s: ResidencySet, b: Buffer) void {
        if (!s.record.active or !b.id.alive or !b.id.enrolled) s.record.bad += 1;
        b.id.enrolled = false;
    }
    pub fn commit(s: ResidencySet) void {
        if (s.record.set_commits == 0) for (s.record.allocations.items) |p| {
            if (p.alive and !p.enrolled) s.record.initial_missing += 1;
        };
        s.record.set_commits += 1;
    }
    pub fn requestResidency(_: ResidencySet) void {}
    pub fn deinit(s: ResidencySet) void {
        for (s.record.allocations.items) |p| p.enrolled = false;
        s.record.active = false;
    }
};

pub const Queue = struct {
    record: *Record,
    pub fn deinit(q: Queue) void {
        q.record.queue_releases += 1;
    }
    pub fn commandBuffer(q: Queue) CommandBuffer {
        return .{ .record = q.record };
    }
    pub fn commandBufferUnretained(q: Queue) CommandBuffer {
        return q.commandBuffer();
    }
    pub fn addResidencySet(q: Queue, _: ResidencySet) void {
        q.record.attached = true;
    }
    pub fn removeResidencySet(q: Queue, _: ResidencySet) void {
        q.record.attached = false;
    }
};

pub const CommandBuffer = struct {
    record: *Record,
    id: objc.Id = undefined,
    pub fn useResidencySet(c: CommandBuffer, _: ResidencySet) void {
        c.record.uses += 1;
    }
    pub fn blit(c: CommandBuffer) BlitEncoder {
        return .{ .record = c.record };
    }
    pub fn compute(c: CommandBuffer, _: DispatchType) ComputeEncoder {
        return .{ .record = c.record };
    }
    pub fn commit(c: CommandBuffer) void {
        c.record.commits += 1;
    }
    pub fn wait(c: CommandBuffer) void {
        c.record.waits += 1;
    }
    pub fn failure(_: CommandBuffer) ?[]const u8 {
        return null;
    }
    pub fn gpuStart(_: CommandBuffer) f64 {
        return 0;
    }
    pub fn gpuEnd(_: CommandBuffer) f64 {
        return 0;
    }
    pub fn gpuSeconds(_: CommandBuffer) f64 {
        return 0;
    }
    pub fn retain(c: CommandBuffer) CommandBuffer {
        return c;
    }
    pub fn release(_: CommandBuffer) void {}
    pub fn waitFor(_: CommandBuffer, _: SharedEvent, _: u64) void {}
    pub fn signal(_: CommandBuffer, _: SharedEvent, _: u64) void {}
    pub fn status(_: CommandBuffer) types.CommandBufferStatus {
        return .completed;
    }
};

pub const BlitEncoder = struct {
    record: *Record,
    pub fn copy(e: BlitEncoder, from: Buffer, at: usize, to: Buffer, out: usize, n: usize) void {
        if (at > from.length() or n > from.length() - at or out > to.length() or n > to.length() - out) {
            e.record.bad += 1;
            return;
        }
        std.mem.copyForwards(u8, to.contents()[out..][0..n], from.contents()[at..][0..n]);
        e.record.copies += 1;
    }
    pub fn end(_: BlitEncoder) void {}
};

pub const ComputeEncoder = struct {
    record: *Record,
    pub fn setPipeline(e: @This(), p: Pipeline) void {
        e.record.current = .{ .pipe = @intFromPtr(p.id) };
    }
    pub fn setBuffer(e: @This(), b: Buffer, off: usize, at: usize) void {
        e.record.current.buffers[at] = .{ .id = @intFromPtr(b.id), .off = off };
        if (at == 5) e.record.current.output = b;
    }
    pub fn setValue(e: @This(), v: anytype, at: usize) void {
        const bytes = std.mem.asBytes(&v);
        std.debug.assert(bytes.len <= e.record.current.values[at].len);
        e.record.current.lengths[at] = bytes.len;
        @memcpy(e.record.current.values[at][0..bytes.len], bytes);
    }
    pub fn setBytes(e: @This(), bytes: []const u8, at: usize) void {
        std.debug.assert(bytes.len <= e.record.current.values[at].len);
        e.record.current.lengths[at] = bytes.len;
        @memcpy(e.record.current.values[at][0..bytes.len], bytes);
    }
    fn dispatch(e: @This(), grid: Size, threads: Size, direct: bool) void {
        const r = e.record;
        r.current.grid = grid;
        r.current.threads = threads;
        r.current.direct = direct;
        if (r.logits != 0 and r.current.buffers[5].id == r.logits and r.current.lengths[4] == 24) {
            const dims = std.mem.bytesToValue([6]i32, r.current.values[4][0..24]);
            const vocab: usize = @intCast(dims[1]);
            const rows: usize = @intCast(dims[3]);
            const out = r.current.output.?;
            const all = @as([*]u16, @ptrCast(@alignCast(out.contents() + r.current.buffers[5].off)))[0 .. rows * vocab];
            for (0..rows) |row| {
                const l = all[row * vocab ..][0..vocab];
                @memset(l, 0);
                if (row + 1 < vocab) l[row + 1] = 0x3f80; // each row's greedy token is its one-based global row
            }
        }
        if (r.pass_count == r.passes.len) {
            r.bad += 1;
            return;
        }
        r.passes[r.pass_count] = r.current;
        r.pass_count += 1;
    }
    pub fn dispatchThreads(e: @This(), grid: Size, threads: Size) void {
        e.dispatch(grid, threads, true);
    }
    pub fn dispatchGroups(e: @This(), grid: Size, threads: Size) void {
        e.dispatch(grid, threads, false);
    }
    pub fn dispatchIndirect(_: @This(), _: Buffer, _: usize, _: Size) void {}
    pub fn barrier(_: @This()) void {}
    pub fn end(_: @This()) void {}
};
