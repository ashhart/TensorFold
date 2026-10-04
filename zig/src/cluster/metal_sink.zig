//! Loading into Metal: every mapped range becomes a no-copy buffer in one residency set; a tensor is (buffer, offset).
const std = @import("std");
const mtl = @import("metal");
const cluster = @import("cluster");

const loader = cluster.loader;

pub const Placed = struct { range: loader.Range, view_offset: u64, buffer: mtl.Buffer };

pub const Sink = struct {
    gpa: std.mem.Allocator,
    device: mtl.Device,
    set: ?mtl.ResidencySet,
    placed: std.ArrayList(Placed) = .empty,

    /// A residency set when the OS has them (macOS 15+); buffers still work without one.
    pub fn init(gpa: std.mem.Allocator, device: mtl.Device, capacity: usize) Sink {
        return .{ .gpa = gpa, .device = device, .set = device.residencySet(capacity) catch null };
    }

    pub fn deinit(s: *Sink) void {
        for (s.placed.items) |p| p.buffer.deinit();
        s.placed.deinit(s.gpa);
        if (s.set) |set| set.deinit();
    }

    pub fn sink(s: *Sink) loader.Sink {
        return .{ .ptr = s, .run = run };
    }

    fn run(ptr: *anyopaque, r: loader.Range, view_offset: u64, view: []align(loader.page) const u8) anyerror!void {
        const s: *Sink = @ptrCast(@alignCast(ptr));
        const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
        const buffer = try s.device.bufferNoCopy(@constCast(view.ptr), view.len, opts);
        errdefer buffer.deinit();
        if (s.set) |set| set.add(buffer);
        try s.placed.append(s.gpa, .{ .range = r, .view_offset = view_offset, .buffer = buffer });
    }

    /// Make every buffer resident before the first round (the wired bytes are what the plan budgeted).
    pub fn commit(s: *Sink) void {
        const set = s.set orelse return;
        set.commit();
        set.requestResidency();
    }

    /// The buffer and byte offset where file `file`'s byte `offset` lives, when this rank loaded it.
    pub fn find(s: *const Sink, file: u32, offset: u64) ?struct { buffer: mtl.Buffer, offset: u64 } {
        for (s.placed.items) |p| {
            if (p.range.file != file or offset < p.range.offset or offset >= p.range.offset + p.range.len) continue;
            return .{ .buffer = p.buffer, .offset = offset - p.view_offset };
        }
        return null;
    }
};

test "loaded ranges are no-copy Metal buffers whose bytes are the file's, in one residency set" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const img = try cluster.checkpoint.image(arena.allocator(), &.{
        .{ .name = "model.layers.0.mlp.experts.0.down_proj.weight", .dtype = "U8", .shape = &.{ 64, 1024 }, .fill = 3 },
        .{ .name = "model.layers.0.mlp.experts.1.down_proj.weight", .dtype = "U8", .shape = &.{ 64, 1024 }, .fill = 4 },
    });
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = img });
    const ts = try cluster.checkpoint.parseHeader(arena.allocator(), 0, img);
    const files = [_]cluster.checkpoint.File{.{ .name = "model.safetensors", .size = img.len }};
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    var s = Sink.init(a, device, 16);
    defer s.deinit();
    const ranges = [_]loader.Range{.{ .file = 0, .offset = ts[1].start, .len = ts[1].bytes }};
    var rep = try loader.load(io, arena.allocator(), tmp.dir, .{ .files = &files }, &ranges, null, s.sink(), null);
    defer rep.release(arena.allocator());
    s.commit();
    const at = s.find(0, ts[1].start).?;
    const bytes = at.buffer.contents()[@intCast(at.offset)..][0..@intCast(ts[1].bytes)];
    for (bytes) |b| try std.testing.expectEqual(@as(u8, 4), b);
    try std.testing.expectEqual(@as(?@TypeOf(at), null), s.find(0, ts[0].start));
    if (s.set) |set| try std.testing.expectEqual(@as(usize, 1), set.count());
}
