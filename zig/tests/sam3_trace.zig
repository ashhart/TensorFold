const std = @import("std");
const sam = @import("sam3");
const fixture = @import("checkpoint_fixture");

pub const Trace = struct {
    a: std.mem.Allocator,
    io: std.Io,
    dir: std.Io.Dir,
    frames: []const usize,
    layers: bool,
    current_frame: usize = 0,
    stage: usize = 0,

    pub fn init(a: std.mem.Allocator, io: std.Io, path: []const u8, frames: []const usize, layers: bool) !Trace {
        try std.Io.Dir.cwd().createDir(io, path, .default_dir);
        return .{ .a = a, .io = io, .dir = try std.Io.Dir.cwd().openDir(io, path, .{}), .frames = frames, .layers = layers };
    }

    pub fn deinit(self: *Trace) void {
        self.dir.close(self.io);
    }

    pub fn observer(self: *Trace, index: usize) ?sam.observe.Observer {
        self.current_frame = index;
        self.stage = 0;
        if (std.mem.indexOfScalar(usize, self.frames, index) == null) return null;
        return .{ .context = self, .write = check };
    }

    pub fn frame(self: *Trace, session: *sam.Session, image: sam.Image, index: usize) !void {
        if (std.mem.indexOfScalar(usize, self.frames, index) == null) return;
        var arena = std.heap.ArenaAllocator.init(self.a);
        defer arena.deinit();
        const a = arena.allocator();
        var tensors: std.ArrayList(fixture.Entry) = .empty;
        var runtime = session.model.runtime(a);
        defer runtime.deinit();
        var context = Context{ .a = a, .entries = &tensors };
        const prepared = try sam.image_tower.prepare(&runtime, image);
        try feature(a, &tensors, "preprocessed_nhwc", prepared);
        runtime.observer = .{ .context = &context, .write = if (self.layers) record else ignore };
        const trunk = try sam.image_tower.encodeImage(&runtime, image);
        try feature(a, &tensors, "trunk.0", trunk);
        for (session.tracking_levels.?, 0..) |value, i| try feature(a, &tensors, try std.fmt.allocPrint(a, "pyramid.propagation.{d}", .{i}), value);
        if (session.prompt_levels) |levels| for (levels, 0..) |value, i|
            try feature(a, &tensors, try std.fmt.allocPrint(a, "pyramid.interactive.{d}", .{i}), value);
        for (session.groups, 0..) |group, b| {
            const result = group.latest orelse continue;
            const prefix = try std.fmt.allocPrint(a, "group.{d}", .{b});
            try feature(a, &tensors, try std.fmt.allocPrint(a, "{s}.mask_logits", .{prefix}), result.mask_logits);
            if (result.full_logits) |value| try feature(a, &tensors, try std.fmt.allocPrint(a, "{s}.full_logits", .{prefix}), value);
            if (result.readout_logits) |value| try feature(a, &tensors, try std.fmt.allocPrint(a, "{s}.readout_logits", .{prefix}), value);
            try tensor(a, &tensors, try std.fmt.allocPrint(a, "{s}.pointers", .{prefix}), result.pointers, &.{ result.slots, 256 });
            try host(a, &tensors, try std.fmt.allocPrint(a, "{s}.presence_logits", .{prefix}), result.presence[0..result.slots]);
            try host(a, &tensors, try std.fmt.allocPrint(a, "{s}.quality", .{prefix}), result.quality[0..result.slots]);
            if (group.bank.entries.items.len != 0) {
                const memory = group.bank.entries.items[group.bank.entries.items.len - 1];
                if (memory.frame != index) return error.UnexpectedMemoryFrame;
                try tensor(a, &tensors, try std.fmt.allocPrint(a, "{s}.memory", .{prefix}), memory.encoding, &.{ 1, 72, 72, session.model.config.memory_width });
            }
        }
        const bytes = try fixture.image(a, tensors.items);
        const name = try std.fmt.allocPrint(a, "frame-{d:0>6}.safetensors", .{index});
        const file = try self.dir.createFile(self.io, name, .{ .exclusive = true });
        defer file.close(self.io);
        try file.writeStreamingAll(self.io, bytes);
    }
};

fn check(context: *anyopaque, name: []const u8, value: sam.observe.Tensor, shape: []const usize) !void {
    const self: *Trace = @ptrCast(@alignCast(context));
    const input = self.layers and std.mem.endsWith(u8, name, ".input") and !std.mem.startsWith(u8, name, "tracker.");
    if (input or std.mem.startsWith(u8, name, "tracker.") or std.mem.startsWith(u8, name, "prompt.") or std.mem.startsWith(u8, name, "decoder.")) {
        var arena = std.heap.ArenaAllocator.init(self.a);
        defer arena.deinit();
        const a = arena.allocator();
        const bytes = try fixture.image(a, &.{.{ .name = name, .dtype = switch (value.dtype) {
            .f32 => "F32",
            .f16 => "F16",
            .bf16 => "BF16",
        }, .shape = shape, .bytes = value.slice(u8, value.count * value.dtype.size()) }});
        const path = if (input)
            try std.fmt.allocPrint(a, "frame-{d:0>6}-{s}.safetensors", .{ self.current_frame, name })
        else
            try std.fmt.allocPrint(a, "frame-{d:0>6}-stage-{d:0>3}-{s}.safetensors", .{ self.current_frame, self.stage, name });
        self.stage += @intFromBool(!input);
        const file = try self.dir.createFile(self.io, path, .{ .exclusive = true });
        defer file.close(self.io);
        try file.writeStreamingAll(self.io, bytes);
    }
    try value.checkFinite(name);
}

const Context = struct { a: std.mem.Allocator, entries: *std.ArrayList(fixture.Entry) };

fn ignore(_: *anyopaque, _: []const u8, _: sam.observe.Tensor, _: []const usize) !void {}

fn record(context: *anyopaque, name: []const u8, value: sam.observe.Tensor, shape: []const usize) !void {
    const sink: *Context = @ptrCast(@alignCast(context));
    if (std.mem.endsWith(u8, name, ".input")) return;
    try tensor(sink.a, sink.entries, name, value, shape);
}

fn tensor(a: std.mem.Allocator, entries: *std.ArrayList(fixture.Entry), name: []const u8, value: anytype, shape: []const usize) !void {
    var count: usize = 1;
    for (shape) |dimension| count = try std.math.mul(usize, count, dimension);
    if (count != value.count) return error.TraceShapeMismatch;
    try value.checkFinite(name);
    try entries.append(a, .{
        .name = name,
        .dtype = switch (value.dtype) {
            .f32 => "F32",
            .f16 => "F16",
            .bf16 => "BF16",
        },
        .shape = try a.dupe(usize, shape),
        .bytes = try a.dupe(u8, value.slice(u8, value.count * value.dtype.size())),
    });
}

fn feature(a: std.mem.Allocator, entries: *std.ArrayList(fixture.Entry), name: []const u8, value: anytype) !void {
    try tensor(a, entries, name, value.tensor, &.{ 1, value.height, value.width, value.channels });
}

fn host(a: std.mem.Allocator, entries: *std.ArrayList(fixture.Entry), name: []const u8, values: []const f32) !void {
    for (values) |value| if (!std.math.isFinite(value)) return error.NonFiniteTrace;
    try entries.append(a, .{ .name = name, .dtype = "F32", .shape = try a.dupe(usize, &.{values.len}), .bytes = try a.dupe(u8, std.mem.sliceAsBytes(values)) });
}
