//! Real LFM3-VL tower rows from an independent patch fixture, without Python in the native process.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const q = tf.lfm3;
const Dump = struct {
    io: std.Io,
    a: std.mem.Allocator,
    dir: []const u8,

    fn take(ptr: *anyopaque, name: []const u8, value: q.vision.Ref, rows: usize, width: usize) !void {
        const self: *Dump = @ptrCast(@alignCast(ptr));
        const path = try std.fmt.allocPrint(self.a, "{s}/{s}.bf16", .{ self.dir, name });
        defer self.a.free(path);
        try std.Io.Dir.cwd().writeFile(self.io, .{
            .sub_path = path,
            .data = (value.buf.contents() + value.off)[0 .. rows * width * 2],
        });
    }
};

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const photo = args.len >= 3 and std.mem.eql(u8, args[2], "--image");
    if (photo and args.len != 5 and args.len != 6 or !photo and args.len != 6 and args.len != 7)
        return error.ExpectedModelPatchesHeightWidthOutput;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    var gh: u32 = undefined;
    var gw: u32 = undefined;
    var pixels: []f32 = undefined;
    const out_path = if (photo) args[4] else args[5];
    const trace_path: ?[]const u8 = if (photo) (if (args.len == 6) args[5] else null) else (if (args.len == 7) args[6] else null);
    if (photo) {
        const bytes = try std.Io.Dir.cwd().readFileAlloc(init.io, args[3], init.gpa, .limited(32 << 20));
        defer init.gpa.free(bytes);
        const prepared = try q.image.prepare(init.io, init.gpa, bytes, 256);
        gh = prepared.gh;
        gw = prepared.gw;
        pixels = prepared.pixels;
        errdefer init.gpa.free(pixels);
        const patch_path = try std.fmt.allocPrint(init.arena.allocator(), "{s}.patches.f32", .{out_path});
        try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = patch_path, .data = std.mem.sliceAsBytes(pixels) });
        std.debug.print("native photo patches: {d} x {d}, width 1536, f32 little-endian\n", .{ gh, gw });
    } else {
        gh = try std.fmt.parseInt(u32, args[3], 10);
        gw = try std.fmt.parseInt(u32, args[4], 10);
        if (gh == 0 or gw == 0 or gh % 2 != 0 or gw % 2 != 0 or @as(u64, gh) * gw > 16384)
            return error.BadPatchGrid;
        const bytes = try std.Io.Dir.cwd().readFileAlloc(init.io, args[2], init.gpa, .limited(128 << 20));
        defer init.gpa.free(bytes);
        const input = try tf.npy.parse(bytes);
        if (!std.mem.eql(u8, input.descr, "<f4") or input.rank != 2 or input.shape[0] != gh * gw or
            input.shape[1] != 1536) return error.ExpectedPatchMatrix;
        pixels = try init.gpa.alloc(f32, input.count());
        @memcpy(std.mem.sliceAsBytes(pixels), input.data);
    }
    defer init.gpa.free(pixels);
    const tokens = gh * gw / 4;
    const model = try q.Model.load(init.gpa, init.io, args[1]);
    defer model.deinit();
    const vision = try q.vision.Vision.load(init.gpa, model.device, &model.checkpoint, tokens);
    defer vision.deinit();
    const output = try model.device.buffer(@as(usize, tokens) * q.config.hidden * 2, mtl.ResourceOptions.shared);
    defer output.deinit();
    if (trace_path) |path| {
        try std.Io.Dir.cwd().createDirPath(init.io, path);
        var dump: Dump = .{ .io = init.io, .a = init.gpa, .dir = path };
        try vision.imageTrace(model.queue, pixels, gh, gw, .{ .buf = output }, .{ .ctx = &dump, .take = Dump.take });
    } else try vision.image(model.queue, pixels, gh, gw, .{ .buf = output });
    try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = out_path, .data = output.contents()[0..output.length()] });
    std.debug.print("native LFM3-VL tower: {d} image rows, bf16 outputs\n", .{tokens});
}
