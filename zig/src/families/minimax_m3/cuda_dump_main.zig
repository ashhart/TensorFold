//! tf-minimax-cuda: each layer's rows and the logits on one GPU, or (`rows`) every row's logits on this rank (usage).

const std = @import("std");
const cuda = @import("cuda");
const mm3 = @import("minimax_m3_cuda");

const usage = "usage: tf-minimax-cuda <model dir> <ids.json> <layers> <out dir>   each layer's rows and the logits, one GPU\n       tf-minimax-cuda <model dir> <ids.json> 0 <out dir> rows   every row's logits, teacher-forced; at TP set\n       TF_MINIMAX_RANK, TF_MINIMAX_WORLD and NCCL_COMM_ID on every rank\n";

const Dump = struct {
    d: *const cuda.Driver,
    io: std.Io,
    stream: cuda.Stream,
    dir: std.Io.Dir,
    gpa: std.mem.Allocator,
    hidden: usize,
    chunk: usize = 0,

    fn call(ptr: *anyopaque, layer: usize, resid: u64, rows: usize) anyerror!void {
        const self: *Dump = @ptrCast(@alignCast(ptr));
        try self.stream.synchronize();
        const bytes = try self.gpa.alloc(u8, rows * self.hidden * 4);
        defer self.gpa.free(bytes);
        try self.d.check(self.d.api.cuMemcpyDtoH_v2(bytes.ptr, resid, bytes.len), "cuMemcpyDtoH");
        var name: [32]u8 = undefined;
        try self.dir.writeFile(self.io, .{ .sub_path = try std.fmt.bufPrint(&name, "r{d}.c{d}.f32", .{ layer, self.chunk }), .data = bytes });
    }
};

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const gpa = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    const every = args.len == 6 and std.mem.eql(u8, args[5], "rows");
    if (args.len != 5 and !every) {
        std.debug.print("{s}", .{usage});
        return 2;
    }
    const layers = try std.fmt.parseInt(u32, args[3], 10);
    const text = try std.Io.Dir.cwd().readFileAlloc(io, args[2], a, .limited(1 << 24));
    var doc = try std.json.parseFromSlice(std.json.Value, a, text, .{});
    defer doc.deinit();
    const list = switch (doc.value) {
        .array => |arr| arr,
        .object => |o| (o.get("ids") orelse return error.BadIds).array,
        else => return error.BadIds,
    };
    const ids = try a.alloc(i32, list.items.len);
    for (list.items, ids) |v, *id| id.* = @intCast(v.integer);
    if (ids.len == 0) return error.BadIds;
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var stream = try cuda.Stream.init(&driver, false);
    defer stream.deinit();
    var m = try mm3.forward.Modules.load(gpa, &driver, &ctx);
    defer m.deinit();
    if (every) {
        try everyRow(gpa, io, &driver, stream, &m, args[1], ids, args[4]);
        return 0;
    }
    var c = try mm3.config.Config.read(gpa, io, args[1]);
    if (layers == 0 or layers > c.layers) {
        std.debug.print("layers: 1 to the model's {d}\n", .{c.layers});
        return 2;
    }
    c.layers = layers;
    const share = try mm3.shard.Share.of(&c, 0, 1);
    const which = try a.alloc(u32, layers);
    for (which, 0..) |*l, i| l.* = @intCast(i);
    var t0 = std.Io.Clock.awake.now(io).toNanoseconds();
    var w = try mm3.weights.load(gpa, io, m.tools(&driver, stream), args[1], c, share, which);
    defer w.deinit();
    std.debug.print("loaded {d} layers, {d:.2} GiB in {d:.1} s\n", .{ layers, @as(f64, @floatFromInt(w.bytes)) / (1 << 30), @as(f64, @floatFromInt(std.Io.Clock.awake.now(io).toNanoseconds() - t0)) * 1e-9 });
    const context = std.mem.alignForward(usize, ids.len, 128); // the prompt in chunks, its caches whole
    var b = try mm3.state.Buffers.init(&driver, &w, @min(ids.len, mm3.state.prompt_rows), true, context, 1, &m.lane, &.{}, false);
    defer b.deinit();
    try m.reserve(&c, share, @min(ids.len, mm3.state.prompt_rows), true);
    var st = try mm3.state.State.init(gpa, &driver, &c, share.kv.len(), context);
    defer st.deinit();
    try std.Io.Dir.cwd().createDirPath(io, args[4]);
    var out = try std.Io.Dir.cwd().openDir(io, args[4], .{});
    defer out.close(io);
    var dump: Dump = .{ .d = &driver, .io = io, .stream = stream, .dir = out, .gpa = gpa, .hidden = c.hidden };
    const f: mm3.forward.Forward = .{ .d = &driver, .m = &m, .w = &w, .stream = stream, .b = &b, .watch = .{ .ctx = &dump, .call = Dump.call } };
    t0 = std.Io.Clock.awake.now(io).toNanoseconds();
    var start: usize = 0;
    while (start < ids.len) : (start += mm3.state.prompt_rows) {
        const n = @min(ids.len - start, mm3.state.prompt_rows);
        dump.chunk = start / mm3.state.prompt_rows;
        const seg = [_]mm3.forward.Seg{.{ .st = &st, .row0 = 0, .rows = n }};
        try f.run(ids[start..][0..n], &seg);
        try mm3.forward.Forward.commit(&seg, &.{n});
    }
    try stream.synchronize();
    std.debug.print("forward of {d} rows in {d:.2} s\n", .{ ids.len, @as(f64, @floatFromInt(std.Io.Clock.awake.now(io).toNanoseconds() - t0)) * 1e-9 });
    const logits = try gpa.alloc(u8, share.vocab.len() * 2);
    defer gpa.free(logits);
    try driver.check(driver.api.cuMemcpyDtoH_v2(logits.ptr, b.logits, logits.len), "cuMemcpyDtoH");
    try out.writeFile(io, .{ .sub_path = "logits.bf16", .data = logits });
    var best: f32 = -std.math.inf(f32);
    var bid: usize = 0;
    for (0..share.vocab.len()) |i| {
        const v: f32 = @bitCast(@as(u32, std.mem.readInt(u16, logits[2 * i ..][0..2], .little)) << 16);
        if (v > best) {
            best = v;
            bid = i;
        }
    }
    std.debug.print("greedy {d} ({d:.3})\n", .{ bid, best });
    return 0;
}

/// Every row's logits of this rank's vocabulary share in verify windows: logits.r{rank}.bf16 and its .json.
fn everyRow(gpa: std.mem.Allocator, io: std.Io, driver: *const cuda.Driver, stream: cuda.Stream, m: *const mm3.forward.Modules, dir: []const u8, ids: []const i32, out_dir: []const u8) !void {
    const p = try mm3.native.place();
    var lib: ?cuda.nccl.Library = null;
    var comm: cuda.nccl.Comm = null;
    if (p.world > 1) {
        if (std.c.getenv("NCCL_COMM_ID") == null) return error.NoCommId;
        lib = try cuda.nccl.Library.open();
        var id: cuda.nccl.UniqueId = undefined;
        // NCCL_COMM_ID names rank 0's bootstrap root: every rank's id is that address
        try lib.?.check(lib.?.api.ncclGetUniqueId(&id), "ncclGetUniqueId");
        try lib.?.check(lib.?.api.ncclCommInitRank(&comm, @intCast(p.world), id, @intCast(p.rank)), "ncclCommInitRank");
    }
    defer if (lib) |*l| {
        _ = l.api.ncclCommDestroy(comm);
        l.close();
    };
    const c = try mm3.config.Config.read(gpa, io, dir);
    const share = try mm3.shard.Share.of(&c, p.rank, p.world);
    const which = try gpa.alloc(u32, c.layers);
    defer gpa.free(which);
    for (which, 0..) |*l, i| l.* = @intCast(i);
    var w = try mm3.weights.load(gpa, io, m.tools(driver, stream), dir, c, share, which);
    defer w.deinit();
    const window = mm3.state.window_rows;
    const context = std.mem.alignForward(usize, ids.len + window, 128);
    var b = try mm3.state.Buffers.init(driver, &w, window, false, context, p.world, &m.lane, &.{}, false);
    defer b.deinit();
    try m.reserve(&c, share, window, false);
    var st = try mm3.state.State.init(gpa, driver, &c, share.kv.len(), context);
    defer st.deinit();
    const f: mm3.forward.Forward = .{ .d = driver, .m = m, .w = &w, .stream = stream, .b = &b, .exchange = if (lib) |*l| .{ .nccl = .{ .lib = l, .comm = comm } } else .local };
    const nv = share.vocab.len();
    const all = try gpa.alloc(u8, ids.len * nv * 2);
    defer gpa.free(all);
    var start: usize = 0;
    while (start < ids.len) : (start += window) {
        const n = @min(ids.len - start, window);
        const seg = [_]mm3.forward.Seg{.{ .st = &st, .row0 = 0, .rows = n }};
        try f.run(ids[start..][0..n], &seg);
        try mm3.forward.Forward.commit(&seg, &.{n});
        try stream.synchronize();
        try driver.check(driver.api.cuMemcpyDtoH_v2(all[start * nv * 2 ..].ptr, b.logits, n * nv * 2), "cuMemcpyDtoH");
    }
    try std.Io.Dir.cwd().createDirPath(io, out_dir);
    var out = try std.Io.Dir.cwd().openDir(io, out_dir, .{});
    defer out.close(io);
    var name: [32]u8 = undefined;
    try out.writeFile(io, .{ .sub_path = try std.fmt.bufPrint(&name, "logits.r{d}.bf16", .{p.rank}), .data = all });
    const meta = try std.fmt.allocPrint(gpa, "{{\"rank\": {d}, \"world\": {d}, \"vocab_lo\": {d}, \"vocab_hi\": {d}, \"rows\": {d}}}\n", .{ p.rank, p.world, share.vocab.lo, share.vocab.hi, ids.len });
    defer gpa.free(meta);
    try out.writeFile(io, .{ .sub_path = try std.fmt.bufPrint(&name, "logits.r{d}.json", .{p.rank}), .data = meta });
    std.debug.print("rank {d}: {d} rows, vocabulary [{d}, {d})\n", .{ p.rank, ids.len, share.vocab.lo, share.vocab.hi });
}
