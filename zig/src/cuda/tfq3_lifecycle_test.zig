//! CUDA TFQ3 ownership and malformed input regressions without a GPU.
const std = @import("std");
const tf = @import("tfq3.zig");
const fmt = @import("tfq3_format.zig");
const memory = @import("memory.zig");
const Driver = @import("driver.zig").Driver;
const abi = @import("abi.zig");
const t = std.testing;

const Fake = struct {
    var next: usize = 0;
    var live: [64]bool = @splat(false);
    var duplicate_frees: usize = 0;
    fn alloc(p: *u64, _: usize) callconv(.c) c_int {
        next += 1;
        live[next] = true;
        p.* = next * 256;
        return 0;
    }
    fn free(p: u64) callconv(.c) c_int {
        const i = p / 256;
        if (!live[i]) duplicate_frees += 1;
        live[i] = false;
        return 0;
    }
    fn copy(_: u64, _: ?*const anyopaque, _: usize) callconv(.c) c_int {
        return 0;
    }
    fn sync(_: abi.Stream) callconv(.c) c_int {
        return 0;
    }
    fn driver() Driver {
        next = 0;
        live = @splat(false);
        duplicate_frees = 0;
        var d: Driver = .{ .lib = undefined, .api = undefined };
        d.api.cuMemAlloc_v2 = alloc;
        d.api.cuMemFree_v2 = free;
        d.api.cuMemcpyHtoD_v2 = copy;
        d.api.cuStreamSynchronize = sync;
        return d;
    }
};

fn tables() [fmt.tables_file_bytes]u8 {
    var b: [fmt.tables_file_bytes]u8 = @splat(0);
    std.mem.writeInt(u32, b[0..4], fmt.magic_tables, .little);
    std.mem.writeInt(u32, b[4..8], 3, .little);
    return b;
}

fn shard(duplicate: bool, off: u64) ![]u8 {
    const b = try t.allocator.alloc(u8, 3 * fmt.section_align);
    @memset(b, 0);
    const h: fmt.TfpnHeader = .{
        .magic = fmt.magic_tfpn,
        .version = 3,
        .num_tensors = if (duplicate) 2 else 1,
        .pad0 = 0,
        .off_meta = 64,
        .off_scales = fmt.section_align,
        .off_weights = 2 * fmt.section_align,
        .total_scale_halfs = 1,
        .total_weight_words = fmt.tensorWords(128, 128, 4) + 4,
        .pad1 = 0,
    };
    @memcpy(b[0..64], std.mem.asBytes(&h));
    var m = std.mem.zeroes(fmt.TfpnTensorMeta);
    @memcpy(m.name[0..7], "lm_head");
    m.r = 128;
    m.c = 128;
    m.b_bits = 4;
    m.word_off = off;
    m.word_count = fmt.tensorWords(128, 128, 4);
    @memcpy(b[64..192], std.mem.asBytes(&m));
    if (duplicate) {
        m.name = @splat(0);
        @memcpy(m.name[0..13], "model.lm_head");
        @memcpy(b[192..320], std.mem.asBytes(&m));
    }
    return b;
}

fn directory(tmp: *t.TmpDir) ![:0]u8 {
    return tmp.dir.realPathFileAlloc(t.io, ".", t.allocator);
}

test "CUDA TFQ3: reserve allocation failure leaves the old scratch owned" {
    var d = Fake.driver();
    var k: tf.Kernels = undefined;
    k.d = &d;
    k.xrot = null;
    k.yr = null;
    k.act_rot = null;
    k.xrot_elems = 0;
    k.yr_elems = 0;
    k.act_rot_elems = 0;
    try k.reserve(16, 16, 16);
    defer {
        if (k.xrot) |*b| b.free();
        if (k.yr) |*b| b.free();
        if (k.act_rot) |*b| b.free();
    }
    const before = memory.usage(false);
    const old = k.xrot.?.ptr;
    const limit = try memory.setAllocationLimit(before.device + before.host);
    defer _ = memory.setAllocationLimit(limit) catch unreachable;
    try t.expectError(error.OutOfDeviceMemory, k.reserve(32, 16, 16));
    try t.expect(Fake.live[old / 256]);
    try t.expectEqual(old, k.xrot.?.ptr);
    try t.expectEqual(@as(usize, 16), k.xrot_elems);
    try t.expectEqual(before.device, memory.usage(false).device);
    try t.expectEqual(@as(usize, 0), Fake.duplicate_frees);
}

test "CUDA TFQ3: duplicate canonical name fails with one owner per allocation" {
    var d = Fake.driver();
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try directory(&tmp);
    defer t.allocator.free(dir);
    const tbl = tables();
    try tmp.dir.writeFile(t.io, .{ .sub_path = "pw_v3_tables.bin", .data = &tbl });
    const bytes = try shard(true, 0);
    defer t.allocator.free(bytes);
    try tmp.dir.writeFile(t.io, .{ .sub_path = "dup.tffpw", .data = bytes });
    const before = memory.usage(false).device;
    try t.expectError(error.DuplicateTfq3Tensor, tf.TfpnStore.loadFile(t.allocator, t.io, &d, dir, "dup.tffpw"));
    try t.expectEqual(@as(usize, 0), Fake.duplicate_frees);
    try t.expectEqual(before, memory.usage(false).device);
}

test "CUDA TFQ3: later invalid shard frees prior shard and tables once" {
    var d = Fake.driver();
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try directory(&tmp);
    defer t.allocator.free(dir);
    const tbl = tables();
    try tmp.dir.writeFile(t.io, .{ .sub_path = "pw_v3_tables.bin", .data = &tbl });
    const bytes = try shard(false, 0);
    defer t.allocator.free(bytes);
    try tmp.dir.writeFile(t.io, .{ .sub_path = "a.tffpw", .data = bytes });
    try tmp.dir.writeFile(t.io, .{ .sub_path = "b.tffpw", .data = "broken" });
    const before = memory.usage(false).device;
    try t.expectError(error.InvalidTfq3Shard, tf.TfpnStore.loadDir(t.allocator, t.io, &d, dir, ""));
    try t.expectEqual(@as(usize, 0), Fake.duplicate_frees);
    try t.expectEqual(before, memory.usage(false).device);
}

test "CUDA TFQ3: in-bounds four-bit word offsets require uint4 alignment" {
    const bytes = try shard(false, 1);
    defer t.allocator.free(bytes);
    try t.expectError(error.MisalignedTfq3Tensor, fmt.validateTfpn(bytes));
    var name: []const u8 = "";
    try t.expectError(error.MisalignedTfq3Tensor, fmt.validateTfpnNamed(bytes, &name));
    try t.expectEqualStrings("lm_head", name);
}

test "CUDA TFQ3: tables refuse all active NaN and infinity fields" {
    for ([_]u32{ 0x7f800000, 0xff800000, 0x7fc00001 }) |bits| {
        for (0..6) |i| {
            var b = tables();
            std.mem.writeInt(u32, b[8 + 4 * i ..][0..4], bits, .little);
            try t.expectError(error.InvalidTfq3Tables, fmt.validateTables(&b));
        }
    }
    for ([_]u16{ 0x7c00, 0xfc00, 0x7e01 }) |bits| {
        var off: usize = 64;
        while (off < fmt.tables_min_bytes) : (off += 2) {
            var b = tables();
            std.mem.writeInt(u16, b[off..][0..2], bits, .little);
            try t.expectError(error.InvalidTfq3Tables, fmt.validateTables(&b));
        }
    }
}

test "CUDA TFQ3: orphan tables never select a dense checkpoint" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try directory(&tmp);
    defer t.allocator.free(dir);
    const tbl = tables();
    try tmp.dir.writeFile(t.io, .{ .sub_path = "pw_v3_tables.bin", .data = &tbl });
    try t.expect(!fmt.hasDense(t.allocator, t.io, dir));
    try tmp.dir.writeFile(t.io, .{ .sub_path = "orphan.tffpw", .data = "broken" });
    try t.expect(!fmt.hasDense(t.allocator, t.io, dir));
    const b = try shard(false, 0);
    defer t.allocator.free(b);
    try tmp.dir.writeFile(t.io, .{ .sub_path = "dense.tffpw", .data = b });
    try t.expect(fmt.hasDense(t.allocator, t.io, dir));
}

test "CUDA TFQ3: dense planning fixes reduction order across every window" {
    var k: tf.Kernels = undefined;
    k.yr = .{ .d = undefined, .ptr = 1, .len = 64 * 34816 * 8 * 4 };
    k.yr_elems = 64 * 34816 * 8;
    k.xrot_elems = 512 * 5120;
    for ([_]tf.DenseWeight{
        .{ .words = 0, .scales = 0, .tables = 0, .n = 128, .k = 1024, .bits = 4 },
        .{ .words = 0, .scales = 0, .tables = 0, .n = 34944, .k = 128, .bits = 4 },
    }) |w| {
        const plan = try k.densePlan(w);
        try t.expectEqual(w.n <= 34816, plan.split);
        for (1..65) |rows| try t.expect(@min(rows, plan.cap) != 0);
        for ([_]usize{ 65, 96, 128, 256, 512, 513, 1024, 1536, 4218 }) |rows| try t.expect(@min(rows, plan.cap) != 0);
    }
    k.yr = null;
    try t.expect(!(try k.densePlan(.{ .words = 0, .scales = 0, .tables = 0, .n = 128, .k = 1024, .bits = 4 })).split);
}

test "CUDA TFQ3: reserve rolls all slots back at each allocation failure" {
    for ([_]usize{ 0, 64, 192 }) |headroom| {
        var d = Fake.driver();
        var k: tf.Kernels = undefined;
        k.d = &d;
        k.xrot = null;
        k.yr = null;
        k.act_rot = null;
        k.xrot_elems = 0;
        k.yr_elems = 0;
        k.act_rot_elems = 0;
        try k.reserve(16, 16, 16);
        defer {
            k.xrot.?.free();
            k.yr.?.free();
            k.act_rot.?.free();
        }
        const before = memory.usage(false);
        const old = [_]u64{ k.xrot.?.ptr, k.yr.?.ptr, k.act_rot.?.ptr };
        const limit = try memory.setAllocationLimit(before.device + before.host + headroom);
        defer _ = memory.setAllocationLimit(limit) catch unreachable;
        try t.expectError(error.OutOfDeviceMemory, k.reserve(32, 32, 32));
        for (old) |ptr| try t.expect(Fake.live[ptr / 256]);
        try t.expectEqualSlices(u64, &old, &.{ k.xrot.?.ptr, k.yr.?.ptr, k.act_rot.?.ptr });
        try t.expectEqual(before.device, memory.usage(false).device);
        try t.expectEqual(@as(usize, 0), Fake.duplicate_frees);
    }
}
