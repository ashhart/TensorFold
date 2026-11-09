//! The actual CUDA family facts and shared key reopen real disk states without a GPU.
const std = @import("std");
const cuda = @import("cuda");
const qwen = @import("qwen35");
const disk = @import("cuda_disk.zig");
const fixture = @import("persistence_fixture.zig");
const t = std.testing;

const Driver = struct {
    fn name(out: [*]u8, _: c_int, _: cuda.abi.Device) callconv(.c) cuda.abi.Result {
        @memcpy(out[0..5], "fake\x00");
        return 0;
    }
    fn attribute(out: *c_int, _: cuda.abi.DeviceAttribute, _: cuda.abi.Device) callconv(.c) cuda.abi.Result {
        out.* = 12;
        return 0;
    }
    fn version(out: *c_int) callconv(.c) cuda.abi.Result {
        out.* = 13000;
        return 0;
    }
};

fn ropeMiss(comptime F: type) !void {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(t.io, ".", t.allocator);
    defer t.allocator.free(dir);
    try tmp.dir.writeFile(t.io, .{ .sub_path = "config.json", .data = "{}" });
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    var driver: cuda.Driver = undefined;
    driver.api.cuDeviceGetName = Driver.name;
    driver.api.cuDeviceGetAttribute = Driver.attribute;
    driver.api.cuDriverGetVersion = Driver.version;
    const gpu: cuda.Context = .{ .d = &driver, .device = 0, .handle = null };
    var e: qwen.Engine = undefined;
    e.kv8 = false;
    e.n_taps = 0;
    e.int8.on = false;
    e.g.rope = .plain;
    var target: qwen.reuse.Target = .{ .e = &e, .caches = undefined, .seen = undefined };
    const profile: @import("disk_identity.zig").Profile = .{ .family = F.model_type, .backend = "cuda" };
    const before = try disk.identityOf(arena.allocator(), t.io, &gpu, dir, null, &F.kept_disk.facts(&target), 42, profile);
    const matching = try disk.identityOf(arena.allocator(), t.io, &gpu, dir, null, &F.kept_disk.facts(&target), 42, profile);
    e.g.rope = .{ .yarn = .{ .factor = 4, .trained = 262144, .table = @splat(1) } };
    const scaled = try disk.identityOf(arena.allocator(), t.io, &gpu, dir, null, &F.kept_disk.facts(&target), 42, profile);
    e.g.rope.yarn.table[17] = 2;
    const changed_table = try disk.identityOf(arena.allocator(), t.io, &gpu, dir, null, &F.kept_disk.facts(&target), 42, profile);
    try fixture.misses(before, matching, &.{ scaled, changed_table });
    try t.expect(scaled != changed_table);
}

test "disk identity: dense Qwen CUDA rotary policy misses with the same probe" {
    try ropeMiss(qwen.native);
}

test "disk identity: MoE Qwen CUDA rotary policy misses with the same probe" {
    try ropeMiss(qwen.native_moe);
}

fn indexedMiss(comptime family: []const u8) !void {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const dir = try tmp.dir.realPathFileAlloc(t.io, ".", a);
    try tmp.dir.createDirPath(t.io, "tensors");
    try tmp.dir.writeFile(t.io, .{ .sub_path = "config.json", .data = "{}" });
    try tmp.dir.writeFile(t.io, .{ .sub_path = "model.safetensors.index.json", .data = "{\"weight_map\":{\"unprobed.weight\":\"tensors/native-QB.safetensors\"}}" });
    const header = "{\"unprobed.weight\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}";
    var shard: [8 + header.len + 4]u8 = @splat(0);
    std.mem.writeInt(u64, shard[0..8], header.len, .little);
    @memcpy(shard[8..][0..header.len], header);
    try tmp.dir.writeFile(t.io, .{ .sub_path = "tensors/native-QB.safetensors", .data = &shard });
    var driver: cuda.Driver = undefined;
    driver.api.cuDeviceGetName = Driver.name;
    driver.api.cuDeviceGetAttribute = Driver.attribute;
    driver.api.cuDriverGetVersion = Driver.version;
    const gpu: cuda.Context = .{ .d = &driver, .device = 0, .handle = null };
    const profile: @import("disk_identity.zig").Profile = .{ .family = family, .backend = "cuda" };
    const before = try disk.identityOf(a, t.io, &gpu, dir, null, &.{1}, 42, profile);
    const same = try disk.identityOf(a, t.io, &gpu, dir, null, &.{1}, 42, profile);
    shard[shard.len - 1] = 1;
    try fixture.rewrite(tmp.dir, "tensors/native-QB.safetensors", &shard);
    try fixture.misses(before, same, &.{try disk.identityOf(a, t.io, &gpu, dir, null, &.{1}, 42, profile)});
    try tmp.dir.deleteFile(t.io, "tensors/native-QB.safetensors");
    try t.expectError(error.IndexedShardMissing, disk.identityOf(a, t.io, &gpu, dir, null, &.{1}, 42, profile));
}

test "disk identity: Nemotron CUDA indexed shard misses with the same probe" {
    try indexedMiss("nemotron_h");
}
test "disk identity: dense Qwen CUDA indexed shard misses with the same probe" {
    try indexedMiss("qwen3_5");
}
test "disk identity: MoE Qwen CUDA indexed shard misses with the same probe" {
    try indexedMiss("qwen3_5_moe");
}
test "disk identity: Flash Next CUDA indexed shard misses with the same probe" {
    try indexedMiss("qwen4_exp");
}

/// A checkpoint folder and a recording device: what the shared key reads besides a family's words.
const Fixture = struct {
    tmp: std.testing.TmpDir,
    arena: std.heap.ArenaAllocator,
    dir: []const u8,
    driver: cuda.Driver,

    fn init(x: *Fixture) !void {
        x.tmp = t.tmpDir(.{});
        x.arena = .init(t.allocator);
        x.dir = try x.tmp.dir.realPathFileAlloc(t.io, ".", x.arena.allocator());
        try x.tmp.dir.writeFile(t.io, .{ .sub_path = "config.json", .data = "{}" });
        x.driver.api.cuDeviceGetName = Driver.name;
        x.driver.api.cuDeviceGetAttribute = Driver.attribute;
        x.driver.api.cuDriverGetVersion = Driver.version;
    }
    fn deinit(x: *Fixture) void {
        x.arena.deinit();
        x.tmp.cleanup();
    }
    fn gpu(x: *Fixture) cuda.Context {
        return .{ .d = &x.driver, .device = 0, .handle = null };
    }
};

const flash = @import("flashnext").native;

fn flashKey(x: *Fixture, c: *flash.Cuda) !u64 {
    const g = x.gpu();
    const profile: @import("disk_identity.zig").Profile = .{ .family = flash.model_type, .backend = "cuda" };
    return disk.keptKey(flash.kept_disk, x.arena.allocator(), t.io, &g, x.dir, null, @as(*anyopaque, c), 42, profile);
}

test "disk identity: Flash Next CUDA resolved prompt rows, kernel forms and expert grids miss with the same probe" {
    var x: Fixture = undefined;
    try x.init();
    defer x.deinit();
    var e: @typeInfo(@FieldType(flash.Cuda, "engine")).pointer.child = undefined;
    e.chunk = 8192;
    e.busy_chunk = 1536;
    const k = &e.batch_k;
    k.forms = .{};
    k.expert_tile = null;
    k.native_lane.prompt = null;
    k.native_lane.prompt_sms = 0;
    k.native_lane.half_below = cuda.qmmf.half_below_default;
    k.nvfp4.warps = 4;
    k.nvfp4.resident = @splat(48);
    k.fp8.resident = @splat(@splat(48));
    var c: flash.Cuda = undefined;
    c.engine = &e;
    c.head = null;
    c.set = null;
    c.glue.mixed = 0;
    c.depth = 6;
    c.parts.on = true;
    const before = try flashKey(&x, &c);
    var changed: std.ArrayList(u64) = .empty;
    defer changed.deinit(t.allocator);
    inline for (.{ "chunk", "busy_chunk" }) |field| { // another chunk height, solo or beside a decoding stream
        @field(e, field) += 512;
        try changed.append(t.allocator, try flashKey(&x, &c));
        @field(e, field) -= 512;
    }
    inline for (.{ "gdn_fused", "hc_fuse", "in_pass" }) |field| {
        @field(k.forms, field) = false;
        try changed.append(t.allocator, try flashKey(&x, &c));
        @field(k.forms, field) = true;
    }
    inline for (.{ "readout_rows", "fact_rows" }) |field| {
        @field(k.forms, field) = std.math.maxInt(usize); // 0 set: never the tiles
        try changed.append(t.allocator, try flashKey(&x, &c));
        @field(k.forms, field) = 64;
    }
    k.expert_tile = .{ .mod = undefined, .fns = undefined, .resident = .{ 64, 64 }, .selection = .up };
    const up = try flashKey(&x, &c);
    k.expert_tile.?.selection = .both; // the same grids: the selection alone moves the key
    const both = try flashKey(&x, &c);
    try t.expect(up != both);
    try changed.appendSlice(t.allocator, &.{ up, both });
    k.expert_tile = null;
    k.native_lane.prompt = @as([2][2]cuda.Function, undefined); // TF_FLASHNEXT_PROMPT_BF16's tiles loaded
    const tiles = try flashKey(&x, &c);
    k.native_lane.prompt_sms = 48; // the SMs those tiles' blocks are weighed against, on the same expert grids
    const weighed = try flashKey(&x, &c);
    try t.expect(tiles != weighed);
    try changed.appendSlice(t.allocator, &.{ tiles, weighed });
    k.native_lane.prompt = null;
    k.native_lane.prompt_sms = 0;
    k.native_lane.half_below = 0; // TF_QMMF_HALF_BELOW=0: a short pass's BF16 and MXFP8 rows on the 64-column tile
    try changed.append(t.allocator, try flashKey(&x, &c));
    k.native_lane.half_below = cuda.qmmf.half_below_default;
    k.nvfp4.warps = 1;
    try changed.append(t.allocator, try flashKey(&x, &c));
    k.nvfp4.warps = 4;
    k.nvfp4.resident[0] = 47; // another SM count under the same device name
    try changed.append(t.allocator, try flashKey(&x, &c));
    k.nvfp4.resident[0] = 48;
    c.glue.mixed = 3; // without a captured set our glue runs every entry anyway
    try t.expectEqual(before, try flashKey(&x, &c));
    c.set = @as(cuda.aot.Set, undefined);
    try changed.append(t.allocator, try flashKey(&x, &c));
    c.glue.mixed = 0;
    try changed.append(t.allocator, try flashKey(&x, &c));
    c.set = null;
    c.depth = 3; // a draft chain's length and lane buffers' reuse touch no prompt pass
    c.parts.on = false;
    try fixture.misses(before, try flashKey(&x, &c), changed.items);
}

test "disk identity: Flash Next CUDA prompt forms take each setting's value at load" {
    const env = @import("sys").env;
    const Engine = @typeInfo(@FieldType(flash.Cuda, "engine")).pointer.child;
    const Forms = @FieldType(@FieldType(Engine, "batch_k"), "forms");
    const names = [_][:0]const u8{ "TF_FLASHNEXT_GDN_FUSED", "TF_FLASHNEXT_HC_FUSE", "TF_FLASHNEXT_KEEP_INPASS", "TF_FLASHNEXT_READOUT_ROWS", "TF_FLASHNEXT_FACT_ROWS" };
    for (names) |n| _ = env.unsetenv(n);
    defer for (names) |n| {
        _ = env.unsetenv(n);
    };
    try t.expectEqualDeep(Forms{}, Forms.resolve());
    for (names) |n| _ = env.setenv(n, "0", 1);
    const never = std.math.maxInt(usize);
    try t.expectEqualDeep(Forms{ .gdn_fused = false, .hc_fuse = false, .in_pass = false, .readout_rows = never, .fact_rows = never }, Forms.resolve());
    _ = env.setenv("TF_FLASHNEXT_READOUT_ROWS", "128", 1);
    _ = env.setenv("TF_FLASHNEXT_FACT_ROWS", "32", 1);
    try t.expectEqual(@as(usize, 128), Forms.resolve().readout_rows);
    try t.expectEqual(@as(usize, 32), Forms.resolve().fact_rows);
}

const nemotron = @import("nemotron");

fn nemotronKey(x: *Fixture, e: anytype) !u64 {
    const g = x.gpu();
    const profile: @import("disk_identity.zig").Profile = .{ .family = nemotron.native.model_type, .backend = "cuda" };
    return disk.withPolicy(try disk.identityOf(x.arena.allocator(), t.io, &g, x.dir, null, &.{0}, 42, profile), e);
}

test "disk identity: Nemotron CUDA resolved checkpoint math, expert ranks and grids miss with the same probe" {
    var x: Fixture = undefined;
    try x.init();
    defer x.deinit();
    const Target = nemotron.native.kept_states.Target;
    var e: @typeInfo(@FieldType(Target, "e")).pointer.child = undefined;
    e.c.ck = false;
    e.c.world = 1;
    e.k.experts.resident = @splat(48);
    e.k.modelopt = null;
    e.k.ck = null;
    e.k.triton = null;
    e.segments = 1;
    e.nch = 32;
    const before = try nemotronKey(&x, &e);
    var changed: std.ArrayList(u64) = .empty;
    defer changed.deinit(t.allocator);
    e.c.ck = true; // --precision checkpoint
    try changed.append(t.allocator, try nemotronKey(&x, &e));
    e.c.ck = false;
    e.c.world = 2; // TF_NEMOTRON_WORLD=2: the experts' sums over two ranks
    try changed.append(t.allocator, try nemotronKey(&x, &e));
    e.c.world = 1;
    e.k.experts.resident[2] = 47;
    try changed.append(t.allocator, try nemotronKey(&x, &e));
    e.k.experts.resident[2] = 48;
    var m: @typeInfo(@FieldType(@TypeOf(e.k), "modelopt")).optional.child = undefined;
    m.experts.warps = 4;
    m.experts.resident = @splat(48);
    m.lane.half_below = cuda.qmmf.half_below_default;
    e.k.modelopt = m;
    const four = try nemotronKey(&x, &e);
    e.k.modelopt.?.experts.warps = 1; // TF_NVFP4_DECODE_WARPS=1 on the same grids
    const one = try nemotronKey(&x, &e);
    try t.expect(four != one);
    try changed.appendSlice(t.allocator, &.{ four, one });
    e.k.modelopt.?.experts.warps = 4;
    e.k.modelopt.?.lane.half_below = 0; // TF_QMMF_HALF_BELOW=0: a short pass's rows on the 64-column tile
    const full_tile = try nemotronKey(&x, &e);
    try t.expect(four != full_tile);
    try changed.append(t.allocator, full_tile);
    e.k.modelopt = null;
    var ck: @typeInfo(@FieldType(@TypeOf(e.k), "ck")).optional.child = undefined;
    ck.resident = @splat(@splat(48));
    ck.prompt_resident = @splat(48);
    e.k.ck = ck;
    try changed.append(t.allocator, try nemotronKey(&x, &e));
    e.k.ck = null;
    e.k.triton = @as(cuda.aot.Set, undefined); // a captured set: its NCH variant follows --context
    const short = try nemotronKey(&x, &e);
    e.nch = 64;
    const long = try nemotronKey(&x, &e);
    try t.expect(short != long);
    try changed.appendSlice(t.allocator, &.{ short, long });
    e.k.triton = null;
    e.segments = 4; // staggered segments run only passes that keep no state
    e.nch = 64; // a longer --context on our glue: more chunk programs a row, whose merge reads only its own
    try fixture.misses(before, try nemotronKey(&x, &e), changed.items);
}
