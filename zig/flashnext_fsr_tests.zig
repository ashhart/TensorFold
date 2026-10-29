const std = @import("std");
const mtl = @import("metal");
const fz = @import("src/families/flashnext/replay.zig");
const storage = @import("src/families/flashnext/fsr_storage.zig");
const Runtime = @import("src/families/flashnext/fsr_runtime.zig").Runtime;
const File = @import("src/core/fsr/file.zig").File;
const cat = @import("src/core/fsr/catalog.zig");

test "FSR native owner joins forecasts, loads actual multi-row demand and fails before publication on truncation" {
    if (@import("builtin").os.tag != .macos) return error.SkipZigTest;
    var mem: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer mem.deinit();
    const a = mem.allocator();
    mtl.allocator = a;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const io = std.testing.io;
    const stride = 4;
    const tensor_bytes = 512 * stride;
    const bytes = try a.alloc(u8, 49 * 9 * tensor_bytes);
    for (bytes, 0..) |*v, i| v.* = @intCast((i / stride + i % stride) % 251);
    try tmp.dir.writeFile(io, .{ .sub_path = "experts", .data = bytes });
    const path = try tmp.dir.realPathFileAlloc(io, "experts", a);
    const file = try File.open(try std.fmt.allocPrintSentinel(a, "{s}", .{path}, 0));
    defer file.close();
    var r: fz.Run = .{ .arena = a, .device = .{}, .queue = .{}, .xnew = true, .serial = true };
    var name: [256]u8 = undefined;
    for (0..49) |layer| for (cat.projections, 0..) |proj, pi| for (cat.suffixes, 0..) |suffix, ci| {
        const tensor = if (layer < 48)
            try std.fmt.bufPrint(&name, "language_model.model.layers.{d}.mlp.switch_mlp.{s}.{s}", .{ layer, proj, suffix })
        else
            try std.fmt.bufPrint(&name, "mtp.layers.0.mlp.switch_mlp.{s}.{s}", .{ proj, suffix });
        try r.index.put(a, try a.dupe(u8, tensor), .{ .fd = file.fd, .at = (layer * 9 + 3 * pi + ci) * tensor_bytes, .len = tensor_bytes });
    };
    r.window.ssd_experts = 512 * 9 * stride;
    r.fsr = try Runtime.init(&r, "mtp", null, 42);
    const s = r.fsr.?;
    defer r.deinit();
    r.route_pipe = .{ .name = "record-only-route" };
    r.router_pipe = .{ .name = "record-only-router" };
    r.ul = .{ .b = try r.buffer((1 + 2 * 320 + 320 * 32) * 4) };
    const pick: fz.Buf = .{ .b = try r.buffer(20 * 4) };
    const rows = try fz.i32Buf(&r, &.{2});
    const lg: fz.Buf = .{ .b = try r.buffer(2 * 513 * 4) };
    const wts: fz.Buf = .{ .b = try r.buffer(20 * 4) };
    for (pick.b.slice(u32, 20), 0..) |*id, i| id.* = @intCast(i);
    @memset(s.forecast_pick.b.slice(u32, 20), 42);
    const cb = r.queue.commandBuffer();
    r.begin(cb, .serial);
    try s.prepare(&r, 0, lg, rows, pick, wts, 2, lg, lg, false);
    for (0..20) |id| {
        const slot = (try s.pool.?.arena.clock.resident(id)).?;
        for (s.pool.?.buffers, 0..) |buffer, part| {
            const expected = bytes[part * tensor_bytes + id * stride ..][0..stride];
            try std.testing.expectEqualSlices(u8, expected, buffer.b.contents()[slot * stride ..][0..stride]);
        }
    }
    const signals_before = mtl.fsr_fence_updates;
    const waits_before = mtl.fsr_fence_waits;
    try s.after(&r);
    try std.testing.expectEqual(signals_before + 1, mtl.fsr_fence_updates);
    try std.testing.expectEqual(waits_before + 1, mtl.fsr_fence_waits);
    try std.testing.expect(s.commands.pending != null);
    @memset(pick.b.slice(u32, 20), 7);
    try s.prepare(&r, 1, lg, rows, pick, wts, 2, lg, null, false);
    try std.testing.expect(s.commands.pending == null);
    try std.testing.expect(s.pool.?.arena.clock.pending(512 + 42) == null);
    if (try s.pool.?.arena.clock.peek(512 + 42)) |forecast_slot| {
        for (s.pool.?.buffers, 0..) |buffer, part|
            try std.testing.expectEqualSlices(u8, bytes[(9 + part) * tensor_bytes + 42 * stride ..][0..stride], buffer.b.contents()[forecast_slot * stride ..][0..stride]);
    }
    try std.testing.expect(try s.pool.?.arena.clock.resident(512 + 7) != null);
    const actual = (try s.pool.?.arena.clock.resident(512 + 7)).?;
    try std.testing.expectEqualSlices(u8, bytes[9 * tensor_bytes + 7 * stride ..][0..stride], s.pool.?.buffers[0].b.contents()[actual * stride ..][0..stride]);
    s.predicted[0] = 512 + 7;
    s.predicted_count = 1;
    const warm_signals = mtl.fsr_fence_updates;
    try s.after(&r);
    try std.testing.expectEqual(warm_signals, mtl.fsr_fence_updates);
    try std.testing.expect(s.commands.pending == null);
    // Unguarded six-bit rows still read their encoded upper bound when a verdict ignores their results.
    rows.b.slice(u32, 1)[0] = 0;
    try s.prepare(&r, 48, lg, rows, pick, wts, 1, lg, null, true);
    try std.testing.expectEqual(@as(?usize, null), try s.pool.?.arena.clock.resident(48 * 512 + 7));
    try s.prepare(&r, 48, lg, rows, pick, wts, 1, lg, null, false);
    try std.testing.expect(try s.pool.?.arena.clock.resident(48 * 512 + 7) != null);
    rows.b.slice(u32, 1)[0] = 2;
    var m: fz.Model = undefined;
    m.r = &r;
    m.t.mixed, m.t.lg, m.t.pick, m.t.wts, m.t.act, m.t.ydown = .{ lg, lg, pick, wts, lg, lg };
    m.mtp.router, m.mtp.ex, m.mtp.xw = .{ lg, @splat(lg), null };
    // Router output is recorded; selected IDs are owned fixture bytes, not GPU arithmetic results.
    r.skip = 1 << 2;
    for ([_]mtl.DispatchType{ .serial, .concurrent }, 0..) |mode, pass| {
        r.abort();
        r.serial = mode == .serial;
        r.begin(r.queue.commandBuffer(), mode);
        r.rows = 2;
        for (pick.b.slice(u32, 20), 0..) |*id, i| id.* = @intCast(100 + pass * 20 + i);
        try @import("src/families/flashnext/batch_draft.zig").experts(&m, rows, 2);
        try std.testing.expectEqual(mode, s.commands.mode);
        for (pick.b.slice(u32, 20)) |id| {
            const slot = (try s.pool.?.arena.clock.resident(48 * 512 + id)) orelse return error.PackedDraftExpertMissing;
            for (s.pool.?.buffers, 0..) |buffer, part| {
                const expected = bytes[(48 * 9 + part) * tensor_bytes + id * stride ..][0..stride];
                try std.testing.expectEqualSlices(u8, expected, buffer.b.contents()[slot * stride ..][0..stride]);
            }
        }
    }
    @memset(pick.b.slice(u32, 20), 7);
    r.skip = 0;
    try tmp.dir.writeFile(io, .{ .sub_path = "experts", .data = "truncated" });
    try std.testing.expectError(error.CheckpointChanged, s.prepare(&r, 2, lg, rows, pick, wts, 2, lg, null, false));
    try std.testing.expectEqual(@as(?usize, null), try s.pool.?.arena.clock.resident(2 * 512 + 7));
    r.abort();
    try std.testing.expect(s.commands.active == null and s.commands.pending == null);
    r.deinit();
    try std.testing.expect(r.fsr == null and r.held.items.len == 0 and r.closed);
    try std.testing.expect(s.pool.?.arena.worker == null);
}

test "FSR native admission replaces only installed cold loaders and charges all pool metadata" {
    const layout = @import("src/families/flashnext/fsr_layout.zig");
    const weights = @import("flashnext_memory").weights_plan;
    const inv: weights.Inventory = .{ .raw = 1024 * 1024 * 1024, .routed = 800 * 1024 * 1024, .ple = 128 * 1024 * 1024, .expert_parts = @splat(4) };
    const plain = try layout.of(inv, fz.WindowSettings{});
    try std.testing.expectEqual(inv.raw, plain.raw);
    try std.testing.expectEqual(@as(u64, 0), plain.extra);
    const pool = try layout.of(inv, fz.WindowSettings{ .ssd_experts = 512 * 36 });
    try std.testing.expectEqual(inv.raw - inv.routed + try pool.pool.?.charged(), pool.raw);
    try std.testing.expect(pool.extra >= 1024 * 1024);
    try std.testing.expect(pool.pool.?.metadata >= 49 * 512 * @sizeOf(@import("src/core/fsr/arena.zig").Expert));
    const ple = try layout.of(inv, fz.WindowSettings{ .ple_on_ssd = true });
    try std.testing.expectEqual(inv.raw - inv.ple + ple.ple.?, ple.raw);
    try std.testing.expect(ple.pool == null and ple.extra > 0);
    try std.testing.expectError(error.ExpertPoolTooSmall, layout.of(inv, fz.WindowSettings{ .ssd_experts = 511 * 36 }));
    try std.testing.expectError(error.FsrNeedsSingleMetalDevice, (fz.WindowSettings{ .ssd_experts = 512 * 36 }).validate(true));
}

test "FSR native PLE stages duplicate and shard-edge rows without changing logical IDs" {
    if (@import("builtin").os.tag != .macos) return error.SkipZigTest;
    var mem: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer mem.deinit();
    const a = mem.allocator();
    mtl.allocator = a;
    const widths = [3]usize{ 120, 10, 10 };
    const data = try a.alloc(u8, 128 * 4 * 140);
    for (data, 0..) |*v, i| v.* = @intCast(i % 251);
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "ple", .data = data });
    const path = try tmp.dir.realPathFileAlloc(std.testing.io, "ple", a);
    const file = try File.open(try std.fmt.allocPrintSentinel(a, "{s}", .{path}, 0));
    defer file.close();
    var r: fz.Run = .{ .arena = a, .device = .{}, .queue = .{}, .serial = true };
    var name: [256]u8 = undefined;
    var offset: usize = 0;
    for (0..128) |shard| for (cat.suffixes, widths) |suffix, width| {
        const tensor = try std.fmt.bufPrint(&name, "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shard_{d}.{s}", .{ shard, suffix });
        try r.index.put(a, try a.dupe(u8, tensor), .{ .fd = file.fd, .at = offset, .len = 4 * width });
        offset += 4 * width;
    };
    r.window.ple_on_ssd = true;
    r.fsr = try Runtime.init(&r, "unused", null, 42);
    const s = r.fsr.?;
    defer s.deinit();
    const ids: fz.Buf = .{ .b = try r.buffer(32 * 4) };
    const pattern = [_]u32{ 0, 3, 4, 128, 511, 128, 3, 4 };
    const logical = pattern ++ pattern ++ pattern ++ pattern;
    @memcpy(ids.b.slice(u32, 32), &logical);
    r.begin(r.queue.commandBuffer(), .serial);
    const tables = (try s.tables(&r, ids, 2)).?;
    try std.testing.expectEqualSlices(u32, &logical, ids.b.slice(u32, 32));
    for (tables[1].b.slice(u32, 8)) |start| try std.testing.expectEqual(@as(u32, 0), start);
    const gathered = tables[0].b.slice(u32, 32);
    for (logical, gathered) |global, local| {
        var part_at: usize = (global / 4) * 4 * 140;
        for (widths, 0..) |width, part| {
            const got = tables[2 + 7 * 3 + part];
            const expected = data[part_at + (global % 4) * width ..][0..width];
            try std.testing.expectEqualSlices(u8, expected, got.b.contents()[local * width ..][0..width]);
            part_at += 4 * width;
        }
    }
    try std.testing.expectEqual(@as(usize, 5), s.ple.?.table.used);
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "ple", .data = "short" });
    try std.testing.expectError(error.CheckpointChanged, s.tables(&r, ids, 2));
    try std.testing.expectEqual(@as(usize, 0), s.ple.?.table.used);
    r.abort();
}

test "FSR native corrupt checkpoint headers refuse before device allocation" {
    if (@import("builtin").os.tag != .macos) return error.SkipZigTest;
    var mem: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer mem.deinit();
    const a = mem.allocator();
    mtl.allocator = a;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const malformed: [8]u8 = @splat(0xff);
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "bad.safetensors", .data = &malformed });
    const path = try tmp.dir.realPathFileAlloc(std.testing.io, "bad.safetensors", a);
    var r: fz.Run = .{ .arena = a, .device = .{}, .queue = .{} };
    defer for (r.files.items) |fd| {
        _ = @import("sys").c.close(fd);
    };
    try std.testing.expectError(error.BadSafetensors, r.indexFile(try std.fmt.allocPrintSentinel(a, "{s}", .{path}, 0)));
    try std.testing.expectEqual(@as(usize, 0), r.held.items.len);
}

test "FSR native prompt emits every actual stored width on both fragment paths" {
    const ks = @import("kernel_sources");
    const Spec = @import("src/families/flashnext/spec.zig").Spec;
    const prompt = @import("src/families/flashnext/prompt_w.zig");
    const frags = @import("src/core/frags.zig");
    const source = @import("src/families/flashnext/fsr_prompt_source.zig");
    var memory: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer memory.deinit();
    const a = memory.allocator();
    mtl.allocator = a;
    for ([_]bool{ false, true }) |tensor_units| {
        var r: fz.Run = .{ .arena = a, .device = .{ .tensor_units = tensor_units }, .queue = .{} };
        const six = try source.patch(a, ks.flashnext_qmm6, false);
        const assembled = try frags.source(r.device, a, six);
        try std.testing.expect(std.mem.endsWith(u8, assembled, @embedFile("kernels/metal/prefill/qmm6_nax_b.metal")));
        try std.testing.expectEqual(!tensor_units, std.mem.startsWith(u8, assembled, "#define TF_SIMD_FRAGS 1\n"));
        for ([_]usize{ 2, 3, 4, 5, 6, 8 }) |bits| for ([_]usize{ 32, 64, 128 }) |group| {
            const recipe = try Spec.init(bits, group);
            // Both routed gate/up K2560 and down K640 are served with every admitted recipe.
            _ = try recipe.words(fz.D);
            _ = try recipe.words(640);
            r.window.ssd_experts = null;
            const resident = try prompt.compile(&r, @intCast(bits), @intCast(group));
            r.window.ssd_experts = 1;
            const slotted = try prompt.compile(&r, @intCast(bits), @intCast(group));
            const plain = resident.qmm.text;
            const patched = slotted.qmm.text;
            const head = try std.fmt.allocPrint(a, "#define WB {d}\n#define WG {d}\n", .{ bits, group });
            try std.testing.expect(std.mem.indexOf(u8, patched, head) != null);
            try std.testing.expect(std.mem.indexOf(u8, patched, "#include \"../nax.h\"") == null);
            try std.testing.expectEqual(!tensor_units, std.mem.startsWith(u8, patched, "#define TF_SIMD_FRAGS 1\n"));
            const marker = "// y[r] = x[r] W_e^T for rows sorted";
            const before = std.mem.indexOf(u8, plain, marker).?;
            const after = std.mem.indexOf(u8, patched, marker).?;
            try std.testing.expectEqualStrings(plain[0..before], patched[0..after]);
            try std.testing.expectEqual(@as(usize, 2), std.mem.count(u8, patched, "FSR_MAP [[buffer(27)]]"));
            for ([_][]const u8{ "k_loop", "fma(", "store_act<", "frag<float>", "expert_tile<BM>", "row + tm + SM <= M" }) |expr|
                try std.testing.expectEqual(std.mem.count(u8, plain, expr), std.mem.count(u8, patched, expr));
            for (resident.gather, slotted.gather) |old, new| try std.testing.expectEqualStrings(old.name, new.name);
            for (resident.gu, slotted.gu) |old, new| try std.testing.expectEqualStrings(old.name, new.name);
        };
    }
}
