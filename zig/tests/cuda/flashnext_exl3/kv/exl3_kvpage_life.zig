//! exl3-kvpage snapshot and trace: a paged sequence saved and restored, and the selection trace run.

const std = @import("std");
const cuda = @import("cuda");
const check = @import("../../check.zig");
const flash = @import("flashnext");
const Gpu = check.Gpu;
const prompt = flash.prompt;
const xpre = flash.exl3.prefill;
const kvp = flash.exl3.kvpage;
const engs = @import("exl3_kvpage_eng.zig");

const Eng = engs.Eng;
const parseList = engs.parseList;
const readIds = engs.readIds;

pub fn snapshotLifecycle(gpu: Gpu) !void {
    const snaps = @import("flashnext").snapshot;
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();
    var engine: prompt.Engine = undefined;
    engine.gpa = gpu.gpa;
    engine.driver = @constCast(gpu.d);
    engine.stream = &stream;
    engine.chunk = 256;
    engine.eos = 0;
    engine.round_graph = .{};
    engine.page_config = .{ .window = 2052, .slots = 16, .rows = 256 };
    engine.shared_arena = null;
    defer flash.exl3.recycle.drain(&engine);
    var destination = try engine.newSeq(8192);
    defer destination.deinit(gpu.gpa);
    for ([_]usize{ 0, 1, 3, 4, 2051, 2052, 4096, 4099, 6147 }) |at| {
        var source: ?prompt.Seq = try engine.newSeq(8192);
        defer if (source) |*sequence| sequence.deinit(gpu.gpa);
        const pg = source.?.paged.?;
        source.?.fed = at;
        source.?.history = .{ @intCast(at), 73 };
        for (0..12) |layer| {
            const origin = (at -| 2052) / 4 * 4;
            pg.win0[layer] = origin;
            pg.rebases[layer] = layer + 3;
            pg.written[layer] = origin;
            for (0..2) |side| {
                const host = pg.hostMut(layer, side == 1);
                const words: []u16 = std.mem.bytesAsSlice(u16, @as([]align(2) u8, @alignCast(host)));
                for (words, 0..) |*word, index| word.* = @truncate(index + layer * 997 + side * 32768);
            }
        }
        try pg.dev.fill8(0xFF, stream.handle);
        const words = try gpu.gpa.alloc(u32, pg.tabs.len / 4);
        defer gpu.gpa.free(words);
        try pg.tabs.download(0, std.mem.sliceAsBytes(words));
        const ctl_at = 24 * pg.nblocks;
        const inverse_at = ctl_at + 12 * 8;
        for (0..12) |layer| {
            words[ctl_at + layer * 8] = 73;
            words[ctl_at + layer * 8 + 6] = @intCast(31 + layer);
            if (at >= 4) {
                words[layer * pg.nblocks] = @intCast(layer);
                words[12 * pg.nblocks + layer * pg.nblocks] = @intCast(31 + layer);
                words[inverse_at + layer] = @intCast(layer * pg.nblocks);
                words[inverse_at + pg.cfg.slots + layer] = @intCast(19 + layer);
            }
        }
        try pg.tabs.upload(0, std.mem.sliceAsBytes(words));
        const saved = try snaps.save(gpu.gpa, gpu.d, stream, &source.?, null);
        defer snaps.drop(gpu.gpa, stream, saved);
        source.?.deinit(gpu.gpa);
        source = null;
        for (0..2) |_| {
            try snaps.restore(stream, &destination, null, saved);
            const again = try snaps.save(gpu.gpa, gpu.d, stream, &destination, null);
            defer snaps.drop(gpu.gpa, stream, again);
            try check.expect(std.meta.eql(saved.paged.?, again.paged.?), "snapshot pager counters at {d}", .{at});
            try check.expect(std.mem.eql(u8, saved.host.?.bytes[0..saved.buf.len], again.host.?.bytes[0..again.buf.len]), "snapshot exact BF16, windows, slots, tables and ordinary state at {d}", .{at});
        }
    }
    check.pass("BF16 paged snapshot source-drop, distinct destination, repeated restore at nine boundaries", .{});
}

pub fn traceRun(gpu: Gpu, dir: []const u8, rest: []const [:0]const u8) !void {
    const a = gpu.gpa;
    if (rest.len < 4) return error.MissingArgument;
    const all = try readIds(gpu, rest[0]);
    defer a.free(all);
    var ns_buf: [16]usize = undefined;
    const ns = ns_buf[0..try parseList(rest[1], &ns_buf)];
    const steps = try std.fmt.parseInt(usize, rest[2], 10);
    const out_prefix = rest[3];
    const with_prompt = rest.len > 4 and std.mem.eql(u8, rest[4], "prompt");
    var max_n: usize = 0;
    for (ns) |n| max_n = @max(max_n, n);
    if (max_n > all.len) return error.NotEnoughIds;
    const o = xpre.optionsFromEnv();
    const g = try Eng.init(gpu, dir, max_n + steps + 64, if (o.enabled) @min(o.rows, (max_n + 255) / 256 * 256) else 0);
    defer g.deinit();
    const toks = try a.alloc(u32, steps);
    defer a.free(toks);
    for (ns) |n| {
        var t = try kvp.Trace.init(a, with_prompt);
        defer t.deinit();
        kvp.trace = &t;
        defer kvp.trace = null;
        const ms_p = try g.prefill(all[0..n]);
        const prompt_rows = t.rows;
        const ms_d = try g.decode(toks);
        std.debug.print("TRACE {d} tokens: prefill {d:.0} ms (traced, {d} rows recorded), {d} decode rounds {d:.1} ms/round (traced: one sync a layer); {d} rows recorded; {d:.1} MiB\n", .{ n, ms_p, prompt_rows, steps, ms_d, t.rows, @as(f64, @floatFromInt(t.out.items.len)) / (1 << 20) });
        std.debug.print("  greedy: {any}\n", .{toks[0..@min(steps, 48)]});
        var buf: [512]u8 = undefined;
        const path = try std.fmt.bufPrint(&buf, "{s}-{d}.bin", .{ out_prefix, n });
        try std.Io.Dir.cwd().writeFile(gpu.io, .{ .sub_path = path, .data = t.out.items });
    }
    check.pass("exl3-kvpage trace written", .{});
}
