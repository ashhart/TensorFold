//! Flash Next TFQ3 shard checks against a valid control and named refusals, and the checked sidecar scan (no GPU).
const std = @import("std");
const check = @import("tfq3_check.zig");
const files = @import("tfq3_files.zig");
const t = std.testing;

const at_meta: usize = 16384;
const at_scales: usize = 49152;
const at_weights: usize = 65536;
/// One gate, up and down room at 5 bits: every expert's records point at them (a small valid shard).
const proj_words: usize = 640 * 2560 * 5 / 32;
const file_len: usize = at_weights + 3 * proj_words * 4;

fn header(layer: u32) check.LayerHeader {
    var h = std.mem.zeroes(check.LayerHeader);
    h.magic = check.magic_layer;
    h.version = check.format_version;
    h.layer_idx = layer;
    h.num_experts = check.num_experts;
    h.num_tensors = 3 * check.num_experts;
    h.hidden_size = check.hidden_size;
    h.moe_inter = check.moe_inter;
    h.stream_words = 3 * proj_words;
    h.scale_halfs = 300;
    h.off_meta = at_meta;
    h.off_scales = at_scales;
    h.off_weights = at_weights;
    return h;
}

fn expert(bits: u32) check.ExpertMeta {
    var e = std.mem.zeroes(check.ExpertMeta);
    e.b_bits = bits;
    e.up_word_off = proj_words;
    e.down_word_off = 2 * proj_words;
    e.up_sc_off = 100;
    e.down_sc_off = 200;
    return e;
}

/// A whole shard in `buf`: header, 512 expert records (`e` each; expert 7 at 5 bits), zero sections.
fn shard(buf: []u8, h: check.LayerHeader, e: check.ExpertMeta) []u8 {
    @memset(buf, 0);
    @memcpy(buf[0..@sizeOf(check.LayerHeader)], std.mem.asBytes(&h));
    const off: usize = @intCast(@min(h.off_meta, buf.len));
    for (0..check.num_experts) |i| {
        var m = e;
        if (i == 7 and e.b_bits == 4) m.b_bits = 5;
        const at = off + i * @sizeOf(check.ExpertMeta);
        if (at + @sizeOf(check.ExpertMeta) <= buf.len) @memcpy(buf[at..][0..@sizeOf(check.ExpertMeta)], std.mem.asBytes(&m));
    }
    return buf;
}

test "Flash Next TFQ3 check: a valid shard passes, and each bad header, section or record is refused by name" {
    const buf = try t.allocator.alloc(u8, file_len);
    defer t.allocator.free(buf);
    const ok = try check.layer(shard(buf, header(3), expert(4)), 3);
    try t.expect(ok.has_5b);
    try t.expectEqual(at_weights, ok.weights.off);
    try t.expect(!(try check.layer(shard(buf, header(3), expert(3)), 3)).has_5b);
    // the MTP shard is checked whole with no layer to match
    _ = try check.layer(shard(buf, header(47), expert(4)), null);
    // metadata off its 8-byte alignment is decoded by copy, never read through a misaligned pointer (E332)
    var odd = header(3);
    odd.off_meta = 145;
    _ = try check.layer(shard(buf, odd, expert(4)), 3);
    // E332: layer 1's shard named layer_00.tffpw
    try t.expectError(error.Tfq3Layer, check.layer(shard(buf, header(1), expert(4)), 0));
    const Case = struct { anyerror, check.LayerHeader, check.ExpertMeta };
    var cases: [13]Case = undefined;
    for (&cases) |*c| c.* = .{ error.Tfq3Truncated, header(3), expert(4) };
    cases[0][1].magic = 0xDEADBEEF;
    cases[0][0] = error.UnsupportedQuantization;
    cases[1][1].num_tensors = 1;
    cases[1][0] = error.Tfq3Geometry;
    cases[2][1].num_experts = 256;
    cases[2][0] = error.Tfq3Geometry;
    cases[3][1].hidden_size = 4096;
    cases[3][0] = error.Tfq3Geometry;
    cases[4][1].off_meta = std.math.maxInt(u64) - 16383; // wrapped to 16384 unchecked (E332)
    cases[4][0] = error.Tfq3Overflow;
    cases[5][1].scale_halfs = @as(u64, 1) << 63; // times two wrapped to zero unchecked (E332)
    cases[5][0] = error.Tfq3Overflow;
    cases[6][1].stream_words = @as(u64, 1) << 62; // times four wrapped to zero unchecked (E332)
    cases[6][0] = error.Tfq3Overflow;
    cases[7][1].off_weights = at_scales;
    cases[7][0] = error.Tfq3Overlap;
    cases[8][1].off_meta = 100;
    cases[8][0] = error.Tfq3Overlap;
    cases[9][1].stream_words = 3 * proj_words + 1; // past the file
    cases[10][2].b_bits = 6;
    cases[10][0] = error.Tfq3Bits;
    cases[11][2].down_word_off = 3 * proj_words - 640 * 2560 * 4 / 32 + 4; // the 4-bit down's last uint4 past it
    cases[12][2].down_sc_off += 1;
    for (cases) |c| try t.expectError(c[0], check.layer(shard(buf, c[1], c[2]), 3));
}

test "Flash Next TFQ3 check: E332's 65,540-byte shard cannot back its experts' matrices" {
    const buf = try t.allocator.alloc(u8, 65540);
    defer t.allocator.free(buf);
    var h = header(0);
    h.scale_halfs = 1;
    h.stream_words = 1;
    try t.expectError(error.Tfq3Bits, check.layer(shard(buf, h, std.mem.zeroes(check.ExpertMeta)), 0)); // width 0
    var e = std.mem.zeroes(check.ExpertMeta);
    e.b_bits = 4; // every projection at offset zero: 204,800 words wanted, one present
    try t.expectError(error.Tfq3Truncated, check.layer(shard(buf, h, e), 0));
}

test "Flash Next TFQ3 check: the table file is version 3, the kernels' whole table, its values finite" {
    var tbl: [16384]u8 = @splat(0);
    std.mem.writeInt(u32, tbl[0..4], check.magic_v3_tables, .little);
    std.mem.writeInt(u32, tbl[4..8], check.format_version, .little);
    try check.tables(&tbl, tbl.len); // a published file, padded to 16 KiB
    try check.tables(tbl[0..check.tables_bytes], check.tables_bytes); // the table alone: main's 16 KiB floor refused it
    try t.expectError(error.Tfq3Table, check.tables(tbl[0 .. check.tables_bytes - 2], check.tables_bytes - 2));
    var bad = tbl;
    std.mem.writeInt(u32, bad[8..12], @bitCast(std.math.nan(f32)), .little); // a3
    try t.expectError(error.Tfq3Table, check.tables(&bad, bad.len));
    bad = tbl;
    std.mem.writeInt(u16, bad[9280..][0..2], 0x7c00, .little); // +Inf in c64_3
    try t.expectError(error.Tfq3Table, check.tables(&bad, bad.len));
    std.mem.writeInt(u32, tbl[4..8], 2, .little);
    try t.expectError(error.UnsupportedQuantization, check.tables(&tbl, tbl.len));
}

test "Flash Next TFQ3 check: a stray table file beside a checkpoint with no shards turns nothing on" {
    const io = t.io;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try std.fmt.allocPrint(t.allocator, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    defer t.allocator.free(dir);
    try tmp.dir.writeFile(io, .{ .sub_path = "pw_v3_tables.bin", .data = "not a table" });
    // engine_load maps pw_v3_tables.bin only when hasRecords: here it is neither mapped nor read, and the scan is clean
    try t.expect(!files.hasRecords(io, dir, 48));
    try t.expect(!(try files.scan(io, dir, 48)).any);
    const buf = try t.allocator.alloc(u8, file_len);
    defer t.allocator.free(buf);
    try tmp.dir.writeFile(io, .{ .sub_path = "mtp_00.tffpw", .data = shard(buf, header(48), expert(4)) });
    try t.expect(files.hasRecords(io, dir, 48));
}

test "Flash Next TFQ3 check: the sidecar scan admits checked shards only, and names a bad one" {
    const io = t.io;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try std.fmt.allocPrint(t.allocator, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    defer t.allocator.free(dir);
    const buf = try t.allocator.alloc(u8, file_len);
    defer t.allocator.free(buf);
    // no sidecars: nothing present, nothing refused
    try t.expect(!(try files.scan(io, dir, 48)).any);
    try tmp.dir.writeFile(io, .{ .sub_path = "layer_00.tffpw", .data = shard(buf, header(0), expert(4)) });
    // a valid shard without the shared table is refused
    try t.expectError(error.Tfq3Table, files.scan(io, dir, 48));
    var tbl: [check.tables_bytes]u8 = @splat(0);
    std.mem.writeInt(u32, tbl[0..4], check.magic_v3_tables, .little);
    std.mem.writeInt(u32, tbl[4..8], check.format_version, .little);
    try tmp.dir.writeFile(io, .{ .sub_path = "pw_v3_tables.bin", .data = &tbl });
    try tmp.dir.writeFile(io, .{ .sub_path = "mtp_00.tffpw", .data = shard(buf, header(48), expert(4)) });
    const p = try files.scan(io, dir, 48);
    try t.expect(p.any and p.layers[0] and !p.layers[1] and p.mtp);
    try t.expectEqual(@as(u64, 2 * file_len), p.routed_bytes);
    try t.expectEqual(@as(u64, 2 * file_len + check.tables_bytes), p.total_bytes);
    // E332: a 16-byte sidecar relaxed admission by its presence alone; now it is refused before any construction
    try tmp.dir.writeFile(io, .{ .sub_path = "layer_01.tffpw", .data = "0123456789abcdef" });
    try t.expectError(error.Tfq3Truncated, files.scan(io, dir, 48));
    // a layer file holding another layer's shard
    try tmp.dir.writeFile(io, .{ .sub_path = "layer_01.tffpw", .data = shard(buf, header(0), expert(4)) });
    try t.expectError(error.Tfq3Layer, files.scan(io, dir, 48));
    // a layer past the model's is not scanned
    try t.expect((try files.scan(io, dir, 1)).layers[0]);
}

test "Flash Next TFQ3 check: a 4-bit projection's words start on the 16 bytes the kernels' uint4 loads read (E333)" {
    const buf = try t.allocator.alloc(u8, file_len);
    defer t.allocator.free(buf);
    // E333's witness and its siblings: each 4-bit projection one word off is refused by name
    for (0..3) |p| {
        var e = expert(4);
        switch (p) {
            0 => e.gate_word_off = 1,
            1 => e.up_word_off += 1,
            else => e.down_word_off += 2,
        }
        try t.expectError(error.Tfq3Misaligned, check.layer(shard(buf, header(3), e), 3));
    }
    // controls: gate at 0 is E333's aligned control, and 3/5-bit words read one at a time take any offset
    var aligned = expert(4);
    aligned.gate_word_off = 4;
    _ = try check.layer(shard(buf, header(3), aligned), 3);
    for ([_]u32{ 3, 5 }) |bits| {
        var odd = expert(bits);
        odd.gate_word_off = 1;
        odd.up_word_off += 1;
        odd.down_word_off = 3 * proj_words - 640 * 2560 * bits / 32 - 1;
        _ = try check.layer(shard(buf, header(3), odd), 3);
    }
}

test "Flash Next TFQ3 check: each projection is checked whole before the next, so a gate past its section wins" {
    const buf = try t.allocator.alloc(u8, file_len);
    defer t.allocator.free(buf);
    // gate on a uint4 but past the words, up off one: CUDA's all-alignments-first order refuses this as misaligned
    var e = expert(4);
    e.gate_word_off = 3 * proj_words;
    e.up_word_off += 1;
    try t.expectError(error.Tfq3Truncated, check.layer(shard(buf, header(3), e), 3));
}
