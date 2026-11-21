//! ZONOS3's phoneme, start and pause tokens (zonos3.tts.superbpe.payload_ids): the text around its markers.
const std = @import("std");
const errors = @import("errors.zig");
const text_mod = @import("model_text.zig");
const req = @import("zonos_request");
const Allocator = std.mem.Allocator;

const prefix = "ZONOS3 request: ";
/// A pronunciation group opens and closes with U+E000 and U+E001; U+E002 is the start marker; U+E003 on, a pause.
pub const group_open = "\xee\x80\x80";
pub const group_close = "\xee\x80\x81";
pub const start_marker = "\xee\x80\x82";
const first_pause = 0x83; // the last byte of U+E003
const last_marker = 0x8a; // U+E00A

/// The release's extra text rows by what they are (text_rows/pause_tokens_s5000.json): text-table row ids.
pub const Spec = struct {
    open_id: u32,
    close_id: u32,
    /// The IPA code points a group may hold: symbol i is row open_id + 2 + i.
    symbols: []const u21,
    start_id: u32,
    /// Row ids of the pause markers U+E003 + j.
    pause_ids: []const u32,

    /// Where a release keeps the spec, under its lm folder; null when it has none.
    pub fn load(a: Allocator, io: std.Io, lm_dir: []const u8) !?Spec {
        const path = try std.fs.path.join(a, &.{ lm_dir, "text_rows", "pause_tokens_s5000.json" });
        const text = std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 22)) catch |e| switch (e) {
            error.FileNotFound => return null,
            else => return e,
        };
        const root = try std.json.parseFromSliceLeaky(std.json.Value, a, text, .{});
        if (root != .object) return error.BadTextRows;
        const o = root.object;
        const phonemes = (o.get("phonemes") orelse return error.BadTextRows).object;
        const listed = (phonemes.get("symbols") orelse return error.BadTextRows).array.items;
        const symbols = try a.alloc(u21, listed.len);
        for (symbols, listed) |*s, v| {
            if (v != .string or (std.unicode.utf8CountCodepoints(v.string) catch 0) != 1) return error.BadTextRows;
            s.* = std.unicode.utf8Decode(v.string) catch return error.BadTextRows;
        }
        const pauses = ((o.get("pauses") orelse return error.BadTextRows).object.get("ids") orelse return error.BadTextRows).array.items;
        const pause_ids = try a.alloc(u32, pauses.len);
        for (pause_ids, pauses) |*id, v| id.* = @intCast(v.integer);
        const spec: Spec = .{
            .open_id = @intCast((phonemes.get("open_id") orelse return error.BadTextRows).integer),
            .close_id = @intCast((phonemes.get("close_id") orelse return error.BadTextRows).integer),
            .symbols = symbols,
            .start_id = @intCast(((o.get("start") orelse return error.BadTextRows).object.get("id") orelse return error.BadTextRows).integer),
            .pause_ids = pause_ids,
        };
        // the rows follow the padding row: the group ids, the symbols, the start row, the pauses
        const first = req.config.text_vocab + 1;
        if (spec.open_id != first or spec.close_id != first + 1 or spec.start_id != first + 2 + symbols.len or pause_ids.len == 0 or pause_ids.len > 8) return error.BadTextRows;
        for (pause_ids, 0..) |id, j| if (id != spec.start_id + 1 + j) return error.BadTextRows;
        return spec;
    }
};

/// A marker (U+E000..U+E00A) at byte `i`: its last byte.
fn markerAt(s: []const u8, i: usize) ?u8 {
    return if (i + 2 < s.len and s[i] == 0xee and s[i + 1] == 0x80 and s[i + 2] >= 0x80 and s[i + 2] <= last_marker) s[i + 2] else null;
}

pub fn hasMarkers(s: []const u8) bool {
    for (0..s.len) |i| if (markerAt(s, i) != null) return true;
    return false;
}

/// A stretch without markers as SuperBPE ids, each less the tokenizer's offset of 4 as prompt.build adds it.
fn stretch(cx: *errors.Cx, text: text_mod.Text, s: []const u8, out: *std.ArrayList(u32)) errors.Refused!void {
    if (s.len == 0) return;
    const ids = text.encode(cx.a, s, false) catch |e| return switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        else => cx.other("the tokenizer cannot encode this prompt"),
    };
    try out.appendSlice(cx.a, ids);
}

/// A row id as prompt.build takes it: less the tokenizer's offset.
fn row(id: u32) u32 {
    return id - req.prompt.token_offset;
}

/// superbpe.payload_ids: the start marker first, a pause marker an id, a group its open id, phones and close id.
pub fn payload(cx: *errors.Cx, text: text_mod.Text, spec: ?Spec, s: []const u8) errors.Refused![]const u32 {
    var out: std.ArrayList(u32) = .empty;
    if (!hasMarkers(s)) {
        try stretch(cx, text, s, &out);
        return out.items;
    }
    const sp = spec orelse return cx.refuse(prefix ++ "this release has no phoneme, start or pause tokens: remove the U+E000 to U+E00A markers");
    var at: usize = 0;
    if (std.mem.startsWith(u8, s, start_marker)) {
        try out.append(cx.a, row(sp.start_id));
        at = 3;
    }
    var from = at;
    while (at < s.len) {
        const m = markerAt(s, at) orelse {
            at += 1;
            continue;
        };
        try stretch(cx, text, s[from..at], &out);
        if (m == 0x82) return cx.refuse(prefix ++ "the start marker (U+E002) must be the first character of the text, once");
        if (m == 0x81) return cx.refuse(prefix ++ "unbalanced pronunciation group markers");
        if (m >= first_pause) {
            if (m - first_pause >= sp.pause_ids.len) return cx.fail(.request, prefix ++ "pause marker U+E0{X:0>2} is outside the {d} pause buckets", .{ m - 0x80, sp.pause_ids.len });
            try out.append(cx.a, row(sp.pause_ids[m - first_pause]));
            at += 3;
            from = at;
            continue;
        }
        // a pronunciation group: its phones up to the close marker
        const end = std.mem.indexOfPos(u8, s, at + 3, group_close) orelse return cx.refuse(prefix ++ "unbalanced or empty pronunciation group");
        const group = s[at + 3 .. end];
        if (group.len == 0 or hasMarkers(group)) return cx.refuse(prefix ++ "unbalanced or empty pronunciation group");
        try out.append(cx.a, row(sp.open_id));
        var phones = (std.unicode.Utf8View.init(group) catch return cx.refuse("input must be valid UTF-8 text")).iterator();
        while (phones.nextCodepoint()) |phone| {
            const i = std.mem.indexOfScalar(u21, sp.symbols, phone) orelse return cx.fail(.request, prefix ++ "a pronunciation group holds U+{X:0>4}, which is outside the phone inventory", .{phone});
            try out.append(cx.a, row(sp.open_id + 2 + @as(u32, @intCast(i))));
        }
        try out.append(cx.a, row(sp.close_id));
        at = end + 3;
        from = at;
    }
    try stretch(cx, text, s[from..], &out);
    return out.items;
}

/// A stand-in tokenizer for tests: an id a byte.
pub const Bytes = struct {
    fn encode(_: *anyopaque, a: Allocator, s: []const u8, _: bool) text_mod.Error![]u32 {
        const ids = try a.alloc(u32, s.len);
        for (ids, s) |*id, c| id.* = c;
        return ids;
    }

    fn decode(_: *anyopaque, a: Allocator, ids: []const u32) Allocator.Error![]u8 {
        const s = try a.alloc(u8, ids.len);
        for (s, ids) |*c, id| c.* = @intCast(id);
        return s;
    }

    var unused: u8 = 0;
    const vtable: text_mod.Text.VTable = .{ .encode = encode, .decode = decode, .token_id = undefined, .token_string = undefined, .vocab_size = undefined, .eos_ids = undefined, .render = undefined, .template_source = undefined };

    pub fn text() text_mod.Text {
        return .{ .ctx = &unused, .vtable = &vtable };
    }
};

test "markers become their rows' ids, the text around them encoded on its own" {
    const t = std.testing;
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    var cx: errors.Cx = .{ .a = arena.allocator() };
    const spec: Spec = .{ .open_id = 8197, .close_id = 8198, .symbols = &.{ 'a', 0x259 }, .start_id = 8201, .pause_ids = &.{ 8202, 8203 } };
    const off = req.prompt.token_offset;
    // plain text passes through; without a spec a marker is refused
    try t.expectEqualSlices(u32, &.{ 'h', 'i' }, try payload(&cx, Bytes.text(), null, "hi"));
    try t.expectError(error.Refused, payload(&cx, Bytes.text(), null, "hi" ++ start_marker));
    const got = try payload(&cx, Bytes.text(), spec, start_marker ++ "a\xee\x80\x84b" ++ group_open ++ "\xc9\x99a" ++ group_close ++ "c");
    try t.expectEqualSlices(u32, &.{ 8201 - off, 'a', 8203 - off, 'b', 8197 - off, 8200 - off, 8199 - off, 8198 - off, 'c' }, got);
    const bad = [_][2][]const u8{
        .{ "a" ++ start_marker, "must be the first character" },
        .{ "a\xee\x80\x85", "outside the 2 pause buckets" },
        .{ group_open ++ "a", "unbalanced or empty pronunciation group" },
        .{ group_open ++ group_close, "unbalanced or empty pronunciation group" },
        .{ "a" ++ group_close, "unbalanced pronunciation group markers" },
        .{ group_open ++ "z" ++ group_close, "outside the phone inventory" },
    };
    for (bad) |c| {
        try t.expectError(error.Refused, payload(&cx, Bytes.text(), spec, c[0]));
        try t.expect(std.mem.indexOf(u8, cx.message, c[1]) != null);
    }
}
