//! ML-1 past cues: English via past.zig, German and Finnish via lang_tables.zig.
const std = @import("std");
const past = @import("past.zig");
const tables = @import("lang_tables.zig");

pub const Lang = enum {
    en,
    de,
    fi,

    pub fn parse(s: []const u8) ?Lang {
        if (std.mem.eql(u8, s, "en")) return .en;
        if (std.mem.eql(u8, s, "de")) return .de;
        if (std.mem.eql(u8, s, "fi")) return .fi;
        return null;
    }
};

/// Tests only: drop this alternative index in the active language's list (threadlocal: other tests call isPast).
pub threadlocal var plant_drop: ?usize = null;

const Cp = struct { cp: u21, len: u3 };

fn at(s: []const u8, i: usize) Cp {
    const n = std.unicode.utf8ByteSequenceLength(s[i]) catch return .{ .cp = s[i], .len = 1 };
    if (i + n > s.len) return .{ .cp = s[i], .len = 1 };
    const cp = std.unicode.utf8Decode(s[i..][0..n]) catch return .{ .cp = s[i], .len = 1 };
    return .{ .cp = cp, .len = n };
}

fn before(s: []const u8, i: usize) ?u21 {
    if (i == 0) return null;
    var j = i - 1;
    while (j > 0 and i - j < 4 and s[j] & 0xC0 == 0x80) j -= 1;
    const c = at(s, j);
    return if (j + c.len == i) c.cp else s[i - 1];
}

fn spelled(s: []const u8, i: usize, word: []const u8) ?usize {
    var j = i;
    var k: usize = 0;
    while (k < word.len) {
        if (j >= s.len) return null;
        const want = at(word, k);
        const got = at(s, j);
        if (!tables.letterEq(got.cp, want.cp)) return null;
        j += got.len;
        k += want.len;
    }
    return j;
}

fn endsWord(s: []const u8, j: usize) bool {
    return j >= s.len or !tables.isWord(at(s, j).cp);
}

fn startsWord(s: []const u8, i: usize) bool {
    if (i >= s.len) return false;
    const c = at(s, i);
    if (!tables.isWord(c.cp)) return false;
    const prev = before(s, i);
    return prev == null or !tables.isWord(prev.?);
}

fn skipSpaces(s: []const u8, i: usize) ?usize {
    if (i >= s.len) return null;
    const first = at(s, i);
    if (!tables.isSpace(first.cp)) return null;
    var j = i + first.len;
    while (j < s.len) {
        const c = at(s, j);
        if (!tables.isSpace(c.cp)) break;
        j += c.len;
    }
    return j;
}

fn skipWordRun(s: []const u8, i: usize) ?usize {
    if (i >= s.len) return null;
    const first = at(s, i);
    if (!tables.isWord(first.cp)) return null;
    var j = i + first.len;
    while (j < s.len) {
        const c = at(s, j);
        if (!tables.isWord(c.cp)) break;
        j += c.len;
    }
    return j;
}

const de_words = [_][]const u8{
    "war",        "waren",  "warst",   "wart",
    "hatte",      "hatten", "hattest",
    "früher",
    "vorher",     "zuvor",  "damals",
    "ursprünglich",
    "einst",      "je",     "jemals",  "immer",
    "geändert",
    "verändert",
    "gewechselt", "bis",    "seit",    "seitdem",
};

const de_units = [_][]const u8{ "Jahren", "Monaten", "Wochen", "Tagen" };

const fi_words = [_][]const u8{
    "oli",        "olivat",    "olin",        "olit",
    "olimme",     "olitte",    "ollut",       "olleet",
    "ennen",      "aiemmin",   "aikaisemmin", "alkujaan",
    "entinen",    "entisen",   "joskus",      "koskaan",
    "ikinä",
    "aina",       "muuttunut", "muuttuneet",  "muuttui",
    "muuttuivat", "vaihtunut", "vaihtui",     "vaihtanut",
    "asti",       "saakka",
    "lähtien",
    "alkaen",     "sitten",
};

const fi_clitics = [_][]const u8{ "ko", "kö", "kin", "kaan", "kään", "han", "hän" };

fn live(id: usize) bool {
    return plant_drop == null or plant_drop.? != id;
}

fn yearAt(s: []const u8, i: usize) bool {
    if (i + 2 > s.len) return false;
    const a = s[i];
    const b = s[i + 1];
    if (!((a == '1' and b == '9') or (a == '2' and b == '0'))) return false;
    var j: usize = i + 2;
    var k: usize = 0;
    while (k < 2) : (k += 1) {
        if (j >= s.len) return false;
        const c = at(s, j);
        if (!tables.isDigit(c.cp)) return false;
        j += c.len;
    }
    return endsWord(s, j);
}

fn schonImmerAt(s: []const u8, i: usize) bool {
    const j0 = spelled(s, i, "schon") orelse return false;
    const j1 = skipSpaces(s, j0) orelse return false;
    const j2 = spelled(s, j1, "immer") orelse return false;
    return endsWord(s, j2);
}

fn vorUnitAt(s: []const u8, i: usize) bool {
    const j0 = spelled(s, i, "vor") orelse return false;
    const j1 = skipSpaces(s, j0) orelse return false;
    const j2 = skipWordRun(s, j1) orelse return false;
    const j3 = skipSpaces(s, j2) orelse return false;
    for (de_units) |u| {
        if (spelled(s, j3, u)) |end| if (endsWord(s, end)) return true;
    }
    return false;
}

fn alunPerinAt(s: []const u8, i: usize) bool {
    const j0 = spelled(s, i, "alun") orelse return false;
    const j1 = skipSpaces(s, j0) orelse return false;
    const j2 = spelled(s, j1, "perin") orelse return false;
    return fiEnd(s, j2);
}

fn fiEnd(s: []const u8, after_stem: usize) bool {
    if (endsWord(s, after_stem)) return true;
    for (fi_clitics) |c| {
        if (spelled(s, after_stem, c)) |end| if (endsWord(s, end)) return true;
    }
    return false;
}

fn deAt(s: []const u8, i: usize) bool {
    var id: usize = 0;
    for (de_words) |w| {
        if (live(id)) if (spelled(s, i, w)) |j| if (endsWord(s, j)) return true;
        id += 1;
    }
    if (live(id) and schonImmerAt(s, i)) return true;
    id += 1;
    if (live(id) and vorUnitAt(s, i)) return true;
    id += 1;
    if (live(id) and yearAt(s, i)) return true;
    return false;
}

fn fiAt(s: []const u8, i: usize) bool {
    var id: usize = 0;
    for (fi_words) |w| {
        if (live(id)) if (spelled(s, i, w)) |j| if (fiEnd(s, j)) return true;
        id += 1;
    }
    if (live(id) and alunPerinAt(s, i)) return true;
    id += 1;
    if (live(id) and yearAt(s, i)) return true;
    return false;
}

fn search(s: []const u8, lang: Lang) bool {
    var i: usize = 0;
    while (i < s.len) {
        if (startsWord(s, i)) {
            const hit = switch (lang) {
                .de => deAt(s, i),
                .fi => fiAt(s, i),
                .en => unreachable,
            };
            if (hit) return true;
        }
        i += at(s, i).len;
    }
    return false;
}

pub fn isPast(q: []const u8, lang: Lang) bool {
    return switch (lang) {
        .en => past.isPast(q),
        .de, .fi => search(q, lang),
    };
}

pub fn deAltCount() usize {
    return de_words.len + 3;
}

pub fn fiAltCount() usize {
    return fi_words.len + 2;
}

test "en parse and routing" {
    try std.testing.expectEqual(Lang.en, Lang.parse("en").?);
    try std.testing.expectEqual(Lang.de, Lang.parse("de").?);
    try std.testing.expectEqual(Lang.fi, Lang.parse("fi").?);
    try std.testing.expect(Lang.parse("sv") == null);
    try std.testing.expect(isPast("Was ist die Augenfarbe meiner Katze?", .en));
    try std.testing.expect(!isPast("Was ist die Augenfarbe meiner Katze?", .de));
}

test "de cues" {
    try std.testing.expect(isPast("Wo war mein Zuhause früher?", .de));
    try std.testing.expect(isPast("vor drei Jahren", .de));
    try std.testing.expect(isPast("vor 3 Jahren", .de));
    try std.testing.expect(!isPast("vor  Jahren", .de));
    try std.testing.expect(isPast("1999", .de));
    try std.testing.expect(!isPast("Was ist die Augenfarbe meiner Katze?", .de));
    try std.testing.expect(!isPast("Wo wohne ich?", .de));
}

test "fi clitics" {
    try std.testing.expect(isPast("oliko se", .fi));
    try std.testing.expect(isPast("alun perin", .fi));
}

const cases_json = @embedFile("testdata/lang_cases.json");

fn asLang(s: []const u8) Lang {
    return Lang.parse(s) orelse unreachable;
}

fn containsId(ids: []const std.json.Value, id: usize) bool {
    for (ids) |v| if (v.integer == @as(i64, @intCast(id))) return true;
    return false;
}

test "every lang_cases.json row, and plant_drop flips a live alternative" {
    const parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, cases_json, .{});
    defer parsed.deinit();
    const obj = parsed.value.object;
    const de_n: usize = @intCast(obj.get("de_alt_count").?.integer);
    const fi_n: usize = @intCast(obj.get("fi_alt_count").?.integer);
    try std.testing.expectEqual(deAltCount(), de_n);
    try std.testing.expectEqual(fiAltCount(), fi_n);
    const cases = obj.get("cases").?.array.items;
    try std.testing.expect(cases.len > 0);
    for (cases) |c| {
        const lang = asLang(c.object.get("lang").?.string);
        const text = c.object.get("text").?.string;
        const want = c.object.get("want").?.bool;
        try std.testing.expectEqual(want, isPast(text, lang));
    }
    const de_sub = obj.get("de_subsumed").?.array.items;
    const fi_sub = obj.get("fi_subsumed").?.array.items;
    for (0..de_n) |id| {
        if (containsId(de_sub, id)) continue;
        plant_drop = id;
        var flipped = false;
        for (cases) |c| {
            if (!std.mem.eql(u8, c.object.get("lang").?.string, "de")) continue;
            if (!c.object.get("want").?.bool) continue;
            if (!isPast(c.object.get("text").?.string, .de)) {
                flipped = true;
                break;
            }
        }
        plant_drop = null;
        try std.testing.expect(flipped);
    }
    for (0..fi_n) |id| {
        if (containsId(fi_sub, id)) continue;
        plant_drop = id;
        var flipped = false;
        for (cases) |c| {
            if (!std.mem.eql(u8, c.object.get("lang").?.string, "fi")) continue;
            if (!c.object.get("want").?.bool) continue;
            if (!isPast(c.object.get("text").?.string, .fi)) {
                flipped = true;
                break;
            }
        }
        plant_drop = null;
        try std.testing.expect(flipped);
    }
}
