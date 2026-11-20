//! ZONOS3's LACI on the speech route: a folder of alignment-head files (--laci-heads) and the `laci` request field.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const errors = @import("errors.zig");
const req = @import("zonos_request");
const textnorm = @import("zonos_textnorm.zig");
const Allocator = std.mem.Allocator;

const prefix = "ZONOS3 request: ";
pub const field = "laci";
/// TensorFold's duration target (the reference servers' LaciTakeRequest.target_frames, in seconds) and its rate rule.
pub const duration_field = "duration_seconds";
pub const adjust_field = "duration_adjust_rate";

/// A request's duration target: generated rows (15 a second) and whether a miss is retaken in another rate bucket.
pub const Duration = struct { frames: f32, adjust_rate: bool };

/// plan.check_laci with a duration target: a LACI request's length to hold, in seconds; null without one.
pub fn duration(cx: *errors.Cx, raw: json.Value, watched: bool, max_frames: u32) errors.Refused!?Duration {
    const v = raw.field(duration_field) orelse {
        if (raw.field(adjust_field) != null) return cx.refuse(prefix ++ "duration_adjust_rate needs duration_seconds");
        return null;
    };
    const seconds = v.float64() orelse return cx.refuse(prefix ++ "duration_seconds must be a number of seconds");
    if (!watched) return cx.refuse(prefix ++ "A LACI duration target needs laci=true");
    const most = @as(f64, @floatFromInt(max_frames)) / 15.0;
    if (!(seconds > 0) or !(seconds <= most)) return cx.fail(.request, prefix ++ "duration_seconds must be above 0 and at most {d:.2} (the frame budget)", .{most});
    var adjust = true;
    if (raw.field(adjust_field)) |a| adjust = if (a == .bool) a.bool else return cx.refuse(prefix ++ "duration_adjust_rate must be a boolean");
    return .{ .frames = @floatCast(seconds * 15.0), .adjust_rate = adjust };
}

/// The heads of each request language, in the languages' order: its speech code's file, else the folder's default.
pub const Table = struct { heads: [textnorm.languages.len]*const api.SpeechLaci };

/// The code a head file is named by: the language's speech code, which is its grammar's but for Norwegian.
fn speechCode(l: textnorm.Language) []const u8 {
    return if (std.mem.eql(u8, l.code, "no")) "no" else l.grammar;
}

fn parse(a: Allocator, text: []const u8) !*const api.SpeechLaci {
    const parsed = try req.laci.Heads.parse(a, text);
    const heads = try a.alloc(api.SpeechLaci.Head, parsed.heads.len);
    for (heads, parsed.heads) |*to, from| to.* = .{ .layer = from.layer, .head = from.head, .radius = from.radius };
    const out = try a.create(api.SpeechLaci);
    out.* = .{ .heads = heads, .frames_per_token = parsed.frames_per_token };
    return out;
}

/// discover_laci_heads: one `*_heads.json` is the default; a `*_heads_<code>.json` serves that speech code.
pub fn load(a: Allocator, io: std.Io, path: []const u8) !*const Table {
    var dir = try std.Io.Dir.cwd().openDir(io, path, .{ .iterate = true });
    defer dir.close(io);
    var default: ?*const api.SpeechLaci = null;
    var by_code: [textnorm.languages.len]?*const api.SpeechLaci = @splat(null);
    var files = dir.iterate();
    while (try files.next(io)) |entry| {
        if (entry.kind != .file or !std.mem.endsWith(u8, entry.name, ".json")) continue;
        const stem = entry.name[0 .. entry.name.len - ".json".len];
        const marker = "_heads";
        const at = std.mem.lastIndexOf(u8, stem, marker) orelse continue;
        const tail = stem[at + marker.len ..];
        if (tail.len != 0 and tail[0] != '_') continue;
        const heads = try parse(a, try dir.readFileAlloc(io, entry.name, a, .limited(1 << 20)));
        if (tail.len == 0) {
            if (default != null) return error.LaciHeadsNeedOneDefault;
            default = heads;
            continue;
        }
        var spoken = false;
        for (textnorm.languages, &by_code) |l, *slot| if (std.mem.eql(u8, speechCode(l), tail[1..])) {
            slot.* = heads;
            spoken = true;
        };
        if (!spoken) return error.LaciHeadsOfNoLanguage;
    }
    const fallback = default orelse return error.LaciHeadsNeedOneDefault;
    const table = try a.create(Table);
    for (&table.heads, by_code) |*to, from| to.* = from orelse fallback;
    return table;
}

/// The heads of a request language.
pub fn of(t: *const Table, language: textnorm.Language) *const api.SpeechLaci {
    for (textnorm.languages, t.heads) |l, heads| if (std.mem.eql(u8, l.code, language.code)) return heads;
    unreachable;
}

/// plan.check_laci: a `laci: true` request needs the server's heads and no guidance of either kind.
pub fn plan(table: ?*const Table, cx: *errors.Cx, raw: json.Value, language: textnorm.Language, guided: bool) errors.Refused!?*const api.SpeechLaci {
    const v = raw.field(field) orelse return null;
    if (v != .bool) return cx.refuse(prefix ++ "laci must be a boolean");
    if (!v.bool) return null;
    const t = table orelse return cx.refuse(prefix ++ "LACI is not enabled on this server (start it with --laci-heads DIR)");
    if (guided) return cx.refuse(prefix ++ "LACI needs speaker_cfg_scale = controls_cfg_scale = 1");
    return of(t, language);
}

test "laci: off without the field, refused without heads or with guidance" {
    const t = std.testing;
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var cx: errors.Cx = .{ .a = a };
    const en = textnorm.language("en_us").?;
    try t.expect(try plan(null, &cx, (try json.parse(a, "{}")).ok, en, false) == null);
    try t.expect(try plan(null, &cx, (try json.parse(a, "{\"laci\": false}")).ok, en, true) == null);
    const on = (try json.parse(a, "{\"laci\": true}")).ok;
    try t.expectError(error.Refused, plan(null, &cx, on, en, false));
    try t.expect(std.mem.indexOf(u8, cx.message, "--laci-heads") != null);
    const heads = try parse(a, "{\"frames_per_token\": 3.5, \"heads\": [{\"layer\": 7, \"head\": 14, \"radius\": 6}]}");
    const table: Table = .{ .heads = @splat(heads) };
    try t.expectError(error.Refused, plan(&table, &cx, on, en, true));
    try t.expect(std.mem.indexOf(u8, cx.message, "speaker_cfg_scale = controls_cfg_scale = 1") != null);
    try t.expect((try plan(&table, &cx, on, en, false)).? == heads);
    try t.expectEqualStrings("no", speechCode(textnorm.language("no").?));
    try t.expectEqualStrings("zh", speechCode(textnorm.language("cmn").?));
    try t.expectError(error.Refused, plan(&table, &cx, (try json.parse(a, "{\"laci\": 1}")).ok, en, false));
}
