//! A host fixture chooses one valid path in the emitted structural tag; it does not replace xgrammar.
const std = @import("std");
const json = @import("json");
const lanes = @import("lanes");
const V = json.Value;
const A = std.mem.Allocator;

fn sample(a: A, s: V) A.Error!V {
    if (s.get("const")) |v| return v;
    const t = s.strField("type") orelse "object";
    if (std.mem.eql(u8, t, "string")) return .{ .string = "v,other:value {x:y}" };
    if (std.mem.eql(u8, t, "integer")) return json.intValue(a, 1);
    if (std.mem.eql(u8, t, "boolean")) return .{ .bool = true };
    const o = try json.newObject(a);
    if (s.get("properties")) |props| for (props.object.keys(), props.object.values()) |key, val| try o.put(a, key, try sample(a, val));
    return .{ .object = o };
}

pub fn text(a: A, n: V) A.Error![]const u8 {
    const t = n.strField("type").?;
    if (std.mem.eql(u8, t, "const_string")) return n.strField("value").?;
    if (std.mem.eql(u8, t, "regex") or std.mem.eql(u8, t, "star")) return "";
    if (std.mem.eql(u8, t, "structural_tag")) return text(a, n.get("format").?);
    if (std.mem.eql(u8, t, "or")) return text(a, n.get("elements").?.array[0]);
    var out: std.ArrayList(u8) = .empty;
    if (std.mem.eql(u8, t, "sequence")) {
        for (n.get("elements").?.array) |part| try out.appendSlice(a, try text(a, part));
    } else {
        std.debug.assert(std.mem.eql(u8, t, "json_schema"));
        const v = try sample(a, n.get("json_schema").?);
        if (std.mem.eql(u8, n.strField("style").?, "qwen_xml")) {
            for (v.object.keys(), v.object.values()) |key, val| try out.print(a, "<parameter={s}>{s}</parameter>", .{ key, if (val == .string) val.string else try json.stringify(a, val, .{ .compact = true }) });
        } else try out.appendSlice(a, try json.stringify(a, v, .{ .compact = true }));
    }
    return out.items;
}

pub const Result = struct { tokens: []u32, drafted: u64 };

/// The actual lane engine walks the fixture rules, masks first/verify draws and rolls speculative paths back.
pub fn run(want: []const u32, depth: u32, seed: u64) !Result {
    const a = std.testing.allocator;
    var seq: lanes.grammar.Sequence = .{ .want = want, .stop = 96, .gpa = a };
    defer seq.log.deinit(a);
    var constraint: lanes.grammar.Constraint = .init(seq.rules(), 4, null);
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const first = (try constraint.next(arena.allocator())).?;
    try std.testing.expect(lanes.grammar.allows(first, want[0]));
    try std.testing.expect(!lanes.grammar.allows(first, 96));
    try std.testing.expect(!lanes.grammar.allows(first, 'A' - 32));
    var cfg = try lanes.Config.init(a, .{ .exact_width = 8, .mtp = true, .speculate = true, .drafts = depth, .hidden_rows = true, .streams_exact = true, .batch_rows = 8, .max_streams = 2, .draft_streams = true }, 8, 7);
    defer cfg.deinit(a);
    var fake: lanes.fake.Fake = .{ .gpa = a };
    defer fake.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var engine = lanes.Engine.init(a, &cfg, fake.backend(), clock.clock());
    defer engine.deinit();
    var s = try lanes.Stream.init(a, .{ .id = "tool", .prompt = &.{ 3, 1, 4, 1 }, .max_new = 512, .sampling = .{ .seed = seed, .temperature = 1 }, .drafts = depth > 0, .grammar = &constraint, .eos = &.{96} });
    defer s.deinit(a);
    try engine.addStream(&s);
    while (engine.activeCount() > 0) try engine.step();
    try std.testing.expect(seq.at > want.len);
    return .{ .tokens = try a.dupe(u32, s.emitted()), .drafted = s.drafted };
}
