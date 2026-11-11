//! The opt-in reply binding (`"execution_witness": true`): the witness at admission and at the end, and what ran.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const errors = @import("errors.zig");
const witness = @import("witness.zig");
const parts = @import("witness_parts.zig");
const tokens = @import("tokens.zig");
const Server = @import("server.zig").Server;
const Sha256 = std.crypto.hash.sha2.Sha256;
const Value = json.Value;
const Allocator = std.mem.Allocator;
const Cx = errors.Cx;

/// Whether the body asks for a binding: a boolean, and true only on a server started with --execution-witness.
pub fn asked(srv: *const Server, cx: *Cx, body: Value) errors.Refused!bool {
    const v = body.field("execution_witness") orelse return false;
    if (v != .bool) return cx.refuse("execution_witness must be a boolean");
    if (v.bool and srv.witness == null) return cx.refuse("execution_witness needs a server started with --execution-witness");
    return v.bool;
}

/// What a reply binds before the engine takes it: the witness as admitted, the prompt ids and the resolved request.
pub const Admitted = struct { snap: witness.Snapshot, prompt: [64]u8, request: Value, request_sha: [64]u8 };

/// Read just before submit, while the request holds the idle manager's active count, so no reload can come between.
pub fn admit(srv: *Server, a: Allocator, request: *const api.Request, stops: []const []const u8) Allocator.Error!Admitted {
    const resolved = try resolvedOf(a, request, stops);
    return .{ .snap = srv.witness.?.read(srv), .prompt = tokens.tokenSha256(request.prompt), .request = resolved, .request_sha = std.fmt.bytesToHex(requestDigest(resolved), .lower) };
}

/// Every resolved value the engine reads to choose tokens, after defaults, clamps and the seed the server drew.
fn resolvedOf(a: Allocator, r: *const api.Request, stops: []const []const u8) Allocator.Error!Value {
    const o = try json.newObject(a);
    try o.put(a, "max_tokens", try json.intValue(a, r.max_tokens));
    try o.put(a, "sampling", if (r.sampling) |s| try samplingOf(a, s) else .null);
    try o.put(a, "presence_penalty", .{ .float = r.penalties.presence });
    try o.put(a, "frequency_penalty", .{ .float = r.penalties.frequency });
    try o.put(a, "eos", try idsOf(a, r.eos));
    try o.put(a, "stop", try stringsOf(a, stops));
    try o.put(a, "drafts", .{ .bool = r.drafts });
    try o.put(a, "think_budget", try json.intValue(a, r.think_budget));
    try o.put(a, "think_close", try idsOf(a, r.think_close));
    try o.put(a, "think_end", if (r.think_end) |e| try json.intValue(a, e) else .null);
    try o.put(a, "answer_max", try json.intValue(a, r.answer_max));
    try o.put(a, "loop_guard", .{ .bool = r.loop_guard });
    try o.put(a, "loop_stop", .{ .bool = r.loop_stop });
    try o.put(a, "call", if (r.call) |c| try callOf(a, c) else .null);
    try o.put(a, "structure", if (r.structure) |s| try structureOf(a, s) else .null);
    try o.put(a, "logprobs", if (r.logprobs) |n| try json.intValue(a, n) else .null);
    try o.put(a, "prompt_logprobs", if (r.prompt_logprobs) |n| try json.intValue(a, n) else .null);
    try o.put(a, "decide", if (r.decide) |d| try json.intValue(a, d) else .null);
    try o.put(a, "untaught", .{ .bool = r.untaught });
    return .{ .object = o };
}

fn samplingOf(a: Allocator, s: api.Sampling) Allocator.Error!Value {
    const o = try json.newObject(a);
    try o.put(a, "seed", try json.intValue(a, s.seed));
    try o.put(a, "temperature", .{ .float = s.temperature });
    try o.put(a, "top_k", try json.intValue(a, s.top_k));
    try o.put(a, "top_p", .{ .float = s.top_p });
    try o.put(a, "min_p", .{ .float = s.min_p });
    return .{ .object = o };
}

fn callOf(a: Allocator, c: api.CallGate) Allocator.Error!Value {
    const o = try json.newObject(a);
    try o.put(a, "opener", try json.intValue(a, c.opener));
    try o.put(a, "lead", .{ .string = c.lead });
    try o.put(a, "tail", .{ .string = c.tail });
    try o.put(a, "names", try stringsOf(a, c.names));
    try o.put(a, "think_open", if (c.think_open) |x| try json.intValue(a, x) else .null);
    try o.put(a, "think_end", if (c.think_end) |x| try json.intValue(a, x) else .null);
    return .{ .object = o };
}

fn structureOf(a: Allocator, s: api.Structure) Allocator.Error!Value {
    const o = try json.newObject(a);
    try o.put(a, "kind", .{ .string = @tagName(s.kind) });
    try o.put(a, "text_sha256", .{ .string = try a.dupe(u8, &sha256Hex(s.text)) });
    try o.put(a, "after", if (s.after) |x| try json.intValue(a, x) else .null);
    try o.put(a, "budget", .{ .bool = s.budget });
    return .{ .object = o };
}

fn idsOf(a: Allocator, list: []const u32) Allocator.Error!Value {
    const out = try a.alloc(Value, list.len);
    for (list, out) |id, *slot| slot.* = try json.intValue(a, id);
    return .{ .array = out };
}

fn stringsOf(a: Allocator, list: []const []const u8) Allocator.Error!Value {
    const out = try a.alloc(Value, list.len);
    for (list, out) |s, *slot| slot.* = .{ .string = s };
    return .{ .array = out };
}

fn sha256Hex(bytes: []const u8) [64]u8 {
    var d: [32]u8 = undefined;
    Sha256.hash(bytes, &d, .{});
    return std.fmt.bytesToHex(d, .lower);
}

/// `request_sha256`: the resolved request walked in order, each value tagged and length-prefixed.
pub fn requestDigest(v: Value) [32]u8 {
    var h = Sha256.init(.{});
    parts.field(&h, "tensorfold resolved request v1");
    encode(&h, v);
    return h.finalResult();
}

/// null n, bool b and 0 or 1, int i and its decimal text, float f and its IEEE bits, string s, array a, object o.
fn encode(h: *Sha256, v: Value) void {
    switch (v) {
        .null => parts.field(h, "n"),
        .bool => |b| tagged(h, "b", if (b) "1" else "0"),
        .int => |text| tagged(h, "i", text),
        .float => |f| {
            var bits: [8]u8 = undefined;
            std.mem.writeInt(u64, &bits, @bitCast(f), .little);
            tagged(h, "f", &bits);
        },
        .string => |s| tagged(h, "s", s),
        .array => |items| {
            tagged(h, "a", &count(items.len));
            for (items) |item| encode(h, item);
        },
        .object => |o| {
            tagged(h, "o", &count(o.count()));
            for (o.keys(), o.values()) |k, item| {
                parts.field(h, k);
                encode(h, item);
            }
        },
    }
}

fn tagged(h: *Sha256, tag: []const u8, bytes: []const u8) void {
    parts.field(h, tag);
    parts.field(h, bytes);
}

fn count(n: usize) [8]u8 {
    var out: [8]u8 = undefined;
    std.mem.writeInt(u64, &out, n, .little);
    return out;
}

/// `binding`: the witness, prompt, request and token digests (as their hex text) and the finish reason.
pub fn bindingDigest(witness_hex: []const u8, prompt: []const u8, request: []const u8, reply_tokens: []const u8, finish_reason: []const u8) [32]u8 {
    var h = Sha256.init(.{});
    parts.field(&h, "tensorfold execution binding v1");
    for ([_][]const u8{ witness_hex, prompt, request, reply_tokens, finish_reason }) |f| parts.field(&h, f);
    return h.finalResult();
}

/// The reply's binding: known only when the witness was known at admission and is the same, same epoch, at the end.
pub fn value(a: Allocator, srv: *Server, admitted: Admitted, reply_tokens: []const u32, finish_reason: []const u8) Allocator.Error!Value {
    const end = srv.witness.?.read(srv);
    const o = try json.newObject(a);
    try o.put(a, "schema", .{ .int = "1" });
    const instance = std.fmt.bytesToHex(end.instance, .lower);
    const known: ?witness.Composed = switch (admitted.snap.result) {
        .unknown => |u| try unknownInto(a, o, @tagName(u.reason), u.feature),
        .known => |first| switch (end.result) {
            .unknown => |u| try unknownInto(a, o, @tagName(u.reason), u.feature),
            .known => |last| if (admitted.snap.epoch != end.epoch)
                try unknownInto(a, o, "epoch_changed", null)
            else if (!std.mem.eql(u8, &first.witness, &last.witness))
                try unknownInto(a, o, "witness_changed", null)
            else
                last,
        },
    };
    const token_sha = tokens.tokenSha256(reply_tokens);
    if (known) |k| {
        const w = std.fmt.bytesToHex(k.witness, .lower);
        const binding = std.fmt.bytesToHex(bindingDigest(&w, &admitted.prompt, &admitted.request_sha, &token_sha, finish_reason), .lower);
        try o.put(a, "state", .{ .string = "known" });
        try o.put(a, "binding", .{ .string = try a.dupe(u8, &binding) });
        try o.put(a, "witness", .{ .string = try a.dupe(u8, &w) });
        try o.put(a, "path", .{ .string = k.path });
    }
    try o.put(a, "instance", .{ .string = try a.dupe(u8, &instance) });
    try o.put(a, "epoch", try json.intValue(a, end.epoch));
    if (known == null) try o.put(a, "admitted_epoch", try json.intValue(a, admitted.snap.epoch));
    try o.put(a, "prompt_sha256", .{ .string = try a.dupe(u8, &admitted.prompt) });
    try o.put(a, "request_sha256", .{ .string = try a.dupe(u8, &admitted.request_sha) });
    try o.put(a, "token_sha256", .{ .string = try a.dupe(u8, &token_sha) });
    try o.put(a, "finish_reason", .{ .string = finish_reason });
    try o.put(a, "request", admitted.request);
    return .{ .object = o };
}

fn unknownInto(a: Allocator, o: *json.Object, reason: []const u8, feature: ?[]const u8) Allocator.Error!?witness.Composed {
    try o.put(a, "state", .{ .string = "unknown" });
    try o.put(a, "reason", .{ .string = reason });
    if (feature) |f| try o.put(a, "feature", .{ .string = f });
    return null;
}

/// A reply that asked but took a path that binds nothing (compaction, Cortheon): unknown, with the witness's reason.
pub fn missing(a: Allocator, srv: *Server) Allocator.Error!Value {
    const snap = srv.witness.?.read(srv);
    const o = try json.newObject(a);
    try o.put(a, "schema", .{ .int = "1" });
    switch (snap.result) {
        .unknown => |u| _ = try unknownInto(a, o, @tagName(u.reason), u.feature),
        .known => _ = try unknownInto(a, o, "unbound", null),
    }
    const instance = std.fmt.bytesToHex(snap.instance, .lower);
    try o.put(a, "instance", .{ .string = try a.dupe(u8, &instance) });
    try o.put(a, "epoch", try json.intValue(a, snap.epoch));
    return .{ .object = o };
}
