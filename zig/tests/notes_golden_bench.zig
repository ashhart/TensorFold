//! notes-golden MODEL_DIR [--index K] [--lang L] [--hop H] SNAPSHOT REF.json...: each question as the server asks it.
const std = @import("std");
const server = @import("server");
const Storage = server.notes_storage.Storage;
const retrieve = server.notes_retrieve.retrieve;
const lang_mod = server.notes_lang;
const index = server.notes_index;

pub fn main(init: std.process.Init) !u8 {
    const gpa = init.gpa;
    const io = init.io;
    const a = init.arena.allocator();
    const argv = try init.minimal.args.toSlice(a);
    if (argv.len < 4) {
        std.debug.print("usage: notes-golden MODEL_DIR [--index postings|sa (default sa)] [--lang en|de|fi (default en)] [--hop subject|subject_value|none] SNAPSHOT REF.json [SNAPSHOT REF.json ...]\n", .{});
        return 2;
    }
    var at: usize = 2;
    var kind: index.Kind = index.default_kind;
    var lang: lang_mod.Lang = .en;
    var hop: ?index.Hop = null;
    while (at + 1 < argv.len and std.mem.startsWith(u8, argv[at], "--")) {
        if (std.mem.eql(u8, argv[at], "--index")) {
            kind = std.meta.stringToEnum(index.Kind, argv[at + 1]) orelse {
                std.debug.print("--index: postings or sa\n", .{});
                return 2;
            };
        } else if (std.mem.eql(u8, argv[at], "--lang")) {
            lang = lang_mod.Lang.parse(argv[at + 1]) orelse {
                std.debug.print("--lang: en, de or fi\n", .{});
                return 2;
            };
        } else if (std.mem.eql(u8, argv[at], "--hop")) {
            hop = std.meta.stringToEnum(index.Hop, argv[at + 1]) orelse {
                std.debug.print("--hop: subject, subject_value or none\n", .{});
                return 2;
            };
        } else {
            std.debug.print("unknown flag {s}\n", .{argv[at]});
            return 2;
        }
        at += 2;
    }
    if ((argv.len - at) % 2 != 0 or argv.len == at) {
        std.debug.print("SNAPSHOT REF.json pairs\n", .{});
        return 2;
    }
    var problem: []const u8 = "";
    const ht = try server.hf_text.HfText.load(gpa, io, argv[1], null, a, &problem);
    defer ht.deinit();
    const text = ht.text();
    var out: std.Io.Writer.Allocating = .init(a);
    const Tally = struct { questions: usize = 0, all_equal: usize = 0, block: usize = 0, ids: usize = 0, tokens: usize = 0, past: usize = 0, candidates: usize = 0, retrieving: usize = 0 };
    var total: Tally = .{};
    var k = at;
    while (k < argv.len) : (k += 2) {
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, argv[k], gpa, .unlimited);
        defer gpa.free(bytes);
        const store = Storage.loadKind(gpa, bytes, text, null, kind, &problem) catch |e| {
            std.debug.print("{s}: {s} ({s})\n", .{ argv[k], problem, @errorName(e) });
            return 1;
        };
        defer store.deinit();
        const ref = try std.Io.Dir.cwd().readFileAlloc(io, argv[k + 1], a, .limited(64 << 20));
        const doc = (try server.json.parse(a, ref)).ok;
        var t: Tally = .{};
        for (doc.get("questions").?.items().?) |q| {
            var qa = std.heap.ArenaAllocator.init(gpa);
            defer qa.deinit();
            const qa_a = qa.allocator();
            const question = q.strField("question").?;
            const given: ?bool = if (q.field("past_given")) |g| g.bool else null;
            const past = given orelse lang_mod.isPast(question, lang);
            var ex: std.ArrayList(u32) = .empty;
            if (q.field("exclude")) |xs| for (xs.items().?) |x| try ex.append(qa_a, @intCast(x.int64().?));
            const qids = try text.encode(qa_a, question, false);
            const cap: u32 = if (q.field("cap")) |c| blk: {
                const n = c.int64() orelse return 2;
                if (n < 1 or n > std.math.maxInt(u32)) return 2;
                break :blk @intCast(n);
            } else store.model.settings.cap;
            const found = try retrieve(store, io, qa_a, text, qids, past, cap, ex.items, .{ .hop = hop });
            const want_ids = q.get("ids").?.items().?;
            var ids_eq = want_ids.len == found.ids.len;
            if (ids_eq) for (want_ids, found.ids) |w, f| {
                ids_eq = ids_eq and w.int64().? == f;
            };
            const block_eq = std.mem.eql(u8, q.strField("block").?, found.block);
            const tokens_eq = q.get("tokens").?.int64().? == @as(i64, @intCast(found.tokens));
            const past_eq = q.get("past").?.bool == past;
            const cand_eq = q.get("candidates").?.int64().? == found.candidates;
            t.questions += 1;
            t.block += @intFromBool(block_eq);
            t.ids += @intFromBool(ids_eq);
            t.tokens += @intFromBool(tokens_eq);
            t.past += @intFromBool(past_eq);
            t.candidates += @intFromBool(cand_eq);
            t.all_equal += @intFromBool(block_eq and ids_eq and tokens_eq and past_eq and cand_eq);
            t.retrieving += @intFromBool(found.block.len > 0);
        }
        try out.writer.print("{{\"snapshot\":\"{s}\",\"ref\":\"{s}\",\"index\":\"{s}\",\"questions\":{d},\"all_equal\":{d},\"block\":{d},\"ids\":{d},\"tokens\":{d},\"past\":{d},\"candidates\":{d},\"retrieving\":{d}}}\n", .{
            std.fs.path.basename(argv[k]), std.fs.path.basename(argv[k + 1]), @tagName(kind), t.questions, t.all_equal, t.block, t.ids, t.tokens, t.past, t.candidates, t.retrieving,
        });
        inline for (@typeInfo(Tally).@"struct".field_names) |f| @field(total, f) += @field(t, f);
    }
    try out.writer.print("{{\"total\":true,\"index\":\"{s}\",\"questions\":{d},\"all_equal\":{d},\"block\":{d},\"ids\":{d},\"tokens\":{d},\"past\":{d},\"candidates\":{d},\"retrieving\":{d}}}\n", .{
        @tagName(kind), total.questions, total.all_equal, total.block, total.ids, total.tokens, total.past, total.candidates, total.retrieving,
    });
    try std.Io.File.stdout().writeStreamingAll(io, out.written());
    return if (total.all_equal == total.questions) 0 else 1;
}
