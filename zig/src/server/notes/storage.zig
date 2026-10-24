//! Journaled model ownership and its n-gram index; each write and erase rebuilds the whole index.
const std = @import("std");
const model_mod = @import("model.zig");
const write = @import("write.zig");
const journal_mod = @import("journal.zig");
const records = @import("../notes_records.zig");
const model_text = @import("../model_text.zig");
const zero = @import("zero_alloc.zig");
const tokenizer_check = @import("tokenizer_check.zig");
const index = @import("index.zig");
const alias = @import("alias.zig");
const Allocator = std.mem.Allocator;
pub const tokenizer_limit: usize = 256 << 20;
pub const Check = enum { sha256, vocabulary_only };
const Identity = struct { io: std.Io, model_dir: []const u8 };

pub const Storage = struct {
    parent: Allocator,
    zeroing: zero.Zeroing,
    model: *model_mod.Model,
    text: model_text.Text,
    log: records.Log,
    journal: ?journal_mod.Journal = null,
    snapshot: []u8,
    snapshot_sha: [64]u8,
    snapshot_path: ?[]u8 = null,
    io: ?std.Io = null,
    replayed: usize = 0,
    check: Check = .vocabulary_only,
    started: bool = false,
    start_source: u64 = 0,
    kind: index.Kind = index.default_kind,
    idx: ?*index.Index = null,

    /// File-backed startup checks tokenizer identity before opening or replaying the journal.
    pub fn open(parent: Allocator, io: std.Io, path: []const u8, text: model_text.Text, model_dir: []const u8, problem: *[]const u8) !*Storage {
        return openKind(parent, io, path, text, model_dir, index.default_kind, problem);
    }

    pub fn openKind(parent: Allocator, io: std.Io, path: []const u8, text: model_text.Text, model_dir: []const u8, kind: index.Kind, problem: *[]const u8) !*Storage {
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, parent, .unlimited);
        defer parent.free(bytes);
        const journal_path = try std.mem.concat(parent, u8, &.{ path, ".journal" });
        defer parent.free(journal_path);
        const s = try loadWith(parent, bytes, text, journal_path, .{ .io = io, .model_dir = model_dir }, kind, problem);
        errdefer s.deinit();
        s.snapshot_path = try s.zeroing.allocator().dupe(u8, path);
        s.io = io;
        return s;
    }

    pub fn load(parent: Allocator, bytes: []const u8, text: model_text.Text, path: ?[]const u8, problem: *[]const u8) !*Storage {
        return loadKind(parent, bytes, text, path, index.default_kind, problem);
    }

    pub fn loadKind(parent: Allocator, bytes: []const u8, text: model_text.Text, path: ?[]const u8, kind: index.Kind, problem: *[]const u8) !*Storage {
        return loadWith(parent, bytes, text, path, null, kind, problem);
    }

    fn loadWith(parent: Allocator, bytes: []const u8, text: model_text.Text, path: ?[]const u8, identity: ?Identity, kind: index.Kind, problem: *[]const u8) !*Storage {
        const s = try parent.create(Storage);
        errdefer parent.destroy(s);
        s.zeroing = .{ .child = parent };
        const g = s.zeroing.allocator();
        const m = try model_mod.parse(g, bytes);
        errdefer m.deinit();
        if (m.vkey.len != text.vocabSize()) {
            problem.* = "the snapshot's vocabulary is not the model tokenizer's size";
            return error.BadSnapshot;
        }
        var checked: Check = .vocabulary_only;
        if (identity) |id| if (m.tokenizer_sha256.len > 0) {
            const sha = try tokenizer_check.tokenizerSha(g, id.io, id.model_dir, tokenizer_limit, problem);
            if (!std.mem.eql(u8, &sha, m.tokenizer_sha256)) {
                problem.* = "the snapshot names a tokenizer.json SHA-256 that is not this model's";
                return error.TokenizerMismatch;
            }
            checked = .sha256;
        };
        const snapshot = try g.dupe(u8, bytes);
        errdefer g.free(snapshot);
        var sha: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(bytes, &sha, .{});
        s.* = .{
            .parent = parent,
            .zeroing = s.zeroing,
            .model = m,
            .text = text,
            .log = .{ .gpa = g },
            .snapshot = snapshot,
            .snapshot_sha = std.fmt.bytesToHex(sha, .lower),
            .check = checked,
            .kind = kind,
        };
        errdefer s.log.deinit();
        if (path) |p| {
            var arena = std.heap.ArenaAllocator.init(g);
            defer arena.deinit();
            var recs: std.ArrayList(write.Record) = .empty;
            s.journal = try journal_mod.Journal.open(g, arena.allocator(), p, s.snapshot_sha, &recs, problem);
            errdefer s.journal.?.deinit();
            for (recs.items) |old_record| {
                var r = old_record;
                if (r.id == 0) r.id = m.nextId();
                var out: write.Outcome = undefined;
                var undo: write.Undo = .{};
                defer undo.deinit(g);
                try write.apply(m, text, arena.allocator(), r, &out, &undo);
                write.commit(m, &undo);
                if (old_record.id == 0 and out.note == 0) r.id = 0;
                if (r.id != 0) m.id_floor = @max(m.nextId(), r.id + 1);
                try s.log.add(r, out, m);
            }
            try m.trim();
            s.replayed = recs.items.len;
            s.log.next_source = @max(s.log.next_source, s.journal.?.next_source);
            m.id_floor = @max(m.nextId(), s.journal.?.next_id);
            s.journal.?.next_id = m.id_floor;
            if (s.journal.?.upgraded) {
                var stable: std.ArrayList(write.Record) = .empty;
                for (s.log.items.items) |e| try stable.append(arena.allocator(), e.r);
                try s.journal.?.rewrite(arena.allocator(), stable.items, s.log.next_source);
            }
        }
        s.idx = index.Index.buildKind(g, m, kind) catch |e| {
            if (e == error.Unsupported) problem.* = "the suffix-array index needs a vocabulary whose keys are all real and whose ids of one key agree on being a word";
            return e;
        };
        errdefer s.idx.?.deinit();
        try alias.populate(s.idx.?, m, text);
        return s;
    }

    pub fn deinit(s: *Storage) void {
        const parent = s.parent;
        if (s.idx) |idx| idx.deinit();
        if (s.journal) |*j| j.deinit();
        s.log.deinit();
        s.model.deinit();
        s.zeroing.allocator().free(s.snapshot);
        if (s.snapshot_path) |path| s.zeroing.allocator().free(path);
        parent.destroy(s);
    }

    /// Record this process's actual writer flags before accepting any new write.
    pub fn beginRun(s: *Storage, a: Allocator, run: journal_mod.Run) !void {
        const j = if (s.journal) |*x| x else return error.ReadOnly;
        if (j.poisoned) return error.JournalPoisoned;
        if (s.started) return error.AlreadyStarted;
        j.addRun(a, run);
        if (j.poisoned) return error.JournalPoisoned;
        s.start_source = s.log.next_source;
        s.started = true;
    }

    /// The caller holds its writer lock; output slices belong to a, not to the durable log.
    pub fn apply(s: *Storage, a: Allocator, r: write.Record) !write.Outcome {
        var pending = try s.prepare(a, r);
        defer pending.deinit();
        try pending.commit(a);
        return pending.out;
    }

    /// Prepare all fallible reply/index allocations before commit; caller holds its writer lock throughout.
    pub fn prepare(s: *Storage, a: Allocator, r: write.Record) !Pending {
        const j = if (s.journal) |*x| x else return error.ReadOnly;
        if (j.poisoned) return error.JournalPoisoned;
        if (!s.started) return error.MissingRun;
        const g = s.zeroing.allocator();
        var out: write.Outcome = undefined;
        var undo: write.Undo = .{};
        errdefer undo.deinit(g);
        var record = r;
        record.id = s.model.nextId();
        try write.apply(s.model, s.text, a, record, &out, &undo);
        errdefer write.revert(s.model, &undo);
        var entry = try s.log.prepare(record, out, s.model);
        errdefer records.freeEntry(g, &entry);
        const idx = try index.Index.buildKind(g, s.model, s.kind);
        errdefer idx.deinit();
        try alias.populate(idx, s.model, s.text);
        return .{ .storage = s, .record = record, .out = out, .undo = undo, .entry = entry, .idx = idx };
    }
};

pub const Pending = struct {
    storage: *Storage,
    record: write.Record,
    out: write.Outcome,
    undo: write.Undo,
    entry: records.Entry,
    idx: *index.Index,
    committed: bool = false,

    pub fn deinit(p: *Pending) void {
        const s = p.storage;
        const g = s.zeroing.allocator();
        if (!p.committed) {
            write.revert(s.model, &p.undo);
            records.freeEntry(g, &p.entry);
            p.idx.deinit();
        }
        p.undo.deinit(g);
    }

    pub fn commit(p: *Pending, a: Allocator) !void {
        if (p.committed) return error.AlreadyCommitted;
        const s = p.storage;
        try s.journal.?.append(a, p.record);
        write.commit(s.model, &p.undo);
        s.log.push(p.entry);
        s.model.id_floor = @max(s.model.nextId(), p.record.id + 1);
        s.journal.?.next_id = s.model.nextId();
        const old = s.idx;
        s.idx = p.idx;
        if (old) |x| x.deinit();
        p.committed = true;
    }
};
