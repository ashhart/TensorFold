const std = @import("std");
const runtime = @import("runtime.zig");
const engine = @import("engine.zig");
const prompts = @import("prompts.zig");
const slots = @import("slots.zig");
const mask = @import("mask.zig");
const profile = @import("profile.zig");
const outcome = @import("outcome.zig");
const memory_bank = @import("memory_bank.zig");
const memory_writer = @import("memory_writer.zig");
const recall_plan = @import("recall_plan.zig");
const frame_step = @import("frame_step.zig");
const image_tower = @import("image_tower.zig");
const elementwise = @import("elementwise.zig");
const spatial = @import("spatial.zig");

const Runtime = runtime.Runtime;
const Tensor = runtime.Tensor;
const FeatureMap = runtime.FeatureMap;
const Model = engine.Model;
const Outcome = outcome.Outcome;
const Entry = memory_bank.Entry;
const Levels = [3]FeatureMap;

pub const Direction = recall_plan.Direction;

pub const Options = struct {
    max_objects: usize = 16,
    frame_count: usize,
    direction: Direction = .forward,
    profile: bool = false,
};

pub const Seed = struct { id: slots.ObjectId, prompt: prompts.Prompt };

pub const TrackedMask = struct { id: slots.ObjectId, mask: mask.Mask };

pub const Prediction = struct {
    allocator: std.mem.Allocator,
    frame_index: usize,
    objects: []TrackedMask,

    pub fn deinit(self: *Prediction) void {
        for (self.objects) |object| self.allocator.free(object.mask.pixels);
        self.allocator.free(self.objects);
        self.* = undefined;
    }
};

pub const Group = struct {
    bank: memory_bank.Bank = .{},
    latest: ?Outcome = null,
    prompted_bits: u16 = 0,
};

pub fn checkFrameOrder(previous: ?usize, index: usize, frames: usize, direction: Direction) error{ SessionNotSeeded, FrameOutOfRange, NonConsecutiveFrame }!void {
    const before = previous orelse return error.SessionNotSeeded;
    if (index >= frames) return error.FrameOutOfRange;
    const consecutive = switch (direction) {
        .forward => index == before + 1,
        .reverse => before != 0 and index == before - 1,
    };
    if (!consecutive) return error.NonConsecutiveFrame;
}

const Keeper = struct {
    allocator: std.mem.Allocator,
    list: std.ArrayList(Tensor) = .empty,

    fn levels(self: *Keeper, maps: ?Levels) !void {
        if (maps) |value| for (value) |level| try self.list.append(self.allocator, level.tensor);
    }

    fn map(self: *Keeper, value: ?FeatureMap) !void {
        if (value) |m| try self.list.append(self.allocator, m.tensor);
    }

    fn outcomes(self: *Keeper, values: []const ?Outcome) !void {
        var buffer: outcome.TensorBuffer = undefined;
        for (values) |value| if (value) |o| try self.list.appendSlice(self.allocator, o.tensors(&buffer));
    }

    fn entries(self: *Keeper, values: []const ?Entry) !void {
        var buffer: [11]Tensor = undefined;
        for (values) |value| if (value) |e| try self.list.appendSlice(self.allocator, e.tensors(&buffer));
    }
};

// SAM3 squashes a prompted frame's memory with the step only when its prompt has points or boxes.
fn hasPointsOrBoxes(prompt: prompts.Prompt) bool {
    return prompt.points.len + prompt.boxes.len != 0;
}

fn releaseLevels(rt: *Runtime, levels: Levels) void {
    for (levels) |map| rt.release(map.tensor);
}

fn releaseOutcome(rt: *Runtime, value: Outcome) void {
    var buffer: outcome.TensorBuffer = undefined;
    for (value.tensors(&buffer)) |tensor| rt.release(tensor);
}

pub const Session = struct {
    allocator: std.mem.Allocator,
    model: *Model,
    options: Options,
    runtime: Runtime,
    objects: slots.Slots,
    groups: []Group,
    frame_index: ?usize = null,
    size: ?prompts.Size = null,
    tracking_levels: ?Levels = null,
    prompt_levels: ?Levels = null,
    trunk: ?FeatureMap = null,
    last_profile: profile.Frame = .{},

    pub fn init(allocator: std.mem.Allocator, model: *Model, options: Options) !Session {
        if (options.frame_count == 0) return error.EmptyVideo;
        const count = try model.config.buckets(options.max_objects);
        var objects = try slots.Slots.init(allocator, model.config, options.max_objects);
        errdefer objects.deinit();
        const groups = try allocator.alloc(Group, count);
        for (groups) |*group| group.* = .{};
        return .{
            .allocator = allocator,
            .model = model,
            .options = options,
            .runtime = model.runtime(allocator),
            .objects = objects,
            .groups = groups,
        };
    }

    pub fn deinit(self: *Session) void {
        self.model.mutex.lockUncancelable(self.model.io);
        defer self.model.mutex.unlock(self.model.io);
        self.runtime.deinit();
        for (self.groups) |*group| group.bank.deinit(self.allocator);
        self.allocator.free(self.groups);
        self.objects.deinit();
        self.* = undefined;
    }

    pub fn reset(self: *Session) !void {
        try self.model.mutex.lock(self.model.io);
        defer self.model.mutex.unlock(self.model.io);
        self.clear();
    }

    pub fn seed(self: *Session, allocator: std.mem.Allocator, image: prompts.Image, index: usize, seeds: []const Seed) !Prediction {
        if (index >= self.options.frame_count) return error.FrameOutOfRange;
        if (seeds.len == 0 or seeds.len > self.objects.ids.len) return error.InvalidObjectCount;
        try image.validate();
        for (seeds, 0..) |item, i| {
            try item.prompt.validate(image.size);
            for (seeds[0..i]) |earlier| if (earlier.id == item.id) return error.DuplicateObjectId;
        }
        try self.model.mutex.lock(self.model.io);
        defer self.model.mutex.unlock(self.model.io);
        var arena = std.heap.ArenaAllocator.init(self.allocator);
        defer arena.deinit();
        const saved = self.runtime.scratch;
        self.runtime.scratch = arena.allocator();
        defer self.runtime.scratch = saved;

        self.clear();
        errdefer self.clear();
        self.last_profile = .{};
        self.size = image.size;
        const rt = &self.runtime;
        const mark = rt.mark();
        const config = self.model.config;
        const sam31 = config.version == .sam31;
        const per_group = config.bucket_size;

        const encoded = try self.encode(image);
        var start = self.stamp();
        const prompt_levels = if (sam31) try image_tower.levels(rt, encoded.trunk, .prompt) else encoded.tracking;
        self.record(start, &self.last_profile.pyramid);

        const parts = try rt.scratch.alloc(?Outcome, self.groups.len * per_group);
        @memset(parts, null);
        const hard = try rt.scratch.alloc(bool, self.groups.len);
        @memset(hard, false);
        for (seeds) |item| {
            start = self.stamp();
            const slot = try self.objects.add(item.id);
            parts[slot.bucket * per_group + slot.slot] = try frame_step.prompted(rt, prompt_levels, item.prompt, image.size);
            hard[slot.bucket] = hard[slot.bucket] or hasPointsOrBoxes(item.prompt);
            self.record(start, &self.last_profile.tracker);
        }
        const merged = try rt.scratch.alloc(?Outcome, self.groups.len);
        const written = try rt.scratch.alloc(?Entry, self.groups.len);
        const bits = try rt.scratch.alloc(u16, self.groups.len);
        for (self.groups, 0..) |*group, g| {
            merged[g] = null;
            written[g] = null;
            bits[g] = self.activeBits(g);
            if (bits[g] == 0) continue;
            try group.bank.reserve(self.allocator);
            start = self.stamp();
            merged[g] = try outcome.merge(rt, parts[g * per_group ..][0..per_group]);
            self.record(start, &self.last_profile.tracker);
            start = self.stamp();
            written[g] = try memory_writer.write(rt, encoded.tracking[2].tensor, merged[g].?, .{
                .active = bits[g],
                .prompted = bits[g],
                .hard = config.version == .sam3 and hard[g],
                .frame = index,
            });
            self.record(start, &self.last_profile.memory);
        }
        var keeper = Keeper{ .allocator = rt.scratch };
        try keeper.levels(encoded.tracking);
        if (sam31) try keeper.levels(prompt_levels);
        try keeper.outcomes(merged);
        try keeper.entries(written);

        start = self.stamp();
        const prediction = try self.readout(allocator, merged, index);
        self.record(start, &self.last_profile.readback);

        start = self.stamp();
        rt.rewind(mark, keeper.list.items);
        for (self.groups, merged, written, bits) |*group, value, entry, active| {
            group.latest = value;
            group.prompted_bits = active;
            if (entry) |e| group.bank.put(rt, e);
        }
        self.frame_index = index;
        self.tracking_levels = encoded.tracking;
        self.prompt_levels = prompt_levels;
        self.record(start, &self.last_profile.state);
        return prediction;
    }

    pub fn track(self: *Session, allocator: std.mem.Allocator, image: prompts.Image, index: usize) !Prediction {
        try self.model.mutex.lock(self.model.io);
        defer self.model.mutex.unlock(self.model.io);
        try checkFrameOrder(self.frame_index, index, self.options.frame_count, self.options.direction);
        const size = self.size orelse return error.SessionNotSeeded;
        if (!std.meta.eql(image.size, size)) return error.VideoSizeChanged;
        try image.validate();
        var arena = std.heap.ArenaAllocator.init(self.allocator);
        defer arena.deinit();
        const saved = self.runtime.scratch;
        self.runtime.scratch = arena.allocator();
        defer self.runtime.scratch = saved;

        const rt = &self.runtime;
        const mark = rt.mark();
        errdefer rt.rewind(mark, &.{});
        self.last_profile = .{};
        const config = self.model.config;
        const sam31 = config.version == .sam31;
        const encoded = try self.encode(image);

        const results = try rt.scratch.alloc(?Outcome, self.groups.len);
        const written = try rt.scratch.alloc(?Entry, self.groups.len);
        for (self.groups, 0..) |*group, g| {
            results[g] = null;
            written[g] = null;
            const bits = self.activeBits(g);
            if (bits == 0) continue;
            try group.bank.reserve(self.allocator);
            var begin = self.stamp();
            const entries = group.bank.entries.items;
            results[g] = try frame_step.tracked(rt, encoded.tracking, entries, index, self.options.frame_count, self.options.direction, bits);
            self.record(begin, &self.last_profile.tracker);
            begin = self.stamp();
            written[g] = try memory_writer.write(rt, encoded.tracking[2].tensor, results[g].?, .{
                .active = bits,
                .prompted = 0,
                .hard = false,
                .frame = index,
            });
            self.record(begin, &self.last_profile.memory);
        }
        var keeper = Keeper{ .allocator = rt.scratch };
        try keeper.levels(encoded.tracking);
        if (sam31) try keeper.map(encoded.trunk);
        try keeper.outcomes(results);
        try keeper.entries(written);

        var start = self.stamp();
        const prediction = try self.readout(allocator, results, index);
        self.record(start, &self.last_profile.readback);

        start = self.stamp();
        rt.rewind(mark, keeper.list.items);
        if (self.tracking_levels) |old| releaseLevels(rt, old);
        if (sam31) if (self.prompt_levels) |old| releaseLevels(rt, old);
        if (self.trunk) |old| rt.release(old.tensor);
        for (self.groups, results, written) |*group, value, entry| {
            if (group.latest) |old| releaseOutcome(rt, old);
            group.latest = value;
            group.prompted_bits = 0;
            if (entry) |e| group.bank.put(rt, e);
            group.bank.prune(rt, index, self.options.frame_count, config.grouped());
        }
        self.frame_index = index;
        self.tracking_levels = encoded.tracking;
        self.prompt_levels = if (sam31) null else encoded.tracking;
        self.trunk = if (sam31) encoded.trunk else null;
        self.record(start, &self.last_profile.state);
        return prediction;
    }

    pub fn prompt(self: *Session, id: slots.ObjectId, value: prompts.Prompt) !void {
        try self.model.mutex.lock(self.model.io);
        defer self.model.mutex.unlock(self.model.io);
        const size = self.size orelse return error.SessionNotSeeded;
        try value.validate(size);
        const global = self.slotOf(id) orelse self.freeSlot() orelse return error.TooManyObjects;
        try self.amend(id, global, value);
    }

    /// Masks of the current frame as they stand, including amendments made since the frame was seeded or tracked.
    pub fn currentMasks(self: *Session, allocator: std.mem.Allocator) !Prediction {
        try self.model.mutex.lock(self.model.io);
        defer self.model.mutex.unlock(self.model.io);
        const index = self.frame_index orelse return error.SessionNotSeeded;
        var arena = std.heap.ArenaAllocator.init(self.allocator);
        defer arena.deinit();
        const saved = self.runtime.scratch;
        self.runtime.scratch = arena.allocator();
        defer self.runtime.scratch = saved;
        const results = try self.runtime.scratch.alloc(?Outcome, self.groups.len);
        for (self.groups, results) |group, *result| result.* = group.latest;
        return self.readout(allocator, results, index);
    }

    pub fn removeObject(self: *Session, id: slots.ObjectId) !void {
        try self.model.mutex.lock(self.model.io);
        defer self.model.mutex.unlock(self.model.io);
        const global = self.slotOf(id) orelse return error.UnknownObject;
        try self.amend(id, global, null);
    }

    fn amend(self: *Session, id: slots.ObjectId, global: usize, value: ?prompts.Prompt) !void {
        const current = self.frame_index orelse return error.SessionNotSeeded;
        const config = self.model.config;
        const per_group = config.bucket_size;
        const g = global / per_group;
        const s = global % per_group;
        if (self.objects.epochs[g] == std.math.maxInt(u64)) return error.EpochOverflow;
        var arena = std.heap.ArenaAllocator.init(self.allocator);
        defer arena.deinit();
        const saved = self.runtime.scratch;
        self.runtime.scratch = arena.allocator();
        defer self.runtime.scratch = saved;

        const rt = &self.runtime;
        const mark = rt.mark();
        errdefer rt.rewind(mark, &.{});
        const group = &self.groups[g];
        try group.bank.reserve(self.allocator);
        var built: ?Levels = null;
        if (value != null and self.prompt_levels == null) {
            const trunk = self.trunk orelse return error.MissingObjectState;
            built = try image_tower.levels(rt, trunk, .prompt);
        }
        const tracking = self.tracking_levels orelse return error.MissingObjectState;
        const active = self.activeBits(g);
        const bit = @as(u16, 1) << @intCast(s);
        const parts = try rt.scratch.alloc(?Outcome, per_group);
        @memset(parts, null);
        for (0..per_group) |other| {
            if (other == s or active & (@as(u16, 1) << @intCast(other)) == 0) continue;
            const latest = group.latest orelse return error.MissingObjectState;
            parts[other] = try outcome.single(rt, latest, other);
        }
        var bits = active & ~bit;
        if (value) |request| {
            const levels = built orelse self.prompt_levels.?;
            parts[s] = try frame_step.prompted(rt, levels, request, self.size.?);
            bits |= bit;
        }
        const prompted_bits = (group.prompted_bits & bits & ~bit) | (if (value != null) bit else 0);
        var merged: [1]?Outcome = .{null};
        var written: [1]?Entry = .{null};
        if (bits != 0) {
            merged[0] = try outcome.merge(rt, parts);
            written[0] = try memory_writer.write(rt, tracking[2].tensor, merged[0].?, .{
                .active = bits,
                .prompted = prompted_bits,
                .hard = config.version == .sam3 and prompted_bits != 0 and (if (value) |v| hasPointsOrBoxes(v) else true),
                .frame = current,
            });
        }
        var keeper = Keeper{ .allocator = rt.scratch };
        try keeper.levels(built);
        try keeper.outcomes(&merged);
        try keeper.entries(&written);

        rt.rewind(mark, keeper.list.items);
        if (built) |levels| {
            self.prompt_levels = levels;
            if (self.trunk) |trunk| rt.release(trunk.tensor);
            self.trunk = null;
        }
        if (value != null)
            _ = self.objects.add(id) catch unreachable
        else
            _ = self.objects.remove(id) catch unreachable;
        // The amended frame's entry replaces that frame's entry; every other entry stays.
        if (written[0]) |entry| group.bank.put(rt, entry) else group.bank.clear(rt);
        if (group.latest) |old| releaseOutcome(rt, old);
        group.latest = merged[0];
        group.prompted_bits = prompted_bits;
    }

    const Encoded = struct { trunk: FeatureMap, tracking: Levels };

    fn encode(self: *Session, image: prompts.Image) !Encoded {
        const rt = &self.runtime;
        var start = self.stamp();
        const prepared = try image_tower.prepare(rt, image);
        self.record(start, &self.last_profile.preprocess);
        start = self.stamp();
        const trunk = try image_tower.encode(rt, prepared);
        self.record(start, &self.last_profile.trunk);
        start = self.stamp();
        const tracking = try image_tower.levels(rt, trunk, .tracking);
        self.record(start, &self.last_profile.pyramid);
        return .{ .trunk = trunk, .tracking = tracking };
    }

    fn readout(self: *Session, allocator: std.mem.Allocator, results: []const ?Outcome, frame: usize) !Prediction {
        const rt = &self.runtime;
        const size = self.size.?;
        const per_group = self.model.config.bucket_size;
        var objects: std.ArrayList(TrackedMask) = .empty;
        errdefer {
            for (objects.items) |object| allocator.free(object.mask.pixels);
            objects.deinit(allocator);
        }
        for (self.objects.ids, 0..) |maybe_id, global| {
            const id = maybe_id orelse continue;
            const result = results[global / per_group] orelse return error.MissingObjectState;
            const s = global % per_group;
            const logits = result.shown();
            const mark = rt.mark();
            defer rt.rewind(mark, &.{});
            const channel = try elementwise.columns(rt, logits.tensor, logits.height * logits.width, logits.channels, s, 1);
            const probability = @as(f32, 1) / (1 + @exp(-result.presence[s]));
            try objects.ensureUnusedCapacity(allocator, 1);
            const value = try spatial.sourceMask(rt, allocator, FeatureMap.of(channel, logits.height, logits.width, 1), size, probability);
            objects.appendAssumeCapacity(.{ .id = id, .mask = value });
        }
        return .{ .allocator = allocator, .frame_index = frame, .objects = try objects.toOwnedSlice(allocator) };
    }

    fn clear(self: *Session) void {
        self.runtime.rewind(.{ .held = 0 }, &.{});
        for (self.groups) |*group| {
            group.bank.entries.clearRetainingCapacity();
            group.latest = null;
            group.prompted_bits = 0;
        }
        @memset(self.objects.ids, null);
        for (self.objects.epochs) |*epoch| epoch.* +%= 1;
        self.frame_index = null;
        self.size = null;
        self.tracking_levels = null;
        self.prompt_levels = null;
        self.trunk = null;
    }

    fn activeBits(self: *const Session, g: usize) u16 {
        const per_group = self.model.config.bucket_size;
        var bits: u16 = 0;
        for (0..per_group) |s| {
            const global = g * per_group + s;
            if (global < self.objects.ids.len and self.objects.ids[global] != null) bits |= @as(u16, 1) << @intCast(s);
        }
        return bits;
    }

    fn slotOf(self: *const Session, id: slots.ObjectId) ?usize {
        for (self.objects.ids, 0..) |value, i| if (value == id) return i;
        return null;
    }

    fn freeSlot(self: *const Session) ?usize {
        for (self.objects.ids, 0..) |value, i| if (value == null) return i;
        return null;
    }

    fn stamp(self: *const Session) profile.Stamp {
        return profile.Stamp.take(self.model.gpu, self.options.profile);
    }

    fn record(self: *const Session, start: profile.Stamp, timing: *profile.Timing) void {
        start.record(self.model.gpu, self.options.profile, timing);
    }
};

test "SAM tracking session accepts only the next frame in its direction" {
    try checkFrameOrder(12, 13, 30, .forward);
    try checkFrameOrder(12, 11, 30, .reverse);
    try std.testing.expectError(error.SessionNotSeeded, checkFrameOrder(null, 0, 30, .forward));
    try std.testing.expectError(error.NonConsecutiveFrame, checkFrameOrder(12, 11, 30, .forward));
    try std.testing.expectError(error.NonConsecutiveFrame, checkFrameOrder(12, 14, 30, .forward));
    try std.testing.expectError(error.NonConsecutiveFrame, checkFrameOrder(12, 13, 30, .reverse));
    try std.testing.expectError(error.FrameOutOfRange, checkFrameOrder(29, 30, 30, .forward));
    try std.testing.expectError(error.NonConsecutiveFrame, checkFrameOrder(0, 0, 30, .reverse));
}
