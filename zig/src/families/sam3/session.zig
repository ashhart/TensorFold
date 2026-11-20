const std = @import("std");
const ops = @import("ops.zig");
const vision = @import("vision.zig");
const model = @import("model.zig");
const tracker = @import("tracker.zig");
const memory = @import("memory.zig");
const prompts = @import("prompts.zig");
const slots = @import("slots.zig");
const profile = @import("profile.zig");

const selection = @import("memory_selection.zig");
pub const Direction = selection.Direction;
pub const Options = struct { max_objects: usize = 16, frame_count: usize, direction: Direction = .forward, profile: bool = false };
pub const Seed = struct { id: slots.ObjectId, prompt: prompts.Prompt };
pub const TrackedMask = struct { id: slots.ObjectId, mask: model.Mask };
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
const Bucket = struct {
    history: std.ArrayList(memory.Memory) = .empty,
    last: ?tracker.Result = null,
    conditioning: u16 = 0,
};

pub const Session = struct {
    allocator: std.mem.Allocator,
    model: *model.Model,
    options: Options,
    r: ops.Runtime,
    objects: slots.Slots,
    buckets: []Bucket,
    frame_index: ?usize = null,
    size: ?prompts.Size = null,
    features: ?[3]ops.Feature = null,
    interactive: ?[3]ops.Feature = null,
    trunk: ?ops.Feature = null,
    last_profile: profile.Frame = .{},

    pub fn init(allocator: std.mem.Allocator, loaded: *model.Model, options: Options) !Session {
        if (options.frame_count == 0) return error.EmptyVideo;
        var objects = try slots.Slots.init(allocator, loaded.config, options.max_objects);
        errdefer objects.deinit();
        const buckets = try allocator.alloc(Bucket, objects.epochs.len);
        @memset(buckets, .{});
        return .{ .allocator = allocator, .model = loaded, .options = options, .r = loaded.runtime(allocator), .objects = objects, .buckets = buckets };
    }

    pub fn deinit(self: *Session) void {
        const loaded = self.model;
        loaded.mutex.lockUncancelable(loaded.io);
        defer loaded.mutex.unlock(loaded.io);
        self.r.deinit();
        for (self.buckets) |*bucket| bucket.history.deinit(self.allocator);
        self.allocator.free(self.buckets);
        self.objects.deinit();
        self.* = undefined;
    }

    fn clear(self: *Session) void {
        self.r.trim(0, &.{});
        for (self.buckets) |*bucket| {
            bucket.history.clearRetainingCapacity();
            bucket.last = null;
            bucket.conditioning = 0;
        }
        @memset(self.objects.ids, null);
        for (self.objects.epochs) |*epoch| epoch.* +%= 1;
        self.frame_index = null;
        self.size = null;
        self.features = null;
        self.interactive = null;
        self.trunk = null;
    }

    pub fn reset(self: *Session) !void {
        try self.model.mutex.lock(self.model.io);
        defer self.model.mutex.unlock(self.model.io);
        self.clear();
    }

    fn active(self: *const Session, bucket: usize) u16 {
        var bits: u16 = 0;
        for (0..self.model.config.bucket_size) |slot| {
            const at = bucket * self.model.config.bucket_size + slot;
            if (at < self.objects.ids.len and self.objects.ids[at] != null)
                bits |= @as(u16, 1) << @intCast(slot);
        }
        return bits;
    }

    fn encode(self: *Session, image: prompts.Image, needs_interactive: bool) !struct {
        [3]ops.Feature,
        ?[3]ops.Feature,
        ?ops.Feature,
    } {
        const prepare_start = profile.Stamp.take(self.r.g, self.options.profile);
        const input = try model.prepareImage(&self.r, image);
        prepare_start.record(self.r.g, self.options.profile, &self.last_profile.preprocess);
        const trunk_start = profile.Stamp.take(self.r.g, self.options.profile);
        const trunk = try vision.trunk(&self.r, input);
        trunk_start.record(self.r.g, self.options.profile, &self.last_profile.trunk);
        const pyramid_start = profile.Stamp.take(self.r.g, self.options.profile);
        const propagation = try vision.pyramid(&self.r, trunk, self.model.config.pyramidPrefix(false));
        const interactive: ?[3]ops.Feature = if (self.model.config.version == .sam3) propagation else if (needs_interactive) try vision.pyramid(&self.r, trunk, self.model.config.pyramidPrefix(true)) else null;
        pyramid_start.record(self.r.g, self.options.profile, &self.last_profile.pyramid);
        return .{ propagation, interactive, if (interactive == null) trunk else null };
    }

    fn retained(self: *Session, features: [3]ops.Feature, interactive: ?[3]ops.Feature, trunk: ?ops.Feature, results: []const ?tracker.Result, memories: []const ?memory.Memory) ![]ops.Tensor {
        var keep: std.ArrayList(ops.Tensor) = .empty;
        const a = self.r.a;
        for (features) |f| try keep.append(a, f.tensor);
        if (interactive) |levels| for (levels) |f| try keep.append(a, f.tensor);
        if (trunk) |f| try keep.append(a, f.tensor);
        for (results) |result| if (result) |out| {
            var storage: [3]ops.Tensor = undefined;
            try keep.appendSlice(a, out.retained(&storage));
        };
        for (memories) |past| if (past) |m| {
            var storage: [11]ops.Tensor = undefined;
            try keep.appendSlice(a, m.retained(&storage));
        };
        return keep.items;
    }

    fn snapshot(self: *Session, allocator: std.mem.Allocator, index: usize, results: []const ?tracker.Result) !Prediction {
        var count: usize = 0;
        for (self.objects.ids) |id| if (id != null) {
            count += 1;
        };
        const output = try allocator.alloc(TrackedMask, count);
        var ready: usize = 0;
        errdefer {
            for (output[0..ready]) |item| allocator.free(item.mask.pixels);
            allocator.free(output);
        }
        for (self.objects.ids, 0..) |maybe, at| {
            const id = maybe orelse continue;
            const result = results[at / self.model.config.bucket_size] orelse return error.MissingObjectState;
            const slot = at % self.model.config.bucket_size;
            const mark = self.r.mark();
            const masks = result.input_masks orelse result.masks;
            const channel = try self.r.columns(masks.tensor, masks.height * masks.width, result.count, slot, 1);
            output[ready] = .{ .id = id, .mask = try model.readMask(&self.r, allocator, ops.feature(channel, masks.height, masks.width, 1), self.size.?, 1 / (1 + @exp(-result.scores[slot]))) };
            ready += 1;
            self.r.trim(mark, &.{});
        }
        return .{ .allocator = allocator, .frame_index = index, .objects = output };
    }

    pub fn seed(self: *Session, allocator: std.mem.Allocator, image: prompts.Image, index: usize, seeds: []const Seed) !Prediction {
        if (index >= self.options.frame_count) return error.FrameOutOfRange;
        if (seeds.len == 0 or seeds.len > self.objects.ids.len) return error.InvalidObjectCount;
        try image.validate();
        for (seeds, 0..) |item, i| {
            try item.prompt.validate(image.size);
            for (seeds[0..i]) |previous| if (previous.id == item.id) return error.DuplicateObjectId;
        }
        try self.model.mutex.lock(self.model.io);
        defer self.model.mutex.unlock(self.model.io);
        var arena = std.heap.ArenaAllocator.init(self.allocator);
        defer arena.deinit();
        self.r.a = arena.allocator();
        defer self.r.a = self.allocator;
        self.clear();
        errdefer self.clear();
        self.last_profile = .{};
        self.size = image.size;
        const features, const interactive, const trunk = try self.encode(image, true);
        const per_object = try self.r.a.alloc(?tracker.Result, self.objects.ids.len);
        @memset(per_object, null);
        const seed_start = profile.Stamp.take(self.r.g, self.options.profile);
        for (seeds) |item| {
            const slot = try self.objects.add(item.id);
            per_object[slot.bucket * self.model.config.bucket_size + slot.slot] =
                try tracker.seed(&self.r, interactive.?, item.prompt, image.size);
        }
        seed_start.record(self.r.g, self.options.profile, &self.last_profile.tracker);
        const results = try self.r.a.alloc(?tracker.Result, self.buckets.len);
        const memories = try self.r.a.alloc(?memory.Memory, self.buckets.len);
        @memset(results, null);
        @memset(memories, null);
        for (self.buckets, 0..) |*bucket, b| {
            const bits = self.active(b);
            if (bits == 0) continue;
            try bucket.history.ensureUnusedCapacity(self.allocator, 1);
            const first = b * self.model.config.bucket_size;
            const combine_start = profile.Stamp.take(self.r.g, self.options.profile);
            results[b] = try combine(&self.r, per_object[first..@min(first + self.model.config.bucket_size, per_object.len)]);
            combine_start.record(self.r.g, self.options.profile, &self.last_profile.tracker);
            const from_points = self.model.config.version == .sam31 or
                seeds[b].prompt.points.len != 0 or seeds[b].prompt.boxes.len != 0;
            const memory_start = profile.Stamp.take(self.r.g, self.options.profile);
            memories[b] = try memory.encode(&self.r, features[2].tensor, results[b].?, bits, bits, from_points, index);
            memory_start.record(self.r.g, self.options.profile, &self.last_profile.memory);
        }
        const readback_start = profile.Stamp.take(self.r.g, self.options.profile);
        var output = try self.snapshot(allocator, index, results);
        errdefer output.deinit();
        readback_start.record(self.r.g, self.options.profile, &self.last_profile.readback);
        const state_start = profile.Stamp.take(self.r.g, self.options.profile);
        self.r.trim(0, try self.retained(features, interactive, trunk, results, memories));
        for (self.buckets, 0..) |*bucket, b| {
            bucket.last = results[b];
            bucket.conditioning = self.active(b);
            if (memories[b]) |past| bucket.history.appendAssumeCapacity(past);
        }
        self.frame_index = index;
        self.features = features;
        self.interactive = interactive;
        self.trunk = trunk;
        state_start.record(self.r.g, self.options.profile, &self.last_profile.state);
        return output;
    }

    pub fn track(self: *Session, allocator: std.mem.Allocator, image: prompts.Image, index: usize) !Prediction {
        try self.model.mutex.lock(self.model.io);
        defer self.model.mutex.unlock(self.model.io);
        try validateNext(self.frame_index, index, self.options);
        if (!std.meta.eql(image.size, self.size.?)) return error.VideoSizeChanged;
        try image.validate();
        var arena = std.heap.ArenaAllocator.init(self.allocator);
        defer arena.deinit();
        self.r.a = arena.allocator();
        defer self.r.a = self.allocator;
        const start = self.r.mark();
        errdefer self.r.trim(start, &.{});
        self.last_profile = .{};
        const features, const interactive, const trunk = try self.encode(image, false);
        const results = try self.r.a.alloc(?tracker.Result, self.buckets.len);
        const memories = try self.r.a.alloc(?memory.Memory, self.buckets.len);
        @memset(results, null);
        @memset(memories, null);
        for (self.buckets, 0..) |*bucket, b| {
            const bits = self.active(b);
            if (bits == 0) continue;
            try bucket.history.ensureUnusedCapacity(self.allocator, 1);
            const tracker_start = profile.Stamp.take(self.r.g, self.options.profile);
            results[b] = try tracker.propagate(&self.r, features, bucket.history.items, index, self.options.frame_count, self.options.direction, bits);
            tracker_start.record(self.r.g, self.options.profile, &self.last_profile.tracker);
            const memory_start = profile.Stamp.take(self.r.g, self.options.profile);
            memories[b] = try memory.encode(&self.r, features[2].tensor, results[b].?, bits, 0, false, index);
            memory_start.record(self.r.g, self.options.profile, &self.last_profile.memory);
        }
        const readback_start = profile.Stamp.take(self.r.g, self.options.profile);
        var output = try self.snapshot(allocator, index, results);
        errdefer output.deinit();
        readback_start.record(self.r.g, self.options.profile, &self.last_profile.readback);
        const state_start = profile.Stamp.take(self.r.g, self.options.profile);
        self.r.trim(start, try self.retained(features, interactive, trunk, results, memories));
        self.releaseFeatures();
        for (self.buckets, 0..) |*bucket, b| {
            self.releaseLast(bucket);
            bucket.last = results[b];
            bucket.conditioning = 0;
            if (memories[b]) |past| bucket.history.appendAssumeCapacity(past);
            var i: usize = 0;
            while (i < bucket.history.items.len) {
                const past = &bucket.history.items[i];
                const retain = selection.retention(bucket.history.items, i, index, self.options.frame_count, self.model.config.version == .sam31);
                if (!retain.keep) {
                    past.release(&self.r);
                    _ = bucket.history.orderedRemove(i);
                } else {
                    if (!retain.spatial) past.expireSpatial(&self.r);
                    i += 1;
                }
            }
        }
        self.frame_index = index;
        self.features = features;
        self.interactive = interactive;
        self.trunk = trunk;
        state_start.record(self.r.g, self.options.profile, &self.last_profile.state);
        return output;
    }

    fn releaseFeatures(self: *Session) void {
        if (self.features) |features| for (features) |f| self.r.release(f.tensor);
        if (self.model.config.version == .sam31) if (self.interactive) |features|
            for (features) |f| self.r.release(f.tensor);
        if (self.trunk) |trunk| self.r.release(trunk.tensor);
    }

    fn releaseLast(self: *Session, bucket: *Bucket) void {
        if (bucket.last) |last| {
            var storage: [3]ops.Tensor = undefined;
            for (last.retained(&storage)) |tensor| self.r.release(tensor);
        }
        bucket.last = null;
    }

    pub fn removeObject(self: *Session, id: slots.ObjectId) !void {
        try self.model.mutex.lock(self.model.io);
        defer self.model.mutex.unlock(self.model.io);
        const slot = self.objects.get(id) orelse return error.UnknownObject;
        try self.recondition(id, slot, null);
    }

    pub fn prompt(self: *Session, id: slots.ObjectId, value: prompts.Prompt) !void {
        try self.model.mutex.lock(self.model.io);
        defer self.model.mutex.unlock(self.model.io);
        const size = self.size orelse return error.SessionNotSeeded;
        try value.validate(size);
        const slot = self.objects.get(id) orelse free: {
            for (self.objects.ids, 0..) |maybe, at| if (maybe == null) break :free slots.Slot{
                .bucket = at / self.model.config.bucket_size,
                .slot = at % self.model.config.bucket_size,
                .epoch = self.objects.epochs[at / self.model.config.bucket_size],
            };
            return error.TooManyObjects;
        };
        try self.recondition(id, slot, value);
    }

    fn recondition(self: *Session, id: slots.ObjectId, slot: slots.Slot, value: ?prompts.Prompt) !void {
        const index = self.frame_index orelse return error.SessionNotSeeded;
        if (self.objects.epochs[slot.bucket] == std.math.maxInt(u64)) return error.EpochOverflow;
        var arena = std.heap.ArenaAllocator.init(self.allocator);
        defer arena.deinit();
        self.r.a = arena.allocator();
        defer self.r.a = self.allocator;
        const start = self.r.mark();
        errdefer self.r.trim(start, &.{});
        const bucket = &self.buckets[slot.bucket];
        try bucket.history.ensureUnusedCapacity(self.allocator, 1);
        const cached: ?[3]ops.Feature = if (value != null and self.interactive == null)
            try vision.pyramid(&self.r, self.trunk.?, self.model.config.pyramidPrefix(true))
        else
            null;
        var items: [16]?tracker.Result = @splat(null);
        const size = self.model.config.bucket_size;
        var bits = self.active(slot.bucket);
        for (0..size) |s| {
            if (s == slot.slot or bits & (@as(u16, 1) << @intCast(s)) == 0) continue;
            const last = bucket.last.?;
            items[s] = last;
            items[s].?.masks = ops.feature(try self.r.columns(last.masks.tensor, 288 * 288, size, s, 1), 288, 288, 1);
            if (last.input_masks) |input| items[s].?.input_masks = ops.feature(try self.r.columns(input.tensor, input.height * input.width, size, s, 1), input.height, input.width, 1);
            items[s].?.pointer = ops.view(last.pointer, s * 256, 256);
            items[s].?.scores[0] = last.scores[s];
            items[s].?.ious[0] = last.ious[s];
        }
        if (value) |p| {
            items[slot.slot] = try tracker.seed(&self.r, self.interactive orelse cached.?, p, self.size.?);
            bits |= @as(u16, 1) << @intCast(slot.slot);
        } else bits &= ~(@as(u16, 1) << @intCast(slot.slot));
        var result: ?tracker.Result = null;
        var past: ?memory.Memory = null;
        const slot_bit = @as(u16, 1) << @intCast(slot.slot);
        const conditioning = (bucket.conditioning & bits & ~slot_bit) | if (value != null) slot_bit else @as(u16, 0);
        if (bits != 0) {
            result = try combine(&self.r, items[0..size]);
            const from_points = if (value) |p| p.points.len != 0 or p.boxes.len != 0 else false;
            past = try memory.encode(&self.r, self.features.?[2].tensor, result.?, bits, conditioning, from_points, index);
        }
        var keep: std.ArrayList(ops.Tensor) = .empty;
        if (cached) |levels| for (levels) |f| try keep.append(self.r.a, f.tensor);
        if (result) |out| {
            var storage: [3]ops.Tensor = undefined;
            try keep.appendSlice(self.r.a, out.retained(&storage));
        }
        if (past) |m| {
            var storage: [11]ops.Tensor = undefined;
            try keep.appendSlice(self.r.a, m.retained(&storage));
        }
        self.r.trim(start, keep.items);
        if (cached) |levels| {
            self.interactive = levels;
            self.r.release(self.trunk.?.tensor);
            self.trunk = null;
        }
        if (value != null) _ = try self.objects.add(id) else _ = try self.objects.remove(id);
        for (bucket.history.items) |m| m.release(&self.r);
        bucket.history.clearRetainingCapacity();
        self.releaseLast(bucket);
        bucket.last = result;
        bucket.conditioning = conditioning;
        if (past) |m| bucket.history.appendAssumeCapacity(m);
    }
};

fn combine(r: *ops.Runtime, items: []const ?tracker.Result) !tracker.Result {
    const count = r.config.bucket_size;
    const empty = try r.alloc(288 * 288);
    @memset(empty.slice(f32, empty.count), -1024);
    const zero = try r.alloc(256);
    @memset(zero.slice(f32, zero.count), 0);
    var masks: [16]ops.Tensor = @splat(empty);
    var pointers: [16]ops.Tensor = @splat(zero);
    var out: tracker.Result = .{ .masks = undefined, .pointer = undefined, .count = count, .scores = @splat(-10), .ious = @splat(0) };
    for (items, 0..) |maybe, slot| if (maybe) |item| {
        masks[slot] = item.masks.tensor;
        pointers[slot] = item.pointer;
        out.scores[slot] = item.scores[0];
        out.ious[slot] = item.ious[0];
    };
    const channels = try r.concat(masks[0..count]);
    out.masks = ops.feature(try r.transpose(channels, count, 288 * 288), 288, 288, count);
    out.pointer = try r.concat(pointers[0..count]);
    const input_size: ?[2]usize = for (items) |maybe| {
        if (maybe) |item| if (item.input_masks) |input| break .{ input.height, input.width };
    } else null;
    if (input_size) |size| {
        const empty_high = try r.alloc(size[0] * size[1]);
        @memset(empty_high.slice(f32, empty_high.count), -1024);
        var high: [16]ops.Tensor = @splat(empty_high);
        for (items, 0..) |maybe, slot| if (maybe) |item| {
            const input = item.input_masks orelse item.masks;
            high[slot] = (try r.resize(input, size[0], size[1], true)).tensor;
        };
        out.input_masks = ops.feature(try r.transpose(try r.concat(high[0..count]), count, size[0] * size[1]), size[0], size[1], count);
    }
    return out;
}

fn validateNext(previous: ?usize, index: usize, options: Options) !void {
    const before = previous orelse return error.SessionNotSeeded;
    if (index >= options.frame_count) return error.FrameOutOfRange;
    const valid = switch (options.direction) {
        .forward => before < index and index - before == 1,
        .reverse => index < before and before - index == 1,
    };
    if (!valid) return error.NonConsecutiveFrame;
}

test "video propagation accepts nonzero anchors and enforces the requested direction" {
    try validateNext(7, 8, .{ .frame_count = 10 });
    try validateNext(7, 6, .{ .frame_count = 10, .direction = .reverse });
    try std.testing.expectError(error.SessionNotSeeded, validateNext(null, 1, .{ .frame_count = 10 }));
    try std.testing.expectError(error.NonConsecutiveFrame, validateNext(7, 6, .{ .frame_count = 10 }));
    try std.testing.expectError(error.NonConsecutiveFrame, validateNext(7, 9, .{ .frame_count = 10 }));
    try std.testing.expectError(error.FrameOutOfRange, validateNext(9, 10, .{ .frame_count = 10 }));
}
