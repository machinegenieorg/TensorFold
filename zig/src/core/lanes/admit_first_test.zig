//! First-token handoff precedes drafting without changing tokens or transferring stream ownership.
const std = @import("std");
const be = @import("backend.zig");
const sm = @import("stream.zig");
const fake = @import("fake.zig");
const Engine = @import("engine.zig").Engine;
const Config = @import("config.zig").Config;
const gpa = std.testing.allocator;

const Probe = struct {
    target: fake.Fake = .{ .gpa = gpa },
    vtable: be.Backend.VTable = undefined,
    calls: usize = 0,
    handoffs: usize = 0,
    releases: usize = 0,
    fail: bool = false,
    early: bool = true,
    cancel: bool = false,
    last_first: ?u32 = null,
    list: []const *sm.Stream = &.{},

    fn backend(x: *Probe) be.Backend {
        var b = x.target.backend();
        x.vtable = b.vtable.*;
        x.vtable.draft = draft;
        x.vtable.release = release;
        b.vtable = &x.vtable;
        b.first_before_draft = x.early;
        return b;
    }

    fn from(ptr: *anyopaque) *Probe {
        const target: *fake.Fake = @ptrCast(@alignCast(ptr));
        return @fieldParentPtr("target", target);
    }

    fn draft(ptr: *anyopaque, requests: []const be.DraftRequest) !void {
        const x = from(ptr);
        x.calls += 1;
        for (requests) |r| {
            if (x.early and r.first != null) {
                try std.testing.expectEqual(@as(usize, 1), r.stream.emitted().len);
                try std.testing.expect(!r.stream.finished);
                try std.testing.expectEqual(r.stream.pending.?, r.first.?.value);
                try std.testing.expect(x.handoffs > 0 or x.list.len == 0);
                x.last_first = r.first.?.value;
            }
        }
        if (x.fail) return error.InitialDraftFailed;
        return x.target.backend().draft(requests);
    }

    fn release(ptr: *anyopaque, s: *sm.Stream) void {
        const x = from(ptr);
        x.releases += 1;
        x.target.backend().release(s);
    }

    fn handoff(ptr: *anyopaque, index: usize) void {
        const x: *Probe = @ptrCast(@alignCast(ptr));
        std.debug.assert(x.calls == 0 and x.list[index].emitted().len == 1);
        x.handoffs += 1;
    }

    fn cancelled(ptr: *anyopaque) bool {
        const x: *Probe = @ptrCast(@alignCast(ptr));
        return x.cancel;
    }

    fn cancelHandoff(ptr: *anyopaque, index: usize) void {
        handoff(ptr, index);
        const x: *Probe = @ptrCast(@alignCast(ptr));
        x.cancel = true;
    }
};

fn config() !Config {
    return Config.init(gpa, .{ .exact_width = 8, .mtp = true, .speculate = true, .speculate_early = false, .drafts = 4, .window_costs = &.{ .{ .width = 1, .ms = 5 }, .{ .width = 8, .ms = 10 } }, .mtp_step_ms = 0.5, .hidden_rows = true, .draft_streams = true }, 8, 4);
}

test "first handoff is committed and terminal streams survive another stream's initial draft failure" {
    var cfg = try config();
    defer cfg.deinit(gpa);
    var x: Probe = .{ .fail = true };
    defer x.target.deinit();
    var clock: fake.FixedClock = .{};
    var e = Engine.init(gpa, &cfg, x.backend(), clock.clock());
    defer e.deinit();
    var terminal = try sm.Stream.init(gpa, .{ .id = "terminal", .prompt = &.{ 2, 7 }, .max_new = 1 });
    defer terminal.deinit(gpa);
    var live = try sm.Stream.init(gpa, .{ .id = "live", .prompt = &.{ 1, 8 }, .max_new = 20 });
    defer live.deinit(gpa);
    x.list = &.{ &terminal, &live };
    var errs: [2]?anyerror = undefined;
    try e.addStreamsCommitted(x.list, &errs, .{ .ptr = &x, .call = Probe.handoff });
    try std.testing.expectEqual(@as(usize, 2), x.handoffs);
    try std.testing.expectEqual(@as(?anyerror, null), errs[0]);
    try std.testing.expectEqual(error.InitialDraftFailed, errs[1].?);
    try std.testing.expect(terminal.finished);
    try std.testing.expectEqual(@as(usize, 1), live.emitted().len);
    try std.testing.expectEqual(@as(usize, 1), x.releases);
    e.discard(&live);
    try std.testing.expectEqual(@as(usize, 2), x.releases);
    try std.testing.expectEqual(@as(usize, 0), x.target.lanes.count());
}

test "cancellation after first handoff releases once and never prepares proposals" {
    var cfg = try config();
    defer cfg.deinit(gpa);
    var x: Probe = .{};
    defer x.target.deinit();
    var clock: fake.FixedClock = .{};
    var e = Engine.init(gpa, &cfg, x.backend(), clock.clock());
    defer e.deinit();
    var s = try sm.Stream.init(gpa, .{ .id = "cancel", .prompt = &.{2}, .max_new = 20, .cancel_check = .{ .ptr = &x, .check = Probe.cancelled } });
    defer s.deinit(gpa);
    x.list = &.{&s};
    var errs: [1]?anyerror = undefined;
    try e.addStreamsCommitted(x.list, &errs, .{ .ptr = &x, .call = Probe.cancelHandoff });
    try std.testing.expectEqual(error.Cancelled, errs[0].?);
    try std.testing.expectEqual(@as(usize, 0), x.calls);
    try std.testing.expectEqual(@as(usize, 1), x.releases);
    try std.testing.expectEqual(@as(usize, 0), e.activeCount());
}

test "forced first and logprobs match default admission and subsequent exact tokens" {
    var cfg = try config();
    defer cfg.deinit(gpa);
    var a: Probe = .{ .early = false };
    defer a.target.deinit();
    var b: Probe = .{};
    defer b.target.deinit();
    var clock: fake.FixedClock = .{};
    var old = Engine.init(gpa, &cfg, a.backend(), clock.clock());
    defer old.deinit();
    var new = Engine.init(gpa, &cfg, b.backend(), clock.clock());
    defer new.deinit();
    const spec: sm.Spec = .{ .id = "forced", .prompt = &.{ 2, 7, 1 }, .max_new = 20, .logprobs = 3 };
    var left = try sm.Stream.init(gpa, spec);
    defer left.deinit(gpa);
    var right = try sm.Stream.init(gpa, spec);
    defer right.deinit(gpa);
    try left.force.append(gpa, 45);
    try right.force.append(gpa, 45);
    a.list = &.{&left};
    b.list = &.{&right};
    var errs: [1]?anyerror = undefined;
    try old.addStreamsCommitted(a.list, &errs, .{ .ptr = &a, .call = Probe.handoff });
    try new.addStreamsCommitted(b.list, &errs, .{ .ptr = &b, .call = Probe.handoff });
    try std.testing.expectEqual(@as(usize, 0), a.handoffs);
    try std.testing.expectEqual(@as(usize, 1), b.handoffs);
    try std.testing.expectEqual(@as(u32, 45), b.last_first.?);
    while (old.activeCount() > 0) try old.step();
    while (new.activeCount() > 0) try new.step();
    try std.testing.expectEqualSlices(u32, left.emitted(), right.emitted());
    try std.testing.expectEqual(left.rows.items.len, right.rows.items.len);
    for (left.rows.items, right.rows.items) |l, r| {
        try std.testing.expectEqual(l.token, r.token);
        try std.testing.expectEqual(@as(u32, @bitCast(l.logprob)), @as(u32, @bitCast(r.logprob)));
        try std.testing.expectEqual(l.count, r.count);
        try std.testing.expectEqualSlices(u32, l.ids[0..l.count], r.ids[0..r.count]);
        for (l.logprobs[0..l.count], r.logprobs[0..r.count]) |lp, rp|
            try std.testing.expectEqual(@as(u32, @bitCast(lp)), @as(u32, @bitCast(rp)));
    }
}
