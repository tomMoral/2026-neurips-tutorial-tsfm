"""From one bike counter to EEG: the same tasks on both kinds of series.

Render from slides/anim/:
    manim -qh tasks.py BikesToEEG
"""
import json
from pathlib import Path

import manimpango
import numpy as np
from manim import (
    DOWN, LEFT, RIGHT, UP, UL, Arrow, Brace, Circle, Create, DashedVMobject, Dot, Ellipse,
    FadeIn, FadeOut, FadeTransform, GrowArrow, LaggedStart, Line, Polygon, Rectangle, Restore,
    RoundedRectangle, Scene, Text, Transform, ValueTracker, VGroup, VMobject, config, linear,
)

INK, MUTED, PAPER = "#24221d", "#716b5e", "#f5f0e4"
PRIMARY, ACCENT, TINT, RULE = "#1f6f50", "#a8680f", "#e8e0cb", "#c4b89a"
FONT = "Source Sans Pro"
for font in (Path(__file__).parent / "fonts").glob("*.otf"):
    manimpango.register_font(str(font))
config.background_color = PAPER

DATA = json.loads((Path(__file__).parent / "bikes.json").read_text())
# Plot area on the left, with room for covariate tracks above; data or task list on the right
PLOT_LEFT, PLOT_WIDTH, PLOT_BOTTOM, PLOT_HEIGHT = -6.6, 8.4, -2.7, 3.2
WEEK_WIDTH = PLOT_WIDTH * 168 / 216

TASKS = ["Forecasting", "Classification", "Anomaly detection", "Event detection", "Imputation"]
BIKE_TASKS = [
    "the counts of the next days",
    "a commuter or a leisure street",
    "a broken sensor",
    "a strike day",
    "a missing day",
]
EEG_TASKS = [
    "all channels, a seizure before it starts",
    "the response to a stimulus, sleep stages",
    "a drifting channel, its neighbours are fine",
    "blinks, and where on the scalp",
    "a bad electrode, from its neighbours",
]
CHANNELS = ["Fp1", "F3", "C3", "P3", "O1", "Oz"]
# Where the channels sit on a unit head, the nose up
CHANNEL_XY = [(-.3, .9), (-.42, .5), (-.55, 0), (-.42, -.5), (-.3, -.9), (0, -.97)]
STIMULI = [1.0, 3.3, 6.2, 8.4]
# The Seine through Paris, from Ivry to Issy, as (lat, lon) of a few bridges
SEINE = [
    (48.818, 2.413), (48.838, 2.376), (48.845, 2.366), (48.851, 2.359), (48.857, 2.341), (48.860, 2.329),
    (48.864, 2.320), (48.864, 2.314), (48.863, 2.301), (48.860, 2.293), (48.856, 2.288), (48.850, 2.280),
    (48.847, 2.275), (48.840, 2.270), (48.834, 2.260),
]
WATER, RAIN = "#b7c9d1", "#6f93a4"
DATA_ROWS = ["Series", "Covariates"]
# The closing table: one short example per cell
TABLE = [
    ("Series", "128 counters", "64 electrodes"),
    ("Covariates", "weather, calendar", "the paradigm"),
    ("Forecasting", "the next days", "a seizure"),
    ("Classification", "a commuter street", "a target stimulus"),
    ("Anomaly detection", "a broken sensor", "a drifting channel"),
    ("Event detection", "a strike", "a blink"),
    ("Imputation", "a missing day", "a bad electrode"),
]
COUNTER_DATA = ["one counter, hourly", "weather, holidays, strikes"]
CITY_DATA = ["128 counters across Paris", "weather, holidays, strikes"]
EEG_DATA = ["64 electrodes, hundreds of Hz", "the paradigm: stimuli, eyes closed"]


def text(s, size=22, color=INK, weight="NORMAL"):
    return Text(s, font=FONT, font_size=size, color=color, weight=weight)


def title(s):
    head = text(s, 30, PRIMARY, "SEMIBOLD").to_corner(UL, buff=.45)
    rule = Line([-6.65, 0, 0], [6.65, 0, 0], color=RULE, stroke_width=1.5)
    return VGroup(head, rule.next_to(head, DOWN, buff=.15).set_x(0))


def polyline(values, left, width, bottom, height, vmax, color=PRIMARY, stroke=2.5):
    values = np.asarray(values, dtype=float)
    x = left + width * np.arange(len(values)) / (len(values) - 1)
    y = bottom + height * values / vmax
    line = VMobject(stroke_color=color, stroke_width=stroke)
    return line.set_points_as_corners(np.column_stack([x, y, 0 * x]))


def listing(names, examples, color=PRIMARY):
    rows = VGroup()
    for name, example in zip(names, examples):
        rows.add(VGroup(text(name, 22, color, "SEMIBOLD"), text(example, 18, MUTED)).arrange(DOWN, aligned_edge=LEFT, buff=.08))
    rows.arrange(DOWN, aligned_edge=LEFT, buff=.28).move_to([4.75, -.35, 0]).align_to([2.75, 0, 0], LEFT)
    return rows


def data_card(examples):
    """The data description, boxed and in ochre so that it does not read as a task."""
    rows = listing(DATA_ROWS, examples, ACCENT)
    kicker = text("DATA", 14, MUTED, "SEMIBOLD").next_to(rows, UP, buff=.25).align_to(rows, LEFT)
    box = RoundedRectangle(width=4.1, height=rows.height + kicker.height + .75, corner_radius=.12,
                           fill_color=TINT, fill_opacity=1, stroke_width=0)
    box.move_to(VGroup(kicker, rows)).align_to(rows, LEFT).shift(LEFT * .3)
    return VGroup(box, kicker, rows)


def swap(rows, examples):
    """Transforms that replace the examples of a listing, row by row."""
    return [Transform(row[1], text(e, 18, MUTED).move_to(row[1], aligned_edge=LEFT)) for row, e in zip(rows, examples)]


def hours_x(h):
    return PLOT_LEFT + WEEK_WIDTH * h / 167


def electrodes(center, radius):
    """A schematic 64-electrode montage: rings from the vertex to the rim."""
    points = [np.array(center)]
    for r, n in [(.3, 8), (.55, 16), (.8, 19), (.98, 20)]:
        angles = np.pi / 2 + 2 * np.pi * (np.arange(n) + .5 * (n % 2)) / n
        points += [center + radius * r * np.array([np.cos(a), np.sin(a), 0]) for a in angles]
    return np.array(points)


def eeg(seconds=12, fs=60, seed=0):
    """Synthetic EEG: eyes open then closed with a posterior alpha, evoked responses, a frontal blink."""
    rng = np.random.default_rng(seed)
    t = np.arange(seconds * fs) / fs
    closed = t >= 5
    lag = np.clip(t[None] - np.array(STIMULI)[:, None], 0, None)
    evoked = (np.sin(2 * np.pi * 3 * lag) * np.exp(-lag / .15) * (lag > 0)).sum(0)
    signals = []
    for c in range(len(CHANNELS)):
        # One extra second on each side keeps the moving average away from the edges
        noise = np.cumsum(rng.normal(size=t.size + 2 * fs))
        noise = ((noise - np.convolve(noise, np.ones(fs) / fs, "same")) * .25)[fs:-fs]
        posterior = c / (len(CHANNELS) - 1)
        alpha = (.3 + 1.6 * posterior * closed) * np.sin(2 * np.pi * 10 * t + rng.uniform(0, 6))
        signals.append(noise + alpha + [.8, 1.5, 2.5, 2.5, 1.5, .8][c] * evoked)
    blink = 4 * np.exp(-.5 * ((t - 2) / .12) ** 2)
    signals[0] += blink
    signals[1] += .5 * blink
    signals[2] += .15 * blink
    return t, np.array(signals)


class BikesToEEG(Scene):
    def construct(self):
        week = np.array(DATA["week"])
        vmax = 1.15 * week.max()

        def plot(values, left=PLOT_LEFT, width=WEEK_WIDTH, color=PRIMARY):
            return polyline(values, left, width, PLOT_BOTTOM, PLOT_HEIGHT, vmax, color)

        def band(h0, h1, color, opacity, height=PLOT_HEIGHT + .1, y=PLOT_BOTTOM + PLOT_HEIGHT / 2):
            box = Rectangle(width=hours_x(h1) - hours_x(h0), height=height, fill_color=color, fill_opacity=opacity, stroke_width=0)
            return box.move_to([(hours_x(h0) + hours_x(h1)) / 2, y, 0])

        # 1. The data: one counter, and the covariates that come with it
        head = title("One bike counter, and what comes with it")
        card = data_card(COUNTER_DATA)
        line = plot(week)
        days = VGroup(*[text(d, 16, MUTED).move_to([hours_x(24 * i + 12), PLOT_BOTTOM - .3, 0])
                        for i, d in enumerate(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])])
        caption = text("Boulevard de Sébastopol, hourly, 15-21 September 2025", 16, MUTED).next_to(days, DOWN, buff=.15).align_to(line, LEFT)
        self.play(FadeIn(head), FadeIn(card[:2]), FadeIn(card[2][0]))
        self.play(Create(line), FadeIn(days), FadeIn(caption), run_time=2.5, rate_func=linear)

        # Covariate tracks above the series, known over the forecast window too: hourly weather, and a calendar
        rain_base, cal_y, known = 1.6, .95, len(DATA["wind"]) - 1
        bars = VGroup(*[Rectangle(width=WEEK_WIDTH / 168, height=.6 * mm / 3.5, fill_color=RAIN, fill_opacity=1, stroke_width=0)
                        .move_to([hours_x(h + .5), rain_base, 0], aligned_edge=DOWN)
                        for h, mm in enumerate(DATA["rain"]) if mm > 0])
        rain_axis = Line([PLOT_LEFT, rain_base, 0], [hours_x(known), rain_base, 0], color=RULE, stroke_width=1.5)
        wind = polyline(DATA["wind"], PLOT_LEFT, hours_x(known) - PLOT_LEFT, rain_base, .6, 30, MUTED, 1.5)
        calendar = VGroup(
            band(0, known, TINT, 1, .3, cal_y),
            band(120, 168, MUTED, .35, .3, cal_y),
            band(72, 96, ACCENT, .6, .3, cal_y),
        )
        calendar.add(text("weekend", 14, INK).move_to(calendar[1]), text("strike", 14, INK, "SEMIBOLD").move_to(calendar[2]))
        weather = VGroup(text("weather:", 14, MUTED), text("rain (mm/h)", 14, RAIN, "SEMIBOLD"), text("wind (km/h)", 14, MUTED, "SEMIBOLD"))
        labels = VGroup(weather.arrange(RIGHT, buff=.12).next_to(rain_axis, UP, buff=.7).align_to(rain_axis, LEFT),
                        text("calendar", 14, MUTED).next_to(calendar[0], UP, buff=.06).align_to(calendar[0], LEFT))
        self.play(FadeIn(card[2][1]), FadeIn(rain_axis), FadeIn(labels), LaggedStart(*[FadeIn(b) for b in bars], lag_ratio=.1),
                  Create(wind))
        self.play(FadeIn(calendar))
        self.wait(2)

        # 2. The univariate tasks, with the covariates still in view
        head2 = title("One bike counter, many questions")
        rows = listing(TASKS, BIKE_TASKS)
        for row in rows:
            row.set_opacity(.25)
        self.play(FadeTransform(head, head2), FadeOut(card), FadeIn(rows))

        # Forecasting: the next two days, with a band
        future = np.array(DATA["future"][:48])
        f_left = PLOT_LEFT + WEEK_WIDTH
        f_width = PLOT_WIDTH * 48 / 216
        shade = Rectangle(width=f_width, height=PLOT_HEIGHT + .2, fill_color=TINT, fill_opacity=1, stroke_width=0)
        shade.move_to([f_left + f_width / 2, PLOT_BOTTOM + PLOT_HEIGHT / 2, 0])
        upper, lower = plot(1.25 * future, f_left, f_width), plot(.75 * future, f_left, f_width)
        spread = Polygon(*upper.get_anchors(), *lower.get_anchors()[::-1], fill_color=ACCENT, fill_opacity=.25, stroke_width=0)
        median = DashedVMobject(plot(future, f_left, f_width, ACCENT), num_dashes=60)
        self.play(rows[0].animate.set_opacity(1), FadeIn(shade))
        self.play(FadeIn(spread), Create(median), run_time=1.5)
        self.wait(.8)
        self.play(FadeOut(shade), FadeOut(spread), FadeOut(median))

        # Classification: one label for the whole window, inside its frame
        frame = Rectangle(width=WEEK_WIDTH + .2, height=PLOT_HEIGHT + .2, color=ACCENT, stroke_width=3)
        frame.move_to([PLOT_LEFT + WEEK_WIDTH / 2, PLOT_BOTTOM + PLOT_HEIGHT / 2, 0])
        label = text("commuter street", 20, ACCENT, "SEMIBOLD").move_to([hours_x(144), PLOT_BOTTOM + PLOT_HEIGHT - .2, 0])
        self.play(rows[1].animate.set_opacity(1), Create(frame), FadeIn(label))
        self.wait(.8)
        self.play(FadeOut(frame), FadeOut(label))

        # Anomaly detection: a sensor stuck at zero on Wednesday morning
        start, stop = 2 * 24 + 6, 2 * 24 + 11
        broken = week.copy()
        broken[start:stop] = 0
        ring = Circle(radius=.45, color=ACCENT, stroke_width=3).move_to([hours_x((start + stop) / 2), PLOT_BOTTOM + .2, 0])
        self.play(rows[2].animate.set_opacity(1), line.animate.become(plot(broken)))
        self.play(Create(ring))
        self.wait(.6)
        self.play(FadeOut(ring), line.animate.become(plot(week)))

        # Event detection: the strike, a whole day with an onset and an offset
        event = VGroup(band(72, 96, ACCENT, .15),
                       Line([hours_x(72), PLOT_BOTTOM + PLOT_HEIGHT + .1, 0], [hours_x(96), PLOT_BOTTOM + PLOT_HEIGHT + .1, 0],
                            color=ACCENT, stroke_width=6))
        self.play(rows[3].animate.set_opacity(1), FadeIn(event))
        self.wait(.8)
        self.play(FadeOut(event))

        # Imputation: hide Friday, then fill it back
        gap = band(96, 120, PAPER, 1)
        friday = plot(week[96:121], hours_x(96), WEEK_WIDTH * 24 / 167, ACCENT)
        self.play(rows[4].animate.set_opacity(1), FadeIn(gap))
        self.wait(.4)
        self.play(Create(DashedVMobject(friday, num_dashes=40)), run_time=1.2)
        self.wait(.8)
        self.play(*[FadeOut(m) for m in self.mobjects if m is not head2])

        # 3. Many counters: one day of Paris, scaled per counter
        head3 = title("Many counters, one city")
        card = data_card(CITY_DATA)
        center, radius = np.array([-2.4, -.35, 0]), 2.7
        sites = DATA["sites"]
        lat, lon = np.array(sites["lat"]), np.array(sites["lon"])
        river_lat, river_lon = np.array(SEINE).T

        def project(la, lo):
            return np.column_stack([(lo - lon.mean()) * np.cos(np.radians(lat.mean())), la - lat.mean()])

        scale = radius / np.abs(project(lat, lon)).max()
        positions = [center + np.array([x, y, 0]) for x, y in scale * project(lat, lon)]
        seine = VMobject(stroke_color=WATER, stroke_width=9).set_points_smoothly(
            [center + np.array([x, y, 0]) for x, y in scale * project(river_lat, river_lon)])
        seine_label = text("Seine", 16, WATER, "SEMIBOLD").next_to(seine.get_end(), DOWN, buff=.1)
        profile = np.array(sites["profile"])
        hour = ValueTracker(0)

        def level(i):
            return np.interp(hour.get_value(), np.arange(24), profile[i])

        dots = VGroup(*[Dot(p, radius=.07, color=PRIMARY) for p in positions])
        for i, dot in enumerate(dots):
            dot.add_updater(lambda m, i=i: m.set_fill(opacity=.25 + .75 * level(i)).scale_to_fit_width(.1 + .22 * level(i)))
        clock = text("00:00", 26, ACCENT, "SEMIBOLD").move_to([center[0] + radius + .2, center[1] + radius - .1, 0])
        clock.add_updater(lambda m: m.become(
            text(f"{int(hour.get_value()):02d}:00", 26, ACCENT, "SEMIBOLD").move_to(clock.get_center())))
        map_caption = text("one dot per counter, sized by its traffic at that hour", 16, MUTED).move_to([center[0], -3.55, 0])
        self.play(FadeTransform(head2, head3), FadeIn(card))
        self.play(Create(seine), FadeIn(seine_label), run_time=1.2)
        self.play(LaggedStart(*[FadeIn(d) for d in dots], lag_ratio=.01), FadeIn(clock), FadeIn(map_caption))
        self.play(hour.animate.set_value(23), run_time=6, rate_func=linear)
        for dot in dots:
            dot.clear_updaters()
        clock.clear_updaters()
        self.wait(.5)

        # 4. Counters become electrodes, the Seine curls into the scalp
        head4 = title("Many electrodes, one brain")
        targets = electrodes(center, radius)
        free = list(range(len(dots)))
        moves, kept = [], []
        for target in targets:
            j = min(free, key=lambda k: np.linalg.norm(positions[k] - target))
            free.remove(j)
            kept.append(j)
            moves.append(dots[j].animate.move_to(target).set_fill(PRIMARY, opacity=1).scale_to_fit_width(.13))
        scalp = Circle(radius=radius + .3, color=INK, stroke_width=2).move_to(center)
        nose = VMobject(stroke_color=INK, stroke_width=2).set_points_as_corners(
            [center + [-.3, radius + .27, 0], center + [0, radius + .65, 0], center + [.3, radius + .27, 0]])
        ears = VGroup(*[Circle(radius=.28, color=INK, stroke_width=2).stretch(1.6, 1).move_to(center + [s * (radius + .4), 0, 0])
                        for s in (-1, 1)])
        self.play(FadeTransform(head3, head4), FadeOut(clock), FadeOut(map_caption), FadeOut(seine_label),
                  *[FadeOut(dots[k]) for k in free])
        self.play(*moves, Transform(seine, scalp), run_time=2.5)
        # The EEG covariates come later, with their track, as for the single counter
        self.play(Create(nose), Create(ears), *swap(card[2][:1], EEG_DATA[:1]), FadeOut(card[2][1]), run_time=2)
        card[2][1][1].become(text(EEG_DATA[1], 18, MUTED).move_to(card[2][1][1], aligned_edge=LEFT))

        # The head goes to a corner, and the channels draw under their paradigm track
        electrode_dots = VGroup(*[dots[k] for k in kept])
        brain = VGroup(seine, nose, ears, electrode_dots)
        self.play(brain.animate.scale(.25).move_to([PLOT_LEFT + PLOT_WIDTH - .9, 2.05, 0]))
        head_c, head_r = electrode_dots[0].get_center(), .25 * radius

        def on_head(u):
            return head_c + head_r * np.array([*u, 0])

        def near(point, spread):
            return [np.exp(-np.sum((d.get_center() - point) ** 2) / (2 * (spread * head_r) ** 2)) for d in electrode_dots]

        t, signals = eeg()
        fs, seen = 60, 10
        bottom, step, left, width = -2.8, .52, PLOT_LEFT + .7, PLOT_WIDTH - .7
        seen_width = width * seen / t[-1]

        def tx(sec):
            return left + seen_width * sec / seen

        def trace(c, values=None, start=0, color=PRIMARY):
            values = signals[c, :seen * fs] if values is None else values
            return polyline(values, tx(start), seen_width * (len(values) - 1) / (seen * fs - 1),
                            bottom + step * (len(CHANNELS) - 1 - c), step / 8, 1, color, 1.5)

        traces = VGroup(*[trace(c) for c in range(len(CHANNELS))])
        names = VGroup(*[text(n, 14, MUTED).next_to(tr, LEFT, buff=.1) for n, tr in zip(CHANNELS, traces)])
        track_y = .55
        paradigm = VGroup(
            Rectangle(width=tx(5) - tx(0), height=.3, fill_color=TINT, fill_opacity=1, stroke_width=0),
            Rectangle(width=left + width - tx(5), height=.3, fill_color=ACCENT, fill_opacity=.3, stroke_width=0),
        ).arrange(RIGHT, buff=0).move_to([left, track_y, 0], aligned_edge=LEFT)
        paradigm.add(text("eyes open", 14, INK).move_to([tx(2.15), track_y, 0]),
                     text("eyes closed", 14, INK).move_to([tx(7.3), track_y, 0]),
                     *[Line([tx(s), track_y - .15, 0], [tx(s), track_y + .2, 0], color=INK, stroke_width=3) for s in STIMULI],
                     text("paradigm: stimuli and conditions", 14, MUTED).next_to(paradigm, UP, buff=.06).align_to(paradigm, LEFT))
        eeg_caption = text("synthetic EEG, for illustration", 14, MUTED).next_to(traces, DOWN, buff=.2).align_to(traces, LEFT)
        self.play(FadeIn(names), FadeIn(eeg_caption))
        self.play(*[Create(tr) for tr in traces], run_time=3.5, rate_func=linear)
        self.play(FadeIn(paradigm), FadeIn(card[2][1]))
        self.wait(1.5)

        # 5. The same questions, each using the other channels
        eeg_rows = listing(TASKS, EEG_TASKS)
        for row in eeg_rows:
            row.set_opacity(.25)
        self.play(FadeTransform(card, eeg_rows))

        # Forecasting: every channel at once, with the paradigm known ahead
        ahead = VGroup(*[DashedVMobject(trace(c, signals[c, seen * fs - 1:], seen, ACCENT), num_dashes=25)
                         for c in range(len(CHANNELS))])
        shade = Rectangle(width=left + width - tx(seen), height=step * len(CHANNELS), fill_color=TINT, fill_opacity=1, stroke_width=0)
        shade.move_to([tx(seen), bottom + step * (len(CHANNELS) - 1) / 2, 0], aligned_edge=LEFT)
        self.play(eeg_rows[0].animate.set_opacity(1), FadeIn(shade))
        self.play(*[Create(a) for a in ahead], run_time=1.5)
        self.wait(.8)
        self.play(FadeOut(shade), FadeOut(ahead))

        # Classification: one label for one trial, read from all the channels
        trial = Brace(VGroup(Dot([tx(6.2), bottom - .1, 0]), Dot([tx(7.4), bottom - .1, 0])), DOWN, color=ACCENT)
        verdict = text("target stimulus", 18, ACCENT, "SEMIBOLD").next_to(trial, DOWN, buff=.05)
        epoch = Rectangle(width=tx(7.4) - tx(6.2), height=step * len(CHANNELS), fill_color=ACCENT, fill_opacity=.12, stroke_width=0)
        epoch.move_to([tx(6.2), bottom + step * (len(CHANNELS) - 1) / 2, 0], aligned_edge=LEFT)
        self.play(eeg_rows[1].animate.set_opacity(1), FadeIn(epoch), FadeIn(trial), FadeIn(verdict))
        self.wait(1)
        self.play(FadeOut(epoch), FadeOut(trial), FadeOut(verdict))

        # Anomaly detection: C3 drifts away, while its neighbours stay put
        drift = signals[2, :seen * fs] + np.clip(t[:seen * fs] - 3, 0, None) * 1.2
        ring = Ellipse(width=tx(seen) - tx(3) + .4, height=step * 1.3, color=ACCENT, stroke_width=3)
        ring.move_to([(tx(3) + tx(seen)) / 2, bottom + step * 3 + .2, 0])
        self.play(eeg_rows[2].animate.set_opacity(1), traces[2].animate.become(trace(2, drift)))
        self.play(Create(ring))
        self.wait(.8)
        self.play(FadeOut(ring), traces[2].animate.become(trace(2)))

        # Event detection: the blink, and the electrodes closest to the eyes
        blink = Ellipse(width=.7, height=step * 2.3, color=ACCENT, stroke_width=3).move_to([tx(2), bottom + step * 4.4, 0])
        electrode_dots.save_state()
        glow = [d.animate.set_fill(ACCENT).scale(1 + 1.5 * w) for d, w in zip(electrode_dots, near(on_head((0, 1.1)), .45)) if w > .1]
        self.play(eeg_rows[3].animate.set_opacity(1), Create(blink), *glow)
        self.wait(1)
        self.play(FadeOut(blink), Restore(electrode_dots))

        # Imputation: P3 is lost, and its neighbours bring it back
        hole = Rectangle(width=seen_width + .1, height=step * .9, fill_color=PAPER, fill_opacity=1, stroke_width=0)
        hole.move_to([tx(seen / 2), bottom + step * 2, 0])
        arrows = VGroup(*[Arrow([tx(.4), bottom + step * (len(CHANNELS) - 1 - c), 0], [tx(.4), bottom + step * 2, 0],
                                color=ACCENT, stroke_width=3, buff=.12, max_tip_length_to_length_ratio=.3) for c in (2, 4)])
        lost = Circle(radius=.09, color=ACCENT, stroke_width=3).move_to(on_head(CHANNEL_XY[3]))
        helpers = [d.animate.set_fill(ACCENT) for d, w in zip(electrode_dots, near(on_head(CHANNEL_XY[3]), .3)) if w > .2]
        self.play(eeg_rows[4].animate.set_opacity(1), FadeIn(hole), Create(lost))
        self.play(GrowArrow(arrows[0]), GrowArrow(arrows[1]), *helpers)
        self.play(Create(DashedVMobject(trace(3, color=ACCENT), num_dashes=80)), run_time=1.5)
        self.wait(1.5)

        # 6. The same table for both, the data on top and the tasks below
        self.play(*[FadeOut(m) for m in self.mobjects if m not in (head4, eeg_rows)])
        head6 = title("Same tasks, same models: from bikes to brains")
        def cell(s, color=INK, weight="NORMAL"):
            # A hidden bar, as tall as any glyph, puts every cell of a row on the same baseline
            t = text(s + "|", 22, color, weight)
            return t.add(t.submobjects.pop().set_opacity(0))

        grid = VGroup(VGroup(cell(""), cell("Bikes", ACCENT, "SEMIBOLD"), cell("EEG", ACCENT, "SEMIBOLD")))
        for name, bike, brain_ex in TABLE:
            grid.add(VGroup(cell(name, ACCENT if name in DATA_ROWS else PRIMARY, "SEMIBOLD"), cell(bike), cell(brain_ex)))
        for r, line_ in enumerate(grid):
            for c, cell in enumerate(line_):
                cell.move_to([[-5.2, -1.2, 2.6][c], 2.2 - .6 * r - .25 * (r > 2), 0], aligned_edge=LEFT)
        split = Line([-5.2, 2.2 - .6 * 2.5 - .12, 0], [5.2, 2.2 - .6 * 2.5 - .12, 0], color=RULE, stroke_width=1.5)
        self.play(FadeTransform(head4, head6), FadeOut(eeg_rows), FadeIn(grid), FadeIn(split))
        self.wait(3)
