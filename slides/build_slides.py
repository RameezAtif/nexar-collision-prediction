"""Build the presentation deck slides/Anticipating_Dashcam_Collisions.pptx.

Inputs: results/*.csv (every number on the slides is read from there) and slides/frames/*.jpg (made by slides/extract_frames.py from the raw videos).
The frames come from the Nexar Dashcam Collision Prediction dataset (Moura, Zhu, Zvitia, CVPRW 2025), used with attribution under the Nexar open data licence.
Run from the project root:  python slides/extract_frames.py && python slides/build_slides.py
"""

from pathlib import Path

import pandas as pd
from lxml import etree
from PIL import Image
from pptx import Presentation
from pptx.chart.data import CategoryChartData, XyChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_AXIS_CROSSES, XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION, XL_MARKER_STYLE, XL_TICK_LABEL_POSITION
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

FRAMES = Path("slides/frames")
RESULTS = Path("results")
OUT = Path("slides/Anticipating_Dashcam_Collisions.pptx")

# ------------------------------------------------------------------ palette (dark, matches the report: blue = CNN+LSTM, orange = optical flow / event)
BG, PANEL, PANEL2 = "0B1020", "141B30", "1D2644"
TEXT, MUTED, GRIDC = "F3F5FA", "9AA6BF", "2A3350"
BLUE, ORANGE, GREEN, GREY = "3987E5", "EB6834", "1BAF7A", "8A8FA3"
FONT = "Calibri"

# ------------------------------------------------------------------ data from the result files
summary = pd.read_csv(RESULTS / "temporal_ablation_summary.csv")
boot = pd.read_csv(RESULTS / "temporal_ablation_bootstrap.csv")
position = pd.read_csv(RESULTS / "failure_frame_position_ap.csv")
sel = pd.read_csv(RESULTS / "selection_check_fulldata.csv").groupby("tte").agg(final=("final_epoch", "mean"), best=("best_epoch", "mean"))
nb05 = pd.read_csv(RESULTS / "cnn_lstm_results.csv").sort_values("tte")
fail = pd.read_csv(FRAMES / "failure_examples.csv")
TTES = [0.5, 1.0, 1.5]


def ap(variant, tte, col="AP"):
    return float(summary[(summary["variant"] == variant) & (summary["tte"] == tte)].iloc[0][f"{col}_mean"])


def apsd(variant, tte):
    return float(summary[(summary["variant"] == variant) & (summary["tte"] == tte)].iloc[0]["AP_std"])


def bootrow(name, tte):
    r = boot[(boot["comparison"] == name) & (boot["tte"] == tte)].iloc[0]
    return float(r["estimate"]), float(r["ci_low"]), float(r["ci_high"])


def rgb(h):
    return RGBColor.from_string(h)


prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]
SW, SH = 13.333, 7.5
slide_no = [0]


# ------------------------------------------------------------------ helpers
def new_slide(kicker=None, title=None, notes="", footer=True, credit=False, bg=BG):
    s = prs.slides.add_slide(BLANK)
    slide_no[0] += 1
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = rgb(bg)
    if kicker:
        text(s, 0.7, 0.42, 9, 0.3, kicker.upper(), 12, bold=True, color=ORANGE)
    if title:
        text(s, 0.7, 0.72, 11.9, 1.0, title, 34, bold=True, color=TEXT, anchor=MSO_ANCHOR.TOP)
    if footer:
        text(s, 0.7, 7.08, 6, 0.25, "Anticipating Dashcam Collisions  ·  COMS4036A", 10, color=MUTED)
        text(s, 12.0, 7.08, 0.65, 0.25, str(slide_no[0]), 10, color=MUTED, align=PP_ALIGN.RIGHT)
    if credit:
        text(s, 5.2, 7.08, 6.6, 0.25, "Frames: Nexar Dashcam Collision Prediction dataset (Moura et al., CVPRW 2025), Nexar open data licence", 10, color=MUTED, align=PP_ALIGN.RIGHT)
    if notes:
        s.notes_slide.notes_text_frame.text = notes
    return s


def text(slide, x, y, w, h, content, size=16, bold=False, color=TEXT, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, italic=False, spacing=None):
    """content: str, or list of paragraphs; a paragraph is a str or a list of (text, {bold, color, size}) runs."""
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    paras = content if isinstance(content, list) else [content]
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        if spacing:
            p.space_after = Pt(spacing)
        runs = para if isinstance(para, list) else [(para, {})]
        for t, o in runs:
            r = p.add_run()
            r.text = t
            r.font.name = FONT
            r.font.size = Pt(o.get("size", size))
            r.font.bold = o.get("bold", bold)
            r.font.italic = o.get("italic", italic)
            r.font.color.rgb = rgb(o.get("color", color))
    return box


def set_alpha(shape, pct):
    clr = shape.fill._xPr.find(qn("a:solidFill"))[0]
    a = etree.SubElement(clr, qn("a:alpha"))
    a.set("val", str(int(pct * 1000)))


def box(slide, x, y, w, h, fill=PANEL, radius=0.06, alpha=None, shape=MSO_SHAPE.ROUNDED_RECTANGLE):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = rgb(fill)
    s.line.fill.background()
    s.shadow.inherit = False
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = radius
    if alpha is not None:
        set_alpha(s, alpha)
    return s


def picture(slide, name, x, y, w, h, radius=0.05):
    """Crop-to-fill picture with rounded corners."""
    path = FRAMES / f"{name}.jpg"
    iw, ih = Image.open(path).size
    pic = slide.shapes.add_picture(str(path), Inches(x), Inches(y), Inches(w), Inches(h))
    target, actual = w / h, iw / ih
    if actual > target:
        c = (1 - target / actual) / 2
        pic.crop_left = pic.crop_right = c
    else:
        c = (1 - actual / target) / 2
        pic.crop_top = pic.crop_bottom = c
    pic.auto_shape_type = MSO_SHAPE.ROUNDED_RECTANGLE
    prst = pic._element.spPr.find(qn("a:prstGeom"))
    av = prst.find(qn("a:avLst"))
    if av is None:
        av = etree.SubElement(prst, qn("a:avLst"))
    gd = etree.SubElement(av, qn("a:gd"))
    gd.set("name", "adj")
    gd.set("fmla", f"val {int(radius * 100000 * min(1.0, h / w * 1.4))}")
    return pic


def full_bleed(slide, name):
    path = FRAMES / f"{name}.jpg"
    iw, ih = Image.open(path).size
    pic = slide.shapes.add_picture(str(path), 0, 0, Inches(SW), Inches(SH))
    c = max((1 - (SW / SH) / (iw / ih)) / 2, 0)
    pic.crop_left = pic.crop_right = c if (iw / ih) > (SW / SH) else 0
    if (iw / ih) < (SW / SH):
        pic.crop_top = pic.crop_bottom = (1 - (iw / ih) / (SW / SH)) / 2
    return pic


def cell_borders(cell, color=GRIDC, width_pt=0.75):
    tcPr = cell._tc.get_or_add_tcPr()
    for tag in ("a:lnL", "a:lnR", "a:lnT", "a:lnB"):
        old = tcPr.find(qn(tag))
        if old is not None:
            tcPr.remove(old)
        ln = etree.SubElement(tcPr, qn(tag))
        ln.set("w", str(int(width_pt * 12700)))
        sf = etree.SubElement(ln, qn("a:solidFill"))
        etree.SubElement(sf, qn("a:srgbClr")).set("val", color)
    # border elements must precede the fill in tcPr
    fill = tcPr.find(qn("a:solidFill"))
    if fill is not None:
        tcPr.remove(fill)
        tcPr.append(fill)


def pill(slide, x, y, w, label, fill=PANEL2, color=TEXT, size=12, h=0.3):
    b = box(slide, x, y, w, h, fill=fill, radius=0.5)
    tf = b.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = label
    r.font.name, r.font.size, r.font.bold = FONT, Pt(size), True
    r.font.color.rgb = rgb(color)
    return b


def number_circle(slide, x, y, n, d=0.5, fill=BLUE):
    c = box(slide, x, y, d, d, fill=fill, shape=MSO_SHAPE.OVAL)
    tf = c.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = str(n)
    r.font.name, r.font.size, r.font.bold = FONT, Pt(18), True
    r.font.color.rgb = rgb(TEXT)


def line(slide, x1, y1, x2, y2, color=MUTED, width=1.5, dash=False):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = rgb(color)
    c.line.width = Pt(width)
    if dash:
        c.line.dash_style = 4  # dash
    return c


def stat(slide, x, y, w, big, small, color=BLUE, big_size=44):
    text(slide, x, y, w, 0.8, big, big_size, bold=True, color=color)
    text(slide, x, y + 0.82, w, 0.6, small, 14, color=MUTED)


def style_chart(chart, legend=True, size=14):
    chart.font.size = Pt(size)
    chart.font.name = FONT
    chart.font.color.rgb = rgb(MUTED)
    chart.has_legend = legend
    if legend:
        chart.legend.position = XL_LEGEND_POSITION.BOTTOM
        chart.legend.include_in_layout = False
        chart.legend.font.size = Pt(size)
        chart.legend.font.color.rgb = rgb(TEXT)
    cs = chart._chartSpace
    for parent, before in [(cs, cs.find(qn("c:txPr")))]:
        sp = etree.Element(qn("c:spPr"))
        etree.SubElement(sp, qn("a:noFill"))
        ln = etree.SubElement(sp, qn("a:ln"))
        etree.SubElement(ln, qn("a:noFill"))
        if before is not None:
            before.addprevious(sp)
        else:
            parent.append(sp)
    pa = chart.plots[0]._element.getparent()
    sp = etree.SubElement(pa, qn("c:spPr"))
    etree.SubElement(sp, qn("a:noFill"))


def style_axes(chart, vmin, vmax, major, fmt="0.00", vtitle=None, ctitle=None):
    va, ca = chart.value_axis, chart.category_axis
    va.minimum_scale, va.maximum_scale, va.major_unit = vmin, vmax, major
    va.has_major_gridlines = True
    va.major_gridlines.format.line.color.rgb = rgb(GRIDC)
    va.major_gridlines.format.line.width = Pt(0.75)
    va.format.line.fill.background()
    va.tick_labels.number_format = fmt
    va.tick_labels.number_format_is_linked = False
    va.tick_labels.font.size = Pt(13)
    va.tick_labels.font.color.rgb = rgb(MUTED)
    ca.format.line.color.rgb = rgb(GRIDC)
    ca.has_major_gridlines = False
    ca.tick_labels.font.size = Pt(14)
    ca.tick_labels.font.color.rgb = rgb(TEXT)
    for ax, t in ((va, vtitle), (ca, ctitle)):
        if t:
            ax.has_title = True
            ax.axis_title.text_frame.text = t
            f = ax.axis_title.text_frame.paragraphs[0].runs[0].font
            f.size, f.bold, f.name = Pt(13), False, FONT
            f.color.rgb = rgb(MUTED)


def color_series(series, color, width=3.0, marker=True, labels=None, label_pos=XL_LABEL_POSITION.ABOVE):
    series.format.line.color.rgb = rgb(color)
    series.format.line.width = Pt(width)
    series.smooth = False
    if marker:
        series.marker.style = XL_MARKER_STYLE.CIRCLE
        series.marker.size = 9
        series.marker.format.fill.solid()
        series.marker.format.fill.fore_color.rgb = rgb(color)
        series.marker.format.line.color.rgb = rgb(BG)
    if labels:
        dl = series.data_labels
        dl.show_value = True
        dl.number_format = labels
        dl.number_format_is_linked = False
        dl.position = label_pos
        dl.font.size = Pt(14)
        dl.font.bold = True
        dl.font.color.rgb = rgb(color)


def fill_series(series, color, labels=None):
    series.format.fill.solid()
    series.format.fill.fore_color.rgb = rgb(color)
    series.invert_if_negative = False
    if labels:
        dl = series.data_labels
        dl.show_value = True
        dl.number_format = labels
        dl.number_format_is_linked = False
        dl.position = XL_LABEL_POSITION.OUTSIDE_END
        dl.font.size = Pt(12)
        dl.font.bold = True
        dl.font.color.rgb = rgb(TEXT)


# ================================================================== 1. title
s = new_slide(footer=False, notes=(
    "Intro (about 30 s). We asked: can a dashcam model see a collision coming, and how early? We compared a naive baseline, optical flow, and a CNN plus LSTM, at 0.5, 1.0 and 1.5 s before the event. "
    "The short answer: learned features beat motion cues clearly, but when we tried to credit the temporal model, a single frame did just as well. "
    "The background frame is from the Nexar dataset (attribution on the last slide)."))
full_bleed(s, "bg_title")
ov = box(s, 0, 0, SW, SH, fill="05070F", alpha=62, shape=MSO_SHAPE.RECTANGLE)
pill(s, 0.9, 1.5, 3.4, "COMS4036A  ·  COMPUTER VISION", fill=ORANGE, size=12)
text(s, 0.9, 2.15, 11.8, 1.2, "Anticipating Dashcam Collisions", 50, bold=True)
text(s, 0.9, 3.45, 10.5, 1.0, "Do learned temporal features beat motion cues, and what do they actually use?", 26, color=MUTED)
text(s, 0.9, 5.5, 8, 0.8, [[("Muhammad Hoosen", {"bold": True}), ("  ·  2552770", {"color": MUTED})],
                           [("Rameez Atif", {"bold": True}), ("  ·  2612521", {"color": MUTED})]], 18, spacing=4)
text(s, 0.9, 6.7, 8, 0.3, "University of the Witwatersrand  ·  27 October 2026", 14, color=MUTED)

# ================================================================== 2. the question
s = new_slide("The problem", "Can a model see a crash coming — and how early?", notes=(
    "A warning only helps if it comes before the impact, so the interesting variable is how far ahead the model must predict: the time to event, TTE. "
    "Our hypothesis from the proposal: performance falls as TTE grows, but a learned temporal model degrades more slowly than a classical motion baseline. "
    "We defined success up front: the learned model beats optical flow in AP at every TTE with a confidence interval that excludes zero, and the gap widens with TTE. "
    "These four frames are one collision from the dataset: a car merges into our lane. Last frame is the annotated event."), credit=True)
items = [("How does prediction quality change as the warning horizon grows from 0.5 to 1.5 s?", BLUE),
         ("Does a learned temporal model beat classical motion cues?", ORANGE),
         ("Hypothesis: it degrades more slowly than optical flow. Success = it wins at every horizon, with a CI that excludes zero.", GREEN)]
for i, (t, c) in enumerate(items):
    y = 2.0 + i * 1.45
    box(s, 0.7, y, 5.7, 1.25, fill=PANEL)
    number_circle(s, 0.95, y + 0.37, i + 1, fill=c)
    text(s, 1.75, y + 0.18, 4.45, 0.95, t, 17, anchor=MSO_ANCHOR.MIDDLE)
grid = [("hero1024_m3.0", "−3.0 s"), ("hero1024_m1.5", "−1.5 s"), ("hero1024_m0.5", "−0.5 s"), ("hero1024_ev", "event")]
for k, (n, lab) in enumerate(grid):
    x, y = 6.85 + (k % 2) * 3.0, 2.0 + (k // 2) * 2.35
    picture(s, n, x, y, 2.85, 1.6)
    pill(s, x, y + 1.7, 0.95, lab, fill=ORANGE if lab == "event" else PANEL2, size=12)
text(s, 6.85, 6.55, 5.9, 0.4, "One collision from the dataset: a car merges into our lane.", 13, color=MUTED, italic=True)

# ================================================================== 3. the data
s = new_slide("The data", "Nexar dashcam videos: 750 collisions, 750 normal drives", notes=(
    "1,500 labelled videos from the Nexar collision prediction dataset, about 40 seconds each, 1280 by 720 at about 30 fps. Half contain a collision or near-collision, half are normal driving. "
    "Collision videos carry the annotated event time and an alert time, when annotators judged the danger to be apparent, on average 1.6 s before the event. "
    "The mosaic shows the variety: day, night, rain, dense city traffic. Licence: free use with attribution; it prohibits resale, re-identifying people or vehicles, weapons use and harmful systems. We redistribute no videos."), credit=True)
stat(s, 0.7, 2.0, 2.8, "1,500", "labelled videos")
stat(s, 3.7, 2.0, 2.8, "750 / 750", "collision vs normal", color=ORANGE, big_size=38)
stat(s, 0.7, 3.85, 2.8, "≈ 40 s", "per video, 1280×720, 30 fps", color=GREEN)
stat(s, 3.7, 3.85, 2.8, "2 labels", "event time and alert time (collisions)", color=BLUE)
box(s, 0.7, 5.75, 5.8, 0.95, fill=PANEL)
text(s, 0.95, 5.88, 5.3, 0.7, "Licence: free use with attribution. No resale, no re-identification, no weapons or harmful systems.", 14, color=TEXT, anchor=MSO_ANCHOR.MIDDLE)
mos = [("mos_c872", "collision · day"), ("mos_c677", "collision · rain"), ("mos_c540", "collision · night"),
       ("mos_c13", "collision · autumn"), ("mos_n1736", "normal · sun glare"), ("mos_n1770", "normal · night city")]
for k, (n, lab) in enumerate(mos):
    x, y = 6.95 + (k % 2) * 2.95, 1.95 + (k // 2) * 1.62
    picture(s, n, x, y, 2.8, 1.4)
    pill(s, x + 0.12, y + 1.0, 1.75, lab, fill="05070F", size=11, h=0.26)

# ================================================================== 4. clips and TTE
s = new_slide("Defining the task", "“Before the crash” becomes a measurement", notes=(
    "For a collision video with event time t_e and a chosen TTE, the prediction time is t_p = t_e minus TTE, and the input is 16 frames sampled evenly from the two seconds ending at t_p. "
    "We checked that no sampled frame lies after t_p, so there is no peeking. The bars are the three clip definitions; below them is the last frame the model sees in each case, from the same real video. "
    "For normal videos there is no event, so we draw the prediction time once from the distribution of the collision prediction times, so clip position is not a shortcut. A classifier using prediction time alone scores AUC about 0.5."), credit=True)
x0, scale = 0.9, 2.45  # time axis: event at x0 + 4.2*scale
ex = x0 + 4.2 * scale
for k, tte in enumerate(TTES):
    y = 2.0 + k * 0.62
    left, right = ex - (tte + 2.0) * scale, ex - tte * scale
    b = box(s, left, y, right - left, 0.46, fill=BLUE, alpha=70 if tte != 1.0 else 85)
    text(s, left + 0.12, y + 0.08, right - left - 0.2, 0.3, f"TTE {tte} s  ·  16 frames from the 2 s before t_p", 13, bold=True)
    text(s, right + 0.1, y + 0.08, 1.2, 0.3, "t_p", 13, color=MUTED)
line(s, ex, 1.85, ex, 4.0, color=ORANGE, width=3)
text(s, ex - 0.4, 1.5, 1.4, 0.3, "event  t_e", 14, bold=True, color=ORANGE)
shots = [("hero1024_m1.5", "TTE 1.5 s — last frame seen"), ("hero1024_m1.0", "TTE 1.0 s — last frame seen"),
         ("hero1024_m0.5", "TTE 0.5 s — last frame seen"), ("hero1024_ev", "the annotated event")]
for k, (n, lab) in enumerate(shots):
    x = 0.7 + k * 3.0
    picture(s, n, x, 4.3, 2.7, 1.52)
    text(s, x, 5.95, 2.7, 0.5, lab, 13, color=ORANGE if k == 3 else TEXT, bold=(k == 3))

# ================================================================== 5. protocol
s = new_slide("Experimental design", "One fair protocol for every model", notes=(
    "Four design choices. One: the split is by video, before any clip is cut: 1,200 training and 300 held-out videos, 50/50 balanced, seed 42, zero overlap checked. "
    "Two: every model sees exactly the same saved clips and TTEs. Three: the epoch is chosen on 20 percent of the training videos, never on the test videos; we learned this the hard way, see slide 13. "
    "Four: five training seeds, and a paired bootstrap over the 300 test clips, because 300 clips is few and seed standard deviation only shows training noise. "
    "Metrics: average precision is primary; we also report recall at a false-alarm rate of at most 10 percent of normal clips, as the proposal promised. Skip detail here if short on time."))
cards = [("Split by video", "1,200 train · 300 test, stratified 50/50, zero overlap. Split before cutting clips.", BLUE),
         ("Same clips, same metric", "All models share the saved clips and TTEs. Primary metric: average precision (AP).", ORANGE),
         ("No peeking at the test set", "Epoch chosen on 20% of the training videos. Test set scored once per run.", GREEN),
         ("Uncertainty that counts", "5 seeds, and a paired bootstrap over the 300 test clips (2,000 resamples).", BLUE)]
for k, (h, b_, c) in enumerate(cards):
    x = 0.7 + k * 3.05
    box(s, x, 2.0, 2.85, 3.9, fill=PANEL)
    text(s, x + 0.25, 2.2, 1.5, 0.9, f"0{k + 1}", 44, bold=True, color=c)
    text(s, x + 0.25, 3.25, 2.4, 0.8, h, 20, bold=True)
    text(s, x + 0.25, 4.2, 2.4, 1.6, b_, 15, color=MUTED)
bar_x = 0.95
box(s, bar_x, 5.35, 2.35 * 0.8, 0.22, fill=BLUE, radius=0.5)
box(s, bar_x + 2.35 * 0.8 + 0.04, 5.35, 2.35 * 0.2 - 0.04, 0.22, fill=ORANGE, radius=0.5)
text(s, 0.7, 6.25, 12, 0.5, "Also reported: recall at ≤ 10% false-alarm rate (naive chance = 0.10).", 15, color=MUTED)

# ================================================================== 6. three models
s = new_slide("Method", "Three models, from trivial to learned", notes=(
    "Model 0 predicts collision for every clip; its AP equals the 0.5 prevalence. "
    "Model 1 is classical: Farneback dense optical flow between consecutive sampled frames, about 0.13 s apart, seven statistics per pair (mean and std of horizontal and vertical flow and of magnitude, plus max magnitude), summarised by mean, std and max over the 15 pairs into 21 features, then standardised logistic regression. The image is a real flow field from our video. "
    "Model 2: a frozen ImageNet ResNet-18 gives a 512-d vector per frame, cached once; a one-layer LSTM with 128 units reads the 16 vectors in order and a linear layer outputs the collision probability. Adam, learning rate 1e-3, batch 32, 10 epochs. "
    "What we reused: the ResNet weights, Farneback from OpenCV, PyTorch and scikit-learn. What we wrote: sampling, split, features, training and all analyses."), credit=True)
for k, (h, c) in enumerate([("Model 0 · Naive", GREY), ("Model 1 · Optical flow", ORANGE), ("Model 2 · CNN + LSTM", BLUE)]):
    x = 0.7 + k * 4.1
    box(s, x, 1.95, 3.9, 4.8, fill=PANEL)
    text(s, x + 0.3, 2.15, 3.3, 0.4, h, 20, bold=True, color=c)
x = 0.7
text(s, x + 0.3, 3.0, 3.3, 1.0, "AP 0.50", 48, bold=True, color=GREY)
text(s, x + 0.3, 4.2, 3.3, 1.6, "Predict “collision” for every clip. The floor: with balanced classes AP equals the prevalence, 0.5.", 16, color=MUTED)
x = 4.8
picture(s, "flow_vis", x + 0.3, 2.7, 3.3, 1.86)
text(s, x + 0.3, 4.7, 3.3, 1.9, [[("Farnebäck flow", {"bold": True}), (" between frames 0.13 s apart", {})], "7 motion statistics per pair → 21 features → logistic regression"], 15, color=MUTED, spacing=6)
x = 8.9
for k in range(8):
    picture(s, f"clip16_{k * 2:02d}", x + 0.3 + (k % 4) * 0.84, 2.7 + (k // 4) * 0.55, 0.78, 0.48, radius=0.08)
text(s, x + 0.3, 3.85, 3.3, 0.3, "16 frames per clip (8 shown)", 12, color=MUTED)
for j, (lab, w) in enumerate([("ResNet-18 · frozen", 3.3), ("LSTM · 128 units", 3.3), ("P(collision)", 3.3)]):
    box(s, x + 0.3, 4.3 + j * 0.78, w, 0.58, fill=BLUE if j < 2 else PANEL2, radius=0.2)
    text(s, x + 0.3, 4.3 + j * 0.78, w, 0.58, lab, 16, bold=True, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)

# ================================================================== 7. result 1
s = new_slide("Results", "Learned features beat motion cues at every horizon", notes=(
    "Average precision on the 300 held-out test videos, mean over five training seeds. The CNN plus LSTM scores 0.762, 0.749 and 0.675; optical flow is barely above chance at about 0.57; naive is 0.5. "
    "The gap over optical flow is plus 0.197, 0.180 and 0.117, and every paired bootstrap interval excludes zero. At a 10 percent false-alarm limit the learned model recovers 0.38, 0.43 and 0.30 of collisions versus 0.17, 0.15 and 0.21 for flow; chance would be 0.10."))
cd = CategoryChartData()
cd.categories = ["0.5 s", "1.0 s", "1.5 s"]
cd.add_series("CNN + LSTM", [ap("lstm", t) for t in TTES])
cd.add_series("Optical flow", [ap("optical_flow", t) for t in TTES])
cd.add_series("Naive (chance)", [0.5, 0.5, 0.5])
gf = s.shapes.add_chart(XL_CHART_TYPE.LINE_MARKERS, Inches(0.6), Inches(1.75), Inches(8.0), Inches(5.1), cd)
ch = gf.chart
style_chart(ch)
style_axes(ch, 0.5, 0.9, 0.1, "0.0", vtitle="average precision (AP)", ctitle="time to event")
color_series(ch.series[0], BLUE, labels="0.000")
color_series(ch.series[1], ORANGE, labels="0.000", label_pos=XL_LABEL_POSITION.ABOVE)
color_series(ch.series[2], GREY, width=2, marker=False)
ch.series[2].format.line.dash_style = 4
d = [bootrow("AP diff: CNN+LSTM − Optical flow", t) for t in TTES]
stat(s, 9.1, 2.0, 3.6, f"+{d[0][0]:.2f}", "AP over optical flow at 0.5 s", color=BLUE)
stat(s, 9.1, 3.75, 3.6, f"+{d[2][0]:.2f}", "AP over optical flow at 1.5 s", color=BLUE)
box(s, 9.1, 5.5, 3.6, 1.2, fill=PANEL)
text(s, 9.3, 5.6, 3.2, 1.0, "All three paired 95% CIs exclude zero.", 16, anchor=MSO_ANCHOR.MIDDLE)

# ================================================================== 8. degradation + hypothesis
s = new_slide("Results", "Performance falls with horizon — but the hypothesis fails", notes=(
    "The CNN plus LSTM loses 0.086 AP from 0.5 to 1.5 s, with a 95 percent interval of 0.03 to 0.14. The 0.5 to 1.0 s drop, 0.016, is not distinguishable from zero. "
    "Optical flow is flat: 0.007, interval minus 0.05 to plus 0.06. But that is only because it is already near chance, so there is nothing to lose. "
    "So the hypothesis that the learned model degrades more slowly than flow is not supported: the learned model is the one that degrades. That is an honest negative result."))
dl_ = {"lstm": [bootrow("AP drop 0.5s→1.0s: CNN + LSTM", 1.0), bootrow("AP drop 0.5s→1.5s: CNN + LSTM", 1.5)],
       "flow": [bootrow("AP drop 0.5s→1.0s: Optical flow", 1.0), bootrow("AP drop 0.5s→1.5s: Optical flow", 1.5)]}
cd = CategoryChartData()
cd.categories = ["0.5 s → 1.0 s", "0.5 s → 1.5 s"]
cd.add_series("CNN + LSTM", [r[0] for r in dl_["lstm"]])
cd.add_series("Optical flow", [max(r[0], 0.0) for r in dl_["flow"]])
gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.6), Inches(1.75), Inches(6.9), Inches(5.1), cd)
ch = gf.chart
style_chart(ch)
style_axes(ch, 0.0, 0.12, 0.03, "0.00", vtitle="AP lost relative to 0.5 s", ctitle=None)
fill_series(ch.series[0], BLUE, labels="0.000")
fill_series(ch.series[1], ORANGE, labels="0.000")
ch.plots[0].gap_width = 60
ch.plots[0].overlap = -10
box(s, 7.9, 1.95, 4.75, 4.75, fill=PANEL)
text(s, 8.2, 2.15, 4.2, 0.4, "HYPOTHESIS", 12, bold=True, color=MUTED)
text(s, 8.2, 2.5, 4.2, 1.2, "The learned model degrades more slowly than optical flow", 18, bold=True)
text(s, 8.2, 3.9, 4.2, 0.8, "NOT SUPPORTED", 30, bold=True, color=ORANGE)
text(s, 8.2, 4.75, 4.2, 1.9, [f"CNN + LSTM drops {dl_['lstm'][1][0]:.3f} AP by 1.5 s (95% CI {dl_['lstm'][1][1]:.2f} to {dl_['lstm'][1][2]:.2f}).",
                              "Optical flow is “flat” only because it is already near chance."], 15, color=MUTED, spacing=8)

# ================================================================== 9. temporal ablation
s = new_slide("Probe", "A single frame does just as well as the temporal model", notes=(
    "This is the central probe. On the same frozen ResNet features we tried: the last frame only, the mean of the 16 frames (no order), an LSTM trained and tested on shuffled frames, and our LSTM. "
    "At 0.5 and 1.0 s the single frame matches the LSTM (differences plus 0.009 and plus 0.002, intervals include zero), and it is better at 1.5 s (minus 0.025, interval just excludes zero in the single frame's favour). "
    "Shuffling frames at test time costs the trained LSTM about 0.03 AP at 0.5 s and 0.02 at 1.0 s, so it uses a little order, but a model that never sees order does as well. "
    "So we cannot credit temporal modelling; the advantage over optical flow comes from the pretrained visual features."), credit=True)
cd = CategoryChartData()
cd.categories = ["0.5 s", "1.0 s", "1.5 s"]
for var, nm in [("lstm", "CNN + LSTM"), ("last_frame", "Last frame only"), ("mean_pool", "Mean-pool (no order)"), ("lstm_train_shuffled", "LSTM, shuffled frames")]:
    cd.add_series(nm, [ap(var, t) for t in TTES])
gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.6), Inches(1.75), Inches(8.2), Inches(5.1), cd)
ch = gf.chart
style_chart(ch)
style_axes(ch, 0.5, 0.8, 0.1, "0.0", vtitle="average precision (AP); chance = 0.5", ctitle="time to event")
for sr, c in zip(ch.series, [BLUE, GREEN, GREY, "B58CF5"]):
    fill_series(sr, c, labels="0.00")
    sr.data_labels.font.size = Pt(11)
ch.plots[0].gap_width = 55
ch.plots[0].overlap = -8
d1 = bootrow("AP diff: CNN+LSTM − Last frame (no temporal)", 1.0)
picture(s, "hero1024_m0.5", 9.2, 1.95, 3.45, 1.94)
pill(s, 9.2, 4.0, 2.4, "one frame, no motion", fill=PANEL2, size=12)
box(s, 9.2, 4.55, 3.45, 2.15, fill=PANEL)
text(s, 9.4, 4.65, 3.1, 0.7, f"{d1[0]:+.3f}", 36, bold=True, color=GREEN)
text(s, 9.4, 5.4, 3.1, 1.2, f"LSTM minus last frame at 1.0 s; 95% CI [{d1[1]:+.3f}, {d1[2]:+.3f}] includes zero", 14, color=MUTED)

# ================================================================== 10. what does it use
s = new_slide("Probe", "Most of the signal is there long before the event", notes=(
    "We trained the single-frame classifier on each of the 16 frame positions separately and plotted AP against how many seconds before the event that frame was taken. "
    "A frame from three or more seconds before the event already gives AP of about 0.69, with chance at 0.5; frames within about 1.2 s of the event raise it only to about 0.75. "
    "The three curves, from the three clip definitions, line up. So most of the signal is not a late pre-collision cue. There is a hazard-related part: collision clips that end after the annotated alert are detected better, AUC 0.72 to 0.77, than clips that end before it, 0.57 to 0.66, but both are above chance."), credit=True)
xy = XyChartData()
for tte, nm in zip(TTES, ["clip ends 0.5 s before event", "clip ends 1.0 s before event", "clip ends 1.5 s before event"]):
    sr = xy.add_series(nm)
    sub = position[position["tte"] == tte].sort_values("seconds_before_event")
    for _, r in sub.iterrows():
        sr.add_data_point(float(r["seconds_before_event"]), float(r["AP_mean"]))
gf = s.shapes.add_chart(XL_CHART_TYPE.XY_SCATTER_LINES, Inches(0.6), Inches(1.75), Inches(8.2), Inches(5.1), xy)
ch = gf.chart
style_chart(ch)
va, ca = ch.value_axis, ch.category_axis
va.minimum_scale, va.maximum_scale, va.major_unit = 0.5, 0.85, 0.1
va.major_gridlines.format.line.color.rgb = rgb(GRIDC)
va.format.line.fill.background()
va.tick_labels.font.size = Pt(13)
va.tick_labels.font.color.rgb = rgb(MUTED)
va.tick_labels.number_format = "0.0"
va.tick_labels.number_format_is_linked = False
va.has_title = True
va.axis_title.text_frame.text = "single-frame AP; chance = 0.5"
ca.minimum_scale, ca.maximum_scale, ca.major_unit = 0.0, 4.0, 0.5
ca.reverse_order = True
ca.crosses = XL_AXIS_CROSSES.MAXIMUM
ca.has_major_gridlines = False
ca.format.line.color.rgb = rgb(GRIDC)
ca.tick_labels.font.size = Pt(13)
ca.tick_labels.font.color.rgb = rgb(TEXT)
ca.has_title = True
ca.axis_title.text_frame.text = "seconds before the event at which the single frame was taken"
for ax in (va, ca):
    f = ax.axis_title.text_frame.paragraphs[0].runs[0].font
    f.size, f.bold, f.name = Pt(13), False, FONT
    f.color.rgb = rgb(MUTED)
ca.tick_label_position = XL_TICK_LABEL_POSITION.LOW
for sr, c in zip(ch.series, [BLUE, ORANGE, GREEN]):
    color_series(sr, c, width=2.25, marker=False)
picture(s, "pos329_far", 9.2, 1.95, 3.45, 1.94)
pill(s, 9.2, 4.0, 2.7, "3.0 s before the event", fill=PANEL2, size=12)
picture(s, "pos329_near", 9.2, 4.5, 3.45, 1.94)
pill(s, 9.2, 6.55, 2.7, "0.5 s before the event", fill=ORANGE, size=12)

# ================================================================== 11. failures
s = new_slide("Failures", "Where it goes wrong: dense night traffic and murky scenes", notes=(
    "71 of the 300 test videos are wrong at every horizon: 36 false alarms and 35 missed collisions. "
    "We looked at twelve of them. The false alarms (top row) are mostly dense night traffic with a vehicle close ahead, which reads as risky but nothing happened. "
    "The misses (bottom row) are mostly rain, glare or empty road, where a collision gives little visible warning in the clip. "
    "The probabilities are the CNN plus LSTM seed-ensemble at TTE 1.0 s. This is a qualitative impression from a dozen videos, not a result; the quantitative version is on the previous slide."), credit=True)
rows = [("FALSE ALARMS · normal videos the model flags", ORANGE, "fa", 1.95), ("MISSED COLLISIONS · collision videos scored low", BLUE, "miss", 4.6)]
for lab, c, kind, y in rows:
    text(s, 0.7, y - 0.05, 8, 0.3, lab, 12, bold=True, color=c)
    sub = fail[fail["kind"] == kind]
    for k, (_, r) in enumerate(sub.iterrows()):
        x = 0.7 + k * 4.05
        picture(s, f"err_{kind}_{int(r['id'])}", x, y + 0.3, 3.8, 2.0)
        pill(s, x + 0.12, y + 0.42, 1.55, f"P(collision) = {r['p_lstm_tte1.0']:.2f}", fill="05070F", size=11, h=0.28)

# ================================================================== 12. id blocks
s = new_slide("Caveat", "The two classes live in separate id blocks", notes=(
    "Every collision video has id up to 1039 and every normal video has id 1040 or above, so the video id alone separates the classes. No model sees the id, so no number is invalid, but the blocks may differ in recording conditions. "
    "What we checked on a 431-video sample of the raw videos: resolution, frame rate, duration, brightness and the Nexar overlay do not differ between blocks, and those properties together predict the class at AUC 0.58, inside what shuffled labels reach. "
    "What the cached features show: the collision score drifts down with id inside the collision block (Spearman minus 0.13, p 0.0005), and class separation is weaker near the boundary, AUC 0.59 within 100 ids versus 0.73 farther away. "
    "What we could not test: scene content such as location, season and traffic. So we say the model separates the two video groups, most plausibly by recognising high-risk scenes, not that it detects imminent collisions."))
total = 11.9
cw = total * 1040 / 2140
box(s, 0.7, 2.05, cw - 0.03, 0.75, fill=BLUE)
box(s, 0.7 + cw, 2.05, total - cw, 0.75, fill=GREY)
text(s, 0.95, 2.05, cw - 0.3, 0.75, "collision videos · ids 0 – 1039", 17, bold=True, anchor=MSO_ANCHOR.MIDDLE)
text(s, 0.95 + cw, 2.05, total - cw - 0.3, 0.75, "normal videos · ids 1040 – 2139", 17, bold=True, anchor=MSO_ANCHOR.MIDDLE)
line(s, 0.7 + cw - 0.015, 1.9, 0.7 + cw - 0.015, 2.95, color=ORANGE, width=3)
cards = [("What we checked", GREEN, ["Resolution, frame rate, duration, brightness, overlay: no difference between blocks (431 videos)", "These properties predict the class at AUC 0.58 ≈ chance"]),
         ("What the features show", ORANGE, ["Collision score drifts down with id: ρ = −0.13, p = 0.0005", "Class separation weaker near the boundary: AUC 0.59 vs 0.73"]),
         ("What we could not test", BLUE, ["Scene content: location, season, traffic", "So: “separates two video groups”, not “predicts imminent collisions”"])]
for k, (h, c, pts) in enumerate(cards):
    x = 0.7 + k * 4.05
    box(s, x, 3.3, 3.8, 3.45, fill=PANEL)
    text(s, x + 0.3, 3.5, 3.2, 0.4, h, 20, bold=True, color=c)
    text(s, x + 0.3, 4.2, 3.2, 2.5, pts, 15, color=TEXT, spacing=12)

# ================================================================== 13. protocol lesson
s = new_slide("Lesson", "Choosing the epoch on the test set inflated AP by up to 0.07", notes=(
    "Our first CNN plus LSTM run kept the epoch with the best AP on the evaluation videos and reported that score: 0.793, 0.774, 0.739 (the chart shows the 5-seed mean of the same procedure: 0.793, 0.780, 0.742). "
    "We then trained on the full training set for a fixed 10 epochs with no selection: 0.754, 0.753, 0.676. So the selection inflated AP by about 0.04, 0.03 and 0.07, most at 1.5 s, which also flattened the apparent degradation curve. "
    "Holding out 20 percent of the training videos to pick the epoch costs almost nothing: 0.762, 0.749, 0.675. That is the protocol we report. "
    "The lesson: tuning on the evaluation set flatters you most where the task is hardest."))
cd = CategoryChartData()
cd.categories = ["0.5 s", "1.0 s", "1.5 s"]
cd.add_series("Best epoch picked on the test videos (5-seed mean)", [float(sel.loc[t, "best"]) for t in TTES])
cd.add_series("Fixed 10 epochs, no selection", [float(sel.loc[t, "final"]) for t in TTES])
cd.add_series("Epoch picked on held-out training videos (reported)", [ap("lstm", t) for t in TTES])
gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.6), Inches(1.75), Inches(8.8), Inches(5.1), cd)
ch = gf.chart
style_chart(ch, size=13)
style_axes(ch, 0.5, 0.85, 0.1, "0.0", vtitle="CNN + LSTM average precision", ctitle="time to event")
for sr, c in zip(ch.series, [ORANGE, GREY, BLUE]):
    fill_series(sr, c, labels="0.000")
    sr.data_labels.font.size = Pt(11)
ch.plots[0].gap_width = 55
ch.plots[0].overlap = -6
infl = [float(sel.loc[t, "best"]) - float(sel.loc[t, "final"]) for t in TTES]
stat(s, 9.8, 2.0, 2.9, f"+{infl[2]:.2f}", "AP inflation at 1.5 s from selecting on the test videos", color=ORANGE, big_size=48)
box(s, 9.8, 4.4, 2.85, 2.3, fill=PANEL)
text(s, 10.0, 4.55, 2.5, 2.0, "Holding out 20% of the training videos costs almost nothing, so that is what we report.", 15, anchor=MSO_ANCHOR.MIDDLE)

# ================================================================== 14. limitations and next steps
s = new_slide("Honest limits", "Limitations and what we would do next", notes=(
    "Limitations: there is no separate untouched test set, so the 300 held-out videos were used for earlier choices such as the scaled flow variant, about 0.01 AP. With 300 clips an AP has a 95 percent interval of about plus or minus 0.07, so differences near 0.03 are within noise. "
    "One frozen backbone and one clip design. The flow baseline uses frames 0.13 s apart, so classical motion cues may be underestimated. And the id-block structure. "
    "Next steps: compare the id blocks on scene content with all 1,500 videos and evaluate on a split that holds out id ranges; pick epoch and regularisation by nested cross-validation; compute flow between adjacent frames and try a fine-tuned or two-stream model; add seeds and intervals to every ablation."))
box(s, 0.7, 1.95, 5.85, 4.8, fill=PANEL)
text(s, 1.0, 2.15, 5.2, 0.4, "LIMITATIONS", 14, bold=True, color=ORANGE)
text(s, 1.0, 2.75, 5.3, 3.9, ["No untouched test set: the 300 videos informed earlier choices", "300 clips: an AP has a 95% interval of about ±0.07", "One frozen backbone, one clip design, 10 epochs", "Flow baseline uses frames 0.13 s apart: may understate classical cues", "Class-specific id blocks; scene content untested"], 18, spacing=12)
box(s, 6.8, 1.95, 5.85, 4.8, fill=PANEL)
text(s, 7.1, 2.15, 5.2, 0.4, "NEXT STEPS", 14, bold=True, color=GREEN)
text(s, 7.1, 2.75, 5.3, 3.9, ["Compare the id blocks on scene content; evaluate on a split that holds out id ranges", "Choose epoch and regularisation by nested cross-validation", "Optical flow between adjacent frames; fine-tuned or two-stream model", "Seeds and intervals for every ablation"], 18, spacing=12)

# ================================================================== 15. takeaways
s = new_slide(footer=False, notes=(
    "Three take-aways. One: pretrained visual features clearly beat optical flow, at every horizon. Two: we could not credit temporal modelling, because a single frame does as well, and much of the signal is present seconds before the event. "
    "Three: evaluation protocol matters; selecting the epoch on the test set flattered us by up to 0.07 AP. "
    "The code, notebooks and the experiment log are in the repository; every number traces to a result file. Invite questions."))
bgpic = s.shapes.add_picture(str(FRAMES / "bg_close.jpg"), 0, 0, Inches(SW), Inches(SH))
iw, ih = Image.open(FRAMES / "bg_close.jpg").size
c = (1 - (SW / SH) / (iw / ih)) / 2
bgpic.crop_top = bgpic.crop_bottom = max(c, 0)
box(s, 0, 0, SW, SH, fill="05070F", alpha=70, shape=MSO_SHAPE.RECTANGLE)
text(s, 0.9, 0.9, 11, 0.4, "TAKE-AWAYS", 14, bold=True, color=ORANGE)
tk = [("Learned features beat motion cues", f"AP {ap('lstm', 0.5):.3f} / {ap('lstm', 1.0):.3f} / {ap('lstm', 1.5):.3f} vs {ap('optical_flow', 0.5):.3f} / {ap('optical_flow', 1.0):.3f} / {ap('optical_flow', 1.5):.3f} for optical flow", BLUE),
      ("…but we cannot credit the temporal model", "A single frame does as well; much of the signal is there 3 s before the event", ORANGE),
      ("Evaluation protocol matters", "Picking the epoch on the test set inflated AP by up to 0.07", GREEN)]
for k, (h, b_, cc) in enumerate(tk):
    y = 1.6 + k * 1.45
    number_circle(s, 0.9, y + 0.1, k + 1, d=0.6, fill=cc)
    text(s, 1.8, y, 10.5, 0.5, h, 28, bold=True)
    text(s, 1.8, y + 0.58, 10.5, 0.5, b_, 17, color=MUTED)
text(s, 0.9, 6.15, 11.5, 0.4, "Code and experiment log: github.com/RameezAtif/nexar-collision-prediction", 15, color=TEXT)
text(s, 0.9, 6.6, 11.5, 0.4, "Questions?   Muhammad Hoosen · Rameez Atif", 20, bold=True, color=ORANGE)

# ================================================================== appendix divider
s = new_slide("Appendix", "Backup slides", notes="Backup material for questions.", footer=True)
picture(s, "mos_c677", 7.6, 1.95, 5.0, 4.6, radius=0.04)
text(s, 0.7, 2.2, 6.5, 3, ["A1  Full results table", "A2  What we wrote and what we reused", "A3  Reproducing the results", "A4  Data licence and credits"], 24, color=MUTED, spacing=14)

# ================================================================== A1 results table
s = new_slide("Appendix A1", "Full results (test set, 300 videos)", notes=(
    "Mean plus or minus standard deviation over five training seeds. Optical flow and naive are deterministic. Recall is at a false-alarm rate of at most 10 percent of normal clips; the naive value is the chance level. "
    "Source files: results/temporal_ablation_summary.csv and temporal_ablation_bootstrap.csv; the report tables are generated from the same files."))
hdr = ["Model", "AP 0.5 s", "AP 1.0 s", "AP 1.5 s", "R@10% 0.5 s", "R@10% 1.0 s", "R@10% 1.5 s"]
rows_ = [("Naive", "naive"), ("Optical flow", "optical_flow"), ("CNN + LSTM", "lstm"), ("Last frame only", "last_frame"), ("Mean-pool", "mean_pool"), ("LSTM, shuffled train+test", "lstm_train_shuffled"), ("LSTM, shuffled at test", "lstm_test_shuffled")]
tbl = s.shapes.add_table(len(rows_) + 1, 7, Inches(0.7), Inches(1.9), Inches(11.9), Inches(4.4)).table
for j, h in enumerate(hdr):
    c = tbl.cell(0, j)
    c.text = h
for i, (nm, var) in enumerate(rows_, 1):
    vals = [nm]
    for t in TTES:
        m, sd = ap(var, t), apsd(var, t)
        vals.append(f"{m:.3f}" if pd.isna(sd) else f"{m:.3f} ± {sd:.3f}")
    for t in TTES:
        r = ap(var, t, "Recall@FPR10")
        vals.append(f"{r:.2f}")
    for j, v in enumerate(vals):
        tbl.cell(i, j).text = v
for i in range(len(rows_) + 1):
    for j in range(7):
        c = tbl.cell(i, j)
        c.fill.solid()
        c.fill.fore_color.rgb = rgb(PANEL2 if i == 0 else (PANEL if i % 2 else BG))
        c.margin_left = c.margin_right = Inches(0.1)
        c.vertical_anchor = MSO_ANCHOR.MIDDLE
        cell_borders(c)
        for p in c.text_frame.paragraphs:
            p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
            for r in p.runs:
                r.font.size = Pt(14 if i else 13)
                r.font.name = FONT
                r.font.bold = (i == 0) or (rows_[i - 1][1] == "lstm" if i else False)
                r.font.color.rgb = rgb(BLUE if (i and rows_[i - 1][1] == "lstm") else TEXT)
tbl.columns[0].width = Inches(3.5)
for j in range(1, 7):
    tbl.columns[j].width = Inches((11.9 - 3.5) / 6)

# ================================================================== A2 wrote vs reused
s = new_slide("Appendix A2", "What we wrote and what we reused", notes=(
    "Reused: the Nexar videos and labels; ImageNet ResNet-18 weights from torchvision; Farneback optical flow from OpenCV; LSTM and Adam from PyTorch; logistic regression and the AP and AUC metrics from scikit-learn; the IEEE report class. "
    "Written by us: the video-level split, the TTE clip sampler and leakage checks, negative sampling, the flow-statistic features, feature caching, training and evaluation code, the ablation variants, the bootstrap and permutation analyses, the failure and id-block analyses, and all notebooks and scripts."))
box(s, 0.7, 1.95, 5.85, 4.8, fill=PANEL)
text(s, 1.0, 2.15, 5.2, 0.4, "REUSED", 14, bold=True, color=MUTED)
text(s, 1.0, 2.75, 5.3, 3.9, ["Nexar videos and labels", "ImageNet ResNet-18 weights (torchvision)", "Farnebäck optical flow (OpenCV)", "LSTM, Adam (PyTorch)", "Logistic regression, AP and AUC (scikit-learn)"], 16, spacing=12)
box(s, 6.8, 1.95, 5.85, 4.8, fill=PANEL)
text(s, 7.1, 2.15, 5.2, 0.4, "WRITTEN BY US", 14, bold=True, color=ORANGE)
text(s, 7.1, 2.75, 5.3, 3.9, ["Video-level split, TTE clip sampler, leakage checks", "Negative sampling; flow-statistic features", "Feature caching, training and evaluation code", "Ablation variants; bootstrap and permutation analyses", "Failure and id-block analyses; all notebooks and scripts"], 16, spacing=12)

# ================================================================== A3 reproducibility
s = new_slide("Appendix A3", "Reproducing the results", notes=(
    "From a fresh git clone with no raw data, python tools/run_notebooks.py reruns notebooks 06 to 09 from the committed cached features in about five minutes on a CPU, and regenerates all 24 result CSVs identically, difference below 1e-9, on the same machine. "
    "Seeds: 42 for the split and sampling, 42 to 46 for training. The README has a table mapping each reported number to its file."))
box(s, 0.7, 1.95, 11.9, 3.3, fill="0E1428")
code = ["pip install -r requirements.txt", "python tools/run_notebooks.py        # notebooks 06-09 (~5 min, CPU)",
        "python tools/make_report_assets.py   # report figures and tables from results/*.csv", "cd report && sh build.sh               # report PDF (Docker)"]
mono = text(s, 1.0, 2.2, 11.3, 2.9, code, 15, color=GREEN, spacing=14)
for para in mono.text_frame.paragraphs:
    for r in para.runs:
        r.font.name = 'Courier New'
text(s, 0.7, 5.55, 11.9, 1.2, [[("Verified: ", {"bold": True}), ("a fresh clone with no raw data regenerated all 24 result files identically (difference < 1e-9).", {})],
                               [("Seeds: ", {"bold": True}), ("42 for split and sampling; 42–46 for training; fixed seeds for bootstrap and permutation tests.", {})]], 16, color=MUTED, spacing=8)

# ================================================================== A4 licence and credits
s = new_slide("Appendix A4", "Data licence and credits", notes=(
    "The Nexar open data licence permits free use, copying and redistribution with attribution to Nexar Inc. and the copyright notice retained; it prohibits resale, attempts to re-identify people or vehicles, weapons use and systems intended to cause harm. "
    "All frames in these slides are from that dataset and are used for research and teaching only; we redistribute no videos."))
text(s, 0.7, 2.0, 11.9, 4.6, [
    [("Dataset. ", {"bold": True}), ("D. C. Moura, S. Zhu, O. Zvitia, “Nexar Dashcam Collision Prediction Dataset and Challenge,” CVPR Workshops 2025, pp. 2608–2616. © Nexar Inc.", {})],
    [("Licence. ", {"bold": True}), ("Nexar open data licence: free use, copying and redistribution with attribution; no resale, no re-identification of people or vehicles, no weapons use, no systems intended to cause harm.", {})],
    [("This deck. ", {"bold": True}), ("All frames are from the dataset, used for research and teaching with attribution. No raw videos are redistributed.", {})],
    [("Models and libraries. ", {"bold": True}), ("ResNet-18 (He et al., 2016), LSTM (Hochreiter and Schmidhuber, 1997), Farnebäck flow (2003), PyTorch, scikit-learn, OpenCV.", {})]], 17, color=TEXT, spacing=16)

try:
    prs.save(OUT)
except PermissionError:                      # the deck is open in PowerPoint: save next to it instead of failing
    OUT = OUT.with_name(OUT.stem + "_updated.pptx")
    prs.save(OUT)
print("saved", OUT, "with", len(prs.slides), "slides")
