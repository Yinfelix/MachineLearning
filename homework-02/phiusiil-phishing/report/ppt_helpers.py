"""Small layout toolkit for a clean 16:9 academic deck."""
from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt
from PIL import Image

SURFACE = RGBColor(0xFC, 0xFC, 0xFB)
INK = RGBColor(0x0B, 0x0B, 0x0B)
INK_2 = RGBColor(0x52, 0x51, 0x4E)
MUTED = RGBColor(0x89, 0x87, 0x81)
GRID = RGBColor(0xE1, 0xE0, 0xD9)
BLUE = RGBColor(0x2A, 0x78, 0xD6)
ORANGE = RGBColor(0xEB, 0x68, 0x34)
AQUA = RGBColor(0x1B, 0xAF, 0x7A)
YELLOW = RGBColor(0xED, 0xA1, 0x00)
MAGENTA = RGBColor(0xE8, 0x7B, 0xA4)
VIOLET = RGBColor(0x62, 0x50, 0xD6)
RED = RGBColor(0xE3, 0x49, 0x48)
LIGHT_BLUE = RGBColor(0xED, 0xF2, 0xF9)

FONT = "Microsoft YaHei"
FONT_EN = "Segoe UI"


def new_deck() -> Presentation:
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    return prs


def blank(prs: Presentation):
    return prs.slides.add_slide(prs.slide_layouts[6])


def rect(slide, x, y, w, h, fill=None, line=None, radius=False, line_width=1.0):
    shape_type = MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE
    shp = slide.shapes.add_shape(shape_type, Inches(x), Inches(y), Inches(w), Inches(h))
    if fill is None:
        shp.fill.background()
    else:
        shp.fill.solid()
        shp.fill.fore_color.rgb = fill
    if line is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line
        shp.line.width = Pt(line_width)
    shp.shadow.inherit = False
    if radius:
        try:
            shp.adjustments[0] = 0.08
        except Exception:
            pass
    return shp


def textbox(slide, text, x, y, w, h, size=16, bold=False, color=INK, font=FONT,
            align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, line_spacing=1.15, margins=(0.03, 0.02)):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = Inches(margins[0]); tf.margin_right = Inches(margins[0])
    tf.margin_top = Inches(margins[1]); tf.margin_bottom = Inches(margins[1])
    p = tf.paragraphs[0]
    p.alignment = align
    p.line_spacing = line_spacing
    r = p.add_run()
    r.text = text
    r.font.name = font
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.color.rgb = color
    return tb


def bullets(slide, items, x, y, w, h, size=16, color=INK, bullet_color=BLUE, spacing=8, font=FONT):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.02); tf.margin_right = Inches(0.02)
    tf.margin_top = Inches(0.02); tf.margin_bottom = Inches(0.02)
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(spacing)
        p.line_spacing = 1.25
        p.level = 0
        # bullet glyph as a colored run, then text
        rb = p.add_run()
        rb.text = "•  "
        rb.font.name = font
        rb.font.size = Pt(size)
        rb.font.bold = True
        rb.font.color.rgb = bullet_color
        if isinstance(item, tuple):
            text, bold = item
        else:
            text, bold = item, False
        rt = p.add_run()
        rt.text = text
        rt.font.name = font
        rt.font.size = Pt(size)
        rt.font.bold = bold
        rt.font.color.rgb = color
    return tb


def title(slide, kicker, headline, page=None, total=None):
    # thin accent bar
    rect(slide, 0, 0, 13.333, 0.08, fill=BLUE)
    textbox(slide, kicker, 0.7, 0.42, 6.0, 0.35, size=11, bold=True, color=BLUE)
    textbox(slide, headline, 0.7, 0.78, 11.9, 0.75, size=27, bold=True, color=INK)



def footer(slide, page, total=None):
    rect(slide, 0.7, 7.05, 11.93, 0.012, fill=GRID)
    textbox(slide, "PhiUSIIL 钓鱼网站检测 ｜ 机器学习课程项目 2", 0.7, 7.08, 7.5, 0.28, size=9, color=MUTED)
    txt = f"{page:02d}" + (f" / {total:02d}" if total else "")
    textbox(slide, txt, 11.7, 7.08, 0.9, 0.28, size=9, color=MUTED, align=PP_ALIGN.RIGHT)


def image_fit(slide, path, x, y, w, h):
    """Place an image inside the box, preserving aspect ratio and centring."""
    with Image.open(path) as im:
        iw, ih = im.size
    scale = min(w / iw, h / ih)
    nw, nh = iw * scale, ih * scale
    left = x + (w - nw) / 2
    top = y + (h - nh) / 2
    return slide.shapes.add_picture(str(path), Inches(left), Inches(top), Inches(nw), Inches(nh))


def stat_tile(slide, value, label, x, y, w, h, accent=BLUE, note=None):
    rect(slide, x, y, w, h, fill=SURFACE, line=GRID)
    rect(slide, x, y, 0.06, h, fill=accent)
    textbox(slide, value, x + 0.18, y + 0.12, w - 0.28, h * 0.52, size=28, bold=True, color=accent)
    textbox(slide, label, x + 0.18, y + h * 0.58, w - 0.28, h * 0.28, size=11.5, color=INK_2)
    if note:
        textbox(slide, note, x + 0.18, y + h * 0.82, w - 0.28, h * 0.16, size=9, color=MUTED)


def callout(slide, text, x, y, w, h, accent=ORANGE, size=13, fill=None):
    rect(slide, x, y, w, h, fill=fill or RGBColor(0xFF, 0xF4, 0xEE), line=None, radius=True)
    rect(slide, x, y, 0.06, h, fill=accent)
    textbox(slide, text, x + 0.18, y + 0.08, w - 0.26, h - 0.12, size=size, color=INK_2,
            anchor=MSO_ANCHOR.MIDDLE)


def table(slide, headers, rows, x, y, w, h, col_widths=None, font_size=10.5, header_fill=LIGHT_BLUE):
    shape = slide.shapes.add_table(len(rows) + 1, len(headers), Inches(x), Inches(y), Inches(w), Inches(h))
    tbl = shape.table
    if col_widths:
        for i, cw in enumerate(col_widths):
            tbl.columns[i].width = Inches(cw)
    for j, head in enumerate(headers):
        cell = tbl.cell(0, j)
        cell.fill.solid(); cell.fill.fore_color.rgb = header_fill
        cell.text = ""
        p = cell.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run(); r.text = str(head)
        r.font.name = FONT; r.font.size = Pt(font_size); r.font.bold = True; r.font.color.rgb = INK
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    for i, row in enumerate(rows, start=1):
        for j, val in enumerate(row):
            cell = tbl.cell(i, j)
            cell.fill.solid(); cell.fill.fore_color.rgb = SURFACE
            cell.text = ""
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER
            r = p.add_run(); r.text = str(val)
            r.font.name = FONT; r.font.size = Pt(font_size); r.font.color.rgb = INK_2
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    return shape


def section_number(slide, number, title_text, subtitle=""):
    """A full-bleed section divider."""
    rect(slide, 0, 0, 13.333, 7.5, fill=SURFACE)
    textbox(slide, number, 0.9, 1.8, 2.2, 1.3, size=72, bold=True, color=BLUE)
    textbox(slide, title_text, 0.95, 3.2, 10.8, 0.9, size=34, bold=True, color=INK)
    if subtitle:
        textbox(slide, subtitle, 0.98, 4.15, 10.5, 0.6, size=15, color=INK_2)
    rect(slide, 0.95, 5.05, 1.5, 0.06, fill=BLUE)
