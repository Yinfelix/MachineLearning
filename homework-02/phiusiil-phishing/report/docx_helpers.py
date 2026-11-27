"""Reusable python-docx helpers matching the course report template."""
from __future__ import annotations

from docx.enum.section import WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Cm, Inches, Pt, RGBColor

BODY_EAST_FONT = "宋体"
BODY_ASCII_FONT = "Times New Roman"
HEAD_FONT = "黑体"


def set_run_font(run, size=12, bold=False, italic=False, ascii_font=BODY_ASCII_FONT,
                 east_font=BODY_EAST_FONT, color=None):
    run.font.name = ascii_font
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run._element.rPr.rFonts.set(qn("w:eastAsia"), east_font)
    if color is not None:
        run.font.color.rgb = RGBColor.from_string(color)


def setup_document(doc):
    sec = doc.sections[0]
    sec.page_width = Cm(21.0)
    sec.page_height = Cm(29.7)
    sec.left_margin = Inches(1.0)
    sec.right_margin = Inches(1.0)
    sec.top_margin = Inches(1.0)
    sec.bottom_margin = Inches(1.0)
    sec.header_distance = Inches(0.5)
    sec.footer_distance = Inches(0.5)

    normal = doc.styles["Normal"]
    normal.font.name = BODY_ASCII_FONT
    normal.font.size = Pt(10.5)
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), BODY_EAST_FONT)
    normal.paragraph_format.line_spacing = 1.25
    normal.paragraph_format.space_before = Pt(0)
    normal.paragraph_format.space_after = Pt(0)


def add_page_number_footer(doc, prefix=""):
    footer = doc.sections[0].footer
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if prefix:
        r = p.add_run(prefix)
        set_run_font(r, 9, color="898781")
    run = p.add_run()
    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = " PAGE "
    fld_sep = OxmlElement("w:fldChar")
    fld_sep.set(qn("w:fldCharType"), "separate")
    txt = OxmlElement("w:t")
    txt.text = "1"
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    run._r.append(fld_begin)
    run._r.append(instr)
    run._r.append(fld_sep)
    run._r.append(txt)
    run._r.append(fld_end)
    set_run_font(run, 9, color="898781")


def add_paragraph(doc, text="", size=10.5, bold=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY,
                  first_indent=True, line=1.25, before=0, after=0, east_font=BODY_EAST_FONT,
                  ascii_font=BODY_ASCII_FONT, color=None, keep_with_next=False):
    p = doc.add_paragraph()
    p.alignment = align
    pf = p.paragraph_format
    pf.line_spacing = line
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    if first_indent:
        pf.first_line_indent = Pt(size * 1.75)
    else:
        pf.first_line_indent = Pt(0)
    pf.keep_with_next = keep_with_next
    if text:
        r = p.add_run(text)
        set_run_font(r, size=size, bold=bold, east_font=east_font, ascii_font=ascii_font, color=color)
    return p


def add_rich_paragraph(doc, parts, size=10.5, align=WD_ALIGN_PARAGRAPH.JUSTIFY, first_indent=True,
                       line=1.25, before=0, after=0):
    """parts: list of (text, bold) tuples."""
    p = doc.add_paragraph()
    p.alignment = align
    pf = p.paragraph_format
    pf.line_spacing = line
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    pf.first_line_indent = Pt(size * 1.75) if first_indent else Pt(0)
    for text, bold in parts:
        r = p.add_run(text)
        set_run_font(r, size=size, bold=bold)
    return p


def add_h1(doc, text):
    p = doc.add_paragraph(style="List Paragraph")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pf = p.paragraph_format
    pf.space_before = Pt(10)
    pf.space_after = Pt(6)
    pf.line_spacing = 1.25
    pf.first_line_indent = Pt(0)
    pf.keep_with_next = True
    r = p.add_run(text)
    set_run_font(r, 14, bold=True, east_font=HEAD_FONT)
    return p


def add_h2(doc, text):
    p = doc.add_paragraph(style="List Paragraph")
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    pf = p.paragraph_format
    pf.space_before = Pt(8)
    pf.space_after = Pt(4)
    pf.line_spacing = 1.25
    pf.first_line_indent = Pt(0)
    pf.keep_with_next = True
    r = p.add_run(text)
    set_run_font(r, 12, bold=True, east_font=HEAD_FONT)
    return p


def add_h3(doc, text):
    p = doc.add_paragraph(style="List Paragraph")
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    pf = p.paragraph_format
    pf.space_before = Pt(6)
    pf.space_after = Pt(3)
    pf.line_spacing = 1.25
    pf.first_line_indent = Pt(0)
    pf.keep_with_next = True
    r = p.add_run(text)
    set_run_font(r, 12, bold=True, east_font=HEAD_FONT)
    return p


def add_caption(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pf = p.paragraph_format
    pf.space_before = Pt(4)
    pf.space_after = Pt(6)
    pf.line_spacing = 1.0
    pf.first_line_indent = Pt(0)
    pf.keep_with_next = text.startswith("表")
    r = p.add_run(text)
    set_run_font(r, 9.5)
    return p


def add_bullet(doc, text, size=10.5, symbol="•"):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    pf = p.paragraph_format
    pf.line_spacing = 1.25
    pf.space_before = Pt(0)
    pf.space_after = Pt(0)
    pf.left_indent = Pt(21)
    pf.first_line_indent = Pt(-12)
    r = p.add_run(f"{symbol} {text}")
    set_run_font(r, size=size)
    return p


def add_figure(doc, path, width_in=5.6, caption=None):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.first_line_indent = Pt(0)
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(path), width=Inches(width_in))
    if caption:
        add_caption(doc, caption)


def _shade_cell(cell, fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    tcPr.append(shd)


def _set_cell_margins(table, top=40, bottom=40, left=70, right=70):
    tblPr = table._tbl.tblPr
    mar = OxmlElement("w:tblCellMar")
    for side, value in (("top", top), ("bottom", bottom), ("left", left), ("right", right)):
        el = OxmlElement(f"w:{side}")
        el.set(qn("w:w"), str(value))
        el.set(qn("w:type"), "dxa")
        mar.append(el)
    tblPr.append(mar)


def add_table(doc, headers, rows, widths_in=None, font_size=9, caption=None,
              header_fill="EDF2F9", align_center=True):
    if caption:
        add_caption(doc, caption)
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    _set_cell_margins(table)

    hdr = table.rows[0]
    trPr = hdr._tr.get_or_add_trPr()
    tblHeader = OxmlElement("w:tblHeader")
    tblHeader.set(qn("w:val"), "true")
    trPr.append(tblHeader)

    for i, h in enumerate(headers):
        cell = hdr.cells[i]
        cell.text = ""
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.line_spacing = 1.0
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(0)
        r = p.add_run(str(h))
        set_run_font(r, font_size, bold=True)
        _shade_cell(cell, header_fill)
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER

    for row in rows:
        cells = table.add_row().cells
        for i, val in enumerate(row):
            cell = cells[i]
            cell.text = ""
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if align_center else WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.line_spacing = 1.0
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            r = p.add_run(str(val))
            set_run_font(r, font_size)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER

    for ri, row in enumerate(table.rows):
        trPr = row._tr.get_or_add_trPr()
        cant = OxmlElement("w:cantSplit")
        trPr.append(cant)
        if ri < len(table.rows) - 1:
            for cell in row.cells:
                for p in cell.paragraphs:
                    p.paragraph_format.keep_with_next = True

    if widths_in:
        for i, w in enumerate(widths_in):
            for row in table.rows:
                row.cells[i].width = Inches(w)
    return table


# --------------------------------------------------------------------------- #
# OMML equation helpers
# --------------------------------------------------------------------------- #
def mr(text: str, italic: bool = True, bold: bool = False) -> str:
    """A math run.  Set italic=False for Chinese labels / operators."""
    rpr = "" if italic else '<m:rPr><m:sty m:val="p"/></m:rPr>'
    b = "<w:b/>" if bold else '<w:b w:val="0"/>'
    return (
        f'<m:r>{rpr}<w:rPr><w:rFonts w:ascii="Cambria Math" w:hAnsi="Cambria Math" '
        f'w:eastAsia="宋体"/>{b}<w:sz w:val="24"/></w:rPr>'
        f'<m:t xml:space="preserve">{text}</m:t></m:r>'
    )


def mfrac(num: str, den: str) -> str:
    return f"<m:f><m:num>{num}</m:num><m:den>{den}</m:den></m:f>"


def msup(base: str, sup: str) -> str:
    return f"<m:sSup><m:e>{base}</m:e><m:sup>{sup}</m:sup></m:sSup>"


def msub(base: str, sub: str) -> str:
    return f"<m:sSub><m:e>{base}</m:e><m:sub>{sub}</m:sub></m:sSub>"


def add_equation(doc, body_xml: str):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pf = p.paragraph_format
    pf.space_before = Pt(4)
    pf.space_after = Pt(4)
    pf.line_spacing = 1.15
    pf.first_line_indent = Pt(0)
    xml = (
        f'<m:oMathPara {nsdecls("m", "w")}>'
        f'<m:oMathParaPr><m:jc m:val="center"/></m:oMathParaPr>'
        f"<m:oMath>{body_xml}</m:oMath></m:oMathPara>"
    )
    p._p.append(parse_xml(xml))
    return p


def page_break(doc):
    doc.add_page_break()
