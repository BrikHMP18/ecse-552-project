"""Build tools/reference.docx, the Word template used by `make docx`.

Starts from pandoc's default template and makes the Word copy look like the
NeurIPS PDF: Times New Roman, 10 pt body, black text except red team notes,
justified
paragraphs, US Letter with a 5.5 in text block.

Usage: python3 tools/make_reference_docx.py
"""

import re
import subprocess
import zipfile
from pathlib import Path

OUT = Path(__file__).with_name("reference.docx")
FONT = "Times New Roman"
RFONTS = f'<w:rFonts w:ascii="{FONT}" w:hAnsi="{FONT}" w:eastAsia="{FONT}" w:cs="{FONT}" />'

# Run properties per style: (size in half-points, bold).
HEADINGS = {
    "Title": (34, True),
    "Subtitle": (24, False),
    "Author": (20, False),
    "Heading1": (24, True),
    "Heading2": (22, True),
    "Heading3": (20, True),
    "Heading4": (20, True),
}
JUSTIFIED = ["BodyText", "FirstParagraph", "Compact", "Bibliography", "BlockText"]


def style_block(styles, style_id):
    m = re.search(rf'<w:style [^>]*w:styleId="{style_id}".*?</w:style>', styles, re.S)
    return m


def set_ppr(block, extra):
    """Append paragraph properties (e.g. justification) to a style block."""
    if "<w:pPr>" in block:
        return block.replace("<w:pPr>", "<w:pPr>" + extra, 1)
    if "<w:pPr />" in block:
        return block.replace("<w:pPr />", f"<w:pPr>{extra}</w:pPr>", 1)
    # Schema order puts pPr before rPr.
    anchor = "<w:rPr>" if "<w:rPr>" in block else "</w:style>"
    return block.replace(anchor, f"<w:pPr>{extra}</w:pPr>{anchor}", 1)


def fix_styles(styles):
    # One font everywhere: drop theme fonts, then force Times New Roman.
    styles = re.sub(r"<w:rFonts [^>]*/>", RFONTS, styles)
    # Black only: every explicit colour becomes black, theme colours removed.
    styles = re.sub(r"<w:color [^>]*/>", '<w:color w:val="000000" />', styles)
    # Default run: 10 pt Times; default paragraph: 6 pt after, single spacing.
    styles = re.sub(
        r"<w:rPrDefault>.*?</w:rPrDefault>",
        f'<w:rPrDefault><w:rPr>{RFONTS}<w:color w:val="000000" /><w:sz w:val="20" />'
        '<w:szCs w:val="20" /><w:lang w:val="en-US" /></w:rPr></w:rPrDefault>',
        styles,
        flags=re.S,
    )
    styles = re.sub(
        r"<w:pPrDefault>.*?</w:pPrDefault>",
        '<w:pPrDefault><w:pPr><w:spacing w:after="80" w:line="240" w:lineRule="auto" />'
        "</w:pPr></w:pPrDefault>",
        styles,
        flags=re.S,
    )
    for sid, (size, bold) in HEADINGS.items():
        m = style_block(styles, sid)
        if not m:
            continue
        rpr = (
            f'<w:rPr>{RFONTS}{"<w:b /><w:bCs />" if bold else ""}'
            f'<w:color w:val="000000" /><w:sz w:val="{size}" /><w:szCs w:val="{size}" /></w:rPr>'
        )
        block = re.sub(r"<w:rPr>.*?</w:rPr>", rpr, m.group(0), flags=re.S)
        if "<w:rPr>" not in block:
            block = block.replace("</w:style>", rpr + "</w:style>")
        block = re.sub(r"<w:spacing [^>]*/>", '<w:spacing w:before="180" w:after="60" />', block)
        styles = styles.replace(m.group(0), block)
    # Body paragraphs: no extra space before, 4 pt after (pandoc's default is 9 pt).
    for sid in JUSTIFIED:
        m = style_block(styles, sid)
        if m:
            block = re.sub(r"<w:spacing [^>]*/>", '<w:spacing w:before="0" w:after="80" />', m.group(0))
            styles = styles.replace(m.group(0), block)
    for sid in JUSTIFIED:
        m = style_block(styles, sid)
        if m and '<w:jc w:val="both"' not in m.group(0):
            styles = styles.replace(m.group(0), set_ppr(m.group(0), '<w:jc w:val="both" />'))
    # The only colour: red for notes the team still has to complete.
    styles = styles.replace(
        "</w:styles>",
        '<w:style w:type="character" w:customStyle="1" w:styleId="TeamNote">'
        '<w:name w:val="Team Note" /><w:basedOn w:val="DefaultParagraphFont" />'
        '<w:rPr><w:color w:val="FF0000" /></w:rPr></w:style>'
        '<w:style w:type="character" w:customStyle="1" w:styleId="TeamName">'
        '<w:name w:val="Team Name" /><w:basedOn w:val="DefaultParagraphFont" />'
        '<w:rPr><w:color w:val="E06666" /></w:rPr></w:style></w:styles>',
    )
    # Links (citations, URLs) black and not underlined, like the PDF's hidelinks.
    m = style_block(styles, "Hyperlink")
    if m:
        block = re.sub(r"<w:u [^>]*/>", "", m.group(0))
        styles = styles.replace(m.group(0), block)
    return styles


def fix_document(doc):
    # US Letter, 1.5 in side margins (5.5 in text width as in NeurIPS), 1 in top/bottom.
    sect = (
        '<w:sectPr><w:pgSz w:w="12240" w:h="15840" />'
        '<w:pgMar w:top="1440" w:right="2160" w:bottom="1440" w:left="2160" '
        'w:header="720" w:footer="720" w:gutter="0" /></w:sectPr>'
    )
    if "<w:sectPr" in doc:
        return re.sub(r"<w:sectPr.*?</w:sectPr>|<w:sectPr\s*/>", sect, doc, flags=re.S)
    return doc.replace("</w:body>", sect + "</w:body>")


def main():
    default = subprocess.run(
        ["pandoc", "--print-default-data-file", "reference.docx"],
        check=True,
        capture_output=True,
    ).stdout
    tmp = OUT.with_suffix(".tmp")
    tmp.write_bytes(default)
    with zipfile.ZipFile(tmp) as zin, zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "word/styles.xml":
                data = fix_styles(data.decode("utf-8")).encode("utf-8")
            elif item.filename == "word/document.xml":
                data = fix_document(data.decode("utf-8")).encode("utf-8")
            zout.writestr(item, data)
    tmp.unlink()
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
