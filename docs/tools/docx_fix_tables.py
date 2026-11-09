"""Give every table header row a thick bottom border in a pandoc .docx.

The template's table style also asks for this rule, but Google Docs ignores
conditional table-style formatting on import, so the header line disappears.
Writing the border directly on each header cell keeps it in Word and Drive.

Usage: python3 docs/tools/docx_fix_tables.py main.docx [--compact-proposal]
"""

import argparse
import re
import zipfile
from pathlib import Path

# 1.5 pt black rule (sz is in eighths of a point).
RULE = (
    '<w:tcBorders><w:bottom w:val="single" w:sz="12" '
    'w:space="0" w:color="000000" /></w:tcBorders>'
)


def fix_row(row):
    row = row.replace("<w:tcPr />", f"<w:tcPr>{RULE}</w:tcPr>")
    return re.sub(r"<w:tcPr>(?!<w:tcBorders>)", f"<w:tcPr>{RULE}", row)


def fix(xml):
    count = 0

    def header(m):
        nonlocal count
        count += 1
        return fix_row(m.group(0))

    xml = re.sub(
        r'<w:tr><w:trPr><w:tblHeader w:val="on" /></w:trPr>.*?</w:tr>',
        header, xml, flags=re.S,
    )
    return xml, count


def compact_styles(xml):
    """Reduce proposal whitespace without changing fonts or document text."""
    spacing = {
        "Title": (0, 80),
        "Author": (0, 80),
        "Heading1": (80, 20),
        "BodyText": (0, 20),
        "Compact": (0, 0),
    }
    for style_id, (before, after) in spacing.items():
        pattern = rf'<w:style\b[^>]*w:styleId="{style_id}".*?</w:style>'
        match = re.search(pattern, xml, re.S)
        if not match:
            raise ValueError(f"Missing proposal style: {style_id}")
        block = match.group(0)
        tag = f'<w:spacing w:before="{before}" w:after="{after}" />'
        if re.search(r"<w:spacing\b[^>]*/>", block):
            block = re.sub(r"<w:spacing\b[^>]*/>", tag, block)
        else:
            block = block.replace("<w:pPr>", "<w:pPr>" + tag, 1)
        xml = xml[:match.start()] + block + xml[match.end():]
    return xml


def separate_references(xml):
    """Keep the AI disclosure and bibliography after the proposal body."""
    # A 0.75-inch bottom margin keeps the complete team table on the body page.
    xml = re.sub(
        r'(<w:pgMar\b[^>]*\bw:bottom=")\d+("[^>]*/>)',
        r'\g<1>1080\2', xml,
    )

    def paragraph(match):
        block = match.group(0)
        if "Generative-AI disclosure." in block and "<w:pageBreakBefore" not in block:
            block = block.replace("<w:pPr>", "<w:pPr><w:pageBreakBefore />", 1)
        return block

    return re.sub(r"<w:p>.*?</w:p>", paragraph, xml, flags=re.S)


def main(path, compact_proposal=False):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    with zipfile.ZipFile(path) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "word/document.xml":
                text, count = fix(data.decode("utf-8"))
                if compact_proposal:
                    text = separate_references(text)
                data = text.encode("utf-8")
            elif compact_proposal and item.filename == "word/styles.xml":
                data = compact_styles(data.decode("utf-8")).encode("utf-8")
            zout.writestr(item, data)
    tmp.replace(path)
    print(f"{path}: thick rule under {count} table header rows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    parser.add_argument("--compact-proposal", action="store_true")
    args = parser.parse_args()
    main(args.path, args.compact_proposal)
