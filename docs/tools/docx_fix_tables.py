"""Give every table header row a thick bottom border in a pandoc .docx.

The template's table style also asks for this rule, but Google Docs ignores
conditional table-style formatting on import, so the header line disappears.
Writing the border directly on each header cell keeps it in Word and Drive.

Usage: python3 docs/tools/docx_fix_tables.py main.docx
"""

import re
import sys
import zipfile
from pathlib import Path

# 1.5 pt black rule (sz is in eighths of a point).
RULE = '<w:tcBorders><w:bottom w:val="single" w:sz="12" w:space="0" w:color="000000" /></w:tcBorders>'


def fix_row(row):
    row = row.replace("<w:tcPr />", f"<w:tcPr>{RULE}</w:tcPr>")
    return re.sub(r"<w:tcPr>(?!<w:tcBorders>)", f"<w:tcPr>{RULE}", row)


def fix(xml):
    count = 0

    def header(m):
        nonlocal count
        count += 1
        return fix_row(m.group(0))

    xml = re.sub(r'<w:tr><w:trPr><w:tblHeader w:val="on" /></w:trPr>.*?</w:tr>', header, xml, flags=re.S)
    return xml, count


def main(path):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    with zipfile.ZipFile(path) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "word/document.xml":
                text, count = fix(data.decode("utf-8"))
                data = text.encode("utf-8")
            zout.writestr(item, data)
    tmp.replace(path)
    print(f"{path}: thick rule under {count} table header rows")


if __name__ == "__main__":
    main(sys.argv[1])
