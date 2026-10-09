"""Clean up bookmarks in a pandoc .docx for Google Docs.

Pandoc adds a bookmark to every heading and reference, named with a leading
underscore (e.g. `_fd1394...`). Word treats those as hidden and Google Docs drops
them on import, so citation links stop working; Google Docs also shows a marker
icon for every bookmark it keeps. This removes bookmarks no link points to and
gives the remaining ones (the reference-list entries) visible names (`ref_<n>`).

Usage: python3 tools/docx_fix_bookmarks.py main.docx
"""

import re
import sys
import zipfile
from pathlib import Path


def fix(xml):
    targets = set(re.findall(r'<w:hyperlink [^>]*w:anchor="([^"]+)"', xml))

    # Bookmark ids whose name is not a link target: drop start and end tags.
    unused = {
        bid
        for bid, name in re.findall(r'<w:bookmarkStart w:id="(\d+)" w:name="([^"]+)" />', xml)
        if name not in targets
    }
    xml = re.sub(
        r'<w:bookmark(?:Start|End) w:id="(\d+)"[^>]*/>',
        lambda m: "" if m.group(1) in unused else m.group(0),
        xml,
    )

    names = {}

    def visible(name):
        if not name.startswith("_"):
            return name
        if name not in names:
            names[name] = f"ref_{len(names) + 1}"
        return names[name]

    xml = re.sub(
        r'(<w:bookmarkStart [^>]*w:name=")([^"]+)(")',
        lambda m: m.group(1) + visible(m.group(2)) + m.group(3),
        xml,
    )
    xml = re.sub(
        r'(<w:hyperlink [^>]*w:anchor=")([^"]+)(")',
        lambda m: m.group(1) + visible(m.group(2)) + m.group(3),
        xml,
    )
    return xml, len(unused), len(names)


def main(path):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    with zipfile.ZipFile(path) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "word/document.xml":
                text, removed, kept = fix(data.decode("utf-8"))
                data = text.encode("utf-8")
            zout.writestr(item, data)
    tmp.replace(path)
    print(f"{path}: {removed} unused bookmarks removed, {kept} kept for citation links")


if __name__ == "__main__":
    main(sys.argv[1])
