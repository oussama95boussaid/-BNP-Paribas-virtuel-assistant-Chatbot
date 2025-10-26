import re
import zipfile
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parent
ROBOTS_PATH = ROOT / "robots.txt"
SITEMAP_PATH = ROOT / "sitemap.xml"
OUTPUT_XLSX = ROOT / "Filtered_Sitemap_Allowed_Links.xlsx"


def read_robots_disallows(robots_text: str) -> list[str]:
    disallows: list[str] = []
    current_agents: list[str] = []
    in_star_block = False

    for raw_line in robots_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#'):
            continue
        # Split by first ':'
        parts = line.split(':', 1)
        if len(parts) != 2:
            continue
        key = parts[0].strip().lower()
        val = parts[1].strip()

        if key == 'user-agent':
            agent = val
            # Start of a new block (robots.txt semantics are tolerant of multiple lines)
            if not current_agents:
                current_agents = [agent]
            else:
                # Continuation of agents in same block
                current_agents.append(agent)
            # Track if this block includes '*'
            in_star_block = ('*' in current_agents)
        elif key == 'disallow':
            # Only capture rules from blocks that apply to '*'
            if in_star_block:
                # Some robots use empty Disallow to allow-all; ignore empty
                if val:
                    disallows.append(val)
        else:
            # Other directives are ignored for this task
            pass

        # If we encounter a blank line, it's a new group. We handle this implicitly
        # by resetting when we see next user-agent. That's sufficient here.

    return disallows


def pattern_to_regex(pattern: str) -> re.Pattern:
    # Convert a robots wildcard pattern into a regex anchored at the start of the path
    # Supports '*' wildcard and '$' end anchor (Google extensions)
    regex_parts: list[str] = ['^']
    for ch in pattern:
        if ch == '*':
            regex_parts.append('.*')
        elif ch == '$':
            regex_parts.append('$')
        else:
            regex_parts.append(re.escape(ch))
    regex_str = ''.join(regex_parts)
    # Compile as a raw match (case-sensitive by default)
    return re.compile(regex_str)


def build_disallow_regexes(disallow_rules: list[str]) -> list[re.Pattern]:
    regexes: list[re.Pattern] = []
    for rule in disallow_rules:
        try:
            regexes.append(pattern_to_regex(rule))
        except re.error:
            # Skip malformed rules gracefully
            continue
    return regexes


def is_disallowed(url: str, disallow_regexes: list[re.Pattern]) -> bool:
    parsed = urlparse(url)
    # Build the path including query (many robots implementations consider the full path + query)
    path = parsed.path or '/'
    if parsed.query:
        path = f"{path}?{parsed.query}"
    for rx in disallow_regexes:
        if rx.match(path):
            return True
    return False


def extract_sitemap_urls(xml_bytes: bytes) -> list[str]:
    # Parse sitemap and return list of <loc> URLs for urlset
    # Namespaces handling
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as e:
        raise SystemExit(f"Failed to parse sitemap.xml: {e}")

    ns = {
        'sm': 'http://www.sitemaps.org/schemas/sitemap/0.9'
    }

    urls: list[str] = []
    tag = root.tag
    # Strip namespace if present
    if '}' in tag:
        local = tag.split('}', 1)[1]
    else:
        local = tag

    if local == 'urlset':
        for loc in root.findall('.//sm:url/sm:loc', ns):
            if loc.text:
                urls.append(loc.text.strip())
    elif local == 'sitemapindex':
        # Index – we won't crawl sub-sitemaps (no network). Try to read <loc> if they are local files.
        # But for this task, we limit to main file only.
        for loc in root.findall('.//sm:sitemap/sm:loc', ns):
            if loc.text:
                urls.append(loc.text.strip())
    else:
        # Unknown root – attempt generic <loc>
        for loc in root.findall('.//loc'):
            if loc.text:
                urls.append(loc.text.strip())

    return urls


def write_minimal_xlsx(urls: list[str], out_path: Path) -> None:
    # Create a minimal XLSX with a single sheet and a header 'url'
    # using only the Python standard library.
    created = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')

    # XML content builders
    content_types = '''<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>'''

    rels_root = '''<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="/docProps/app.xml"/>
</Relationships>'''

    workbook = '''<?xml version="1.0" encoding="UTF-8"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets>
    <sheet name="AllowedURLs" sheetId="1" r:id="rId1"/>
  </sheets>
</workbook>'''

    workbook_rels = '''<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
</Relationships>'''

    # Build sheet XML with inline strings
    def cell_xml(r: int, c: int, text: str) -> str:
        # Convert 1-based column index to Excel column letter(s)
        name = ''
        x = c
        while x > 0:
            x, rem = divmod(x - 1, 26)
            name = chr(65 + rem) + name
        # Escape XML special characters in text
        esc = (text.replace('&', '&amp;')
                    .replace('<', '&lt;')
                    .replace('>', '&gt;'))
        return f'<c r="{name}{r}" t="inlineStr"><is><t>{esc}</t></is></c>'

    rows_xml = []
    # Header
    rows_xml.append(f'<row r="1">{cell_xml(1, 1, "url")}</row>')
    # Data
    for idx, url in enumerate(urls, start=2):
        rows_xml.append(f'<row r="{idx}">{cell_xml(idx, 1, url)}</row>')

    sheet = f'''<?xml version="1.0" encoding="UTF-8"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <sheetData>
    {''.join(rows_xml)}
  </sheetData>
</worksheet>'''

    core = f'''<?xml version="1.0" encoding="UTF-8"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:creator>BNP Client Assistant</dc:creator>
  <cp:lastModifiedBy>BNP Client Assistant</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{created}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{created}</dcterms:modified>
</cp:coreProperties>'''

    app = '''<?xml version="1.0" encoding="UTF-8"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>BNP Client Assistant</Application>
</Properties>'''

    with zipfile.ZipFile(out_path, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr('[Content_Types].xml', content_types)
        z.writestr('_rels/.rels', rels_root)
        z.writestr('xl/workbook.xml', workbook)
        z.writestr('xl/_rels/workbook.xml.rels', workbook_rels)
        z.writestr('xl/worksheets/sheet1.xml', sheet)
        z.writestr('docProps/core.xml', core)
        z.writestr('docProps/app.xml', app)


def main() -> None:
    robots_text = ROBOTS_PATH.read_text(encoding='utf-8', errors='ignore')
    disallow_rules = read_robots_disallows(robots_text)
    disallow_regexes = build_disallow_regexes(disallow_rules)

    xml_bytes = SITEMAP_PATH.read_bytes()
    urls = extract_sitemap_urls(xml_bytes)

    allowed = [u for u in urls if not is_disallowed(u, disallow_regexes)]

    OUTPUT_XLSX.parent.mkdir(parents=True, exist_ok=True)
    write_minimal_xlsx(allowed, OUTPUT_XLSX)
    print(f"Wrote {len(allowed)} URLs to {OUTPUT_XLSX}")


if __name__ == '__main__':
    main()
