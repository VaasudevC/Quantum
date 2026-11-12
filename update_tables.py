"""Replace <!--NAME-->...<!--/NAME--> blocks in BRICKWALL.md by the content of generated files."""
import re, sys
from pathlib import Path
def rd(f):
    b = Path(f).read_bytes()
    try: return b.decode('utf-8')
    except UnicodeDecodeError: return b.decode('cp1252')
BLOCKS = {"P3TABLES": "results/bw/p3_tables.md", "P3ANALYSIS": "results/bw/p3_analysis.md", "P4TABLE": "results/ext/p4_report.md",
          "P5TABLE": "results/lc/p5_table.md", "P6NOISE": "results/bw/p6_noise.md", "P6COMPILE": "results/bw/p6_compile.md"}
s = Path("BRICKWALL.md").read_text(encoding="utf-8")
for name, f in BLOCKS.items():
    if Path(f).exists() and f"<!--{name}-->" in s:
        s = re.sub(f"<!--{name}-->.*?<!--/{name}-->", lambda m: f"<!--{name}-->\n" + rd(f) + f"\n<!--/{name}-->", s, flags=re.S)
Path("BRICKWALL.md").write_text(s, encoding="utf-8")
