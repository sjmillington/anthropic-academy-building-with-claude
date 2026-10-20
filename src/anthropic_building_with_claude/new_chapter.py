import json
import re
import sys
import uuid
from pathlib import Path

TEMPLATE = Path(__file__).parent / "chapter_template.ipynb"


def main() -> None:
    """Create a new chapter notebook from the template: `uv run new-chapter "Chapter title"`."""
    if len(sys.argv) < 2:
        sys.exit('Usage: uv run new-chapter "Chapter title"')

    title = " ".join(sys.argv[1:])
    filename = "".join(word[:1].upper() + word[1:] for word in re.findall(r"[A-Za-z0-9]+", title))
    path = Path.cwd() / "src" / "anthropic_building_with_claude" / "chapters" / f"{filename}.ipynb"

    if path.exists():
        sys.exit(f"{path.name} already exists")

    notebook = json.loads(TEMPLATE.read_text())
    for cell in notebook["cells"]:
        cell["id"] = uuid.uuid4().hex[:8]

    title_cell = notebook["cells"][0]
    title_cell["source"] = title_cell["source"].replace("{title}", title)

    path.write_text(json.dumps(notebook, indent=1) + "\n")
    print(f"Created {path.name}")
