from __future__ import annotations

from pathlib import Path
import runpy


APP_PATH = Path(__file__).with_name("streamlit_app.py.py")


if __name__ == "__main__":
    runpy.run_path(str(APP_PATH), run_name="__main__")
