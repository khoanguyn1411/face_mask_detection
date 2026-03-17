from pathlib import Path
import runpy


def main() -> None:
    app_path = Path(__file__).resolve().parent / "app.py"
    runpy.run_path(str(app_path), run_name="__main__")


if __name__ == "__main__":
    main()