from pathlib import Path

from vanishcam.ui import run


if __name__ == "__main__":
    run(Path(__file__).resolve().parent / "models")
