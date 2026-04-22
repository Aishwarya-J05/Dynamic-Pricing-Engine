from pathlib import Path
import compileall


def test_project_compiles() -> None:
    root = Path(__file__).resolve().parent.parent
    assert compileall.compile_dir(root / "app", quiet=1)
    assert compileall.compile_dir(root / "scripts", quiet=1)
