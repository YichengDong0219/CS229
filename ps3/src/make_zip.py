"""Package your Python files and generated prediction text files."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

def make_zip():
    root = Path(__file__).resolve().parent
    target = root / 'submission.zip'
    with ZipFile(target, 'w', compression=ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob('*')):
            if path.is_file() and path.suffix in {'.py', '.txt'}:
                if '__pycache__' not in path.parts:
                    archive.write(path, path.relative_to(root))
    print(target)

if __name__ == '__main__':
    make_zip()
