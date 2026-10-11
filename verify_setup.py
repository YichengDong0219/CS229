"""Check dependencies, imports and input files without executing unfinished solutions."""
from pathlib import Path
import ast, gzip, hashlib, importlib.util, json, os, subprocess, sys, tempfile

def main():
    root = Path(__file__).resolve().parent
    import numpy as np
    import scipy
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    import ipykernel, jupyterlab, fitz
    print(f'Python {sys.version.split()[0]} | NumPy {np.__version__} | SciPy {scipy.__version__} | Matplotlib {matplotlib.__version__}')
    manifest = json.loads((root / 'resources.json').read_text())
    for item in manifest['datasets']:
        p = root / item['path']
        assert hashlib.sha256(p.read_bytes()).hexdigest() == item['sha256'], f'Missing/changed data: {p}'
        if p.suffix == '.tiff':
            with Image.open(p) as image:
                image.verify()
        elif p.suffix == '.csv':
            assert p.stat().st_size > 0
        elif p.name.endswith('.csv.gz'):
            with gzip.open(p, 'rt') as fh:
                assert fh.readline().strip(), f'Empty dataset: {p}'
    scripts = list(root.glob('ps*/src/**/*.py'))
    for p in scripts:
        ast.parse(p.read_text(), filename=str(p))
        if p.name == 'make_zip.py':
            continue
        code = ('import importlib.util; '
                's=importlib.util.spec_from_file_location("setup_check",' + repr(str(p)) + '); '
                'm=importlib.util.module_from_spec(s); s.loader.exec_module(m)')
        subprocess.run([sys.executable, '-B', '-c', code], cwd=p.parent,
                       env={**os.environ, 'MPLBACKEND': 'Agg', 'PYTHONPATH': str(p.parent)}, check=True)
    for item in manifest['question_assets']:
        p = root / item['path']
        assert hashlib.sha256(p.read_bytes()).hexdigest() == item['sha256'], f'Missing/changed question asset: {p}'
    for answer in manifest['answer_templates']:
        assert (root / answer).is_file()
    for ps in ('ps1','ps2','ps3'):
        p = root / ps / 'tex' / (ps + '.pdf')
        with fitz.open(p) as doc:
            text = '\n'.join(page.get_text() for page in doc)
            assert len(doc) > 0 and 'Answer:' not in text
    with tempfile.TemporaryDirectory() as d:
        plt.plot([0, 1], [0, 1]); plt.savefig(Path(d) / 'check.png'); plt.close()
    print(f'OK: {len(scripts)} Python files; {len(manifest["datasets"])} unchanged input files; 58 answer templates; 3 question PDFs.')
    print('Environment and input checks passed. Algorithm TODOs are intentionally unimplemented.')

if __name__ == '__main__':
    main()
