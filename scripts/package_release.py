"""Build a Linux installation archive; runtime environments are installed at destination."""
import argparse
from pathlib import Path
import tarfile

root = Path(__file__).resolve().parent.parent
parser = argparse.ArgumentParser()
parser.add_argument('--with-data', action='store_true', help='Include local projects, models, caches under data/')
parser.add_argument('--output', default=str(root/'dist/vits-linux-source.tar.gz'))
args = parser.parse_args()
output = Path(args.output).resolve()
output.parent.mkdir(parents=True, exist_ok=True)

def include(info):
    parts = Path(info.name).parts[1:]
    if root.joinpath(*parts).resolve() == output:
        return None
    if any(part in {'.git', '__pycache__', '.pytest_cache', 'runtime', 'cache', 'dist'} for part in parts):
        return None
    if parts and (parts[0] == 'data' and not args.with_data):
        return None
    if info.name.endswith(('session-cookies.txt', 'server.pid', 'server.log', 'server-error.log', 'upstream-tree.json')):
        return None
    if parts and parts[0] == 'deploy.env':
        return None  # The installer generates the target host's addresses and port settings.
    if info.name.endswith('.sh'):
        info.mode = 0o755
    return info

with tarfile.open(output, 'w:gz', compresslevel=1) as archive:
    archive.add(root, arcname='vits', filter=include)
print(output)
