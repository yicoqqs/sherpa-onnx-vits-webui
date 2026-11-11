"""Pin registry-installed Python packages; Conda packages are locked separately."""
from importlib import metadata
import json
from pathlib import Path
import re
import sys

normalize = lambda name: re.sub(r'[-_.]+', '-', name).lower()
conda = set()
for path in (Path(sys.prefix)/'conda-meta').glob('*.json'):
    conda.add(normalize(json.loads(path.read_text())['name']))
versions = {}
for distribution in metadata.distributions():
    name = distribution.metadata.get('Name')
    if name and normalize(name) not in conda:
        versions[normalize(name)] = distribution.version
for name, version in sorted(versions.items()):
    print(f'{name}=={version}')
