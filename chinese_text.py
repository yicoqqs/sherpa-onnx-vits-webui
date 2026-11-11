"""Normalize Chinese transcription to simplified script without rephrasing it."""
import re
from functools import lru_cache
from cn2an import an2cn
from opencc import OpenCC


@lru_cache(maxsize=1)
def converter():
    return OpenCC('t2s')


def to_simplified(text):
    return converter().convert(text)


def training_text(text):
    """Remove silent editorial brackets before phonemization; keep spoken text."""
    text = re.sub(r'([+-]?[0-9０-９]+(?:[.．][0-9０-９]+)?)\s*[%％]',
                  lambda match: '百分之' + an2cn(match.group(1).translate(str.maketrans('０１２３４５６７８９．', '0123456789.')).lstrip('+')), text)
    return to_simplified(text).translate(str.maketrans('', '', '《》〈〉「」『』【】（）“”‘’'))


def normalize_training_phones(phones, symbols):
    """Map equivalent punctuation only when the base lacks the source symbol."""
    pairs = {',':'，','.':'。','?':'？','!':'！','-':'—',':':'，',';':'，'}
    pairs.update({'，':',','。':'.','？':'?','！':'!','—':'-'})
    table = {ord(a):b for a,b in pairs.items() if a not in symbols and b in symbols}
    return phones.translate(table)


def punctuation_for_symbols(symbols):
    return ' '.join(c for c in dict.fromkeys('，。！？—…,.!?-~─') if c in symbols)


def simplify_project(project):
    changed = 0
    for clip in project.get('clips', []):
        original = clip.get('text', '')
        simplified = to_simplified(original)
        if simplified != original:
            clip.setdefault('original_text', original)
            clip['text'] = simplified
            changed += 1
    return changed
