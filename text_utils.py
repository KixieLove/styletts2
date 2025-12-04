# IPA Phonemizer: https://github.com/bootphon/phonemizer
_pad = "$"
_punctuation = ';:,.!?¡¿—–--…"«»“”’ '
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzáéíóúÁÉÍÓÚñÑüÜ'
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

# Export all symbols:
symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)

dicts = {}
for i in range(len((symbols))):
    dicts[symbols[i]] = i

import unicodedata, re
from text_utils import symbols, _pad  # keep your existing exports

dicts = {s: i for i, s in enumerate(symbols)}
TEXT_VOCAB_LIMIT = 178  # must match ASR embedding size

class TextCleaner:
    def __init__(self, dummy=None):
        self.word_index_dictionary = dicts
        self.pad_id = dicts.get(_pad, 0)

        # Normalize straight apostrophe to curly (low-id punctuation)
        self._premap = str.maketrans({
            "'": "’",         # ASCII apostrophe -> curly
            "\u00A0": " ",    # NBSP -> space
        })

    def _normalize(self, s: str) -> str:
        s = unicodedata.normalize('NFKC', s)
        s = s.replace('\u2011', '-')  # non-breaking hyphen
        s = s.replace('\u2013', '-')  # en dash
        return s.translate(self._premap)

    def __call__(self, text: str):
        text = self._normalize(text)
        out = []
        for ch in text:
            # try direct
            idx = self.word_index_dictionary.get(ch, None)

            # fallback: strip diacritics (á->a) if direct failed
            if idx is None:
                base = ''.join(c for c in unicodedata.normalize('NFKD', ch)
                               if unicodedata.category(c) != 'Mn')
                idx = self.word_index_dictionary.get(base, None)

            # final guard: drop anything that would exceed ASR vocab
            if idx is not None and idx < TEXT_VOCAB_LIMIT:
                out.append(idx)
            # else: silently skip
        return out
