import re
from typing import List
from tokenizers import Tokenizer
from pathlib import Path

def kmers(input: str, k:int = 3) -> List[str]:
    """Splits input string into list of overlapping kmers with step 1. Both words and url input are supported.
    Example:
        input: hello world
        output: ['hel', 'ell', 'llo', 'lo ', 'o w', ' wo', 'wor', 'orl', 'rld']

    Args:
        input (str): string to tokenize
        k (int, optional): token size. Defaults to 3.

    Returns:
        List[str]: list of kmers
    """

    # deal with url input
    input = _url_transform(input)

    kmers = zip(*[input[i:] for i in range(k)])
    return [''.join(kmer) for kmer in kmers]

def _url_transform(url:str) -> str:
    """Transforms input url into words format.
    Example:
        input: https://www.pneuboss.cz/pneu-255-45-r-19-104y-eagle_f1_asymmetric-xl-goodyear
        output: pneu 255 45 r 19 104y eagle f1 asymmetric xl goodyear
    Args:
        url (str): url to transform

    Returns:
        str: url transformed to words format
    """

    if url.startswith('https://'):

        # remove https://www. prefix
        url = url[12:]
        # remove e-shop domain
        url = url[url.index('/') + 1:]
        # transform url separators to word separator - space
        url = url.replace('-', ' ').replace('_', ' ')

    return url

def words(input:str) -> List[str]:
    """Splits input string into tokens. Splitting is done on spaces. Special characters and numbers are also separated from words.
    Both words and url input are supported.
    Example:
        input: PIRELLI P-ZERO (PZ4) SPORT 5
        output: ['PIRELLI', 'P', '-', 'ZERO', '(', 'PZ', '4', ')', 'SPORT', '5']

    Args:
        input (str): string to tokenize

    Returns:
        List[str]: list of tokens
    """

    # deal with url input
    input = _url_transform(input)

    words = re.split(r'(\W)', input)
    tokens = []

    for word in words:
        # split words to number and character parts
        tokens.extend(re.split(r'(\d+)', word))

    # remove empty tokens
    tokens = list(filter(lambda x: x.strip(), tokens))

    # TODO: do we want to keep special character as (, ), -, _, ... ?

    return tokens


def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate

@static_vars(tokenizer = Tokenizer.from_file(str(Path(__file__).parent / '..' / 'resources' / 'tokenizer-trained.json')))
def subwords(input:str) -> List[str]:
    return subwords.tokenizer.encode(input).tokens