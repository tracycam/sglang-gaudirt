"""Identical token inputs for the plain and cached/chunked service checks."""

import os


def feature_prompts(vocabulary):
    prefix_length = int(os.environ.get("GAUDIRT_FEATURE_PREFIX", "160"))
    suffix_length = int(os.environ.get("GAUDIRT_FEATURE_SUFFIX", "9"))
    assert prefix_length > 0 and suffix_length > 0 and vocabulary > 3
    prefix = [3 + (i * 37) % (vocabulary - 3) for i in range(prefix_length)]
    return [prefix + [3 + (j * 11 + i) % (vocabulary - 3) for i in range(suffix_length)]
            for j in (1, 2)]
