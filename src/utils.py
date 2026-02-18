"""Visualization and text-processing utilities.

Provides helpers for saving Matplotlib figures, generating word clouds,
and parsing the ``<think>...</think>`` blocks from DeepSeek-R1 responses.
"""

import re
from typing import List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from wordcloud import WordCloud

from src.config import IMAGES_DIR


def save_plot(fig: Figure, filename: str) -> None:
    """Save a Matplotlib figure to the project images directory.

    Args:
        fig: The Matplotlib ``Figure`` object to persist.
        filename: Target filename (e.g. ``"wordcloud.png"``).

    Returns:
        None.  Side-effect: writes the figure to
        ``config.IMAGES_DIR / filename``.
    """
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    path = IMAGES_DIR / filename
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {path}")


def generate_wordcloud(text_data: List[str], filename: str = "wordcloud.png") -> None:
    """Generate a word cloud image and save it to disk.

    Args:
        text_data: List of text strings to combine into the word cloud.
        filename: Output filename within the images directory.

    Returns:
        None.  Side-effect: writes the word cloud image to
        ``config.IMAGES_DIR / filename``.
    """
    wc = WordCloud(width=800, height=400, background_color="white").generate(
        " ".join(text_data)
    )
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    # Save first
    save_plot(plt.gcf(), filename)
    # Re-display for notebook
    try:
        image_path = IMAGES_DIR / filename
        img = plt.imread(str(image_path))
        plt.figure(figsize=(10, 5))
        plt.imshow(img)
        plt.axis("off")
        plt.show()
    except Exception:
        pass


def parse_deepseek_response(
    text: str,
) -> Tuple[Optional[str], str]:
    """Split a DeepSeek-R1 response into thinking and final answer.

    DeepSeek-R1 often wraps its chain-of-thought reasoning inside
    ``<think>...</think>`` XML tags.  This function extracts the
    thinking block (if present) and returns the cleaned final answer.

    Args:
        text: The raw text returned by the DeepSeek-R1 model.

    Returns:
        A tuple of ``(thinking_process, final_answer)`` where
        *thinking_process* is ``None`` when no ``<think>`` block is
        found.
    """
    pattern: str = r"<think>(.*?)</think>"
    match = re.search(pattern, text, re.DOTALL)

    if match:
        thinking_process: str = match.group(1).strip()
        final_answer: str = re.sub(pattern, "", text, flags=re.DOTALL).strip()
        return thinking_process, final_answer

    return None, text
