import matplotlib.pyplot as plt
from wordcloud import WordCloud
from src.config import IMAGES_DIR
import re


def save_plot(fig, filename):
    """Saves a matplotlib figure to the images directory."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    path = IMAGES_DIR / filename
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {path}")


def generate_wordcloud(text_data, filename="wordcloud.png"):
    """Generates and saves a word cloud."""
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


def parse_deepseek_response(text):
    """Splits text into thinking process and final answer."""
    pattern = r"<think>(.*?)</think>"
    match = re.search(pattern, text, re.DOTALL)

    if match:
        thinking_process = match.group(1).strip()
        final_answer = re.sub(pattern, "", text, flags=re.DOTALL).strip()
        return thinking_process, final_answer

    return None, text
