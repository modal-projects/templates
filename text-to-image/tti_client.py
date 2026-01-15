import time
from pathlib import Path

import modal

if __name__ == "__main__":
    """
    Generate an image using the deployed Modal function via `ImageGenerator`.

    Args:
        prompt (str): The text prompt to generate an image from.
        results_dir (str): Directory to save the generated image.
        image_name (str): Name of the generated image file (without extension).

    Example:
        python text_to_image.py
        # or with overrides:
        python text_to_image.py "A serene mountain landscape" "./outputs" "mountain_img"
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate an image from a text prompt."
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default="A princess riding on a pony",
        help="Text prompt to generate an image from.",
    )
    parser.add_argument(
        "results_dir",
        nargs="?",
        default="./bootstrap-results-text_to_image",
        help="Directory to save the generated image.",
    )
    parser.add_argument(
        "image_name",
        nargs="?",
        default="generated_image",
        help="Name for the generated image file (without extension).",
    )
    args = parser.parse_args()

    prompt = args.prompt
    results_dir = args.results_dir
    image_name = args.image_name

    generator = modal.Cls.from_name("text-to-image", "ImageGenerator")()

    start = time.perf_counter()
    image_bytes = generator.generate.remote(prompt, batch_size=1)
    print(f"Elapsed: {time.perf_counter() - start:.2f} seconds")

    output_dir = Path(results_dir)
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"{image_name}.png"
    output_path.write_bytes(image_bytes[0])

    print(f"✓ Image saved to: {output_path}")
    print("Running Stable Diffusion 3.5 Large Turbo on a cloud H100 GPU")
    print(f"Prompt: {prompt}")
