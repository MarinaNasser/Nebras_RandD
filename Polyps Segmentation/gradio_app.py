"""Interactive Gradio demo for the trained polyp-segmentation model."""

import argparse
import time
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch
import yaml

from inference_demo import annotate_contours, image_confidence, preprocess
from model import build_model


ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_model(config_path: Path, checkpoint_path: Path | None = None):
    with config_path.open("r", encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)

    model = build_model(
        cfg.get("model_variant", "modified"),
        num_classes=cfg.get("num_classes", 1),
        pretrained=False,
        segformer_size=cfg.get("segformer_size", "b0"),
    ).to(DEVICE)

    checkpoint = checkpoint_path or resolve_path(cfg["checkpoint_dir"]) / "best_model.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    state = torch.load(checkpoint, map_location=DEVICE, weights_only=False)
    # Also accept checkpoints that wrap the weights in a training-state dictionary.
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model, cfg, checkpoint


def create_demo(config_path: Path, checkpoint_path: Path | None = None) -> gr.Blocks:
    model, cfg, checkpoint = load_model(config_path, checkpoint_path)
    image_size = int(cfg["image_size"])

    def predict(image_rgb, detection_threshold, mask_threshold):
        if image_rgb is None:
            raise gr.Error("Upload an endoscopy image first.")

        started = time.perf_counter()
        image_rgb = np.asarray(image_rgb, dtype=np.uint8)
        if image_rgb.ndim == 2:
            image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2RGB)
        elif image_rgb.shape[2] == 4:
            image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_RGBA2RGB)

        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        height, width = image_bgr.shape[:2]
        tensor = preprocess(image_bgr, image_size).to(DEVICE)

        with torch.inference_mode():
            logits = model(tensor)
            probability = torch.sigmoid(logits)[0, 0].float().cpu().numpy()

        confidence = image_confidence(probability)
        probability = cv2.resize(probability, (width, height), interpolation=cv2.INTER_LINEAR)
        binary_mask = (probability >= float(mask_threshold)).astype(np.uint8)
        detected = confidence >= float(detection_threshold) and bool(binary_mask.any())

        annotated_bgr = annotate_contours(
            image_bgr, confidence, float(detection_threshold), binary_mask
        )
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        mask_rgb = np.repeat((binary_mask * 255)[..., None], 3, axis=2)
        heatmap_bgr = cv2.applyColorMap((probability * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

        elapsed_ms = (time.perf_counter() - started) * 1000
        status = "Polyp detected" if detected else "Uncertain / no polyp detected"
        summary = {
            "status": status,
            "confidence": round(confidence, 4),
            "confidence_percent": round(confidence * 100, 1),
            "positive_mask_percent": round(float(binary_mask.mean()) * 100, 2),
            "inference_ms": round(elapsed_ms, 1),
            "device": str(DEVICE),
        }
        return annotated_rgb, mask_rgb, heatmap_rgb, status, summary

    with gr.Blocks(title="Polyp Segmentation Inference") as demo:
        gr.Markdown(
            "# Polyp Segmentation Inference\n"
            "Upload an endoscopy frame to visualize the predicted polyp region. "
            "For research use only—not for clinical diagnosis."
        )
        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(type="numpy", label="Endoscopy image")
                detection_threshold = gr.Slider(
                    0.0, 1.0, value=0.85, step=0.01, label="Detection confidence threshold"
                )
                mask_threshold = gr.Slider(
                    0.0, 1.0, value=0.5, step=0.01, label="Mask probability threshold"
                )
                with gr.Row():
                    run_button = gr.Button("Run inference", variant="primary")
                    clear_button = gr.ClearButton()
                gr.Markdown(
                    f"**Model:** `{cfg.get('model_variant', 'modified')}` · "
                    f"**Input:** `{image_size}×{image_size}` · **Device:** `{DEVICE}`\n\n"
                    f"**Checkpoint:** `{checkpoint.name}`"
                )
            with gr.Column(scale=2):
                status = gr.Textbox(label="Result", interactive=False)
                with gr.Tabs():
                    with gr.Tab("Contour overlay"):
                        annotated = gr.Image(label="Prediction", interactive=False)
                    with gr.Tab("Binary mask"):
                        mask = gr.Image(label="Predicted mask", interactive=False)
                    with gr.Tab("Probability heatmap"):
                        heatmap = gr.Image(label="Pixel probabilities", interactive=False)
                details = gr.JSON(label="Inference details")

        outputs = [annotated, mask, heatmap, status, details]
        run_button.click(
            predict,
            inputs=[input_image, detection_threshold, mask_threshold],
            outputs=outputs,
            api_name="predict",
        )
        input_image.upload(
            predict,
            inputs=[input_image, detection_threshold, mask_threshold],
            outputs=outputs,
        )
        clear_button.add([input_image, *outputs])

    return demo


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create a temporary public Gradio link")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    app = create_demo(args.config.resolve(), args.checkpoint.resolve() if args.checkpoint else None)
    app.queue(default_concurrency_limit=1).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )
