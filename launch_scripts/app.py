import os
import tempfile
from pathlib import Path

import gradio as gr

from predict import predict_voice


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = BASE_DIR / "pretrained_models" / "model.ckpt"


def run_prediction(score_path: str | None, model_path: str | None):
    """Run voice prediction and return the path to the generated MEI file."""
    if not score_path:
        return None, "Please upload a score file first."

    model_to_use = (model_path or "").strip() or str(DEFAULT_MODEL_PATH)
    if not Path(model_to_use).exists():
        return None, f"Model checkpoint not found at: {model_to_use}"

    input_path = Path(score_path)
    tmp_dir = Path(tempfile.mkdtemp(prefix="svsep_pred_"))
    output_path = tmp_dir / f"{input_path.stem}_pred.musicxml"

    try:
        predict_voice(model_to_use, str(input_path), str(output_path))
    except Exception as exc:  # pragma: no cover - shown directly in UI
        return None, f"Error during prediction: {exc}"

    return str(output_path), f"Saved prediction to: {output_path}"


with gr.Blocks(title="Piano SVSep Voice Separation") as demo:
    gr.Markdown(
        "## Piano SVSep Voice Separation\n"
        "Upload a MusicXML/MEI score and get back an MEI file with predicted voices."
    )

    with gr.Row():
        with gr.Column():
            score_input = gr.File(
                label="Score file (.musicxml / .xml / .mei)",
                file_types=[".musicxml", ".xml", ".mei"],
                type="filepath",
            )
            model_input = gr.Textbox(
                label="Model checkpoint path",
                value=str(DEFAULT_MODEL_PATH),
                placeholder="Path to model.ckpt",
            )
            run_button = gr.Button("Predict voices")

        with gr.Column():
            output_file = gr.File(label="Predicted MusicXML (download)")
            status_box = gr.Textbox(label="Status", interactive=False)

    run_button.click(
        fn=run_prediction,
        inputs=[score_input, model_input],
        outputs=[output_file, status_box],
    )


if __name__ == "__main__":
    demo.launch()
