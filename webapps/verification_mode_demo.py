# -*- coding: utf-8 -*-

from typing import Literal

import gradio as gr
import numpy as np
import torch
from pydantic.dataclasses import dataclass
from sinapsis.webapp.agent_gradio_helper import add_logo_and_title, css_header
from sinapsis_core.cli.run_agent_from_config import generic_agent_builder
from sinapsis_core.data_containers.data_packet import DataContainer, ImagePacket
from sinapsis_core.utils.env_var_keys import AGENT_CONFIG_PATH, GRADIO_SHARE_APP

CONFIG_FILE = AGENT_CONFIG_PATH or "src/sinapsis_retina_face_trt/configs/face_verification.yml"


@dataclass(frozen=True, slots=True)
class VerificationLabels:
    """Valid labels for face verification mode app."""

    verified: str = "VERIFIED"
    not_verified: str = "NOT-VERIFIED"


LABELS = VerificationLabels()
VerificationType = Literal[LABELS.verified, LABELS.not_verified]
agent = generic_agent_builder(CONFIG_FILE)


def get_embedding_from_container(container: DataContainer) -> torch.Tensor:
    """Extracts face embedding from image annotations.

    Args:
        container (DataContainer): DataContainer with image annotations embeddings.

    Returns:
        torch.Tensor: Extracted face embedding.
    """

    return container.images[0].annotations[0].embedding


def face_verification(reference_img: np.ndarray, input_img: np.ndarray, threshold: float) -> VerificationType:
    """Perform face verification by computing an embedding similarity score between reference and input image.

    Args:
        reference_img (np.ndarray):Image used as reference.
        input_img (np.ndarray): Image to be compared against reference image.
        threshold (float): Threshold for computed similarity score.

    Returns:
        str: Resulting verification label.
    """

    container_1 = agent(DataContainer(images=[ImagePacket(content=reference_img)]))
    container_2 = agent(DataContainer(images=[ImagePacket(content=input_img)]))

    embeddings_1 = get_embedding_from_container(container_1)
    embeddings_2 = get_embedding_from_container(container_2)

    dist = torch.nn.CosineSimilarity(dim=1)(embeddings_1, embeddings_2)

    if dist.max() > threshold:
        return LABELS.verified
    return LABELS.not_verified


def demo() -> gr.Blocks:
    with gr.Blocks(css=css_header()) as face_recognition_demo:
        add_logo_and_title("Sinapsis RetinaFace TensorRT")
        with gr.Row():
            with gr.Column():
                threshold = gr.Number(value=0.5, label="Similarity threshold")
                reference_img = gr.Image(label="Reference ID", sources=["upload", "clipboard"], type="numpy")

            with gr.Column():
                output = gr.Textbox(label="Result")
                input_img = gr.Image(label="Input image", type="numpy")
                submit_btn = gr.Button(value="Predict")

        submit_btn.click(
            face_verification, inputs=[reference_img, input_img, threshold], outputs=output, api_name=False
        )

    return face_recognition_demo


if __name__ == "__main__":
    live_interface = demo()
    live_interface.launch(share=GRADIO_SHARE_APP)
