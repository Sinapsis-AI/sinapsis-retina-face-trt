# -*- coding: utf-8 -*-

import gradio as gr
from sinapsis.webapp.agent_gradio_helper import add_logo_and_title, css_header, init_image_inference
from sinapsis_core.utils.env_var_keys import AGENT_CONFIG_PATH, GRADIO_SHARE_APP

CONFIG_FILE = AGENT_CONFIG_PATH or "src/sinapsis_retina_face_trt/configs/face_recognition.yml"


def demo()-> gr.Blocks:
    with gr.Blocks(css=css_header()) as demo:
        add_logo_and_title("Sinapsis RetinaFace TensorRT")
        init_image_inference(CONFIG_FILE, "", True, app_message="Allow access to your camera and hit the record button to allow live inference.")
    return demo


if __name__ == "__main__":

    live_interface = demo()
    live_interface.launch(share=GRADIO_SHARE_APP)
