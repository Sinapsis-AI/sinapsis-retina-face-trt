<h1 align="center">
<br>
<br>
<a href="https://sinapsis.tech/">
  <img
    src="https://github.com/Sinapsis-AI/brand-resources/blob/main/sinapsis_logo/4x/logo.png?raw=true"
    alt="" width="300">
</a>
<br>
Sinapsis Retina Face TRT
<br>
</h1>

<h4 align="center">Templates for real-time facial recognition with RetinaFace and DeepFace.</h4>

<p align="center">
<a href="#installation">🐍 Installation</a> •
<a href="#features">🚀 Features</a> •
<a href="#usage-example">📚 Example usage</a> •
<a href="#webapp">🌐 Webapp</a>
<a href="#documentation">📙 Documentation</a> •
<a href="#licence">🔍 License</a>
</p>

The `sinapsis-retina-face-trt` module provides templates for real-time facial recognition with RetinaFace and DeepFace, enabling efficient and accurate inference.

</details>
<h2 id="installation">🐍 Installation</h2>

> [!NOTE]
> CUDA-based templates in sinapsis-retina-face-trt require NVIDIA driver version to be 550 or higher.

Install using your package manager of choice. We encourage the use of `uv`

```bash
uv pip install sinapsis-retina-face-trt
```
or wiht raw pip
```bash
pip install sinapsis-retina-face-trt 
```

> [!IMPORTANT]
> To enable tensorflow with CUDA support please install `tensorflow` as follows:

```bash
uv pip install tensorflow[and-cuda]==2.18.0
```
or
```bash
pip install tensorflow[and-cuda]==2.18.0
```

> [!IMPORTANT]
> Templates in sinapsis-retina-face-trt package may require extra dependencies. For development, we recommend installing the package with all the optional dependencies:

```bash
uv pip install sinapsis-retina-face-trt[all] --extra-index-url https://pypi.sinapsis.tech
```
or
```bash
pip install sinapsis-retina-face-trt[all] --extra-index-url https://pypi.sinapsis.tech
```


</details>

<h2 id="features">🚀 Features</h2>

<h3> Templates Supported</h3>

The **Sinapsis Retina Face TRT** module provides multiple templates for real-time facial recognition, leveraging TensorRT optimization and DeepFace embedding search.


- **RetinaFacePytorch**. Runs face detection using RetinaFace implemented in PyTorch.
- **RetinaFacePytorchTRT**. A TensorRT-optimized version of **RetinaFacePytorch** for faster inference. 
- **RetinaFacePytorchTRTTorchOnly**. A Torch-TensorRT optimized version of RetinaFace, focusing solely on Torch-TRT acceleration. 
- **PytorchEmbeddingSearch**. Performs similarity search over a gallery of embeddings.
- **PytorchEmbeddingExtractor**. A base template for extracting embeddings from face images.
- **Facenet512EmbeddingExtractorTRT**. Uses TensorRT for fast embedding extraction based on **Facenet512**.
- **Facenet512EmbeddingExtractorTRTDev**. An alternative version of **Facenet512EmbeddingExtractorTRT** that converts the model at runtime. 
- **FaceVerificationFromGallery**. Perform face verification by direct comparison between predicted face embeddings and face embeddings stored in a gallery file.



<h2 id="usage-example">📚 Example usage</h2>

The following example demonstrates how to use the **RetinaFacePytorchTRT** template for real-time facial detection.

This configuration defines an **agent** and a sequence of **templates** to run real-time facial recognition with **RetinaFace**.

1. **Image Loading (`FolderImageDatasetCV2`)**: Loads images from the specified directory (`data_dir`).
2. **Face Detection (`RetinaFacePytorchTRT`)**: Runs inference using **RetinaFace**, applying a confidence threshold, model configuration, and pretrained weights.
3. **Bounding Box Drawing (`BBoxDrawer`)**: Overlays bounding boxes on detected faces.
4. **Saving Results (`ImageSaver`)**: Saves the processed images to the defined output directory.

   
<details>
  <summary id="docker"><strong><span style="font-size: 1.2em;">Config file</span></strong></summary>
   
```yaml
agent:
  name: face_detection
  description: > 
    Agent to perform face detection by employing an accelerated TRT version of the RetinaFace model.

templates:
- template_name: InputTemplate-1
  class_name: InputTemplate
  attributes: {}

- template_name: FolderImageDatasetCV2-1
  class_name: FolderImageDatasetCV2
  template_input: InputTemplate-1
  attributes:
    data_dir: /opt/app/datasets/vision/detection/lfw
    load_on_init : true
    samples_to_load : 1
    batch_size : 10

- template_name: RetinaFacePytorch-1
  class_name: RetinaFacePytorchTRT
  template_input: FolderImageDatasetCV2-1
  attributes:
    return_key_points: true
    confidence_threshold: 0.3
    local_model_path: "/root/.cache/torch/hub/checkpoints/resnet50_2020-07-20.engine"

- template_name: BBoxDrawer
  class_name: BBoxDrawer
  template_input: RetinaFacePytorch-1
  attributes:
    draw_boxes: true
    draw_key_points: true
    randomized_color: false

- template_name: ImageSaver
  class_name: ImageSaver
  template_input: BBoxDrawer
  attributes:
    save_dir: "examples/inference_results/"
    root_dir: ""
    extension: jpg
    save_full_image: true
    save_bbox_crops: false
```
</details>

To run the agent, you should run:

```bash
sinapsis run /path/to/sinapsis-retina-face-trt/src/sinapsis_retina_face_trt/configs/face_recognition.yml
``` 
</details>

<h2 id="webapp">🌐 Webapp</h2>

The webapps included in this repo provide interactive interfaces to showcase **real-time facial recognition** and **face verification mode** capabilities. 

> [!IMPORTANT]
> To run the apps, you first need to clone this repository:

```bash
git clone https://github.com/sinapsis-ai/sinapsis-retina-face-trt.git
```

> [!NOTE]
> The **face recognition** app requires a dataset of face images organized into folders, where each folder is named after the individual whose face images it contains. Example dataset structure:

```yaml
.
└── gallery/
    ├── person_1/
    │   ├── image_1
    │   ├── image_2
    │   ├── image_3
    │   └── image_4
    ├── person_2/
    │   ├── image_1
    │   ├── image_2
    │   ├── image_3
    │   └── image_4
    └── person_3/
        ├── image_1
        ├── image_2
        └── image_3
```
We have created a small version of the [lfw](http://vis-www.cs.umass.edu/lfw/) dataset in the following [link](https://cortezaai-my.sharepoint.com/:f:/g/personal/natalia_corteza_ai/EtiIJWdgdlNCgr3L4-gbeRIBsLNbl5GHdQrgPgNK-SDIXg?e=AYZ3Xp)

> [!NOTE]
> The **face verification** app don't require you to build a dataset. For demo purposes, the app is designed to perform face validation by using only one image as reference which should be provided through the app interface. 

> [!WARNING]
> If you have cached versions of the retinaface or Facenet models, please remove them before running the app. 
To remove cached versions, use (might need root permissions, in which case use sudo) 

<code> rm -rf ~/.cache/torch/hub/checkpoints/* && rm -rf ~/.cache/sinapsis/.deepface/weights/* </code>

> [!NOTE]
> If you'd like to enable external app sharing in Gradio use: 
`export GRADIO_SHARE_APP=True`

<details>
<summary id="docker"><strong><span style="font-size: 1.4em;">🐳 Docker</span></strong></summary>

1. Build the sinapsis-retina-face-trt image:
```bash
docker compose -f docker/compose.yaml build
```
2. Start the container:

For **face recognition app**, export the variable with the path to your gallery folder
```bash
export GALLERY_ROOT_DIR=/path/to/dataset/
```
and initialize app
```bash
docker compose -f docker/compose_apps.yaml up sinapsis-face-recognition-gradio -d
```

For **face verification app**
```bash
docker compose -f docker/compose_apps.yaml up sinapsis-verification-mode-gradio -d
```

3. Check the status:

For **face recognition app**
```bash
docker logs -f sinapsis-face-recognition-gradio
```

For **face verification app**
```bash
docker logs -f sinapsis-verification-mode-gradio 
```

4. The logs will display the URL to access the webapp:
```bash
Running on local URL:  http://127.0.0.1:7860
```
5. To stop the app:
```bash
docker compose -f docker/compose_apps.yaml down
```

</details>
<details>
<summary id="uv"><strong><span style="font-size: 1.4em;">📦 UV</span></strong></summary>

1. Create the virtual environment and sync the dependencies:

```bash
uv sync --frozen
```

2. Install the sinapsis-retina-face-trt package with all its dependencies:

```bash
uv pip install sinapsis-retina-face-trt[all] --extra-index-url https://pypi.sinapsis.tech
```

3. Install `tensorflow` with cuda support:

```bash
uv pip install tensorflow[and-cuda]==2.18.0
```

4. Run the webapp.

For **face recognition app**:

Update the following attributes in the [face_recognition](https://github.com/sinapsis-ai/sinapsis-retina-face-trt/blob/main/src/sinapsis_retina_face_trt/configs/face_recognition.yml) config file:

* `local_model_path` in the `RetinaFacePytorch-1` template,  to point to the torch hub cache local folder.
* `image_root_dir` in the `PytorchEmbeddingSearch-1` template,  to point to your local gallery folder.

then run:
```bash
uv run webapps/face_recognition_demo.py
```

For **face verification app**:

Update the `local_model_path` attributes of the `RetinaFacePytorch-1` template in the [face_verification](https://github.com/sinapsis-ai/sinapsis-retina-face-trt/blob/main/src/sinapsis_retina_face_trt/configs/face_verification.yml) config file to point to the torch hub cache local folder:

then run:
```bash
uv run webapps/verification_mode_demo.py
```

5. The terminal will display the URL to access the webapp:
```bash
Running on local URL:  http://127.0.0.1:7860
```
**NOTE**: The URL may vary; check the terminal output for the correct address.
</details>

<h2 id="documentation">📙 Documentation</h2>

Documentation is available on the [sinapsis website](https://docs.sinapsis.tech/docs)

Tutorials for different projects within sinapsis are available at [sinapsis tutorials page](https://docs.sinapsis.tech/tutorials)

<h2 id="license">🔍 License</h2>

This project is licensed under the AGPLv3 license, which encourages open collaboration and sharing. For more details, please refer to the [LICENSE](LICENSE) file.

For commercial use, please refer to our [official Sinapsis website](https://sinapsis.tech) for information on obtaining a commercial license.
