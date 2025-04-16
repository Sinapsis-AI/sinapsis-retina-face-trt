<h1 align="center">
<br>
<a href="https://github.com/Corteza-ai/sinapsis"><img src="imgs/sinapsis_logo.png" alt="Sinapsis" width="300"></a>
<br>
Sinapsis Retina Face TRT
<br>
</h1>

<h4 align="center">Templates for real-time facial recognition with RetinaFace and DeepFace.</h4>

<p align="center">
<a href="#requirements">⚙️ Requirements</a> •
<a href="#installation">📥 Installation</a> •
<a href="#features">🚀 Features</a> •
<a href="#webapp">🌐 Webapp</a>
</p>

The `sinapsis-retina-face-trt` module provides templates for real-time facial recognition with RetinaFace and DeepFace, enabling efficient and accurate inference.

<h2 id="requirements">⚙️ Requirements</h2>

Before using **Sinapsis Retina Face TRT**, ensure you have the necessary dependencies installed.
<h3>General Requirements</h3>


These dependencies are required regardless of the installation method:
- **Git with SSH enabled**: Necessary for cloning repositories securely.
- **NVIDIA drivers 550+**: Required for GPU acceleration with CUDA-based templates.
<details>
<summary id="docker"><strong><span style="font-size: 1.4em;">Additional Requirements for Running with UV</span></strong></summary>


If you plan to use a virtual environment (**UV**), you also need:
- **Python 3.10**: Required for running the templates.
- **`uv` package manager**: For managing dependencies, see [official installation guide](https://docs.astral.sh/uv/getting-started/installation/#installation-methods).
</details>
<details>
<summary id="docker"><strong><span style="font-size: 1.4em;">Additional Requirements for Running with Docker</span></strong></summary>

If you prefer a containerized setup with **Docker**, you also need:
- **Docker**: Follow the [official Docker installation guide](https://www.docker.com/get-started/).
- **NVIDIA Container Toolkit**: Needed for GPU support in Docker. Follow the [installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
- **sinapsis-nvidia image**: Follow the **Sinapsis [README](https://github.com/Corteza-ai/sinapsis)** for instructions on setting up the required **sinapsis-nvidia** image.
</details>
<h2 id="installation">📥 Installation</h2>
1. Clone this repo:

```bash
git clone git@github.com:Corteza-ai/sinapsis-retina-face-trt.git
cd sinapsis-retina-face-trt
```
To set up **Sinapsis Retina Face TRT**, you can choose between two installation methods: **Docker** for containerized execution or **UV** for managing dependencies in a virtual environment. Follow the steps below based on your preferred setup.

<details>
<summary id="docker"><strong><span style="font-size: 1.4em;">🐳 Docker</span></strong></summary>



2. Build the sinapsis-retina-face-trt image:
```bash
docker compose -f docker/compose.yaml build
```

</details>
<details>
<summary id="uv"><strong><span style="font-size: 1.4em;">📦 UV</span></strong></summary>


2. Create the virtual environment and sync the dependencies:

```bash
uv sync --frozen --all-extras
```

3. Build and install the wheel:

```bash
uv build
uv pip install dist/sinapsis*-none-any.whl
```

</details>

<h2 id="features">🚀 Features</h2>

<h3> Templates Supported</h3>

The **Sinapsis Retina Face TRT** module provides multiple templates for real-time facial recognition, leveraging TensorRT optimization and DeepFace embedding search.
<details>
<summary><strong><span style="font-size: 1em;">List of templates</span></strong></summary>
<details>
<summary><strong><span style="font-size: 1em;">RetinaFacePytorch</span></strong></summary>

Runs face detection using RetinaFace implemented in PyTorch.

- **`cuda` (bool, optional)**: Whether to use GPU (`True`) or CPU (`False`). Defaults to `True`.
- **`return_key_points` (bool, optional)**: Whether to return facial key points in the annotations. Defaults to `True`.
- **`confidence_threshold` (float, optional)**: Confidence threshold for detections. Defaults to 0.7.
- **`nms_threshold` (float, optional)**: Non-maximum suppression threshold. Defaults to `0.4`
- **`face_class_id` (int, optional)**: The class ID assigned to detected faces. Defaults to `1`.
- **`height` (int, optional)**: Maximum height for the resizing of images. Defaults to `960`.
- **`width` (int, optional)**: Maximum width for the resizing of images. Defaults to `960`.
- **`model_name` (str, optional)**: Name of the pre-trained model. Defaults to `resnet50_2020-07-20`.

</details>
<details>
<summary><strong><span style="font-size: 1em;">RetinaFacePytorchTRT</span></strong></summary>

A TensorRT-optimized version of **RetinaFacePytorch** for faster inference. Inherits all attributes from **RetinaFacePytorch**, introducing two additional attributes:
- **`force_compilation` (bool, optional)**: Whether to force model compilation. Defaults to `False`.
- **`local_model_path` (str, optional)**: Path to the locally saved TRT model. Defaults to `None`.
</details>

<details>
<summary><strong><span style="font-size: 1em;">RetinaFacePytorchTRTTorchOnly</span></strong></summary>

A Torch-TensorRT optimized version of RetinaFace, focusing solely on Torch-TRT acceleration. Inherits all attributes from **RetinaFacePytorchTRT**.
</details>

<details>
<summary><strong><span style="font-size: 1em;">PytorchEmbeddingSearch</span></strong></summary>

Performs similarity search over a gallery of embeddings.

- **`gallery_file` (str, required)**: Path to the gallery folder.
- **`similarity_threshold` (float, optional)**: Threshold for determining similar embeddings. Defaults to `200.0`.
- **`k_value` (int, optional)**: Number of matches to return. Defaults to `3`.
- **`metric` (str, optional)**: Distance metric (`cosine` or `euclidean`). Defaults to `cosine`.
- **`device` (str, optional)**: Device for computation (`cuda` or `cpu`). Defaults to `cuda`.
- **`force_build_from_dir` (bool, optional)**: Whether to rebuild the gallery from a directory. Defaults to `False`.
- **`model_to_use` (str, optional)**: Model used for embedding extraction. Defaults to `Facenet512EmbeddingExtractorTRTDev`.
- **`image_root_dir` (str, optional)**: Root directory for images. Defaults to `None`.
- **`model_kwargs` (dict, optional)**: Additional model parameters. Defaults to `{}`.
</details>

<details>
<summary><strong><span style="font-size: 1em;">PytorchEmbeddingExtractor</span></strong></summary>

A base template for extracting embeddings from face images.

- **`from_bbox_crop` (bool, optional)**: Whether to infer the embedding from the bbox or full image. Defaults to `False`.
- **`force_compilation` (bool, optional)**: Whether to force model compilation. Defaults to `False`.
- **`deep_copy_image` (bool, optional)**: Whether to make a deep copy of the input image. Defaults to `True`.
</details>

<details>
<summary><strong><span style="font-size: 1em;">Facenet512EmbeddingExtractorTRT</span></strong></summary>

Uses TensorRT for fast embedding extraction based on **Facenet512**. Inherits attributes from **PytorchEmbeddingExtractor**, plus three additional attributes:
- **`model_local_path` (str, required)**: Path to the locally stored TRT model.
- **`model_name` (str, optional)**: Name of the model. Defaults to `Facenet512`.
- **`input_shape` (int, optional)**: Expected input shape of the model. Defaults to `160`.
</details>

<details>
<summary><strong><span style="font-size: 1em;">Facenet512EmbeddingExtractorTRTDev</span></strong></summary>

An alternative version of **Facenet512EmbeddingExtractorTRT** that converts the model at runtime. Inherits attributes from **PytorchEmbeddingExtractor**, plus one additional attribute:
- **`model_name` (str, optional)**: Name of the model. Defaults to `Facenet512`.
</details>
</details>

<h3> Usage example</h3>

The following example demonstrates how to use the **RetinaFacePytorchTRT** template for real-time facial detection.

This configuration defines an **agent** and a sequence of **templates** to run real-time facial recognition with **RetinaFace**.

1. **Image Loading (`FolderImageDatasetCV2`)**: Loads images from the specified directory (`data_dir`).
2. **Face Detection (`RetinaFacePytorchTRT`)**: Runs inference using **RetinaFace**, applying a confidence threshold, model configuration, and pretrained weights.
3. **Bounding Box Drawing (`BBoxDrawer`)**: Overlays bounding boxes on detected faces.
4. **Saving Results (`ImageSaver`)**: Saves the processed images to the defined output directory.

   
<details>
  <summary id="docker"><strong><span style="font-size: 1.4em;">Config file</span></strong></summary>
   
```yaml
agent:
  name: face_detection

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



The webapp provides an interactive interface to showcase real-time facial recognition capabilities. By default, it runs with a confidence threshold of `0.3` and employs TensorRT acceleration.
The app requires a dataset with images of people, divided in folders with the names of the people on it:

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

NOTE: If you have cached versions of the retinaface or Facenet models, please remove them before running the app. 

To remove cached versions, use (might need root permissions, in which case use sudo) 

<code> rm -rf ~/.cache/torch/hub/checkpoints/* && rm -rf ~/.cache/sinapsis/.deepface/weights/* </code>

**NOTE: If you'd like to enable external app sharing in Gradio, `export GRADIO_SHARE_APP=True`**

<details>
<summary id="docker"><strong><span style="font-size: 1.4em;">🐳 Docker</span></strong></summary>

1. Export the variable with the path to your gallery folder:
```bash
export GALLERY_ROOT_DIR=/path/to/dataset/
```
2. Start the container:
```bash
docker compose -f docker/compose_apps.yaml up sinapsis-retina-face-gradio -d
```
3. Check the status:
```bash
docker logs -f sinapsis-retina-face-gradio
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

1. Please update the following attributes in the [face_recognition.yml](https://github.com/Corteza-ai/sinapsis-retina-face-trt/blob/main/src/sinapsis_retina_face_trt/configs/face_recognition.yml) file:

`image_root_dir` in the `PytorchEmbeddingSearch-1` template,  to point to your local gallery folder

`local_model_path` in the `RetinaFacePytorch-1` template,  to point to the torch hub cache local folder


```yaml
- template_name: PytorchEmbeddingSearch-1
  class_name: PytorchEmbeddingSearch
  template_input: RetinaFacePytorch-1
  attributes: 
    gallery_file: webapps/lfw_gallery.gallery
    image_root_dir: /path/to/dataset/
    model_kwargs:
      metadata:
        template_instance_name: PytorchEmbeddingSearch-1
        agent_name: face_recognition
```
2. Activate the environment:

```bash
source .venv/bin/activate
```
3. Run the webapp:
```bash
python webapps/gradio_live_demo.py
```
4. The terminal will display the URL to access the webapp:
```bash
Running on local URL:  http://127.0.0.1:7860
```

</details>
