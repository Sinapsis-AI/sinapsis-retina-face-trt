<h1 align="center"><br/><br/><a href="https://sinapsis.tech/"><img alt="" src="https://github.com/Sinapsis-AI/brand-resources/blob/main/sinapsis_logo/4x/logo.png?raw=true" width="300"/></a><br/>Sinapsis Retina Face TRT
<br/></h1>

<h4 align="center">Plantillas para reconocimiento facial en tiempo real con RetinaFace y DeepFace.</h4>

<p align="center"><a href="#installation">🐍 Instalación</a> •
<a href="#features">🚀 Características</a> •
<a href="#usage-example">📚 Uso del ejemplo</a> •
<a href="#webapp">🌐 Aplicación Web</a><a href="#documentation">📙 Documentación</a> •
<a href="#licence">🔍 Licencia</a></p>

<p>El <code>sinapsis-retina-face-trt</code> El módulo proporciona plantillas para el reconocimiento facial en tiempo real con RetinaFace y DeepFace, lo que permite una inferencia eficiente y precisa.</p>

<p></p>

<h2 id="installation">🐍 Instalación</h2>

<blockquote><p>[! NOTA]
Las plantillas basadas en CUDA en sinapsis-retina-cara-trt requieren que la versión de controlador NVIDIA sea 550 o superior.</p></blockquote>

<p>Instala el administrador de tu paquete de elección. Alentamos el uso de <code>uv</code></p>

<p><pre><code class="bash language-bash">uv pip install sinapsis-retina-face-trt
</code></pre><p>or wiht raw pip</p><pre><code class="bash language-bash">pip install sinapsis-retina-face-trt
</code></pre><blockquote><p>[! IMPORTANTE]
Para habilitar tensorflow con soporte CUDA por favor instale <code>tensorflow</code> como sigue:</p></blockquote><pre><code class="bash language-bash">uv pip install tensorflow[and-cuda]==2.18.0
</code></pre><p>o</p><pre><code class="bash language-bash">pip install tensorflow[and-cuda]==2.18.0
</code></pre><blockquote><p>[! IMPORTANTE]
Las plantillas en el paquete sinapsis-retina-cara-trt pueden requerir dependencias adicionales. Para el desarrollo, recomendamos instalar el paquete con todas las dependencias opcionales:</p></blockquote><pre><code class="bash language-bash">uv pip install sinapsis-retina-face-trt[all] --extra-index-url https://pypi.sinapsis.tech
</code></pre><p>o</p><pre><code class="bash language-bash">pip install sinapsis-retina-face-trt[all] --extra-index-url https://pypi.sinapsis.tech
</code></pre><p></p><h2 id="features">🚀 Características</h2><h3> Plantillas apoyadas</h3><p>El <strong>Sinapsis Retina Face TRT</strong> El módulo proporciona múltiples plantillas para reconocimiento facial en tiempo real, optimización TensorRT y búsqueda de incrustación DeepFace.</p><ul><li><strong>RetinaFacePytorch</strong>. Ejecuta la detección de rostros usando RetinaFace implementado en PyTorch.</li><li><strong>RetinaFacePytorchTRT</strong>. Una versión optimizada de TensorRT <strong>RetinaFacePytorch</strong> para una inferencia más rápida. </li><li><strong>RetinaFacePytorchTRTTorchSólo</strong>. Una versión optimizada de Torch-TensorRT de RetinaFace, centrándose exclusivamente en la aceleración de Torch-TRT. </li><li><strong>PytorchEmbeddingSearch</strong>. Realiza búsqueda de similitud sobre una galería de embeddings.</li><li><strong>PytorchEmbeddingExtractor</strong>. Una plantilla base para extraer embeddings de imágenes faciales.</li><li><strong>Facenet512EmbeddingExtractorTRT</strong>. Utiliza TensorRT para la extracción rápida de incrustación basada en <strong>Facenet512</strong>.</li><li><strong>Facenet512EmbeddingExtractorTRTDev</strong>. Una versión alternativa <strong>Facenet512EmbeddingExtractorTRT</strong> que convierte el modelo a tiempo de ejecución. </li><li><strong>FaceVerification De Gallery</strong>. Realizar verificación facial mediante comparación directa entre las incrustaciones faciales predichas y las incrustaciones faciales almacenadas en un archivo galería.</li></ul><h2 id="usage-example">📚 Uso del ejemplo</h2><p>El siguiente ejemplo demuestra cómo utilizar el <strong>RetinaFacePytorchTRT</strong> plantilla para detección facial en tiempo real.</p><p>Esta configuración define un <strong>Agente</strong> y una secuencia de <strong>plantillas</strong> para ejecutar reconocimiento facial en tiempo real con <strong>RetinaFace</strong>.</p><ol><li><strong>Imagen Cargando (<code>FolderImageDatasetCV2</code>)</strong>: Carga imágenes del directorio especificado (<code>data_dir</code>).</li><li><strong>Detección facial (<code>RetinaFacePytorchTRT</code>)</strong>: Corre la inferencia usando <strong>RetinaFace</strong>, aplicando un umbral de confianza, configuración modelo y pesos preentrenados.</li><li><strong>Bounding Box ()<code>BBoxDrawer</code>)</strong>: Superpuestos de cajas en las caras detectadas.</li><li><strong>Resultados de ahorro (<code>ImageSaver</code>)</strong>: Guarda las imágenes procesadas al directorio de salida definido.</li></ol><p><details><summary id="docker"><strong><span style="font-size: 1.2em;">Archivo de Config</span></strong></summary></details></p><pre><code class="yaml language-yaml">agent:
  name: face_detection
  description: &amp;gt;
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
</code></pre><p></p><p>Para dirigir al agente, debe correr:</p><pre><code class="bash language-bash">sinapsis run /path/to/sinapsis-retina-face-trt/src/sinapsis_retina_face_trt/configs/face_recognition.yml
</code></pre><p></p><h2 id="webapp">🌐 Aplicación Web</h2><p>Las aplicaciones web incluidas en este repo proporcionan interfaces interactivas para mostrar <strong>reconocimiento facial en tiempo real</strong> y <strong>modo de verificación facial</strong> capacidades. </p><blockquote><p>[! IMPORTANTE]
Para ejecutar las aplicaciones, primero necesitas clonar este repositorio:</p></blockquote><pre><code class="bash language-bash">git clone https://github.com/sinapsis-ai/sinapsis-retina-face-trt.git
</code></pre><blockquote><p>[! NOTA]
El <strong>reconocimiento facial</strong> app requiere un conjunto de datos de imágenes faciales organizadas en carpetas, donde cada carpeta es nombrada por el individuo cuyas imágenes faciales contiene. Estructura de conjunto de datos:</p></blockquote><pre><code class="yaml language-yaml">.
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
</code></pre><p>Hemos creado una pequeña versión de la <a href="http://vis-www.cs.umass.edu/lfw/">lfw</a> dataset in the following <a href="https://cortezaai-my.sharepoint.com/:f:/g/personal/natalia_corteza_ai/EtiIJWdgdlNCgr3L4-gbeRIBsLNbl5GHdQrgPgNK-SDIXg?e=AYZ3Xp">enlace</a></p><blockquote><p>[! NOTA]
El <strong>verificación facial</strong> app no requiere que construya un conjunto de datos. Para fines de demostración, la aplicación está diseñada para realizar validación facial utilizando sólo una imagen como referencia que debe proporcionarse a través de la interfaz de aplicación. </p><p>[! ¡Ay!
Si tiene versiones en caché de los modelos de retinaface o Facenet, por favor retírelos antes de ejecutar la aplicación.
Para eliminar versiones en caché, utilice (necesita permisos de raíz, en cuyo caso utilice sudo) </p></blockquote><p><code> rm -rf ~/.cache/torch/hub/checkpoints/* &amp;amp;&amp;amp; rm -rf ~/.cache/sinapsis/.deepface/weights/* </code></p><blockquote><p>[! NOTA]
Si desea habilitar el intercambio de aplicaciones externas en el uso de Gradio:
 <code>export GRADIO_SHARE_APP=True</code></p></blockquote><p><details><summary id="docker"><strong><span style="font-size: 1.4em;">🐳 Docker</span></strong></summary></details></p><ol><li>Construir la imagen sinapsis-retina-cara-trt:</li></ol><pre><code class="bash language-bash">docker compose -f docker/compose.yaml build
</code></pre><ol start="2"><li>Iniciar el contenedor:</li></ol><p>Para <strong>aplicación de reconocimiento facial</strong>, exportar la variable con el camino a la carpeta de la galería</p><pre><code class="bash language-bash">export GALLERY_ROOT_DIR=/path/to/dataset/
</code></pre><p>e inicializar la aplicación</p><pre><code class="bash language-bash">docker compose -f docker/compose_apps.yaml up sinapsis-face-recognition-gradio -d
</code></pre><p>Para <strong>aplicación de verificación</strong></p><pre><code class="bash language-bash">docker compose -f docker/compose_apps.yaml up sinapsis-verification-mode-gradio -d
</code></pre><ol start="3"><li>Compruebe el estado:</li></ol><p>Para <strong>aplicación de reconocimiento facial</strong></p><pre><code class="bash language-bash">docker logs -f sinapsis-face-recognition-gradio
</code></pre><p>Para <strong>aplicación de verificación</strong></p><pre><code class="bash language-bash">docker logs -f sinapsis-verification-mode-gradio
</code></pre><ol start="4"><li>Los registros mostrarán la URL para acceder a la aplicación web:</li></ol><pre><code class="bash language-bash">Running on local URL:  http://127.0.0.1:7860
</code></pre><ol start="5"><li>Para detener la aplicación:</li></ol><pre><code class="bash language-bash">docker compose -f docker/compose_apps.yaml down
</code></pre><p><details><summary id="uv"><strong><span style="font-size: 1.4em;">📦 UV</span></strong></summary></details></p><ol><li>Crear el entorno virtual y sincronizar las dependencias:</li></ol><pre><code class="bash language-bash">uv sync --frozen
</code></pre><ol start="2"><li>Instale el paquete sinapsis-retina-cara-trt con todas sus dependencias:</li></ol><pre><code class="bash language-bash">uv pip install sinapsis-retina-face-trt[all] --extra-index-url https://pypi.sinapsis.tech
</code></pre><ol start="3"><li>Instala <code>tensorflow</code> con soporte de cuda:</li></ol><pre><code class="bash language-bash">uv pip install tensorflow[and-cuda]==2.18.0
</code></pre><ol start="4"><li>Corre la aplicación web.</li></ol><p>Para <strong>aplicación de reconocimiento facial</strong>:</p><p>Actualizar los siguientes atributos en los <a href="https://github.com/sinapsis-ai/sinapsis-retina-face-trt/blob/main/src/sinapsis_retina_face_trt/configs/face_recognition.yml">face<em>recognition</a> archivo config:</p><ul><li><code>local_model_path</code> en el <code>RetinaFacePytorch-1</code> plantilla, para apuntar a la carpeta local del centro de antorcha.</li><li><code>image_root_dir</code> en el <code>PytorchEmbeddingSearch-1</code> plantilla, para apuntar a su carpeta de galería local.</li></ul><p>entonces corre:</p><pre><code class="bash language-bash">uv run webapps/face_recognition_demo.py
</code></pre><p>Para <strong>aplicación de verificación</strong>:</p><p>Actualizar <code>local_model_path</code> atributos de los <code>RetinaFacePytorch-1</code> plantilla en el <a href="https://github.com/sinapsis-ai/sinapsis-retina-face-trt/blob/main/src/sinapsis_retina_face_trt/configs/face_verification.yml">face</em>verification</a> config file to point to the torch hub cache local folder:</p><p>entonces corre:</p><pre><code class="bash language-bash">uv run webapps/verification_mode_demo.py
</code></pre><ol start="5"><li>El terminal mostrará la URL para acceder a la aplicación web:</li></ol><pre><code class="bash language-bash">Running on local URL:  http://127.0.0.1:7860
</code></pre><p><strong>NOTA</strong>: La URL puede variar; comprueba la salida de la terminal para la dirección correcta.
</p><h2 id="documentation">📙 Documentación</h2><p>La documentación está disponible <a href="https://docs.sinapsis.tech/docs">web de sinapsis</a></p><p>Tutoriales para diferentes proyectos dentro de sinapsis están disponibles en <a href="https://docs.sinapsis.tech/tutorials">la página de tutoriales de sinapsis</a></p><h2 id="license">🔍 Licencia</h2><p>Este proyecto está licenciado bajo la licencia AGPLv3, que fomenta la colaboración abierta y el intercambio. Para más detalles, consulte el <a href="LICENSE">LICENSE</a> archivo.</p><p>Para uso comercial, consulte nuestra página <a href="https://sinapsis.tech">Sitio web de Sinapsis</a> para información sobre la obtención de una licencia comercial.</p></p>