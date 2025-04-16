FROM sinapsis-nvidia:base


RUN mkdir -p /root/.deepface/weights
# sync all requirements including optional ones
COPY . /app/
WORKDIR /app
RUN uv build
RUN --mount=type=ssh,mode=0666 uv sync --frozen  --all-extras
#
## install the generated wheel for this app
RUN uv pip install /app/dist/sinapsis_retina_face_trt-*.whl
