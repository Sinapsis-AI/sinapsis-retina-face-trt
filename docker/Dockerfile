FROM sinapsis-nvidia:base

RUN mkdir -p /root/.deepface/weights
# sync all requirements including optional ones
COPY . /app/
WORKDIR /app

RUN uv sync --frozen
RUN uv pip install sinapsis-retina-face-trt[all]
RUN uv pip install tensorflow[and-cuda]==2.18.0
