# -*- coding: utf-8 -*-


from typing import Callable, cast


def __getattr__(name: str) -> Callable:
    match name:
        case "RetinaFacePytorch":
            from .retina_face.retina_face_pytorch import RetinaFacePytorch

            returnModule = cast("RetinaFacePytorch", RetinaFacePytorch)
        case "RetinaFacePytorchTRT":
            from .retina_face.retina_face_pytorch_trt import RetinaFacePytorchTRT

            returnModule = cast("RetinaFacePytorchTRT", RetinaFacePytorchTRT)
        case "RetinaFacePytorchTRTTorchOnly":
            from .retina_face.retina_face_pytorch_trt import (
                RetinaFacePytorchTRTTorchOnly,
            )

            returnModule = cast("RetinaFacePytorchTRTTorchOnly", RetinaFacePytorchTRTTorchOnly)
        case "PytorchEmbeddingSearch":
            from .retina_face.pytorch_embedding_search_from_gallery import (
                PytorchEmbeddingSearch,
            )

            returnModule = cast("PytorchEmbeddingSearch", PytorchEmbeddingSearch)
        case "PytorchEmbeddingExtractor":
            from .retina_face.deepface_face_recognition import (
                PytorchEmbeddingExtractor,
            )

            returnModule = cast("PytorchEmbeddingExtractor", PytorchEmbeddingExtractor)
        case "Facenet512EmbeddingExtractorTRT":
            from .retina_face.deepface_face_recognition import (
                Facenet512EmbeddingExtractorTRT,
            )

            returnModule = cast("Facenet512EmbeddingExtractorTRT", Facenet512EmbeddingExtractorTRT)
        case "Facenet512EmbeddingExtractorTRTDev":
            from .retina_face.deepface_face_recognition_dev import (
                Facenet512EmbeddingExtractorTRTDev,
            )

            returnModule = cast("Facenet512EmbeddingExtractorTRTDev", Facenet512EmbeddingExtractorTRTDev)
        case _:
            raise AttributeError(f"module {__name__!r} has no template {name!r}")
    return returnModule


__all__ = [
    "Facenet512EmbeddingExtractorTRT",
    "Facenet512EmbeddingExtractorTRTDev",
    "PytorchEmbeddingExtractor",
    "PytorchEmbeddingSearch",
    "RetinaFacePytorch",
    "RetinaFacePytorchTRT",
    "RetinaFacePytorchTRTTorchOnly",
]
