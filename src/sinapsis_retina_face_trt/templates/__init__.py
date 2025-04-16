# -*- coding: utf-8 -*-
import importlib
from typing import Any, Callable, cast

_root_lib_path = "sinapsis_retina_face_trt.templates"


_template_lookup = {
    "RetinaFacePytorch": f"{_root_lib_path}.retina_face.retina_face_pytorch",
    "RetinaFacePytorchTRT": f"{_root_lib_path}.retina_face.retina_face_pytorch_trt",
    "RetinaFacePytorchTRTTorchOnly": f"{_root_lib_path}.retina_face.retina_face_pytorch_trt",
    "PytorchEmbeddingSearch": f"{_root_lib_path}.retina_face.pytorch_embedding_search_from_gallery",
    "Facenet512EmbeddingExtractorTRT": f"{_root_lib_path}.retina_face.deepface_face_recognition",
    "Facenet512EmbeddingExtractorTRTDev": f"{_root_lib_path}.retina_face.deepface_face_recognition_dev",
    "FaceVerificationFromGallery": f"{_root_lib_path}.retina_face.face_verification_from_gallery",
}


def __getattr__(name: str) -> Callable[..., Any]:
    if name in _template_lookup:
        module = importlib.import_module(_template_lookup[name])
        attr = getattr(module, name)
        if callable(attr):
            return cast(Callable[..., Any], attr)
        raise TypeError(f"Attribute `{name}` in `{_template_lookup[name]}` is not callable.")

    raise AttributeError(f"template `{name}` not found in {_root_lib_path}")


__all__ = list(_template_lookup.keys())
