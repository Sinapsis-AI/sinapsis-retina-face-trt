# -*- coding: utf-8 -*-
from os.path import join
from pathlib import Path

from sinapsis_framework_converter.framework_converter.framework_converter import (
    DLFrameworkConverter,
    ModelExtensions,
)
from sinapsis_framework_converter.framework_converter.framework_converter_keras_tf import (
    FrameworkConverterKerasTF,
)
from sinapsis_framework_converter.framework_converter.framework_converter_tf import (
    FrameworkConverterTFONNX,
)
from sinapsis_framework_converter.framework_converter.framework_converter_trt import (
    FrameworkConverterTRT,
)


class FaceNetConverter(FrameworkConverterTRT, FrameworkConverterTFONNX, FrameworkConverterKerasTF):
    """
    Framework converter used for Deepface models
    The Class inherits functionality from its base classes 'FrameworkConverterTRT',
    'FrameworkConverterTFONNX', and 'FrameworkConverterKerasTF'
    by converting from keras to tensorflow, from tensorflow to onnx and finally
    from onnx to TensorRT.
    """

    PARENT_SAVE_DIR: str = f"{DLFrameworkConverter.PARENT_SAVE_DIR}/.deepface/weights/"

    def onnx_model_file_path(self) -> Path:
        """Returns path to the onnx model"""
        return Path(
            join(
                str(self.model_file_path().absolute()),
                f"{self.TF_DEFAULT_SAVE_NAME}{ModelExtensions.ONNX_FILE_EXTENSION}",
            )
        )

    def trt_model_file_path(self) -> Path:
        """Returns path to trt model"""
        return Path(
            join(
                str(self.model_file_path().absolute()),
                f"{self.TF_DEFAULT_SAVE_NAME}{ModelExtensions.TRT_MODEL_EXTENSION}",
            )
        )
