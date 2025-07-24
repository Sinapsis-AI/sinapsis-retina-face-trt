# -*- coding: utf-8 -*-
from typing import Tuple

import tensorflow as tf
from deepface.basemodels import Facenet
from sinapsis_core.utils.logging_utils import sinapsis_logger
from sinapsis_framework_converter.framework_converter.trt_torch_module_wrapper import (
    TensorrtTorchWrapper,
)

from .deepface_face_recognition import PytorchEmbeddingExtractor
from .model_converters.face_net_converter import FaceNetConverter


def set_memory_limit(memory_limit: int) -> None:
    """Set a memory limit in the GPU devices being used by TensorFlow.

    Args:
        memory_limit (int): The maximum amount of memory allowed per GPU in MB.
    """
    physical_gpu_devices = tf.config.list_physical_devices("GPU")
    if physical_gpu_devices:
        try:
            for gpu_id, gpu in enumerate(physical_gpu_devices):
                tf.config.set_logical_device_configuration(
                    gpu, [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit)]
                )
                sinapsis_logger.debug(f"Memory limit of: {memory_limit} set in TensorFlow GPU Device: {gpu_id}")
        except RuntimeError as rt_error:
            sinapsis_logger.debug(rt_error)
    else:
        sinapsis_logger.debug("No GPU memory limit set due to no physical GPU devices being found.")


class Facenet512EmbeddingExtractorTRTDev(PytorchEmbeddingExtractor):
    """
    Same as Facenet512EmbeddingExtractorTRT except this class converts the model
     at run time as opposed to 'Facenet512EmbeddingExtractorTRTDev' which expects
     the model to already be converted and stored locally.
    This template also has a set of extra dependencies such as 'deepface',
    'keras', and 'tensorflow'.

    Usage example:
    agent:
      name: my_test_agent
    templates:
    - template_name: InputTemplate
      class_name: InputTemplate
      attributes: {}
    - template_name: Facenet512EmbeddingExtractorTRTDev
      class_name: Facenet512EmbeddingExtractorTRTDev
      template_input: InputTemplate
      attributes:
        from_bbox_crop: false
        force_compilation: false
        deep_copy_image: true
        model_name: Facenet512
    """

    class AttributesBaseModel(PytorchEmbeddingExtractor.AttributesBaseModel):
        """
        Attributes for Facenet512EmbeddingExtractorTRTDev template.
        from_bbox_crop (bool) : Establish whether to infer the embedding
            from the bbox or full image.
        force_compilation (bool) : Establish whether to force the model compilation.
        deep_copy_image (bool)  : Establish whether to make a deep copy of the image.
        model_name (str) : Name of the model to use for the embedding.
        tf_memory_limit (int | None): Memory limit for TensorFlow gpu devices in MB.
            If None, no memory limit is set. Defaults to None.
        """

        model_name: str = "Facenet512"
        tf_memory_limit: int | None = None

    def _convert_model(self) -> Tuple[TensorrtTorchWrapper, int]:
        """
        Converts the 'Facenet512' model to trt version, using the Framework converter modules
            The pipeline starts by exporting from keras -> tensorflow -> onnx -> trt
        """

        if self.attributes.tf_memory_limit:
            set_memory_limit(self.attributes.tf_memory_limit)

        model = Facenet.FaceNet512dClient()
        exporter = FaceNetConverter(self.attributes)
        exporter.export_keras_to_tf(model.model)
        exporter.export_tensorflow_to_onnx(opset_version=14)
        exporter.export_onnx_to_trt()
        trt_model = TensorrtTorchWrapper(str(exporter.trt_model_file_path().absolute()), output_as_value_tuple=False)
        input_shape = model.model.input_shape[1:3]
        return trt_model, input_shape

    def _build_model(self) -> Tuple[TensorrtTorchWrapper, int]:
        """
        Executes the model conversion and returns the TensorrtTorchWrapper
        object and the shape of the model inputs
        """
        return self._convert_model()
