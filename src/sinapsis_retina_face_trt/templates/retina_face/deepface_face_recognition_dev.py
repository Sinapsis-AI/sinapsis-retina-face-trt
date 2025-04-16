# -*- coding: utf-8 -*-
from typing import Tuple

from deepface.basemodels import Facenet
from sinapsis_framework_converter.framework_converter.trt_torch_module_wrapper import (
    TensorrtTorchWrapper,
)

from .deepface_face_recognition import PytorchEmbeddingExtractor
from .model_converters.face_net_converter import FaceNetConverter


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
        Attributes for Facenet512EmbeddingExtractorTRTDev template
        from_bbox_crop (bool) : Establish whether to infer the embedding
            from the bbox or full image
        force_compilation (bool) : Establish whether to force the model compilation
        deep_copy_image (bool)  : Establish whether to make a deep copy of the image
        model_name (str) : Name of the model to use for the embedding
        """

        model_name: str = "Facenet512"

    def _convert_model(self) -> Tuple[TensorrtTorchWrapper, int]:
        """
        Converts the 'Facenet512' model to trt version, using the Framework converter modules
            The pipeline starts by exporting from keras -> tensorflow -> onnx -> trt
        """

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
