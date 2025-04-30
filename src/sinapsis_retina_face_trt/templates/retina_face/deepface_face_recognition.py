# -*- coding: utf-8 -*-
from abc import abstractmethod
from copy import deepcopy

import cv2
import numpy as np
import torch
from sinapsis_core.data_containers.annotations import ImageAnnotations
from sinapsis_core.data_containers.data_packet import DataContainer, ImagePacket
from sinapsis_core.template_base import Template
from sinapsis_core.template_base.base_models import (
    OutputTypes,
    TemplateAttributes,
    TemplateAttributeType,
    UIPropertiesMetadata,
)
from sinapsis_framework_converter.framework_converter.trt_torch_module_wrapper import (
    TensorrtTorchWrapper,
)


def crop_bbox_from_img(annotation: ImageAnnotations, image: np.ndarray) -> np.ndarray | None:
    """
    Crops the image using the bounding boxes in the annotations and
    returns the cropped image.
    Args:
        annotation (ImageAnnotations): annotation that contains the bounding boxes
        image (np.ndarray): Original image.
    Returns:
        np.ndarray: cropped image.
    """
    crop = None
    if annotation.bbox:
        crop = image[
            int(annotation.bbox.y) : int(annotation.bbox.y + annotation.bbox.h),
            int(annotation.bbox.x) : int(annotation.bbox.x + annotation.bbox.w),
        ]
    return crop


class PytorchEmbeddingExtractor(Template):
    """
    Base template for pytorch embedding extraction models
    This template is in charge of making pre-process to the images (e.g., cropping
    bboxes), inferring from the image, original or bbox, to get the embeddings and
    add embeddings to the ImagePacket.

    Usage example:

    agent:
      name: my_test_agent
    templates:
    - template_name: InputTemplate
      class_name: InputTemplate
      attributes: {}
    - template_name: PytorchEmbeddingExtractor
      class_name: PytorchEmbeddingExtractor
      template_input: InputTemplate
      attributes:
        from_bbox_crop: false
        force_compilation: false
        deep_copy_image: true

    """

    class AttributesBaseModel(TemplateAttributes):
        """
        Attributes for PytorchEmbeddingExtractor template:
        from_bbox_crop (bool): Establish whether infer the embedding
            from the bbox or full image
        force_compilation (bool): Establish whether force the model compilation
        deep_copy_image (bool): Establish whether to make a deep copy of the image
        """

        from_bbox_crop: bool | None = False
        force_compilation: bool | None = False
        deep_copy_image: bool | None = True
    UIProperties = UIPropertiesMetadata(category="DeepFace", output_type=OutputTypes.IMAGE)

    def __init__(
        self,
        attributes: TemplateAttributeType,
    ) -> None:
        super().__init__(attributes)
        self._model, self.input_shape = self._build_model()
        self.device = "cuda"

    @abstractmethod
    def _build_model(self) -> tuple[TensorrtTorchWrapper, int]: ...

    def _pre_process(self, image: np.ndarray) -> torch.Tensor:
        """
        Perform the following transformations to the input image
            - BGR2RGB
            - Resize to model input shape
            - Pixel value normalization
            - Numpy array to Torch tensor
        Args:
            image (np.ndarray): input image
        Returns:
            torch.Tensor: transformed image
        """

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.input_shape)
        mean, std = image.mean(), image.std()
        image = (image - mean) / std
        return torch.from_numpy(image).to(self.device).float()

    def _infer(self, image: np.ndarray) -> torch.Tensor:
        """
        Performs model inference on input image
        Args:
            image (np.ndarray): input image
        Returns: (np.ndarray): image embeddings as inferred by the model

        """

        if self.attributes.deep_copy_image:
            image = deepcopy(image)
        image_as_tensor: torch.Tensor = self._pre_process(image)
        embedding: torch.Tensor = self._model(image_as_tensor.unsqueeze(0))
        return embedding

    def _infer_from_crops(self, image_packet: ImagePacket) -> None:
        """
        Given an image annotation, gets the embeddings for the cropped image and
        stores in the 'embedding' field of the image
        Args:
            image_packet (ImagePacket): image with the array content
        """
        for ann in image_packet.annotations:
            crop = crop_bbox_from_img(ann, image_packet.content)

            if crop is not None and crop.size >= 4:
                ann.embedding = self._infer(crop)

    def execute(self, container: DataContainer) -> DataContainer:
        """Gets the embedding for each image in the data
        container and assigns to embedding attr."""

        with torch.autocast(device_type=self.device, dtype=torch.float16, cache_enabled=True):
            for img in container.images:
                if self.attributes.from_bbox_crop:
                    self._infer_from_crops(img)
                else:
                    img.embedding = self._infer(img.content)
            return container


class Facenet512EmbeddingExtractorTRT(PytorchEmbeddingExtractor):
    """
    Template for embedding extraction using the TRT version of the 'Facenet512' model.
    This template inherits the functionality from its base class 'PytorchEmbeddingExtractor'
    providing functionality to crop images, and extract embeddings from the crops.

    Usage example:

    agent:
      name: my_test_agent
    templates:
    - template_name: InputTemplate
      class_name: InputTemplate
      attributes: {}
    - template_name: Facenet512EmbeddingExtractorTRT
      class_name: Facenet512EmbeddingExtractorTRT
      template_input: InputTemplate
      attributes:
        from_bbox_crop: false
        force_compilation: false
        deep_copy_image: true
        model_local_path: '/path/to/resnet/model'
        model_name: Facenet512
        input_shape: (160, 160)

    """

    class AttributesBaseModel(PytorchEmbeddingExtractor.AttributesBaseModel):
        local_model_path: str
        model_name: str = "Facenet512"
        input_shape: tuple[int] = (160, 160)

    def _build_model(self) -> tuple[TensorrtTorchWrapper, int]:
        """
        Builds a trt model instance by loading a trt engine file
        from a local path
        """
        trt_model = TensorrtTorchWrapper(self.attributes.local_model_path, output_as_value_tuple=False)
        return trt_model, self.attributes.input_shape
