# -*- coding: utf-8 -*-
"""RetinaFace Template in PyTorch."""

from dataclasses import dataclass
from typing import Any

import torch
from retinaface.pre_trained_models import get_model
from sinapsis_core.data_containers.annotations import (
    BoundingBox,
    ImageAnnotations,
    KeyPoint,
)
from sinapsis_core.data_containers.data_packet import DataContainer, ImagePacket
from sinapsis_core.template_base import Template
from sinapsis_core.template_base.base_models import OutputTypes, TemplateAttributes, UIPropertiesMetadata


@dataclass(frozen=True)
class RetinaFacePytorchOutputKeys:
    """
    Keys needed in the RetinaFacePytorch template.
    bbox (str): key for bounding boxes
    score (str): key for the score
    landmarks (str): key for the landmarks
    """

    bbox: str = "bbox"
    score: str = "score"
    landmarks: str = "landmarks"


class RetinaFacePytorch(Template):
    """Template based on the PyTorch implementation of RetinaFace:
    Deng, J., Guo, J., Zhou, Y., Yu, J., Kotsia, I., & Zafeiriou, S. (2019).
    Retinaface: Single-stage dense face localisation in the wild.
    arXiv preprint arXiv:1905.00641.

    The template runs inference on the ImagePacket and generates the corresponding
    annotations for bounding boxes, and key_points if set through the attributes.

    Usage example:

    agent:
        name: my_test_agent
    templates:
    - template_name: InputTemplate
      class_name: InputTemplate
      attributes: {}
    - template_name: RetinaFacePytorch
      class_name: RetinaFacePytorch
      template_input: InputTemplate
      attributes:
        cuda: true
        return_key_points: true
        confidence_threshold: 0.7
        nms_threshold: 0.4
        face_class_id: 1
        height: 960
        width: 960
        model_name: resnet50_2020-07-20

    """

    UIProperties = UIPropertiesMetadata(category="RetinaFace", output_type=OutputTypes.IMAGE)
    class AttributesBaseModel(TemplateAttributes):
        """Template attributes for the RetinaFacePytorch template.
        cuda (bool): Device to be used. If marked as True, it will use GPU,
        otherwise CPU
        return_key_points (bool):  whether to return the keypoints in the annotations
        confidence_threshold (float): confidence threshold for the predictions
        nms_threshold (float): threshold for the non-maximum suppression value
        face_class_id (int): The class ID for the current face to be added in
        the annotations
        max_size (int): Maximum size for the resizing of images
        model_name (str): Name to be used in the inference step
        """

        cuda: bool = True
        return_key_points: bool = True
        confidence_threshold: float = 0.7
        nms_threshold: float = 0.4
        face_class_id: int = 1
        height: int = 960
        width: int = 960
        model_name: str = "resnet50_2020-07-20"

    def __init__(
        self,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(attributes)
        self.device = "cuda" if self.attributes.cuda else "cpu"
        self.model = self.make_model()

    def make_model(self) -> torch.nn.Module:
        """Setup of the model with the corresponding init attributes.
        Sets in eval mode."""
        max_size = max(self.attributes.height, self.attributes.width)
        model: torch.nn.Module = get_model(
            self.attributes.model_name,
            max_size=max_size,
            device=self.device,
        )
        model.eval()
        return model

    @staticmethod
    def make_key_points(landmarks: list[list[float]], score: float) -> list[KeyPoint]:
        """Make keypoints from landmarks and scores.
        Args:
            landmarks (list[list[float]]): Detected landmarks.
            score (float): Confidence score of the face detection.

        Returns:
            list[KeyPoint]: List of keypoint annotations.
        """
        key_points = [KeyPoint(x=kpt[0], y=kpt[1], score=score) for kpt in landmarks]

        return key_points

    @staticmethod
    def make_bbox(bbox: list[float]) -> BoundingBox:
        """Make BoundingBox annotation.

        Args:
            bbox (list[float]): bounding box coordinates.

        Returns:
            BoundingBox: BoundingBox annotation.
        """
        x = max(0, int(bbox[0]))
        y = max(0, int(bbox[1]))
        return BoundingBox(
            x=x,
            y=y,
            w=int(bbox[2] - x),
            h=int(bbox[3] - y),
        )

    def parse_single_image_results(self, image: ImagePacket, predictions: list[dict[str, Any]]) -> None:
        """Parse single image results and store the parsed annotations
        in the image packet.

        Args:
            image (ImagePacket): image packet where the parsed annotations
            will be stored.
            predictions (list[dict[str, Any]]): List of predictions.
            Each prediction consists of a dictionary with "bbox",
            "score" and "landmarks".
        """
        predicted_anns = []
        for pred in predictions:
            if not pred[RetinaFacePytorchOutputKeys.bbox]:
                continue
            score = pred[RetinaFacePytorchOutputKeys.score] * 100
            predicted_anns.append(
                ImageAnnotations(
                    label=self.attributes.face_class_id,
                    label_str="face",
                    bbox=self.make_bbox(pred[RetinaFacePytorchOutputKeys.bbox]),
                    keypoints=self.make_key_points(pred[RetinaFacePytorchOutputKeys.landmarks], score),
                    confidence_score=score,
                )
            )
        if image.annotations:
            image.annotations += predicted_anns
        else:
            image.annotations = predicted_anns

    def execute(self, container: DataContainer) -> DataContainer:
        """
        Perform pytorch retina face model inference on a batch of
        images contained in received DataContainer

        Args:
            container (DataContainer): Received DataContainer

        Returns:
            DataContainer: Processed DataContainer
        """

        with torch.autocast(device_type=self.device, dtype=torch.float16, cache_enabled=True):
            for image_packet in container.images:
                # for trt models with latest version of retina face
                # torch we need to do proper
                # resizing of images
                #  to keep aspect ratio
                pred = self.model.predict_jsons(
                    image_packet.content,
                    self.attributes.confidence_threshold,
                    self.attributes.nms_threshold,
                )
                self.parse_single_image_results(image_packet, pred)
        return container
