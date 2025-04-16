# -*- coding: utf-8 -*-
from typing import ClassVar

import numpy as np
import torch
from sinapsis_core.data_containers.data_packet import ImageAnnotations, ImagePacket

from sinapsis_retina_face_trt.templates.retina_face.deepface_face_recognition import crop_bbox_from_img
from sinapsis_retina_face_trt.templates.retina_face.pytorch_embedding_search_from_gallery import (
    PytorchEmbeddingSearch,
)


class FaceVerificationFromGallery(PytorchEmbeddingSearch):
    """Template to perform Face Verification by computing a similarity score between received embeddings produced from
    received face detections and face embeddings stored in a gallery file. If gallery file don't exist it's created by
    specifying a path containing reference images.

    Usage example:
        agent:
          name: my_test_agent
        templates:
        - template_name: InputTemplate
          class_name: InputTemplate
          attributes: {}
        - template_name: FaceVerificationFromGallery
          class_name: FaceVerificationFromGallery
          template_input: InputTemplate
          attributes:
            gallery_file: '`replace_me:<class ''str''>`'
            similarity_threshold: 0.5
            k_value: 3
            metric: cosine
            device: cuda
            force_build_from_dir: false
            model_to_use: Facenet512EmbeddingExtractorTRTDev
            image_root_dir: null
            model_kwargs: '`replace_me:<class ''dict''>`'
            face_detector_kwargs: '`replace_me:<class ''dict''>`'
            use_face_detector_for_gallery_creation: false
            face_detector: RetinaFacePytorchTRT
            label_path_index: -2

    """

    VERIFIED = "verified"
    NOT_VERIFIED = "not-verified"

    class AttributesBaseModel(PytorchEmbeddingSearch.AttributesBaseModel):
        """Attributes for FaceVerificationFromGallery template.

        Args:
            label_mapping (dict[str, int]): Labels mapping for face verification results.
                Defaults to {"verified": 0, "not-verified": 1}.
        """

        labels_mapping: ClassVar[dict[str, int]] = {"verified": 0, "not-verified": 1}

    def compute_similarity(
        self,
        img_crop: np.ndarray,
    ) -> float | None:
        """Performs an embedding similarity score according to specified metric.

        Args:
            img_crop (np.ndarray): Input crop to be compared agains gallery.

        Returns:
            float | None: Predicted similarity score.
        """
        crop_embedding = self.infer_model(ImagePacket(content=img_crop))
        if crop_embedding is None:
            self.logger.debug("Not performing similarity computation due to no embedding")
            return None

        if self.attributes.metric == "cosine":
            dist = torch.cosine_similarity(self.gallery.gallery, crop_embedding, dim=1)
            return dist.max()

        dist = torch.norm(self.gallery.gallery - crop_embedding, dim=1, p=None)
        return dist.min()

    def update_annotations(self, similarity_score: float, ann: ImageAnnotations) -> None:
        """Update image annotations according to face similarity results.

        Args:
            similarity_score (float): Computed face similarity score.
            ann (ImageAnnotations): Image annotations to be updated.
        """
        if similarity_score > self.attributes.similarity_threshold:
            ann.label = self.attributes.labels_mapping.get(self.VERIFIED)
            ann.label_str = self.VERIFIED
            ann.confidence_score = similarity_score
        else:
            ann.label = self.attributes.labels_mapping.get(self.NOT_VERIFIED)
            ann.label_str = self.NOT_VERIFIED
            ann.confidence_score = similarity_score

    def _execute_single_image(self, image_packet: ImagePacket) -> None:
        """Executes the pipeline for a single ImagePacket, crops image according to received face detection bboxes
        and update image annotations according to predicted similarity score.

        Args:
            image_packet (ImagePacket): _description_
        """

        for ann in image_packet.annotations:
            crop = crop_bbox_from_img(ann, image_packet.content)
            if crop is not None and crop.size >= 4:
                similarity_score = self.compute_similarity(crop)
                if similarity_score is None:
                    continue
                self.update_annotations(similarity_score, ann)
