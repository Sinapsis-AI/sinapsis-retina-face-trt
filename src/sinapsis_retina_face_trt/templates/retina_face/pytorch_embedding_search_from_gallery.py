# -*- coding: utf-8 -*-

from pathlib import Path
from typing import ClassVar, List, Literal, cast

import joblib
import numpy as np
import polars as pl
import torch
from pydantic import BaseModel, ConfigDict, Field
from pydantic.dataclasses import dataclass
from sinapsis_core.data_containers.annotations import BoundingBox, ImageAnnotations
from sinapsis_core.data_containers.data_packet import DataContainer, ImagePacket
from sinapsis_core.template_base import (
    Template,
    TemplateAttributes,
    TemplateAttributeType,
)
from sinapsis_data_readers.templates.image_readers.image_folder_reader_cv2 import (
    FolderImageDatasetCV2,
)

from .deepface_face_recognition import (
    Facenet512EmbeddingExtractorTRT,
    PytorchEmbeddingExtractor,
    crop_bbox_from_img,
)
from .deepface_face_recognition_dev import Facenet512EmbeddingExtractorTRTDev

DEVICE_TYPES = Literal["cuda", "cpu"]
METRIC_TYPES = Literal["cosine", "euclidean"]
MODEL_LITERAL_TYPE = Literal[
    "Facenet512EmbeddingExtractorTRT",
    "Facenet512EmbeddingExtractorTRTDev",
    "PytorchEmbeddingExtractor",
]


@dataclass(frozen=True)
class MetadataColumnNames:
    """
    Column names for metadata dataframes
    str_label (str): key for the str_label
    img_paths (str): key for the image paths
    bounding_boxes (str) key for the bounding boxes
    """

    str_label: str = "str_label"
    img_paths: str = "img_paths"
    bounding_boxes: str = "bounding_boxes"


def make_metadata(
    str_labels: list[str],
    img_paths: list[str | None] | None = None,
    bounding_boxes: list[BoundingBox] | None = None,
) -> pl.DataFrame:
    """Make a polars Dataframe with metadata consisting of
       labels, image paths and their bounding boxes annotations.

    Args:
        str_labels (List[str]): List of labels.
        img_paths (List[str] | None, optional): List of image paths. Defaults to None.
        bounding_boxes (List[BoundingBox] | None, optional): List of BoundingBox
        annotations. Defaults to None.

    Returns:
        pl.DataFrame: polars Dataframe with metadata.
    """

    num_labels = len(str_labels)
    if img_paths is None:
        img_paths = [None] * num_labels
    if bounding_boxes is None:
        bounding_boxes = [None] * num_labels
    return pl.DataFrame(
        {
            MetadataColumnNames.str_label: str_labels,
            MetadataColumnNames.img_paths: img_paths,
            MetadataColumnNames.bounding_boxes: bounding_boxes,
        }
    )


class EmbeddingGallery(BaseModel):
    """
    A class that represents a gallery of embeddings.

    Attributes:
        gallery (torch.Tensor): The tensor containing the embeddings.
        embedding_metadata (pl.DataFrame): A polars DataFrame containing metadata about
        each embedding.
        model_config (ConfigDict): A configuration dictionary for the model.
    """

    gallery: torch.Tensor
    embedding_metadata: pl.DataFrame
    model_config = ConfigDict(arbitrary_types_allowed=True)


class PytorchEmbeddingSearch(Template):
    """
    This template uses the Pytorch library, to perform embedding search
    in a gallery composed of embeddings.
    It returns the k-top closest vectors from the database.
    The template also provides functionality to build the gallery,
     by generating embedding for a given ImageDataset provided
    in the '_build_from_dir' method, using the model provided in the attributes.

    Usage example:
    agent:
      name: my_test_agent
    templates:
    - template_name: InputTemplate
      class_name: InputTemplate
      attributes: {}
    - template_name: PytorchEmbeddingSearch
      class_name: PytorchEmbeddingSearch
      template_input: InputTemplate
      attributes:
        gallery_file: '`replace_me:<class ''str''>`'
        similarity_threshold: 200.0
        k_value: 3
        metric: cosine
        device: cuda
        force_build_from_dir: false
        model_to_use: Facenet512EmbeddingExtractorTRTDev
        image_root_dir: null
        model_kwargs: '`replace_me:<class ''dict''>`'


    """

    SUPPORTED_EMBEDDING_EXTRACTORS_MAP: ClassVar = {
        "Facenet512EmbeddingExtractorTRT": Facenet512EmbeddingExtractorTRT,
        "Facenet512EmbeddingExtractorTRTDev": Facenet512EmbeddingExtractorTRTDev,
        "PytorchEmbeddingExtractor": PytorchEmbeddingExtractor,
    }

    PREDICTION_CLASS_LABEL = 1

    class AttributesBaseModel(TemplateAttributes):
        """Attributes for the PytorchEmbeddingSearch Template
        gallery_file (str): name of the file containing the gallery
        similarity_threshold (float): Threshold value to determine if two embeddings
            are similar
        k_value (int) : number of matches to return
        metric (METRIC_TYPES): type of metric to calculate distance between embeddings
        device (DEVICE_TYPE) : device used to perform the search
        force_build_from_dir (bool) : Whether to force the gallery build from image dir
        model_to_use (MODEL_LITERAL_TYPE) :  model to use for the embedding generation
        image_root_dir (str | None) : root directory where images are stored
        model_kwargs (dict | None) : Extra arguments for the model
        """

        gallery_file: str
        similarity_threshold: float = 200.0
        k_value: int = 3
        metric: METRIC_TYPES = "cosine"
        device: DEVICE_TYPES = "cuda"
        force_build_from_dir: bool = False
        model_to_use: MODEL_LITERAL_TYPE = "Facenet512EmbeddingExtractorTRTDev"
        image_root_dir: str | None = None
        model_kwargs: dict = Field(default_factory=dict)

    def __init__(self, attributes: TemplateAttributeType) -> None:
        super().__init__(attributes)
        self.embedding_extractor = self.__make_model()
        self.gallery: EmbeddingGallery = self.build_gallery()

    def _add_match_to_extras(self, ann: ImageAnnotations, metadata: pl.DataFrame, scores: List[float]) -> None:
        """Adds a match to the extras of an annotation if the similarity score
           is below or equal to the similarity threshold.

        Args:
            ann (ImageAnnotations): The annotation object where the label and
            its corresponding score will be added.
            metadata (pl.DataFrame): The DataFrame containing the metadata information.
            scores (List[float]): A list of similarity scores for each label in the
            metadata DataFrame.
        """
        for label, score in zip(metadata.to_dict()[MetadataColumnNames.str_label], scores):
            # for metadata_row, score in zip(metadata, scores):
            if score > self.attributes.similarity_threshold:
                continue

            if not ann.label and not ann.label_str:
                ann.label_str = label
                ann.label = 1
                ann.confidence_score = score
            else:
                if not ann.extra_labels:
                    ann.extra_labels = {}
                if label in ann.extra_labels:
                    continue  # don't overwrite the best score
                ann.extra_labels[label] = score

    def _execute_single_image(self, image_packet: ImagePacket) -> None:
        """Executes the pipeline for a single ImagePacket
        Crops image and adds information to the extras field of ImageAnnotations
        """
        for ann in image_packet.annotations:
            crop = crop_bbox_from_img(ann, image_packet.content)
            if crop is not None and crop.size >= 4:
                metadata, scores = self.search(crop)
                if metadata is not None and scores is not None:
                    self._add_match_to_extras(ann, metadata, scores)

    def execute(self, container: DataContainer) -> DataContainer:
        """Template execute method.
        For each of the images in the container, executes the pipeline that
        consists of cropping image, searching the top k values and adding
        any possible match to the image packet
        """

        for image_packet in container.images:
            if image_packet.annotations:
                self._execute_single_image(image_packet)

        return container

    def force_build_from_dir(self) -> bool:
        """Method to build the gallery from the directory,
        Returns True if set in the attributes or if the
        gallery file does not exist.
        Otherwise, returns False
        """
        return self.attributes.force_build_from_dir or not Path(self.attributes.gallery_file).exists()

    def build_gallery(self) -> EmbeddingGallery:
        """Builds the gallery from the directory if it does not exist,
        or build_from_dir is set to True.
        If the gallery exists and build_from_dir is False, it returns the gallery.
        """
        if self.force_build_from_dir():
            return self._build_from_dir()
        return self.load_from_file()

    def load_from_file(self) -> EmbeddingGallery:
        """Note that joblib imports are relative so a gallery won't be loaded
        if it was generated by calling this method from a different file"""
        self.logger.info(f"loading gallery from: {self.attributes.gallery_file}")
        gallery_file_load = joblib.load(self.attributes.gallery_file)
        gallery_file: EmbeddingGallery = cast(EmbeddingGallery, gallery_file_load)
        return gallery_file

    def __make_model(self) -> PytorchEmbeddingExtractor:
        """Returns the corresponding template from the
        PytorchEmbeddingSearch Mapping, initialized with
        the model arguments
        """
        if self.attributes.model_to_use not in PytorchEmbeddingSearch.SUPPORTED_EMBEDDING_EXTRACTORS_MAP:
            raise TypeError(
                f"{self.class_name}: Unsupported model specified. "
                "Supported models: "
                f"{PytorchEmbeddingSearch.SUPPORTED_EMBEDDING_EXTRACTORS_MAP.keys()}"
            )
        return PytorchEmbeddingSearch.SUPPORTED_EMBEDDING_EXTRACTORS_MAP[self.attributes.model_to_use](
            self.attributes.model_kwargs
        )

    def __infer_model(self, img_packet: ImagePacket) -> np.ndarray:
        """Extracts embedding from the image packet
        Args:
            img_packet (ImagePacket): Image packet to extract
                the embedding from
        """
        container = DataContainer()
        container.images = [img_packet]
        out_data_container = self.embedding_extractor.execute(container=container)
        embedding: np.ndarray = out_data_container.images[0].embedding
        return embedding

    def _build_from_dir(self) -> EmbeddingGallery:
        """Creates an embedding gallery from the images in a folder dataset,
        and returns the EmbeddingGallery object
        """
        embeddings = []
        str_labels = []
        img_paths = []
        if not self.attributes.image_root_dir:
            self.logger.error("No image root dir specified")

        for img_packet in FolderImageDatasetCV2(
            {
                "data_dir": self.attributes.image_root_dir,
                "load_on_init": True,
            }
        ).data_collection:
            result_tensor = self.__infer_model(img_packet)
            embeddings.append(result_tensor.squeeze())
            str_labels.append(img_packet.annotations[0].label_str)
            img_paths.append(img_packet.source)

        gallery = EmbeddingGallery(
            gallery=torch.stack(embeddings),
            embedding_metadata=make_metadata(
                str_labels=str_labels,
                img_paths=img_paths,
            ),
        )
        joblib.dump(gallery, self.attributes.gallery_file)
        self.logger.info(f"built gallery and saved to: {self.attributes.gallery_file}")
        return gallery

    def search(
        self,
        search_query: np.ndarray,
    ) -> tuple[pl.DataFrame | None, list[float] | None]:
        """Makes an embedding search
        If there is a match, it inserts that embedding into the gallery
        Args:
            search_query (np.ndarray): Embedding to search in the gallery
        """
        embedding = self.__infer_model(ImagePacket(content=search_query))
        if embedding is None:
            self.logger.debug("Not performing search due to no embedding")
            return None, None
        val, idx = self._search_norm(embedding, self.gallery.gallery)
        metadata: pl.DataFrame = self.gallery.embedding_metadata[idx.tolist()]
        return metadata, val.tolist()

    def _search_norm(self, query: np.ndarray, gallery: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Uses the vector norm to find the top k matches of an embedding
        Args:
            query (np.ndarray): embedding to search in the gallery.
            gallery (torch.Tensor): Gallery to be populated
        Returns:
            the value and index of the top k matches
        """
        dist = torch.norm(gallery - query, dim=1, p=None)
        val, index = dist.topk(self.attributes.k_value, largest=False)
        return val, index
