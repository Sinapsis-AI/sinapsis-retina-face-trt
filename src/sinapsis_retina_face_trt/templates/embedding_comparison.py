from sinapsis_core.data_containers.data_packet import DataContainer, ImagePacket, ImageAnnotations
import torch
from sinapsis_core.template_base.base_models import TemplateAttributes, TemplateAttributeType
from typing import Literal
from sinapsis_core.template_base.template import Template

class EmbeddingComparisonAttributes(TemplateAttributes):
    """Attributes for the template
    threshold: Value below which embeddings are 'match' or not
    distance_method: Method to calculate distance
    dimension: normalization or dimension
    """

    threshold: float = 0.5
    distance_method: Literal['PairwiseDistance', 'CosineSimilarity'] = 'CosineSimilarity'
    dimension: int = 0



class EmbeddingComparison(Template):
    """Template to calculate the PairwiseDistance or CosineSimilarity between two image embeddings.
    If embeddings are close or similar enough, it adds an annotation with the value of similarity.
    Otherwise, adds annotation with no match and the value of similarity.
    """
    AttributesBaseModel =  EmbeddingComparisonAttributes
    def __init__(self, attributes:TemplateAttributeType)->None:
        super().__init__(attributes)
        distance_method = getattr(torch.nn, self.attributes.distance_method)
        self.distance_calculator = distance_method(self.attributes.dimension)

    @staticmethod
    def get_embedding_from_image(image: ImagePacket) -> torch.Tensor:
        """Extracts face embedding from image annotations.

        Args:
            image (ImagePacket): ImagePacket with image annotations embeddings.

        Returns:
            torch.Tensor: Extracted face embedding.
        """
        return image.annotations[-1].embedding[0].unsqueeze(0)


    def add_annotation(self, distance: torch.Tensor, image_packet:ImagePacket)->ImagePacket:
        """Adds annotation to the ImagePacket depending on the distance between embeddings.

        Args:
            distance (torch.Tensor): Tensor with similarity/distance values between the two embeddings
            image_packet (ImagePacket): ImagePacket to add the annotations on

        """

        if self.attributes.distance_method=='PairwiseDistance':

            distance = 1 - distance
        else:
            distance = distance.squeeze(0).mean().item()
        if distance > self.attributes.threshold:

            image_packet.annotations[-1].extra_labels={'match': distance*100}
        else:
            image_packet.annotations[-1].extra_labels = {'No match': distance*100}
            #)
        return image_packet

    def execute(self, container: DataContainer) -> DataContainer:
        images = container.images
        for i in range(0, len(container.images), 2):
            embedding_1 = self.get_embedding_from_image(images[i])
            embedding_2 = self.get_embedding_from_image(images[i+1]) if i + 1 <len(images) else []
            if embedding_1 is not None and embedding_2 is not None:
                similarity = self.distance_calculator(embedding_1, embedding_2)
                images[i] = self.add_annotation(similarity, images[i])
                images[i+1] = self.add_annotation(similarity, images[i+1])

            else:

                self.logger.debug('No embeddings to compare, returning container')
        return container

