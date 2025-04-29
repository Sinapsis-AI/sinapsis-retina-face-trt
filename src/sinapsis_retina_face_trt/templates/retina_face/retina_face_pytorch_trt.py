# -*- coding: utf-8 -*-

from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from retinaface.net import FPN
from retinaface.predict_single import Model as RetinaFaceModel
from sinapsis_core.template_base.base_models import TemplateAttributeType
from sinapsis_framework_converter.framework_converter.framework_converter_torch import (
    FrameworkConverterTorch,
)
from sinapsis_framework_converter.framework_converter.framework_converter_torch_trt import (
    FrameworkConverterTorchTRT,
)
from sinapsis_framework_converter.framework_converter.framework_converter_trt import (
    FrameworkConverterTRT,
)
from sinapsis_framework_converter.framework_converter.trt_torch_module_wrapper import (
    TensorrtTorchWrapper,
)

from .retina_face_pytorch import RetinaFacePytorch


class CustomFPN(FPN):
    """
    Custom model class for retina face Feature Pyramid Network
    """

    def forward(self, x: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        """
        Custom FPN forward pass
        Args:
            x (dict[str, torch.Tensor]): input bindings of the model
        Returns
            the predictions for the inputs
        """
        y = list(x.values())

        output1 = self.output1(y[0])
        output2 = self.output2(y[1])
        output3 = self.output3(y[2])
        up3 = F.interpolate(output3, size=output2.shape[2:], mode="nearest")
        output2 = output2 + up3
        output2 = self.merge2(output2)

        up2 = F.interpolate(output2, size=output1.shape[2:], mode="nearest")
        output1 = output1 + up2
        output1 = self.merge1(output1)
        return [output1, output2, output3]


def update_retinaface_fpn_model(model: RetinaFaceModel) -> None:
    """
    Updates RetinaFaceModel FPN with CustomFPN
    """
    in_channels = 256
    out_channels = 256
    in_channels_stage2 = in_channels
    in_channels_list = [
        in_channels_stage2 * 2,
        in_channels_stage2 * 4,
        in_channels_stage2 * 8,
    ]
    new_fpn = CustomFPN(in_channels_list, out_channels).eval()
    new_fpn.load_state_dict(deepcopy(model.model.fpn.state_dict()), strict=True)
    model.model.fpn = new_fpn
    model.model.fpn.cuda()


class RetinaFaceConverter(FrameworkConverterTorch, FrameworkConverterTRT):
    """
    Framework converter class for RetinaFace models
    """

    PARENT_SAVE_DIR = f"{torch.hub.get_dir()}/checkpoints"


class RetinaFacePytorchTRT(RetinaFacePytorch):
    """
    PytorchTRT version of RetinaFace Template.

    The template extends the functionality of generating the corresponding
    annotations for the inference performed on the image packet passed in
    the DataContainer. The template optimizes the pytorch model using the
    TorchTRT module, allowing for faster inference times

    Usage example:

    agent:
      name: my_test_agent
    templates:
    - template_name: InputTemplate
      class_name: InputTemplate
      attributes: {}
    - template_name: RetinaFacePytorchTRT
      class_name: RetinaFacePytorchTRT
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
        force_compilation: false
        local_model_path: null


    """

    class AttributesBaseModel(RetinaFacePytorch.AttributesBaseModel):
        """
        Attributes for RetinaFacePytorchTRT
        cuda (bool): Device to be used. If marked as True, it will use GPU,
        otherwise CPU
        return_key_points (bool):  whether to return the keypoints in the annotations
        confidence_threshold (float): confidence threshold for the predictions
        nms_threshold (float): threshold for the non-maximum suppression value
        face_class_id (int): The class ID for the current face to be added in the
        annotations
        max_size (int): Maximum size for the resizing of images
        force_compilation (bool): Whether to force model compilation
        local_model_path (str) :  Path to local Pytorch model
        """

        force_compilation: bool = False
        local_model_path: str | None = None

    def __init__(self, attributes: TemplateAttributeType) -> None:
        attributes["device"] = "cuda"  # enforce cuda
        super().__init__(attributes)

        self.model_exporter = RetinaFaceConverter(self.attributes)
        self.convert_model()
        self._replace_retina_model()

    def convert_model(self) -> None:
        """Only convert if no local path provided or if force compilation is True"""
        model_path = Path(self.attributes.local_model_path)
        if self.model_exporter.force_export(model_path):
            self.update_fpn()
            self.model_exporter.export_torch_to_onnx(self.model.model)
            self.model_exporter.export_onnx_to_trt()

    def update_fpn(self) -> None:
        """
        Updates RetinaFace FPN model with CustomFPN
        """
        update_retinaface_fpn_model(self.model)
        self.model.model.fpn.cuda()

    def _replace_retina_model(self) -> None:
        """
        Updates RetinaFace model with trt version
        """
        local_model_path = self.attributes.local_model_path or str(self.model_exporter.trt_model_file_path().absolute())

        self.model.model = TensorrtTorchWrapper(local_model_path, output_as_value_tuple=True)


class RetinaFaceConverterV2(FrameworkConverterTorchTRT):
    """
    Framework converter from pytorch to torch-trt.
    This class extends the functionality from its base class and
    sets the SAVE_DIR to that to torch hub
    """

    PARENT_SAVE_DIR = f"{torch.hub.get_dir()}/checkpoints"


class RetinaFacePytorchTRTTorchOnly(RetinaFacePytorch):
    """
    The template extends the functionality from its base class RetinaFacePytorch,
    running inference on the ImagePacket, and generating the corresponding
    annotations for bounding boxes, and key_points if set through the attributes,
    and forcing the model conversion to torch-trt

    Usage example:

    agent:
        name: my_test_agent
    templates:
    - template_name: InputTemplate
      class_name: InputTemplate
      attributes: {}
    - template_name: RetinaFacePytorchTRTTorchOnly
      class_name: RetinaFacePytorchTRTTorchOnly
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
        force_compilation: false
        local_model_path: null


    """

    AttributesBaseModel = RetinaFacePytorchTRT.AttributesBaseModel

    def __init__(self, attributes: dict[str, Any]) -> None:
        super().__init__(attributes)
        self.model_exporter = RetinaFaceConverterV2(self.attributes)
        self.convert_model()
        self._replace_retina_model()

    def convert_model(self) -> None:
        """
        Converts model from torch to TorchTRT version. The model is saved in
        ExportedProgram format as explained in
        https://pytorch.org/TensorRT/user_guide/saving_models.html
        """
        model_path = Path(self.attributes.local_model_path)
        if self.model_exporter.force_export(model_path):
            self.update_fpn()
            self.model_exporter.export_torch_to_trt(self.model.model)

    def update_fpn(self) -> None:
        """
        Updates RetinaFace FPN with CustomFPN
        """
        update_retinaface_fpn_model(self.model)
        self.model.model.fpn.cuda()

    def _replace_retina_model(self) -> None:
        """
        Updates RetinaFace model with trt version
        """
        local_model_path = self.attributes.local_model_path or str(
            self.model_exporter.torch_trt_model_file_path().absolute()
        )
        self.model.model = self.model_exporter.load_model(local_model_path).cuda()
