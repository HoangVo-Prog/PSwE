import sys
import types

# Cached-feature predictor execution does not instantiate ResNetCOCO.  Keep
# that base path importable in the repository's CPU/offline test environment
# when torchvision is absent; a real visual-backbone construction still fails
# explicitly at the point where the unavailable dependency is needed.
if "torchvision" not in sys.modules:
    try:
        import torchvision  # noqa: F401
    except (ImportError, ModuleNotFoundError):
        torchvision_stub = types.ModuleType("torchvision")
        detection_stub = types.ModuleType("torchvision.models.detection")
        transforms_stub = types.ModuleType("torchvision.transforms")

        class _Weights:
            COCO_V1 = None

        def _missing_backbone(*args, **kwargs):
            raise ImportError("torchvision is required only when constructing ResNetCOCO")

        detection_stub.maskrcnn_resnet50_fpn = _missing_backbone
        detection_stub.MaskRCNN_ResNet50_FPN_Weights = _Weights
        torchvision_stub.models = types.SimpleNamespace(detection=detection_stub)
        torchvision_stub.transforms = transforms_stub
        sys.modules["torchvision"] = torchvision_stub
        sys.modules["torchvision.models"] = torchvision_stub.models
        sys.modules["torchvision.models.detection"] = detection_stub
        sys.modules["torchvision.transforms"] = transforms_stub

from ._load_canonical import load_canonical

_module = load_canonical("models.py", "_isp_senet_air_models")
for _name, _value in vars(_module).items():
    if not _name.startswith("_"):
        globals()[_name] = _value

