from .criterion import MaskedCrossEntropyLoss, MaskedL1Loss, MaskedMSELoss
from .input_adapters import PatchedInputAdapter, SemSegInputAdapter
from .multimae_orig_1 import MultiMAE, MultiViT
from .output_adapters import (ConvNeXtAdapter, DPTOutputAdapter,
                              LinearOutputAdapter,
                              SegmenterMaskTransformerAdapter,
                              SpatialOutputAdapter)
