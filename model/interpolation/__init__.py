"""Models registered for the interpolation task.

Importing this sub-package executes every concrete model file so their
``@register_model`` decorators run and populate the shared registry exposed by
``model.registry``.
"""

from ..registry import MODEL_REGISTRY, build_model, register_model

from . import atten_unet  # noqa: F401
from . import dncnn  # noqa: F401
from . import res_unet  # noqa: F401
from . import unet  # noqa: F401
from . import unet_plusplus  # noqa: F401
from . import gated_transformer_v9  # noqa: F401
from . import gated_transformer_v9_encdec  # noqa: F401
from . import gated_transformer_v11  # noqa: F401
from . import gated_transformer_v11_encdec  # noqa: F401

__all__ = [
    "MODEL_REGISTRY",
    "build_model",
    "register_model",
]
