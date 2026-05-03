"""Models registered for the first-break-picking task.

Note: this folder name contains a hyphen, which prevents normal dotted-import
syntax (``import model.first-break_picking`` is a syntax error). Use
``importlib.import_module("model.first-break_picking")`` or rename the folder
to ``first_break_picking`` if you want to import it with regular syntax.

Importing this sub-package executes every concrete model file so their
``@register_model`` decorators run and populate the shared registry exposed by
``model.registry``.
"""

from ..registry import MODEL_REGISTRY, build_model, register_model

from . import atten_unet  # noqa: F401
from . import dncnn  # noqa: F401
from . import res_unet  # noqa: F401
from . import unet  # noqa: F401

__all__ = [
    "MODEL_REGISTRY",
    "build_model",
    "register_model",
]
