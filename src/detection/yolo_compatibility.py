"""
YOLO Compatibility Module
Provides placeholder classes for YOLO model compatibility across versions
"""

import torch
import torch.nn as nn


class RobustPlaceholder(nn.Module):
    """Drop-in replacement for unknown YOLO layers.

    The original layer definitions (e.g. *C3k2*, *RepC3*, …) may modify the
    number of channels flowing through the network.  Loading a model that was
    trained with such layers would therefore break if we replaced them with a
    pure identity mapping because subsequent convolutions would receive an
    unexpected number of channels.  To preserve the tensor **shape** we do the
    following:

    1. Parse the constructor arguments that the Ultralytics model parser passes
       to each module.  The first positional argument is always the number of
       input channels *c1*.  If a second positional argument is provided *and*
       is an ``int`` we treat it as the desired output channel count *c2*;
       otherwise we fall back to *c1* (i.e. keep channels unchanged).
    2. If ``c1 == c2`` we can safely act as an identity layer.
    3. If the channel count needs to change we insert a lightweight ``1×1``
       convolution to adapt the feature map.  This adds a negligible amount of
       compute while maintaining compatibility with the rest of the network.
    """

    def __init__(self, *args, **kwargs):  # pylint: disable=unused-argument
        super().__init__()

        # --- Parse the incoming channel dimensions ---------------------------------
        c1 = args[0] if args else kwargs.get("c1", 64)

        # If the second positional argument is a bool it is probably the
        # *shortcut* flag, not *c2*.
        if len(args) >= 2 and isinstance(args[1], int):
            c2 = args[1]
        else:
            c2 = c1  # Default to preserving channel count

        self._needs_projection = c1 != c2

        if self._needs_projection:
            # 1×1 convolution to match channel dimensions
            self.proj = nn.Conv2d(c1, c2, kernel_size=1, stride=1, bias=False)
        else:
            self.proj = nn.Identity()

        # Ultralytics sets the attribute ``c`` on every module to signal the
        # number of output channels to the layer parser.
        self.c = c2

    # --------------------------------------------------------------------- forward
    def forward(self, x: torch.Tensor):  # noqa: D401, pylint: disable=arguments-differ
        return self.proj(x)


# Define all compatibility classes as proper classes that can be pickled
class C3k2(RobustPlaceholder):
    """C3k2 compatibility class"""
    pass

class C3k(RobustPlaceholder):
    """C3k compatibility class"""
    pass

class C3(RobustPlaceholder):
    """C3 compatibility class"""
    pass

class C2PSA(RobustPlaceholder):
    """C2PSA compatibility class"""
    pass

class C3TR(RobustPlaceholder):
    """C3TR compatibility class"""
    pass

class C3Ghost(RobustPlaceholder):
    """C3Ghost compatibility class"""
    pass

class RepC3(RobustPlaceholder):
    """RepC3 compatibility class"""
    pass

class GhostBottleneck(RobustPlaceholder):
    """GhostBottleneck compatibility class"""
    pass

class RepConv(RobustPlaceholder):
    """RepConv compatibility class"""
    pass

class C3x(RobustPlaceholder):
    """C3x compatibility class"""
    pass

class C3k2x(RobustPlaceholder):
    """C3k2x compatibility class"""
    pass

class C2f2(RobustPlaceholder):
    """C2f2 compatibility class"""
    pass

class PSABlock(RobustPlaceholder):
    """PSABlock compatibility class"""
    pass

class Attention(RobustPlaceholder):
    """Attention compatibility class"""
    pass

class PSA(RobustPlaceholder):
    """PSA compatibility class"""
    pass

class C2fAttn(RobustPlaceholder):
    """C2fAttn compatibility class"""
    pass

class ImagePoolingAttn(RobustPlaceholder):
    """ImagePoolingAttn compatibility class"""
    pass

class EdgeResidual(RobustPlaceholder):
    """EdgeResidual compatibility class"""
    pass

class C2fCIB(RobustPlaceholder):
    """C2fCIB compatibility class"""
    pass

class C2fPSA(RobustPlaceholder):
    """C2fPSA compatibility class"""
    pass

class SCDown(RobustPlaceholder):
    """SCDown compatibility class"""
    pass


def register_compatibility_classes():
    """Register all compatibility classes with ultralytics modules"""
    try:
        import ultralytics.nn.modules.block as block_module
        import ultralytics.nn.tasks as tasks_module
        
        # Get all compatibility classes from this module
        compatibility_classes = {
            'C3k2': C3k2,
            'C3k': C3k,
            'C3': C3,
            'C2PSA': C2PSA,
            'C3TR': C3TR,
            'C3Ghost': C3Ghost,
            'RepC3': RepC3,
            'GhostBottleneck': GhostBottleneck,
            'RepConv': RepConv,
            'C3x': C3x,
            'C3k2x': C3k2x,
            'C2f2': C2f2,
            'PSABlock': PSABlock,
            'Attention': Attention,
            'PSA': PSA,
            'C2fAttn': C2fAttn,
            'ImagePoolingAttn': ImagePoolingAttn,
            'EdgeResidual': EdgeResidual,
            'C2fCIB': C2fCIB,
            'C2fPSA': C2fPSA,
            'SCDown': SCDown,
        }
        
        for class_name, class_obj in compatibility_classes.items():
            if not hasattr(block_module, class_name):
                print(f"Creating {class_name} placeholder for model compatibility")
                setattr(block_module, class_name, class_obj)
                # Also add to tasks module globals for model parsing
                setattr(tasks_module, class_name, class_obj)
        
        return True
        
    except Exception as e:
        print(f"Warning: Could not register compatibility classes: {e}")
        return False


# Auto-register when module is imported
register_compatibility_classes()
