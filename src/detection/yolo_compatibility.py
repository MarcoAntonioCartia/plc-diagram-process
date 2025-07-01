"""
YOLO Compatibility Module
Provides placeholder classes for YOLO model compatibility across versions
"""

import torch
import torch.nn as nn

try:
    from ultralytics.nn.modules.block import C2f
    BASE_CLASS = C2f
except ImportError:
    # Fallback if C2f is not available
    BASE_CLASS = nn.Module


class RobustPlaceholder(BASE_CLASS):
    """Robust placeholder that handles tensor creation properly and can be pickled"""
    
    def __init__(self, *args, **kwargs):
        # Handle variable arguments from YOLO model parser
        # Common patterns: [c1, c2, n, shortcut, g, e] or [c1, shortcut, e]
        try:
            if len(args) >= 2:
                c1, c2 = args[0], args[0]  # Use same value for both if only one provided
                if len(args) >= 3:
                    # Handle different argument patterns
                    if isinstance(args[1], bool):  # [c1, shortcut, e] pattern
                        shortcut = args[1]
                        e = args[2] if len(args) > 2 else 0.5
                        n = kwargs.get('n', 1)
                        g = kwargs.get('g', 1)
                    else:  # [c1, c2, ...] pattern
                        c2 = args[1]
                        n = args[2] if len(args) > 2 else kwargs.get('n', 1)
                        shortcut = args[3] if len(args) > 3 else kwargs.get('shortcut', False)
                        g = args[4] if len(args) > 4 else kwargs.get('g', 1)
                        e = args[5] if len(args) > 5 else kwargs.get('e', 0.5)
                else:
                    n = kwargs.get('n', 1)
                    shortcut = kwargs.get('shortcut', False)
                    g = kwargs.get('g', 1)
                    e = kwargs.get('e', 0.5)
            else:
                # Fallback defaults
                c1, c2 = 64, 64
                n = kwargs.get('n', 1)
                shortcut = kwargs.get('shortcut', False)
                g = kwargs.get('g', 1)
                e = kwargs.get('e', 0.5)
            
            # Ensure we have valid tensor parameters
            if c1 is None or c2 is None:
                c1, c2 = 64, 64  # Default safe values
            
            if BASE_CLASS != nn.Module:
                super().__init__(c1, c2, n, shortcut, g, e)
            else:
                nn.Module.__init__(self)
                self.c = c1
                
        except Exception as ex:
            # Fallback to simple identity if C2f fails
            print(f"Warning: Placeholder fallback for {args}, {kwargs}: {ex}")
            nn.Module.__init__(self)
            self.c = args[0] if args else 64
    
    def forward(self, x):
        try:
            if BASE_CLASS != nn.Module:
                return super().forward(x)
            else:
                # Simple identity fallback
                return x
        except Exception:
            # Fallback to identity
            return x


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
