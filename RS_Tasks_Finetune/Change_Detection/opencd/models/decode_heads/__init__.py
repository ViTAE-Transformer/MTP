from .bit_head import BITHead
from .changer import Changer
from .general_scd_head import GeneralSCDHead
from .identity_head import DSIdentityHead, IdentityHead
from .multi_head import MultiHeadDecoder
from .sta_head import STAHead
from .tiny_head import TinyHead

from .unet_head import UNetHead
from .uper_head import UPerHead

__all__ = ['BITHead', 'Changer', 'IdentityHead', 'DSIdentityHead', 'TinyHead',
           'STAHead', 'MultiHeadDecoder', 'GeneralSCDHead', 'UNetHead',
           'UPerHead']
