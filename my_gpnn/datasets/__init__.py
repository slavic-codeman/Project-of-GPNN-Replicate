from .CAD120.cad120 import CAD120
from .HICO.hico import HICO
from .VCOCO.vcoco import VCOCO

import utils
from .CAD120 import metadata as cad_metadata
from .HICO import metadata as hico_metadata
from .VCOCO import metadata as vcoco_metadata
"""如何引用才是正确的？至少utils.py不报错了"""
# import CAD120.metadata as cad_metadata
# import HICO.metadata as hico_metadata
# import VCOCO.metadata as vcoco_metadata

__all__ = ('CAD120', 'VCOCO','HICO', 'utils', 'cad_metadata', 'hico_metadata','vcoco_metadata')
