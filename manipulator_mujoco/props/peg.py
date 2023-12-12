from dm_control import mjcf
from . primitive import Primitive
import os



class Peg(Primitive):
    def __init__(self):
        self._xml_path = os.path.join(
            os.path.dirname(__file__),
            './peg/peg.xml',
        )
        self._mjcf_root = mjcf.from_path(self._xml_path)

    @property
    def mjcf_model(self):
        """Returns the `mjcf.RootElement` object corresponding to this robot."""
        return self._mjcf_root