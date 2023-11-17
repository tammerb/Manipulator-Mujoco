from dm_control import mjcf
import os



class Hole:
    def __init__(self):
        self._xml_path = os.path.join(
            os.path.dirname(__file__),
            './hole/hole.xml',
        )
        self._mjcf_root = mjcf.from_path(self._xml_path)

    @property
    def mjcf_model(self):
        """Returns the `mjcf.RootElement` object corresponding to this robot."""
        return self._mjcf_root