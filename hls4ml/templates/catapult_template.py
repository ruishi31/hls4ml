from hls4ml.templates.vivado_template import VivadoBackend
import os
from shutil import copyfile

class CatapultBackend(VivadoBackend):
    def __init__(self):
        super(CatapultBackend, self).__init__()
        self.name = 'Catapult'

