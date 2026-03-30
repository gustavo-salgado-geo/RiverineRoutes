import os
from qgis.core import QgsProcessingProvider
from qgis.PyQt.QtGui import QIcon

from .water_mask_raster_algorithm import WaterMaskRasterAlgorithm
from .water_mask_vector_algorithm import WaterMaskVectorAlgorithm
from .riverine_routes_algorithm import RiverineRoutesAlgorithm
from .land_river_integration_algorithm import LandRiverIntegrationAlgorithm

class RiverineRoutesProvider(QgsProcessingProvider):

    def __init__(self):
        super().__init__()

    def loadAlgorithms(self, *args, **kwargs):
        self.addAlgorithm(WaterMaskRasterAlgorithm())
        self.addAlgorithm(WaterMaskVectorAlgorithm())
        self.addAlgorithm(RiverineRoutesAlgorithm())
        self.addAlgorithm(LandRiverIntegrationAlgorithm())

    def id(self):
        return 'riverineroutes'

    def name(self):
        return 'Riverine Routes'

    def icon(self):
        """
        Retorna um ícone para a pasta do provedor na Caixa de Ferramentas.
        """
        # Pega o caminho absoluto da pasta do plugin e junta com o nome da imagem
        icon_path = os.path.join(os.path.dirname(__file__), 'icon.png') 
        return QIcon(icon_path)