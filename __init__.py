from .riverine_routes_provider import RiverineRoutesProvider

class RiverineRoutesPlugin:
    def __init__(self):
        self.provider = None

    def initProcessing(self):
        self.provider = RiverineRoutesProvider()
        from qgis.core import QgsApplication
        QgsApplication.processingRegistry().addProvider(self.provider)

    def initGui(self):
        self.initProcessing()

    def unload(self):
        from qgis.core import QgsApplication
        if self.provider:
            QgsApplication.processingRegistry().removeProvider(self.provider)

def classFactory(iface):
    return RiverineRoutesPlugin()