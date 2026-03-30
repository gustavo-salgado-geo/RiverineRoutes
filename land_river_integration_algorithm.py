import os
import processing
from qgis.PyQt.QtCore import QCoreApplication
from qgis.core import (QgsProcessing,
                       QgsProcessingAlgorithm,
                       QgsProcessingParameterVectorLayer,
                       QgsProcessingParameterDistance,
                       QgsProcessingParameterFeatureSink,
                       QgsFeatureSink)

class LandRiverIntegrationAlgorithm(QgsProcessingAlgorithm):
    """
    Algoritmo para integrar rotas terrestres e fluviais, garantindo a topologia nos pontos de transição (margens).
    """

    INPUT_WATER_MASK = 'INPUT_WATER_MASK'
    INPUT_LAND_ROUTES = 'INPUT_LAND_ROUTES'
    INPUT_RIVER_ROUTES = 'INPUT_RIVER_ROUTES'
    SNAP_TOLERANCE = 'SNAP_TOLERANCE'
    OUTPUT_INTEGRATED_NETWORK = 'OUTPUT_INTEGRATED_NETWORK'

    def tr(self, string):
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        return LandRiverIntegrationAlgorithm()

    def name(self):
        return 'landriverintegration'

    def displayName(self):
        return self.tr('3. Land-River Routes (Integração de Redes)')

    def group(self):
        return self.tr('Módulo Base')

    def groupId(self):
        return 'modulo_base'

    def shortHelpString(self):
        return self.tr("Funde as rotas terrestres e fluviais numa única rede. Utiliza a fronteira da máscara de água para criar nós de integração exatos (Snap).")

    def initAlgorithm(self, config=None):
        # 1. Máscara de Água (Vetor)
        self.addParameter(QgsProcessingParameterVectorLayer(
            self.INPUT_WATER_MASK, 
            self.tr('Máscara de Água (Polígonos limitadores)'), 
            [QgsProcessing.TypeVectorPolygon]
        ))
        
        # 2. Rede Terrestre
        self.addParameter(QgsProcessingParameterVectorLayer(
            self.INPUT_LAND_ROUTES, 
            self.tr('Rede de Estradas/Caminhos Terrestres (Linhas)'), 
            [QgsProcessing.TypeVectorLine]
        ))
        
        # 3. Rede Fluvial (Gerada na função 2)
        self.addParameter(QgsProcessingParameterVectorLayer(
            self.INPUT_RIVER_ROUTES, 
            self.tr('Rede de Rotas Fluviais (Linhas)'), 
            [QgsProcessing.TypeVectorLine]
        ))

        # 4. Tolerância de Snap
        self.addParameter(QgsProcessingParameterDistance(
            self.SNAP_TOLERANCE, 
            self.tr('Tolerância de Ajuste/Snap (metros)'), 
            defaultValue=50.0
        ))
        
        # OUTPUT: Rede Integrada
        self.addParameter(QgsProcessingParameterFeatureSink(
            self.OUTPUT_INTEGRATED_NETWORK, 
            self.tr('Rede Integrada (Terrestre + Fluvial)')
        ))

    def processAlgorithm(self, parameters, context, feedback):
        water_mask = self.parameterAsVectorLayer(parameters, self.INPUT_WATER_MASK, context)
        land_routes = self.parameterAsVectorLayer(parameters, self.INPUT_LAND_ROUTES, context)
        river_routes = self.parameterAsVectorLayer(parameters, self.INPUT_RIVER_ROUTES, context)
        snap_tolerance = self.parameterAsDouble(parameters, self.SNAP_TOLERANCE, context)

        # 1. Transformar a máscara de água (Polígonos) em Linhas (Fronteiras)
        feedback.pushInfo(self.tr("A extrair fronteiras costeiras/marginais..."))
        boundaries = processing.run("native:polygonstolines", {
            'INPUT': water_mask,
            'OUTPUT': 'TEMPORARY_OUTPUT'
        }, context=context, feedback=feedback)['OUTPUT']

        # 2. Encontrar interseções entre rotas terrestres e as margens da água (potenciais portos)
        feedback.pushInfo(self.tr("A identificar nós de integração (interseções)..."))
        intersections = processing.run("native:lineintersections", {
            'INPUT': land_routes,
            'INTERSECT': boundaries,
            'OUTPUT': 'TEMPORARY_OUTPUT'
        }, context=context, feedback=feedback)['OUTPUT']

        # 3. Aplicar SNAP nas Rotas Terrestres para garantir que tocam nos nós
        feedback.pushInfo(self.tr("A ajustar a topologia da rede terrestre..."))
        snapped_land = processing.run("native:snapgeometries", {
            'INPUT': land_routes,
            'REFERENCE_LAYER': intersections,
            'TOLERANCE': snap_tolerance,
            'BEHAVIOR': 1, # Preferir alinhar nós
            'OUTPUT': 'TEMPORARY_OUTPUT'
        }, context=context, feedback=feedback)['OUTPUT']

        # 4. Aplicar SNAP nas Rotas Fluviais para garantir que tocam nos nós
        feedback.pushInfo(self.tr("A ajustar a topologia da rede fluvial..."))
        snapped_river = processing.run("native:snapgeometries", {
            'INPUT': river_routes,
            'REFERENCE_LAYER': intersections,
            'TOLERANCE': snap_tolerance,
            'BEHAVIOR': 1,
            'OUTPUT': 'TEMPORARY_OUTPUT'
        }, context=context, feedback=feedback)['OUTPUT']

        # 5. Mesclar as Redes (MERGE)
        feedback.pushInfo(self.tr("A compilar a rede intermodal final..."))
        merged_network = processing.run("native:mergevectorlayers", {
            'LAYERS': [snapped_land, snapped_river],
            'CRS': context.project().crs(),
            'OUTPUT': parameters[self.OUTPUT_INTEGRATED_NETWORK]
        }, context=context, feedback=feedback)['OUTPUT']

        return {self.OUTPUT_INTEGRATED_NETWORK: merged_network}