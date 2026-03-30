import os
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtCore import QCoreApplication
from qgis.core import (QgsProcessing,
                       QgsProcessingAlgorithm,
                       QgsProcessingParameterRasterLayer,
                       QgsProcessingParameterVectorDestination,
                       QgsProcessingException)
import processing

class WaterMaskVectorAlgorithm(QgsProcessingAlgorithm):
    """
    Algoritmo para converter um raster binário de água em polígonos vetoriais limpos,
    dissolvidos e em partes simples.
    """
    
    INPUT_RASTER = 'INPUT_RASTER'
    OUTPUT_VECTOR = 'OUTPUT_VECTOR'

    def tr(self, string):
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        return WaterMaskVectorAlgorithm()

    def name(self):
        return 'watermask_vector'

    def displayName(self):
        return self.tr('1B. WaterMask (Extrair Polígonos)')

    def group(self):
        return self.tr('Módulo Base')

    def groupId(self):
        return 'modulo_base'

    def shortHelpString(self):
        return self.tr("Converte um raster binário de água em polígonos, extrai apenas a água, dissolve fronteiras internas e converte para polígonos simples (Singleparts).")

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT_RASTER,
                self.tr('Selecione a Máscara Binária (Raster)')
            )
        )
        
        self.addParameter(
            QgsProcessingParameterVectorDestination(
                self.OUTPUT_VECTOR,
                self.tr('Máscara de Água (Polígonos)')
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        raster_layer = self.parameterAsRasterLayer(parameters, self.INPUT_RASTER, context)

        if raster_layer is None:
            raise QgsProcessingException(self.tr("Erro: Camada Raster inválida."))

        # -----------------------------------------------------------------
        # PASSO 1: Converter o raster para vetor
        # -----------------------------------------------------------------
        feedback.pushInfo(self.tr("Vetorizando a máscara binária..."))
        
        polygonize_params = {
            'INPUT_RASTER': raster_layer.source(),
            'RASTER_BAND': 1,
            'FIELD_NAME': 'is_water',
            'OUTPUT': 'TEMPORARY_OUTPUT'
        }
        
        result_vector = processing.run("native:pixelstopolygons", polygonize_params, context=context, feedback=feedback, is_child_algorithm=True)

        if feedback.isCanceled():
            return {}

        # -----------------------------------------------------------------
        # PASSO 2: Extrair APENAS a água (onde is_water = 1)
        # -----------------------------------------------------------------
        feedback.pushInfo(self.tr("Limpando ruído e extraindo apenas os polígonos de água..."))
        
        extract_params = {
            'INPUT': result_vector['OUTPUT'],
            'FIELD': 'is_water',
            'OPERATOR': 0, # 0 significa igual a (=)
            'VALUE': '1',
            'OUTPUT': 'TEMPORARY_OUTPUT' # Agora é temporário!
        }
        
        result_filtered = processing.run("native:extractbyattribute", extract_params, context=context, feedback=feedback, is_child_algorithm=True)

        if feedback.isCanceled():
            return {}

        # -----------------------------------------------------------------
        # PASSO 3: Dissolver polígonos colados (Remove fronteiras internas)
        # -----------------------------------------------------------------
        feedback.pushInfo(self.tr("A dissolver os polígonos para unir pixels adjacentes..."))
        
        dissolve_params = {
            'INPUT': result_filtered['OUTPUT'],
            'FIELD': [], # Deixar vazio dissolve TUDO numa única feição
            'OUTPUT': 'TEMPORARY_OUTPUT'
        }
        
        result_dissolved = processing.run("native:dissolve", dissolve_params, context=context, feedback=feedback, is_child_algorithm=True)

        if feedback.isCanceled():
            return {}

        # -----------------------------------------------------------------
        # PASSO 4: Partes Múltiplas para Partes Simples (Separa rios isolados)
        # -----------------------------------------------------------------
        feedback.pushInfo(self.tr("A separar a geometria em partes simples (Singleparts)..."))
        
        singleparts_params = {
            'INPUT': result_dissolved['OUTPUT'],
            'OUTPUT': parameters[self.OUTPUT_VECTOR] # Finalmente, o output definitivo
        }
        
        result_singleparts = processing.run("native:multiparttosingleparts", singleparts_params, context=context, feedback=feedback, is_child_algorithm=True)

        return {
            self.OUTPUT_VECTOR: result_singleparts['OUTPUT']
        }
        
    def icon(self):
        """Ícone específico para a ferramenta Vetor"""
        icon_path = os.path.join(os.path.dirname(__file__), 'icon_2.png')
        return QIcon(icon_path)