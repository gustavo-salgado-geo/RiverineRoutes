import os
import numpy as np
import rasterio
from skimage.morphology import closing, disk
import processing
from qgis.PyQt.QtGui import QIcon

from qgis.PyQt.QtCore import QCoreApplication
from qgis.core import (QgsProcessing,
                       QgsProcessingAlgorithm,
                       QgsProcessingParameterMultipleLayers,
                       QgsProcessingParameterRasterDestination,
                       QgsProcessingParameterNumber,
                       QgsProcessingException,
                       QgsRasterLayer,
                       QgsRasterBandStats)

from qgis.analysis import QgsRasterCalculator, QgsRasterCalculatorEntry

class WaterMaskRasterAlgorithm(QgsProcessingAlgorithm):
    """
    Algoritmo para calcular a Máscara de Água (NDWI) a partir de bandas Green e NIR.
    Inclui opções avançadas de limiar (threshold), fechamento morfológico e remoção de ruídos.
    """
    
    INPUT_GREEN = 'INPUT_GREEN'
    INPUT_NIR = 'INPUT_NIR'
    NDWI_THRESHOLD = 'NDWI_THRESHOLD'
    CLOSING_SIZE = 'CLOSING_SIZE'
    SIEVE_SIZE = 'SIEVE_SIZE'
    OUTPUT_RASTER = 'OUTPUT_RASTER'

    def tr(self, string):
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        return WaterMaskRasterAlgorithm()

    def name(self):
        return 'watermask_raster'

    def displayName(self):
        return self.tr('1A. WaterMask (Gerar Raster Binário)')

    def group(self):
        return self.tr('Módulo Base')

    def groupId(self):
        return 'modulo_base'

    def shortHelpString(self):
        return self.tr("Calcula o NDWI com limiar ajustável. Permite usar morfologia matemática para conectar rios fragmentados e filtro Sieve para apagar ruídos.")

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterMultipleLayers(self.INPUT_GREEN, self.tr('Banda Green (1 ou mais)'), layerType=QgsProcessing.TypeRaster))
        self.addParameter(QgsProcessingParameterMultipleLayers(self.INPUT_NIR, self.tr('Banda NIR (1 ou mais)'), layerType=QgsProcessing.TypeRaster))

        # 1. Parâmetro do Threshold do NDWI
        self.addParameter(
            QgsProcessingParameterNumber(
                self.NDWI_THRESHOLD,
                self.tr('Limiar (Threshold) do NDWI (Ex: 0.0 ou -0.1)'),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.0,
                minValue=-0.2,
                maxValue=0
            )
        )

        # 2. Parâmetro de Fechamento Morfológico
        self.addParameter(
            QgsProcessingParameterNumber(
                self.CLOSING_SIZE,
                self.tr('Conectar Rios Desconectados (Raio em pixels, 0 para desativar)'),
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=3,
                minValue=0,
                maxValue=50
            )
        )

        # 3. Parâmetro de Filtro Sieve (Ruídos)
        self.addParameter(
            QgsProcessingParameterNumber(
                self.SIEVE_SIZE,
                self.tr('Remover Ruídos/Ilhas (Tamanho mínimo em pixels, 0 para desativar)'),
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=10,
                minValue=0,
                maxValue=100000
            )
        )

        self.addParameter(QgsProcessingParameterRasterDestination(self.OUTPUT_RASTER, self.tr('Máscara Binária de Água (Raster)')))

    def processAlgorithm(self, parameters, context, feedback):
        green_layers = self.parameterAsLayerList(parameters, self.INPUT_GREEN, context)
        nir_layers = self.parameterAsLayerList(parameters, self.INPUT_NIR, context)
        
        ndwi_thresh = self.parameterAsDouble(parameters, self.NDWI_THRESHOLD, context)
        closing_size = self.parameterAsInt(parameters, self.CLOSING_SIZE, context)
        sieve_size = self.parameterAsInt(parameters, self.SIEVE_SIZE, context)

        if not green_layers or not nir_layers:
            raise QgsProcessingException(self.tr("Erro: Nenhuma camada fornecida para as bandas de entrada."))

        # Caminho final solicitado pelo utilizador
        final_output_path = self.parameterAsOutputLayer(parameters, self.OUTPUT_RASTER, context)
        
        # Gestão de ficheiros temporários baseada nas opções ativas
        do_closing = closing_size > 0
        do_sieve = sieve_size > 0
        calc_out_path = final_output_path if not do_closing and not do_sieve else os.path.join(context.temporaryFolder(), 'calc_ndwi_temp.tif')

        # -----------------------------------------------------------
        # Função auxiliar para gerar mosaico
        # -----------------------------------------------------------
        def prepare_band(layers_list, band_name):
            if len(layers_list) == 1:
                return layers_list[0]
            else:
                feedback.pushInfo(self.tr(f"A criar mosaico para a Banda {band_name}..."))
                merge_params = {'INPUT': layers_list, 'PCT': False, 'SEPARATE': False, 'DATA_TYPE': 5, 'OUTPUT': 'TEMPORARY_OUTPUT'}
                result_merge = processing.run("gdal:merge", merge_params, context=context, feedback=feedback, is_child_algorithm=True)
                return QgsRasterLayer(result_merge['OUTPUT'], f"mosaico_{band_name}")

        green_final_layer = prepare_band(green_layers, "Green")
        nir_final_layer = prepare_band(nir_layers, "NIR")

        # -----------------------------------------------------------
        # Deteção do Limiar de Ruído
        # -----------------------------------------------------------
        max_val = green_final_layer.dataProvider().bandStatistics(1, QgsRasterBandStats.Max).maximumValue
        noise_threshold = 0.01 if max_val <= 2.0 else 5 if max_val <= 300 else 100

        # -----------------------------------------------------------
        # ETAPA 1: Cálculo do NDWI Otimizado
        # -----------------------------------------------------------
        feedback.pushInfo(self.tr(f"A calcular NDWI com Threshold de {ndwi_thresh}..."))
        
        entries = []
        for ref, layer in [('green@1', green_final_layer), ('nir@1', nir_final_layer)]:
            entry = QgsRasterCalculatorEntry()
            entry.ref = ref
            entry.raster = layer
            entry.bandNumber = 1
            entries.append(entry)
        
        # Fórmula: Ignora NoData e aplica o threshold escolhido pelo utilizador
        formula = f'( "green@1" > {noise_threshold} ) * ( "nir@1" > {noise_threshold} ) * ((( 1.0 * "green@1" - 1.0 * "nir@1" ) / ( 1.0 * "green@1" + 1.0 * "nir@1" + 0.00001 )) >= {ndwi_thresh} )'
        
        calc = QgsRasterCalculator(formula, calc_out_path, 'GTiff', green_final_layer.extent(), green_final_layer.width(), green_final_layer.height(), entries)
        if calc.processCalculation(feedback) != 0:
            raise QgsProcessingException(self.tr("Erro ao processar a calculadora raster."))

        current_out_path = calc_out_path

        # -----------------------------------------------------------
        # ETAPA 2: Fechamento Morfológico (skimage)
        # -----------------------------------------------------------
        if do_closing:
            close_out_path = final_output_path if not do_sieve else os.path.join(context.temporaryFolder(), 'closed_temp.tif')
            feedback.pushInfo(self.tr(f"A aplicar fechamento morfológico (Raio: {closing_size} px) para conectar os rios..."))

            with rasterio.open(current_out_path) as src:
                meta = src.meta
                data = src.read(1)

            # Converter para binário absoluto (0 e 1) acelera imensamente o processamento
            data_uint8 = (data > 0).astype(np.uint8)
            closed_data = closing(data_uint8, disk(closing_size))

            meta.update(dtype=rasterio.uint8, nodata=0)

            with rasterio.open(close_out_path, 'w', **meta) as dst:
                dst.write(closed_data, 1)

            current_out_path = close_out_path

        # -----------------------------------------------------------
        # ETAPA 3: Filtro Sieve (gdal)
        # -----------------------------------------------------------
        if do_sieve:
            feedback.pushInfo(self.tr(f"A aplicar filtro Sieve para remover ruídos menores que {sieve_size} px..."))
            
            sieve_params = {
                'INPUT': current_out_path,
                'THRESHOLD': sieve_size,
                'EIGHT_CONNECTEDNESS': True,
                'OUTPUT': final_output_path
            }
            processing.run("gdal:sieve", sieve_params, context=context, feedback=feedback, is_child_algorithm=True)
            current_out_path = final_output_path

        # -----------------------------------------------------------
        # ETAPA 4: Transparência (NoData)
        # -----------------------------------------------------------
        feedback.pushInfo(self.tr("A aplicar transparência nas áreas terrestres..."))
        out_layer = QgsRasterLayer(current_out_path, "temp_out")
        if out_layer.isValid():
            out_layer.dataProvider().setNoDataValue(1, 0.0)

        return {self.OUTPUT_RASTER: current_out_path}
 
    def icon(self):
        """Ícone específico para a ferramenta Raster"""
        icon_path = os.path.join(os.path.dirname(__file__), 'icon_1.png')
        return QIcon(icon_path)