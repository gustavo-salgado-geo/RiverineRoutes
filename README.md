\# RiverineRoutes - QGIS Plugin 🚤🌍

\*\*RiverineRoutes\*\* é um plugin de processamento para o QGIS focado
em análise espacial hidrológica e roteamento intermodal (Terra-Rio). Ele
fornece um conjunto de ferramentas automatizadas para extrair corpos
d\'água de imagens de satélite, modelar rotas de navegação complexas e
integrar topologicamente redes terrestres e fluviais.

\-\--

\## 🛠️ Ferramentas Disponíveis

O plugin adiciona uma nova categoria à \*\*Caixa de Ferramentas de
Processamento\*\* do QGIS, contendo os seguintes algoritmos:

\### 1. WaterMask (Máscara de Água) Calcula o Índice da Diferença
Normalizada da Água (NDWI) a partir das bandas Green e NIR de imagens de
satélite (Landsat, Sentinel-2, CBERS). \* \*\*Saída:\*\* Raster binário
e um ficheiro vetorial (polígonos) delimitando as massas de água.

\### 2. RiverineRoutes (Rede Fluvial) Gera uma rede de navegação
estruturada e topologicamente correta no interior dos polígonos de água.
\* \*\*Rotas Centrais:\*\* Extraídas através da esqueletonização
matemática da máscara binária. \* \*\*Rotas Marginais:\*\* Geradas via
buffers internos baseados em distâncias medianas de navegação
(calculadas ou definidas pelo utilizador). \* \*\*Rotas
Transversais:\*\* Transectos perpendiculares que conectam as margens à
rota central.

\### 3. Land-River Routes (Integração Intermodal) Funde rotas terrestres
(estradas/caminhos) e as rotas fluviais numa única rede intermodal. \*
Utiliza algoritmos de \*snap\* (ancoragem geométrica) para garantir que
os nós de transição (portos/ancoradouros) ocorram perfeitamente nas
fronteiras da máscara de água, permitindo cálculos precisos em análises
de rede (\*Network Analysis\*).

\-\--

\## 📋 Pré-requisitos e Dependências

Este plugin utiliza bibliotecas científicas de Python avançadas para
manipulação de matrizes e geometrias. Como o ambiente Python padrão do
QGIS não inclui todos estes pacotes, é necessário instalá-los
manualmente.

\*\*Bibliotecas Necessárias:\*\* \* \`geopandas\` \* \`rasterio\` \*
\`scikit-image\` \* \`shapely\` \* \`numpy\`

\### Como instalar as dependências (Windows): 1. Procure e abra o
\*\*OSGeo4W Shell\*\* (instalado juntamente com o QGIS) como
Administrador. 2. Navegue até à pasta do plugin ou direcione para o
ficheiro \`requirements.txt\`. 3. Execute o comando: \`\`\`bash python
-m pip install -r requirements.txt
