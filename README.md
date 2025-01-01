# Sistema de detecção facial

## Descrição da atividade

O objetivo principal deste projeto é trabalhar com as bibliotecas e frameworks estudados e analisados em nossas aulas. Neste sentido, a proposta padrão envolve um sistema de detecção e reconhecimento de faces, utilizando o framework TensorFlow em conjuntos com as bibliotecas que o projetista julgue necessárias, de forma ilimitada.  
Coleta de Dados

## Criação do dataset

- Na raíz do diretório, crie a pasta dataset
- Dentro da pasta dataset, crie pastas para adicionar as imagens das pessoas (rostos) a serem detectados. Cada pasta representará um rosto.
- Crie mais uma pasta com o nome "outros", que serão rostos que não são reconhecidos.

É importante que todas as pastas tenham um número similar de imagens, se uma pasta tem 30 imagens, as outras devem ter esse número para melhores resultados.

## Criando um ambiente virtual Python
Comando para criar o ambiente:
```
python3 -m venv myenv
```
myenv pode ser modificado para o nome de seu ambiente

Iniciando o ambiente:
```
source myenv/bin/activate
```

## Passo a Passo para fins de Organização
- Coleta de dados
    - Criar o dataset das pessoas a serem reconhecidas, em diferentes angulos e iluminações
    - Organizar as imagens em pastas separadas por nome.
    
- Pré-processamento das Imagens
    - Use o OpenCV para detectar rostos nas imagens e cortar apenas as áreas relevantes.
    - Converta as imagens para tons de cinza ou normalize para melhorar o desempenho.

- Treinamento do Modelo
    - Utilize uma biblioteca como FaceNet ou DeepFace para gerar embeddings faciais.
    - Treine um classificador (por exemplo, SVM ou k-NN) usando esses embeddings para associar rostos aos nomes.

- Teste o sistema com imagens novas para validar a precisão.
    - Ajuste os limiares de similaridade para evitar falsos positivos ou negativos.

- Implantação
    - Configure o sistema para rodar em um servidor local ou em nuvem (AWS, GCP, Azure).
