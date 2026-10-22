# Detecção Facial com Python (LiveFace)

Projeto de detecção e captura facial em tempo real utilizando **Python**, **OpenCV** e **MediaPipe**.

## Funcionalidades

- **Detecção em Tempo Real:** Identificação rápida e precisa de rostos com MediaPipe.
- **Captura Limpa (Clean Frame):** As fotos são salvas sem anotações ou caixas delimitadoras sobrepostas, prontas para cadastros ou biometria.
- **Recorte Facial Automático:** Salva opcionalmente o recorte (*crop*) aproximado do rosto com margem de segurança.
- **Interface e HUD:** Contador de rostos detectados, monitor de FPS e alertas visuais de confirmação de captura diretamente no vídeo.
- **Controle Flexível via CLI:** Suporte a seleção de câmera, nível de confiança, alcance do modelo e pastas de saída.
- **Prevenção de Colisões:** Nomenclatura com milissegundos para evitar sobrescrita de imagens.

## Tecnologias Utilizadas

- Python 3.8+
- [OpenCV](https://opencv.org/) (`opencv-python`)
- [MediaPipe](https://developers.google.com/mediapipe) (`mediapipe==0.10.21`)
- NumPy

## Instalação

1. Clone o repositório ou navegue até o diretório do projeto:
   ```bash
   git clone <url-do-repositorio>
   cd LiveFace
   ```

2. Crie e ative um ambiente virtual (recomendado):
   ```bash
   python -m venv .venv
   # No Windows:
   .venv\Scripts\activate
   # No Linux/macOS:
   source .venv/bin/activate
   ```

3. Instale as dependências:
   ```bash
   python -m pip install -r requirements.txt
   ```

> **Nota:** A versão `mediapipe==0.10.21` mantém total compatibilidade com a API `mp.solutions` utilizada no projeto.

## Como Executar

### Execução Padrão:
```bash
python Face.py
```

### Opções de Linha de Comando:
```bash
# Usar outra câmera conectada (índice 1)
python Face.py --camera 1

# Ajustar a confiança mínima de detecção
python Face.py --confidence 0.75

# Modelo de longo alcance (> 2 metros de distância)
python Face.py --model 1

# Definir outra pasta de destino
python Face.py --output-dir fotos_visitantes

# Desativar o salvamento do recorte do rosto
python Face.py --no-crop
```

Para ver todos os parâmetros disponíveis:
```bash
python Face.py --help
```

## Comandos do Teclado

| Tecla | Ação |
| :--- | :--- |
| **S** ou **s** | Captura e salva a imagem limpa e o recorte facial. |
| **Q**, **q** ou **ESC** | Encerra a aplicação com segurança. |

## Armazenamento das Imagens

As imagens capturadas são salvas automaticamente no diretório `capturas_faciais/` (ou no caminho especificado via `--output-dir`), seguindo o padrão:
- `captura_YYYYMMDD_HHMMSS_mmm.jpg` (imagem completa original limpa)
- `crop_YYYYMMDD_HHMMSS_mmm.jpg` (recorte facial com margem)

*(O diretório `capturas_faciais/` é ignorado pelo Git por motivos de privacidade e conformidade com a LGPD).*
