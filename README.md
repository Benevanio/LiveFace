# Detecção Facial com Python

Projeto simples de detecção facial em tempo real utilizando Python, OpenCV e MediaPipe.

## Funcionalidades

- Detectar rostos utilizando a webcam.
- Exibir a quantidade de rostos detectados.
- Destacar os rostos identificados na imagem.
- Capturar e salvar imagens pressionando uma tecla.
- Encerrar a execução pelo teclado.

## Tecnologias utilizadas

- Python
- OpenCV
- MediaPipe

## Instalação

Instale as dependências executando:

```bash
python -m pip install opencv-python mediapipe==0.10.21
```

> A versão do MediaPipe acima mantém compatibilidade com a API `mp.solutions` utilizada neste projeto.

## Como executar

Execute o arquivo principal:

```bash
python Face.py
```

A webcam será aberta e o sistema começará a detectar rostos em tempo real.

## Comandos

- **S:** capturar e salvar a imagem.
- **Q:** encerrar o programa.

## Armazenamento das imagens

As imagens capturadas são salvas automaticamente no diretório `capturas_faciais`, criado na raiz do projeto.

## Requisitos

- Python instalado.
- Webcam disponível e acessível.
- Dependências instaladas corretamente.