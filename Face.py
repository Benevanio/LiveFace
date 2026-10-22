from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import time
from typing import Any, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np


class FaceDetector:
    """Encapsula o pipeline de detecção facial do MediaPipe."""

    def __init__(self, min_confidence: float = 0.6, model_selection: int = 0) -> None:
        self.min_confidence = min_confidence
        self.model_selection = model_selection
        self._mp_face = mp.solutions.face_detection
        self._mp_drawing = mp.solutions.drawing_utils
        self._detector: Optional[Any] = None

    def __enter__(self) -> FaceDetector:
        self._detector = self._mp_face.FaceDetection(
            model_selection=self.model_selection,
            min_detection_confidence=self.min_confidence,
        )
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._detector is not None:
            self._detector.close()
            self._detector = None

    def process_frame(self, frame_bgr: np.ndarray) -> List[Any]:
        """Converte frame para RGB, processa com MediaPipe e retorna detecções."""
        if self._detector is None:
            raise RuntimeError("FaceDetector não foi inicializado dentro de um context manager.")

        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        # Otimização recomendada pelo MediaPipe para evitar cópia desnecessária de memória
        rgb_frame.flags.writeable = False
        results = self._detector.process(rgb_frame)
        rgb_frame.flags.writeable = True

        return results.detections or []

    def draw_detections(self, frame_bgr: np.ndarray, detections: List[Any]) -> None:
        """Desenha os pontos de referência e caixas delimitadoras no frame fornecido."""
        for detection in detections:
            self._mp_drawing.draw_detection(frame_bgr, detection)


class LiveFaceApp:
    """Controlador da interface, câmera e loop de processamento."""

    def __init__(
        self,
        camera_index: int = 0,
        output_dir: Path = Path("capturas_faciais"),
        min_confidence: float = 0.6,
        model_selection: int = 0,
        save_crop: bool = True,
    ) -> None:
        self.camera_index = camera_index
        self.output_dir = output_dir
        self.min_confidence = min_confidence
        self.model_selection = model_selection
        self.save_crop = save_crop
        self.window_name = "LiveFace - Detecção e Captura"

        # Notificações visuais na tela (Toast HUD)
        self.notification_msg: str = ""
        self.notification_color: Tuple[int, int, int] = (0, 255, 0)
        self.notification_expiry: float = 0.0

    def set_notification(
        self,
        message: str,
        color: Tuple[int, int, int] = (0, 255, 0),
        duration: float = 2.0,
    ) -> None:
        """Define uma notificação visual na tela com temporizador."""
        self.notification_msg = message
        self.notification_color = color
        self.notification_expiry = time.time() + duration

    def _save_capture(self, raw_frame: np.ndarray, detections: List[Any]) -> None:
        """Salva a imagem original limpa e opcionalmente o recorte facial."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"captura_{timestamp}.jpg"
        filepath = self.output_dir / filename

        success = cv2.imwrite(str(filepath), raw_frame)
        if not success:
            self.set_notification("ERRO AO SALVAR IMAGEM!", (0, 0, 255))
            print(f"[ERRO] Falha ao gravar a imagem em: {filepath.resolve()}")
            return

        print(f"[OK] Imagem salva: {filepath.resolve()}")

        # Salva recorte do primeiro rosto, se habilitado
        if self.save_crop and detections:
            self._save_face_crop(raw_frame, detections[0], timestamp)

        self.set_notification("FOTO CAPTURADA COM SUCESSO!", (0, 255, 0))

    def _save_face_crop(self, raw_frame: np.ndarray, detection: Any, timestamp: str) -> None:
        """Recorta e salva individualmente a face detectada com margem de segurança."""
        h, w, _ = raw_frame.shape
        bbox = detection.location_data.relative_bounding_box
        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        box_w = int(bbox.width * w)
        box_h = int(bbox.height * h)

        # Margem de segurança de 15% ao redor do rosto
        pad_x = int(box_w * 0.15)
        pad_y = int(box_h * 0.15)

        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w, x + box_w + pad_x)
        y2 = min(h, y + box_h + pad_y)

        face_crop = raw_frame[y1:y2, x1:x2]
        if face_crop.size > 0:
            crop_path = self.output_dir / f"crop_{timestamp}.jpg"
            cv2.imwrite(str(crop_path), face_crop)
            print(f"[OK] Recorte facial salvo: {crop_path.resolve()}")

    def run(self) -> None:
        """Inicia a captura da webcam e loop principal de inferência."""
        webcam = cv2.VideoCapture(self.camera_index)
        if not webcam.isOpened():
            print(f"Erro: Não foi possível acessar a câmera no índice {self.camera_index}.")
            return

        prev_time = time.time()

        try:
            with FaceDetector(self.min_confidence, self.model_selection) as detector:
                print("=" * 48)
                print(" LiveFace - Sistema Iniciado")
                print(" Pressione [S] para Capturar | [Q/ESC] para Sair")
                print("=" * 48)

                while True:
                    success, raw_frame = webcam.read()
                    if not success:
                        print("Aviso: Falha ao obter quadro da câmera.")
                        break

                    # Cópia para overlay de exibição (mantém raw_frame limpo para gravação)
                    display_frame = raw_frame.copy()

                    detections = detector.process_frame(raw_frame)
                    face_count = len(detections)

                    # Desenha as marcações faciais apenas no frame de exibição
                    detector.draw_detections(display_frame, detections)

                    # Cálculo de FPS
                    curr_time = time.time()
                    fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0.0
                    prev_time = curr_time

                    # HUD: Contador de rostos e status
                    hud_color = (0, 255, 0) if face_count > 0 else (0, 165, 255)
                    cv2.putText(
                        display_frame,
                        f"Rostos: {face_count} | FPS: {int(fps)}",
                        (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        hud_color,
                        2,
                    )

                    # HUD: Instruções de rodapé
                    cv2.putText(
                        display_frame,
                        "[S] Capturar  |  [Q/ESC] Sair",
                        (15, display_frame.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )

                    # Notificação temporária na tela (Toast HUD)
                    if time.time() < self.notification_expiry:
                        cv2.putText(
                            display_frame,
                            self.notification_msg,
                            (15, 65),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            self.notification_color,
                            2,
                        )

                    cv2.imshow(self.window_name, display_frame)

                    key = cv2.waitKey(1) & 0xFF

                    # Tecla S (maiúsculo ou minúsculo)
                    if key in (ord("s"), ord("S")):
                        if face_count == 0:
                            self.set_notification("Nenhum rosto detectado!", (0, 0, 255))
                            print("Aviso: Nenhum rosto detectado. Captura cancelada.")
                            continue
                        self._save_capture(raw_frame, detections)

                    # Tecla Q ou ESC para sair
                    elif key in (ord("q"), ord("Q"), 27):
                        print("Encerrando aplicação...")
                        break

        except Exception as error:
            print(f"Erro inesperado no módulo facial: {error}")

        finally:
            webcam.release()
            cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="LiveFace - Detecção e Captura Facial em Tempo Real")
    parser.add_argument("--camera", type=int, default=0, help="Índice da câmera (padrão: 0)")
    parser.add_argument("--confidence", type=float, default=0.6, help="Confiança mínima de detecção (padrão: 0.6)")
    parser.add_argument(
        "--model",
        type=int,
        default=0,
        choices=[0, 1],
        help="0 para alcance curto (< 2m) ou 1 para alcance longo (> 2m) (padrão: 0)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="capturas_faciais",
        help="Pasta de saída das capturas (padrão: capturas_faciais)",
    )
    parser.add_argument(
        "--no-crop",
        action="store_true",
        help="Desativa o salvamento do recorte do rosto",
    )

    args = parser.parse_args()

    app = LiveFaceApp(
        camera_index=args.camera,
        output_dir=Path(args.output_dir),
        min_confidence=args.confidence,
        model_selection=args.model,
        save_crop=not args.no_crop,
    )
    app.run()


if __name__ == "__main__":
    main()
