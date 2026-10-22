
import cv2
import mediapipe as mp
import os
from datetime import datetime
CAMERA_INDEX = 0
OUTPUT_DIR = "capturas_faciais"
MIN_CONFIDENCE = 0.6

os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    webcam = cv2.VideoCapture(CAMERA_INDEX)

    if not webcam.isOpened():
        print("Erro: não foi possível acessar a câmera.")
        return

    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils

    try:
        with mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=MIN_CONFIDENCE
        ) as face_detection:

            print("Bem-vindo ao sistema")
            print("S: capturar imagem | Q: sair")

            while True:
                success, frame = webcam.read()

                if not success:
                    print("Erro ao capturar imagem da câmera.")
                    break

                rgb_frame = cv2.cvtColor(
                    frame, cv2.COLOR_BGR2RGB
                )

                results = face_detection.process(rgb_frame)

                detections = results.detections or []
                face_count = len(detections)

                for detection in detections:
                    mp_drawing.draw_detection(
                        frame, detection
                    )

                color = (0, 255, 0) if face_count else (0, 165, 255)

                cv2.putText(
                    frame,
                    f"Rostos detectados: {face_count}",
                    (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
                )

                cv2.putText(
                    frame,
                    "S: Capturar | Q: Sair",
                    (15, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

                cv2.imshow("Face - Portaria", frame)

                key = cv2.waitKey(1) & 0xFF

                if key == ord("s"):
                    if face_count == 0:
                        print("Nenhum rosto detectado. Captura cancelada.")
                        continue

                    timestamp = datetime.now().strftime(
                        "%Y%m%d_%H%M%S"
                    )

                    filename = f"captura_{timestamp}.jpg"
                    filepath = os.path.join(
                        OUTPUT_DIR, filename
                    )

                    if cv2.imwrite(filepath, frame):
                        print(
                            "Imagem salva:",
                            os.path.abspath(filepath)
                        )
                    else:
                        print("Falha ao salvar a imagem.")

                elif key == ord("q"):
                    break

    except Exception as error:
        print(f"Erro no módulo facial: {error}")

    finally:
        webcam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

