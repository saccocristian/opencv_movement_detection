import cv2
from ultralytics import YOLO


def main():
    # 1. Inizializzazione: Carichiamo il modello YOLOv8 Nano (leggero e fulmineo)
    # Se non è presente, lo scaricherà automaticamente nella cartella corrente
    model = YOLO('yolov8n.pt')

    # 2. Apertura Sorgente: 0 è solitamente la webcam integrata
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Errore: Capo Supremo, non riesco a trovare la webcam!")
        return

    print("Sistema avviato. Premi 'q' per uscire.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 3. Inferenza: YOLO analizza il frame
        # stream=True rende l'elaborazione più fluida per i video
        results = model(frame, stream=True, conf = 0.7, classes = [0])

        # 4. Visualizzazione: Disegniamo i risultati sul frame
        for r in results:
            annotated_frame = r.plot()  # Crea un'immagine con i rettangoli e le etichette

            # Mostriamo il risultato in una finestra
            cv2.imshow("YOLOv8 Test - Real Time", annotated_frame)

        # Esci se premi il tasto 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Pulizia finale
    cap.release()
    cv2.destroyAllWindows()
    print("Sistema spento. Alla prossima, Capo Supremo.")


if __name__ == "__main__":
    main()