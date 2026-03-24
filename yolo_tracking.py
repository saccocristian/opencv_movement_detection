import cv2
import torch
from ultralytics import YOLO

# 1. Scarico modello e lo importo
model = YOLO('yolov8s.pt')

# 2. Voglio usare GPU Apple: forzo utilizzo

if torch.backends.mps.is_available():
    device = 'mps' # Metal Performance Shaders
else:
    device = 'cpu'

print("Device:",device)

# 3. Apreo videocamera e imposto variabili
cap = cv2.VideoCapture(0)

oggetti_contati = set()

while cap.isOpened():
    success,frame = cap.read()
    if not success:
        break

    # Inferenza: il modello ricava le bounding box, livello di confidenza dell'oggetto rilevato
    results = model.track(frame,persist=True,device = device, classes = [39], conf = 0.4)

    if results[0].boxes.id is not None: # se non abbiamo bottiglie non faccio niente
        ids = results[0].boxes.id.int().cpu().tolist()
        for obj_id in ids:
            if obj_id not in oggetti_contati:
                oggetti_contati.add(obj_id)
                print(f"Nuova bottiglia rilevata! ID: {obj_id} | Totale: {len(oggetti_contati)}")

            # 5. Visualizzazione professionale
    annotated_frame = results[0].plot()

    # Box informativo in alto a sinistra
    cv2.rectangle(annotated_frame, (10, 10), (350, 100), (0, 0, 0), -1)
    cv2.putText(annotated_frame, f"Conteggio: {len(oggetti_contati)}", (20, 70),
         cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    cv2.imshow("M4 AI - Laboratorio Capo Supremo", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print(f"Analisi conclusa. Bottiglie totali trovate: {len(oggetti_contati)}")