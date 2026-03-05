from xxlimited_35 import Null

import cv2
import time
import os
from datetime import datetime

def mkdirectory(path):
    try:
        os.makedirs(path)
        print(f"Nested directories '{path}' created successfully.")
    except FileExistsError:
        print(f"One or more directories in '{path}' already exist.")
    except PermissionError:
        print(f"Permission denied: Unable to create '{path}'.")
    except Exception as e:
        print(f"An error occurred: {e}")

def fine_registrazione(prev_timestamp, now_timestamp):
    # TRUE: se timestamp attuale - timestamp salvato maggiore del tempo fissato, esco dal ciclo
    # FALSE: Altrimenti
    recording_minutes = 1
    return now_timestamp - prev_timestamp > recording_minutes * 60
    return

def main():
    # apertura camera
    cap = cv2.VideoCapture(0)
    first_frame = None

    # Informazioni dimensioni cattura video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    #Variabili controllo per salvataggio files
    movimento_rilevato = False
    salvataggio_video = False
    prev_timestamp = 0

    print("Sistema pronto. Premi 'r' per resettare lo sfondo, 'q' per uscire.")

    for i in range (2):
        print("Avvio in corso ...", i)
        time.sleep(1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # --- PRE-PROCESSING ---
        # Lavoro in scala di grigi
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # Salvo primo frame per comparison dei pixel
        if first_frame is None:
            first_frame = gray
            continue

        # Calcolo differenza tra i pixel
        delta = cv2.absdiff(first_frame, gray)
        _, thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=4)

        # Trovo tutti i contorni sulla differenza dei pixel;
        # non e' piu' necessario copiare l'oggetto, ma e' buona abitudine farlo
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            movimento_rilevato = True
            # 1. Area del contorno trovato
            area = cv2.contourArea(contour)

            # 2. Elimino bounding box troppo piccole
            if area < 30000:
                continue
            # 3. Coordinate del rettangolo
            (x, y, w, h) = cv2.boundingRect(contour)

            # 4. Disegniamo sul frame originale (BGR)
            # Parametri: (immagine, (x1, y1), (x2, y2), colore, spessore)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Movimento Rilevato!", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Salvo la data e la converto in timestamp
        my_datetime = datetime.now()
        timestamp = my_datetime.timestamp()

        # Creo ambiente per salvataggio video
        if movimento_rilevato and not salvataggio_video:
            salvataggio_video = True
            prev_timestamp = timestamp
            # Ottengo strighe di data e ora per le cartelle
            my_date = str(my_datetime)[0:10].replace("-","_")
            my_time = str(my_datetime)[11:19].replace(":","_")
            print(my_date)
            print(my_time)

            # Creo cartella con il giorno, poi la creo con l'ora (se non esistono)
            nested_directory = "recordings/" + my_date + "/" + my_time
            mkdirectory(nested_directory)
            out = cv2.VideoWriter(nested_directory + "/output.mp4",cv2.VideoWriter.fourcc(*'mp4v'),20,(frame_width,frame_height))

        # Salvataggio video
        if salvataggio_video:
            out.write(frame)
            # se finisce il tempo della registrazione, resetto variabili e interrompo salvataggio video
            if fine_registrazione(prev_timestamp,timestamp):
                out.release()
                prev_timestamp = 0
                salvataggio_video = False
                continue

        # Visualizziamo i risultati
        cv2.imshow("Originale con Bounding Box", frame)
        # cv2.imshow("Thresh", thresh)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            first_frame = None

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()