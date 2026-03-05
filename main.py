import cv2
import time
import os
from datetime import datetime


def main():
    cap = cv2.VideoCapture(0)
    first_frame = None

    print("Sistema pronto. Premi 'r' per resettare lo sfondo, 'q' per uscire.")

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

        # Trovo tutti i contorni sulla differenza dei pixel; non e' piu' necessario copiare l'oggetto,
        # ma e' buona abitudine farlo
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        movimento_rilevato = False
        salvataggio_video = False

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

        # Salviamo video relativo a movimento
        if movimento_rilevato and not salvataggio_video:
            salvataggio_video = True
            # Salvo la data, la converto in timestamp e ottengo strighe di data e ora per le cartelle
            my_datetime = datetime.now()
            timestamp = my_datetime.timestamp()
            my_date = str(my_datetime)[0:10].replace("-","_")
            my_time = str(my_datetime)[11:19].replace(":","_")
            print(my_date)
            print(my_time)
            # Creo cartella con il giorno, poi la creo con l'ora (se non esistono)
            nested_directory = my_date + "/" + my_time

            # Create nested directories
            try:
                os.makedirs(nested_directory)
                print(f"Nested directories '{nested_directory}' created successfully.")
            except FileExistsError:
                print(f"One or more directories in '{nested_directory}' already exist.")
            except PermissionError:
                print(f"Permission denied: Unable to create '{nested_directory}'.")
            except Exception as e:
                print(f"An error occurred: {e}")

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