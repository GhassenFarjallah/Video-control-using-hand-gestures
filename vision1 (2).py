import cv2
import mediapipe as mp
import pyautogui
import time
from collections import deque

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Réduire la résolution de la caméra pour améliorer les performances
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

last_gesture_time = 0  # Pour suivre le temps de la dernière action de geste
current_action = "Aucune"  # Variable pour stocker l'action actuelle à afficher
pause_start_time = 0  # Suivre quand le geste "pause" est détecté

gesture_window = deque(maxlen=10)  # Fenêtre temporelle pour stocker les gestes reconnus

def count_fingers(hand_landmarks):
    """
    Fonction pour compter le nombre de doigts levés en fonction des points de repère.
    Retourne le nombre de doigts levés.
    """
    fingers = []

    # Pouce
    if hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x < hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x:
        fingers.append(1)  # Le pouce est ouvert
    else:
        fingers.append(0)  # Le pouce est fermé

    # 4 Doigts
    for id in range(1, 5):
        if hand_landmarks.landmark[mp_hands.HandLandmark(id * 4)].y < hand_landmarks.landmark[mp_hands.HandLandmark(id * 4 - 2)].y:
            fingers.append(1)  # Doigt est ouvert
        else:
            fingers.append(0)  # Doigt est fermé

    return sum(fingers)

def recognize_gesture(multi_hand_landmarks):
    """
    Fonction pour reconnaître les gestes de la main.
    Retourne le geste reconnu sous forme de chaîne et le nombre total de doigts.
    """
    total_fingers = 0
    for hand_landmarks in multi_hand_landmarks:
        total_fingers += count_fingers(hand_landmarks)

    # Logique pour la reconnaissance des gestes en fonction du nombre de doigts
    if total_fingers == 1:
        return "pause", total_fingers
    elif total_fingers == 2:
        return "avance", total_fingers
    elif total_fingers == 3:
        return "recul", total_fingers
    elif total_fingers == 4:
        return "augmenter-volume", total_fingers
    elif total_fingers == 5:
        return "diminuer-volume", total_fingers
    elif total_fingers == 6:
        return "accelerer-video", total_fingers
    elif total_fingers >= 7:
        return "ralentir-video", total_fingers
    else:
        return "aucun", total_fingers

def determine_consistent_gesture():
    """
    Fonction pour déterminer le geste dominant dans la fenêtre temporelle.
    Retourne le geste le plus fréquent dans la fenêtre.
    """
    if not gesture_window:
        return "aucun", 0
    gesture_counts = {}
    for gesture, num_fingers in gesture_window:
        if gesture in gesture_counts:
            gesture_counts[gesture] += 1
        else:
            gesture_counts[gesture] = 1

    # Retourne le geste avec la fréquence maximale
    consistent_gesture = max(gesture_counts, key=gesture_counts.get)
    for gesture, num_fingers in gesture_window:
        if gesture == consistent_gesture:
            return consistent_gesture, num_fingers
    return "aucun", 0

def control_application(gesture, num_fingers):
    global last_gesture_time, current_action, pause_start_time
    current_time = time.time()

    # Gestion du geste "pause"
    if gesture == "pause":
        if pause_start_time == 0:
            pause_start_time = current_time  # Commence le compte à rebours de 3 secondes
        elif current_time - pause_start_time >= 3:  # Déclenche la pause après 3 secondes
            print("Geste de pause détecté - Appuyer sur espace pour mettre en pause")
            pyautogui.press('space')
            current_action = "En pause"
            last_gesture_time = current_time
            pause_start_time = 0  # Réinitialiser le minuteur
    else:
        pause_start_time = 0  # Réinitialiser si le geste change

    # Gestion de l'avance avec 2 doigts
    if gesture == "avance" and current_time - last_gesture_time >= 0.5:
        print("Geste d'avance détecté - Appuyer sur la flèche droite")
        pyautogui.press('right')
        current_action = "Avance rapide"
        last_gesture_time = current_time

    # Gestion du recul avec 3 doigts
    elif gesture == "recul" and current_time - last_gesture_time >= 0.5:
        print("Geste de recul détecté - Appuyer sur la flèche gauche")
        pyautogui.press('left')
        current_action = "Reculer"
        last_gesture_time = current_time

    # Gestion de l'augmentation du volume avec 4 doigts
    elif gesture == "augmenter-volume" and current_time - last_gesture_time >= 0.5:
        print("Geste d'augmentation du volume détecté - Appuyer sur la flèche haut")
        pyautogui.press('up')
        current_action = "Augmenter le volume"
        last_gesture_time = current_time

    # Gestion de la diminution du volume avec 5 doigts
    elif gesture == "diminuer-volume" and current_time - last_gesture_time >= 0.5:
        print("Geste de diminution du volume détecté - Appuyer sur la flèche bas")
        pyautogui.press('down')
        current_action = "Diminuer le volume"
        last_gesture_time = current_time

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        gesture, num_fingers = recognize_gesture(result.multi_hand_landmarks)
        gesture_window.append((gesture, num_fingers))

        consistent_gesture, consistent_fingers = determine_consistent_gesture()

        if consistent_gesture != "aucun":
            control_application(consistent_gesture, consistent_fingers)

        # Dessiner les points de repère pour chaque main détectée
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Afficher l'action actuelle sur l'image de la caméra
    cv2.putText(frame, f"Action: {current_action}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Contrôle par geste de la main', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
