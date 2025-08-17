# main.py - v35 (New Finger-Length Size Limit)
# This version changes the maximum size of the energy ball to be
# limited by the length of the index finger.

import cv2
import mediapipe as mp
import math
import time
import random
import numpy as np

# --- 1. Initialization ---
mp_hands = mp.solutions.hands
hands = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream from webcam.")
    exit()

# --- 2. Projectile Class ---
projectiles = []
class Projectile:
    def __init__(self, x, y, radius):
        self.x = x
        self.y = y
        self.radius = radius
        self.velocity = 20

    def update(self):
        self.y -= self.velocity

    def draw(self, image):
        draw_final_energy_ball(image, (int(self.x), int(self.y)), self.radius)

# --- 3. Gesture Detection Function ---
def is_specific_gesture(hand_landmarks):
    try:
        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
        middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        middle_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
        ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
        ring_pip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]
        pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
        pinky_pip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]
        index_straight = index_tip.y < index_pip.y
        middle_straight = middle_tip.y < middle_pip.y
        ring_bent = ring_tip.y > ring_pip.y
        pinky_bent = pinky_tip.y > pinky_pip.y
        if index_straight and middle_straight and ring_bent and pinky_bent:
            return True
    except: return False
    return False

# --- 4. Drawing & Effect Functions ---
def draw_final_energy_ball(image, center, radius):
    center_color = (220, 50, 180)
    edge_color = (120, 10, 90)
    for i in range(radius, 0, -2):
        r = int(edge_color[0] + (center_color[0] - edge_color[0]) * (1 - i/radius))
        g = int(edge_color[1] + (center_color[1] - edge_color[1]) * (1 - i/radius))
        b = int(edge_color[2] + (center_color[2] - edge_color[2]) * (1 - i/radius))
        cv2.circle(image, center, i, (b, g, r), -1)
    num_bolts = max(3, int(radius / 15))
    bolt_thickness = max(1, int(radius / 35))
    for _ in range(num_bolts):
        start_angle = random.uniform(0, 2 * math.pi)
        start_point = (int(center[0] + radius * math.cos(start_angle)), int(center[1] + radius * math.sin(start_angle)))
        end_angle = start_angle + math.pi + random.uniform(-math.pi/4, math.pi/4)
        end_point = (int(center[0] + radius * math.cos(end_angle)), int(center[1] + radius * math.sin(end_angle)))
        draw_lightning_bolt(image, start_point, end_point, radius / 4, bolt_thickness)

def draw_lightning_bolt(image, pt1, pt2, max_offset, thickness):
    if max_offset < 1:
        cv2.line(image, pt1, pt2, (255, 255, 255), thickness)
        return
    mid_x = (pt1[0] + pt2[0]) // 2 + int(random.uniform(-max_offset, max_offset))
    mid_y = (pt1[1] + pt2[1]) // 2 + int(random.uniform(-max_offset, max_offset))
    mid_point = (mid_x, mid_y)
    draw_lightning_bolt(image, pt1, mid_point, max_offset / 2, thickness)
    draw_lightning_bolt(image, mid_point, pt2, max_offset / 2, thickness)

def draw_unstable_aura(image, center, radius):
    overlay = image.copy()
    for i in range(4):
        aura_radius = radius + random.randint(5, 20)
        alpha = random.uniform(0.05, 0.15)
        color = (200, 30, 150)
        cv2.circle(overlay, center, aura_radius, color, -1)
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

def apply_lensing_distortion(image, center, radius, intensity):
    """Applies a gravitational lensing distortion effect (Optimized)."""
    if radius <= 0 or intensity <= 0: return

    # Define the region of interest (ROI) around the effect
    x_start = max(0, center[0] - radius)
    y_start = max(0, center[1] - radius)
    x_end = min(image.shape[1], center[0] + radius)
    y_end = min(image.shape[0], center[1] + radius)
    
    roi = image[y_start:y_end, x_start:x_end]
    if roi.size == 0: return

    rows, cols, _ = roi.shape
    roi_center = (center[0] - x_start, center[1] - y_start)
    
    # Create maps only for the ROI, not the whole image
    map_x = np.zeros((rows, cols), dtype=np.float32)
    map_y = np.zeros((rows, cols), dtype=np.float32)

    for r in range(rows):
        for c in range(cols):
            dist_x = c - roi_center[0]
            dist_y = r - roi_center[1]
            distance = math.sqrt(dist_x**2 + dist_y**2)

            if distance < radius:
                new_dist = distance * (1 - intensity * math.pow(1 - distance / radius, 2))
                angle = math.atan2(dist_y, dist_x)
                map_x[r, c] = roi_center[0] + new_dist * math.cos(angle)
                map_y[r, c] = roi_center[1] + new_dist * math.sin(angle)
            else:
                map_x[r, c] = c
                map_y[r, c] = r
                
    distorted_roi = cv2.remap(roi, map_x, map_y, interpolation=cv2.INTER_LINEAR)
    image[y_start:y_end, x_start:x_end] = distorted_roi


# --- 5. State Variables ---
energy_ball_charge = 0.0
prev_wrist_z = 0.0

# --- 6. Main Application Loop ---
while cap.isOpened():
    success, image = cap.read()
    if not success: continue

    for p in projectiles[:]:
        p.update()
        p.draw(image)
        if p.y < 0: projectiles.remove(p)

    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    h, w, _ = image.shape

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
        wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
        
        # *** CHANGE HERE: Max radius is now the distance between the index fingertip and its base knuckle. ***
        max_radius = math.sqrt(((index_tip.x - index_mcp.x) * w)**2 + ((index_tip.y - index_mcp.y) * h)**2)

        if is_specific_gesture(hand_landmarks):
            energy_ball_charge += 4.0
        elif energy_ball_charge > 0:
            energy_ball_charge -= 5.0
        energy_ball_charge = max(0, min(energy_ball_charge, max_radius))

        if energy_ball_charge > 0:
            radius = int(energy_ball_charge)
            center_pt = (int(index_tip.x * w), int(index_tip.y * h))
            distortion_intensity = radius / 50.0 
            apply_lensing_distortion(image, center_pt, radius + 20, distortion_intensity)
            draw_unstable_aura(image, center_pt, radius)

        # Simplified hand wireframe
        glow_color = (200, 30, 150) # Constant purple tint
        glow_thickness = 1
        for connection in mp_hands.HAND_CONNECTIONS:
            start_pt = (int(hand_landmarks.landmark[connection[0]].x * w), int(hand_landmarks.landmark[connection[0]].y * h))
            end_pt = (int(hand_landmarks.landmark[connection[1]].x * w), int(hand_landmarks.landmark[connection[1]].y * h))
            cv2.line(image, start_pt, end_pt, glow_color, glow_thickness)

        current_wrist_z = wrist.z
        # Casting logic now uses the new max_radius to determine if the ball is "full"
        if prev_wrist_z != 0 and (prev_wrist_z - current_wrist_z) > 0.03 and energy_ball_charge >= max_radius * 0.9:
            projectiles.append(Projectile(index_tip.x * w, index_tip.y * h, int(energy_ball_charge)))
            energy_ball_charge = 0
        prev_wrist_z = current_wrist_z

        if energy_ball_charge > 0:
            radius = int(energy_ball_charge)
            if radius > 0:
                draw_final_energy_ball(image, (int(index_tip.x * w), int(index_tip.y * h)), radius)

    cv2.imshow('AR Hand Tracker - Final Version', image)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
