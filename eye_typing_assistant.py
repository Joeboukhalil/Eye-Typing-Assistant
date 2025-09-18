import cv2
import mediapipe as mp
import pyttsx3
import time
import string
import math

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
mp_draw = mp.solutions.drawing_utils

# Initialize Text-to-Speech
engine = pyttsx3.init()

# Keyboard letters + special buttons
keys = list(string.ascii_uppercase) + ['SPACE', 'DEL', 'SPEAK']
num_cols = 8  # letters per row

# Blink detection variables
eyes_closed = False
blink_start = None
blink_thresh = 0.25
last_blink_time = 0
blink_timeout = 1.5
blink_count = 0

# Mouth detection threshold
mouth_open_thresh = 0.05  # adjust if needed
mouth_was_open = False  # track mouth state

# Text output
text_output = ""
current_index = 0  # highlighted letter

# Start video capture
cap = cv2.VideoCapture(0)

def eye_aspect_ratio(landmarks, indices):
    top = landmarks[indices[1]]
    bottom = landmarks[indices[2]]
    left = landmarks[indices[0]]
    right = landmarks[indices[3]]
    vert = math.hypot(top[1]-bottom[1], top[0]-bottom[0])
    horz = math.hypot(left[0]-right[0], left[1]-right[1])
    return vert / horz

def mouth_aspect_ratio(landmarks):
    # Using landmarks 13 (upper lip) and 14 (lower lip)
    vert = math.hypot(landmarks[13].x - landmarks[14].x, landmarks[13].y - landmarks[14].y)
    return vert

def handle_key(k):
    global text_output
    if k == 'DEL':
        text_output = text_output[:-1]
    elif k == 'SPACE':
        text_output += ' '
    elif k == 'SPEAK':
        engine.say(text_output)
        engine.runAndWait()
    else:
        text_output += k

def draw_keyboard(frame, keys, current_index):
    h, w, _ = frame.shape
    start_x, start_y = 50, 50
    step_x, step_y = 80, 50

    for i, k in enumerate(keys):
        row = i // num_cols
        col = i % num_cols
        x = start_x + col * step_x
        y = start_y + row * step_y
        color = (0,0,255) if i == current_index else (200,200,200)
        cv2.putText(frame, k, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Return bottom y coordinate of last row for placing typed text
    last_row = (len(keys)-1) // num_cols
    text_y = start_y + (last_row + 1) * step_y + 30  # 30 pixels below last row
    return text_y

def draw_text_area(frame, text, y_position):
    cv2.rectangle(frame, (50, y_position-40), (900, y_position+20), (50,50,50), -1)
    cv2.putText(frame, "Typed: " + text, (60, y_position), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    current_time = time.time()

    if results.multi_face_landmarks:
        face = results.multi_face_landmarks[0].landmark
        h, w, c = frame.shape

        # Eye landmarks
        left_eye = [face[33], face[133], face[159], face[145]]
        right_eye = [face[362], face[263], face[386], face[374]]

        left_eye_pts = [(int(pt.x*w), int(pt.y*h)) for pt in left_eye]
        right_eye_pts = [(int(pt.x*w), int(pt.y*h)) for pt in right_eye]

        # Draw eye dots
        for pt in left_eye_pts + right_eye_pts:
            cv2.circle(frame, pt, 2, (0,255,0), -1)

        # Blink detection
        left_ear = eye_aspect_ratio(left_eye_pts, [0,2,3,1])
        right_ear = eye_aspect_ratio(right_eye_pts, [0,2,3,1])
        ear = (left_ear + right_ear)/2

        if ear < blink_thresh:
            if not eyes_closed:
                eyes_closed = True
                blink_start = current_time
        else:
            if eyes_closed:
                eyes_closed = False
                # Count sequential blinks
                if current_time - last_blink_time > blink_timeout:
                    blink_count = 1
                else:
                    blink_count += 1
                last_blink_time = current_time

                if blink_count == 2:
                    # 2 blinks → move to next letter
                    current_index = (current_index + 1) % len(keys)
                    blink_count = 0

        # Mouth detection
        mar = mouth_aspect_ratio(face)
        if mar > mouth_open_thresh and not mouth_was_open:
            # Mouth just opened → select current letter
            handle_key(keys[current_index])
            mouth_was_open = True
        elif mar <= mouth_open_thresh and mouth_was_open:
            # Mouth closed → reset state
            mouth_was_open = False

    # Draw keyboard and typed text
    text_y = draw_keyboard(frame, keys, current_index)
    draw_text_area(frame, text_output, text_y)

    cv2.imshow("Eye Typing", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
