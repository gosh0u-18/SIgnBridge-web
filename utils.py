import os
import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial import distance
from scipy.signal import savgol_filter
import warnings
warnings.filterwarnings('ignore')

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

HAND_CONNECTIONS = mp_hands.HAND_CONNECTIONS
NUM_LANDMARKS = 21
SINGLE_HAND_FEATURES = 63 
DUAL_HAND_FEATURES = 126 

def get_hand_detector(max_hands=2, detection_confidence=0.7, tracking_confidence=0.5):
    return mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=max_hands,
        min_detection_confidence=detection_confidence,
        min_tracking_confidence=tracking_confidence,
        model_complexity=1
    )

def extract_hand_landmarks_from_frame(frame, hands_detector):
    if frame is None or frame.size == 0:
        return []
    
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    
    results = hands_detector.process(image_rgb)
    
    image_rgb.flags.writeable = True
    landmarks_list = []
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            coords = []
            for lm in hand_landmarks.landmark:
                coords.extend([lm.x, lm.y, lm.z])
            landmarks_list.append(np.array(coords, dtype=np.float32))
    
    return landmarks_list

def normalize_landmarks(landmarks, reference_idx=0):
    if np.all(landmarks == 0) or len(landmarks) == 0:
        return landmarks
    
    coords = landmarks.reshape(-1, 3)
    
    reference_point = coords[reference_idx]
    coords = coords - reference_point
    
    hand_size = np.linalg.norm(coords[9])
    
    if hand_size > 0.001:
        coords = coords / hand_size
    
    coords = np.clip(coords, -3, 3)
    
    return coords.flatten().astype(np.float32)

def calculate_angles(landmarks):
    if np.all(landmarks == 0):
        return np.zeros(15, dtype=np.float32)
    
    coords = landmarks.reshape(-1, 3)
    angles = []
    
    finger_segments = [
        [0, 1, 2, 3, 4],
        [0, 5, 6, 7, 8],
        [0, 9, 10, 11, 12],
        [0, 13, 14, 15, 16],
        [0, 17, 18, 19, 20]
    ]
    
    for finger in finger_segments:
        for i in range(len(finger) - 2):
            p1 = coords[finger[i]]
            p2 = coords[finger[i + 1]]
            p3 = coords[finger[i + 2]]
            
            v1 = p1 - p2
            v2 = p3 - p2
            
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)
            
            if norm_v1 > 0.001 and norm_v2 > 0.001:
                cos_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.arccos(cos_angle)
                angles.append(angle)
            else:
                angles.append(0.0)
    
    return np.array(angles[:15], dtype=np.float32)

def calculate_finger_distances(landmarks):
    if np.all(landmarks == 0):
        return np.zeros(10, dtype=np.float32)
    
    coords = landmarks.reshape(-1, 3)
    distances = []
    
    fingertips = [4, 8, 12, 16, 20]
    
    for i in range(len(fingertips)):
        for j in range(i + 1, len(fingertips)):
            dist = np.linalg.norm(coords[fingertips[i]] - coords[fingertips[j]])
            distances.append(dist)
    
    return np.array(distances[:10], dtype=np.float32)

def calculate_palm_features(landmarks):
    if np.all(landmarks == 0):
        return np.zeros(5, dtype=np.float32)
    
    coords = landmarks.reshape(-1, 3)
    features = []
    
    base_points = [2, 5, 9, 13, 17]
    
    polygon_points = coords[base_points]
    
    if len(polygon_points) >= 3:
        polygon_2d = polygon_points[:, :2]
        area = polygon_area(polygon_2d)
        features.append(area)
    else:
        features.append(0.0)
    
    for i in range(len(base_points)):
        for j in range(i + 1, len(base_points)):
            if len(features) < 5:
                dist = np.linalg.norm(coords[base_points[i]] - coords[base_points[j]])
                features.append(dist)
    
    while len(features) < 5:
        features.append(0.0)
    
    return np.array(features[:5], dtype=np.float32)

def polygon_area(polygon):
    x = polygon[:, 0]
    y = polygon[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def extract_dynamic_features(landmarks_sequence):
    if len(landmarks_sequence) < 2:
        return np.zeros(63, dtype=np.float32)
    
    velocities = []
    
    for i in range(1, len(landmarks_sequence)):
        velocity = landmarks_sequence[i] - landmarks_sequence[i-1]
        velocities.append(velocity)
    
    if velocities:
        velocities_arr = np.vstack(velocities)
        mean_velocity = np.mean(velocities_arr, axis=0)
        std_velocity = np.std(velocities_arr, axis=0)
    else:
        mean_velocity = np.zeros_like(landmarks_sequence[0])
        std_velocity = np.zeros_like(landmarks_sequence[0])
    
    dynamic_features = np.concatenate([mean_velocity, std_velocity])
    
    if len(dynamic_features) > 63:
        dynamic_features = dynamic_features[:63]
    elif len(dynamic_features) < 63:
        padding = np.zeros(63 - len(dynamic_features))
        dynamic_features = np.concatenate([dynamic_features, padding])
    
    return dynamic_features.astype(np.float32)

def extract_standard_single_hand_features(landmarks):
    if np.all(landmarks == 0):
        return np.zeros(141, dtype=np.float32)
    
    norm_coords = normalize_landmarks(landmarks)
    joint_angles = calculate_angles(landmarks)
    finger_distances = calculate_finger_distances(landmarks)
    palm_features = calculate_palm_features(landmarks)
    
    static_features = np.concatenate([
        norm_coords,
        joint_angles,
        finger_distances,
        palm_features
    ])
    
    if len(static_features) < 141:
        padding = np.zeros(141 - len(static_features))
        static_features = np.concatenate([static_features, padding])
    
    return static_features.astype(np.float32)

def extract_extended_single_hand_features(landmarks):
    if np.all(landmarks == 0):
        return np.zeros(257, dtype=np.float32)
    
    norm_coords = normalize_landmarks(landmarks)
    joint_angles = calculate_angles(landmarks)
    finger_distances = calculate_finger_distances(landmarks)
    palm_features = calculate_palm_features(landmarks)
    
    coords = landmarks.reshape(-1, 3)
    
    wrist_distances = []
    wrist = coords[0]
    for i in range(1, 21):
        dist = np.linalg.norm(coords[i] - wrist)
        wrist_distances.append(dist)
    wrist_distances = np.array(wrist_distances[:20], dtype=np.float32)
    
    finger_lengths = []
    for tip_idx in [4, 8, 12, 16, 20]:
        if tip_idx == 4:
            base_idx = 2
        else:
            base_idx = tip_idx - 3
        length = np.linalg.norm(coords[tip_idx] - coords[base_idx])
        finger_lengths.append(length)
    finger_lengths = np.array(finger_lengths, dtype=np.float32)
    
    finger_angles = []
    finger_pairs = [(4, 8), (8, 12), (12, 16), (16, 20)]
    for tip1, tip2 in finger_pairs:
        v1 = coords[tip1] - wrist
        v2 = coords[tip2] - wrist
        
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        if norm_v1 > 0.001 and norm_v2 > 0.001:
            cos_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            finger_angles.append(angle)
        else:
            finger_angles.append(0.0)
    
    while len(finger_angles) < 10:
        finger_angles.append(0.0)
    finger_angles = np.array(finger_angles[:10], dtype=np.float32)
    
    finger_tips = coords[[4, 8, 12, 16, 20]]
    finger_bases = coords[[2, 5, 9, 13, 17]]
    middle_joints = coords[[3, 6, 10, 14, 18]]
    
    centers = [
        np.mean(finger_tips, axis=0),
        np.mean(finger_bases, axis=0),
        np.mean(middle_joints, axis=0)
    ]
    
    center_features = []
    for center in centers:
        center_features.extend(center)
    center_features = np.array(center_features[:9], dtype=np.float32)
    
    combined = np.concatenate([
        norm_coords,
        joint_angles,
        finger_distances,
        palm_features,
        wrist_distances,
        finger_lengths,
        finger_angles,
        center_features
    ])
    
    if len(combined) < 257:
        padding = np.zeros(257 - len(combined))
        combined = np.concatenate([combined, padding])
    elif len(combined) > 257:
        combined = combined[:257]
    
    return combined.astype(np.float32)

def extract_dual_hand_features(left_landmarks, right_landmarks):
    left_features = extract_extended_single_hand_features(left_landmarks)
    right_features = extract_extended_single_hand_features(right_landmarks)
    
    if not np.all(left_landmarks == 0) and not np.all(right_landmarks == 0):
        left_coords = left_landmarks.reshape(-1, 3)
        right_coords = right_landmarks.reshape(-1, 3)

        wrist_distance = np.linalg.norm(left_coords[0] - right_coords[0])
        
        joint_distances = []
        for i in range(NUM_LANDMARKS):
            dist = np.linalg.norm(left_coords[i] - right_coords[i])
            joint_distances.append(dist)
        
        mean_joint_distance = np.mean(joint_distances)
        std_joint_distance = np.std(joint_distances)
        
        relative_features = np.array([
            wrist_distance,
            mean_joint_distance,
            std_joint_distance,
            0.0
        ], dtype=np.float32)
    else:
        relative_features = np.zeros(4, dtype=np.float32)
    
    combined_features = np.concatenate([
        left_features,
        right_features,
        relative_features
    ])
    
    return combined_features.astype(np.float32)

def extract_features_for_prediction(landmarks_list, hand_mode="single"):
    if not landmarks_list:
        if hand_mode == "dual":
            return np.zeros(518, dtype=np.float32)
        else:
            return np.zeros(257, dtype=np.float32)
    
    if hand_mode == "dual" and len(landmarks_list) >= 2:
        return extract_dual_hand_features(landmarks_list[0], landmarks_list[1])
    else:
        return extract_extended_single_hand_features(landmarks_list[0])

def extract_advanced_features(video_path, sample_fps=10, max_frames=30, hand_mode="auto"):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Не може да се отвори видео: {video_path}")
    
    max_hands = 2 if hand_mode == "dual" or hand_mode == "auto" else 1
    hands_detector = get_hand_detector(max_hands=max_hands)
    
    frame_idx = 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    skip = max(1, int(fps / sample_fps))
    
    left_hand_sequence = []
    right_hand_sequence = []
    single_hand_sequence = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % skip == 0 and (len(single_hand_sequence) < max_frames or 
                                     (len(left_hand_sequence) < max_frames and len(right_hand_sequence) < max_frames)):
            
            landmarks_list = extract_hand_landmarks_from_frame(frame, hands_detector)
            
            if landmarks_list:
                if len(landmarks_list) == 1:
                    single_hand_sequence.append(landmarks_list[0])
                elif len(landmarks_list) == 2:
                    coords1 = landmarks_list[0].reshape(-1, 3)
                    coords2 = landmarks_list[1].reshape(-1, 3)
                    
                    mean_x1 = np.mean(coords1[:, 0])
                    mean_x2 = np.mean(coords2[:, 0])
                    
                    if mean_x1 < mean_x2:
                        left_hand_sequence.append(landmarks_list[0])
                        right_hand_sequence.append(landmarks_list[1])
                    else:
                        left_hand_sequence.append(landmarks_list[1])
                        right_hand_sequence.append(landmarks_list[0])
        
        frame_idx += 1
        
        if (len(single_hand_sequence) >= max_frames or 
            (len(left_hand_sequence) >= max_frames and len(right_hand_sequence) >= max_frames)):
            break
    
    hands_detector.close()
    cap.release()
    
    features_list = []
    
    if hand_mode == "dual" or (hand_mode == "auto" and len(left_hand_sequence) > 5 and len(right_hand_sequence) > 5):
        if len(left_hand_sequence) > 0 and len(right_hand_sequence) > 0:
            min_len = min(len(left_hand_sequence), len(right_hand_sequence))
            left_sequence = left_hand_sequence[:min_len]
            right_sequence = right_hand_sequence[:min_len]
            
            for left, right in zip(left_sequence, right_sequence):
                features = extract_dual_hand_features(left, right)
                features_list.append(features)
    
    if not features_list and len(single_hand_sequence) > 0:
        for landmarks in single_hand_sequence:
            features = extract_extended_single_hand_features(landmarks)
            features_list.append(features)
    
    if not features_list:
        return np.zeros(514, dtype=np.float32)
    
    processed_features = []
    for features in features_list:
        if len(features) == 518:
            left_part = features[:257]
            right_part = features[257:514]
            relative_part = features[514:]
            combined = (left_part + right_part) / 2.0
            combined = np.concatenate([combined, relative_part[:4]])
            if len(combined) < 257:
                padding = np.zeros(257 - len(combined))
                combined = np.concatenate([combined, padding])
            elif len(combined) > 257:
                combined = combined[:257]
            processed_features.append(combined)
        else:
            processed_features.append(features[:257])
    
    features_array = np.vstack(processed_features)
    
    mean_features = np.mean(features_array, axis=0)
    std_features = np.std(features_array, axis=0)
    
    combined = np.concatenate([mean_features, std_features])
    
    if len(combined) != 514:
        if len(combined) > 514:
            combined = combined[:514]
        else:
            padding = np.zeros(514 - len(combined))
            combined = np.concatenate([combined, padding])
    
    return combined.astype(np.float32)

def load_dataset_from_folder(videos_dir, hand_mode="auto", sample_fps=10, max_frames=30):
    print(f" Сканиране на директория: {videos_dir}")
    
    entries = []
    supported_extensions = (".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv")
    
    for filename in os.listdir(videos_dir):
        if not filename.lower().endswith(supported_extensions):
            continue
            
        name_without_ext = os.path.splitext(filename)[0]
        
        if '_' in name_without_ext:
            parts = name_without_ext.split('_')
            
            if parts[-1].isdigit():
                label = '_'.join(parts[:-1])
            elif len(parts) >= 2 and parts[-2].isdigit():
                label = '_'.join(parts[:-2])
            else:
                label = name_without_ext
        else:
            label = name_without_ext
        
        for suffix in ['_left', '_right', '_single', '_dual', '_hand']:
            if label.endswith(suffix):
                label = label[:-len(suffix)]
        
        full_path = os.path.join(videos_dir, filename)
        entries.append((full_path, label.strip()))
    
    if not entries:
        print("  Няма намерени видео файлове!")
        return np.empty((0, 514)), [], []
    
    from collections import Counter
    label_counter = Counter([label for _, label in entries])
    
    print(f"\n Намерени {len(entries)} видео файла с {len(label_counter)} уникални labels:")
    for label, count in sorted(label_counter.items()):
        print(f"  {label}: {count} файла")
    
    print(f"\n  Извличане на характеристики (режим: {hand_mode})...")
    
    X, y, filenames = [], [], []
    success_count = 0
    
    for i, (video_path, label) in enumerate(entries, 1):
        try:
            print(f"  [{i}/{len(entries)}] Обработка на {os.path.basename(video_path):30s}", end="\r")
            
            features = extract_advanced_features(
                video_path, 
                sample_fps=sample_fps, 
                max_frames=max_frames,
                hand_mode=hand_mode
            )
            
            if len(features) != 514:
                print(f"\n   Грешен размер на характеристиките {len(features)} за {video_path}")
                if len(features) > 514:
                    features = features[:514]
                else:
                    padding = np.zeros(514 - len(features))
                    features = np.concatenate([features, padding])
            
            X.append(features)
            y.append(label)
            filenames.append(video_path)
            success_count += 1
            
        except Exception as e:
            print(f"\n   Грешка при обработка на {video_path}: {str(e)[:50]}...")
            continue
    
    print(f"\n Успешно извлечени {success_count}/{len(entries)} видео файла")
    
    if success_count == 0:
        return np.empty((0, 514)), [], []
    
    X_array = np.vstack(X)
    
    print(f"\n Dataset зареден успешно!")  
    print(f"   Проби: {X_array.shape[0]}")
    print(f"   Характеристики: {X_array.shape[1]} (винаги 514)")
    print(f"   Класове: {len(set(y))}")
    
    return X_array, y, filenames

def create_feature_pipeline():
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.decomposition import PCA
    
    pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('pca', PCA(n_components=0.95, random_state=42))
    ])
    
    return pipeline

def visualize_hand_landmarks(image, landmarks_list, connections=True):
    image_copy = image.copy()
    
    for hand_landmarks in landmarks_list:
        if connections:
            h, w = image_copy.shape[:2]
            for connection in HAND_CONNECTIONS:
                start_idx = connection[0]
                end_idx = connection[1]
                
                if start_idx * 3 + 2 < len(hand_landmarks) and end_idx * 3 + 2 < len(hand_landmarks):
                    start_point = (int(hand_landmarks[start_idx * 3] * w), 
                                 int(hand_landmarks[start_idx * 3 + 1] * h))
                    end_point = (int(hand_landmarks[end_idx * 3] * w), 
                               int(hand_landmarks[end_idx * 3 + 1] * h))
                    
                    cv2.line(image_copy, start_point, end_point, (0, 255, 0), 2)
        
        h, w = image_copy.shape[:2]
        for i in range(min(NUM_LANDMARKS, len(hand_landmarks) // 3)):
            x_idx = i * 3
            y_idx = i * 3 + 1
            
            if x_idx < len(hand_landmarks) and y_idx < len(hand_landmarks):
                x = int(hand_landmarks[x_idx] * w)
                y = int(hand_landmarks[y_idx] * h)
                
                if i == 0:
                    color = (255, 0, 0)
                    radius = 6
                elif i in [4, 8, 12, 16, 20]:
                    color = (0, 0, 255)
                    radius = 5
                else:
                    color = (0, 255, 0)
                    radius = 4
                
                cv2.circle(image_copy, (x, y), radius, color, -1)
    
    return image_copy

def extract_multi_hand_features(video_path, **kwargs):
    return extract_advanced_features(video_path, **kwargs)

video_to_feature_vector = extract_advanced_features