import sys
import os

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

import time
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import joblib
import mediapipe as mp
from collections import deque, defaultdict
from scipy.spatial.distance import euclidean
from utils import (
    normalize_landmarks, 
    calculate_angles,
    extract_extended_single_hand_features,
    extract_dual_hand_features
)

MODEL_PATH_DEFAULT = "signbridge_model.joblib"
MP_HANDS = mp.solutions.hands

class SignBridgeApp(tk.Tk):
    def __init__(self, model_path=MODEL_PATH_DEFAULT):
        super().__init__()
        self.title("SignBridge — Български Жестов Език")
        self.geometry("1200x700")
        self.configure(bg='#1a1a2e')
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.model_path = model_path
        self.hand_mode = tk.StringVar(value="single")
        self.confidence_mode = tk.StringVar(value="normal")
        
        # Video capture
        self.cam_index = tk.IntVar(value=0)
        self.cap = None
        self.hands = MP_HANDS.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.6,
            model_complexity=1
        )
        
        self.prediction_buffer = deque(maxlen=20)
        self.last_predictions = deque(maxlen=7)
        self.hand_history = deque(maxlen=10)
        self.prob_buffer = deque(maxlen=5)
        self.locked_label = None
        self.lock_until = 0

        
        self.prediction_count = 0
        self.start_time = time.time()
        
        self.setup_ui()
        
        self.pipeline = None
        self.label_encoder = None
        self.class_names = []
        self.class_centroids = None
        self.feature_mask = None
        self.expected_input_features = None
        self.load_model_silent()
        
        self.running = True
        self.update_delay = 25
        self.last_pred_time = 0
        self.pred_interval = 0.12
        
        self.after(100, self.open_camera_and_loop)
    
    def setup_ui(self):
        main_container = tk.PanedWindow(self, orient=tk.HORIZONTAL, bg='#1a1a2e', sashwidth=5)
        main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        left_panel = tk.Frame(main_container, bg='#0f3460')
        main_container.add(left_panel, width=700)
        
        cam_title = tk.Label(left_panel, text="Камера", font=("Segoe UI", 18, "bold"), 
                           bg='#0f3460', fg='white', pady=10)
        cam_title.pack(fill=tk.X)
        
        video_container = tk.Frame(left_panel, bg='#0f3460')
        video_container.pack(padx=20, pady=10)
        
        self.video_label = tk.Label(video_container, bg='black', relief=tk.RAISED, bd=3)
        self.video_label.pack()
        
        status_frame = tk.Frame(left_panel, bg='#0f3460')
        status_frame.pack(fill=tk.X, padx=20, pady=(5, 0))
        
        self.video_status = tk.Label(status_frame, text="Камера: Готова", fg="#4cd137", 
                                    bg='#0f3460', font=("Segoe UI", 10))
        self.video_status.pack(side=tk.LEFT)
        
        self.hand_count_label = tk.Label(status_frame, text="Ръце: 0", fg="#00a8ff",
                                        bg='#0f3460', font=("Segoe UI", 10))
        self.hand_count_label.pack(side=tk.LEFT, padx=(20, 0))
        
        controls_frame = tk.Frame(left_panel, bg='#0f3460', pady=10)
        controls_frame.pack(fill=tk.X, padx=20)
        
        btn_frame = tk.Frame(controls_frame, bg='#0f3460')
        btn_frame.pack()
        
        tk.Button(btn_frame, text="Стартирай Камера", width=15,
                 command=self.start_camera,
                 bg='#2ecc71', fg='white', font=("Segoe UI", 10),
                 relief=tk.RAISED, bd=2).pack(side=tk.LEFT, padx=5)
        
        tk.Button(btn_frame, text="Изключи Камера", width=15,
                 command=self.stop_camera,
                 bg='#e74c3c', fg='white', font=("Segoe UI", 10),
                 relief=tk.RAISED, bd=2).pack(side=tk.LEFT, padx=5)
    
        
        settings_frame = tk.Frame(controls_frame, bg='#0f3460', pady=5)
        settings_frame.pack(fill=tk.X)
        
        tk.Label(settings_frame, text="Камера:", bg='#0f3460', fg='white',
                font=("Segoe UI", 10)).pack(side=tk.LEFT)
        
        self.cam_spin = tk.Spinbox(
            settings_frame,
            from_=0,
            to=10,
            width=6,
            textvariable=self.cam_index,
            command=self.on_cam_change,
            font=("Segoe UI", 10),
            bg='#162447',
            fg='white',
            relief=tk.FLAT
        )
        self.cam_spin.pack(side=tk.LEFT, padx=(5, 20))
        
        tk.Label(settings_frame, text="Режим:", bg='#0f3460', fg='white',
                font=("Segoe UI", 10)).pack(side=tk.LEFT)
        
        mode_menu = ttk.Combobox(settings_frame, textvariable=self.hand_mode,
                                values=["single", "dual"], state="readonly",
                                width=8, font=("Segoe UI", 10))
        mode_menu.set("single")
        mode_menu.pack(side=tk.LEFT, padx=5)
        
        right_panel = tk.Frame(main_container, bg='#1a1a2e')
        main_container.add(right_panel, width=500)
        
        translation_frame = tk.Frame(right_panel, bg='#162447', relief=tk.RAISED, bd=2)
        translation_frame.pack(fill=tk.X, padx=20, pady=(20, 10))
        
        tk.Label(translation_frame, text="Превод в реално време", 
                font=("Segoe UI", 14, "bold"), bg='#162447', fg='white').pack(pady=(10, 5))
        
        self.pred_var = tk.StringVar(value="Покажете ръка...")
        self.pred_display = tk.Label(
            translation_frame,
            textvariable=self.pred_var,
            font=("Segoe UI", 32, "bold"),
            bg="#1a1a2e",
            fg="white",
            height=3,
            wraplength=450,
            justify="center",
            relief=tk.SUNKEN,
            bd=2
        )
        self.pred_display.pack(fill=tk.X, padx=20, pady=(5, 20))
        
        conf_frame = tk.Frame(translation_frame, bg='#162447')
        conf_frame.pack(fill=tk.X, padx=20, pady=(0, 10))
        
        tk.Label(conf_frame, text="Сигурност:", font=("Segoe UI", 11), 
                bg='#162447', fg='white').pack(side=tk.LEFT)
        
        self.conf_bar = ttk.Progressbar(
            conf_frame,
            orient="horizontal",
            length=200,
            mode="determinate",
            style="blue.Horizontal.TProgressbar"
        )
        self.conf_bar.pack(side=tk.LEFT, padx=(10, 5))
        
        self.conf_var = tk.StringVar(value="0.00")
        self.conf_label = tk.Label(
            conf_frame,
            textvariable=self.conf_var,
            font=("Segoe UI", 11, "bold"),
            width=5,
            bg='#162447',
            fg='#00a8ff'
        )
        self.conf_label.pack(side=tk.LEFT)
        
        settings_section = tk.LabelFrame(right_panel, text=" Настройки ", 
                                        font=("Segoe UI", 12, "bold"),
                                        bg='#162447', fg='white', padx=15, pady=15)
        settings_section.pack(fill=tk.X, padx=20, pady=10)
        
        thresh_frame = tk.Frame(settings_section, bg='#162447')
        thresh_frame.pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(thresh_frame, text="Праг на доверие:", bg='#162447', 
                fg='white', font=("Segoe UI", 10)).pack(anchor="w")
        
        self.threshold = tk.DoubleVar(value=0.7)
        self.thresh_scale = tk.Scale(
            thresh_frame,
            from_=0.4,
            to=0.95,
            resolution=0.05,
            variable=self.threshold,
            orient="horizontal",
            length=250,
            bg='#162447',
            fg='white',
            highlightbackground='#162447',
            troughcolor='#1a1a2e',
            sliderrelief=tk.RAISED
        )
        self.thresh_scale.pack(fill=tk.X, pady=(5, 0))
        
        security_frame = tk.Frame(settings_section, bg='#162447')
        security_frame.pack(fill=tk.X, pady=(0, 15))
        
        tk.Label(security_frame, text="Режим на сигурност:", bg='#162447', 
                fg='white', font=("Segoe UI", 10)).pack(anchor="w")
        
        security_menu = ttk.Combobox(security_frame, textvariable=self.confidence_mode,
                                    values=["strict", "normal", "loose"], state="readonly",
                                    width=15, font=("Segoe UI", 10))
        security_menu.set("normal")
        security_menu.pack(fill=tk.X, pady=5)
        
        memory_frame = tk.Frame(settings_section, bg='#162447')
        memory_frame.pack(fill=tk.X, pady=(10, 0))
        
        memory_label = tk.Label(memory_frame, text="Управление на памет:", 
                               font=("Segoe UI", 11, "bold"), bg='#162447', fg='white')
        memory_label.pack(anchor="w", pady=(0, 10))
        
        memory_btns = tk.Frame(memory_frame, bg='#162447')
        memory_btns.pack(fill=tk.X)
        
        tk.Button(memory_btns, text="Импортирай", width=12,
                 command=self.import_memory,
                 bg='#9b59b6', fg='white', font=("Segoe UI", 9),
                 relief=tk.RAISED).pack(side=tk.LEFT, padx=2)
        
        tk.Button(memory_btns, text="Експортирай", width=12,
                 command=self.export_memory,
                 bg='#9b59b6', fg='white', font=("Segoe UI", 9),
                 relief=tk.RAISED).pack(side=tk.LEFT, padx=2)
        
        tk.Button(memory_btns, text="Добави Видео", width=12,
                 command=self.add_video_sample,
                 bg='#3498db', fg='white', font=("Segoe UI", 9),
                 relief=tk.RAISED).pack(side=tk.LEFT, padx=2)
        
        tk.Button(memory_btns, text="Обнови Жестове", width=12,
                 command=self.update_gestures,
                 bg='#2ecc71', fg='white', font=("Segoe UI", 9),
                 relief=tk.RAISED).pack(side=tk.LEFT, padx=2)
        
        control_btns = tk.Frame(right_panel, bg='#1a1a2e')
        control_btns.pack(fill=tk.X, padx=20, pady=(10, 20))
        
        tk.Button(control_btns, text="Зареди Модел", width=15,
                 command=self.load_model_local,
                 bg='#3498db', fg='white', font=("Segoe UI", 10),
                 relief=tk.RAISED).pack(side=tk.LEFT, padx=2)
        
        tk.Button(control_btns, text="Презареди", width=15,
                 command=self.reload_model_local,
                 bg='#2ecc71', fg='white', font=("Segoe UI", 10),
                 relief=tk.RAISED).pack(side=tk.LEFT, padx=2)
        
        tk.Button(control_btns, text="Статистика", width=15,
                 command=self.show_stats_local,
                 bg='#f39c12', fg='white', font=("Segoe UI", 10),
                 relief=tk.RAISED).pack(side=tk.LEFT, padx=2)
        
        tk.Button(control_btns, text="Изход", width=15,
                 command=self.on_close,
                 bg='#e74c3c', fg='white', font=("Segoe UI", 10),
                 relief=tk.RAISED).pack(side=tk.LEFT, padx=2)
        
        status_frame = tk.Frame(self, bg='#0f3460', height=30)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.status_var = tk.StringVar(value="Готов - Заредете модел за начало")
        self.status_label = tk.Label(
            status_frame,
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W,
            bg='#0f3460',
            fg='white',
            font=("Segoe UI", 9)
        )
        self.status_label.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)
        
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("blue.Horizontal.TProgressbar", 
                       troughcolor='#1a1a2e',
                       background='#00a8ff',
                       lightcolor='#00a8ff',
                       darkcolor='#00a8ff')
    
    def start_camera(self):
        if self.cap is None:
            self.open_camera_and_loop()
            self.status_var.set("Камера стартирана")
    
    def stop_camera(self):
        if self.cap:
            self.cap.release()
            self.cap = None
            self.video_status.config(text="Камера: Изключена", foreground="#e74c3c")
            self.status_var.set("Камера изключена")
    
    def retrain_system(self):
        response = messagebox.askyesno("Преобучение", "Желаете ли да преобучите системата с текущи данни?")
        if response:
            self.status_var.set("Започва преобучение...")
            self.retrain_model()
    
    def retrain_model(self):
        try:
            import subprocess
            cmd = [sys.executable, "train.py", "--videos", "dataset_videos"]
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            self.status_var.set("Преобучение в процес...")
            
            def check_process():
                if process.poll() is not None:
                    stdout, stderr = process.communicate()
                    if process.returncode == 0:
                        self.status_var.set("Преобучение завършено успешно!")
                        self.load_model_silent()
                        messagebox.showinfo("Успех", "Моделът е преобучен успешно!")
                    else:
                        self.status_var.set("Грешка при преобучение")
                        messagebox.showerror("Грешка", f"Преобучението неуспешно:\n{stderr.decode()}")
                else:
                    self.after(1000, check_process)
            
            self.after(1000, check_process)
            
        except Exception as e:
            self.status_var.set(f"Грешка при стартиране на обучение: {e}")
            messagebox.showerror("Грешка", f"Не може да се стартира обучение: {e}")
    
    def import_memory(self):
        filename = filedialog.askopenfilename(
            title="Импортирай Модел",
            filetypes=[("Joblib файлове", "*.joblib"), ("Всички файлове", "*.*")]
        )
        if filename:
            self.model_path = filename
            self.load_model_silent()
            messagebox.showinfo("Успех", f"Модел зареден от:\n{filename}")
    
    def export_memory(self):
        filename = filedialog.asksaveasfilename(
            title="Експортирай Модел",
            defaultextension=".joblib",
            filetypes=[("Joblib файлове", "*.joblib"), ("Всички файлове", "*.*")]
        )
        if filename and self.pipeline is not None:
            try:
                bundle = {
                    "pipeline": self.pipeline,
                    "label_encoder": self.label_encoder,
                    "class_centroids": self.class_centroids,
                    "feature_mask": self.feature_mask
                }
                joblib.dump(bundle, filename)
                messagebox.showinfo("Успех", f"Модел експортиран в:\n{filename}")
            except Exception as e:
                messagebox.showerror("Грешка", f"Грешка при експорт: {e}")
    
    def add_video_sample(self):
        filename = filedialog.askopenfilename(
            title="Добави Видео Пример",
            filetypes=[("Видео файлове", "*.mp4 *.avi *.mov *.mkv"), ("Всички файлове", "*.*")]
        )
        if filename:
            try:
                dataset_dir = "dataset_videos"
                if not os.path.exists(dataset_dir):
                    os.makedirs(dataset_dir)
                
                from tkinter import simpledialog
                gesture_name = simpledialog.askstring("Име на жест", "Въведете име на жест:")
                if not gesture_name:
                    return
                
                gesture_name = gesture_name.strip()
                
                existing_files = [f for f in os.listdir(dataset_dir) 
                                if f.startswith(gesture_name) and f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
                
                numbers = []
                for f in existing_files:
                    try:
                        import re
                        match = re.search(r'_(\d+)\.', f)
                        if match:
                            numbers.append(int(match.group(1)))
                    except:
                        pass
                
                next_number = max(numbers) + 1 if numbers else 1
                
                new_filename = f"{gesture_name}_{next_number}.mp4"
                new_path = os.path.join(dataset_dir, new_filename)
                
                import shutil
                shutil.copy2(filename, new_path)
                
                self.status_var.set(f"Видео добавено: {new_filename}")
                messagebox.showinfo("Успех", f"Видео добавено като: {new_filename}")
                
            except Exception as e:
                messagebox.showerror("Грешка", f"Грешка при добавяне на видео: {e}")
    
    def update_gestures(self):
        response = messagebox.askyesno("Обновяване", "Желаете ли да обновите списъка с жестове?\nТова ще презареди модела.")
        if response:
            self.status_var.set("Обновяване на жестове...")
            self.load_model_silent()
            self.status_var.set("Жестове обновени")
    
    def load_model_local(self):
        filename = filedialog.askopenfilename(
            title="Избери Модел",
            filetypes=[("Joblib файлове", "*.joblib"), ("Всички файлове", "*.*")]
        )
        if filename:
            self.model_path = filename
            self.load_model_silent()
            messagebox.showinfo("Модел Зареден", f"Модел зареден от:\n{filename}")
    
    def reload_model_local(self):
        try:
            if os.path.exists(self.model_path):
                bundle = joblib.load(self.model_path)
                self.pipeline = bundle.get("pipeline")
                self.label_encoder = bundle.get("label_encoder")
                self.feature_mask = bundle.get("feature_mask")
                self.class_centroids = bundle.get("class_centroids")
                
                self.feature_mask = None
                
                if self.label_encoder:
                    self.class_names = list(self.label_encoder.classes_)
                    self.status_var.set(f"Модел презареден: {len(self.class_names)} жеста (без feature_mask)")
                    messagebox.showinfo("Успех", "Модел презареден успешно!")
                else:
                    messagebox.showwarning("Предупреждение", "Модел зареден но без енкодер")
            else:
                messagebox.showerror("Грешка", "Файл с модел не е намерен")
        except Exception as e:
            messagebox.showerror("Грешка", f"Грешка при презареждане:\n{e}")
    
    def show_stats_local(self):
        uptime = time.time() - self.start_time
        model_info = ""
        if self.pipeline:
            model_info = f"Модел: {os.path.basename(self.model_path)}\n"
            if self.class_names:
                model_info += f"Жестове: {len(self.class_names)}\n"
                model_info += f"Пример: {', '.join(self.class_names[:5])}"
                if len(self.class_names) > 5:
                    model_info += f"... и още {len(self.class_names) - 5}"
        
        stats_text = f"""
Статистика:
-----------
{model_info}
Време на работа: {uptime:.1f} секунди
Направени предсказания: {self.prediction_count}
Честота: {self.prediction_count/max(uptime, 1):.1f}/сек
Режим: {self.hand_mode.get()}
        """
        messagebox.showinfo("Статистика", stats_text)
    
    def open_camera_and_loop(self):
        if self.cap is None:
            try:
                self.cap = cv2.VideoCapture(int(self.cam_index.get()))
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                
                if not self.cap.isOpened():
                    raise RuntimeError(f"Не може да се отвори камера {self.cam_index.get()}")
                    
                self.video_status.config(text="Камера: Активна", foreground="#2ecc71")
                self.status_var.set("Камера готова - Покажете ръка")
                
            except Exception as e:
                messagebox.showerror("Грешка с Камера", f"Грешка при отваряне: {e}")
                self.status_var.set("Грешка с камера")
                self.video_status.config(text="Камера: Грешка", foreground="#e74c3c")
                return
        
        self._loop_frame()
    
    def _loop_frame(self):
        if not self.running:
            return
            
        ret, frame = self.cap.read() if self.cap is not None else (False, None)
        if not ret:
            self.status_var.set("Няма сигнал от камера")
            self.video_status.config(text="Няма сигнал", foreground="#f39c12")
            self.after(1000, self._loop_frame)
            return
        
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        
        drawn = frame.copy()
        hands_detected = 0
        all_landmarks = []
        
        if results.multi_hand_landmarks:
            hands_detected = len(results.multi_hand_landmarks)
            self.hand_count_label.config(text=f"Ръце: {hands_detected}")
            
            for hand_landmarks in results.multi_hand_landmarks:
                h, w, _ = drawn.shape
                
                for landmark in hand_landmarks.landmark:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    cv2.circle(drawn, (x, y), 5, (0, 255, 0), -1)
                
                for connection in mp.solutions.hands.HAND_CONNECTIONS:
                    start_idx = connection[0]
                    end_idx = connection[1]
                    start_point = (int(hand_landmarks.landmark[start_idx].x * w),
                                 int(hand_landmarks.landmark[start_idx].y * h))
                    end_point = (int(hand_landmarks.landmark[end_idx].x * w),
                               int(hand_landmarks.landmark[end_idx].y * h))
                    cv2.line(drawn, start_point, end_point, (0, 255, 0), 2)
                
                coords = []
                for landmark in hand_landmarks.landmark:
                    coords.extend([landmark.x, landmark.y, landmark.z])
                
                landmarks_vec = np.array(coords, dtype=np.float32)
                all_landmarks.append(landmarks_vec)
            
            self.video_status.config(text=f"Ръце открити: {hands_detected}", foreground="#2ecc71")
        else:
            self.hand_count_label.config(text="Ръце: 0")
            self.video_status.config(text="Няма открити ръце", foreground="#e74c3c")
        
        if all_landmarks:
            if len(all_landmarks) == 2 and self.hand_mode.get() == "dual":
                combined_features = self._combine_two_hand_features(all_landmarks[0], all_landmarks[1])
            else:
                combined_features = self._extract_single_hand_features(all_landmarks[0])
            
            self.prediction_buffer.append(combined_features)
            self.hand_history.append(hands_detected)
        else:
            self.prediction_buffer.clear()
            self.hand_history.append(0)
        
        now = time.time()
        if now - self.last_pred_time >= self.pred_interval:
            self.last_pred_time = now
            if all_landmarks and len(self.prediction_buffer) >= 8:
                self._make_advanced_prediction()
            elif all_landmarks:
                self.pred_var.set("Събирам данни...")
                self.conf_bar['value'] = 0
                self.conf_var.set("0.00")
                self.pred_display.config(bg="#1a1a2e", fg="white")
        
        self._update_video_image(drawn)
        self.after(self.update_delay, self._loop_frame)
    
    def _combine_two_hand_features(self, left_hand, right_hand):
        dual_features = extract_dual_hand_features(left_hand, right_hand)
        
        if len(dual_features) == 518:
            left_part = dual_features[:257]
            right_part = dual_features[257:514]
            relative_part = dual_features[514:]
            combined = (left_part + right_part) / 2.0
            combined = np.concatenate([combined, relative_part[:4]])
            if len(combined) < 257:
                padding = np.zeros(257 - len(combined))
                combined = np.concatenate([combined, padding])
            elif len(combined) > 257:
                combined = combined[:257]
            return combined
        else:
            return dual_features[:257]
    
    def _extract_single_hand_features(self, landmarks):
        features = extract_extended_single_hand_features(landmarks)
        return features
    
    def _make_advanced_prediction(self):
        if self.pipeline is None or self.label_encoder is None:
            self.status_var.set("Няма зареден модел")
            return
        
        recent_frames = list(self.prediction_buffer)
        if len(recent_frames) < 5:
            return
        
        
        features_array = np.vstack(recent_frames)
        
        mean_features = np.mean(features_array, axis=0)
        std_features = np.std(features_array, axis=0)
        
        combined_features = np.concatenate([mean_features, std_features])
        
        X = combined_features.reshape(1, -1)

        if self.expected_input_features is not None and X.shape[1] != self.expected_input_features:
            try:
                print(f"Предупреждение: Моделът очаква {self.expected_input_features} характеристики, но подаваме {X.shape[1]}. Адаптиране...")
                if X.shape[1] > self.expected_input_features:
                    X = X[:, :self.expected_input_features]
                else:
                    pad = np.zeros((X.shape[0], self.expected_input_features - X.shape[1]), dtype=X.dtype)
                    X = np.hstack([X, pad])
            except Exception as e:
                print(f"Грешка при адаптиране на характеристиките: {e}")
        
        try:
            if hasattr(self.pipeline, 'predict_proba'):
                proba = self.pipeline.predict_proba(X)[0]
                idx = np.argmax(proba)
                conf = float(proba[idx])
                label = self.label_encoder.inverse_transform([idx])[0]
                self.prob_buffer.append(proba)

                sum_probs = np.sum(self.prob_buffer, axis=0)
                idx = int(np.argmax(sum_probs))
                conf = float(sum_probs[idx] / len(self.prob_buffer))
                label = self.label_encoder.inverse_transform([idx])[0]

                mode = self.confidence_mode.get()
                if mode == "strict":
                    base_threshold = 0.48
                elif mode == "loose":
                    base_threshold = 0.32
                else:
                    base_threshold = 0.42
                
                ui_threshold = float(self.threshold.get())
                effective_threshold = max(0.35, ui_threshold)
                
                self.prediction_count += 1
                
                if conf >= effective_threshold:
                    self.last_predictions.append((label, conf))
                    
                    if len(self.last_predictions) >= 3:
                        weighted_votes = defaultdict(float)
                        for pred_label, pred_conf in list(self.last_predictions):
                            weighted_votes[pred_label] += pred_conf
                        
                        final_label = max(weighted_votes.items(), key=lambda x: x[1])[0]
                        final_conf = weighted_votes[final_label] / len(self.last_predictions)
                    else:
                        final_label = label
                        final_conf = conf
                    
                    current_time = time.time()

                    if self.locked_label is not None and current_time < self.lock_until:
                        final_label = self.locked_label
                    else:
                        if len(self.last_predictions) >= 3:
                            last3 = [p[0] for p in list(self.last_predictions)[-3:]]
                            if len(set(last3)) == 1:
                                self.locked_label = last3[0]
                                self.lock_until = current_time + 0.4
                                final_label = self.locked_label

                    display_text = f"{final_label}"

                    
                    if final_conf > 0.85:
                        color = '#27ae60'
                    elif final_conf > 0.7:
                        color = '#2ecc71'
                    elif final_conf > 0.6:
                        color = '#f39c12'
                    else:
                        color = '#e67e22'
                    
                    self.pred_display.config(bg=color, fg="white")
                    self.status_var.set(f" Предсказание: {final_label} ({final_conf:.2f})")
                else:
                    display_text = "(ниско доверие...)"
                    self.pred_display.config(bg="#1a1a2e", fg="white")
                    self.status_var.set(f" Недостатъчно данни ({conf:.2f}/{effective_threshold:.2f})")
                
                self.pred_var.set(display_text)
                self.conf_bar['value'] = int(min(100, conf * 100))
                self.conf_var.set(f"{conf:.2f}")
                
            else:
                pred = self.pipeline.predict(X)[0]
                label = self.label_encoder.inverse_transform([int(pred)])[0]
                self.pred_var.set(label)
                self.conf_bar['value'] = 70
                self.conf_var.set("0.70")
                self.status_var.set(f" Предсказание: {label}")
        
        except Exception as e:
            print(f"Грешка при предсказване: {e}")
            self.pred_var.set("Грешка")
            self.conf_bar['value'] = 0
            self.conf_var.set("0.00")
            self.status_var.set(f"Грешка при предсказване: {str(e)[:50]}")
    
    def _update_video_image(self, frame_bgr):
        try:
            h, w = frame_bgr.shape[:2]
            target_w, target_h = 640, 480
            
            aspect = w / h
            if w > h:
                new_w = target_w
                new_h = int(target_w / aspect)
            else:
                new_h = target_h
                new_w = int(target_h * aspect)
            
            resized = cv2.resize(frame_bgr, (new_w, new_h))
            
            if new_w < target_w or new_h < target_h:
                border_top = (target_h - new_h) // 2
                border_bottom = target_h - new_h - border_top
                border_left = (target_w - new_w) // 2
                border_right = target_w - new_w - border_left
                
                resized = cv2.copyMakeBorder(
                    resized,
                    border_top,
                    border_bottom,
                    border_left,
                    border_right,
                    cv2.BORDER_CONSTANT,
                    value=[40, 40, 40]
                )
            
            img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            imgtk = ImageTk.PhotoImage(image=img_pil)
            
            self.video_label.imgtk = imgtk
            self.video_label.config(image=imgtk)
            
        except Exception as e:
            print(f"Грешка при обновяване на видео: {e}")
    
    def load_model_silent(self):
        if not os.path.exists(self.model_path):
            self.pipeline = None
            self.label_encoder = None
            self.status_var.set("Модел не е намерен. Моля тренирайте модел първо.")
            return
            
        try:
            bundle = joblib.load(self.model_path)
            self.pipeline = bundle.get("pipeline")
            self.label_encoder = bundle.get("label_encoder")
            self.class_centroids = bundle.get("class_centroids")
            self.feature_mask = bundle.get("feature_mask")

            self.feature_mask = None
            
            print("Модел зареден (feature_mask игнориран за съвместимост)")

            self.expected_input_features = None
            try:
                if self.pipeline is not None and hasattr(self.pipeline, 'named_steps'):
                    for step in self.pipeline.named_steps.values():
                        if hasattr(step, 'n_features_in_'):
                            self.expected_input_features = int(getattr(step, 'n_features_in_'))
                            break
                elif self.pipeline is not None and hasattr(self.pipeline, 'steps') and len(self.pipeline.steps) > 0:
                    first = self.pipeline.steps[0][1]
                    if hasattr(first, 'n_features_in_'):
                        self.expected_input_features = int(getattr(first, 'n_features_in_'))
            except Exception:
                self.expected_input_features = None

            if self.expected_input_features:
                print(f"Pipeline expected input features: {self.expected_input_features}")

            if self.label_encoder:
                self.class_names = list(self.label_encoder.classes_)
                self.status_var.set(f"Модел зареден: {len(self.class_names)} жеста налични")
            else:
                self.status_var.set("Модел зареден (без енкодер)")
                
        except Exception as e:
            self.pipeline = None
            self.label_encoder = None
            error_msg = str(e)[:100]
            self.status_var.set(f"Грешка при зареждане на модел: {error_msg}...")
            print(f"Грешка при зареждане на модел: {e}")
    
    def on_cam_change(self):
        if self.cap:
            self.cap.release()
            self.cap = None
        self.after(100, self.open_camera_and_loop)
    
    def on_close(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.hands.close()
        self.destroy()

if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else MODEL_PATH_DEFAULT
    app = SignBridgeApp(model_path=model_path)
    app.mainloop()