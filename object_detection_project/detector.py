# --- KÜTÜPHANE İÇE AKTARMA ---
import cv2
import numpy as np
from tensorflow import lite as tflite
from time import sleep

# --- AI MODEL KURULUMU ---
MODEL_PATH = "detect.tflite"
LABEL_PATH = "coco_labels.txt"

# TensorFlow Lite modelini yükle
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Model giriş ve çıkış detaylarını al
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Etiket dosyasını yükle
with open(LABEL_PATH, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# --- KAMERA BAŞLATMA ---
camera = cv2.VideoCapture(0)  # 0 = varsayılan kamera
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Kamera başlatıldı. Çıkmak için 'q' tuşuna basın.")

# --- GERÇEK ZAMANLI ALGILAMA DÖNGÜSÜ ---
while True:
    # Kameradan görüntü yakala
    ret, frame = camera.read()
    
    if not ret:
        print("Kamera görüntüsü alınamadı!")
        break
    
    # Görüntüyü modelin beklediği boyuta getir
    resized_frame = cv2.resize(frame, (input_details[0]['shape'][1], input_details[0]['shape'][2]))
    input_data = np.float32(resized_frame)
    input_data = np.expand_dims(input_data, axis=0)
    
    # AI modelini çalıştır
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    # Sonuçları al
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]
    num_detections = interpreter.get_tensor(output_details[3]['index'])[0]
    
    # Tespit edilen nesneleri görüntüye çiz
    for i in range(int(num_detections)):
        if scores[i] > 0.5:  # %50'den yüksek güven skoru
            # Kutu koordinatlarını hesapla
            ymin, xmin, ymax, xmax = boxes[i]
            height, width, _ = frame.shape
            xmin = int(xmin * width)
            xmax = int(xmax * width)
            ymin = int(ymin * height)
            ymax = int(ymax * height)
            
            # Yeşil dikdörtgen çiz
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            
            # Etiket ve güven skorunu ekle
            label_id = int(classes[i])
            if label_id < len(labels):
                label = labels[label_id]
            else:
                label = "Unknown"
            
            confidence = int(scores[i] * 100)
            label_text = f"{label}: {confidence}%"
            
            cv2.putText(frame, label_text, (xmin, ymin - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Sonucu ekranda göster
    cv2.imshow('Nesne Tespiti - Cikmak icin Q basin', frame)
    
    # 'q' tuşuna basılırsa çık
    if cv2.waitKey(1) == ord('q'):
        break

# --- TEMİZLİK ---
camera.release()
cv2.destroyAllWindows()
print("Program sonlandırıldı.")