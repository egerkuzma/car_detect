import cv2
import argparse
from ultralytics import YOLO
import time
from pathlib import Path
import sys
import numpy as np
import torch
import threading
from queue import Queue

print(f"CUDA доступен: {torch.cuda.is_available()}")
print(f"Текущее устройство: {torch.cuda.get_device_name(0)}")

torch.backends.cudnn.benchmark = True  # Оптимизация для фиксированного размера входных данных
torch.backends.cudnn.deterministic = False  # Отключаем детерминизм для лучшей производительности

class VehicleCounter:
    def __init__(self, line_position=0.5):
        self.tracked_vehicles = {}
        self.vehicle_counts = {'car': 0}
        self.line_position = line_position
        self.already_counted = set()
        self.debug = False

    def update(self, detections, frame_width):
        line_pos = int(frame_width * self.line_position)
        
        if not hasattr(detections.boxes, 'id'):
            return self.vehicle_counts
            
        for box in detections.boxes:
            if not hasattr(box, 'id'):
                continue
                
            vehicle_id = int(box.id[0]) if box.id is not None else None
            if vehicle_id is None:
                continue

            cls = int(box.cls[0])
            vehicle_type = 'car'
            
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center_x = (x1 + x2) // 2

            if vehicle_id not in self.tracked_vehicles:
                self.tracked_vehicles[vehicle_id] = {
                    'type': vehicle_type,
                    'last_x': center_x,
                    'counted': False
                }
            
            last_x = self.tracked_vehicles[vehicle_id]['last_x']
            
            if not self.tracked_vehicles[vehicle_id]['counted']:
                if last_x >= line_pos and center_x < line_pos:
                    self.vehicle_counts[vehicle_type] += 1
                    self.tracked_vehicles[vehicle_id]['counted'] = True
                elif last_x <= line_pos and center_x > line_pos:
                    self.vehicle_counts[vehicle_type] += 1
                    self.tracked_vehicles[vehicle_id]['counted'] = True
            
            self.tracked_vehicles[vehicle_id]['last_x'] = center_x

        return self.vehicle_counts

class VideoStream:
    def __init__(self, url):
        self.url = url
        self.cap = None
        self.frame_queue = Queue(maxsize=2)
        self.stopped = False
        self.thread = None

    def start(self):
        self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not self.cap.isOpened():
            raise Exception("Could not connect to stream")

        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                print("Error reading frame, reconnecting...")
                self.reconnect()
                continue

            if not self.frame_queue.full():
                self.frame_queue.put(frame)
            else:
                try:
                    self.frame_queue.get_nowait()
                except:
                    pass

    def reconnect(self):
        if self.cap:
            self.cap.release()
        time.sleep(1)
        self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
        self.cap.set(cv2.CAP_PROP_FPS, 30)

    def read(self):
        try:
            return self.frame_queue.get_nowait()
        except:
            return None

    def stop(self):
        self.stopped = True
        if self.thread:
            self.thread.join()
        if self.cap:
            self.cap.release()

def zoom_area(frame, x, y, width, height, zoom_factor=2.0):
    x = max(0, min(x, frame.shape[1]))
    y = max(0, min(y, frame.shape[0]))
    width = min(width, frame.shape[1] - x)
    height = min(height, frame.shape[0] - y)
    
    roi = frame[y:y+height, x:x+width]
    zoomed = cv2.resize(roi, None, fx=zoom_factor, fy=zoom_factor)
    
    return zoomed

def main():
    parser = argparse.ArgumentParser(description='Vehicle detection and counting in video stream')
    parser.add_argument('--input', type=str, 
                      default='rtsp://rtsp:12345678@192.168.1.128:554/av_stream/ch0',
                      help='RTSP stream URL')
    parser.add_argument('--output', type=str, default='', help='Path to save output (optional)')
    parser.add_argument('--conf', type=float, default=0.1, help='Confidence threshold (0.0-1.0)')
    parser.add_argument('--roi-x', type=int, required=True, help='ROI X coordinate')
    parser.add_argument('--roi-y', type=int, required=True, help='ROI Y coordinate')
    parser.add_argument('--roi-width', type=int, required=True, help='ROI width')
    parser.add_argument('--roi-height', type=int, required=True, help='ROI height')
    parser.add_argument('--zoom-factor', type=float, default=2.0, help='Zoom factor')
    parser.add_argument('--speed', type=float, default=1.0, help='Playback speed multiplier')
    parser.add_argument('--show-full-frame', action='store_true', help='Показывать полный кадр вместо только области интереса')
    args = parser.parse_args()

    print('Loading YOLO model...')
    model = YOLO('yolo11s.pt')

    print(f'Connecting to stream: {args.input}')
    video_stream = VideoStream(args.input)
    video_stream.start()

    if args.show_full_frame:
        window_name = 'Полный кадр (Нажмите Q для выхода)'
    else:
        window_name = 'Область обнаружения (Нажмите Q для выхода)'
        
    zoomed_width = int(args.roi_width * args.zoom_factor)
    zoomed_height = int(args.roi_height * args.zoom_factor)

    output_writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        if args.show_full_frame:
            # Для определения размера выходного файла при отображении полного кадра
            # Получим первый кадр, чтобы определить его размеры
            first_frame = None
            while first_frame is None:
                first_frame = video_stream.read()
                if first_frame is None:
                    time.sleep(0.1)
            
            # Рассчитаем размер после масштабирования
            display_height = 720
            aspect_ratio = first_frame.shape[1] / first_frame.shape[0]
            display_width = int(display_height * aspect_ratio)
            
            output_writer = cv2.VideoWriter(args.output, fourcc, 30, (display_width, display_height))
        else:
            output_writer = cv2.VideoWriter(args.output, fourcc, 30, (zoomed_width, zoomed_height))

    counter = VehicleCounter()
    frame_count = 0
    processed_count = 0
    start_time = time.time()
    last_fps_time = start_time
    fps_counter = 0
    fps = 0

    try:
        while True:
            frame = video_stream.read()
            if frame is None:
                time.sleep(0.001)  # Небольшая задержка для снижения нагрузки на CPU
                continue
                
            frame_count += 1
            processed_count += 1
            fps_counter += 1
            
            current_time = time.time()
            if current_time - last_fps_time >= 1.0:
                fps = fps_counter / (current_time - last_fps_time)
                fps_counter = 0
                last_fps_time = current_time
                print(f"Current FPS: {fps:.1f}")
            
            try:
                x = max(0, min(args.roi_x, frame.shape[1] - args.roi_width))
                y = max(0, min(args.roi_y, frame.shape[0] - args.roi_height))
                roi_frame = frame[y:y+args.roi_height, x:x+args.roi_width]
                
                if roi_frame.size == 0:
                    continue
                    
            except Exception as e:
                continue
            
            results = model.track(
                roi_frame,
                classes=[2],
                conf=args.conf,
                show=False,
                persist=True,
                tracker="botsort.yaml",
                verbose=False,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            
            if results and results[0].boxes:
                counts = counter.update(results[0], roi_frame.shape[1])
            else:
                counts = counter.vehicle_counts

            if args.show_full_frame:
                # Создаем копию полного кадра для отображения
                display_frame = frame.copy()
                
                # Рисуем прямоугольник области интереса
                cv2.rectangle(display_frame, 
                             (x, y), 
                             (x + args.roi_width, y + args.roi_height), 
                             (0, 255, 255), 2)
                
                # Рисуем линию подсчета в области интереса
                line_pos = x + int(args.roi_width * 0.5)
                cv2.line(display_frame, 
                        (line_pos, y), 
                        (line_pos, y + args.roi_height), 
                        (0, 255, 255), 2)

                if results and results[0].boxes:
                    for box in results[0].boxes:
                        b = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0])
                        
                        # Корректируем координаты для полного кадра
                        b[0] += x
                        b[1] += y
                        b[2] += x
                        b[3] += y
                        
                        cv2.rectangle(display_frame, 
                                    (int(b[0]), int(b[1])), 
                                    (int(b[2]), int(b[3])), 
                                    (0, 255, 0), 2)
                        
                        label = f"car: {conf:.2f}"
                        
                        try:
                            if hasattr(box, 'id') and box.id is not None:
                                track_id = box.id[0]
                                if track_id is not None:
                                    label += f" ID:{int(track_id)}"
                        except (AttributeError, IndexError, TypeError):
                            pass
                            
                        cv2.putText(display_frame, label, 
                                  (int(b[0]), int(b[1]-10)), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.5, (0, 255, 0), 2)

                # Изменяем размер для отображения
                display_height = 720  # Фиксированная высота для отображения
                aspect_ratio = display_frame.shape[1] / display_frame.shape[0]
                display_width = int(display_height * aspect_ratio)
                display_frame = cv2.resize(display_frame, (display_width, display_height))
            else:
                # Отображаем только область интереса (ROI)
                display_frame = roi_frame.copy()
                
                # Рисуем линию подсчета
                line_pos = int(display_frame.shape[1] * 0.5)
                cv2.line(display_frame, 
                        (line_pos, 0), 
                        (line_pos, display_frame.shape[0]), 
                        (0, 255, 255), 2)

                if results and results[0].boxes:
                    for box in results[0].boxes:
                        b = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0])
                        
                        cv2.rectangle(display_frame, 
                                    (int(b[0]), int(b[1])), 
                                    (int(b[2]), int(b[3])), 
                                    (0, 255, 0), 2)
                        
                        label = f"car: {conf:.2f}"
                        
                        try:
                            if hasattr(box, 'id') and box.id is not None:
                                track_id = box.id[0]
                                if track_id is not None:
                                    label += f" ID:{int(track_id)}"
                        except (AttributeError, IndexError, TypeError):
                            pass
                            
                        cv2.putText(display_frame, label, 
                                  (int(b[0]), int(b[1]-10)), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.5, (0, 255, 0), 2)
                
                # Применяем масштабирование к области интереса
                display_frame = cv2.resize(display_frame, None, 
                                         fx=args.zoom_factor, 
                                         fy=args.zoom_factor)

            stats_text = f"FPS: {fps:.1f} | Cars: {counts['car']}"
            cv2.putText(display_frame, stats_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

            cv2.imshow(window_name, display_frame)
            
            if output_writer:
                output_writer.write(display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nStream interrupted by user")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        video_stream.stop()
        if output_writer:
            output_writer.release()
        cv2.destroyAllWindows()
        
        print(f'Stream ended. Processed {processed_count} frames in {time.time() - start_time:.1f} sec')
        print(f'Final counts - Cars: {counter.vehicle_counts["car"]}')

if __name__ == "__main__":
    main()