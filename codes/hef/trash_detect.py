import cv2
import time
import numpy as np

from common import (
    HailoPythonInferenceEngine, 
    DetectionPostProcessor,
    letterbox_image,
    scale_detections_to_original
)

# ---------------------------------------------------------
# 1. 拦截器：自定义 YOLO 解码引擎
# ---------------------------------------------------------
class CustomYOLOEngine(HailoPythonInferenceEngine):
    def __init__(self, hef_path, num_classes=4):
        super().__init__(hef_path)
        self.num_classes = num_classes
        
    def _run_python_head(self, dequantized_results, conf_threshold):
        tensors_80 = [data for data in dequantized_results.values() if data.shape[1] == 80]
        tensors_40 = [data for data in dequantized_results.values() if data.shape[1] == 40]
        tensors_20 = [data for data in dequantized_results.values() if data.shape[1] == 20]
        
        def split_cls_reg(t1, t2):
            if t1.min() < t2.min(): return t1, t2
            else: return t2, t1
            
        cls_80, reg_80 = split_cls_reg(tensors_80[0], tensors_80[1])
        cls_40, reg_40 = split_cls_reg(tensors_40[0], tensors_40[1])
        cls_20, reg_20 = split_cls_reg(tensors_20[0], tensors_20[1])
        
        STRIDES = [8, 16, 32]
        GRID_SIZES = [80, 40, 20]
        logit_threshold = -np.log(1.0 / conf_threshold - 1.0)
        
        results = []
        coco_classes = DetectionPostProcessor.COCO_CLASSES
        
        for scale_idx, (cls_data, reg_data) in enumerate([(cls_80, reg_80), (cls_40, reg_40), (cls_20, reg_20)]):
            stride = STRIDES[scale_idx]
            grid_dim = GRID_SIZES[scale_idx]
            
            cls_flat = cls_data[0].reshape(-1, self.num_classes)
            reg_flat = reg_data[0].reshape(-1, 4)
            
            max_logits = cls_flat.max(axis=1)
            class_ids = cls_flat.argmax(axis=1)
            
            mask = max_logits > logit_threshold
            if not mask.any(): continue
                
            indices = np.where(mask)[0]
            scores = 1.0 / (1.0 + np.exp(-max_logits[indices]))
            cls = class_ids[indices]
            
            rows = indices // grid_dim
            cols = indices % grid_dim
            
            l, t, r, b = reg_flat[indices, 0], reg_flat[indices, 1], reg_flat[indices, 2], reg_flat[indices, 3]
            
            x1 = (cols + 0.5 - l) * stride
            y1 = (rows + 0.5 - t) * stride
            x2 = (cols + 0.5 + r) * stride
            y2 = (rows + 0.5 + b) * stride
            
            for j in range(len(indices)):
                results.append({
                    'x1': float(x1[j]), 'y1': float(y1[j]),
                    'x2': float(x2[j]), 'y2': float(y2[j]),
                    'conf': float(scores[j]),
                    'cls_id': int(cls[j]),
                    'cls_name': coco_classes.get(int(cls[j]), 'Unknown')
                })
        return results

# ---------------------------------------------------------
# 2. 专属高级画框函数（解决颜色乱跳问题）
# ---------------------------------------------------------
def draw_smart_bboxes(image, detections, thickness=2):
    img = image.copy()
    h, w = img.shape[:2]
    
    # 🌟 专属颜色字典 (OpenCV 使用 BGR 格式)
    # 你可以随意修改这里的 RGB 值来调整你的 UI 配色
    class_colors = {
        0: (0, 0, 255),    # 有害垃圾 (Hazardous): 红色
        1: (0, 255, 0),    # 厨余垃圾 (Kitchen): 绿色
        2: (0, 165, 255),  # 其他垃圾 (Other): 橙色/黄色
        3: (255, 0, 0)     # 可回收物 (Recyclable): 蓝色
    }
    
    for det in detections:
        x1, y1 = int(max(0, det['x1'])), int(max(0, det['y1']))
        x2, y2 = int(min(w, det['x2'])), int(min(h, det['y2']))
        
        cls_id = det['cls_id']
        color = class_colors.get(cls_id, (255, 255, 255)) # 未知类别用白色
        
        # 1. 画主体边框
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        
        # 2. 画文字的底色标签背景 (提升 UI 质感)
        label = f"{det['cls_name']} {det['conf']:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
        
        # 3. 写入白色文字
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
    return img

# ---------------------------------------------------------
# 3. 主程序
# ---------------------------------------------------------
def run_smart_trash_bin():
    print("正在连接 Hailo-8L 视觉大脑...")
    engine = CustomYOLOEngine("my_trash_model.hef", num_classes=4)
    
    DetectionPostProcessor.COCO_CLASSES = {
        0: 'Hazardous',
        1: 'Kitchen',
        2: 'Other',
        3: 'Recyclable'
    }

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("🚀 智能垃圾桶视觉大脑已完美启动！按 'q' 退出。")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
            
        start_time = time.perf_counter()
        orig_h, orig_w = frame.shape[:2]
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        padded, scale, pad_w, pad_h = letterbox_image(rgb_frame, target_size=640)
        input_data = np.expand_dims(padded, axis=0).astype(np.uint8)
        
        # 🌟 优化1：大幅度降低门槛到 0.45，解决漏检的问题！
        results, stats = engine.infer(input_data, conf_threshold=0.45)
        
        if results:
            results = scale_detections_to_original(results, orig_h, orig_w, scale, pad_w, pad_h)
            # 🌟 优化2：调用我们专属的高级颜色分类画框函数！
            frame = draw_smart_bboxes(frame, results, thickness=2)
        
        fps = 1.0 / (time.perf_counter() - start_time)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("Smart Trash Bin Vision", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_smart_trash_bin()