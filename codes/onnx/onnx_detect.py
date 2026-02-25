import cv2
import time
import numpy as np
import onnxruntime as ort

# ========================================================
# 1. 核心图像预处理 (Letterbox 保持比例填充)
# ========================================================
def letterbox(img, target_size=640, color=(114, 114, 114)):
    h, w = img.shape[:2]
    scale = min(target_size / h, target_size / w)
    new_w, new_h = int(w * scale), int(h * scale)
    
    resized = cv2.resize(img, (new_w, new_h))
    
    pad_w = (target_size - new_w) // 2
    pad_h = (target_size - new_h) // 2
    
    padded = np.full((target_size, target_size, 3), color, dtype=np.uint8)
    padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
    
    return padded, scale, pad_w, pad_h

# ========================================================
# 2. 高级 UI 画框函数 (专属颜色绑定)
# ========================================================
def draw_smart_bboxes(image, detections, conf_threshold=0.45):
    img = image.copy()
    h, w = img.shape[:2]
    
    # 你的专属垃圾分类字典
    COCO_CLASSES = {
        0: 'Hazardous waste',
        1: 'Kitchen waste',
        2: 'Other waste',
        3: 'Recyclable waste'
    }
    
    # 专属 UI 配色 (BGR)
    class_colors = {
        0: (0, 0, 255),    # 有害: 红
        1: (0, 255, 0),    # 厨余: 绿
        2: (0, 165, 255),  # 其他: 橙
        3: (255, 0, 0)     # 可回收: 蓝
    }
    
    for det in detections:
        x1, y1, x2, y2, conf, cls_id = det
        cls_id = int(cls_id)
        
        if conf < conf_threshold:
            continue
            
        x1, y1 = int(max(0, x1)), int(max(0, y1))
        x2, y2 = int(min(w, x2)), int(min(h, y2))
        
        color = class_colors.get(cls_id, (255, 255, 255))
        cls_name = COCO_CLASSES.get(cls_id, f"Unknown {cls_id}")
        
        # 画主体边框
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # 画文字标签背景
        label = f"{cls_name} {conf:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
        
        # 写白字
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
    return img

# ========================================================
# 3. 主程序
# ========================================================
def run_onnx_camera():
    onnx_model_path = "best.onnx"  # 确保你的模型名字和路径正确
    print(f"正在加载 ONNX 模型: {onnx_model_path} ...")
    
    # 初始化 ONNX Runtime 推理引擎
    try:
        session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        print("✓ 模型加载成功！")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("🚀 ONNX 视觉检测已启动！按 'q' 退出。")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
            
        start_time = time.perf_counter()
        orig_h, orig_w = frame.shape[:2]
        
        # --- [A] 图像预处理 ---
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        padded, scale, pad_w, pad_h = letterbox(img_rgb, target_size=640)
        
        # 归一化到 0-1 并转换通道 CHW
        input_tensor = padded.astype(np.float32) / 255.0
        input_tensor = input_tensor.transpose(2, 0, 1)
        input_tensor = np.expand_dims(input_tensor, axis=0)
        
        # --- [B] ONNX 推理 ---
        outputs = session.run([output_name], {input_name: input_tensor})
        
        # 输出形状是 [1, 300, 6]
        raw_detections = outputs[0][0]
        
        # --- [C] 坐标还原与画框 ---
        final_detections = []
        for det in raw_detections:
            x1_lb, y1_lb, x2_lb, y2_lb, conf, cls_id = det
            
            # 去除 padding 并缩放回真实摄像头尺寸
            x1 = (x1_lb - pad_w) / scale
            y1 = (y1_lb - pad_h) / scale
            x2 = (x2_lb - pad_w) / scale
            y2 = (y2_lb - pad_h) / scale
            
            final_detections.append([x1, y1, x2, y2, conf, cls_id])
            
        # 调用我们的智能画框引擎 (置信度设为 0.45)
        frame = draw_smart_bboxes(frame, final_detections, conf_threshold=0.45)
        
        # 计算 FPS
        fps = 1.0 / (time.perf_counter() - start_time)
        cv2.putText(frame, f"ONNX FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow("ONNX Smart Trash Bin", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_onnx_camera()