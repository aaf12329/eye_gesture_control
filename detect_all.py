import argparse
import time
import cv2
import numpy as np
import torch
from torch.nn import DataParallel
import onnxruntime
import threading
from collections import deque
import serial
import sys
import dlib

sys.path.append('/retinaface_arcface')

# 导入人脸识别所需模块
from models import *
from retinaface import Retinaface
from utils.facealign import (Config, cosin_metric, getTransMatrix, img_floder,
                             load_image, process)


class FaceGestureDetector:
    def __init__(self,
                 gesture_model_path,
                 gesture_input_shape=(640, 640),
                 gesture_conf_thres=0.4,
                 gesture_iou_thres=0.5,
                 face_pic_dir='test/',
                 face_distance=0.5,
                 face_landmark_model="shape_predictor_68_face_landmarks.dat",
                 serial_port=None,
                 baudrate=9600,
                 command_interval=0.5):

        # 手势识别相关初始化
        self.gesture_model_path = gesture_model_path
        self.gesture_input_shape = gesture_input_shape
        self.gesture_conf_thres = gesture_conf_thres
        self.gesture_iou_thres = gesture_iou_thres
        self.gesture_session = None
        self.gesture_input_name = None
        self.gesture_output_names = None
        self.gesture_class_names = ['0', '1', '2', '3', '4', '5']  # 手势类别名称
        self.last_detected_class = None  # 记录上次检测到的类别

        # 眼动追踪相关初始化
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(face_landmark_model)
        self.eye_threshold = 0.3
        self.gaze_directions = ["注视", "左转", "右转", "前进", "后退", "停止"]
        self.command_mapping = {
            0: "1",  # 注视中央 -> 启动前进
            1: "3",  # 向左看 -> 左转
            2: "4",  # 向右看 -> 右转
            3: "1",  # 向上看 -> 前进
            4: "2",  # 向下看 -> 后退
            5: "0"  # 闭眼 -> 停止
        }
        self.eye_closed = False
        self.close_eye_frames = 0
        self.open_eye_frames = 0
        self.min_closed_frames = 60
        self.last_eye_closed_time = 0
        self.current_direction = 0
        self.last_direction = 0
        self.direction_stability_count = 0
        self.stability_threshold = 30
        self.eye_weight = 0.7
        self.face_weight = 0.3
        self.eye_center_history = deque(maxlen=10)
        self.eye_ratio_history = deque(maxlen=10)
        self.calibration_complete = False
        self.calibration_eye_centers = []
        self.calibration_top = None
        self.calibration_bottom = None
        self.calibration_left = None
        self.calibration_right = None
        self.calibration_center = None

        # 串口通信相关初始化
        self.last_command_time = 0
        self.command_interval = command_interval
        self.serial_port = serial_port
        self.baudrate = baudrate
        self.serial_connection = None
        self.command_queue = deque()

        # FPS计算相关
        self.fps = 0
        self.fps_queue = deque(maxlen=10)

        # 人脸识别相关初始化
        self.face_pic_dir = face_pic_dir
        self.face_distance = face_distance
        self.retinaface = Retinaface()

        # 人脸验证状态
        self.face_verified = False
        self.verified_users = set()
        self.last_face_verify_time = 0
        self.face_verify_interval = 30.0
        self.face_verify_frames = 0
        self.frame_count = 0

        # 控制模式
        self.control_mode = "gesture"  # 默认为手势控制模式
        self.last_mode_switch_time = 0
        self.mode_switch_interval = 1.0  # 模式切换最小间隔（秒）

        # 初始化各个模型
        self.init_face_model()
        self.load_gesture_model()
        self.init_serial()

    def init_face_model(self):
        """初始化人脸识别模型"""
        opt = Config()
        if opt.backbone == 'resnet18':
            self.face_model = resnet_face18(opt.use_se)
        elif opt.backbone == 'resnet34':
            self.face_model = resnet34()
        elif opt.backbone == 'resnet50':
            self.face_model = resnet50()

        self.face_model = DataParallel(self.face_model)
        self.face_model.load_state_dict(torch.load(opt.test_model_path, map_location='cpu'))
        self.face_model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.face_model.eval()

        # 人脸对齐点
        self.dst_pts = np.array([[66, 75], [128, 75],
                                 [93, 90], [69, 130],
                                 [122, 130]], dtype=np.float32) / 192 * 150

        # 加载人脸库
        self.known_face_names, self.known_face_encodings = img_floder(
            self.face_pic_dir, self.retinaface, self.dst_pts, self.face_model)
        print('已加载人脸库，发现以下人物:', ' ，'.join(self.known_face_names))

    def init_serial(self):
        """初始化串口连接"""
        if self.serial_port:
            try:
                self.serial_connection = serial.Serial(self.serial_port, self.baudrate, timeout=1)
                print(f"成功连接到串口 {self.serial_port}")
                # 启动指令发送线程
                threading.Thread(target=self.send_commands_thread, daemon=True).start()
            except Exception as e:
                print(f"无法连接到串口 {self.serial_port}: {e}")
                self.serial_connection = None

    def send_command(self, command):
        """将指令添加到队列"""
        if self.serial_connection:
            self.command_queue.append(command)
            print(f"[串口输出] 入队待发: {command.strip()}")
        else:
            print(f"[串口输出] 串口未连接，指令被丢弃: {command.strip()}")

    def send_commands_thread(self):
        """线程函数，持续从队列发送指令"""
        while True:
            if self.command_queue:
                command = self.command_queue.popleft()
                try:
                    data = command.encode('utf-8')
                    self.serial_connection.write(data)
                    print(f"[串口输出] 已发送指令 {command.strip()} -> {data!r}")
                except Exception as e:
                    print(f"发送指令失败: {e}")
            time.sleep(0.1)  # 避免CPU占用过高

    def load_gesture_model(self):
        """加载手势识别ONNX模型"""
        # 尝试使用CUDA加速，如果失败则回退到CPU
        try:
            options = onnxruntime.SessionOptions()
            options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
            options.enable_profiling = False
            options.enable_mem_pattern = True
            options.intra_op_num_threads = 4  # 使用4个线程进行计算

            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            provider_options = [
                {
                    'device_id': 0,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                    'do_copy_in_default_stream': True,
                },
                {}
            ]

            self.gesture_session = onnxruntime.InferenceSession(
                self.gesture_model_path,
                sess_options=options,
                providers=providers,
                provider_options=provider_options
            )
            print("成功启用CUDA加速")
        except:
            providers = ['CPUExecutionProvider']
            self.gesture_session = onnxruntime.InferenceSession(
                self.gesture_model_path,
                providers=providers
            )
            print("CUDA不可用，使用CPU进行推理")

        self.gesture_input_name = self.gesture_session.get_inputs()[0].name
        self.gesture_output_names = [output.name for output in self.gesture_session.get_outputs()]

        # 打印模型输入输出详细信息
        print(f"手势模型输入名称: {self.gesture_input_name}, 形状: {self.gesture_session.get_inputs()[0].shape}")
        for i, output in enumerate(self.gesture_session.get_outputs()):
            print(f"手势模型输出{i + 1}名称: {output.name}, 形状: {output.shape}")

    def preprocess_frame_for_gesture(self, frame):
        """预处理帧用于手势识别"""
        # 直接resize到模型输入尺寸
        input_data = cv2.resize(frame, self.gesture_input_shape)
        # BGR转RGB并归一化
        input_data = cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        # 调整维度顺序 (HWC -> CHW) 并添加batch维度
        return np.transpose(input_data, (2, 0, 1))[np.newaxis, ...]

    def xywh2xyxy(self, x):
        """将边界框从xywh格式转换为xyxy格式"""
        y = np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
        y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
        y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
        y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
        return y

    def box_iou(self, box1, box2):
        """计算两个边界框之间的IoU"""
        # 获取边界框坐标
        (x1, y1, x2, y2) = box1
        (x1b, y1b, x2b, y2b) = box2.T

        # 计算交集区域
        inter_area = np.maximum(0, np.minimum(x2, x2b) - np.maximum(x1, x1b)) * \
                     np.maximum(0, np.minimum(y2, y2b) - np.maximum(y1, y1b))

        # 计算并集区域
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x2b - x1b) * (y2b - y1b)
        union_area = box1_area + box2_area - inter_area

        # 计算IoU
        return inter_area / union_area

    def non_max_suppression(self, prediction, conf_thres=0.25, iou_thres=0.45):
        """非极大值抑制，过滤重叠的边界框"""
        # 检查输入是否为空
        if len(prediction) == 0:
            return []

        # 提取置信度和类别分数
        scores = prediction[..., 4] * np.max(prediction[..., 5:], axis=-1)

        # 过滤低置信度的预测
        prediction = prediction[scores > conf_thres]
        if len(prediction) == 0:
            return []

        # 提取边界框和类别
        boxes = self.xywh2xyxy(prediction[..., :4])
        class_scores = prediction[..., 5:]
        classes = np.argmax(class_scores, axis=-1)
        confidences = prediction[..., 4] * class_scores[np.arange(len(classes)), classes]

        # 按置信度排序
        detections = np.column_stack((boxes, confidences, classes)).astype(np.float32)
        detections = detections[np.argsort(-detections[:, 4])]

        # 非极大值抑制
        keep = []
        while detections.shape[0] > 0:
            # 保留置信度最高的框
            keep.append(detections[0])

            # 计算与其余框的IoU
            iou = self.box_iou(keep[-1][:4], detections[1:, :4])

            # 移除IoU大于阈值的框
            detections = detections[1:][iou <= iou_thres]

        return np.array(keep) if keep else np.zeros((0, 6))

    def process_face_recognition(self, frame):
        """处理人脸识别"""
        flag, feature, b1s = process(self.retinaface, frame, self.dst_pts, self.face_model)

        if not flag:
            return frame, False, []

        result = np.zeros((self.known_face_encodings.shape[0], feature.shape[0]))
        match = np.zeros((self.known_face_encodings.shape[0], feature.shape[0]))

        # 计算每个人脸与库中人脸的相似度
        for i in range(self.known_face_encodings.shape[0]):
            for j in range(feature.shape[0]):
                cv2.rectangle(frame, (b1s[j][0], b1s[j][1]), (b1s[j][2], b1s[j][3]), (255, 0, 0), 2)
                result[i][j] = cosin_metric(self.known_face_encodings[i], feature[j].T)
                if result[i][j] > self.face_distance:
                    match[i][j] = 1

        # 按相似度从高到低排序
        after_sort = np.argsort(-result, axis=0)

        recognized_faces = []

        # 在图像上显示识别结果
        for i in range(after_sort.shape[1]):
            if int(match[after_sort[0][i]][i]) == 1:
                face_name = self.known_face_names[after_sort[0][i]]
                face_score = result[after_sort[0][i]][i]
                recognized_faces.append(face_name)

                # 将已识别的人脸添加到已验证集合
                self.verified_users.add(face_name)

                cv2.putText(frame,
                            f"{face_name} {face_score:.2f}",
                            (b1s[i][0], b1s[i][1]),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            1,
                            (20, 216, 48),
                            thickness=2)

        # 如果识别到人脸，设置人脸验证状态
        if recognized_faces:
            self.face_verified = True
            self.last_face_verify_time = time.time()

        return frame, True, recognized_faces

    def process_gesture_detection(self, frame):
        """处理手势识别"""
        try:
            # 获取原始帧尺寸
            orig_h, orig_w = frame.shape[:2]

            # 预处理输入
            input_data = self.preprocess_frame_for_gesture(frame)

            # 执行推理
            outputs = self.gesture_session.run(self.gesture_output_names, {self.gesture_input_name: input_data})

            # 解析输出
            if len(outputs) == 1:
                detections = outputs[0][0]  # 形状通常是[1, num_det, 85]
            else:
                detections = outputs[0][0]

            # 应用非极大值抑制
            detections = self.non_max_suppression(detections, self.gesture_conf_thres, self.gesture_iou_thres)

            # 绘制检测结果
            current_classes = set()  # 当前帧检测到的所有类别
            for det in detections:
                x1, y1, x2, y2, conf, cls = det

                # 缩放坐标到原始图像尺寸
                x1 = int(x1 * orig_w / self.gesture_input_shape[1])
                y1 = int(y1 * orig_h / self.gesture_input_shape[0])
                x2 = int(x2 * orig_w / self.gesture_input_shape[1])
                y2 = int(y2 * orig_h / self.gesture_input_shape[0])

                # 确保坐标在图像范围内
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(orig_w, x2)
                y2 = min(orig_h, y2)

                # 获取类别名称
                class_id = int(cls)
                current_classes.add(class_id)

                # 使用英文标签
                gesture_names = {
                    0: "Stop",
                    1: "Forward",
                    2: "Backward",
                    3: "Left",
                    4: "Right",
                    5: "Invalid"
                }
                class_name = gesture_names.get(class_id, f"Unknown{class_id}")

                # 绘制检测框
                color = (0, 255, 0)  # 绿色边框
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # 创建标签背景
                label = f"{class_name} {conf:.2f}"
                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)

                # 绘制标签文本
                cv2.putText(frame, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # 发送指令逻辑
            # if current_classes and self.face_verified:  # 已注释：不做人脸验证
            if current_classes:
                # 选择置信度最高的类别（detections已按置信度排序）
                highest_conf_class = int(detections[0][5]) if len(detections) > 0 else None

                # 检查是否需要发送指令（类别变化且在0-4范围内）
                if (highest_conf_class is not None and
                        highest_conf_class != self.last_detected_class and
                        highest_conf_class <= 4 and
                        time.time() - self.last_command_time > self.command_interval):

                    self.last_detected_class = highest_conf_class
                    self.last_command_time = time.time()

                    # 构建指令并发送
                    command = f"{highest_conf_class}\n"
                    self.send_command(command)
                    print(f"Detected gesture class {highest_conf_class}, sending command: {command.strip()}")
                elif 5 in current_classes:
                    print("Detected gesture class 5, no command sent")

            return frame, current_classes

        except Exception as e:
            print(f"Gesture detection error: {e}")
            return frame, set()

    def update_face_verify_status(self):
        """更新人脸验证状态"""
        # 如果人脸已验证但超过指定时间，重置验证状态
        if self.face_verified and time.time() - self.last_face_verify_time > self.face_verify_interval:
            print(f"人脸验证已过期，需要重新验证")
            self.face_verified = False

    def process_eye_tracking(self, frame):
        """处理眼动追踪"""
        try:
            # 图像预处理
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 人脸检测
            faces = self.detector(gray)

            if len(faces) > 0:
                # 只处理第一个检测到的人脸
                face = faces[0]

                # 获取人脸关键点
                shape = self.predictor(gray, face)

                # 获取眼睛标记点
                eyes_info = self.get_eyes_landmarks(shape, frame)

                # 在两眼中间绘制中心点
                self.draw_eyes_center(frame, eyes_info)

                # 检查眼睛是否闭合
                is_closed = self.check_eyes_closed(eyes_info)

                # 提取眼球特征（仅在眼睛睁开时）
                eye_features = self.extract_eye_features(frame, eyes_info)

                # 如果检测到左眼或右眼（且眼睛睁开）
                if (not is_closed) and ("left_eye" in eye_features or "right_eye" in eye_features):
                    # 选择一个眼睛用于校准和追踪
                    eye_name = "left_eye" if "left_eye" in eye_features else "right_eye"

                    if eye_name in eye_features:
                        # 校准眼球位置
                        self.calibrate_eyes(eye_features[eye_name]["rel_pos"], eye_name)

                # 确定眼球方向
                eye_direction = self.determine_gaze_direction(eye_features, is_closed)

                # 处理方向稳定性
                if eye_direction == self.current_direction:
                    self.direction_stability_count += 1
                else:
                    self.current_direction = eye_direction
                    self.direction_stability_count = 1

                # 如果方向足够稳定，且与上次发送的不同，则发送指令
                if (self.direction_stability_count >= self.stability_threshold and
                        (self.current_direction != self.last_direction or is_closed) and
                        (self.calibration_complete or is_closed) and
                        time.time() - self.last_command_time > self.command_interval):
                    self.last_direction = self.current_direction
                    self.last_command_time = time.time()

                    # 根据方向映射为实际命令
                    command_code = self.command_mapping.get(self.current_direction, "0")
                    command = f"{command_code}\n"
                    self.send_command(command)

                    action_name = self.gaze_directions[self.current_direction]
                    print(f"检测到眼动: {action_name}，发送指令: {command.strip()}")

                return frame, True

            return frame, False

        except Exception as e:
            print(f"眼动追踪错误: {e}")
            return frame, False

    def get_eyes_landmarks(self, shape, frame):
        """获取眼睛的关键点"""
        # 定义左右眼的地标索引
        left_eye_indices = list(range(36, 42))
        right_eye_indices = list(range(42, 48))

        # 提取左眼和右眼的地标坐标
        left_eye_points = np.array([(shape.part(i).x, shape.part(i).y) for i in left_eye_indices])
        right_eye_points = np.array([(shape.part(i).x, shape.part(i).y) for i in right_eye_indices])

        # 计算眼睛纵横比
        left_ear = self.eye_aspect_ratio(left_eye_points)
        right_ear = self.eye_aspect_ratio(right_eye_points)

        # 平均眼睛纵横比
        avg_ear = (left_ear + right_ear) / 2.0

        # 计算左右眼的中心点
        left_eye_center = np.mean(left_eye_points, axis=0).astype(int)
        right_eye_center = np.mean(right_eye_points, axis=0).astype(int)

        # 计算左右眼的轮廓区域
        left_eye_min_x = np.min(left_eye_points[:, 0])
        left_eye_max_x = np.max(left_eye_points[:, 0])
        left_eye_min_y = np.min(left_eye_points[:, 1])
        left_eye_max_y = np.max(left_eye_points[:, 1])

        right_eye_min_x = np.min(right_eye_points[:, 0])
        right_eye_max_x = np.max(right_eye_points[:, 0])
        right_eye_min_y = np.min(right_eye_points[:, 1])
        right_eye_max_y = np.max(right_eye_points[:, 1])

        # 返回眼睛关键信息
        return {
            "left_eye": {
                "points": left_eye_points,
                "center": left_eye_center,
                "ear": left_ear,
                "roi": (left_eye_min_x, left_eye_min_y, left_eye_max_x, left_eye_max_y)
            },
            "right_eye": {
                "points": right_eye_points,
                "center": right_eye_center,
                "ear": right_ear,
                "roi": (right_eye_min_x, right_eye_min_y, right_eye_max_x, right_eye_max_y)
            },
            "avg_ear": avg_ear
        }

    def eye_aspect_ratio(self, eye_points):
        """计算眼睛纵横比，用于判断眨眼"""
        # 计算垂直眼睛地标的欧几里德距离
        vertical_1 = np.linalg.norm(eye_points[1] - eye_points[5])
        vertical_2 = np.linalg.norm(eye_points[2] - eye_points[4])

        # 计算水平眼睛地标的欧几里德距离
        horizontal = np.linalg.norm(eye_points[0] - eye_points[3])

        # 眼睛纵横比
        ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
        return ear

    def draw_eyes_center(self, frame, eyes_info):
        """在两眼中间绘制中心点"""
        if "left_eye" in eyes_info and "right_eye" in eyes_info:
            # 获取左右眼的中心点
            left_center = eyes_info["left_eye"]["center"]
            right_center = eyes_info["right_eye"]["center"]

            # 计算两眼之间的中心点
            center_x = (left_center[0] + right_center[0]) // 2
            center_y = (left_center[1] + right_center[1]) // 2

            # 绘制中心点
            cv2.circle(frame, (center_x, center_y), 2, (0, 255, 0), -1)

    def check_eyes_closed(self, eyes_info):
        """检查是否闭眼"""
        if "avg_ear" not in eyes_info:
            return False

        avg_ear = eyes_info["avg_ear"]
        current_time = time.time()

        # 如果眼睛纵横比小于阈值，认为闭眼
        is_closed = avg_ear < self.eye_threshold

        # 更新闭眼/睁眼帧计数
        if is_closed:
            self.close_eye_frames += 1
            self.open_eye_frames = 0
            if self.last_eye_closed_time == 0:
                self.last_eye_closed_time = current_time
        else:
            self.open_eye_frames += 1
            self.close_eye_frames = 0
            self.last_eye_closed_time = 0

        # 只有连续多帧检测到闭眼且持续时间超过2秒才认为真的闭眼
        self.eye_closed = (self.close_eye_frames >= self.min_closed_frames and
                           current_time - self.last_eye_closed_time >= 2.0)

        return self.eye_closed

    def extract_eye_features(self, frame, eyes_info):
        """提取眼睛的特征用于追踪眼球位置"""
        features = {}

        # 如果眼睛闭合，直接返回空特征
        if self.eye_closed:
            return features

        for eye_name, eye_info in eyes_info.items():
            if eye_name == "avg_ear":
                continue

            # 提取眼睛ROI
            x_min, y_min, x_max, y_max = eye_info["roi"]
            roi = frame[y_min:y_max, x_min:x_max]

            if roi.size == 0:
                continue

            # 转换为灰度图
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            # 使用高斯模糊减少噪声
            blurred = cv2.GaussianBlur(gray_roi, (7, 7), 0)

            # 使用自适应阈值二值化图像
            _, threshold = cv2.threshold(blurred, 40, 255, cv2.THRESH_BINARY_INV)

            # 查找轮廓
            contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # 找出最大的轮廓，可能是瞳孔
            if contours:
                max_contour = max(contours, key=cv2.contourArea)

                # 确保轮廓面积足够大，避免错误检测
                if cv2.contourArea(max_contour) > 10:
                    # 计算轮廓的矩
                    M = cv2.moments(max_contour)

                    # 防止除以0
                    if M["m00"] != 0:
                        # 计算瞳孔中心坐标（相对于ROI）
                        pupil_x = int(M["m10"] / M["m00"])
                        pupil_y = int(M["m01"] / M["m00"])

                        # 转换为原始图像坐标
                        pupil_x_abs = pupil_x + x_min
                        pupil_y_abs = pupil_y + y_min

                        # 计算瞳孔相对于眼睛中心的位置（归一化）
                        center_x, center_y = eye_info["center"]
                        eye_width = x_max - x_min
                        eye_height = y_max - y_min

                        rel_x = (pupil_x_abs - center_x) / (eye_width / 2) if eye_width > 0 else 0
                        rel_y = (pupil_y_abs - center_y) / (eye_height / 2) if eye_height > 0 else 0

                        # 保存特征
                        features[eye_name] = {
                            "pupil_pos": (pupil_x_abs, pupil_y_abs),
                            "rel_pos": (rel_x, rel_y)
                        }

                        # 在原始图像上绘制十字准星
                        self.draw_crosshair(frame, (pupil_x_abs, pupil_y_abs))

        return features

    def draw_crosshair(self, frame, center, size=6, color=(0, 255, 0), thickness=1):
        """在指定位置绘制十字准星"""
        x, y = center
        # 绘制水平线
        cv2.line(frame, (x - size, y), (x + size, y), color, thickness)
        # 绘制垂直线
        cv2.line(frame, (x, y - size), (x, y + size), color, thickness)

    def calibrate_eyes(self, rel_pos, eye_name):
        """校准眼球位置"""
        if not self.calibration_complete:
            rel_x, rel_y = rel_pos

            # 添加当前位置到校准历史
            self.calibration_eye_centers.append((rel_x, rel_y))

            # 至少收集30个样本进行校准
            if len(self.calibration_eye_centers) >= 30:
                # 计算平均值作为中心参考点
                center_x = np.mean([p[0] for p in self.calibration_eye_centers])
                center_y = np.mean([p[1] for p in self.calibration_eye_centers])

                self.calibration_center = (center_x, center_y)

                # 计算标准差，用于设置阈值
                std_x = np.std([p[0] for p in self.calibration_eye_centers])
                std_y = np.std([p[1] for p in self.calibration_eye_centers])

                # 设置方向阈值
                self.calibration_left = center_x - 1.5 * std_x
                self.calibration_right = center_x + 1.5 * std_x
                self.calibration_top = center_y - 1.5 * std_y
                self.calibration_bottom = center_y + 1.5 * std_y

                self.calibration_complete = True
                print(
                    f"眼球校准完成: 中心 ({center_x:.2f}, {center_y:.2f}), 阈值: 左 {self.calibration_left:.2f}, 右 {self.calibration_right:.2f}, 上 {self.calibration_top:.2f}, 下 {self.calibration_bottom:.2f}")

    def determine_gaze_direction(self, eye_features, is_closed):
        """确定眼球注视方向"""
        # 如果眼睛闭合，返回停止指令
        if is_closed:
            return 5  # Stop

        # 如果没有特征或校准未完成，返回中央
        if not eye_features or not self.calibration_complete:
            return 0  # Center

        # 优先使用左眼，如果左眼没有检测到，使用右眼
        eye_name = "left_eye" if "left_eye" in eye_features else "right_eye"

        if eye_name not in eye_features:
            return 0  # Center

        rel_x, rel_y = eye_features[eye_name]["rel_pos"]

        # 增加方向判断的稳定性
        # 只有当眼球移动超过阈值的一定比例时才判定为方向变化
        direction_threshold = 0.5  # 方向判断阈值

        # 判断方向
        if rel_x < self.calibration_left * direction_threshold:
            return 1  # Left
        elif rel_x > self.calibration_right * direction_threshold:
            return 2  # Right
        elif rel_y < self.calibration_top * direction_threshold:
            return 3  # Forward
        elif rel_y > self.calibration_bottom * direction_threshold:
            return 4  # Backward
        else:
            return 0  # Center

    def run(self, camera_id=0, show_face=True, show_gesture=True, show_eye=True):
        """主循环"""
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise RuntimeError("无法打开摄像头")

        # 获取摄像头帧率和分辨率
        cap_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"摄像头参数: 分辨率 {width}x{height}, 帧率 {cap_fps:.1f} FPS")

        # 创建输出窗口
        cv2.namedWindow("Control System", cv2.WINDOW_NORMAL)

        # 人脸验证标志
        face_verification_done = False

        try:
            while True:
                start_time = time.time()
                ret, frame = cap.read()
                if not ret:
                    break

                self.frame_count += 1

                # 只在开始时进行一次人脸验证
                # --- 已注释：不做人脸验证 ---
                # if not face_verification_done and show_face:
                #     frame, face_found, recognized_faces = self.process_face_recognition(frame)
                #     if face_found and recognized_faces:
                #         print(f"识别到人脸: {', '.join(recognized_faces)}")
                #         face_verification_done = True
                #         self.face_verified = True

                # 只有在人脸验证通过后才执行手势和眼动识别
                # if self.face_verified:  # 已注释：不做人脸验证
                if True:
                    # 处理手势识别
                    if show_gesture:
                        frame, gesture_classes = self.process_gesture_detection(frame)
                        if gesture_classes:
                            self.control_mode = "gesture"
                            self.last_mode_switch_time = time.time()

                    # 处理眼动追踪
                    if show_eye and (time.time() - self.last_mode_switch_time > self.mode_switch_interval):
                        frame, eye_tracked = self.process_eye_tracking(frame)
                        if eye_tracked and not gesture_classes:  # 只有在没有检测到手势时才切换到眼动控制
                            self.control_mode = "eye"
                else:
                    # 如果没有进行人脸验证，显示警告
                    cv2.putText(frame, "Please verify face first", (width // 2 - 150, height // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

                # 计算FPS
                self.fps_queue.append(1.0 / (time.time() - start_time))
                self.fps = sum(self.fps_queue) / len(self.fps_queue)

                # 绘制状态信息
                frame = self.draw_status(frame)

                # 显示结果
                cv2.imshow("Control System", frame)

                # 按'q'键退出
                if cv2.waitKey(1) == ord('q'):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()
            if self.serial_connection:
                self.serial_connection.close()

    def draw_status(self, frame):
        """在帧上绘制当前状态信息"""
        # 显示FPS
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 显示人脸验证状态
        status_color = (0, 255, 0) if self.face_verified else (0, 0, 255)
        status_text = "Face Verified" if self.face_verified else "Face Not Verified"
        cv2.putText(frame, status_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        # 显示已验证的用户
        if self.verified_users:
            users_text = f"Verified Users: {', '.join(self.verified_users)}"
            cv2.putText(frame, users_text, (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # 显示当前控制模式
        mode_text = f"Control Mode: {'Gesture' if self.control_mode == 'gesture' else 'Eye'}"
        cv2.putText(frame, mode_text, (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)

        return frame


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="人脸识别、手势识别和眼动追踪参数设置")
    parser.add_argument("--gesture-model", default="best.onnx", help="手势识别模型路径")
    parser.add_argument("--pic-dir", default='test/', help="存放人脸照片的文件夹")
    parser.add_argument("--video", default=0, help="视频地址，默认为摄像头")
    parser.add_argument("--distance", default=0.5, type=float, help="人脸识别距离阈值")
    parser.add_argument("--landmark-model", default="shape_predictor_68_face_landmarks.dat",
                        help="dlib面部特征点模型路径")
    parser.add_argument("--serial-port", default="COM5", help="串口号，如COM3")
    parser.add_argument("--baudrate", default=9600, type=int, help="波特率")
    parser.add_argument("--show-face", default=True, type=bool, help="是否显示人脸识别结果")
    parser.add_argument("--show-gesture", default=True, type=bool, help="是否显示手势识别结果")
    parser.add_argument("--show-eye", default=True, type=bool, help="是否显示眼动追踪结果")
    parser.add_argument("--face-verify-interval", default=30.0, type=float, help="人脸验证间隔时间(秒)")
    args = parser.parse_args()

    # 初始化融合检测器
    detector = FaceGestureDetector(
        gesture_model_path=args.gesture_model,
        gesture_input_shape=(640, 640),
        gesture_conf_thres=0.4,
        gesture_iou_thres=0.5,
        face_pic_dir=args.pic_dir,
        face_distance=args.distance,
        face_landmark_model=args.landmark_model,
        serial_port=args.serial_port,
        baudrate=args.baudrate,
        command_interval=0.5
    )

    # 设置人脸验证间隔
    detector.face_verify_interval = args.face_verify_interval

    # 运行检测
    detector.run(
        camera_id=args.video if args.video != 0 else 0,
        show_face=args.show_face,
        show_gesture=args.show_gesture,
        show_eye=args.show_eye
    )