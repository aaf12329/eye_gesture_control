import cv2
import numpy as np
import dlib
import torch
import onnxruntime
import serial
import threading
from collections import deque
import time
from scipy import stats

class EyeController:
    def __init__(self, camera_id=0, serial_port=None, baud_rate=9600):
        # 初始化摄像头
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        # 初始化人脸检测器
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

        # 图像处理参数
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.gaussian_kernel = (5, 5)
        self.adaptive_threshold_block = 11
        self.adaptive_threshold_C = 2

        # 瞳孔检测参数
        self.pupil_min_area = 10
        self.pupil_max_area = 1000
        self.pupil_circularity_threshold = 0.7

        # 控制参数
        self.origin = None
        self.sensitivity = 0.18
        self.dynamic_scaling = True
        self.buffer_size = 7
        self.position_buffer = deque(maxlen=self.buffer_size)
        self.calibration_data = []
        self.thresholds = (20, 15)
        self.last_output = 0
        self.calibration_frames = 60

        # 状态参数
        self.calibrating = True
        self.show_debug = True
        self.runtime_data = deque(maxlen=100)
        self.kalman_state = None
        self.kalman_matrix = None
        self.confirm_counter = 0

        # Smoothing parameters for pupil
        self.left_eye_smooth_pos = None
        self.right_eye_smooth_pos = None
        self.pupil_smooth_factor = 0.7  # Weight for the previous smoothed position

        # 串口通信
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        self.ser = None
        self._init_serial()

        # 性能监控
        self.fps = 0
        self.fps_queue = deque(maxlen=10)
        self.last_frame_time = time.time()

    def _init_serial(self):
        """初始化串口通信"""
        if self.serial_port:
            try:
                self.ser = serial.Serial(self.serial_port, self.baud_rate, timeout=1)
                print(f"已连接到串口：{self.serial_port}")
            except Exception as e:
                print(f"串口连接失败：{e}")
                self.ser = None

    def _send_command(self, command):
        """发送命令到串口"""
        if self.ser and self.ser.is_open:
            try:
                self.ser.write(f"{command}\n".encode())
                print(f"发送命令：{command} - {self._get_command_description(command)}")
            except Exception as e:
                print(f"发送命令失败：{e}")

    def _get_command_description(self, command):
        """获取命令描述"""
        command_desc = {
            "0": "停止",
            "1": "前进",
            "2": "后退",
            "3": "左转",
            "4": "右转"
        }
        return command_desc.get(command, "未知")

    def _calculate_dynamic_thresholds(self):
        """计算动态阈值"""
        if len(self.runtime_data) < 20:
            return (
                self.thresholds[0] * self.sensitivity,
                self.thresholds[1] * self.sensitivity
            )

        dx = [x for x, y in self.runtime_data]
        dy = [y for x, y in self.runtime_data]

        mad_x = stats.median_abs_deviation(dx, scale='normal') * 2.0
        mad_y = stats.median_abs_deviation(dy, scale='normal') * 2.0

        range_x = np.percentile(dx, 95) - np.percentile(dx, 5)
        range_y = np.percentile(dy, 95) - np.percentile(dy, 5)

        auto_sensitivity = min(0.3, max(0.1, 0.5 / (range_x + range_y)))

        th_x = max(10, mad_x * self.sensitivity * (auto_sensitivity if self.dynamic_scaling else 1))
        th_y = max(8, mad_y * self.sensitivity * (auto_sensitivity if self.dynamic_scaling else 1))

        return (th_x, th_y)

    def _adaptive_filter(self, new_point):
        """卡尔曼滤波"""
        if self.kalman_state is None:
            self.kalman_state = np.array([[new_point[0]], [new_point[1]], [0], [0]], dtype=np.float32)
            self.kalman_matrix = np.eye(4, dtype=np.float32) * 0.95

        predicted_state = np.dot(self.kalman_matrix, self.kalman_state)
        measurement = np.array([[new_point[0]], [new_point[1]]])
        innovation = measurement - predicted_state[:2]
        self.kalman_state = predicted_state + 0.5 * np.vstack((innovation, np.zeros((2, 1))))
        return self.kalman_state[:2, 0].flatten().tolist()

    def _get_smoothed_position(self):
        """获取平滑位置"""
        if not self.position_buffer:
            return (0, 0)

        buffer_array = np.array(self.position_buffer)
        
        if len(buffer_array) < 3:
            return np.median(buffer_array, axis=0)

        # 分离处理x/y轴
        sorted_x = buffer_array[buffer_array[:, 0].argsort()]
        sorted_y = buffer_array[buffer_array[:, 1].argsort()]

        # 使用加权平均
        weights = np.linspace(1, 0.5, len(buffer_array))
        
        weighted_median = np.array([
            np.average(sorted_x[:, 0], weights=weights),
            np.average(sorted_y[:, 1], weights=weights)
        ])

        return self._adaptive_filter(weighted_median)

    def _direction_detection(self, dx, dy):
        """方向判断逻辑"""
        th_x, th_y = self.thresholds
        magnitude = np.sqrt(dx ** 2 + dy ** 2)
        angle = np.degrees(np.arctan2(dy, dx)) % 360

        if magnitude < (th_x + th_y) / 2 * 0.6:
            return 0  # 停止

        if -45 <= angle < 45:
            return 4  # 右转
        elif 45 <= angle < 135:
            return 1  # 前进
        elif 135 <= angle < 225:
            return 3  # 左转
        else:
            return 2  # 后退

    def _remove_outliers(self, data, iterations=2):
        """异常值过滤"""
        clean = data.copy()
        for _ in range(iterations):
            if len(clean) < 10:
                break
            x = [d[0] for d in clean]
            y = [d[1] for d in clean]
            q1_x, q3_x = np.percentile(x, [25, 75])
            q1_y, q3_y = np.percentile(y, [25, 75])
            iqr_x = q3_x - q1_x
            iqr_y = q3_y - q1_y
            clean = [d for d in clean
                     if (q1_x - 1.5 * iqr_x <= d[0] <= q3_x + 1.5 * iqr_x)
                     and (q1_y - 1.5 * iqr_y <= d[1] <= q3_y + 1.5 * iqr_y)]
        return clean

    def _process_eye(self, frame, eye_roi, eye_center, roi_x_start, roi_y_start): # MODIFIED: Added roi_x_start, roi_y_start
        """处理单个眼睛区域"""
        if eye_roi.size == 0:
            return None

        # 图像预处理
        gray_roi = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
        enhanced_roi = self.clahe.apply(gray_roi)
        blurred = cv2.GaussianBlur(enhanced_roi, self.gaussian_kernel, 0)

        # 自适应阈值处理
        threshold = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, self.adaptive_threshold_block,
            self.adaptive_threshold_C
        )

        # 形态学操作
        kernel = np.ones((3, 3), np.uint8)
        threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
        threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)

        # 查找轮廓
        contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if self.pupil_min_area < area < self.pupil_max_area:
                    perimeter = cv2.arcLength(contour, True)
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    
                    if circularity > self.pupil_circularity_threshold:
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            pupil_x = int(M["m10"] / M["m00"])
                            pupil_y = int(M["m01"] / M["m00"])
                            
                            # 转换为原始图像坐标
                            pupil_x_abs = pupil_x + roi_x_start # MODIFIED: Corrected transformation
                            pupil_y_abs = pupil_y + roi_y_start # MODIFIED: Corrected transformation
                            
                            # 绘制瞳孔轮廓
                            # MODIFIED: Offset contour by ROI start coordinates for drawing on the main frame
                            offset_contour = contour + np.array([roi_x_start, roi_y_start])
                            cv2.drawContours(frame, [offset_contour], -1, (0, 255, 0), 1)
                            
                            return (pupil_x_abs, pupil_y_abs)
        
        return None

    def _print_eye_movement(self, x, y, dx, dy, current):
        """打印眼球运动情况"""
        print(f"\r眼球位置: ({x:.1f}, {y:.1f}) | 移动: ({dx:.1f}, {dy:.1f}) | 动作: {self._get_command_description(str(current))}", end="")

    def _draw_face_box(self, frame, face):
        """绘制人脸检测框"""
        x, y = face.left(), face.top()
        w, h = face.right() - x, face.bottom() - y
        # 绘制主框
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # 绘制四个角
        corner_length = min(w, h) // 4
        # 左上角
        cv2.line(frame, (x, y), (x + corner_length, y), (0, 255, 0), 2)
        cv2.line(frame, (x, y), (x, y + corner_length), (0, 255, 0), 2)
        # 右上角
        cv2.line(frame, (x + w - corner_length, y), (x + w, y), (0, 255, 0), 2)
        cv2.line(frame, (x + w, y), (x + w, y + corner_length), (0, 255, 0), 2)
        # 左下角
        cv2.line(frame, (x, y + h - corner_length), (x, y + h), (0, 255, 0), 2)
        cv2.line(frame, (x, y + h), (x + corner_length, y + h), (0, 255, 0), 2)
        # 右下角
        cv2.line(frame, (x + w - corner_length, y + h), (x + w, y + h), (0, 255, 0), 2)
        cv2.line(frame, (x + w, y + h), (x + w, y + h - corner_length), (0, 255, 0), 2)

    def _draw_pupil_point(self, frame, point, color=(0, 0, 255)):
        """绘制瞳孔点"""
        if point is not None:
            x, y = point
            # 绘制十字准星
            size = 5  # '+'号的大小，可以根据实际效果调整
            cv2.line(frame, (int(x - size), int(y)), (int(x + size), int(y)), color, 2) # 水平线
            cv2.line(frame, (int(x), int(y - size)), (int(x), int(y + size)), color, 2) # 垂直线

    def _draw_center_point(self, frame, left_eye_center, right_eye_center):
        """绘制两眼之间的中心点"""
        center_x = (left_eye_center[0] + right_eye_center[0]) // 2
        center_y = (left_eye_center[1] + right_eye_center[1]) // 2
        
        # 绘制十字准星
        size = 10
        cv2.line(frame, (center_x - size, center_y), (center_x + size, center_y), (0, 255, 255), 1)
        cv2.line(frame, (center_x, center_y - size), (center_x, center_y + size), (0, 255, 255), 1)
        
        # 绘制空心圆
        cv2.circle(frame, (center_x, center_y), 12, (0, 255, 255), 1)
        # 绘制中心点
        cv2.circle(frame, (center_x, center_y), 2, (0, 255, 255), -1)

    def run(self):
        """主循环"""
        print("开始眼球追踪...")
        print("校准期间请保持头部静止并直视前方...")
        print("控制说明：")
        print("- 向上看 -> 前进")
        print("- 向左看 -> 左转")
        print("- 向右看 -> 右转")
        print("- 向下看 -> 后退")
        print("- 直视 -> 停止")
        
        calibration_counter = 0
        last_valid = None

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # 人脸检测
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray)

            if len(faces) > 0:
                face = faces[0]
                shape = self.predictor(gray, face)

                # 绘制人脸框 (已移除)
                # self._draw_face_box(frame, face)

                # 获取眼睛区域的特征点
                left_eye_landmarks = np.array([(shape.part(i).x, shape.part(i).y) for i in range(36, 42)]) # MODIFIED: Renamed for clarity
                right_eye_landmarks = np.array([(shape.part(i).x, shape.part(i).y) for i in range(42, 48)]) # MODIFIED: Renamed for clarity

                # 计算眼睛中心点
                left_eye_center = np.mean(left_eye_landmarks, axis=0).astype(int)
                right_eye_center = np.mean(right_eye_landmarks, axis=0).astype(int)

                # 绘制两眼之间的中心点 (已移除)
                # self._draw_center_point(frame, left_eye_center, right_eye_center)

                # 初始化瞳孔位置
                left_pupil = None
                right_pupil = None
                
                h_frame, w_frame = frame.shape[:2] # Frame dimensions for boundary checks

                # 处理左眼
                if left_eye_landmarks.size > 0:
                    left_roi_x_start = np.min(left_eye_landmarks[:, 0])
                    left_roi_y_start = np.min(left_eye_landmarks[:, 1])
                    left_roi_x_end = np.max(left_eye_landmarks[:, 0])
                    left_roi_y_end = np.max(left_eye_landmarks[:, 1])

                    # Ensure ROI coordinates are valid and within frame bounds
                    left_roi_x_start = max(0, left_roi_x_start)
                    left_roi_y_start = max(0, left_roi_y_start)
                    left_roi_x_end = min(w_frame, left_roi_x_end)
                    left_roi_y_end = min(h_frame, left_roi_y_end)

                    if left_roi_x_start < left_roi_x_end and left_roi_y_start < left_roi_y_end:
                        left_eye_roi = frame[
                            left_roi_y_start:left_roi_y_end,
                            left_roi_x_start:left_roi_x_end
                        ]
                        if left_eye_roi.size > 0: # Ensure ROI is not empty
                           left_pupil = self._process_eye(frame, left_eye_roi, left_eye_center, left_roi_x_start, left_roi_y_start) # MODIFIED: Pass ROI start coords
                    # else: left_pupil remains None

                # 处理右眼
                if right_eye_landmarks.size > 0:
                    right_roi_x_start = np.min(right_eye_landmarks[:, 0])
                    right_roi_y_start = np.min(right_eye_landmarks[:, 1])
                    right_roi_x_end = np.max(right_eye_landmarks[:, 0])
                    right_roi_y_end = np.max(right_eye_landmarks[:, 1])

                    # Ensure ROI coordinates are valid and within frame bounds
                    right_roi_x_start = max(0, right_roi_x_start)
                    right_roi_y_start = max(0, right_roi_y_start)
                    right_roi_x_end = min(w_frame, right_roi_x_end)
                    right_roi_y_end = min(h_frame, right_roi_y_end)
                    
                    if right_roi_x_start < right_roi_x_end and right_roi_y_start < right_roi_y_end:
                        right_eye_roi = frame[
                            right_roi_y_start:right_roi_y_end,
                            right_roi_x_start:right_roi_x_end
                        ]
                        if right_eye_roi.size > 0: # Ensure ROI is not empty
                            right_pupil = self._process_eye(frame, right_eye_roi, right_eye_center, right_roi_x_start, right_roi_y_start) # MODIFIED: Pass ROI start coords
                    # else: right_pupil remains None

                # Smooth left pupil position
                if left_pupil is not None:
                    if self.left_eye_smooth_pos is None:
                        self.left_eye_smooth_pos = left_pupil
                    else:
                        smooth_x = int(left_pupil[0] * (1 - self.pupil_smooth_factor) + self.left_eye_smooth_pos[0] * self.pupil_smooth_factor)
                        smooth_y = int(left_pupil[1] * (1 - self.pupil_smooth_factor) + self.left_eye_smooth_pos[1] * self.pupil_smooth_factor)
                        self.left_eye_smooth_pos = (smooth_x, smooth_y)
                # If left_pupil is None, self.left_eye_smooth_pos retains its last value (or remains None if never detected)

                # Smooth right pupil position
                if right_pupil is not None:
                    if self.right_eye_smooth_pos is None:
                        self.right_eye_smooth_pos = right_pupil
                    else:
                        smooth_x = int(right_pupil[0] * (1 - self.pupil_smooth_factor) + self.right_eye_smooth_pos[0] * self.pupil_smooth_factor)
                        smooth_y = int(right_pupil[1] * (1 - self.pupil_smooth_factor) + self.right_eye_smooth_pos[1] * self.pupil_smooth_factor)
                        self.right_eye_smooth_pos = (smooth_x, smooth_y)
                # If right_pupil is None, self.right_eye_smooth_pos retains its last value

                # 绘制平滑后的瞳孔点
                self._draw_pupil_point(frame, self.left_eye_smooth_pos, (0, 0, 255))  # 红色
                self._draw_pupil_point(frame, self.right_eye_smooth_pos, (255, 0, 0)) # 蓝色

                # 使用平滑后的左眼或右眼的位置（优先使用左眼）
                # pupil_pos will be the one used for control logic and calibration
                # It should be the most reliable smoothed position available
                
                # Determine the primary pupil position for control based on smoothed values
                current_smoothed_pupil = self.left_eye_smooth_pos if self.left_eye_smooth_pos is not None else self.right_eye_smooth_pos

                if current_smoothed_pupil is not None:
                    last_valid = current_smoothed_pupil  # Update last_valid with the current smoothed position
                    pupil_pos = current_smoothed_pupil
                else:
                    # If no current smoothed pupil (neither eye detected for a while), use the last known valid smoothed position
                    pupil_pos = last_valid 
                
                if pupil_pos is not None:
                    last_valid = pupil_pos
                    x, y = pupil_pos

                    if self.calibrating:
                        if calibration_counter < self.calibration_frames:
                            self.calibration_data.append((x, y))
                            calibration_counter += 1
                            cv2.putText(frame, f"校准中... {calibration_counter}/{self.calibration_frames}",
                                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        else:
                            clean_data = self._remove_outliers(self.calibration_data)
                            self.origin = (
                                np.median([d[0] for d in clean_data]),
                                np.median([d[1] for d in clean_data])
                            )
                            self.calibrating = False
                            print("\n校准完成！")
                    else:
                        # 计算位移
                        dx = x - self.origin[0]
                        dy = y - self.origin[1]
                        self.position_buffer.append((dx, dy))

                        # 获取平滑坐标
                        smooth_dx, smooth_dy = self._get_smoothed_position()
                        self.runtime_data.append((smooth_dx, smooth_dy))

                        # 更新阈值
                        self.thresholds = self._calculate_dynamic_thresholds()

                        # 方向判断
                        current = self._direction_detection(smooth_dx, smooth_dy)

                        # 打印眼球运动情况
                        self._print_eye_movement(x, y, smooth_dx, smooth_dy, current)

                        # 状态确认
                        if current != self.last_output:
                            self.confirm_counter += 1
                            if self.confirm_counter >= 2:
                                self.last_output = current
                                self._send_command(str(current))
                                self.confirm_counter = 0
                        else:
                            self.confirm_counter = 0

                        # 显示调试信息
                        if self.show_debug:
                            debug_info = [
                                f"位置: ({x:.1f}, {y:.1f})",
                                f"移动: ({smooth_dx:.1f}, {smooth_dy:.1f})",
                                f"阈值: {self.thresholds}",
                                f"动作: {self._get_command_description(str(current))}"
                            ]
                            y_pos = 30
                            for text in debug_info:
                                cv2.putText(frame, text, (10, y_pos),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                                y_pos += 25

            # 更新FPS
            current_time = time.time()
            self.fps_queue.append(1.0 / (current_time - self.last_frame_time))
            self.fps = sum(self.fps_queue) / len(self.fps_queue)
            self.last_frame_time = current_time

            # 显示FPS
            cv2.putText(frame, f"帧率: {self.fps:.1f}", (10, frame.shape[0] - 10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 显示主窗口
            cv2.imshow("眼球控制器", frame)

            if cv2.waitKey(1) == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()
        if self.ser:
            self.ser.close()
        print("\n眼球追踪已停止。")

if __name__ == "__main__":
    controller = EyeController(camera_id=0, serial_port="COM5", baud_rate=9600)
    controller.run()
