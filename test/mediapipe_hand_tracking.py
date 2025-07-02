import cv2
import mediapipe as mp

# 初始化MediaPipe Hands组件
mp_hands = mp.solutions.hands
# Hands()有很多参数可以设置:
# static_image_mode: False表示处理视频流/连续帧，True表示处理单张静态图片。默认为False。
# max_num_hands: 最多检测的手的数量。默认为2。
# min_detection_confidence: 手部检测模型的最小置信度阈值 (0.0-1.0)。默认为0.5。
# min_tracking_confidence: 手部关键点跟踪模型的最小置信度阈值 (0.0-1.0)。默认为0.5。
# 对于视频追踪，较高的跟踪置信度有助于减少抖动，但过高可能会导致跟丢。
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,  # 根据你的视频中可能出现的最大手数进行调整
    min_detection_confidence=0.2,
    min_tracking_confidence=0.2) # 稍微提高跟踪置信度可能使追踪更稳定

# 初始化MediaPipe Drawing组件，用于绘制手部关键点和连接线
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles # 包含一些预设的绘制样式

# 指定你的视频文件路径
video_file_path = 'shoplift1.mp4' # 请将这里替换成你的视频文件实际路径

# 打开视频文件
cap = cv2.VideoCapture(video_file_path)

# 检查视频是否成功打开
if not cap.isOpened():
    print(f"错误：无法打开视频文件: {video_file_path}")
    exit()

# 获取视频的宽度和高度，用于后续可能的大小调整或窗口创建
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps = int(cap.get(cv2.CAP_PROP_FPS)) # 如果需要，也可以获取FPS

print(f"正在处理视频: {video_file_path}")
print(f"分辨率: {frame_width}x{frame_height}")

while cap.isOpened():
    # 读取一帧视频
    success, frame = cap.read()
    if not success:
        print("视频处理完成或读取下一帧失败。")
        break

    # ---- 新增：缩放图像帧 ----
    target_width = 1280  # 或者 640，可以尝试不同值
    h, w, _ = frame.shape
    aspect_ratio = w / h
    target_height = int(target_width / aspect_ratio)
    resized_frame = cv2.resize(frame, (target_width, target_height))
    # -------------------------

    # MediaPipe Hands期望输入的是RGB格式的图像
    # 使用缩放后的帧进行处理
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)  # 注意这里用 resized_frame

    rgb_frame.flags.writeable = False
    results = hands.process(rgb_frame)
    rgb_frame.flags.writeable = True

    # 在缩放后的帧上绘制 (或者在原始帧上绘制，但坐标需要相应转换回去，初次排查先在缩放帧上绘制)
    # output_display_frame = resized_frame.copy() # 我们要在 resized_frame 上画
    output_display_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)  # 如果想从处理过的rgb_frame转回

    if results.multi_hand_landmarks:
        print(f"检测到 {len(results.multi_hand_landmarks)} 只手。")  # 调试信息
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image=output_display_frame,  # 在缩放后的帧上绘制
                landmark_list=hand_landmarks,
                connections=mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style())
    else:
        print("当前帧未检测到手。")  # 调试信息

    cv2.imshow('MediaPipe Hands - Hand Tracking', output_display_frame)  # 显示缩放后的帧

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
# 释放资源
hands.close() # 关闭MediaPipe Hands实例
cap.release() # 释放视频捕获对象
cv2.destroyAllWindows() # 关闭所有OpenCV创建的窗口

print("程序执行完毕。")