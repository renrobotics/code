import cv2
import numpy as np
import time
from ultralytics import YOLO
import argparse
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description='使用YOLO检测视频中的盗窃行为')
    parser.add_argument('--video', type=str, default='shoplift1111.mp4', help='输入视频路径')
    parser.add_argument('--conf', type=float, default=0.25, help='置信度阈值')
    parser.add_argument('--output', type=str, default='output_detection.mp4', help='输出视频路径')
    return parser.parse_args()


class ShopliftingDetector:
    def __init__(self, video_path, conf_threshold=0.25, output_path='output_detection.mp4'):
        self.video_path = video_path
        self.conf_threshold = conf_threshold
        self.output_path = output_path

        # 初始化YOLO模型
        self.model = YOLO('yolov8n.pt')  # 使用预训练的YOLOv8模型

        # 人物跟踪的数据结构
        self.tracks = {}
        self.track_id = 0
        self.suspicious_actions = defaultdict(int)

        # 定义感兴趣的类别 (基于COCO数据集)
        self.person_class_id = 0  # COCO中的人类ID
        self.product_class_ids = [39, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
                                  52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67]  # 物品ID

    def process_video(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"无法打开视频: {self.video_path}")
            return

        # 获取视频信息
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 设置输出视频
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))

        frame_id = 0
        start_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_id += 1
            if frame_id % 10 == 0:  # 显示处理进度
                elapsed = time.time() - start_time
                fps_processing = frame_id / elapsed
                print(
                    f"处理: {frame_id}/{frame_count} 帧 ({100 * frame_id / frame_count:.1f}%) - 速度: {fps_processing:.1f} FPS")

            # 使用YOLO检测对象
            results = self.model(frame, conf=self.conf_threshold)

            # 分析检测结果
            self.analyze_frame(frame, results[0], frame_id)

            # 在输出视频中标注检测和警告
            annotated_frame = self.annotate_frame(frame, results[0])
            out.write(annotated_frame)

        cap.release()
        out.release()
        print(f"处理完成，输出保存至: {self.output_path}")

    def analyze_frame(self, frame, result, frame_id):
        """分析当前帧并更新跟踪信息"""
        height, width = frame.shape[:2]

        # 获取检测到的物体
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        cls_ids = result.boxes.cls.cpu().numpy().astype(int)

        # 处理所有检测到的人
        current_persons = []
        current_products = []

        for i, (box, conf, cls_id) in enumerate(zip(boxes, confs, cls_ids)):
            if cls_id == self.person_class_id:
                # 保存人的位置
                current_persons.append({
                    'box': box,
                    'center': ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2),
                    'area': (box[2] - box[0]) * (box[3] - box[1]),
                    'conf': conf
                })
            elif cls_id in self.product_class_ids:
                # 保存产品的位置
                current_products.append({
                    'box': box,
                    'center': ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2),
                    'area': (box[2] - box[0]) * (box[3] - box[1]),
                    'conf': conf,
                    'cls_id': cls_id
                })

        # 更新人物跟踪
        self.update_tracks(current_persons, current_products, frame_id)

        # 检测可疑行为
        self.detect_suspicious_behavior(frame_id)

    def update_tracks(self, current_persons, current_products, frame_id):
        """更新人物跟踪信息"""
        # 如果是第一帧，初始化跟踪
        if not self.tracks:
            for person in current_persons:
                self.track_id += 1
                self.tracks[self.track_id] = {
                    'positions': [person['center']],
                    'last_seen': frame_id,
                    'products_near': [],
                    'suspicious_actions': 0,
                    'history': [(frame_id, person['box'], [])]  # (帧ID, 人物边界框, 附近产品)
                }
            return

        # 匹配当前检测结果与现有跟踪
        matched_tracks = set()
        unmatched_persons = []

        for person in current_persons:
            best_match = None
            min_dist = float('inf')

            for track_id, track in self.tracks.items():
                if track['last_seen'] == frame_id:  # 已经匹配过了
                    continue

                last_pos = track['positions'][-1]
                dist = np.sqrt((person['center'][0] - last_pos[0]) ** 2 +
                               (person['center'][1] - last_pos[1]) ** 2)

                if dist < min_dist and dist < 100:  # 假设距离阈值为100像素
                    min_dist = dist
                    best_match = track_id

            if best_match is not None:
                # 更新匹配的轨迹
                self.tracks[best_match]['positions'].append(person['center'])
                self.tracks[best_match]['last_seen'] = frame_id

                # 查找此人附近的产品
                nearby_products = []
                for product in current_products:
                    prod_dist = np.sqrt((person['center'][0] - product['center'][0]) ** 2 +
                                        (person['center'][1] - product['center'][1]) ** 2)
                    if prod_dist < 150:  # 产品在人物附近的阈值
                        nearby_products.append(product)

                self.tracks[best_match]['history'].append((frame_id, person['box'], nearby_products))
                matched_tracks.add(best_match)
            else:
                unmatched_persons.append(person)

        # 为未匹配的人创建新轨迹
        for person in unmatched_persons:
            self.track_id += 1
            self.tracks[self.track_id] = {
                'positions': [person['center']],
                'last_seen': frame_id,
                'products_near': [],
                'suspicious_actions': 0,
                'history': [(frame_id, person['box'], [])]
            }

    def detect_suspicious_behavior(self, frame_id):
        """检测可疑行为模式"""
        for track_id, track in self.tracks.items():
            # 只分析活跃的轨迹
            if frame_id - track['last_seen'] > 30:  # 如果超过30帧未见，则认为不再活跃
                continue

            # 至少需要一定的历史记录才能分析
            if len(track['history']) < 5:
                continue

            # 分析最近的历史记录
            recent_history = track['history'][-10:]  # 取最近10条记录
            product_counts = defaultdict(int)

            # 统计产品出现和消失的情况
            for i in range(1, len(recent_history)):
                prev_products = {tuple(p['box']): p for p in recent_history[i - 1][2]}
                curr_products = {tuple(p['box']): p for p in recent_history[i][2]}

                # 检查产品突然出现或消失的情况
                for prod_box, prod in curr_products.items():
                    if prod_box not in prev_products:
                        product_counts[prod['cls_id']] += 1

                for prod_box, prod in prev_products.items():
                    if prod_box not in curr_products:
                        product_counts[prod['cls_id']] -= 1

            # 如果有物品突然消失，增加可疑度
            suspicious_score = 0
            for cls_id, count in product_counts.items():
                if count < -1:  # 物品消失
                    suspicious_score += abs(count)

            if suspicious_score > 0:
                self.tracks[track_id]['suspicious_actions'] += suspicious_score
                self.suspicious_actions[track_id] += suspicious_score

    def annotate_frame(self, frame, result):
        """标注帧并添加警告信息"""
        # 首先绘制YOLO的标准检测结果
        annotated_frame = result.plot()

        # 添加自定义盗窃警告
        for track_id, track in self.tracks.items():
            if track['suspicious_actions'] > 3:  # 可疑行为阈值
                if len(track['positions']) > 0:
                    x, y = int(track['positions'][-1][0]), int(track['positions'][-1][1])
                    cv2.putText(annotated_frame, f"ID {track_id}: 可疑行为!", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    cv2.circle(annotated_frame, (x, y), 5, (0, 0, 255), -1)

        # 添加统计信息
        cv2.putText(annotated_frame,
                    f"检测到的人数: {sum(1 for t in self.tracks.values() if t['last_seen'] > result.frame_id - 30)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(annotated_frame,
                    f"可疑行为警告: {sum(1 for t in self.tracks.values() if t['suspicious_actions'] > 3)}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return annotated_frame


def main():
    args = parse_args()
    detector = ShopliftingDetector(
        video_path=args.video,
        conf_threshold=args.conf,
        output_path=args.output
    )
    detector.process_video()


if __name__ == "__main__":
    main()