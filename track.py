import cv2
import yaml
import os
from ultralytics import YOLO

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

model = YOLO(config["model"])

for seq_name in config["sequences"]:
    print(f"\n{'=' * 50}\nОбработка последовательности: {seq_name}\n{'=' * 50}")

    seq_path = os.path.join(config["dataset_path"], seq_name, "img1")
    out_dir = "output"
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(seq_path):
        print(f"Папка {seq_path} не найдена.")
        continue

    # Получение кадров
    frames = sorted([f for f in os.listdir(seq_path) if f.endswith(('.jpg', '.png'))])
    if not frames:
        continue

    first_frame = cv2.imread(os.path.join(seq_path, frames[0]))
    h, w, _ = first_frame.shape

    video_writer = cv2.VideoWriter(
        os.path.join(out_dir, f"{seq_name}_result.mp4"),
        cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h)
    )

    mot_results = []

    for frame_idx, frame_name in enumerate(frames, start=1):
        frame = cv2.imread(os.path.join(seq_path, frame_name))

        results = model.track(
            frame,
            persist=True,
            tracker=config["tracker"],
            conf=config["conf_threshold"],
            iou=config["iou_threshold"],
            classes=config["classes"],
            verbose=False
        )

        annotated_frame = results[0].plot()
        video_writer.write(annotated_frame)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            confs = results[0].boxes.conf.cpu().numpy()

            for box, tid, conf in zip(boxes, ids, confs):
                x_center, y_center, width, height = box
                x_top_left = x_center - width / 2
                y_top_left = y_center - height / 2
                mot_results.append([frame_idx, tid, x_top_left, y_top_left, width, height, conf, -1, -1, -1])

    video_writer.release()

    txt_path = os.path.join(out_dir, f"{seq_name}.txt")
    with open(txt_path, "w") as f:
        for row in mot_results:
            f.write(','.join(map(str, row)) + '\n')

    print(f"Готово. Сохранено: {seq_name}_result.mp4 и {seq_name}.txt")

print("\nВсе последовательности обработаны.")