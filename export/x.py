from ultralytics import YOLO


def test(m):
    model = YOLO(m)
    results = model('./bus.jpg', half=True, imgsz=160)
    for result in results:
        result.save(filename=f"{m}-result.jpg")  # save to disk


test('yolo11n_ncnn_model')
test('yolo11n.mnn')
