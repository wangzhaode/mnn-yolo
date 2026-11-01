from ultralytics import YOLO

model = YOLO("yolo11n.pt")

model.export(format='mnn')

mnn_model = YOLO("yolo11n.mnn")
results = mnn_model('https://ultralytics.com/images/bus.jpg', half=True)


for result in results:
    result.show()  # display to screen
    result.save(filename="result.jpg")  # save to disk
