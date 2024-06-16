from ultralytics import YOLO

model = YOLO("/content/runs/detect/train3/weights/best.pt")
#model training train on Googlecolab A100
#results = model.train(data="/content/data.yaml", epochs=100, batch=32, imgsz=224, patience=15, device=[0])

#inference
results = model(source='/content/GAN',
                conf=0.5,
                save=True,
                save_txt=True,
                project="/content/res", name="prediction")