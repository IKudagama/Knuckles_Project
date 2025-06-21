from roboflow import Roboflow

rf = Roboflow(api_key="WIz1jGSuFHxImCaj6i2x")
project = rf.workspace("duka").project("trees-hjbhh")
version = project.version(1)
dataset = version.download("yolov11")
                