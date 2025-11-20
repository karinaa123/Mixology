from roboflow import Roboflow

rf = Roboflow(api_key="5n5yW3l54FDmCaKLwocD")
project = rf.workspace("self-zb436").project("bottles-q0ajc")
version = project.version(1)
dataset = version.download("yolov8")
