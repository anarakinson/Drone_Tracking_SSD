import subprocess
import os
import wget

from utils.paths import paths, files
from utils.labelmap import create_label_map

def main():

    for path in paths.values():
        if not os.path.exists(path):
            os.mkdir(path)


    # Install tf object-detection
    if os.name == 'posix':
        subprocess.call("apt install protobuf-compiler", shell=True)
        script = f'cd Tensorflow/models/research && \
        protoc object_detection/protos/*.proto --python_out=. && \
        cp object_detection/packages/tf2/setup.py . && \
        python -m pip install .'
        subprocess.call(script, shell=True)

    if os.name == 'nt':
        url = "https://github.com/protocolbuffers/protobuf/releases/download/v3.15.6/protoc-3.15.6-win64.zip"
        wget.download(url)

        subprocess.call(f"move protoc-3.15.6-win64.zip {paths['PROTOC_PATH']}", shell=True)
        subprocess.call(f"cd {paths['PROTOC_PATH']} && tar -xf protoc-3.15.6-win64.zip", shell=True)

        os.environ['PATH'] += os.pathsep + os.path.abspath(os.path.join(paths['PROTOC_PATH'], 'bin'))

        script = f'cd Tensorflow/models/research && \
        protoc object_detection/protos/*.proto --python_out=. && \
        copy object_detection\\packages\\tf2\\setup.py setup.py && \
        python setup.py build && \
        python setup.py install'
        subprocess.call(script, shell=True)

        subprocess.call("cd Tensorflow/models/research/slim && pip install -e", shell=True)
        # (--editable) Install a project from a local project path


    create_label_map()

if __name__ == "__main__":
    main()
