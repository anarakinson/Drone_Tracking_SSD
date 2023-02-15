import subprocess
import os
import wget

from utils.paths import paths, files


# Verify instalation
VERIFICATION_SCRIPT = os.path.join(
    paths['APIMODEL_PATH'], 'research', 'object_detection', 'builders', 'model_builder_tf2_test.py'
)
subprocess.call(VERIFICATION_SCRIPT, shell=True)


if os.name == 'posix':
    subprocess.call(f"wget {PRETRAINED_MODEL_URL}", shell=True)
    subprocess.call(f"mv {PRETRAINED_MODEL_NAME + '.tar.gz'} {paths['PRETRAINED_MODEL_PATH']}", shell=True)
    subprocess.call(f"cd {paths['PRETRAINED_MODEL_PATH']} && tar -zxvf {PRETRAINED_MODEL_NAME + '.tar.gz'}", shell=True)

if os.name == 'nt':
    wget.download(PRETRAINED_MODEL_URL)
    subprocess.call(f"move {PRETRAINED_MODEL_NAME + '.tar.gz'} {paths['PRETRAINED_MODEL_PATH']}", shell=True)
    subprocess.call(f"cd {paths['PRETRAINED_MODEL_PATH']} && tar -zxvf {PRETRAINED_MODEL_NAME + '.tar.gz'}", shell=True)
