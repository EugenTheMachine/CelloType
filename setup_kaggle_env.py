import subprocess
import signal
import os

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Execution interrupted due to timeout")

def create_conda_env(env_name, packages, channels):
    command = [
        "conda", "create", "--name", env_name, "--yes", "--no-deps"
    ] + packages + channels
    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(60)  # Set timeout to 60 seconds (1 minute)
        subprocess.run(command)
        signal.alarm(0)  # Disable the alarm if command finishes in time
    except TimeoutException:
        print("Execution interrupted due to timeout")

def setup_env():
    # Clone the CelloType repository
    subprocess.run(["git", "clone", "https://github.com/EugenTheMachine/CelloType.git"])
    
    # Create the myenv conda environment with specific packages
    env_name = "myenv"
    packages = ["pytorch==1.9.0", "torchvision==0.10.0", "cudatoolkit=11.1"]
    channels = ["-c", "pytorch", "-c", "nvidia"]
    create_conda_env(env_name, packages, channels)
    
    # Install detectron2 from GitHub
    subprocess.run(["python", "-m", "pip", "install", "git+https://github.com/facebookresearch/detectron2.git"])
    
    # Clone the Deformable-DETR repository
    subprocess.run(["git", "clone", "https://github.com/fundamentalvision/Deformable-DETR.git"])
    
    # Run the make.sh script in Deformable-DETR/models/ops
    os.chdir('/kaggle/working/Deformable-DETR/models/ops')
    subprocess.run(["sh", "./make.sh"])
    
    # Change directory back to CelloType
    os.chdir('/kaggle/working/CelloType')

# Call the setup_env function to execute all steps
if __name__ == "__main__":
    setup_env()