import subprocess
import os

def run_shell(cmd):
    try:
        result = subprocess.check_output(
            cmd,
            shell=True,
            stderr=subprocess.STDOUT,
            text=True
        )
        return result[:5000]
    except subprocess.CalledProcessError as e:
        return e.output


def read_file(path):
    try:
        with open(path, "r") as f:
            return f.read()[:5000]
    except Exception as e:
        return str(e)


def list_files(path="."):
    try:
        return "\n".join(os.listdir(path))
    except Exception as e:
        return str(e)