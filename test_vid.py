import subprocess

res = subprocess.run(
    ["ffmpeg", "-buildconf"],
    capture_output=True,
    text=True
)
print("Return code:", res.returncode)
print(res.stdout)
