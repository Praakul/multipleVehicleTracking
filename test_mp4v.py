import subprocess
try:
    res = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"],
        capture_output=True,
        text=True,
        check=True
    )
    encoders = [line for line in res.stdout.splitlines() if "V" in line[:6] and "h264" in line.lower()]
    print("Found H.264 encoders:")
    for enc in encoders:
        print(enc.strip())
except Exception as e:
    print("Error:", e)
