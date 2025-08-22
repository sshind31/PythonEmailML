import pyautogui
import subprocess
import time

# Step 1: Launch the app (replace with actual path)
exe_path=r"C:\Users\S950325\AppData\Local\Timetracker\app-6.0.0.1\Timetracker.Client.Win.exe"


try:
    subprocess.Popen(exe_path)
    print("App launched!")
except Exception as e:
    print("Failed to launch app:", e)

# Wait for the app to load
time.sleep(1)

# Step 2: Move and click (coordinates will vary — get via pyautogui.position())
pyautogui.click(x=1640, y=1180)  # Example coordinates for the Start button
time.sleep(1)
pyautogui.click(x=1550, y=1078)
time.sleep(4)
pyautogui.click(x=1862, y=935)

#1640 1180
#1550 1078
#1852 9935

