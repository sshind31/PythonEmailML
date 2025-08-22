import pyautogui
import time
import psutil

# Replace with the actual name of your EXE
process_name = "Timetracker.Client.Win.exe"

# Check if any process matches the name
running = any(proc.name() == process_name for proc in psutil.process_iter(['name']))

if running:
    print(f"{process_name} is running.")
else:
    print(f"{process_name} is NOT running.")

print("Move your mouse to the desired spot. You'll get the position in 5 seconds...")
time.sleep(5)
print("Position:", pyautogui.position())


