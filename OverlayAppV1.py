import tkinter as tk

root = tk.Tk()
root.title("Overlay")
root.geometry("300x100+100+100")  # width x height + x + y

# Remove window border
root.overrideredirect(True)

# Always on top
root.wm_attributes("-topmost", True)

# Semi-transparent
root.wm_attributes("-alpha", 0.7)

label = tk.Label(root, text="🔔 Overlay Example", font=("Arial", 16), bg="yellow")
label.pack(expand=True, fill="both")


# Button to close overlay
btn = tk.Button(root, text="❌ Close", command=root.destroy)
btn.pack()

root.mainloop()
