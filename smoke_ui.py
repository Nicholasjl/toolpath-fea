"""Exercise Tk rendering and the asynchronous solve without leaving a window."""
import time
import tkinter as tk
from app import App

root=tk.Tk()
root.withdraw()
app=App(root)
app.run()
deadline=time.monotonic()+30
while app.busy and time.monotonic()<deadline:
    root.update()
    time.sleep(.02)
assert app.result is not None, 'GUI solve did not finish'
for kind in ('位移幅值 / mm','等效应力 / MPa','沿线应力 σ₁ / MPa','走线角度 / °'):
    app.plot_type.set(kind)
    app.draw()
app.plot_type.set('位移幅值 / mm')
app.draw()
app.figure.savefig('example_result.png',dpi=150)
print(app.status.get())
root.destroy()
