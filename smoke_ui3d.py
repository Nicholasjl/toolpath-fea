import time
import tkinter as tk
from app3d import SolidApp

root=tk.Tk();root.withdraw();app=SolidApp(root)
app.start(True);deadline=time.monotonic()+60
while app.busy and time.monotonic()<deadline:
    root.update();time.sleep(.02)
assert app.result is not None,'GUI solve failed'
for name in ['位移 / mm','等效应力 / MPa','层间正应力 / MPa','路径距离 / mm']:
    app.field.set(name);app.draw();root.update()
app.cut.set(.5);app.draw();root.update()
app.cut.set(1);app.field.set('位移 / mm');app.deform.set(True);app.draw();root.update()
app.figure.savefig('example_solid.png',dpi=150)
print(app.status.get());root.destroy()
