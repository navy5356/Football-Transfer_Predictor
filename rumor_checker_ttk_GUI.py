# -*- coding: utf-8 -*-
from tkinter import *
from tkinter import ttk
from PIL import ImageTk,Image  
import ftr_predict_ttk as ftr_pred

def getRumour(*args):
    try:
        retval = 'Not Sure'
        value = rumour_text.get()
        retval = ftr_pred.detecting_rumour_validity(model, value)
        prediction.set(retval)

    except ValueError:
        pass

root = Tk()
root.option_add('*tearOff', FALSE)
root.title("Football Transfer Rumour Checker")

mainframe = ttk.Frame(root, padding="3 3 12 12")
mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)

rumour_text = StringVar()
prediction = StringVar()
result_label = StringVar()
result_label = ' '

logo = ImageTk.PhotoImage(file='Messi_small.gif')

ttk.Label(mainframe, image=logo).grid(column=1, row=1, sticky=E)
rumour_entry = ttk.Entry(mainframe, width=50, textvariable=rumour_text)
rumour_entry.grid(column=3, row=1, sticky=(W, E))

ttk.Label(mainframe, textvariable=prediction).grid(column=3, row=2,sticky=(W, E))
ttk.Button(mainframe, text="Check Rumour", command=getRumour, default='active').grid(column=3, row=3, sticky=W)

ttk.Label(mainframe, text="Rumour").grid(column=2, row=1, sticky=E)
ttk.Label(mainframe, text='is Likely to be').grid(column=2, row=2, sticky=W)

for child in mainframe.winfo_children(): child.grid_configure(padx=5, pady=5)
rumour_entry.focus()
root.bind('<Return>', getRumour)

model = ftr_pred.load_ftr_model()
root.mainloop()


