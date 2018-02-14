from tkinter import *
from tkinter import filedialog
from PIL import Image,ImageTk
import cv2
import numpy as np

novi_fajl = "slika/slagalica.jpg"

ispremestana_slagalica = "slika/pocetak.jpg"

root = Tk()

class Aplikacija:

    def __init__(self):
        self.desniOkvir = Frame(root, width=400, height=300, highlightbackground="red", highlightcolor="red",
                           highlightthickness=1, bd=0)
        self.desniOkvir.pack(side=RIGHT, pady=5, padx=5)

        self.leviOkvir = Frame(root, width=200, height=300)
        self.leviOkvir.pack(side=LEFT)

        self.slikaDugme = Button(self.leviOkvir, text="Odaberite sliku", width=15, height=3, relief="groove", command=self.dodaj_sliku)
        self.slikaDugme.pack(side=TOP, pady=5, padx=10)

        self.sloziDugme = Button(self.leviOkvir, text="Slozite slagalicu", width=15, height=3, relief="groove")
        self.sloziDugme.pack(side=TOP, pady=5, padx=10)

        self.labela = Label(self.leviOkvir, text="x", borderwidth=2, relief="groove", width=200, height=200)
        self.labela.pack(side=BOTTOM, pady=5, padx=5)

        self.labelaSlagalica = Label(self.desniOkvir, text="", width=50,height=50)
        self.labelaSlagalica.pack()


    def dodaj_sliku(self):
        fajl = filedialog.askopenfile(parent=root, mode='rb', title='Izaberite sliku',
                                      filetypes=[('Image files', '*.png *.jpg *.gif *.bmp')])
        if fajl != None:
            data = fajl.read()
            fajl.close()
            with open(novi_fajl, 'wb') as fajl_upis:
                fajl_upis.write(data)

            self.prikazi_originalnu_sliku()
            self.iseci_sliku()

    def prikazi_originalnu_sliku(self):

        slika = Image.open(novi_fajl)

        smanjena = slika.resize((200, 200), Image.ANTIALIAS)

        self.slika = ImageTk.PhotoImage(smanjena)

        self.labela.destroy()
        self.labela = Label(self.leviOkvir, image=self.slika, borderwidth=2, relief="groove", width=200, height=200)
        self.labela.image = self.slika
        self.labela.pack(side=BOTTOM, pady=5, padx=5)


if __name__ == '__main__':

    root.title("Resavanje slagalice")
    root.geometry('600x300+400+100')

    root.minsize(600,300)

    app = Aplikacija()

    root.mainloop()