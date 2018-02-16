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
        self.desniOkvir = Frame(root, width=550, height=550, highlightbackground="red", highlightcolor="red",
                           highlightthickness=1, bd=0)
        self.desniOkvir.pack(side=RIGHT, pady=5, padx=5)

        self.leviOkvir = Frame(root, width=250, height=350)
        self.leviOkvir.pack(side=LEFT)

        self.slikaDugme = Button(self.leviOkvir, text="Odaberite sliku", width=15, height=3, relief="groove", command=self.dodaj_sliku)
        self.slikaDugme.pack(side=TOP, pady=5, padx=10)

        self.sloziDugme = Button(self.leviOkvir, text="Slazite slagalicu", width=15, height=3, relief="groove", state="disabled",
                                 command=self.zamena_mesta)
        self.sloziDugme.pack(side=TOP, pady=5, padx=10)

        self.labela = Label(self.leviOkvir, text="x", borderwidth=2, relief="groove", width=35, height=17)
        self.labela.pack(side=BOTTOM, pady=5, padx=5)

        self.labelaSlagalica = Label(self.desniOkvir, text="", width=75, height=50)
        self.labelaSlagalica.pack(side=RIGHT)


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
            self.sloziDugme["state"] = "active"

    def prikazi_originalnu_sliku(self):

        slika = Image.open(novi_fajl)

        smanjena = slika.resize((250, 250), Image.ANTIALIAS)

        self.slika = ImageTk.PhotoImage(smanjena)

        self.labela.destroy()
        self.labela = Label(self.leviOkvir, image=self.slika, borderwidth=2, relief="groove", width=250, height=250)
        self.labela.image = self.slika
        self.labela.pack(side=BOTTOM, pady=5, padx=5)

    def prikazi_novu_slagalicu(self):

        slika = Image.open(ispremestana_slagalica)

        smanjena = slika.resize((500, 500), Image.ANTIALIAS)

        self.slika1 = ImageTk.PhotoImage(smanjena)

        self.labelaSlagalica.destroy()
        self.labelaSlagalica = Label(self.desniOkvir, image=self.slika1, width=540, height=550)
        self.labelaSlagalica.image = self.slika1
        self.labelaSlagalica.pack()

    def iseci_sliku(self):

        slika = cv2.imread(novi_fajl)

        smanjena_slika = cv2.resize(slika, (500, 500))
        puzla = slika[0:100, 0:100]
        visina = len(puzla)
        puta = visina

        sve_slike = []

        for i in range(5):
            for j in range(5):
                nova_slika = smanjena_slika[i * puta:i * puta + puta, j * puta:j * puta + puta]
                sve_slike.append(nova_slika)

        vis = np.zeros((max(500, 500), 500, 3), np.float32)

        np.random.shuffle(sve_slike)

        vis1 = np.zeros((max(100, 100), 100, 3), np.float32)
        vis2 = np.zeros((max(100, 100), 100, 3), np.float32)
        vis3 = np.zeros((max(100, 100), 100, 3), np.float32)
        vis4 = np.zeros((max(100, 100), 100, 3), np.float32)
        vis5 = np.zeros((max(100, 100), 100, 3), np.float32)

        vis1 = np.concatenate((sve_slike[0], sve_slike[1], sve_slike[2],
                               sve_slike[3], sve_slike[4]), axis=0)

        vis2 = np.concatenate((sve_slike[5], sve_slike[6], sve_slike[7],
                               sve_slike[8], sve_slike[9]), axis=0)

        vis3 = np.concatenate((sve_slike[10], sve_slike[11], sve_slike[12],
                               sve_slike[13], sve_slike[14]), axis=0)

        vis4 = np.concatenate((sve_slike[15], sve_slike[16], sve_slike[17],
                               sve_slike[18], sve_slike[19]), axis=0)

        vis5 = np.concatenate((sve_slike[20], sve_slike[21], sve_slike[22],
                               sve_slike[23], sve_slike[24]), axis=0)

        vis = np.concatenate((vis1, vis2, vis3, vis4, vis5), axis=1)

        cv2.imwrite(ispremestana_slagalica, vis)

        slika = Image.open(ispremestana_slagalica)

        smanjena = slika.resize((500, 500), Image.ANTIALIAS)

        self.slika1 = ImageTk.PhotoImage(smanjena)

        self.labelaSlagalica.destroy()
        self.labelaSlagalica = Label(self.desniOkvir, image=self.slika1, width=540, height=550)
        self.labelaSlagalica.image = self.slika1
        self.labelaSlagalica.pack()

    def zamena_mesta(self):
        menjanje_mesta()
        self.prikazi_novu_slagalicu()


def menjanje_mesta():

    #koordinate_1, koordinate_2 - koordinate koje cemo dobaviti

    koordinate_1 = [0, 0]

    koordinate_2 = [0, 400]

    slika = cv2.imread(ispremestana_slagalica)

    kopija = slika.copy()


    x1 = koordinate_1[0]
    y1 = koordinate_1[1]

    x2 = koordinate_2[0]
    y2 = koordinate_2[1]

    puzla1 = slika[y1:y1+100,x1:x1+100]

    puzla2 = slika[y2:y2+100,x2:x2+100]

    kopija[y1:y1+100,x1:x1+100] = puzla2

    kopija[y2:y2+100,x2:x2+100] = puzla1

    cv2.imwrite(ispremestana_slagalica, kopija)





if __name__ == '__main__':

    root.title("Resavanje slagalice")
    root.geometry('800x550+400+100')

    root.minsize(800,550)

    app = Aplikacija()

    root.mainloop()