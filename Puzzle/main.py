from tkinter import *
from tkinter import filedialog
from PIL import Image,ImageTk
import cv2
import numpy as np
import tensorflow as tf
from podaci import *

novi_fajl = "slika/slagalica.jpg"

ispremestana_slagalica = "slika/pocetak.jpg"

razlika_fajl = "slika/razlika.jpg"

root = Tk()
with open('slike_treniranje.pickle', 'rb') as f:
    x = pickle.load(f)
train_x, train_y, test_x, test_y = x
n_nodes_hl1 = 800
n_nodes_hl2 = 800
n_nodes_hl3 = 800
n_classes = 5
x = tf.placeholder('float', [None, len(train_x[0])])
y = tf.placeholder('float')
v1 = tf.get_variable("v1", shape=[3])
saver = tf.train.Saver()

def neural_network_model(data):
    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([len(train_x[0]), n_nodes_hl1])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    'biases': tf.Variable(tf.random_normal([n_classes]))}

    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

    return output
def test(vektor):
   pass
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
                                      filetypes=[('Image files', '*.jpeg *.png *.jpg *.gif *.bmp')])
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

    def prikazi_novu_slagalicu(self,slika):

        b, g, r = cv2.split(slika)
        s = cv2.merge((r, g, b))
        s=Image.fromarray(s)
        self.slika1 = ImageTk.PhotoImage(s)

        self.labelaSlagalica.destroy()
        self.labelaSlagalica = Label(self.desniOkvir, image=self.slika1, width=540, height=550)
        self.labelaSlagalica.pack()

    def iseci_sliku(self):

        slika = cv2.imread(novi_fajl)

        smanjena_slika = cv2.resize(slika, (400, 400))
        puzla = slika[0:200, 0:200]
        visina = len(puzla)
        puta = visina

        sve_slike = []

        for i in range(2):
            for j in range(2):
                nova_slika = smanjena_slika[i * puta:i * puta + puta, j * puta:j * puta + puta]
                sve_slike.append(nova_slika)

        vis = np.zeros((max(400, 400), 400, 3), np.float32)

        np.random.shuffle(sve_slike)

        vis1 = np.zeros((max(200, 200), 200, 3), np.float32)
        vis2 = np.zeros((max(200, 200), 200, 3), np.float32)


        vis1 = np.concatenate((sve_slike[0], sve_slike[1]), axis=0)

        vis2 = np.concatenate((sve_slike[2], sve_slike[3]), axis=0)



        vis = np.concatenate((vis1, vis2), axis=1)

        cv2.imwrite(ispremestana_slagalica, vis)

        slika = Image.open(ispremestana_slagalica)

        smanjena = slika.resize((400, 400), Image.ANTIALIAS)

        self.slika1 = ImageTk.PhotoImage(smanjena)

        self.labelaSlagalica.destroy()
        self.labelaSlagalica = Label(self.desniOkvir, image=self.slika1, width=540, height=550)
        self.labelaSlagalica.image = self.slika1
        self.labelaSlagalica.pack()

    def zamena_mesta(self):
        self.promeni_mesto()

    def promeni_mesto(self):

        rez = [1]
        puta = 200
        prediction = neural_network_model(x)

        slika = cv2.imread(ispremestana_slagalica)

        slika_original = cv2.imread(novi_fajl)

        slika2 = cv2.resize(slika_original, (400, 400))

        if slike_iste(slika, slika2) is True:
            self.sloziDugme["state"] = "disabled"

        else:
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                # Restore variables from disk.
                saver.restore(sess, "/tmp/model.ckpt")
                # print("Model restored.")

                kopija = slika.copy()

                slika_gore_levo = kopija[0 * puta:0 * puta + puta, 0 * puta:0 * puta + puta]
                slika_dole_levo = kopija[1 * puta:1 * puta + puta, 0 * puta:0 * puta + puta]
                slika_gore_desno = kopija[0 * puta:0 * puta + puta, 1 * puta:1 * puta + puta]
                slika_dole_desno = kopija[1 * puta:1 * puta + puta, 1 * puta:1 * puta + puta]


                vektor1 = jedan_vektor(slika_gore_levo)
                vektor2 = jedan_vektor(slika_dole_levo)
                vektor3 = jedan_vektor(slika_gore_desno)
                vektor4 = jedan_vektor(slika_dole_desno)

                vek12 = spajanje_vektora(vektor1, vektor2)

                vek13 = spajanje_vektora(vektor1, vektor3)

                vek21 = spajanje_vektora(vektor2, vektor1)

                vek34 = spajanje_vektora(vektor3, vektor4)

                rez21 = (sess.run(tf.argmax(prediction.eval(feed_dict={x: [vek21]}), 1)))

                indeks = (sess.run(tf.argmax(prediction.eval(feed_dict={x: [vek12]}), 1)))

                for i in range(100):
                    if (indeks[0] == 4 and rez21[0] ==4):
                        #print("dijagonalno dole lijeva ide dole desno")
                        slika[1 * puta:1 * puta + puta, 0 * puta:0 * puta + puta],slika[1 * puta:1 * puta + puta, 1 * puta:1 * puta + puta] = slika_dole_desno, slika_dole_levo
                        self.prikazi_novu_slagalicu(slika)
                        break

                    elif(indeks[0] == 3 and rez21[0]==1):
                        #print("sa desne strane dole lijeva ide gore desno")
                        slika[1 * puta:1 * puta + puta, 0 * puta:0 * puta + puta],slika[0 * puta:0 * puta + puta, 1 * puta:1 * puta + puta]= slika_gore_desno, slika_dole_levo
                        self.prikazi_novu_slagalicu(slika)
                        break

                    elif(indeks[0] == 0 and rez21[0]==2):
                        #print("sa donje dole levo ide gore levo")
                        slika[0 * puta:0 * puta + puta, 0 * puta:0 * puta + puta],slika[1 * puta:1 * puta + puta, 0 * puta:0 * puta + puta] = slika_dole_levo,slika_gore_levo
                        self.prikazi_novu_slagalicu(slika)
                        break

                    elif(indeks[0] == 1 and rez21[0]==3):
                        #print("sa leve gore levo ide dole desno")
                        slika[0 * puta:0 * puta + puta, 0 * puta:0 * puta + puta],slika[1 * puta: 1 * puta + puta, 1 * puta: 1 * puta + puta] = slika_dole_desno, slika_gore_levo
                        self.prikazi_novu_slagalicu(slika)
                        break


                for i in range(0, 1):
                    for j in range(0, 2):
                        nova_slika = slika[i * puta:i * puta + puta, j * puta:j * puta + puta]
                        vekt1 = jedan_vektor(nova_slika)
                        nova_slika1 = slika[(i + 1) * puta:(i + 1) * puta + puta, j * puta:j * puta + puta]
                        vekt2 = jedan_vektor(nova_slika1)

                        vektor_ponovo = spajanje_vektora(vekt1, vekt2)
                        vektor_ponovo2 = spajanje_vektora(vekt2, vekt1)

                        rezultat = (sess.run(tf.argmax(prediction.eval(feed_dict={x: [vektor_ponovo]}), 1)))
                        rezultat2 = (sess.run(tf.argmax(prediction.eval(feed_dict={x: [vektor_ponovo2]}), 1)))

                        if (rezultat[0] == 0):
                            #print("mijenja po vrstama")
                            koordinate1 = i * puta, i * puta + puta, j * puta, j * puta + puta
                            koordinate2 = (i + 1) * puta, (i + 1) * puta + puta, j * puta, j * puta + puta
                            slika = menjanje_mesta(slika, koordinate1, koordinate2)
                            self.prikazi_novu_slagalicu(slika)

                        nova_slika2 = slika[j * puta:j * puta + puta, i * puta:i * puta + puta]
                        vektor_ponovo3 = jedan_vektor(nova_slika2)
                        nova_slika3 = slika[j * puta:j * puta + puta, (i + 1) * puta:(i + 1) * puta + puta]
                        vektor_ponovo4 = jedan_vektor(nova_slika3)

                        vekt3 = spajanje_vektora(vektor_ponovo3, vektor_ponovo4)
                        vekt4 = spajanje_vektora(vektor_ponovo4, vektor_ponovo3)
                        rezultat3 = (sess.run(tf.argmax(prediction.eval(feed_dict={x: [vekt3]}), 1)))
                        rezultat4 = (sess.run(tf.argmax(prediction.eval(feed_dict={x: [vekt4]}), 1)))

                        if (rezultat3[0] == 3):
                            #print("mijenja po kolonama")
                            koordinate1 = j * puta, j * puta + puta, i * puta, i * puta + puta
                            koordinate2 = j * puta, j * puta + puta, (i + 1) * puta, (i + 1) * puta + puta
                            slika = menjanje_mesta(slika, koordinate1, koordinate2)
                            self.prikazi_novu_slagalicu(slika)



def slike_iste(slika1, slika2):

    a1=slika1[0:5,0:5]
    b1=slika2[0:5,0:5]
    a2=slika1[200:5,0:5]
    b2=slika2[200:5,0:5]
    a3=slika1[0:5,200:5]
    b3=slika2[0:5,200:5]


    if (np.array_equal(a1,b1) and np.array_equal(a2,b2) and np.array_equal(a3,b3)):
        return True
    else:
        return False

def menjanje_mesta(slika,koordinate_1, koordinate_2):

    kopija = slika.copy()

    ax1 = koordinate_1[0]
    ax2 = koordinate_1[1]
    ay1 = koordinate_1[2]
    ay2 = koordinate_1[3]

    bx1 = koordinate_2[0]
    bx2 = koordinate_2[1]
    by1 = koordinate_2[2]
    by2 = koordinate_2[3]

    puzla1 = slika[ax1:ax2, ay1:ay2]

    puzla2 = slika[bx1:bx2, by1:by2]

    kopija[ax1:ax2, ay1:ay2] = puzla2

    kopija[bx1:bx2, by1:by2] = puzla1

    return kopija




if __name__ == '__main__':

    root.title("Resavanje slagalice")
    root.geometry('800x550+400+100')

    root.minsize(800,550)

    app = Aplikacija()

    root.mainloop()