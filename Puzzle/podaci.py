import tensorflow as tf
import pickle
import cv2
import os
import random
import numpy as np

def iseci_sliku(slika,visina,sirina):
    slike=[]
    for i in range(int(len(slika)/visina)):
        for j in range(int(len(slika[0])/sirina)):
            crop_img = slika[(i)*visina:(i)*visina + visina, (j)*sirina:(j)*sirina + sirina]
            slike.append(crop_img)
    return slike

def broj_linija_na_slici(slika):
    gray = cv2.cvtColor(slika, cv2.COLOR_BGR2GRAY)
    lines1 = []

    edges1 = cv2.Canny(gray, 50, 150, apertureSize=3)
    try:
        lines1= cv2.HoughLinesP(edges1,1,np.pi/180,100)
        len(lines1)
    except:
        lines1 = []
    return len(lines1)

def procenat(vrednost,brojPiksela):
    if int(vrednost/brojPiksela*100)==0:
        return 1
    return int(vrednost/brojPiksela*100)


def razne_tacke(slika):
    #BGR
    min=150
    BLUE_MIN = np.array([min, 0, 0], np.uint8)
    BLUE_MAX = np.array([255, 200, 200], np.uint8)
    YELLOW_MIN = np.array([0, min,min], np.uint8)
    YELLOW_MAX = np.array([200, 255, 255], np.uint8)
    GREEN_MIN = np.array([0, min, 0], np.uint8)
    GREEN_MAX = np.array([200, 255, 200], np.uint8)
    RED_MIN = np.array([0, 0, min], np.uint8)
    RED_MAX = np.array([200, 200, 255], np.uint8)
    L_MIN = np.array([200, 200, 200], np.uint8)
    L_MAX = np.array([255, 255, 255], np.uint8)
    D_MIN = np.array([0, 0, 0], np.uint8)
    D_MAX = np.array([50, 50, 50], np.uint8)
    MID_MIN = np.array([70, 70, 70], np.uint8)
    MID_MAX = np.array([170, 170, 170], np.uint8)
    plava_slika = cv2.inRange(slika, BLUE_MIN, BLUE_MAX)
    zuta_slika=cv2.inRange(slika, YELLOW_MIN, YELLOW_MAX)
    #cv2.imshow("zuta_slika",zuta_slika)
    #cv2.waitKey(0)
    zelena_slika = cv2.inRange(slika, GREEN_MIN, GREEN_MAX)
    crvena_slika = cv2.inRange(slika, RED_MIN, RED_MAX)
    svetla_slika = cv2.inRange(slika, L_MIN, L_MAX)
    tamna_slika = cv2.inRange(slika, D_MIN, D_MAX)
    srednja_slika = cv2.inRange(slika, MID_MIN, MID_MAX)

    broj_plavih = cv2.countNonZero(plava_slika)
    broj_zutih=cv2.countNonZero(zuta_slika)
    broj_crvenih = cv2.countNonZero(zelena_slika)
    broj_zelenih = cv2.countNonZero(crvena_slika)
    broj_svetlih = cv2.countNonZero(svetla_slika)
    broj_tamnih = cv2.countNonZero(tamna_slika)
    broj_srednjih= cv2.countNonZero(srednja_slika)
    return (broj_plavih,broj_zutih,broj_crvenih,broj_zelenih,broj_svetlih,broj_tamnih,broj_srednjih)


def jedan_vektor(slika):
    ret_vektor=[]
    siva = cv2.cvtColor(slika, cv2.COLOR_RGB2GRAY)
    siva = cv2.resize(siva, (50, 50))
    #cv2.imshow("slika",slika)
    #cv2.waitKey(0)
    velicina_prozora=2
    gornji_pikseli=siva[0:velicina_prozora,0:50]
    desni_pikseli = siva[0:50, 50-velicina_prozora:50]
    #desni ide od gore na dole a trebaju ici u redu i onda je to transponovanje
    desni_pikseli=list(map(list, zip(*desni_pikseli)))
    dole_pikseli = siva[50-velicina_prozora:50, 0:50]
    levo_pikseli = siva[0:50, 0:velicina_prozora]
    levo_pikseli = list(map(list, zip(*levo_pikseli)))
    for i in range(len(gornji_pikseli)):
        ret_vektor+=list(gornji_pikseli[i])+desni_pikseli[i]+list(dole_pikseli[i])+levo_pikseli[i]
    return ret_vektor

def napravi_vektor(slike):
    #BGR
    povratni_vektor=[]
    for slika in slike:
        povratni_vektor.append(jedan_vektor(slika))
    return povratni_vektor
def spajanje_vektora(vektor_jedan,vektor_dva):
    spojeni_vektori = []
    for k in range(len(vektor_jedan)):
        spojeni_vektori += [vektor_jedan[k]] + [vektor_dva[k]]
    #print(spojeni_vektori)
    return spojeni_vektori


def resenje_vektora(i,j):
    ix=i%2
    iy=i//2
    jx=j%2
    jy =j//2
    if iy==jy:
        if ix+1==jx:#i sa leve strane od j
            #print(ix,iy,jx,jy)
            return [0,0,0,1,0]#gore,desno,dole,levo

        if ix-1==jx:
            #print(ix, iy, jx, jy)
            return [0, 1, 0, 0,0]
    if ix==jx:
        if iy+1==jy:#i sa gornje strane od j
            return [1,0,0,0,0]#gore,desno,dole,levo
        if iy-1==jy:
            return [0, 0, 1, 0,0]
    return [0,0,0,0,1]#nisu susedni



    povratni_vektor=[]
    return povratni_vektor


def krajnji_vektor(vektori):
    povratni_vektor=[]
    drugi_povratini_vektor=[]
    test_vektor = []
    test_vektor_y = []
    broj_nedgovarajucih=0
    for i in range(len(vektori)):
        for j in range(len(vektori)):
            rv=resenje_vektora(i,j)
            if rv[4]==0:
                povratni_vektor.append(spajanje_vektora(vektori[i], vektori[j]))
                drugi_povratini_vektor.append(resenje_vektora(i, j))
                test_vektor.append(spajanje_vektora(vektori[i],vektori[j]))
                test_vektor_y.append(rv)
            else:
                if broj_nedgovarajucih < 2:
                    povratni_vektor.append(spajanje_vektora(vektori[i], vektori[j]))
                    drugi_povratini_vektor.append(resenje_vektora(i, j))
                    broj_nedgovarajucih += 1

    return  [povratni_vektor,drugi_povratini_vektor,test_vektor,test_vektor_y]

def vrati_vektore_od_slike(jpg):
    try:
        krajnji_vekotr=[]
        slika=cv2.imread(jpg)
        slika = cv2.resize(slika, (400, 400))
        visina=len(slika)
        sirina=len(slika[0])
        i = 1
        velicine = [200]

        #if visina>1200:
            #slika=slika[0:1200,0:1200]
        print(str(visina)+","+str(sirina),jpg)
        slike=[]
        for velicina in velicine:
            slike+=iseci_sliku(slika,i*velicina,i*velicina)
        vektori=napravi_vektor(slike)
        krajnji_vekotr=krajnji_vektor(vektori)
        print(len(krajnji_vekotr[0]))
        return krajnji_vekotr
    except:
        return []


if __name__ == '__main__':
    slike_putanja=os.listdir()
    samo_slike=[]
    print(slike_putanja)
    for putanja in slike_putanja:
        try:
            if putanja.split('.')[1]=="jpg":
                samo_slike.append(putanja)
            elif putanja.split('.')[1]=="jpeg":
                samo_slike.append(putanja)
        except:
            pass
    print(samo_slike)
    krajnji_vekotr=[[],[],[],[]]
    for slika in samo_slike:
        nova_lista=vrati_vektore_od_slike(slika)
        if len(nova_lista)>0:
            print("nova_lista:"+str(len(nova_lista[0])))
            for i in range(len(krajnji_vekotr)):
                krajnji_vekotr[i]=krajnji_vekotr[i]+nova_lista[i]
            print("ukupno:"+str(len(krajnji_vekotr[0])))
        if len(krajnji_vekotr[0])>500000:
            break
    print(len(krajnji_vekotr[0]))
    with open('slike_treniranje.pickle','wb') as f:
        pickle.dump(krajnji_vekotr,f)

