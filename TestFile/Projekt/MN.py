import math
def betrag(vektor):
    sum=0
    for i in range(len(vektor)):
        sum=sum+vektor[i]*vektor[i]
    return math.sqrt(sum)
'''
def summe(vektorA, vektorB):
    vektorC=[]
    for i in range(len(vektorA)):
        vektorC.append(0)
        vektorC[i]=vektorA[i]+vektorB[i]
    return vektorC

def matrixMalVektor(matrix,vektor):
    ergebnis=[]
    for i in range(len(matrix)):
        if len (matrix[i]) is not len(vektor):
            #print("Dimensionsfehler")
            ergebnis=["Dimensionsfehler"]
            break
        else:
            vektorKomponente=0
            for j in range(len(matrix[i])):
                vektorKomponente+=(matrix[i][j]*vektor[j])
        ergebnis.append(vektorKomponente)
    print(ergebnis)
    return ergebnis

def matrixMalMatrix(matrixA,matrixB):
    ergebnis=[]
    for i in range(len(matrixA)):
        ergebnis.append([])
        for l in range(len(matrixB[0])):
            if len (matrixA[i]) is not len(matrixB):
                #print("Dimensionsfehler")
                ergebnis=["Dimensionsfehler"]
                break
            else:
                matrixElement=0
                for j in range(len(matrixA[i])):
                    matrixElement+=(matrixA[i][j]*matrixB[j][l])
            ergebnis[i].append(matrixElement)
        print(ergebnis[i])
    return ergebnis
'''    
def differenz(vektorA, vektorB):
    vektorC=[]
    for i in range(len(vektorA)):
        vektorC.append(0)
        vektorC[i]=vektorA[i]-vektorB[i]
    return betrag(vektorC)

def duplikateloeschen(liste):
    o=[]
    for e in liste:
        if e not in o:
            o.append(e)
    return o    
    
#Erstellt eine Matrix mit Listen als Elemente
def uebersicht(measurements,Startzeit,N):
    uebersichtMatrix=[]
    for i in range(N):
        uebersichtMatrix.append([])
        for j in range(len(measurements[Startzeit-1])):
            uebersichtMatrix[i].append([])
    return uebersichtMatrix
#Prueft ob Werte in kandidaten schon mal in uebersichtMatrix zur Zeit i aufgetaucht sind,
#wenn nicht dann wird die Werte in uebersichtMatrixgespeichert.    
def checkAndAdd(uebersichtMatrix,kandidaten,i,j):
    temp=kandidaten.copy()
    for a in temp:
        for b in range(len(uebersichtMatrix[i])):
            if a in uebersichtMatrix[i][b]:
                kandidaten.remove(a)
                break
                
    uebersichtMatrix[i][j].extend(kandidaten)
    return uebersichtMatrix

def mnLogic(M,N,Startzeit,measurements):                            #(Anzahl der benoetigten Detektionen, Anzahl der Scans, Startzeitpunkt, Messdaten)  
    n=0                                                             #Anzahl der geschaetzten Objekte
    uebersichtMatrix=uebersicht(measurements,Startzeit,N)
    
    if Startzeit+N-1>(len(measurements)) or Startzeit<1:            #Ueberpruefung ob genug Messungen fuer den gewaehlten Startpunkt da sind
        return "Keine weiteren Messungen vorhanden"
    else:    
        for j in range (len(measurements[Startzeit-1])):            #Duch die erste Zeile von measurements iterieren
            kandidaten=[measurements[Startzeit-1][j]]               #moegliches Objekt Speichern
            m=1                                                     #erste Messung zaehlt als erste Detektion                          
            m_bar=0                                                 #Anzahl der Fehldetektionen
            s=1                                                     #Aktueller Scan
            #############################################################Durch die Zeilen/Zeit von measurements iterieren
            for i in range(Startzeit-1,Startzeit-1+N-1):
                z=checkAndAdd(uebersichtMatrix,kandidaten,i-Startzeit+1,j)
                alteAnzahl=len(kandidaten)             
                s+=1
                #########################################################wird ausgefuehrt wenn man sich am Startzeitpunkt befindet
                if i==Startzeit-1:                          #
                    for k in measurements[i+1]:             #       fuer alle measurements in der i+1-ten Zeile Messpunkte suchen,
                        if differenz(kandidaten[0],k)<=1:         #       die die Bedingungen als Kandidat erfuellen 
                            kandidaten.append(k)            #       und zur Kandidatenliste hinzufuegen  
                                                            #
                    if len(kandidaten)>alteAnzahl:          #       wenn neue Kandidaten im Naechsten Zeitschritt gefunden wurden,            
                        del kandidaten[0:alteAnzahl]        #       loesche die Kandidaten aus dem Vorherigen Zeitschritt,
                        m+=1                                #       erhoehe die Anzahl der erfolgreichen Detektionen um 1
                    else:                                   #
                        m_bar+=1                            #       keine neue Kandidaten-> Anzahl der Fehldetektionen erhoehen
                #############################################
                #########################################################wird ausgefuehrt wenn man sich NICHT am Startzeitpunkt befindet
                else:                                           #   
                    newKandidaten=kandidaten.copy()             #       Kandidatenzwischenspeicher
                    for l in kandidaten:                        #       fuer alle Kandidaten in der Kandidatenliste pruefen
                        if l not in measurements[i]:            #       ob Kandidat nicht in der aktuellen i-ten Messung vorkommt,
                            for k in measurements[i+1]:         #       wenn Kandidat nicht in der aktuellen i-ten Messung steht,
                                if differenz(l,k)<=2:                 #       andere Bedingung fuer Kandidatenzugehoerigkeit pruefen
                                    newKandidaten.append(k)     #       und zu Kandidatenzwischenspeicher hinzufuegen
                                                                #
                        else:                                   #
                            for k in measurements[i+1]:         #       fuer alle measurements in der i+1-ten Zeile Messpunkte suchen,
                                if differenz(l,k)<=1:                 #       die die Bedingungen als Kandidat erfuellen 
                                    newKandidaten.append(k)     #       und zur Kandidatenliste hinzufuegen  
                                                                #
                                                                #
                                                                #    
                    if len(newKandidaten)>alteAnzahl:           #       wenn neue Kandidaten im Naechsten Zeitschritt gefunden wurden,        
                        del newKandidaten[0:alteAnzahl]         #       loesche die Kandidaten aus dem vorherigen Zeitschritt,
                        m+=1                                    #       erhoehe die Anzahl der erfolgreichen Detektionen um 1
                    else:                                       #
                        m_bar+=1                                #       keine neue Kandidaten-> Anzahl der Fehldetektionen erhoehen
                                                                #
                    kandidaten=duplikateloeschen(newKandidaten) #       Kandidatenliste aktualisieren 
                                                                #
                #################################################

                #########################################################Abbruchbedingungen    
                if m_bar>(N-M):                             #       wenn die zuviele Fehldetektionen
                    kandidaten=[]
                    break                                   #       Abbruch und neue Iteration bei measurements[0][j+1]
            if m>=M:                                        #       wenn die benoetigte Anzahl an Detektionen erreicht,
                n+=1                                        #       erhoehe die Anzahl der geschaetzten Objekte
                #############################################
            z[N-1][j]=kandidaten    
    for x in range(len(z)):
        print(z[x])
    list2 = [x for x in z[N-1] if x != []]
    return n,list2

#testecke
   
measurements=[[[1,1],[2,2],[3,3],[0,0]],
            [[1.5,1.5],[2.5,2.5],[6,6],[5,5]],
            [[2,2],[3,3],[0,0],[1,1]],
            [[0,0],[8,8],[3,3]],
            [[2,2],[3.5,3.5]]]

print(mnLogic(4,5,1,measurements))

