import math
import numpy

def duplikateloeschen(liste):
    o=[]
    for e in liste:
        if e not in o:
            o.append(e)
    return o    
def betrag(vektor):
    sum=0
    for i in range(len(vektor)):
        sum=sum+vektor[i]*vektor[i]
    return math.sqrt(sum)


def differenz(vektorA, vektorB):
    vektorC=[]
    for i in range(len(vektorA)):
        vektorC.append(0)
        vektorC[i]=vektorA[i]-vektorB[i]
    return betrag(vektorC)


#Normierung von 2D-Daten
def norm(data):
    data_x=[]
    data_y=[]
    min_x=0
    min_y=0
    max_x=640
    max_y=480
    normed_x=[]
    normed_y=[]
    normed_data=[]
    for i in range(len(data)):
        data_x_temp=[]
        data_y_temp=[]
        for j in range(len(data[i])):
            data_x_temp.append(data[i][j][0])
            data_y_temp.append(data[i][j][1])
        data_x+=[data_x_temp]
        data_y+=[data_y_temp]        
        
    for k in range(len(data)):
        data_x_temp=[]
        data_y_temp=[]
        for l in range(len(data[k])):
            data_x_temp.append((data_x[k][l]-min_x)/(max_x-min_x))
            data_y_temp.append((data_y[k][l]-min_y)/(max_y-min_y))
        normed_x+=[data_x_temp]
        normed_y+=[data_y_temp]
       
    for m in range(len(data)):
        data_points_temp=[]
        for n in range(len(data[m])):
            data_points_temp+=[[normed_x[m][n],normed_y[m][n]]]
        normed_data+=[data_points_temp]    
    
    return min_x, min_y, max_x, max_y, normed_data


#Normierung von 1D-Daten (für Horizont)
def horizonNorm(data):
    data_y=data
    min_y=0
    max_y=480
   
    normed_data=[]
        
    for k in range(len(data)):
        data_y_temp=[]
        for l in range(len(data[k])):
            data_y_temp.append((data_y[k][l]-min_y)/(max_y-min_y))
        normed_data+=[data_y_temp]
               
    return min_y, max_y, normed_data


#Erstellt eine Matrix mit Listen als Elemente
#Prueft ob Werte in kandidaten schon mal in uebersichtMatrix zur Zeit i aufgetaucht sind,
#wenn nicht dann wird die Werte in uebersichtMatrixgespeichert.
def uebersicht(measurements,N):
    uebersichtMatrix=[]
    for i in range(N):
        uebersichtMatrix.append([])
        for j in range(len(measurements[len(measurements)-1])):
            uebersichtMatrix[i].append([])
    return uebersichtMatrix
    

#M/N Logik für den Anfang###########################################################################################################################
def initMnLogic(M,N,measurements,velocities,T, est,treshhold,n_old,P,P_i_init):#(Anzahl der benoetigten Detektionen, Anzahl der Scans, Messdaten 2D, Geschwindigkeiten, Abtastzeit, est, treshhold position, n_old)  
    min_x, min_y, max_x, max_y, normed_data=norm(measurements)
    #normed_data=measurements
    n=0                                                             #Anzahl der geschätzten Objekte
    AnfangsWerteGNN=[]
    uebersichtMatrix=uebersicht(measurements,len(measurements))
    #Duch die erste Zeile von measurements iterieren
    for j in range (len(normed_data[len(normed_data)-1])):          
        kandidaten=[normed_data[len(normed_data)-1][j]]             #moegliches Objekt Speichern
        uebersichtMatrix[len(measurements)-1][j]=kandidaten
        m=1                                                         #erste Messung zaehlt als erste Detektion                          
        m_bar=0                                                     #Anzahl der Fehldetektionen
        #Durch die Zeilen/Zeit von measurements iterieren
        for i in range(len(measurements)-1,len(measurements)-N,-1):
            alteAnzahl=len(kandidaten)                                     
            newKandidaten=kandidaten.copy()             #           Kandidatenzwischenspeicher
            for l in kandidaten:                        #           fuer alle Kandidaten in der Kandidatenliste pruefen
                if l not in normed_data[i]:             #           ob Kandidat nicht in der aktuellen i-ten Messung vorkommt,
                    for k in normed_data[i-1]:          #           wenn Kandidat nicht in der aktuellen i-ten Messung steht,
                        list1 = [x for x in uebersichtMatrix[i-1] if x != []]
                        list2=[e for sl in list1 for e in sl]
                        if (abs(l[0]-k[0])<=2*treshhold)and(abs(l[1]-k[1])<=2*treshhold) and k not in list2:                
                            uebersichtMatrix[i-1][j].append(k)
                            newKandidaten.append(k)            
                else:                                   
                    for k in normed_data[i-1]:          
                        list1 = [x for x in uebersichtMatrix[i-1] if x != []]
                        list2=[e for sl in list1 for e in sl]
                        if (abs(l[0]-k[0])<=treshhold)and(abs(l[1]-k[1])<=treshhold) and k not in list2:                
                            uebersichtMatrix[i-1][j].append(k)
                            newKandidaten.append(k)              
            if len(newKandidaten)>alteAnzahl:           #       wenn neue Kandidaten im Naechsten Zeitschritt gefunden wurden,        
                del newKandidaten[0:alteAnzahl]         #       loesche die Kandidaten aus dem vorherigen Zeitschritt,
                m+=1                                    #       erhoehe die Anzahl der erfolgreichen Detektionen um 1
            else:                                       
                m_bar+=1                                #       keine neue Kandidaten-> Anzahl der Fehldetektionen erhoehen
            kandidaten=duplikateloeschen(newKandidaten) #       Kandidatenliste aktualisieren 
            #Abbruchbedingungen    
            if m_bar>(N-M):                             #       wenn zuviele Fehldetektionen
                #kandidaten=[]
                break                                   #       Abbruch und neue Iteration bei measurements[0][j+1]
        if m>=M:                                        #       wenn die benoetigte Anzahl an Detektionen erreicht,
            AnfangsWerteGNN.append([(max_x-min_x)*normed_data[len(measurements)-1][j][0]+min_x,(max_y-min_y)*normed_data[len(measurements)-1][j][1]+min_y])
            n+=1                                        #       erhoehe die Anzahl der geschaetzten Objekte
        kandidaten=[]   
    #for x in range(len(uebersichtMatrix)):
    #    print(uebersichtMatrix[x])
    #return n, AnfangsWerteGNN
    return deathsBirths(n,AnfangsWerteGNN,est,n_old,P,P_i_init)


#M/N Logik über Geschwindigkeiten###################################################################################################################
def veloMnLogic(M,N,measurements,velocities,T, est,treshhold,n_old,P,P_i_init):#(Anzahl der benoetigten Detektionen, Anzahl der Scans, Messdaten 2D , Geschwindigkeiten, Abtastzeit, est, treshhold position, n_old)  
    min_x, min_y, max_x, max_y, normed_data=norm(measurements)
    min_x, min_y, max_x, max_y, normed_velocities=norm(velocities)
    #normed_data=measurements
    n=0                                                             #Anzahl der geschaetzten Objekte
    AnfangsWerteGNN=[]
    uebersichtMatrix=uebersicht(measurements,len(measurements))
    trimmedVelocity=normed_velocities[len(normed_velocities)-4:]
    #Duch die erste Zeile von measurements iterieren
    for j in range (len(normed_data[len(normed_data)-1])):          
        kandidaten=[normed_data[len(normed_data)-1][j]]             #moegliches Objekt Speichern
        uebersichtMatrix[len(measurements)-1][j]=kandidaten
        m=1                                                         #erste Messung zaehlt als erste Detektion                          
        m_bar=0                                                     #Anzahl der Fehldetektionen
        #Durch die Zeilen/Zeit von measurements iterieren
        for i in range(len(measurements)-1,len(measurements)-N,-1):
            alteAnzahl=len(kandidaten)             
            newKandidaten=kandidaten.copy()             #           Kandidatenzwischenspeicher
            for l in kandidaten:                        #           fuer alle Kandidaten in der Kandidatenliste pruefen
                if l not in normed_data[i]:             #           ob Kandidat nicht in der aktuellen i-ten Messung vorkommt,
                    for k in normed_data[i-1]:          #           wenn Kandidat nicht in der aktuellen i-ten Messung steht,
                        for v in trimmedVelocity[i-1]:
                            for v1 in trimmedVelocity[i]:
                                list1 = [x for x in uebersichtMatrix[i-1] if x != []]
                                list2=[e for sl in list1 for e in sl]
                                if (abs(k[0]-l[0]+(v[0]+v1[0])*T)<=treshhold)and(abs(k[1]-l[1]+(v[1]+v1[1])*T)<=treshhold) and k not in list2:                
                                    uebersichtMatrix[i-1][j].append(k)
                                    newKandidaten.append(k)    
                else:                                   
                    for k in normed_data[i-1]:          
                        for v in trimmedVelocity[i-1]:
                            list1 = [x for x in uebersichtMatrix[i-1] if x != []]
                            list2=[e for sl in list1 for e in sl]
                            if (abs(k[0]-l[0]+v[0]*T)<=treshhold)and(abs(k[1]-l[1]+v[1]*T)<=treshhold) and k not in list2:                
                                uebersichtMatrix[i-1][j].append(k)
                                newKandidaten.append(k) 
            if len(newKandidaten)>alteAnzahl:           #       wenn neue Kandidaten im Naechsten Zeitschritt gefunden wurden,        
                del newKandidaten[0:alteAnzahl]         #       loesche die Kandidaten aus dem vorherigen Zeitschritt,
                m+=1                                    #       erhoehe die Anzahl der erfolgreichen Detektionen um 1
            else:                                       #
                m_bar+=1                                #       keine neue Kandidaten-> Anzahl der Fehldetektionen erhoehen
            kandidaten=duplikateloeschen(newKandidaten) #       Kandidatenliste aktualisieren 
            #Abbruchbedingungen    
            if m_bar>(N-M):                             #       wenn die zuviele Fehldetektionen
                #kandidaten=[]
                break                                   #       Abbruch und neue Iteration bei measurements[0][j+1]
        if m>=M:                                        #       wenn die benoetigte Anzahl an Detektionen erreicht,
            AnfangsWerteGNN.append([(max_x-min_x)*normed_data[len(measurements)-1][j][0]+min_x,(max_y-min_y)*normed_data[len(measurements)-1][j][1]+min_y])
            n+=1                                        #       erhoehe die Anzahl der geschaetzten Objekte
        kandidaten=[]   
    #for x in range(len(uebersichtMatrix)):
    #    print(uebersichtMatrix[x])
    #return n, AnfangsWerteGNN
    return deathsBirths(n,AnfangsWerteGNN,est,n_old,P,P_i_init)


#M/N Logik für Horizont###########################################################################################################################
def horizonMnLogic(measurements):#(Messdaten)  
    trimmedMeasurements=measurements[len(measurements)-4:]
    temp_path=[]#Zwischenspeicher erstellen
    path=[]#Pfadspeicher
    for a in trimmedMeasurements[0]:
        for b in trimmedMeasurements[1]:
            for c in trimmedMeasurements[2]:
                for d in trimmedMeasurements[3]:
                    temp_path.extend([a,b,c,d])#Pfad erstellen
                    if len(path)==0 or numpy.var(temp_path)<numpy.var(path):#Varianzen vergleichen 
                        path=temp_path.copy()#Pfad speichern mit kleinster Varianz
                    temp_path=[]#Zwischenspeicher leeren
    return path[3]

def deathsBirths(n_new,anfangsWerte,est,n_old,P,P_i_init):
    
    H= numpy.array([[1,0,0,0],[0,0,1,0]]) #Ausgangsmatrix
    H_velocity = numpy.array([[0,1,0,0],[0,0,0,1]])
    number_states = 4
    
    
    try: #Erfolg wenn n nicht zwei mal in Folge gleich null ist
        vel_old = numpy.matmul(H_velocity,est) #Geschwindigkeit alt
        pos_old = numpy.matmul(H,est) #Positionen alt (Eingangs des MN)
        pos_new = numpy.transpose(numpy.array(anfangsWerte))
        est_updated = numpy.zeros((4,n_new))
        
        
    #Initialisierung 
        if n_old < 1:
            est_updated[0,:] = pos_new[0,:]
            est_updated[2,:] = pos_new[1,:]
            P = numpy.zeros((number_states,n_new*number_states))
           
            for i in range(n_new):
                P[0:number_states,i*number_states:number_states*(i+1)] = P_i_init #Kovarianzmatrix des Schätzfehlers
                
                
        
        elif n_new > n_old: #Births: Die Koordinaten die die minimale Abstände von den alten Objekten aufweisen, werden als "schon vorhandetes Objekt" betrachtet und daher werden als neues Objekt ausgeschlossen
            
            vel_new = numpy.zeros((n_new-n_old)) #Geschwinigkeit neu. Erstmal als 0 annehmen. 
            P_new = numpy.zeros((number_states,(n_new-n_old)*number_states)) #Initialisierung der neuen Varianz: Bei Geburt werden die neulich erzeugten Objete der Varianz = P_init zugewiesen
            for i in range(n_new-n_old):
                P_new[0:number_states,i*number_states:number_states*(i+1)] = P_i_init #Kovarianzmatrix des Schätzfehlers
            for i in range(n_old):
                distances = [] #Liste mit Abständen zu den alten koordinaten
                for j in range(pos_new.shape[1]):
                    distances.append(differenz(pos_old[:,i],pos_new[:,j])) #Vektorbetrag hinzufügen
                index_min = min(range(len(distances)), key=distances.__getitem__) #minimalen index ausrechnen
                pos_new = numpy.delete(pos_new,index_min,1)
            est_updated[0,:] = numpy.concatenate((pos_old[0],pos_new[0]),0)
            est_updated[1,:] = numpy.concatenate((vel_old[0],vel_new),0)
            est_updated[2,:] = numpy.concatenate((pos_old[1],pos_new[1]),0)
            est_updated[3,:] = numpy.concatenate((vel_old[1],vel_new),0)
            P = numpy.concatenate((P,P_new),1)
            
            
        elif n_new < n_old: #Deaths: Die Koordinaten die die maximale Abstände von den alten Objekten aufweisen, werden als "sterbende Objekte" betrachtet und daher werden von den Zuständen gelöscht 
            P_new = numpy.zeros((number_states,n_new*number_states))# Bei deaths werden die Varianzen der sterbenden Objete gelöscht
            for i in range(n_new):
                distances = [] #Liste mit Abständen zu den alten koordinaten
                for j in range(n_old):
                    distances.append(differenz(pos_old[:,j],pos_new[:,i])) #Vektorbetrag hinzufügen
                index_min =  min(range(len(distances)), key=distances.__getitem__) #maximalen index ausrechnen
                est_updated[0,i] = pos_old[0,index_min]
                est_updated[1,i] = vel_old[0,index_min]
                est_updated[2,i] = pos_old[1,index_min]
                est_updated[3,i] = vel_old[1,index_min]
                P_new[0:number_states,i*number_states:number_states*(i+1)] = P[0:number_states,index_min*number_states:number_states*(index_min+1)]
            P = P_new
            
        elif n_new == n_old:
            est_updated = est
            P = P
     


        
    except: #Für den Fall, dass für zwei aufeinander folgenden Zeitschritten n gleich null ist
        
        if n_new == 0:
            est_updated = numpy.array([])
            P = numpy.array([])
        else:
            est_updated = numpy.zeros((4,n_new))
            pos_new = numpy.transpose(numpy.array(anfangsWerte))
            P = numpy.zeros((number_states,n_new*number_states))
                
            for i in range(n_new):
                est_updated[0,i] = pos_new[0,i]
                est_updated[1,i] = 0
                est_updated[2,i] = pos_new[1,i]
                est_updated[3,i] = 0
                P[0:number_states,i*number_states:number_states*(i+1)] = P_i_init #Kovarianzmatrix des Schätzfehlers gleich P_init initialisiert
    return n_new,est_updated,P