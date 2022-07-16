# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 14:12:19 2021

@author: gbeno
"""
# Importation des packages
import random
import pickle
import numpy as np
import pandas as pd
import openpyxl
import sklearn
import heapq
import time
from datetime import timedelta
from sklearn.model_selection \
    import train_test_split
from sklearn.linear_model \
    import LogisticRegression
from sklearn.metrics import confusion_matrix,\
    accuracy_score, recall_score,\
    precision_score, classification_report
from sklearn import model_selection
from scipy.stats import bernoulli, binom
from statistics import mean


# Fonction qui permet de lire le
# fichier avec les données
def read(filename, target):
    # Stocke les données dans une variable
    data = pd.read_csv(filename, index_col=None, delimiter=",")
    # use .loc to select all columns except the targeted one
    D = data.loc[:, data.columns != target]
    #X=data.describe().transpose()
    #print(D.shape[1])
    #print(X)
    #print(data["class"].describe())
    return data, D.shape[1]

def vectprobas(length, p0 = 0.5) :
    # initie un vecteur de probas à 0.5 
    #probas = np.full(length, p0)  
    probas = [p0]*length
    #print(probas)
    return probas

'''
def create_population(taille_pop, nb_var) : #, probas):
    # Fonction qui permet de créer une population d'individus à partir d'un vecteur de probas
    # Choisir dans 
    boolean = [True, False]
    # le vecteur de probas
    probas = vectprobas(nb_var, p0 = 0.5)
    # les dimensions de la population
    pop = np.zeros((taille_pop, nb_var))
    for i in range(taille_pop):
        #ind = []
        for j in range(nb_var):
            pop[i,j] = np.random.choice(boolean, 1, probas[j])

        # verifier que les indivs ne sont pas que des 0
        sum = 0
        for k in ind:
            if k is False:
                sum = sum + 1
        if sum == nb_var:
            rand2 = random.randint(0, nb_var-1)
            ind[rand2] = True
            
        #pop.append(ind)
        
    #print("------------------POPULATION DE BOOLEENS6------------------")
    #print(pop)
    #print("-----------------------------------------------------------")
    return pop
'''


def create_population(taille_pop, nb_var, probas):
    # Fonction qui permet de créer une population d'individus à partir d'un vecteur de probas
    # Choisir dans 
    boolean = [True, False]
    # le vecteur de probas
    #probas = vectprobas(nb_var, p0 = 0.5)
    # les dimensions de la population
    pop = []
    for i in range(taille_pop):
        ind = []
        for j in range(nb_var):
            rand = np.random.choice(boolean, 1, probas[j])
            ind.append(rand)
        '''
        # verifier que les indivs ne sont pas que des 0
        sum = 0
        for k in ind:
            if k is False:
                sum = sum + 1
        if sum == nb_var:
            rand2 = random.randint(0, nb_var-1)
            ind[rand2] = True
        '''    
        pop.append(ind)
       
    #print("------------------POPULATION DE BOOLEENS6------------------")
    #print(pop)
    #print("-----------------------------------------------------------")
    return pop



def preparation(data, ind, target):
    # Sélectionne les colonnes en fonction de la valeur d'un individus

    # print(data.columns)
    copy = data.copy()
    copy_target = copy[target]
    copy = copy.drop([target], axis=1)
    cols = copy.columns

    # Pour chaque colonne si la valeur de l'individu est True
    # La colonne est sélectionnée
    cols_selection = []
    for c in range(len(cols)):
        if ind[c] == 1:
            cols_selection.append(cols[c])

    # Récupère les données correspondantes
    copy = copy[cols_selection]
    copy[target] = copy_target
    #print(copy)
    return copy


def cross_validation(nfold, X, y, model, matrix):
    # validation croisée
    k = model_selection.KFold(nfold)

    y_test_lst = []
    y_pred_lst = []

    # Permet de séparer les données en k répartitions
    # Pour chaque répartition on effectue un apprentissage
    for train_index, test_index in k.split(X, y):

        X_train, X_test = X[train_index],\
                          X[test_index]
        y_train, y_test = y[train_index], \
                          y[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Somme des matrices de confusions
        # Pour chacune des répartitions
        matrix = matrix + confusion_matrix(y_test,
                                           y_pred)

        # Ajout des valeurs réelles (y_test)
        # dans la liste
        y_test_lst.extend((y_test))

        # Ajout des valeurs prédites par le modèle (y_pred)
        # dans une autre liste
        y_pred_lst.extend((y_pred))

    return matrix, y_test_lst, y_pred_lst


def learning(data, target):
    X = data.drop([target], axis=1).values
    y = data[target].values

    # Initialise une matrice carrée de zéros
    # de taille 2
    matrix = np.zeros((2, 2), dtype=int)

    model = LogisticRegression(solver='liblinear')

    matrix, y_test, y_pred =\
        cross_validation(10, X, y, model, matrix)
    #print("-----------------MATRICE DE CONFUSION-----------------------")
    #print(matrix)
    #print("------------------------------------------------------------")
    #print(classification_report(y_test, y_pred))
    #print("---------------SCORE : RECALL modèle complet----------------")
    #print(recall_score(y_test, y_pred, average="macro"))

    return recall_score(y_test, y_pred, average="macro")

def fitness(d, pop, target_name):
    # Calcul du score pour chacun des individus
    
    score_list = []
    # pop.shape[0]
    for i in range(len(pop)):
        data = preparation(d, pop[i], target_name)
        score_list.append(learning(data, target_name))
    return score_list

def selection(pop, score_list, n=1):
    # Obtenir les n meilleurs individus de la population

    best_score = heapq.nlargest(n, score_list)
    #print(best_score)

    # Récupérer l'indice des n meilleurs scores
    best_score_index = heapq.nlargest(n, range(len(score_list)), key=score_list.__getitem__)
    #print(best_score_index)
    best_score_index = best_score_index[0]
    #print(best_score_index)
    # Récupérer les n meilleurs individus
    #new_pop = []
    #for x in best_score_index:
        #new_pop.append(pop[x])
    #print(pop)

    # Récu le meilleur indiv de la pop 
    best_indiv = pop[best_score_index]
    #print(best_indiv)
    #return new_pop
    return best_indiv, best_score

    
'''
-MAJ DU VECTEUR DE PROBA EN FONCTION DU MEILLEUR INDIV
LR = 0.1
probas[i] = probas[i]*(1-LR) + (LR*best_indiv[i])
'''

def maj_vect_probas(probas, best_indiv, LR=0.1) :
    # Mettre à jour un vecteur de probas à partir du meilleur individu de la population 
    for i in range(len(probas)) :
        probas[i] = probas[i]*(1-LR) + (LR*best_indiv[i])
    #print(probas)
    return probas
        
'''
-verifier s il faut faire une mutation avec MP
MP = 0.2
 si oui : 
     MS = 0.05
     probas[i] = probas[i]*(1-MS) + random.choice([0,1])*MS
'''

def mutation(new_probas, MS=0.2) : 
    # Fait une mutation du nouveau vecteur de probas à partir de MS
    for i in range(len(new_probas)) : 
        new_probas[i] = new_probas[i]*(1-MS) + random.choice([0,1])*MS
    #print(new_probas)
    return new_probas

def choix_mutation(MP) :
    # Détermine aléatoirement s'il faut faire une mutation ou pas 
    # Choisir dans 
    boolean = [True, False]
    choix = np.random.choice(boolean, 1, MP)
    #print(choix[0])
    return choix[0]
'''
-critère d arret
    si g<G=nb de géné voulue(5) alors retourner à l étape 2 : create pop
    sinon stop
'''

def PBIL(data, target_name, LR=0.1, MS=0.2, MP=0.2, taille_pop=10, G=3) : 
    
    # Répertoire de sortie 
    fichier_sorties = 'C:/PBIL_FILES/output_file.txt'
    fichier_best_all_generations = 'C:/PBIL_FILES/best_all_generations.txt'
    
    # Progression des meilleurs éléments
    score_Max = []
    indiv_score_Max = []
        # liste des indiv ayant un score max à chaque génération
    #tab_best_indiv = []
    
    # Progression du temps d'exécution
    T = []
    
    # Vecteur de probas initial
    p0 = 0.5
    probs = vectprobas(d[1], p0)
    

    g = 1
    while g <= G : 
        
        # Initier le temps d'exécution pour cette génération
        debut = time.time()
        
        # Créer une population 
        pop = create_population(taille_pop, nb_var=d[1], probas=probs)
        
        # Evaluer la pop
        score_list = fitness(d[0], pop, target)
        
        # Récup le meilleur indiv
        best_elts = selection(pop, score_list, 1)
        best_indiv = best_elts[0]
        best_score = best_elts[1]
        
        # Màj du vecteur de probas sur le meilleur indiv
        probs = maj_vect_probas(probs, best_indiv)
        
        # Mutation
        if choix_mutation(MP) == True : 
            probs = mutation(probs,  MS)
        else : 
            pass
        
        # Récupère la moyenne des scores des indivs de la génération
        score_avg = mean(score_list)
        
        # Récupère les meilleurs éléments de la génération
            # meilleur individu : s'ajoute à la liste des meilleurs indivs
            # /!\ faire un tableau pour mieux récup le meilleur indiv après
        indiv_score_Max.append(str(best_indiv))
            # son score : s'ajoute au vecteur des meilleurs scores
        score_Max.append(str(best_score))
            
        # Récupère la progression de la durée d'exécution pour g
        tps = timedelta(seconds=time.time()-debut)
        T.append(str(tps))
        
        # /!\Ecrit le meilleur indiv et la moyenne des scores dans un fichier texte
        #best_indiv
        #avg()
        #ecrire dans un fichier texte / répertoire
        with open(fichier_sorties, 'wb') as file:
            pickle.dump("-------GENERATION " + str(g) + ":\n", file)
            pickle.dump(str(best_indiv), file)
            pickle.dump(str(score_avg), file)
            pickle.dump("\n---------------------------------\n", file)
        
        #print("------------------------------------------------")
        #print(probs)
        #print("------------------------------------------------")
        # affiche 
        # la génération
        # le meilleur indiv
        #print("----------MEILLEUR INDIVIDU------------")
        # la durée d'execution 
        #print("génération : " + str(g) +
         #     " meilleur : " + str(best_indiv) +
          #    " temps : " + str(tps))
        
        g += 1
    
    # out of looping
        
    print(T)
    print(score_Max)
    
    # meilleur ttes génération confondues
    score_Max_all_gen = max(score_Max)
    # meilleure génération pour obtenir les paramètres associés
    best_g = score_Max.index(score_Max_all_gen)
    # meilleur individu
    best_indiv_all_gen = indiv_score_Max[best_g]
    
    print(score_Max_all_gen)
    
    
   
    # /!\ Ecrire le meilleur individu ttes générations confondues 
    # et les paramètres dans un fichier texte
    # quels paramètres ?
    with open(fichier_best_all_generations, 'wb') as file:
            pickle.dump("----- Meilleure Génération : " + str(best_g) + "e\n", file)
            pickle.dump("------Meilleur Indiv Ttes Gen :-------\n", file)
            pickle.dump(str(best_indiv_all_gen), file)
            pickle.dump("-------Meilleur score Ttes Gen :--------\n", file)
            pickle.dump(str(score_Max_all_gen), file)
            pickle.dump("------Meilleur Temps Ttes Gen :--------\n", file)
            pickle.dump(str(T[best_g]), file)
            pickle.dump("\n---------------------------------\n", file)
    
    
    
    
    #----FIN PBIL------------
    
    
    
        
# Fonction principale point d'entrée du programme
if __name__ == '__main__':
    # Appelle de la fonction read
    
    target = "class"
    d = read("C:\\Users\\gbeno\\Documents\\M1 IES\\PROG\\projetinfo.csv", target)
    #1#probs = vectprobas(d[1], 0.5)
    #learning(d[0], target)
    #2#pop = create_population(taille_pop=3, nb_var=d[1], probas=probs)
    #3#score_list = fitness(d[0], pop, target)  
    #4#best_indiv = selection(pop, score_list, 1)
    #5#new_probas = maj_vect_probas(probs, best_indiv)

    #6#new_probas = mutation(new_probas)
    # reste à modifier la create pop pour qu'il prenne en entrée un vecteur de proba crée
    #2-2#pop = create_population(taille_pop=3, nb_var=d[1], probas=new_probas)
    #2-3#score_list = fitness(d[0], pop, target)  
    #2-4#best_indiv = selection(pop, score_list, 1)
    
    PBIL(d, target)
    
    