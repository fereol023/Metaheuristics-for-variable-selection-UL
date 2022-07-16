# -*- coding: utf-8 -*-

#GBENOU Merveille
#JUGAND Tess

#Importation des packages
import pickle
import random
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from datetime import timedelta
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, recall_score
from sklearn import model_selection
from statistics import mean

#Note : 
#Tous les "#print" présents dans les algorithmes peuvent être excécuter en enlevant le # devant
#Peut être utile pour avoir une meilleure visualisation de ce que chaque algorithme fait

########################### ALGORITHMES NECESSAIRES A LA CONFIGURATION DE LA METAHEURISTIQUE ###########################

#Lire le fichier avec les données
def read(filename):
    data = pd.read_csv(filename, delimiter = ",", index_col = None)
    return data

#Validation croisée pour garantir que l'apprentissage soit représentatif de l'ensemble des données
def cross_validation(nfold, X, Y, model, matrix):
    k = model_selection.KFold(nfold)
    Y_test_lst = []
    Y_pred_lst = []

    #Séparer les données en k répartitions,
    #et pour chaque répartition on effectue un apprentissage
    for train_index, test_index in k.split(X, Y):

        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)

        #Somme des matrices de confusions
        matrix = matrix + confusion_matrix(Y_test, Y_pred)

        #Ajout des valeurs réelles (Y_test) dans la liste
        Y_test_lst.extend(Y_test)

        #Ajout des valeurs prédites par le modèle (Y_pred) dans une autre liste
        Y_pred_lst.extend(Y_pred)

    return matrix, Y_pred_lst, Y_test_lst

#Permet de faire un apprentissage
def learning(data, target):
    X = data.drop([target], axis=1).values #Toutes les variables sauf la variable cible
    Y = data[target].values

    #Initialise une matrice carrée de zéros de taille 2 x 2
    matrix = np.zeros((2, 2), dtype=int)
    
    #Régression logistique
    model = LogisticRegression(solver='liblinear')
    
    matrix, Y_test, Y_pred = cross_validation(nfold, X, Y, model, matrix)

    #print("-----------------MATRICE DE CONFUSION-----------------------")
    #print(matrix)
    #print("------------------------------------------------------------")
    #print(classification_report(Y_test, Y_pred))
    #print("---------------SCORE : RECALL modèle complet----------------")
    #print(recall_score(Y_test, Y_pred, average="macro"))

    return recall_score(Y_test, Y_pred, average="macro")

#Créer une population d'individus
def create_population(taille_pop, taille_var):
    pop = []
    for i in range(taille_pop):
        ind = []
        for j in range(taille_var):
            rand = random.choice([True, False])
            ind.append(rand)
            
        #Vérifier que les individus ne sont pas tous nuls
        sum = 0
        for k in ind:
            if k is False:
                sum = sum + 1
        if sum == taille_var:
            rand2 = random.randint(0, taille_var-1)
            ind[rand2] = True
        #Ajout de l'individu après vérification 
        pop.append(ind) 
        
    #print("------------------POPULATION DE BOOLEENS------------------")
    #print(pop)
    #print("-----------------------------------------------------------")
    return pop

#Sélectionne les colonnes en fonction de la valeur d'un individu
def preparation(data, ind, target):
    #Faire une copie du dataset pour pouvoir le modifier
    #print(data.columns)
    copy = data.copy()
    copy_target = copy[target]
    copy = copy.drop([target], axis=1)
    cols = copy.columns

    #Pour chaque colonne si la valeur de l'individu est True, la colonne est sélectionnée
    cols_selection = []
    for c in range(len(cols)):
        if ind[c] == True:
            cols_selection.append(cols[c])

    #Récupérer les données correspondantes qui seront utilisées pour l'apprentissage
    copy = copy[cols_selection]
    copy[target] = copy_target
    
    #print(copy)
    return copy

#Calcul du score pour chacun des individus
def fitness(d, pop, target):
    score_list = []
    for i in range(len(pop)):
        data = preparation(d, pop[i], target)
        score_list.append(learning(data, target))
        
    #print("\n------------------SCORE DES INDIVIDUS BOOLEENS----------------\n")
    #print(score_list)
    #print("------------------------------------------------------------")
    return score_list

#Choisir des individus dans la population de manière aléatoire
def choix_indiv(pop, number_of_choices, taille_pop, taille_var) :
    #Taille de population attendue : 
    dim_mutants = (taille_pop, taille_var) 
    pop = np.reshape(pop, dim_mutants)
    #Manipulation des lignes : 
    number_of_rows = pop.shape[0]
    random_indices = np.random.choice(number_of_rows, 
                               size = number_of_choices, 
                               replace=False) #replace false == indiv differents
    choosen_rows = pop[random_indices, :]
    
    #display choosen rows
    #print("\n-----------Random Indivs-------------")
    #print(choosen_rows)
    #print(random_indices)
    #print("---------------------------------------")
    return choosen_rows
    
#Créer des mutants
def create_mutants(pop, taille_pop, taille_var, F, R) :
    mutants = []
    for i in range(taille_pop):
        m = []
        
        #On récupère les individus pour la mutation
        r1 = R[0]
        r2 = R[1]
        r3 = R[2] 
        
        #On convertit les booléens en 0 et 1
        r1 = r1.astype(int)
        r2 = r2.astype(int)
        r3 = r3.astype(int)
        
        #Calcul des mutants
        for j in range(taille_var) :
            m_j = r1[j] + F*(r2[j] - r3[j])
            m.append(m_j)
            #Convertir les valeurs > 1 en 1 et les valeurs < 0 en 0
            if m[j] >= 1 :
                m[j] = 1
            if m[j] <= 0 :
                m[j] = 0
            #Puis on convertit en booléens les 0 1
            if m[j] == 1 : 
                m[j] = True
            else :
                m[j] = False
        
        #Ajout de l'individu        
        mutants.append(m) 
        
    #print("------------------POPULATION DE MUTANTS------------------")
    #print(mutants)
    #print("-----------------------------------------------------------")
    return mutants
        
#Fonction de croisement
def croisement(CR, pop, taille_pop, taille_var, mutants):
    U = []
    for i in range(taille_pop) :
        u = []
        v = pop[i]
        m = mutants[i]
        for j in range(taille_var) :
            r = random.uniform(0,1)
            if r > CR :
                u_j = v[j]
            else :
                u_j = m[j]
            u.append(u_j)
        #Rajouter l'individu issu du croisement à la nouvelle pop
        U.append(u)
    
    #print(U)
    return U
        
################################## METAHEURISTIQUE : EVOLUTION DIFFERENTIELLE ##########################################

def evolution_diff(data, target, CR, G, F, taille_pop) :

    #Répertoire de sortie
    #Il faut créer les deux fichiers dans votre ordinateur avec ces noms et chemins précis
    fichier_meilleurs_indivs_moy_scores = 'C:/EVOLUTION_DIFF_FILES/fichier_meilleurs_indivs_moy_scores.txt'
    fichier_meilleur_ttes_generations = 'C:/EVOLUTION_DIFF_FILES/fichier_meilleur_ttes_generations.txt'    

    #Ouvrir le fichier_meilleur_ttes_generations
    with open(fichier_meilleurs_indivs_moy_scores, 'wb') as file:
    
        best_scores_list = []
        best_ind_list = []
        T = [0] #Vecteur qui enregistre le temps
        generations = [0] #Vecteur qui enregistre les générations
    
        #Définir le nombre de variables sur les dimensions des données
        N = data.shape
        taille_var = N[1]
    
        #Créer la population initiale "pop"
        pop = create_population(taille_pop, taille_var)
        
        #Evaluer la population
        scores_pop = fitness(data, pop, target)
    
        #Calcul de la moyenne des scores sur la population
        avg_scores_pop = mean(scores_pop)

        #Récupérer le meilleur score et le meilleur individu de la population
        best_score_pop = np.max(scores_pop)
        argmax_pop = np.argmax(scores_pop)
        best_ind_pop = pop[argmax_pop]
    
        #Sauvegarde
        best_scores_list.append(best_score_pop)
        best_ind_list.append(best_ind_pop)
        
        #Sauvegarde dans le fichier_meilleurs_indivs_moy_scores pour la population
        pickle.dump("-------POPULATION INITIALE------", file)
        pickle.dump("Meilleur individu", file)
        pickle.dump(str(best_ind_pop), file)
        pickle.dump("Moyenne des scores", file)
        pickle.dump(str(avg_scores_pop), file)
        pickle.dump("---------------------------------", file)
        
        #Compteur
        g = 1
        while g <= G : 
        
            #Mesure le temps d'excécution
            debut = time.time()

            #Selectionner les individus dans la population
            R = choix_indiv(pop, number_of_choices, taille_pop, taille_var)
        
            #Création de la population de mutants
            mutants = create_mutants(pop, taille_pop, taille_var, F, R)   
        
            #Croisement
            U = croisement(CR, pop, taille_pop, taille_var, mutants)
        
            #Evaluer la nouvelle population issue du croisement
            scores_U = fitness(d, U, target)

            
            
            #Calcul de la moyenne des scores
            avg_scores_gen = mean(scores_U)
            
            #Récupérer le meilleur individu de la nouvelle population
            best_score_U = np.max(scores_U)
            argmax_U = np.argmax(scores_U)
            best_ind_U = U[argmax_U]
            
            #Comparer score nouvelle population et ancienne 
            if best_score_U < best_scores_list[-1] : 
                best_score_U = best_scores_list[-1]
                
            
            #Sauvegarde
            best_scores_list.append(best_score_U)
            best_ind_list.append(best_ind_U)
        
            #Sauvegarde dans le fichier_meilleurs_indivs_moy_scores pour chaque génération
            pickle.dump("-------GENERATION " + str(g) + " :---------", file)
            pickle.dump("Meilleur individu", file)
            pickle.dump(str(best_ind_U), file)
            pickle.dump("Moyenne des scores", file)
            pickle.dump(str(avg_scores_gen), file)
            pickle.dump("\n---------------------------------\n", file)
            
            #Affichage du du temps d'exécution
            fin = time.time()
            tps = timedelta(seconds=fin - debut)
            #Enregistrement du temps
            T.append(fin-debut)
            
            #print("génération : "+str(g)+ ", meilleur score : " + str(best_score_U) + ", temps: " + str(tps))
            
            #Vecteur des générations pour la représentation graphique
            generations.append(g)
            
            #Incrémenation des générations
            g +=1
        
        #Meilleur individu/score toutes générations confondues
        meilleur_score_ttes_generations = np.max(best_scores_list)
        indice_meilleur_ttes_generations = np.argmax(best_scores_list)
        meilleur_ind_ttes_generations = best_ind_list[indice_meilleur_ttes_generations]
        
        #Sauvegarde dans le fichier_meilleur_ttes_generations
        with open(fichier_meilleur_ttes_generations, 'wb') as file2 :
            pickle.dump("Meilleur score toutes générations : " + str(meilleur_score_ttes_generations), file2)
            pickle.dump("Paramètres F : " + str(F), file2)
            pickle.dump("Nombre de générations G : " + str(G), file2)
            pickle.dump("CR : " + str(CR), file2)
            pickle.dump("Meilleur individu toutes générations confondues :", file2)
            pickle.dump(str(meilleur_ind_ttes_generations), file2)
                
        #Représentations graphiques :
            
        #1. Evolution des scores
        #print(generations)
        #print(best_scores_list)
        plt.plot(generations, best_scores_list)
        plt.xlabel("Générations")
        plt.ylabel("Scores")
        plt.title("Evolution du score du meilleur individu par génération")
        plt.show()
        
        #2. Evolution du temps d'execution
        #print(T)
        plt.plot(generations, T)
        plt.xlabel("Générations")
        plt.ylabel("Temps d'exécution")
        plt.title("Evolution du temps d'exécution de l'Evolution Différentielle par génération")
        plt.show()

            
            
############################### FONCTION PRINCIPALE : POINT D'ENTREE DU PROGRAMME ######################################

if __name__ == '__main__':
    
    #Appel des données du fichier
    #Enregistrer au préalable le fichier dans le même dossier que celui où est enregistré ce code
    #Ou alors modifier la fonction read avec le chemin employé jusqu'à l'emplacement du fichier "projetinfo"
    d = read("C:\\Users\\gbeno\\Documents\\M1 IES\\S1\\1-PROG\\Projet\\projetinfo.csv")
    target = "class"

    #Définir les paramètres ici, pour permettre d'ajuster ceux-ci de manière simple et efficace
    F = 1
    G = 10000
    CR = 0.5
    taille_pop = 10
    number_of_choices = 3
    nfold = 10
    
    #Appel de la métaheuristique
    evolution_diff(d, target, CR, G, F, taille_pop)
    