# -*- coding: utf-8 -*-
import copy
import pyAgrum as gum
import StrategyTree as st
import numpy as np
from tqdm import tqdm
from itertools import permutations
from numpy import inf


def shallow_copy_list_of_copyable(l):
    """
    Une méthode statique du module qui crée une copie superficielle d'une liste passée en remplissant de plus chaque
    élément d'une nouvelle liste par un appel de méthode copy() d'un élément corespondant de la liste donnée.

    Args :
        l : une liste dont la copie il faut créer, objet du type list<Copyable> où Copyable est une interface abstraite
            qui dispose d'une méthode copy().

    Returns :
        cl : une copie superficielle d'une liste donnée qu'il fallait créer, objet du type list<Copyable>.
    """
    if l is None:
        return None
    return [subl.copy() for subl in l]


def shallow_copy_parent(parent):
    """
    Une méthode statique auxiliaire qui crée une copie superficielle d'un `` parent '' (cf une méthode
    TroubleShootingProblem._evaluate_all_st ci-dessous).

    Args :
        parent : un parent dont la copie il faut créer, objet du type list<tuple<NodeST, NodeST, StrategyTree>>.

    Returns :
        parent_copy : une copie superficielle d'un parent passé, objet du type
            list<tuple<NodeST, NodeST, StrategyTree>>.
    """
    if parent is None:
        return None
    return [tuple([elem.copy() if elem is not None else None for elem in par]) for par in parent]


def merge_dicts(left, right):
    """
    Uné méthode qui fusionne deux dictionnaire passés ne pas les changeant. Une couple (clé, valeur) de dictionnaire
    right est plus prioritaire que celle de left ; c'est-à-dire, s'il existe une valeur associée à la même clé k dans
    tous les deux dictionnaire, on ajoute dans le résultat celle qui appartient à right.

    Args :
        left : l'un de dictionnaires à fusionner, celui qui est moins prioritaire, objet du type dict.
        right : l'un autre dictionnaire à fusionner, celui qui est plus prioritaire, objet du type dict.

    Returns :
        res : le résultat de la fusion, objet du type dict.
    """
    res = left.copy()
    res.update(right)
    return res


class bcolors:
    """
    Une classe statique auxiliare qui n'est utilisé que pour stocker des couleurs différentes.
    """
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class TroubleShootingProblem:
    """
    Classe créée pour représenter un problème de Troubleshooting.
    Contient des méthodes divers pour résoudre le problème.
    Utilise le module pyAgrum pour manipuler le réseau bayesien utilisé pour 
    répresenter le problème.
    Les noeuds du réseau bayésien sont réferencés par des strings.
    
    Attributs :
        bayesian_network : Objet du type pyAgrum.BayesNet qui répresent le 
             réseau bayésien (BN) qui modélise un problème donné.
        bay_lp : Objet du type pyAgrum.LazyPropagation qui fait l'inference
             exacte pour le BN passé en argument.
        costs_rep : Dictionnaire de coûts où les clés représentent les noeuds 
            du BN et les valeurs leurs coûts de reparation (float).
        costs_rep_interval : Dictionnaire de coûts où les clés représentent les
            noeuds du BN et les valeurs des listes avec les coûts minimum et 
            maximum de reparation (floats).
        costs_obs : Dictionnaire de coûts où les clés représentent les noeuds 
            du BN et les valeurs leurs coûts d'observation (float).  
        repairable_nodes : Ensemble de noeuds qui correspondent aux éléments du 
            système concerné qui peuvent être réparés.
        unrepairable_nodes : Ensemble de noeuds qui correspondent aux éléments 
            d'un système qui ne peuvent pas être réparés.
        problem_defining_node : Noeud qui répresent le problème a être reglé
            (système fonctionnel où pas).
        observation_nodes : Ensemble de noeuds qui correspondent aux éléments 
            du système qui peuvent être observés.
        service_node : Noeud qui répresent l'appel au service (appel à la 
            réparation sûre du système).   
        evidences : Dictionnaire ou les clés répresentent les élements du 
            système qui ont des evidences modifiés dans bay_lp (donc qui ont 
            été réparés/observés) et les valeurs sont les inferences faites.
    """

    def __init__(self, bayesian_network, costs, nodes_types):
        """
        Crée un objet du type TroubleShootingProblem.
        Initialise bay_lp et ajoute des inférences vides aux noeuds du BN qui 
        peuvent être modifiés (réparés/observés/appelés). 
        
        Args :
            bayesian_network : Objet du type pyAgrum.BayesNet qui répresent le 
                réseau bayésien (BN) modélisant un problème donné.
            costs : Liste avec deux dictionnaires, le premier avec les coûts de 
                réparation (exactes ou avec des minimun/maximun) et le deuxième
                avec les coûts d'observation des noeuds.
            nodes_types : Dictionnaire où les clés représent les noeuds du BN 
                    et les valeurs leurs types associés (set de string).       
        """
        self.bayesian_network = gum.BayesNet(bayesian_network)
        self.bay_lp = gum.LazyPropagation(self.bayesian_network)
        self.costs_rep, self.costs_rep_interval = self._compute_costs(costs[0])
        self.costs_obs = costs[1].copy()
        self.repairable_nodes = {node for node in nodes_types.keys() \
                                if 'repairable' in nodes_types[node]}
        self.unrepairable_nodes = {node for node in nodes_types.keys() \
                                if 'unrepairable' in  nodes_types[node]}
        self.problem_defining_node = [node for node in nodes_types.keys() \
                                if 'problem-defining' in nodes_types[node]][0]
        self.observation_nodes = {node for node in nodes_types.keys() \
                                if 'observable' in nodes_types[node]}
        self.service_node = [node for node in nodes_types.keys() \
                                if 'service' in nodes_types[node]][0]
        self._start_bay_lp()
        
        # Variables internes utilisées pour le calcul exacte 
        self._nodes_ids_db_brute_force = []
        self._best_st = None
        self._best_ecr = None

# =============================================================================
# Méthodes fonctionnelles
# =============================================================================

    def _compute_costs(self, costs):
        """
        Prend en argument un dictionnaire de couts qui peut avoir des valeurs 
        exactes ou des intervalles de valeurs (de la forme [minimum, maximum])
        et le transforme en 2 dictionnaires, un avec les esperances de cout 
        pour chaque clé et l'autre avec des intervalles de valeurs pour chaque 
        clé.
        
        Args :
            costs : Dictionnaire de couts où les clés représentent les noeuds 
                du BN et les valeurs sont de nombres ou de listes de deux 
                nombres.
        Returns : 
            expected_cost : Dictionnaire où les clés représentent les noeuds du
                BN et les valeurs l'esperance de cout de ce noeud. Si la valeur
                initiale était déjà un nombre, ce nombre est seulement copié, 
                sinon on considère que la valeur est une variable aléatoire 
                avec une distribution uniforme dans l'intervalle et donc
                l'esperance est la moyenne des extremités de l'intervalle.
            interval_cost : Dictionnaire où les clés représentent les noeuds du
                BN et les valeurs sont des listes contenant les deux extremités
                des intervalles dans lequels les couts se trouvent. Si la 
                valeur initiale était déjà un nombre, ce nombre est copié comme
                les deux extremités. Si la valeur initiale était un iterable, 
                on le transforme en liste. 
        """
        # On initialise les dictionnaires
        expected_cost = {}
        interval_cost = {}
        
        # Pour chaque clé et valeur associé du dictionnaire passé en argument
        # on remplit les deux dictionnaires
        for k, v in costs.items():
            if isinstance(v, int) or isinstance(v, float):
                expected_cost[k] = v
                interval_cost[k] = [v, v]
            else:
                expected_cost[k] = sum(v) / 2
                interval_cost[k] = list(v)  
        return expected_cost, interval_cost

    def _start_bay_lp(self):
        """
        Ajoute des inférences vides aux noeuds du BN qui peuvent être modifiés 
        (réparés/observés/appelés).Ces évidences ne changent pas les 
        probabilités, elles servent pour qu'on puisse utiliser la méthode 
        chgEvidence de pyAgrum à la suite.
        """
        # On recupere les noeuds qui peuvent changer
        modifiable_nodes = self.repairable_nodes.union(self.observation_nodes)
        modifiable_nodes.add(self.service_node)
        modifiable_nodes.add(self.problem_defining_node)
        # On ajoute une evidence "vide" dans les noeuds 
        for node in modifiable_nodes:
            self.bay_lp.addEvidence(node, [1, 1]) 
            
        # Evidences commence vide et on l'utilise pour suivre quels sont les
        # noeuds de bay_lp qui ont des vraies evidences à un moment donné
        self.evidences = {}
            
    def reset_bay_lp(self, dict_inf = {}):
        """
        Reinitialise les inférences des noeuds du BN qui peuvent être modifiés 
        (réparés/observés/appelés). Pour les noeuds dans dict_inf, l'inférence 
        est mis à la valeur associé au noeud dans dict_inf, pour les autres 
        l'inférence est mis à 1.
        
        Args :
            dict_inf (facultatif) : Dictionnaire où les clés sont des noeuds et
                les valeurs sont des inférences. 
        """
        # On recupere les noeuds qui peuvent changer
        modifiable_nodes = self.repairable_nodes.union(self.observation_nodes)
        modifiable_nodes.add(self.service_node)
        modifiable_nodes.add(self.problem_defining_node)
        # Pour chaque noeud, soit on remet l'evidence à 1, soit on met la 
        # valeur trouvée dans dict_inf
        for node in modifiable_nodes:
            if node in dict_inf:
                self.add_evidence(node, dict_inf[node])
            else:
                self.remove_evidence(node)

    def add_evidence(self, node, evidence):
        """
        Fonction wrapper pour la fonction chgEvidence de l'objet bay_lp du 
        type pyAgrum.LazyPropagation qui additionne une inference et mantient 
        le dictionnaire evidences actualisé. L'evidence passé en argument ne 
        doit pas être une evidence "vide" (des 1, utilisé plutôt la fonction
        remove_evidence). 
        
        Args :
            node : String, nom du noeud de bay_lp qui va être modifié.
            evidence : nouvelle inference pour le noeud traité (généralement
                une string ici, cf. les types acceptés par chgEvidence)
        """
        self.bay_lp.chgEvidence(node, evidence)
        # On ajout le noeud au dictionnaire evidences
        self.evidences[node] = evidence
        
    def remove_evidence(self, node):
        """
        Fonction wrapper pour la fonction chgEvidence de l'objet bay_lp du 
        type pyAgrum.LazyPropagation qui retire une inference et mantient le 
        dictionnaire evidences actualisé. 
        
        Args :
            node : String, nom du noeud de bay_lp qui va être modifié.
        """
        self.bay_lp.chgEvidence(node, [1]*\
                            len(self.bayesian_network.variable(node).labels()))
        # Si l'evidence d'un noeud est remis à 1, le noeud n'est plus
        # un des noeuds qui ont une vraie evidence, donc on le retire
        # d'evidences
        self.evidences.pop(node, None)
        
    def get_proba(self, node, value):
        """
        Récupère à partir du réseau bayésien la probabilité que le noeud node
        ait la valeur value.
        
        Args:
            node : String, nom du noeud de bay_lp dont on veut calculer la
                probabilité
            value : String, valeur du noeud dont on veut calculer la
                probabilité
        Returns:
            La probabilité P(node = value)
        """
        p_tab = self.bay_lp.posterior(node)
        inst = gum.Instantiation(p_tab)
        inst.chgVal(node, value)
        return p_tab[inst]

    def draw_true_prices(self):
        """
        Tire au hasard des prix de réparation selon des lois uniformes sur les
        intervalles stockés dans self.costs_rep_interval.
        
        Returns :
            Dictionnaire avec prix de réparation.
        """
        return {n: np.random.uniform(*self.costs_rep_interval[n])\
                for n in self.costs_rep_interval}

    def simple_solver(self, debug = False):
        """
        Solveur simple pour le problème du TroubleShooting.
        On ne prend pas en considèration des observations et on ne révise pas 
        les probabilités, c'est-à-dire on ne met pas à jour les probabilités 
        si on répare une composante.
        À cause de cela, ce solveur n'est pas iteractif et renvoie l'ordre de 
        réparation entière (jusqu'au appel au service).
        
        Args : 
            debug (facultatif) : si True, affiche des messages montrant le 
                deroulement de l'algorithme.
        Returns :
            Un tuple avec la séquence des noeuds à être réparés dans l'ordre et
            l'esperance du coût de réparation de cette séquence.
        """     
        # On crée un dictionnaire avec les efficacités liées à chaque noeud 
        # réparable + service
        dic_eff = {}
        
        # On copie les évidences actuelles pour remettre le réseau à son état
        # de départ
        evid = self.evidences.copy()
        
        # Pour chaque noeud réparable + service (et qui n'est pas irréparable):
        for node in (self.repairable_nodes|{self.service_node}) - \
        (self.unrepairable_nodes):
                        
            # On rajoute le fait que le dispositif est en panne
            self.add_evidence(self.problem_defining_node, "yes")               
            
            # On calcule la probabilité que le noeud actuel soit cassé                              
            # p(node != Normal|Ei)
            # Ei = les informations connues au tour de boucle actuel
            if node != self.service_node:
                p_noeud_casse = self.get_proba(node, "yes")
            else:
                p_noeud_casse = self.get_proba(node, "no")

            # on ne sait plus si le dispositif est en panne ou pas
            self.remove_evidence(self.problem_defining_node)
            
            dic_eff[node] =  p_noeud_casse / self.costs_rep[node] 
            
            if debug == True:
                print("noeud consideré : " + node)
                print("proba p(node != Normal|Ei) : ", \
                      p_noeud_casse)            
                print("éfficacité du noeud : ", dic_eff[node])
                print()
            
            # On retourne l'évidence du noeud à celle qui ne change pas les 
            # probabilité du départ
            self.remove_evidence(node)
        # On trie les noeuds par rapport aux efficacités
        rep_seq = sorted(dic_eff.items(), key = lambda x: x[1], reverse = True)           
        # On ne veut que les noeuds, pas les valeurs des efficacités
        rep_seq = [r[0] for r in rep_seq]
        # On renvoie la liste jusqu'au appel au service, pas plus
        rep_seq = rep_seq[:rep_seq.index(self.service_node) + 1]

        # On calcule le coût espéré de la sequence de réparation
        # On commence par le cout de réparation du prémier noeud de la séquence
        proba = 1
        exp_cost = self.costs_rep[rep_seq[0]] * proba
        
        # On calcule maintenant la probabilité que la réparation de ce noeud
        # n'a pas résolu le problème, P(e != Normal | repair(ci), Ei). Comme
        # on n'a qu'un seul défaut, cela vaut P(ci = Normal | Ei).
        self.add_evidence(self.problem_defining_node, "yes")
        
        if node != self.service_node:
            p = self.get_proba(rep_seq[0], "no")
        else:
            p = self.get_proba(rep_seq[0], "yes")
            
        proba *= p

        # Le premier répare n'a pas résolu le problème, on change l'évidence
        # du noeud pour réfletir cela         
        self.add_evidence(rep_seq[0], "no")         
  
        if debug == True:
            print("Calcul de l'esperance de coût \n")
            print ("premier noeud réparé : ", rep_seq[0])
            print("esperance partiel du coût de la séquence : ", exp_cost)

        for node in rep_seq[1:]:
            # On somme le coût de réparation du noeud courant * proba
            exp_cost += self.costs_rep[node] * proba
            
            # Proba que la réparation de ce noeud n'a pas résolu le problème,
            # P(e != Normal | repair(ci), Ei) = P(ci = Normal | Ei).
            if node != self.service_node:
                p = self.get_proba(node, "no")
            else:
                p = self.get_proba(node, "yes")
        
            proba *= p

            if debug == True:
                print("proba p(e != Normal|réparation des tous les noeuds " \
                               + "déjà considerés) : ", proba)     
                print()
                print("noeud réparé : ", node)
                print("esperance partiel du coût de la séquence : ", exp_cost)
            # On actualise l'évidence du noeud concerné
            self.add_evidence(node, "no")  

        # On remet le reseau à l'état de départ
        self.reset_bay_lp(evid)
        
        return rep_seq, exp_cost
           
    
    def simple_solver_obs(self, debug = False):
        """
        Solveur simple pour le problème du Troubleshooting.
        On prend en considèration des paires "observation-réparation" (cf. 
        définition dans l'état de l'art) mais pas les observations globales 
        et on révise les probabilités, c'est-à-dire on met à jour les 
        probabilités quand on "répare" une composante avant de calculer le 
        prochaine composante de la séquence. 
        
        Le solveur n'est pas encore iteractif et renvoie l'ordre de réparation
        entière (jusqu'au appel au service). Cette choix à été fait car on 
        utilise cet algorithme comme part de l'agorithme plus complexe et 
        iteratif. 
         
        Args : 
            debug (facultatif) : si True, affiche des messages montrant le 
                deroulement de l'algorithme.
        Returns :
            Un tuple avec la séquence des noeuds à être réparés dans l'ordre et
            l'esperance du coût de réparation de cette séquence.
        """
        # On initialise la sequence de réparation vide
        rep_seq = []
        
        # On initialise l'esperance de cout de la réparation en 0
        exp_cost = 0
        
        # proba_cost est la probabilité que le système ne fonctionne pas à 
        # l'étape actuel. Commence à 1 car on sait que le système est en panne
        proba_cost = 1
        
        # On copie les évidences actuelles pour remettre le réseau à son état
        # de départ
        evid = self.evidences.copy()
        
        # On recupère les noeuds qui ont des évidences
        evidence_nodes = {k for k in self.evidences}
        
        # On recopie les noeuds qui ne sont pas reparables et on ajoute les
        # noeuds déjà réparés
        unrep_nodes = self.unrepairable_nodes.copy() | evidence_nodes
        
        # On itère jusqu'à ce qu'il n'existe plus de noeud qui peut être reparé
        reparables = (self.repairable_nodes | {self.service_node}) \
        - unrep_nodes
        while len(reparables) != 0:  
            if debug == True:
                print("Les noeuds considerés dans ce tour de boucle : ",\
                      reparables)
                    
            # On crée un dictionnaire avec les efficacités liées à chaque noeud 
            # réparable + service dans ce tour de boucle
            dic_eff = {}  
             
            # On crée un dictionnaire avec les couts utilisés pour calculer 
            # l'esperance de cout de la séquence.
            # On se sert que du cout lié au noeud choisi à la fin de chaque 
            # tour de boucle.
            dic_costs = {}
            
            # On crée un dictionnaire avec les probabilités que chaque noeud
            # soit cassé.
            dic_p = {}
            
            # Pour chaque noeud réparable + service (et qui n'est pas 
            # irréparable):
            for node in reparables:   
                # On rajoute le fait que le dispositif est en panne
                self.add_evidence(self.problem_defining_node, "yes")               
                
                # On calcule la probabilité que le noeud actuel soit cassé                              
                # p(node != Normal|Ei)
                # Ei = les informations connues au tour de boucle actuel
                if node != self.service_node:
                    p = self.get_proba(node, "yes")
                else:
                    p = self.get_proba(node, "no")
                dic_p[node] = p
                
                # On calcule le cout esperé du pair observation-repair pour le
                # noeud actuel
                if node in self.observation_nodes:
                    cost = self.costs_obs[node] + dic_p[node] \
                        * self.costs_rep[node]
                else:
                    cost = self.costs_rep[node]
                
                # Une fois qu'on aura fait l'observation-repair, on ne sait
                # plus si le dispositif est en panne ou pas
                self.remove_evidence(self.problem_defining_node)
                # On récupere le cout pour le calcul de l'ésperance
                dic_costs[node] = cost
                
                # On calcule alors l'efficacité du noeud
                dic_eff[node] = dic_p[node] / cost
                

                if debug == True:
                    print("noeud consideré : " + node)
                    print("proba p(node != Normal|Ei) : ",\
                          dic_p[node])
                    print("coût esperé du pair observation-repair : ", cost)
                    print("éfficacité du noeud : ", dic_eff[node])
                    print()
            
            # On trie les noeuds par rapport aux efficacités
            seq = sorted(dic_eff.items(), key = lambda x: x[1], \
                             reverse = True) 
            # Le noeud choisi est ceux avec la meilleure efficacité dans le 
            # tour de boucle actuel
            chosen_node = seq[0][0]
            rep_seq.append(chosen_node)
            
            # On calcule la contribution à l'ésperance du cout de la sequence
            # de ce noeud
            exp_cost += dic_costs[chosen_node] * proba_cost

            proba_cost *= (1 - dic_p[chosen_node])
            
            if debug == True:
                print("noeud choisi dans ce tour de boucle : ", chosen_node)
                print("contribution à l'ésperance du coût de la séquence : ",\
                      dic_costs[chosen_node])
                print("ésperance du coût partiel : ", exp_cost)
                print()
                            
            # On garde ce noeud au dictionnaire repares pour qu'on puisse
            # mantenir le reseau à jour a chaque tour de la boucle while 
            if chosen_node != self.service_node:
                unrep_nodes.add(chosen_node)
                reparables = (self.repairable_nodes |set([self.service_node]))\
                    - unrep_nodes
                self.add_evidence(chosen_node, "no")
            else:
                break
        # On retourne aux évidences du début
        self.reset_bay_lp(evid)           
        return rep_seq, exp_cost
 
    
    def myopic_solver(self, debug = False, esp_obs = False):
        """
        Implémente une étape du solveur myope. Étant donné l'état actuel du
        réseau, ce solveur utilise dans un premier temps le simple_solver_obs
        pour déterminer quelle action du type "observation-réparation" serait
        la meilleure. Ensuite, il calcule les coûts myopes espérés avec chaque
        observation possible et choisit à la fin la meilleure action à être
        prise.
        
        Cette fonction est itérative et ne fait qu'un seul tour de
        l'algorithme myope car elle attend des nouvelles informations venues
        de l'utilisateur (résultat de l'observation si c'est le cas).
        
        Args : 
            debug (facultatif) : si True, affiche des messages montrant le 
                deroulement de l'algorithme.
            esp_obs (facultatif) : si True, retourne en plus un dictionnaire
                indexé par les observations possibles et contenants leurs couts
                myopes espérés respectifs.
        Returns :
            Le meilleur noeud de ce tour et son type ("repair" ou "obs"). Si
            esp_obs == True, retourne aussi le dictionnaire des couts des
            observations.
        """
        # Liste des observations generales qu'on peut faire
        nd_obs = self.observation_nodes.intersection(self.unrepairable_nodes)
        nd_obs = list(nd_obs.difference(self.evidences.keys()))
        
        # Ésperance du coût de la séquence calculé au état actuel sans 
        # observations suplementáires
        seq_ecr, ecr = self.simple_solver_obs()
        if debug:
            print("Liste des observations possibles :")
            print(nd_obs)
            print("Resolution sans obsérvations :")
            print(seq_ecr)
            print("Ecr : " + str(ecr))
        # Dictionnaire pour les ésperances de coût si on fait une observation 
        # generale avant de calculer la séquence
        eco = {}
        
        # Pour chaque noeud d'obsérvation génerale, on calcule l'ésperance de 
        # coût quand on fait l'observation
        for node in nd_obs:
            # On récupere les probabilités des valeurs possibles du noeud
            p = self.bay_lp.posterior(node)                
            inst = gum.Instantiation(p)
            
            # ECO[node] = cost_obs[node] + somme(ésperance de la séquence 
            # calculé après chaque observation possible du node * probabilité 
            # de chaque observation)
            eco[node] = self.costs_obs[node]
            
            # Pour chaque valeur possible du noeud
            for k in self.bayesian_network.variable(node).labels():
                inst.chgVal(node, k)  
                # On recupere la probabilité de la valeur actuelle
                proba_obs = p[inst]
                # On dit qu'on a observé la valeur actuelle
                self.bay_lp.chgEvidence(node, k)
                # On calcule l'ésperance de coût de la séquence generé avec
                # l'observation du noeud avec la valeur actuel
                _, ecr_node_k = self.simple_solver_obs()
                eco[node] += ecr_node_k * proba_obs
                # On retourne l'évidence du noeud à celle qui ne change pas les 
                # probabilités du départ du tour de boucle actuel
                self.bay_lp.chgEvidence(node,\
                    [1]*len(self.bayesian_network.variable(node).labels()))
        
        if debug:
            print(eco)
        # On ordonne les ECO de façon croissant
        seq = sorted(eco.items(), key = lambda x: x[1]) 
        # Si ecr < min(eco), alors on fait pas d'observation dans ce tour, 
        # sinon on fait l'observation avec le plus petit ECO
        if seq == [] or ecr < seq[0][1]:
            # On récupere le premier élément de la séquence liée au ecr
            chosen_node = seq_ecr[0]
            type_node = "repair"
        else:
             # On recupere le noeud observation avec le plus petit ECO
            chosen_node = seq[0][0]
            type_node = "obs"
 
        # On retourne aux évidences du début
        self.reset_bay_lp(self.evidences) 

        if esp_obs:
            return chosen_node, type_node, eco
        return chosen_node, type_node
    
    def myopic_wraper(self, debug = False):
        """
        Interface textuelle pour le solveur myope. Utilise myopic_solver à
        chaque tour de boucle pour déterminer la meilleure action à prendre.
        Si c'est une observation, le résultat de l'observation est demandé,
        sinon on demande juste si l'action a résolu le problème. Les
        élicitations de couts ne sont pas implémentées. Les entrées de
        l'utilisateur ne sont pas sécurisées.
        
        Args : 
            debug (facultatif) : si True, affiche des messages montrant le 
                deroulement de l'algorithme.
        """
        print("Bienvenue! Suivez les instructions pour réparer votre"
              " dispositif")
        fixed = False
        while not fixed:
            node, type_node = self.myopic_solver(debug) 
            if type_node == "obs":
                print("Faire l'observation du node " + node)
                possibilites = self.bayesian_network.variable(node).labels()
                print("Résultats possibles :")
                for i, p in enumerate(possibilites):
                    print("{} : {}".format(i, p))
                print("Tapez le numero du résultat observé")
                val_obs = int(input())
                self.add_evidence(node, possibilites[val_obs])
            else:
                print("Faire l'observation-réparation suivante : " + node)
                if node == self.service_node:
                    break
                print("Problème résolu ? (Y/N)")
                val_rep = input()
                if val_rep in "Yy":
                    fixed = True
                else:
                    # Après une réparation, certaines observations peuvent
                    # être devenues obsolètes, il faut les exclure et permettre
                    # de les refaire.
                    obsoletes = self.observation_obsolete(node) 
                    self.add_evidence(node, "no")
                    for obs in obsoletes:
                        self.evidences.pop(obs)
                    self.reset_bay_lp(self.evidences)
        print("Merci d'avoir utilisé le logiciel !")
        self.reset_bay_lp()
                
        
    def noeud_ant(self, node, visites):
        """
        Détermine tous les noeuds d'observation impactés par un changement
        du noeud node et qui sont antécesseurs de node, sans visiter les noeuds
        déjà dans l'ensemble des visites. Cette fonction est auxiliaire et n'a
        pas vocation à être appellée en dehors de la fonction principale
        observation_obsolete.
        
        Args :
            node : String, nom du noeud don l'information a changé.
            visites : Ensemble, contient les noeuds déjà visités.
            
        Returns :
            Ensemble des noeuds d'observation affectés par node et qui sont
            antecesseurs de node sans être dans visites.
        """
        # Ensemble des noeuds d'observation impactés
        ant_obs = set()
        
        # Parents du noeud courant
        parents = {self.bayesian_network.names()[p] \
                   for p in self.bayesian_network.parents(node)}
        parents = parents.difference(visites)
        
        # Pour chaque parent non-visité
        for p in parents:
            # On l'ajoute aux visités
            visites.add(p)
            
            # S'il contient une évidence et qu'il est un noeud d'observation
            # globale, alors on le rajoute à l'ensemble des noeuds
            # d'observation impactés et on continue de façon récursive sur
            # ses parents.
            if p in self.evidences:
                if p in self.unrepairable_nodes\
                    and p in self.observation_nodes:
                    ant_obs.add(p)
                    ant_obs.update(self.noeud_ant(p, visites))
                
                # S'il contient une évidence mais il n'est pas un noeud
                # d'observation, on ne fait plus d'appel récusif.
                
            # S'il ne contient pas d'évidence, on continue de façon récursive
            # sur ses parents.
            else:
                ant_obs.update(self.noeud_ant(p, visites))
        return ant_obs
        
    def observation_obsolete(self, node):
        """
        Étant donné un noeud dont l'information a changé, on détermine, à
        partir du réseau bayésien, tous les noeuds d'observation impactés par
        ce chagement.
        
        Args :
            node : String, nom du noeud dont l'information a changé.
            
        Returns :
            Ensemble contenant les noeuds d'observation impactés.
        """
        # La racine est visitée
        visites = {node}
        # Ajout des observations impactées antécesseures de la racine dans
        # la liste d'observations impactées
        obs = self.noeud_ant(node, visites)
        
        # Parcours avec une pile
        stack = [node]
        while stack != []:
            n = stack.pop()
            
            # Enfants du noeud courant qui n'ont pas encore été visités
            enfants = {self.bayesian_network.names()[p]\
                       for p in self.bayesian_network.children(n)}
            enfants = enfants.difference(visites)
            
            # Pour chaque enfant
            for en in enfants:
                # On le marque comme visité
                visites.add(en)
                
                # S'il contient une évidence et qu'il est un noeud
                # d'observation, alors on le rajoute dans la liste des
                # observations impactées et on visite ses enfants.
                if en in self.evidences:
                    if en in self.unrepairable_nodes \
                        and en in self.observation_nodes:    
                        obs.add(en)
                        stack.append(en)
                        
                # S'il contient une évidence mais c'est un noeud de réparation,
                # cette évidence reste mais les antécesseurs de ce noeud sont
                # potentiellement impactés, il faut appeler noeud_and en ce
                # noeud.
                    elif en in self.repairable_nodes:
                        obs.update(self.noeud_ant(en, visites))
                        
                # S'il ne contient pas d'évidence, on le rajoute à la pile pour
                # visiter ses enfants.
                else:
                    stack.append(en)
        return obs
    
    def compute_EVOIs(self):
        """
        Calcule les valeurs espérées d'information (EVOIs) correspondant à
        avoir plus d'information sur l'intervalle de valeur des couts de
        réparation pour chaque composante réparable.

        Returns :
            Dictionnaire indexé par les noeuds réparables contenant la valeur
            d'une information plus précise du cout de réparation de ces noeuds.

        """
        # Noeuds encore réparables
        nd_rep = (self.repairable_nodes | {self.service_node})
        nd_rep = list(nd_rep.difference(self.evidences.keys()))
        
        # Dictionnaire indexé par les noeuds encore réparables contenant
        # la valeur d'une information plus précise du cout de réparation de
        # ces noeuds
        evoi = {}
        
        # Cout espéré de réparation avec les informations courantes
        _, expected_cost_repair = self.simple_solver_obs()
        
        # Boucle sur les noeuds encore réparables
        for noeud in nd_rep:
            # Initialisation du calcul de l'EVOI
            evoi[noeud] = expected_cost_repair
            
            # Espérance courante du cout de réparation
            alpha = self.costs_rep[noeud]
            
            # On considère que le cout de réparation est dans l'intervalle
            # [alpha, Cmax] et on recalcule le cout espéré de réparation avec
            # cet intervalle
            self.costs_rep[noeud] = (self.costs_rep_interval[noeud][0]\
                                     + alpha) / 2
            _, expected_cost_repair_minus = self.simple_solver_obs()
            
            # L'EVOI est mis à jour avec ce cout multiplié par la proba 0.5
            # d'être dans cet intervalle
            evoi[noeud] -= expected_cost_repair_minus * 0.5
            
            # On considère que le cout de réparation est dans l'intervalle
            # [Cmin, alpha] et on recalcule le cout espéré de réparation avec
            # cet intervalle
            self.costs_rep[noeud] = (self.costs_rep_interval[noeud][1]\
                                     + alpha) / 2
            _, expected_cost_repair_plus = self.simple_solver_obs()
            
            # L'EVOI est mis à jour avec ce cout multiplié par la proba 0.5
            # d'être dans cet intervalle
            evoi[noeud] -= expected_cost_repair_plus * 0.5

            # L'espérance de cout de réparation revient à sa valeur initiale
            self.costs_rep[noeud] = alpha
        return evoi
    
    def best_EVOI(self):
        """
        Détermine la composante qui a la plus grande valeur espérée
        d'information (EVOI) correspondant à avoir plus d'information sur
        l'intervalle de valeur de son cout.
        
        Returns :
            Tuple avec le nom du noeud de réparation avec la plus grande EVOI
            et la valeur d'EVOI correspondante.
        """
        evois = self.compute_EVOIs()
        return sorted(evois.items(), key = lambda x: x[1], reverse = True)[0]
    
    def elicitation(self, noeud, islower):
        """
        Met à jour l'intervalle de valeurs de cout pour le noeud et son
        espérance en fonction de la réponse de l'utilisateur.
        
        Args :
            noeud : String, nom du noeud à mettre à jour.
            islower : Booléan, représente la réponse à la question : Est-ce que
                le cout est plus petit que l'espérance courante ?
        """
        if islower:
            self.costs_rep_interval[noeud][1] = self.costs_rep[noeud]
        else:
            self.costs_rep_interval[noeud][0] = self.costs_rep[noeud]
        self.costs_rep[noeud] = sum(self.costs_rep_interval [noeud]) / 2
        
    
    def ECR_ECO_wrapper(self, debug = False):      
        """
        Calcule l'ECR myope pour chaque prochaine "observation-réparation"
        possible et l'ECO pour chaque prochaine observation globale possible.
        
        Args :
            debug (facultatif) : si True, affiche des messages montrant le 
                deroulement de l'algorithme.

        Returns :
            chosen_node : noeud choisi
            type_node : type du noeud choisi
            list_ecr : liste d'ECRs des noeuds d'"observation-réparation"
            list_eco : liste d'ECOs des noeuds d'observation globale
        """
        ecr = {}
        _, _, eco = self.myopic_solver(debug, esp_obs = True)     
        
        # On recupère les noeuds qui ont des évidences
        evidence_nodes = {k for k in self.evidences}
        
        # On recopie les noeuds qui ne sont pas reparables et on ajoute les
        # noeuds déjà réparés
        unrep_nodes = self.unrepairable_nodes.copy() | evidence_nodes
        
        # On itère dans les noeuds qui peuvent être reparés
        reparables = (self.repairable_nodes | {self.service_node}) \
        - unrep_nodes        
        for node in reparables:
            # On rajoute le fait que le dispositif est en panne
            self.add_evidence(self.problem_defining_node, "yes")               
            
            # On calcule la probabilité que le noeud actuel soit cassé                              
            # p(node != Normal|Ei)
            # Ei = les informations connues au tour de boucle actuel
            if node != self.service_node:
                p_noeud_casse = self.get_proba(node, "yes")
            else:
                p_noeud_casse = self.get_proba(node, "no")
            
            # On calcule le cout esperé du pair observation-repair pour le
            # noeud actuel
            if node in self.observation_nodes:
                cost = self.costs_obs[node] + \
                p_noeud_casse * self.costs_rep[node]
            else:
                cost = self.costs_rep[node]
            
            # Une fois qu'on aura fait l'observation-repair, on ne sait
            # plus si le dispositif est en panne ou pas
            self.remove_evidence(self.problem_defining_node)
             
            if node != self.service_node:
                # On calcule le ecr du reste de la séquence (étant donné qu'on 
                # a réparé le noeud actuel en premier)
                self.add_evidence(node, "no")
                _, cost_seq = self.simple_solver_obs(debug)
                self.remove_evidence(node)

                ecr[node] = cost + (1 - p_noeud_casse) * cost_seq
            else:
                ecr[node] = cost
                
        list_eco = sorted(eco.items(), key = lambda x: x[1])
        list_ecr = sorted(ecr.items(), key = lambda x: x[1])
        
        if list_eco != [] and list_eco[0][1] < list_ecr[0][1]:
            chosen_node = list_eco[0][0]
            type_node = "obs"
        else:
            chosen_node = list_ecr[0][0]
            type_node = "repair"
            
        return chosen_node, type_node, list_ecr, list_eco

    def simple_solver_tester(self, true_prices, epsilon, nb_min = 100, \
                             nb_max = 200):
        """
        Test empirique de la méthode simple_solver. Cette méthode calcule la
        séquence d'actions à l'aide de simple_solver et réalise au plus nb_max
        repétitions d'un système tiré au hasard : à chaque fois qu'on a une 
        probabilité qu'une action résoud le problème, on tire au hasard pour 
        déterminer si le problème a effectivement été résolu ou pas suite à 
        cette action. Si après nb_min repétitions l'erreur estimée est plus
        petite que epsilon, on fait une sortie anticipée. La fonction calcule
        aussi les couts empiriques de réparation, en utilisant pour cela
        true_prices. Cette méthode utilise la single fault assumption.
        
        
        Args :
            true_prices : Dictionnaire de prix de réparation des composantes
                réparables
            epsilon : Tolerance de la moyenne
            nb_min : Entier, nombre minimum de répétitions à être realisées
            nb_max : Entier, nombre maximum de répétitions à être realisées
            
        Returns :
            booléeen, True en cas de sortie anticipée, False sinon
            moyenne des couts observés
            tableau avec le nombre de composantes réparées à chaque répétition
        """
        
        costs = np.zeros(nb_max)
        cpt_repair = np.zeros(nb_max)
        sortie_anti = False
        # Comme la sequence ne change pas, on peut la calculer en dehors de 
        # la boucle
        seq, _ = self.simple_solver()
        # On fait au plus nb_max tests
        for i in tqdm(range(nb_max)):
            self.reset_bay_lp()
            for node in seq:
                costs[i] += true_prices[node]
                cpt_repair[i] += 1
                # On effectue une réparation
                if node == self.service_node:
                    break
                else:
                    # on récupere une probabilité:
                    # proba_sys = p(e = Normal| repair(node), Ei)
                    # = p(node != Normal | Ei)
                    self.add_evidence(self.problem_defining_node, "yes")
                    p = self.get_proba(node, "yes")
                    self.remove_evidence(self.problem_defining_node)

                    # On teste pour voir si le système marche 
                    if np.random.rand() <= p:
                        break
                    # Le noeud node marche surement
                    self.add_evidence(node, "no")
                    
            if i >= nb_min\
                and 1.96 * costs[:i + 1].std()/np.sqrt(i + 1) < epsilon:
                sortie_anti = True
                break
            
        self.reset_bay_lp()
        return sortie_anti, costs[:i + 1].mean(), cpt_repair[:i + 1]

    def simple_solver_obs_tester(self, true_prices, epsilon, nb_min = 100, \
                             nb_max = 200):
        """
        Test empirique de la méthode simple_solver_obs. Cette méthode calcule
        la séquence d'actions à l'aide de simple_solver_obs et réalise au plus
        nb_max répétitions d'un système tiré au hasard : si on a une paire
        "observation-réparation", on tire au hasard si la composante
        correspondante marche ou pas. Si oui, on ajoute juste le cout de
        l'observation et on continue, si non, on ajoute les couts d'observation
        et de réparation et on s'arrête (single fault assumption). Si on a
        une réparation simple sans observation associée, on ajoute directement
        le cout de réparation de la composante. Si après nb_min repétitions
        l'erreur estimée est plus petite que epsilon, on fait une sortie
        anticipée. La fonction calcule les couts empiriques de réparation, en
        utilisant pour cela true_prices.
        
        Args :
            true_prices : Dictionnaire de prix de réparation des composantes
                réparables
            epsilon : Tolerance de la moyenne
            nb_min : Entier, nombre minimum de répétitions à être realisées
            nb_max : Entier, nombre maximum de répétitions à être realisées
            
        Returns :
            booléeen, True en cas de sortie anticipée, False sinon
            moyenne des couts observés
            tableau avec le nombre de composantes réparées à chaque répétition
        """
        costs = np.zeros(nb_max)
        cpt_repair = np.zeros(nb_max)
        sortie_anti = False
        # Comme la sequence ne change pas, on peut la calculer en dehors de 
        # la boucle
        seq, _ = self.simple_solver_obs()
        # On fait au plus nb_max tests
        for i in tqdm(range(nb_max)):
            self.reset_bay_lp()
            for node in seq:
                # On fait l'observation du noeud si possible, sinon, on le 
                # repare
                if node in self.observation_nodes: 
                    costs[i] += self.costs_obs[node]
                else:
                    costs[i] += true_prices[node]
                cpt_repair[i] += 1
                
                # On effectue un pair observation-réparation
                if node == self.service_node:
                    break
                else:
                    # on récupere une probabilité:
                    # proba_sys = p(e = Normal| repair(node), Ei)
                    # = p(node != Normal | Ei)
                    self.add_evidence(self.problem_defining_node, "yes")
                    p = self.get_proba(node, "yes")
                    self.remove_evidence(self.problem_defining_node)

                    # On teste pour voir si le système marche 
                    if np.random.rand() <= p:
                        # Si le noeud était cassé et est observable, 
                        # on somme son cout de réparation
                        if node in self.observation_nodes: 
                            costs[i] += true_prices[node]
                        break

                    # Le noeud node marche surement
                    self.add_evidence(node, "no")
            if i >= nb_min\
                and 1.96 * costs[:i + 1].std()/np.sqrt(i + 1) < epsilon:
                sortie_anti = True
                break
        self.reset_bay_lp()
        return sortie_anti, costs[:i + 1].mean(), cpt_repair[:i + 1]


    def myopic_solver_tester(self, true_prices, epsilon, nb_min = 100, \
                             nb_max = 200, debug = False):
        """
        Test empirique de la méthode myopic_solver. Cette méthode calcule
        la séquence d'actions itérativement à l'aide de myopic_solver et
        réalise au plus nb_max repetitions d'un système tiré au hasard. À
        chaque observation globale, son résultat est tiré au hasard. Pour les
        paires "observation-réparation", on tire au hasard si la composante
        correspondante marche ou pas. Si oui, on ajoute juste le cout de
        l'observation et on continue, si non, on ajoute les couts d'observation
        et de réparation et on s'arrête (single fault assumption). Si on a
        une réparation simple sans observation associée, on ajoute directement
        le cout de réparation de la composante. Si après nb_min repétitions
        l'erreur estimée est plus petite que epsilon, on fait une sortie
        anticipée. La fonction calcule les couts empiriques de réparation, en
        utilisant pour cela true_prices.
        
        Args :
            true_prices : Dictionnaire de prix de réparation des composantes
                réparables
            epsilon : Tolerance de la moyenne
            nb_min : Entier, nombre minimum de répétitions à être realisées
            nb_max : Entier, nombre maximum de répétitions à être realisées
            debug (facultatif) : si True, affiche des messages montrant le 
                deroulement de l'algorithme.
                
        Returns :
            booléeen, True en cas de sortie anticipée, False sinon
            moyenne des couts observés
            tableau avec le nombre de composantes réparées à chaque répétition
            tableau avec le nombre d'observations globales faites à chaque
                répétition
        """
        costs = np.zeros(nb_max)
        cpt_obs = np.zeros(nb_max)
        cpt_repair = np.zeros(nb_max)
        
        sortie_anti = False
 
        # On fait au plus nb_max tests
        for i in tqdm(range(nb_max)):
            self.reset_bay_lp()
            while True:
                node, type_node = self.myopic_solver()
                if debug:
                    print("Noeud suggéré :", node)
                if type_node == "obs":
                    costs[i] += self.costs_obs[node]
                    obs_res = self.bay_lp.posterior(node).draw()
                    self.add_evidence(node, obs_res)
                    cpt_obs[i] += 1
                    if debug:
                        labels = self.bayesian_network.variable(node).labels()
                        print("Résultat de l'observation :", labels[obs_res])
                else:
                    # On fait l'observation du noeud si possible, sinon, on le 
                    # repare
                    if node in self.observation_nodes: 
                        costs[i] += self.costs_obs[node]
                    else:
                        costs[i] += true_prices[node]
                    cpt_repair[i] += 1
                    
                    # On effectue un pair observation-réparation
                    if node == self.service_node:
                        break
                    else:
                        # on récupere une probabilité:
                        # proba_sys = p(e = Normal| repair(node), Ei)
                        # = p(node != Normal | Ei)
                        self.add_evidence(self.problem_defining_node, "yes")
                        p = self.get_proba(node, "yes")
                        self.remove_evidence(self.problem_defining_node)
    
                        # On teste pour voir si le système marche 
                        if np.random.rand() <= p:
                            # Si le noeud était cassé et est observable, 
                            # on somme son cout de réparation
                            if node in self.observation_nodes: 
                                costs[i] += true_prices[node]
                            break
    
                        # Après une réparation, certaines observations peuvent
                        # être devenues obsolètes, il faut les exclure et
                        # permettre de les refaire.
                        obsoletes = self.observation_obsolete(node) 
                        self.add_evidence(node, "no")
                        for obs in obsoletes:
                            self.evidences.pop(obs)
                        self.reset_bay_lp(self.evidences)
            if i >= nb_min\
                and 1.96 * costs[:i + 1].std()/np.sqrt(i + 1) < epsilon:
                sortie_anti = True
                break
        self.reset_bay_lp()
        
        return sortie_anti, costs[:i + 1].mean(), cpt_repair[:i + 1],\
            cpt_obs[:i + 1]
    
    def elicitation_solver_tester(self, true_prices, epsilon, nb_min = 100, \
                             nb_max = 200, debug = False):
        """
        Test empirique de la résolution avec élicitation. À chaque fois qu'on
        doit prendre une action, on vérifie d'abord s'il y a des questions à
        répondre et, si oui, on les répond toutes correctement selon
        true_prices. Ensuite, la méthode calcule la séquence d'actions
        itérativement à l'aide de myopic_solver et réalise au plus nb_max
        repetitions d'un système tiré au hasard, le tirage au hasard étant
        identique à celui de myopic_solver_tester. Si après nb_min repétitions
        l'erreur estimée est plus petite que epsilon, on fait une sortie
        anticipée. La fonction calcule les couts empiriques de réparation, en
        utilisant pour cela true_prices.

        Args :
            true_prices : Dictionnaire de prix de réparation des composantes
                réparables
            epsilon : Tolerance de la moyenne
            nb_min : Entier, nombre minimum de répétitions à être realisées
            nb_max : Entier, nombre maximum de répétitions à être realisées
            debug (facultatif) : si True, affiche des messages montrant le 
                deroulement de l'algorithme.
                
        Returns :
            booléeen, True en cas de sortie anticipée, False sinon
            moyenne des couts observés
            tableau avec le nombre de composantes réparées à chaque répétition
            tableau avec le nombre d'observations globales faites à chaque
                répétition        
            tableau avec le nombre de questions répondues à chaque répétition
        """
        costs = np.zeros(nb_max)
        cpt_obs = np.zeros(nb_max)
        cpt_repair = np.zeros(nb_max)
        cpt_questions = np.zeros(nb_max)
        
        sortie_anti = False
        
        costs_rep_save = self.costs_rep.copy()
        costs_rep_interval_save = copy.deepcopy(self.costs_rep_interval)
 
        # On fait au plus nb_max tests
        for i in tqdm(range(nb_max)):
            self.reset_bay_lp()
            while True:
                has_ques = True
                while has_ques:
                    # On teste s'il y a une question disponible
                    eli_nd, val = self.best_EVOI()
                    if not np.allclose(0, val) and val > 0:
                        if debug:
                            print("Question sur :",eli_nd)
                        cpt_questions[i] += 1
                        # On doit répondre correctement à la question
                        lower = (true_prices[eli_nd] < self.costs_rep[eli_nd])
                        if debug:
                            print("Réponse :", lower)
                        self.elicitation(eli_nd, lower)
                    else:
                        has_ques = False
                
                node, type_node = self.myopic_solver()
                if debug:
                    print("Noeud suggéré :", node)
                if type_node == "obs":
                    costs[i] += self.costs_obs[node]
                    obs_res = self.bay_lp.posterior(node).draw()
                    self.add_evidence(node, obs_res)
                    cpt_obs[i] += 1
                    if debug:
                        labels = self.bayesian_network.variable(node).labels()
                        print("Résultat de l'observation :", labels[obs_res])
                else:
                    # On fait l'observation du noeud si possible, sinon, on le 
                    # repare
                    if node in self.observation_nodes: 
                        costs[i] += self.costs_obs[node]
                    else:
                        costs[i] += true_prices[node]
                    cpt_repair[i] += 1
                    
                    # On effectue un pair observation-réparation
                    if node == self.service_node:
                        break
                    else:
                        # on récupere une probabilité:
                        # proba_sys = p(e = Normal| repair(node), Ei)
                        # = p(node != Normal | Ei)
                        self.add_evidence(self.problem_defining_node, "yes")
                        p = self.get_proba(node, "yes")
                        self.remove_evidence(self.problem_defining_node)
    
                        # On teste pour voir si le système marche 
                        if np.random.rand() <= p:
                            # Si le noeud était cassé et est observable, 
                            # on somme son cout de réparation
                            if node in self.observation_nodes: 
                                costs[i] += true_prices[node]
                            break
    
                        # Après une réparation, certaines observations peuvent
                        # être devenues obsolètes, il faut les exclure et
                        # permettre de les refaire.
                        obsoletes = self.observation_obsolete(node) 
                        self.add_evidence(node, "no")
                        for obs in obsoletes:
                            self.evidences.pop(obs)
                        self.reset_bay_lp(self.evidences)
            self.costs_rep = costs_rep_save.copy()
            self.costs_rep_interval = copy.deepcopy(costs_rep_interval_save)
            if i >= nb_min\
                and 1.96 * costs[:i + 1].std()/np.sqrt(i + 1) < epsilon:
                sortie_anti = True
                break

        self.reset_bay_lp()
        
        return sortie_anti, costs[:i + 1].mean(), cpt_repair[:i + 1],\
            cpt_obs[:i + 1], cpt_questions[:i + 1]

# =============================================================================
# Calcul Exacte
# =============================================================================      

    def expected_cost_of_repair_seq_of_actions(self, seq):
        """
        Une fonction qui calcule un coût espéré de réparation à partir d'une séquence d'actions donnée.
        Ici on utilise une formule
        ECR = coût(C1) + (1 - P(e = Normal | C1 = Normal)) * coût(C2) +
              (1 - P(e = Normal | C1 = Normal, C2 = Normal)) * coût(C3) + ...

        Args :
            seq : une séquence d'actions de réparations dont le coût espéré il faut calculer, objet du type list<str>.

        Returns :
            ecr : le coût espéré de réparations d'une séquence soumise, objet du type float.
        """

        ecr = 0.0
        prob = 1.0
        self.reset_bay_lp()

        # Parcours par tous les compasants à réparer
        for node in seq:
            # On ajoute un terme à ECR
            ecr += self.costs_rep[node] * prob
            # On propage dans notre réseau une évidence que C_next = Normal
            if node == self.service_node:
                self.bay_lp.chgEvidence(node, "yes")
            else:
                self.bay_lp.chgEvidence(node, "no")
            pot = self.bay_lp.posterior(self.problem_defining_node)
            inst = gum.Instantiation(pot)
            inst.chgVal(self.problem_defining_node, "no")
            # On met-à-jour une proba
            prob = (1 - pot.get(inst))

        self.reset_bay_lp()

        return ecr

    def brute_force_solver_actions_only(self, debug=False):
        """
        Une fonction qui cherche une séquence optimale de réparation par une recherche exhaustive en choisissant
        une séquence de meilleur ECR. Pour le cas où on ne considère que les actions de réparation il suffit de
        dénombrer toutes les permutations possibles d'un ensemble des actions admissibles.

        Args :
            debug (facultatif) : un flag qui indique s'il faut afficher des séquences intermédiaires ou pas,
                objet du type bool.

        Returns :
            min_seq : une séquence optimale trouvée dont le coût est le plus petit que possible ;
                objet du type list<str>.
            min_ecr : le côut espéré de réparation correspondant à min_seq, objet du type float.
        """
        min_seq = [self.service_node] + list(self.repairable_nodes).copy()
        min_ecr = self.expected_cost_of_repair_seq_of_actions(min_seq)

        # Parcours par toutes les permutations de l'union de noeuds réparables avec un noeud de service
        for seq in [list(t) for t in permutations(list(self.repairable_nodes) + [self.service_node])]:
            ecr = self.expected_cost_of_repair_seq_of_actions(seq)
            if debug:
                print("seq : {0}\necr : {1}\n\n".format(seq, ecr))
            # Si on trouve une séquence meilleure que celle courante on la sauvegarde
            if ecr < min_ecr:
                min_ecr = ecr
                min_seq = seq.copy()

        return min_seq, min_ecr

    def expected_cost_of_repair(self, strategy_tree, obs_rep_couples=False):
        """
        Une méthode qui calcule le coût espéré de réparation étant donné un arbre de décision.

        Args :
            strategy_tree : un arbre de stratégie dont le côut il faut calculer, objet du type StrategyTree.
            obs_rep_couples (facultatif) : une variable boléenne qui indique si on suppose l'existance de couples
            ``observation-réparation'' dans un arbre de stratégie soumis, objet du type bool.

        Returns :
            ecr : le côut espéré de réparation d'un arbre de stratégie fourni, objet du type float.
        """
        ecr = self._expected_cost_of_repair_internal(strategy_tree, obs_rep_couples=obs_rep_couples)
        self.bay_lp.setEvidence({})
        self._start_bay_lp()
        self.reset_bay_lp()
        return ecr

    def _expected_cost_of_repair_internal(self, strategy_tree, evid_init=None, obs_rep_couples=False):
        """
        Une partie récursive d'une fonction expected_cost_of_repair au-dessus

        Args :
            strategy_tree : un arbre de stratégie dont le côut il faut calculer, objet du type StrategyTree.
            evid_init (facultatif) : un dictionnaire d'évidences utilisé dans les appels récursifs mais vide par
                défaut, objet du type dict<str, str>.
            obs_rep_couples (facultatif): une variable boléenne qui indique si on suppose l'existance de couples
                ``observation-réparation'' dans un arbre de stratégie soumis, objet du type bool.

        Returns :
            ecr : le côut espéré de réparation d'un arbre de stratégie fourni, objet du type float.
        """
        if not isinstance(strategy_tree, st.StrategyTree):
            raise TypeError('strategy_tree must have type StrategyTre')

        # On ajoute dans ECR un terme lié à la racine d'un arbre donné
        ecr = 0.0
        evidence = evid_init if isinstance(evid_init, dict) else {}
        self.bay_lp.setEvidence(evidence)
        prob = 1.0 if evidence == {} else (1 - self.prob_val(self.problem_defining_node, 'no'))
        node = strategy_tree.get_root()
        node_name = strategy_tree.get_root().get_name()
        aelem = 0.0
        if obs_rep_couples and node_name in self.repairable_nodes.intersection(self.observation_nodes):
            cost = self.costs_obs[node_name]
            self.bay_lp.setEvidence(merge_dicts(evidence, {node_name: 'yes'}))
            prob_next = 1 - self.prob_val(self.problem_defining_node, 'no')
            self.bay_lp.setEvidence(evidence)
            aelem = prob_next * self.prob_val(node_name, 'yes') * self.costs_rep[node_name]
        else:
            cost = self.costs_rep[node_name] if isinstance(node, st.Repair) else self.costs_obs[node_name]
        ecr += prob * cost + aelem

        # On relance récursivement cette fonction-là pour chaque sous-arbre qui a un enfant (s'il en existe)
        # d'une racine courante pour sa propre racine
        if len(node.get_list_of_children()) == 0:
            return ecr
        if isinstance(node, st.Repair):
            # Généralement un enfant pour une action de réparation puisque on les suppose parfaites ...
            ecr += self._expected_cost_of_repair_internal(
                strategy_tree.get_sub_tree(node.get_child()),
                merge_dicts(evidence, {node_name: 'yes' if node_name == self.service_node else 'no'}), obs_rep_couples)
            self.bay_lp.setEvidence(evidence)
        else:
            # ... et plusieurs enfants pour des observations
            for obs_label in self.bayesian_network.variable(node_name).labels():
                child = node.bn_labels_children_association()[obs_label]
                new_evidence = (merge_dicts(evidence, {node_name: obs_label})
                                if evidence.get(node_name) is None else evidence.copy())
                ecr += (self.prob_val(node_name, obs_label) *
                        self._expected_cost_of_repair_internal(strategy_tree.get_sub_tree(child), new_evidence,
                                                               obs_rep_couples)
                        )
                self.bay_lp.setEvidence(evidence)

        return ecr

    # En fait, la même méthode que get_proba au-dessus
    # On a donc implémenté de manière indépendante deux méthodes similaires :)
    def prob_val(self, var, value):
        """
        Une méthode auxiliare qui retourne une probabilité à posteriori qu'une variable var égal à value sachant une
        évidence courante de notre réseau bayésien (cette évidence doit être mise en place en dehors de cette fonction).

        Args :
            var : un label d'une variable aléatoire, objet du type str.
            value : une valeur d'une variable aléatoire, objet du type str.

        Returns :
            prob : une probabilité que var égal à value sachant une évidence courante de notre réseau bayésien, ie
                prob = P(var = value | E), où E est une évidence courante.
        """
        pot_var = self.bay_lp.posterior(var)
        inst_var = gum.Instantiation(pot_var)
        inst_var.chgVal(var, value)
        return pot_var[inst_var]

    def _create_nodes(self, names, rep_string='_repair', obs_string='_observation', obs_rep_couples=False):
        """
        Une méthode auxiliaire qui crée des noeuds de StrategyTree à partir de leurs noms dans un modèle.

        Args :
            names : des noms des réparations/observations/observations-réparations dans un modèle à partir desquels
                il faut créer des noeuds ; objet du type list<str>.
            rep_string (facultatif) : dans le cas où on ne considère pas des couples, on utilise ce paramètre comme
                un suffixe pour les noeuds de réparation pour les séparer de ceux d'observation, objet du type str.
            obs_string (facultatif) : un suffixe pour les noeuds d'observation, objet du type str.
            obs_rep_couples (facultatif): une variable boléenne qui indique si on suppose l'existance de couples
                ``observation-réparation'' dans un arbre de stratégie soumis, objet du type bool.

        Returns :
            nodes : une liste de noeuds qu'il faut créer, objet du type list<NodeST>
        """
        nodes = []
        temp = self.observation_nodes.intersection(self.repairable_nodes)
        for name, i in zip(names, range(len(names))):
            if obs_rep_couples and name in self.observation_nodes.intersection(self.repairable_nodes):
                node = st.Repair(str(i), -1, name, obs_rep_couples=obs_rep_couples)
            elif name.endswith(rep_string) or name == self.service_node:
                node = st.Repair(str(i), self.costs_rep[name.replace(rep_string, '')], name.replace(rep_string, ''))
            else:
                node = st.Observation(str(i), self.costs_obs[name.replace(obs_string, '')],
                                      name.replace(obs_string, ''))
            nodes.append(node)
            self._nodes_ids_db_brute_force.append(str(i))
        return nodes

    # Une méthode qui permet d'obtenir une prochaine valeur de id pour un noeud de StrategyTree
    def _next_node_id(self):
        next_id = str(int(self._nodes_ids_db_brute_force[-1]) + 1)
        self._nodes_ids_db_brute_force.append(next_id)
        return next_id

    def brute_force_solver(self, debug=False, mode='all', obs_rep_couples=False):
        if debug is False:
            debug = (False, False)
        elif debug is True:
            debug = (True, True)
        rep_string, obs_string = '_repair', '_observation'
        rep_nodes = {n + ('' if obs_rep_couples and n in self.observation_nodes else rep_string)
                     for n in self.repairable_nodes}
        obs_nodes = {n + ('' if obs_rep_couples and n in self.repairable_nodes else obs_string)
                     for n in self.observation_nodes}
        feasible_nodes_names = {self.service_node}.union(rep_nodes).union(obs_nodes)
        self._nodes_ids_db_brute_force = []
        feasible_nodes = self._create_nodes(feasible_nodes_names, rep_string, obs_string, obs_rep_couples)

        call_service_node = st.Repair('0', self.costs_rep[self.service_node], self.service_node)
        call_service_tree = st.StrategyTree(call_service_node, [call_service_node])
        self._best_st = call_service_tree
        self._best_ecr = self.expected_cost_of_repair(call_service_tree, obs_rep_couples)

        if mode == 'dp':
            self._best_st, self._best_ecr = self.dynamic_programming_solver(
                feasible_nodes, debug_iter=debug[0], debug_st=debug[1], obs_rep_couples=obs_rep_couples)
        elif mode == 'all':
            self._evaluate_all_st(feasible_nodes, debug_iter=debug[0], debug_st=debug[1],
                                  obs_rep_couples=obs_rep_couples)

        self._nodes_ids_db_brute_force = []

        return self._best_st, self._best_ecr

    def _evaluate_all_st(self, feasible_nodes, obs_next_nodes=None, parent=None, fn_immutable=None, debug_nb_call=0,
                         debug_iter=False, debug_st=False, obs_rep_couples=False):
        if parent is not None:
            par_tmp = parent[-1][0]
        parent_c = shallow_copy_parent(parent)
        fn_immutable_c = shallow_copy_list_of_copyable(fn_immutable)
        if len(feasible_nodes) > 0:
            for i in range(len(feasible_nodes)):
                node = feasible_nodes[i]
                obs_next_nodes_tmp = shallow_copy_list_of_copyable(obs_next_nodes) \
                    if obs_next_nodes is not None else None
                if parent_c is None or debug_nb_call == 0:
                    if debug_iter:
                        print(
                            f'{bcolors.OKGREEN}\n########################\nIter %d\n########################\n{bcolors.ENDC}' % i)
                    par_mutable = node.copy()
                    par_immutable = node.copy() if isinstance(node, st.Observation) else None

                    par_tree_mutable = st.StrategyTree(node)
                    onn = ([[l for l in self.bayesian_network.variable(node.get_name()).labels()]]
                           if isinstance(node, st.Observation) else None)
                    par = [(par_mutable, par_immutable, par_tree_mutable.copy())]
                    fn_im = []
                    fn_im.append([n.copy() for n in (feasible_nodes[:i] + feasible_nodes[i + 1:])])
                    self._evaluate_all_st(feasible_nodes[:i] + feasible_nodes[i + 1:],
                                          shallow_copy_list_of_copyable(onn) if onn is not None else None, par, fn_im,
                                          debug_nb_call + 1, debug_iter, debug_st, obs_rep_couples)
                else:
                    par_mutable, par_immutable, par_tree_mutable = parent_c[-1]
                    branch_attr = obs_next_nodes_tmp[-1][0] if obs_next_nodes_tmp is not None else None
                    par_tmp = par_tree_mutable.get_node(par_tmp)
                    par_tmp_ch = par_tmp.get_child_by_attribute(branch_attr)
                    sth_removed = par_tree_mutable.remove_sub_tree(
                        par_tree_mutable.get_node(par_tmp_ch.get_id() if par_tmp_ch is not None else None))
                    if sth_removed:
                        par_mutable = par_tmp
                    try:
                        par_tree_mutable.add_node(node)
                        par_tree_mutable.add_edge(par_mutable, node, branch_attr)
                    except ValueError:
                        node = node.copy()
                        node.set_id(self._next_node_id())
                        par_tree_mutable.add_node(node)
                        par_tree_mutable.add_edge(par_mutable, node, branch_attr)
                    par_mutable = node.copy()
                    if isinstance(node, st.Repair):
                        parent_c[-1] = (par_mutable, par_immutable, par_tree_mutable.copy())
                        self._evaluate_all_st(feasible_nodes[:i] + feasible_nodes[i + 1:],
                                              shallow_copy_list_of_copyable(obs_next_nodes_tmp)
                                              if obs_next_nodes_tmp is not None else None,
                                              parent_c, fn_immutable_c, debug_nb_call + 1, debug_iter, debug_st,
                                              obs_rep_couples)
                    elif isinstance(node, st.Observation):
                        fn_im = fn_immutable_c.copy()
                        appended_parent, appended_obs = False, False
                        if len(feasible_nodes) != 1:
                            appended_obs = True
                            par_immutable = node.copy()
                            if len(parent_c) == 1 and parent_c[0][1] is None:
                                parent_c[0] = (par_mutable, par_immutable, par_tree_mutable.copy())
                            else:
                                parent_c.append((par_mutable, par_immutable, par_tree_mutable.copy()))
                                appended_parent = True
                            branch_attrs = [l for l in self.bayesian_network.variable(node.get_name()).labels()]
                            obs_next_nodes_tmp = [branch_attrs] \
                                if obs_next_nodes_tmp is None else obs_next_nodes_tmp + [branch_attrs]
                            fn_im.append([n.copy() for n in (feasible_nodes[:i] + feasible_nodes[i + 1:])])
                        self._evaluate_all_st(feasible_nodes[:i] + feasible_nodes[i + 1:],
                                              shallow_copy_list_of_copyable(obs_next_nodes_tmp)
                                              if obs_next_nodes_tmp is not None else None,
                                              parent_c, fn_im, debug_nb_call + 1, debug_iter, debug_st, obs_rep_couples)
                        if appended_parent:
                            parent_c.pop()
                        if appended_obs:
                            obs_next_nodes_tmp.pop()
                            obs_next_nodes_tmp = obs_next_nodes_tmp if len(obs_next_nodes_tmp) > 0 else None
        if len(feasible_nodes) == 1:
            complete_tree = False
            ecr = inf
            strat_tree = None
            if parent_c is None:
                complete_tree = True
                strat_tree = st.StrategyTree(root=feasible_nodes[0].copy())
                if debug_st:
                    print(strat_tree)
                ecr = self.expected_cost_of_repair(strat_tree, obs_rep_couples)
            else:
                par_mutable, par_immutable, par_tree_mutable = parent_c[-1]
                if obs_next_nodes_tmp is None:
                    complete_tree = True
                    strat_tree = par_tree_mutable.copy()
                    if debug_st:
                        print(strat_tree)
                    ecr = self.expected_cost_of_repair(strat_tree, obs_rep_couples)
                else:
                    if len(obs_next_nodes_tmp[-1]) != 1:
                        par_mutable = par_immutable.copy()
                        parent_c[-1] = (par_mutable, par_immutable, par_tree_mutable.copy())
                        obs_next_nodes_tmp[-1].pop(0)
                        self._evaluate_all_st(fn_immutable_c[-1],
                                              shallow_copy_list_of_copyable(obs_next_nodes_tmp)
                                              if obs_next_nodes_tmp is not None else None,
                                              parent_c, fn_immutable_c, debug_nb_call + 1, debug_iter, debug_st,
                                              obs_rep_couples)
                    else:
                        if len(parent_c) != 1 and any([len(lbls) > 1 for lbls in obs_next_nodes_tmp]):
                            while len(obs_next_nodes_tmp[-1]) == 1:
                                lp = parent_c.pop()
                                obs_next_nodes_tmp.pop()
                                fn_immutable_c.pop()
                                parent_c[-1] = (parent_c[-1][1].copy(), parent_c[-1][1], lp[2].copy())
                            obs_next_nodes_tmp[-1].pop(0)
                            self._evaluate_all_st(fn_immutable_c[-1],
                                                  shallow_copy_list_of_copyable(obs_next_nodes_tmp)
                                                  if obs_next_nodes_tmp is not None else None,
                                                  parent_c, fn_immutable_c, debug_nb_call + 1, debug_iter, debug_st,
                                                  obs_rep_couples)
                        else:
                            complete_tree = True
                            strat_tree = par_tree_mutable.copy()
                            if debug_st:
                                print(strat_tree)
                            ecr = self.expected_cost_of_repair(strat_tree, obs_rep_couples)
            if complete_tree and ecr < self._best_ecr:
                self._best_st = strat_tree
                self._best_ecr = ecr
        elif len(feasible_nodes) == 0:
            return

    def dynamic_programming_solver(self, feasible_nodes, evidence=None, debug_iter=False, debug_st=False,
                                   obs_rep_couples=False):
        if len(feasible_nodes) == 0:
            return None, 0.0
        if evidence is None:
            evidence = {}

        self.bay_lp.setEvidence(evidence)
        prob_ev = 1.0 if len(evidence.keys()) == 0 else (1 - self.prob_val(self.problem_defining_node, 'no'))

        call_service_node = st.Repair(self._next_node_id(), self.costs_rep[self.service_node], self.service_node)
        best_tree = st.StrategyTree(call_service_node, [call_service_node])
        best_ecr = self.costs_rep[self.service_node]

        for i in range(len(feasible_nodes)):
            if debug_iter and len(evidence.keys()) == 0:
                print(f'{bcolors.OKGREEN}\n########################\nIter %d\n########################\n{bcolors.ENDC}'
                      % i)
            node = feasible_nodes[i].copy()
            node.set_id(self._next_node_id())
            strat_tree = st.StrategyTree(root=node)
            aelem = 0.0
            # cost = node.get_cost()
            if obs_rep_couples and node.get_name() in self.repairable_nodes.intersection(self.observation_nodes):
                cost = self.costs_obs[node.get_name()]
                # + self.prob_val(node.get_name(), 'yes') * self.costs_rep[node.get_name()]
                self.bay_lp.setEvidence(merge_dicts(evidence, {node.get_name(): 'yes'}))
                prob_next = 1 - self.prob_val(self.problem_defining_node, 'no')
                self.bay_lp.setEvidence(evidence)
                aelem = prob_next * self.prob_val(node.get_name(), 'yes') * self.costs_rep[node.get_name()]
            else:
                cost = (
                    self.costs_rep[node.get_name()]
                    if isinstance(node, st.Repair)
                    else self.costs_obs[node.get_name()]
                )
            ecr = prob_ev * cost + aelem
            for label in (self.bayesian_network.variable(node.get_name()).labels()
            if isinstance(node, st.Observation)
            else (['yes'] if node.get_name() == self.service_node else ['no'])):
                if isinstance(node, st.Observation) and evidence.get(node.get_name()) is not None:
                    new_evidence = evidence.copy()
                else:
                    new_evidence = merge_dicts(evidence, {node.get_name(): label})
                best_sub_tree, best_sub_ecr = self.dynamic_programming_solver(
                    feasible_nodes[:i] + feasible_nodes[i + 1:], new_evidence, debug_iter, debug_st, obs_rep_couples
                )
                self.bay_lp.setEvidence(evidence)
                if best_sub_tree is not None:
                    strat_tree = best_sub_tree.connect(strat_tree, label)
                else:
                    strat_tree.remove_sub_tree(strat_tree.get_root().get_child_by_attribute(label))
                if isinstance(node, st.Observation):
                    ecr += self.prob_val(node.get_name(), label) * best_sub_ecr
                elif isinstance(node, st.Repair):
                    ecr += best_sub_ecr
            if ecr < best_ecr:
                best_tree = strat_tree.copy()
                best_ecr = ecr

        if len(evidence.keys()) == 0:
            self.bay_lp.setEvidence({})
            self._start_bay_lp()
            self.reset_bay_lp()

        if debug_st:
            print(best_tree)
            print(best_ecr)

        return best_tree, best_ecr

# =============================================================================
# Méthodes pas encore fonctionnelles            
# =============================================================================

    # Une méthode qui implémente un algorithme le plus simple de résolution du problème de Troubleshooting
    def solve_static(self):
        rep_nodes = list(self.repairable_nodes.copy())
        rep_seq = []
        # L'efficacité de l'appel à service
        service_ef = 1.0 / self.costs_rep[self.service_node]
        ie = gum.LazyPropagation(self.bayesian_network)
        # Tant qu'on n'a pas encore observer tous les noeuds réparables
        while len(rep_nodes) > 0:
            # On suppose par défaut que l'appel à service est une action la plus efficace
            # on cherche ensuite une action plus efficace
            action_to_put = self.service_node
            ef = service_ef
            # On observe tous les actions pas encore observés
            for rnode in range(len(rep_nodes)):
                if rnode != 0:
                    ie.eraseEvidence(rep_nodes[rnode-1])
                ie.addEvidence(rep_nodes[rnode], "no")
                # On vérifie si une action courante est plus efficace que le dernier  max par efficacité
                #print(ie.posterior(self.problem_defining_node).tolist()[0])
                if ef < ie.posterior(self.problem_defining_node).tolist()[0] / self.costs_rep[rep_nodes[rnode]]:
                    ef = ie.posterior(self.problem_defining_node).tolist()[0] / self.costs_rep[rep_nodes[rnode]]
                    #print(ef)
                    action_to_put = rep_nodes[rnode]
            rep_seq.append(action_to_put)
            # Si on a trouvé quelque part qu'un appel au service est plus efficace que toutes les actions possibles,
            # on ajoute donc 'service' dans une séquence de réparation et on s'arrête car cet appel réparera un appareil
            # avec un certain
            if action_to_put == self.service_node:
                return rep_seq, self.expected_cost_of_repair_seq_of_actions(rep_seq)
            ie.eraseEvidence(rep_nodes[-1])
            ie.addEvidence(action_to_put, "no")
            # on met-à-jour les noeuds réparables
            rep_nodes.remove(action_to_put)
        rep_seq.append(self.service_node)
        return rep_seq, self.expected_cost_of_repair_seq_of_actions(rep_seq)
