import pyAgrum as gum
from itertools import permutations

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
        costs_obs : Dictionnaire de coûts où les clés représentent les noeuds 
            du BN et les valeurs leurs coûts d'observation (float).  
        repairable_nodes : Liste de noeuds qui correspondent aux éléments du 
            système concerné qui peuvent être réparés.
        unrepairable_nodes : Liste de noeuds qui correspondent aux éléments 
            d'un système qui ne peuvent pas être réparés.
        problem_defining_node : Noeud qui répresent le problème a être reglé
            (système fonctionnel où pas).
        observation_nodes : Liste de noeuds qui correspondent aux éléments du 
            système qui peuvent être observés.
        service_node : Noeud qui répresent l'appel au service (appel à la 
            réparation sûre du système).
    """
    
    # Constructeur
    def __init__(self, bayesian_network, costs, nodes_types):
        """
        Crée un objet du type TroubleShootingProblem.
        Initialise bay_lp et ajoute des inférences vides aux noeuds du BN qui 
        peuvent être modifiés (réparés/appelés). 
        
        Args :
            bayesian_network : Objet du type pyAgrum.BayesNet qui répresent le 
                réseau bayésien (BN) modélisant un problème donné.
            costs : Liste avec deux dictionnaires, le premier avec les coûts de 
                réparation et le deuxième avec les coûts d'observation des 
                noeuds.
            nodes_types : Dictionnaire où les clés représent les noeuds du BN 
                    et les valeurs leurs types associés (set de string).       
        """
        self.bayesian_network = gum.BayesNet(bayesian_network)
        self.bay_lp = gum.LazyPropagation(self.bayesian_network)
        self.costs_rep = costs[0].copy()
        self.costs_obs = costs[1].copy()
        self.nodes_types = nodes_types.copy()
        self.repairable_nodes = [node for node in nodes_types.keys() \
                                if 'repairable' in nodes_types[node]]
        self.unrepairable_nodes = [node for node in nodes_types.keys() \
                                if 'unrepairable' in  nodes_types[node]]
        self.problem_defining_node = [node for node in nodes_types.keys() \
                                if 'problem-defining' in nodes_types[node]][0]
        self.observation_nodes = [node for node in nodes_types.keys() \
                                if 'observable' in nodes_types[node]]
        self.service_node = [node for node in nodes_types.keys() \
                                if 'service' in nodes_types[node]][0]
        self.start_bay_lp()

# =============================================================================
# Méthodes fonctionnelles
# =============================================================================

    def start_bay_lp(self):
        """
        Ajoute des inférences vides aux noeuds du BN qui peuvent être modifiés 
        (réparés/appelés).Ces évidences ne changent pas les probabilités, elles
        servent pour qu'on puisse utiliser la méthode chgEvidence de pyAgrum 
        à la suite.
        """
        for node in self.repairable_nodes + [self.service_node]:
            self.bay_lp.addEvidence(node, [1, 1]) 
            
    def reset_bay_lp(self, dict_inf = {}):
        """
        Reinitialise les inférences des noeuds du BN qui peuvent être modifiés 
        (réparés/appelés). Pour les noeuds dans dict_inf, l'inférence est mis à
        la valeur associé au noeud dans dict_inf, pour les autres l'inférence 
        est mis à [1, 1].
        
        Args :
            dict_inf (facultatif) : Dictionnaire où les clés sont des noeuds et
                les valeurs sont des inférences. 
        """
        for node in self.repairable_nodes + [self.service_node]:
            if node in dict_inf:
                self.bay_lp.chgEvidence(node, dict_inf[node])
            else:
                self.bay_lp.chgEvidence(node, [1, 1])

    def simple_solver(self, debug = False):
        """
        Solveur simple pour le TroubleShooting problem.
        On ne prend pas en considèration les observations et on ne révise pas 
        les probabilités, c'est-à-dire on ne met pas à jour les probabilités 
        si on répare une composante.
        À cause de cela, ce solveur n'est pas iteractive et renvoie l'ordre de 
        réparation entière (jusqu'au appel au service).
        
        Args : 
            debug (facultatif) : si True, affiche des messages montrant le 
                de l'algorithme.
        Returns :
            Un tuple avec la séquence des noeuds à être réparés dans l'ordre et
            l'esperance du coût de réparation de cette séquence.
        """     
        # On crée un dictionnaire avec les efficacités liées à chaque noeud 
        # réparable + service
        dic_eff = {}
        
        # Pour chaque noeud réparable + service (et qui n'est pas irréparable):
        for node in set(self.repairable_nodes + [self.service_node]) \
        - set(self.unrepairable_nodes):
            # On actualise l'évidence liée au noeud en traitement: 
            # On dit que le noeud courrant n'est pas cassé 
            # (On considere qu'il a été réparé)
            if node != "callService":
                self.bay_lp.chgEvidence(node, "no")
            else:
                self.bay_lp.chgEvidence(node, "yes")
            
            # On calcule l'efficacité du noeud:
            # p(e = Normal|repair(noeud)) / coût(repair(noeud))
            p = self.bay_lp.posterior(self.problem_defining_node)
            
            # On utilise une instantion pour récuperer la bonne probabilité 
            # gardé en p sans dépendre d'un indice
            inst = gum.Instantiation(p)
            inst.chgVal(self.problem_defining_node, "no")
            dic_eff[node] = p[inst] / self.costs_rep[node] 
            if debug == True:
                print("noeud consideré : " + node)
                print("proba p(e = Normal|repair(noeud)) : ", p[inst])            
                print("éfficacité du noeud : ", dic_eff[node])
                print()
            
            # On retourne l'évidence du noeud à celle qui ne change pas les 
            # probabilité du départ
            self.bay_lp.chgEvidence(node, [1, 1])
        
        # On sort les noeuds par rapport aux efficacités
        rep_seq = sorted(dic_eff.items(), key = lambda x: x[1], reverse = True)           
        # On veut que les noeuds, pas les valeurs des efficacités
        rep_seq = [r[0] for r in rep_seq]
        # On renvoie la liste jusqu'au appel au service, pas plus
        rep_seq = rep_seq[:rep_seq.index("callService") + 1]

        # On calcule le coût espéré de la sequence de réparation
        # On commence par le cût de réparation du prémier noeud de la séquence
        exp_cost = self.costs_rep[rep_seq[0]]
        # Le premier répare n'a pas résolu le problème, on change l'évidence
        # du noeud pour réfletir cela
        self.bay_lp.chgEvidence(rep_seq[0], "no")            
  
        if debug == True:
            print("Calcul de l'esperance de coût \n")
            print ("premier noeud réparé : ", rep_seq[0])
            print("esperance partiel du coût de la séquence : ", exp_cost)

        for node in rep_seq[1:]:
            # On récupere la probabilité que le problème persiste après la 
            # réparations des noeuds déjà pris en considération
            # p = p(e != Normal|repair(tous les noeuds déjà considerés)) 
            p = self.bay_lp.posterior(self.problem_defining_node)
            # On utilise une instantion pour récuperer la bonne probabilité 
            # gardé en p sans dépendre d'un indice
            inst = gum.Instantiation(p)
            inst.chgVal(self.problem_defining_node, "yes")
            # On somme le coût de réparation du noeud courant * p
            exp_cost += self.costs_rep[node] * p[inst]
            if debug == True:
                print("proba p(e != Normal|réparation des toutes les noeuds " \
                               + "déjà considerés) : ", p[inst])     
                print()
                print("noeud réparé : ", node)
                print("esperance partiel du coût de la séquence : ", exp_cost)
            # On actualise l'évidence du noeud concerné
            self.bay_lp.chgEvidence(node, "no")

        self.reset_bay_lp()
        
        return rep_seq, exp_cost

    # Une fonction qui calcule un coût espéré de réparation à partir d'une séquence d'actions donnée
    # Ici on utilise une formule
    # ECR = coût(C1) +
    #       (1 - P(e = Normal | C1 = Normal)) * coût(C2) +
    #       (1 - P(e = Normal | C1 = Normal, C2 = Normal)) * coût(C3) + ...
    def estimated_repair_cost_seq_of_actions(self, seq):

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

    # Une fonction qui cherche une séquence optimale de réparation par une recherche exhaustive en choisissant
    # une séquence de meilleur ECR
    def brute_force_solver(self, debug=False):
        min_seq = [self.service_node] + self.repairable_nodes.copy()
        min_ecr = self.estimated_repair_cost_seq_of_actions(min_seq)

        # Parcours par toutes les permutations de l'union de noeuds réparables avec un noeud de service
        for seq in [list(t) for t in permutations(self.repairable_nodes + [self.service_node])]:
            ecr = self.estimated_repair_cost_seq_of_actions(seq)
            if ecr < min_ecr:
                min_ecr = ecr
                min_seq = seq.copy()

        return min_seq, min_ecr
    
# =============================================================================
# Méthodes pas encore fonctionnelles            
# =============================================================================
            
    def solveur_simple_iter(self, debug = False):
        """
        """
        rep_seq = []
        repares = {}
        unrep_nodes = set(self.unrepairable_nodes.copy())
        
        #on itère jusqu'à ce qu'il n'existe plus de noeud que peut être reparé
        reparables = list(set(self.repairable_nodes + [self.service_node]) \
        - unrep_nodes)
        while reparables != []:  
            if debug == True:
                print("Les noeuds considerés dans ce tour de boucle : ",\
                      reparables)
            # On crée un dictionnaire avec les efficacités liées à chaque noeud 
            # réparable + service dans ce tour de boucle
            dic_eff = {}            
            # Pour chaque noeud réparable + service (et qui n'est pas 
            # irréparable):
            for node in reparables:   
                # On calcule la probabilité que le noeud actuel soit cassé                              
                # p(node != Normal|Ei)
                # Ei = les informations connues au tour de boucle actuel
                p_noeud_casse = self.bay_lp.posterior(node)        
                inst_noeud_casse = gum.Instantiation(p_noeud_casse)
                inst_noeud_casse.chgVal(node, "yes")
                
                # On calcule le coût esperé du pair observation-repair pour le
                # noeud actuel
                if node in self.observation_nodes:
                    cost = self.costs_obs[node] + \
                    p_noeud_casse[inst_noeud_casse] * self.costs_rep[node]
                else:
                    cost = self.costs_rep[node]
                # On actualise l'évidence liée au noeud en traitement: 
                # On dit que le noeud courrant n'est pas cassé 
                # (On considere qu'il a été réparé)
                if node != "callService":
                    self.bay_lp.chgEvidence(node, "no")
                else:
                    self.bay_lp.chgEvidence(node, "yes")          
                # On calcule l'efficacité du noeud:
                # p(e = Normal|repair(node), Ei) / cost(node)
                p = self.bay_lp.posterior(self.problem_defining_node)                
                inst = gum.Instantiation(p)
                inst.chgVal(self.problem_defining_node, "no")    
                dic_eff[node] = p[inst] / cost
                
                # On retourne l'évidence du noeud à celle qui ne change pas les 
                # probabilités du départ du tour de boucle actuel
                self.bay_lp.chgEvidence(node, [1, 1])
                
                if debug == True:
                    print("noeud consideré : " + node)
                    print("proba p(node != Normal|Ei) : ",\
                          p_noeud_casse[inst_noeud_casse])
                    print("proba p(e = Normal|repair(node), Ei) : ", p[inst]) 
                    print ("coût esperé du pair observation-repair : ", cost)
                    print("éfficacité du noeud : ", dic_eff[node])
                    print()
            
            # On sort les noeuds par rapport aux efficacités
            seq = sorted(dic_eff.items(), key = lambda x: x[1], \
                             reverse = True) 
            # Le noeud choisi est ce avec la meilleure efficacité dans le tour
            # de boucle actuel
            chosen_node = seq[0][0]
            rep_seq.append(chosen_node)
            if debug == True:
                print("noeud choisi dans ce tour de boucle : ", chosen_node)
            
            # On garde ce noeud au dictionnaire repares pour qu'on puisse
            # mantenir le reseau à jour a chaque tour de la boucle while 
            if chosen_node != "callService":
                repares[chosen_node] =  "no"
                unrep_nodes.add(chosen_node)
                reparables = list(set(self.repairable_nodes + \
                                      [self.service_node]) - unrep_nodes)
                self.reset_bay_lp(repares)
            else:
                break
        # On retourne aux évidences du début
        self.reset_bay_lp()           
        return rep_seq

    # Une méthode qui implémente un algorithme le plus simple de résolution du problème de Troubleshooting
    def solve_static(self):
        rep_nodes = self.repairable_nodes.copy()
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
                return rep_seq, self.estimated_repair_cost_seq_of_actions(rep_seq)
            ie.eraseEvidence(rep_nodes[-1])
            ie.addEvidence(action_to_put, "no")
            # on met-à-jour les noeuds réparables
            rep_nodes.remove(action_to_put)
        rep_seq.append(self.service_node)
        return rep_seq, self.estimated_repair_cost_seq_of_actions(rep_seq)
