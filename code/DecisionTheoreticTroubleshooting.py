import pyAgrum as gum
import StrategyTree as st
from itertools import permutations
from numpy import inf
from copy import deepcopy


def shallow_copy_list_of_copyable(l):
    if l is None:
        return None
    return [subl.copy() for subl in l]


def shallow_copy_parent(parent):
    if parent is None:
        return None
    return [tuple([elem.copy() if elem is not None else None for elem in par]) for par in parent]


def merge_dicts(left, right):
    res = left.copy()
    res.update(right)
    return res


class bcolors:
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
        self.start_bay_lp()
        self._nodes_ids_db_brute_force = []
        self._best_st = None
        self._best_ecr = None

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
        modifiable_nodes = self.repairable_nodes.union(self.observation_nodes)
        modifiable_nodes.add(self.service_node)
        modifiable_nodes.add(self.problem_defining_node)
        for node in modifiable_nodes:
            self.bay_lp.addEvidence(node, [1, 1]) 
        self.evidences = {}
            
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
        modifiable_nodes = self.repairable_nodes.union(self.observation_nodes)
        modifiable_nodes.add(self.service_node)
        modifiable_nodes.add(self.problem_defining_node)
        for node in modifiable_nodes:
            if node in dict_inf:
                self.change_evidence(node, dict_inf[node])
            else:
                self.bay_lp.chgEvidence(node, [1, 1])
                self.evidences.pop(node, None)

    def change_evidence(self, node, evidence):
        """
        """
        self.bay_lp.chgEvidence(node, evidence)
        self.evidences[node] = evidence

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
                deroulement de l'algorithme.
        Returns :
            Un tuple avec la séquence des noeuds à être réparés dans l'ordre et
            l'esperance du coût de réparation de cette séquence.
        """     
        # On crée un dictionnaire avec les efficacités liées à chaque noeud 
        # réparable + service
        dic_eff = {}
        
        # Pour chaque noeud réparable + service (et qui n'est pas irréparable):
        for node in (self.repairable_nodes|{self.service_node}) - \
        (self.unrepairable_nodes):
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
            # On récupere la probabilité que le problème persiste après les 
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
           
    def simple_solver_obs(self, debug = False):
        """
        """
        rep_seq = []
        exp_cost = 0
        proba_cost = 1
        repares = [k for k in self.evidences]
        unrep_nodes = self.unrepairable_nodes.copy() | (set(repares))
        
        # On itère jusqu'à ce qu'il n'existe plus de noeud que peut être reparé
        reparables = (self.repairable_nodes | {self.service_node}) \
        - unrep_nodes
        while len(reparables) != 0:  
            if debug == True:
                print("Les noeuds considerés dans ce tour de boucle : ",\
                      reparables)
            # On crée un dictionnaire avec les efficacités liées à chaque noeud 
            # réparable + service dans ce tour de boucle
            dic_eff = {}  
            
            # On crée un dictionnaire avec les coûts utilisés pour calculer 
            # l'esperance de coût de la séquence.
            # On se sert que du coût lié au noeud choisi à la fin de chaque 
            # tour de boucle.
            dic_costs = {}
            
            # Pour chaque noeud réparable + service (et qui n'est pas 
            # irréparable):
            for node in reparables:   
                # On rajoute le fait que le dispositif est en panne
                self.bay_lp.chgEvidence(self.problem_defining_node, "yes")
                
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
                
                # Une fois qu'on aura fait l'observation-repair, on ne sait
                # plus si le dispositif est en panne ou pas
                self.bay_lp.chgEvidence(self.problem_defining_node, [1, 1])
                
                # On récupere le coût pour le calcul de l'ésperance
                dic_costs[node] = cost
                
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
                    print("coût esperé du pair observation-repair : ", cost)
                    print("éfficacité du noeud : ", dic_eff[node])
                    print()
            
            # On sort les noeuds par rapport aux efficacités
            seq = sorted(dic_eff.items(), key = lambda x: x[1], \
                             reverse = True) 
            # Le noeud choisi est ce avec la meilleure efficacité dans le tour
            # de boucle actuel
            chosen_node = seq[0][0]
            rep_seq.append(chosen_node)
            
            # On calcule la contribution à l'ésperance de coût de la sequence
            # de ce noeud
            exp_cost += dic_costs[chosen_node] * proba_cost
            p = self.bay_lp.posterior(self.problem_defining_node)                
            inst = gum.Instantiation(p)
            inst.chgVal(self.problem_defining_node, "yes")
            proba_cost = p[inst]
            
            if debug == True:
                print("noeud choisi dans ce tour de boucle : ", chosen_node)
                print("contribution à l'ésperance du coût de la séquence : ",\
                      dic_costs[chosen_node])
                print("ésperance du coût partiel : ", exp_cost)
                print()
                            
            # On garde ce noeud au dictionnaire repares pour qu'on puisse
            # mantenir le reseau à jour a chaque tour de la boucle while 
            if chosen_node != "callService":
                unrep_nodes.add(chosen_node)
                reparables = (self.repairable_nodes | set([self.service_node]))\
                    - unrep_nodes
                self.bay_lp.chgEvidence(chosen_node, "no")
            else:
                break
        # On retourne aux évidences du début
        self.reset_bay_lp(self.evidences)           
        return rep_seq, exp_cost
    
    def myopic_solver(self, debug = False):
        """
        """
        # Liste des observations generales qu'on peut faire
        nd_obs = set(self.observation_nodes).intersection(\
                      set(self.unrepairable_nodes))
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
            
            if debug:
                print(node)
                print(p)
            
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
                self.bay_lp.chgEvidence(node, [1]*len(self.bayesian_network.variable(node).labels()))
        
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
        return chosen_node, type_node

    # Une fonction qui calcule un coût espéré de réparation à partir d'une séquence d'actions donnée
    # Ici on utilise une formule
    # ECR = coût(C1) +
    #       (1 - P(e = Normal | C1 = Normal)) * coût(C2) +
    #       (1 - P(e = Normal | C1 = Normal, C2 = Normal)) * coût(C3) + ...
    def expected_cost_of_repair_seq_of_actions(self, seq):

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
    def brute_force_solver_actions_only(self, debug=False):
        min_seq = [self.service_node] + list(self.repairable_nodes).copy()
        min_ecr = self.expected_cost_of_repair_seq_of_actions(min_seq)

        # Parcours par toutes les permutations de l'union de noeuds réparables avec un noeud de service
        for seq in [list(t) for t in permutations(list(self.repairable_nodes) + [self.service_node])]:
            ecr = self.expected_cost_of_repair_seq_of_actions(seq)
            if ecr < min_ecr:
                min_ecr = ecr
                min_seq = seq.copy()

        return min_seq, min_ecr

    # Une méthode qui calcule le coût espéré de réparation étant donné un arbre de décision
    def expected_cost_of_repair(self, strategy_tree, obs_rep_couples=False):
        ecr = self._expected_cost_of_repair_internal(strategy_tree, obs_rep_couples=obs_rep_couples)
        self.bay_lp.setEvidence({})
        self.start_bay_lp()
        self.reset_bay_lp()
        return ecr

    # Une partie récursive d'une fonction expected_cost_of_repair ci-dessus
    def _expected_cost_of_repair_internal(self, strategy_tree, evid_init=None, obs_rep_couples=False):
        
        if not isinstance(strategy_tree, st.StrategyTree):
            raise TypeError('strategy_tree must have type StrategyTre')

        ecr = 0.0
        evidence = evid_init if isinstance(evid_init, dict) else {}
        self.bay_lp.setEvidence(evidence)
        prob = 1.0 if evidence == {} else (1 - self.prob_val(self.problem_defining_node, 'no'))
        node = strategy_tree.get_root()
        node_name = strategy_tree.get_root().get_name()
        aelem = 0.0
        # cost = node.get_cost()
        if obs_rep_couples and node_name in self.repairable_nodes.intersection(self.observation_nodes):
            cost = self.costs_obs[node_name]  # + self.prob_val(node_name, 'yes') * self.costs_rep[node_name]
            self.bay_lp.setEvidence(merge_dicts(evidence, {node_name: 'yes'}))
            prob_next = 1 - self.prob_val(self.problem_defining_node, 'no')
            self.bay_lp.setEvidence(evidence)
            aelem = prob_next * self.prob_val(node_name, 'yes') * self.costs_rep[node_name]
        else:
            cost = self.costs_rep[node_name] if isinstance(node, st.Repair) else self.costs_obs[node_name]
        ecr += prob * cost + aelem

        if len(node.get_list_of_children()) == 0:
            return ecr
        if isinstance(node, st.Repair):
            ecr += self._expected_cost_of_repair_internal(
                strategy_tree.get_sub_tree(node.get_child()),
                merge_dicts(evidence, {node_name: 'yes' if node_name == self.service_node else 'no'}), obs_rep_couples)
            self.bay_lp.setEvidence(evidence)
        else:
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

    # Une méthode auxiliare qui retourne une probabilité posterioiri qu'une variable var égal à value
    def prob_val(self, var, value):
        tmp = self.bay_lp.hardEvidenceNodes()
        pot_var = self.bay_lp.posterior(var)
        inst_var = gum.Instantiation(pot_var)
        inst_var.chgVal(var, value)
        return pot_var[inst_var]

    # Une méthode auxiliare qui crée des noeuds de StrategyTree à partir de leurs noms dans un modèle
    def _create_nodes(self, names, rep_string='_repair', obs_string='_observation', obs_rep_couples=False):
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
                        print(f'{bcolors.OKGREEN}\n########################\nIter %d\n########################\n{bcolors.ENDC}' % i)
                    par_mutable = node.copy()
                    par_immutable = node.copy() if isinstance(node, st.Observation) else None

                    par_tree_mutable = st.StrategyTree(node)
                    onn = ([[l for l in self.bayesian_network.variable(node.get_name()).labels()]]
                           if isinstance(node, st.Observation) else None)
                    par = [(par_mutable, par_immutable, par_tree_mutable.copy())]
                    fn_im = []
                    fn_im.append([n.copy() for n in (feasible_nodes[:i] + feasible_nodes[i+1:])])
                    self._evaluate_all_st(feasible_nodes[:i] + feasible_nodes[i+1:],
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
                        self._evaluate_all_st(feasible_nodes[:i] + feasible_nodes[i+1:],
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
                            obs_next_nodes_tmp = [branch_attrs]\
                                if obs_next_nodes_tmp is None else obs_next_nodes_tmp + [branch_attrs]
                            fn_im.append([n.copy() for n in (feasible_nodes[:i] + feasible_nodes[i+1:])])
                        self._evaluate_all_st(feasible_nodes[:i] + feasible_nodes[i+1:],
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
                if obs_rep_couples:
                    if not self._is_feasible_st_obs_rep_couples(strat_tree):
                        return
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
                    feasible_nodes[:i] + feasible_nodes[i+1:], new_evidence, debug_iter, debug_st, obs_rep_couples
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
            self.start_bay_lp()
            self.reset_bay_lp()

        if debug_st:
            print(best_tree)
            print(best_ecr)

        return best_tree, best_ecr

    def _is_feasible_st_obs_rep_couples(self, strat_tree, not_working_component_val="yes"):
        ids_to_check = []
        for node in strat_tree.get_nodes():
            if (node.get_name() in self.repairable_nodes.intersection(self.observation_nodes) and
                    isinstance(node, st.Observation)):
                ids_to_check.append(node.get_id())
        for node_id in ids_to_check:
            node = strat_tree.get_node(node_id)
            child_to_check = node.get_child_by_attribute(not_working_component_val)
            if child_to_check is None or child_to_check.get_name() != node.get_name():
                return False
        return True

# =============================================================================
# Méthodes pas encore fonctionnelles            
# =============================================================================
    def myopic_wraper(self, debug = False):
        """
        """
        print("Bienvenue! Suivez les instructions pour réparer votre dispositif")
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
                self.change_evidence(node, possibilites[val_obs])
            else:
                print("Faire l'observation-réparation suivante : " + node)
                print("Problème résolu ? (Y/N)")
                val_rep = input()
                if val_rep in "Yy":
                    fixed = True
                else:
                    obsoletes = self.observation_obsolete(node) 
                    if node != "callService":
                        self.change_evidence(node, "no")
                    else:
                        self.change_evidence(node, "yes")  
                    for obs in obsoletes:
                        self.evidences.pop(obs)
                    self.reset_bay_lp(self.evidences)
        print("Merci d'avoir utilisé le logiciel !")
        self.reset_bay_lp()    
                
        
    def noeud_ant(self, node, visites):
        """
        """
        ant_obs = set()
        parents = {self.bayesian_network.names()[p] for p in self.bayesian_network.parents(node)}
        parents = parents.difference(visites)
        for p in parents:
            visites.add(p)
            if p in self.evidences:
                if p in self.unrepairable_nodes and p in self.observation_nodes:
                    ant_obs.add(p)
                    ant_obs.update(self.noeud_ant(p, visites))
                
            else:
                ant_obs.update(self.noeud_ant(p, visites))
        return ant_obs
        
    def observation_obsolete(self, node):
        """
        """
        visites = {node}
        obs = self.noeud_ant(node, visites)
        stack = [node]
        while stack != []:
            n = stack.pop()
            enfants = {self.bayesian_network.names()[p] for p in self.bayesian_network.children(n)}
            enfants = enfants.difference(visites)
            for en in enfants:
                visites.add(en)
                if en in self.evidences:
                    if en in self.unrepairable_nodes and en in self.observation_nodes:    
                        obs.add(en)
                        stack.append(en)
                    elif en in self.repairable_nodes:
                        obs.update(self.noeud_ant(en, visites))
                else:
                    stack.append(en)
        return obs

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
