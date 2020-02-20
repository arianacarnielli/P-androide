import pyAgrum as gum


# Une classe qui représente un problème de Troubleshooting
# Elle se compose des attributs suivants
# bayesian_network : un réseau bayésien (BN) qui modélise un problème donné
# costs : un dictionnaire de coûts où des clés représentent des noeuds de BN et des valeurs -- les valeurs de ces coûts
# nodes_type : un dictionnaire de types ______________________________ de BN et des valeurs -- un type dans le format
# de string
# repairable_nodes : une liste de noeuds qui correspondent aux éléments d'un appareil qui peuvent être réparés ; de même
# façon on définit les attributs unrepairable_nodes, problem_defining_node et observation_nodes
class TroubleShootingProblem:

    # Constructeur
    def __init__(self, bayesian_network, costs, nodes_types):
        self.bayesian_network = gum.BayesNet(bayesian_network)
        self.costs = costs.copy()
        self.nodes_types = nodes_types.copy()
        self.repairable_nodes = [node for node in nodes_types.keys() if nodes_types[node] == 'repairable']
        self.unrepairable_nodes = [node for node in nodes_types.keys() if nodes_types[node] == 'unrepairable']
        self.problem_defining_node = [node for node in nodes_types.keys() if nodes_types[node] == 'problem-defining'][0]
        self.observation_nodes = [node for node in nodes_types.keys() if nodes_types[node] == 'observation']

    # Une méthode qui implémente un algorithme le plus simple de résolution du problème de Troubleshooting
    def solve_static(self):
        rep_nodes = self.repairable_nodes.copy()
        rep_seq = []
        # L'efficacité de l'appel à service
        service_ef = 1.0 / self.costs['service']
        ie = gum.LazyPropagation(self.bayesian_network)
        # Tant qu'on n'a pas encore observer tous les noeuds réparables
        while len(rep_nodes) > 0:
            # On suppose par défaut que l'appel à service est une action la plus efficace
            # on cherche ensuite une action plus efficace
            action_to_put = 'service'
            ef = service_ef
            # On observe tous les actions pas encore observés
            for rnode in range(len(rep_nodes)):
                if rnode != 0:
                    ie.eraseEvidence(rep_nodes[rnode-1])
                ie.setEvidence({rep_nodes[rnode]: False})
                # On vérifie si une action courante est plus efficace que le dernier  max par efficacité
                if ef < ie.posterior(self.problem_defining_node).tolist()[0] / self.costs[rep_nodes[rnode]]:
                    ef = ie.posterior(self.problem_defining_node).tolist()[0] / self.costs[rep_nodes[rnode]]
                    action_to_put = rep_nodes[rnode]
            rep_seq.append(action_to_put)
            # Si on a trouvé quelque part qu'un appel au service est plus efficace que toutes les actions possibles,
            # on ajoute donc 'service' dans une séquence de réparation et on s'arrête car cet appel réparera un appareil
            # avec un certain
            if action_to_put == 'service':
                return rep_seq
            ie.eraseEvidence(rep_nodes[-1])
            ie.setEvidence({action_to_put: False})
            # on met-à-jour les noeuds réparables
            rep_nodes.remove(action_to_put)
        rep_seq.append('service')
        return rep_seq
