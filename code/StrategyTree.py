from graphviz import Digraph

class NodeST:

    """
    Classe qui représente un noeud plutôt abstrait d'un arbre de stratégie ; on remarque que cette classe ne dispose pas
    d'un enfant (il n'y a pas d'un attribut qui correspond à un noeud suivant), pourtant, on suppose que ses sous-
    classes en auront.

    Attributs :
        _id : identificateur unique d'un noeud, objet du type str.
        _cost : un attribut qui correspond à "coût" d'un noeud, objet du type float.
        _name : un nom d'un noeud qui peut être pas unique, objet du type str; si rien a été passé, on pose que
                _name = _id.
    """

    def __init__(self, id, cost, name=None):
        """
        Un constructeur qui crée un objet de type NodeST initialisant ses attributs par des valeurs fournies.

        Args :
            id : identificateur unique d'un noeud, objet du type str.
            cost : un attribut qui correspond à "coût" d'un noeud, objet du type float.
            name (facultatif) : un nom d'un noeud qui peut être pas unique, objet du type str; si rien a été soumis,
                on pose que _name = _id.
        """
        self._id = id
        self._cost = cost
        self._name = name if name is not None else id

    def set_id(self, id):
        """
        Setter d'un attribut _id.

        Args :
            id : identificateur unique d'un noeud pour mettre en place, objet du type str.
        """
        self._id = id

    def get_id(self):
        """
        Getter d'un attribut _id.

        Returns :
            _id : identificateur unique courant d'un noeud, objet du type str.
        """
        return self._id

    def set_cost(self, cost):
        """
        Setter d'un attribut _cost.

        Args :
            cost : une valeur de coût d'un noeud à mettre en place, objet du type float.
        """
        self._cost = cost

    def get_cost(self):
        """
        Getter d'un attribut _cost.

        Returns :
            _cost : une valeur de coût d'un noeud, objet du type float.
        """
        return self._cost

    def set_name(self, name):
        """
        Setter d'un attribut _name.

        Args :
            name : un nom d'un noeud pour mettre en place, objet du type str.
        """
        self._name = name

    def get_name(self):
        """
        Getter d'un attribut _name.

        Returns :
            _name : un nom d'un noeud, objet du type str.
        """
        return self._name

    def get_child_by_attribute(self, attr):
        """
        Une méthode abstraite qui retournerait un enfant d'un noeud correspondant à attr.

        Args :
            attr : un attribut (un type) d'enfant qu'il faudrait retourner, objet du type str.

        Returns :
            child : un enfant d'un noeud qui correspond à un attribut soumis, objet du type NodeST.
        """
        raise NotImplementedError('Abstract method! Should be implemented in subclass!')

    def set_child_by_attribute(self, attr, child=None):
        """
        Une méthode abstraite qui mettrait en place un enfant d'un noeud correspondant à attr.

        Args :
            attr : un attribut (un type) d'enfant qu'il faudrait mettre en place, objet du type str.
            child (facultatif) : un enfant d'un noeud qui correspond à un attribut soumis et qu'il faut mettre en place.
        """
        raise NotImplementedError('Abstract method! Should be implemented in subclass!')

    def get_list_of_children(self):
        """
        Une méthode abstraite qui permettrait d'obtenir une liste de tous les enfants d'un noeud.

        Returns :
            list_of_children : une liste de tous les enfants d'un noeud.
        """
        raise NotImplementedError('Abstract method! Should be implemented in subclass!')

    def add_child(self, child):
        """
        Une méthode abstraite qui ajouterait un enfant d'un noeud.

        Args :
            child : un enfant à ajouter, objet du type NodeST.
        """
        raise NotImplementedError('Abstract method! Should be implemented in subclass!')

    def copy(self):
        """
        Une méthode qui retournerait une copie superficielle d'un noeud.

        Returns :
            copy : une copie superficielle d'un noeud, objet du type NodeST.
        """
        raise NotImplementedError('Abstract method! Should be implemented in subclass!')

    def __eq__(self, other):
        """
        Overloading d'opérateur __eq__ ; on dit que deux noeuds sont égaux ssi ils ont les mêmes ids.

        Args :
            other : un autre noeud à comparer avec celui-ci, objet du type NodeST.

        Returns :
            comp_res : True, si self._id == other._id ET si self et other ont le même type ; False, sinon.
        """
        if (self is None and other is not None) or (self is not None and other is None):
            return False
        return self._id == other._id and isinstance(other, type(self))

    def __str__(self):
        """
        Overloading d'opérateur __str__.

        Returns :
            corr_str : une représentation d'un noeud sous une forme de str, objet du type str.
        """
        return '(' + self._id + ': ' + str(self._cost) + ', ' + self._name + ')'

    def bn_labels_children_association(self):
        """
        Une méthode abstraite qui retournera un dictionnaire des associations entre des labels d'un réseau
        bayésien et des enfants d'un noeud.

        Returns :
            da : un dictionnaire des associations concerné.
        """
        raise NotImplementedError('Abstract method! Should be implemented in subclass!')


class Repair(NodeST):

    """
    Classe qui représente un noeud de type plus précis : celui correspondant à une action de réparation.

    Attributs :
        _id : identificateur unique d'un noeud, objet du type str.
        _cost : un attribut qui correspond à "coût" d'un noeud, objet du type float.
        _name : un nom d'un noeud qui peut être pas unique, objet du type str; si rien a été passé, on pose que
            _name = _id.
        _child : un enfant d'un noeud, c'est-à-dire, un noeud suivant dans un arbre ; objet du type NodeST.
        _obs_rep_couples : un attribut sous la forme d'une variable boléenne qui indique si le noeud
            représente une couple observation-réparation ou pas
    """

    def __init__(self, id, cost, name=None, child=None, obs_rep_couples=False):
        """
        Un constructeur qui crée un objet de type Repair(NodeST) initialisant ses attributs par des valeurs fournies.

        Args :
            id : identificateur unique d'un noeud, objet du type str.
            cost : un attribut qui correspond à "coût" d'un noeud, objet du type float.
            name (facultatif) : un nom d'un noeud qui peut être pas unique, objet du type str; si rien a été soumis,
                on pose que _name = _id.
            child (facultatif) : un enfant d'un noeud, objet du type NodeST.
            obs_rep_couples (facultatif) : un attribut sous la forme d'une variable boléenne qui indique si le noeud
                représente une couple observation-réparation ou pas.
        """
        super().__init__(id, cost, name)
        self._child = child
        self._obs_rep_couples = obs_rep_couples

    def set_child(self, child=None):
        """
        Setter d'un attribut _child.

        Args :
            child (facultatif) : un enfant d'un noeud, objet du type NodeST.
        """
        self._child = child

    def get_child(self):
        """
        Getter d'un attribut _child.

        Returns :
            _child : un enfant d'un noeud, objet du type NodeST.
        """
        return self._child

    def set_obs_rep_couples(self, obs_rep_couples):
        """
        Setter d'un attribut _obs_rep_couples

        Args :
            obs_rep_couples : un attribut sous la forme d'une variable boléenne qui indique si le noeud
                représente une couple observation-réparation ou pas.
        """
        self._obs_rep_couples = obs_rep_couples

    def get_obs_rep_couples(self):
        """
        Getter d'un attribut _obs_rep_couples

        Returns :
            _obs_rep_couples : un attribut sous la forme d'une variable boléenne qui indique si le noeud
                représente une couple observation-réparation ou pas
        """
        return self._obs_rep_couples

    def get_child_by_attribute(self, attr):
        """
        Une méthode qui réalise une version abstraite de superclass ; comme ce type de noeud ne dispose que d'un seul
        enfant, on y retourne toujours cet enfant pour n'importe quel attr indiqué

        Args :
            attr : un attribut (un type) d'enfant qu'il faudrait retourner, objet du type str.

        Returns :
            child : un enfant d'un noeud, objet du type NodeST.
        """
        return self._child

    def set_child_by_attribute(self, attr, child=None):
        """
        Une méthode qui met en place un enfant d'un noeud correspondant à attr réalisant une méthode correspondante
        superclass.

        Args :
            attr : un attribut (un type) d'enfant qu'il faut mettre en place, objet du type str.
            child (facultatif) : un enfant d'un noeud qui correspond à un attribut soumis et qu'il faut mettre en place.
        """
        self.set_child(child)

    def get_list_of_children(self):
        """
        Une méthode qui retourne une liste qui contient tous les enfants d'un noeud ; pour ce cas-là alors, soit
        une liste avec un seul élément, soit une liste vide.

        Returns :
            list_of_children : une liste de tous les enfants d'un noeud (soit une liste avec un seul élément,
                soit une liste vide).
        """
        return [self._child] if self._child is not None else []

    def add_child(self, child):
        """
        Une méthode qui ajoute littéralement un enfant dans une liste des enfants d'un noeud.
        ATTENTION : ce méthode ne peut pas changer un enfant qui existe déjà ; pour cela, veuillez utiliser set_child !

        Args :
            child : un enfant à ajouter, objet du type NodeST.
        """
        if child is None:
            return
        if self._child is not None:
            raise OverflowError('It is not permitted to add more than one child to this type of nodes!')
        self.set_child(child)

    def copy(self):
        """
        Une méthode qui retourne une copie superficielle d'un noeud.

        Returns :
            copy : une copie superficielle d'un noeud, objet du type Repair(NodeST).
        """
        new_node = Repair(self._id, self._cost, self._name, self._child.copy() if self._child is not None else None,
                          self._obs_rep_couples)
        return new_node

    def bn_labels_children_association(self):
        """
        Une méthode plutôt auxiliare qui retourne un dictionnaire des associations entre des labels d'un réseau
        bayésien et des enfants d'un noeud.

        Returns :
            da : un dictionnaire des associations concerné.
        """
        return {'': self.get_child()}


class Observation(NodeST):

    """
    Classe qui représente un noeud de type correspondant à une action d'observation.

    Attributs :
        _id : identificateur unique d'un noeud, objet du type str.
        _cost : un attribut qui correspond à "coût" d'un noeud, objet du type float.
        _name : un nom d'un noeud qui peut être pas unique, objet du type str; si rien a été passé, on pose que
                _name = _id.
        _yes_child : un enfant d'un noeud qui correspond à une branche "yes", objet du type NodeST.
        _no_child : un enfant d'un noeud qui correspond à une branche "no", objet du type NodeST.
    """

    def __init__(self, id, cost, name=None, yes_child=None, no_child=None):
        """
        Un constructeur qui crée un objet de type Observation(NodeST) initialisant ses attributs par des valeurs
        fournies.

        Args :
            id : identificateur unique d'un noeud, objet du type str.
            cost : un attribut qui correspond à "coût" d'un noeud, objet du type float.
            name (facultatif) : un nom d'un noeud qui peut être pas unique, objet du type str; si rien a été soumis,
                on pose que _name = _id.
            yes_child (facultatif) : un enfant d'un noeud qui correspond à une branche "yes", objet du type NodeST.
            no_child : un enfant d'un noeud qui correspond à une branche "no", objet du type NodeST.
        """
        super().__init__(id, cost, name)
        self._yes_child = yes_child
        self._no_child = no_child

    def set_yes_child(self, yes_child=None):
        """
        Setter d'un attribut _yes_child.

        Args :
            yes_child (facultatif) : un enfant d'un noeud qui correspond à une branche "yes", objet du type NodeST.
        """
        self._yes_child = yes_child

    def get_yes_child(self):
        """
        Getter d'un attribut _yes_child.

        Returns :
            _yes_child : un enfant d'un noeud qui correspond à une branche "yes", objet du type NodeST.
        """
        return self._yes_child

    def set_no_child(self, no_child=None):
        """
        Setter d'un attribut _no_child.

        Args :
            no_child (facultatif) : un enfant d'un noeud qui correspond à une branche "no", objet du type NodeST.
        """
        self._no_child = no_child

    def get_no_child(self):
        """
        Getter d'un attribut _no_child.

        Returns :
            _no_child : un enfant d'un noeud qui correspond à une branche "no", objet du type NodeST.
        """
        return self._no_child

    def get_child_by_attribute(self, attr):
        """
        Une méthode qui retourne un enfant d'un noeud correspondant à l'attribut de la branche soumise.

        Args :
            attr : attribut de la branche, objet du type str.
        Returns :
            child : un enfant correspondant à l'attribut, objet du type NodeST.
        """

        # Si attr == "yes", on retourne _yes_child
        if str(attr) == 'yes':
            return self._yes_child
        # Si attr == "no", on retourne _no_child
        elif str(attr) == 'no':
            return self._no_child
        # Sinon l'attribut est donc pas connu et on retourne None
        return None

    def set_child_by_attribute(self, attr, child=None):
        """
        Une méthode qui met en place un enfant d'un noeud correspondant à l'attribut de la branche soumise.

        Args :
            attr : attribut de la branche, objet du type str.
            child (facultatif) : un enfant qu'il faut mettre en place, objet du type NodeST.
        """

        # Si on reconnaît l'attribut, on met en place un enfant
        if str(attr) == 'yes':
            self.set_yes_child(child)
        elif str(attr) == 'no':
            self.set_no_child(child)
        # Sinon on évoque une exception
        else:
            raise ValueError('Unknown attribute "%s" for the type of node "Observation"' % str(attr))

    def get_list_of_children(self):
        """
        Une méthode qui retourne une liste avec tous les enfants d'un noeud.

        Returns :
            list_of_children : une liste avec tous les enfants d'un noeud.
        """
        # On ajoute les enfants
        res = [self._yes_child, self._no_child]
        # On supprime tous les valeurs de None
        while None in res:
            res.remove(None)
        return res

    def add_child(self, child):
        """
        Une méthode qui ajoute littéralement un enfant dans une liste des enfants d'un noeud.
        ATTENTION : ce méthode ne peut pas changer un enfant qui existe déjà ; pour cela, veuillez utiliser set_child !

        Args :
            child : un enfant à ajouter, objet du type NodeST.
        """
        if child is None:
            return
        # Par défaut, on ajoute d'abord _yes_child
        if self._yes_child is None:
            self.set_yes_child(child)
        # Ensuite, _no_child
        elif self._no_child is None:
            self.set_no_child(child)
        # Finalement, on ne peut ajouter que deux enfants@
        else:
            raise OverflowError('It is not permitted to add more than two children to this type of nodes'
                                '("Observation")!')

    def copy(self):
        """
        Une méthode qui retourne une copie superficielle d'un noeud.

        Returns :
            copy : une copie superficielle d'un noeud, objet du type Observation(NodeST).
        """
        new_node = Observation(self._id, self._cost, self._name,
                               self._yes_child.copy() if self._yes_child is not None else None,
                               self._no_child.copy() if self._no_child is not None else None)
        return new_node

    def bn_labels_children_association(self):
        """
        Une méthode plutôt auxiliare qui retourne un dictionnaire des associations entre des labels d'un réseau
        bayésien et des enfants d'un noeud.

        Returns :
            da : un dictionnaire des associations concerné.
        """
        return {'no': self.get_no_child(), 'yes': self.get_yes_child()}


class StrategyTree:
    """
    Une classe pour représenter un arbre de stratégie qui fait face au problème de Troubleshooting.

    Attributs :
        _root : une racine de l'arbre, i.e. une action pour commencer ; objet du type NodeST
        _nodes : une liste des noeuds de l'arbre.
        _adj_dict : un dictionnaire pour indiquer lesquels noeuds sont liés par des arcs.
    """

    def __init__(self, root=None, nodes=None):
        """
        Un constructeur qui crée un nouvelle arbre de stratégie donc un nouveau objet du type StrategyTree.

        Attributs :
            root (facultatif) : une racine de l'arbre, i.e. une action pour commencer ; objet du type NodeST.
            nodes (facultatif) : une liste des noeuds de l'arbre.
        """

        # une initialisation par défaut qui crée un arbre vide
        self._root = root
        self._nodes = []
        self._adj_dict = {}

        # si on a précisé une racine, on l'ajoute dans un arbre
        if root is not None:
            if not isinstance(root, NodeST):
                raise TypeError('Type of parameter "root" must be NodeST!')
            self._adj_dict[self._root.get_id()] = [n.get_id() for n in self._root.get_list_of_children()]
            self._nodes.append(self._root)
        # si on a précisé une liste des noeuds, on les ajoute tous dans un arbre
        if nodes is not None:
            if not isinstance(nodes, list) or not sum([isinstance(n, NodeST) for n in nodes]) == len(nodes):
                raise TypeError('Type of parameter "nodes" must be NodeST or list<NodeST>!')
            if len([n.get_id() for n in nodes]) != len(set([n.get_id() for n in nodes])):
                raise ValueError('All the nodes\' ids must be unique!')
            if self._root is not None and not self._root.get_id() in [n.get_id() for n in nodes]:
                raise ValueError('The passed root is not contained in nodes!')
            self._nodes.extend([n for n in nodes if n != self._root])
            for n in [n for n in self._nodes if n != self._root]:
                self._adj_dict[n.get_id()] = [ch.get_id() for ch in n.get_list_of_children()]
        # ATTENTION : un dictionnaire _adj_dict est rempli automatiquement en utilisant des valeurs mises dans root et
        # nodes

    def __str__(self):
        """
        Une méthode qui réalise transformation de l'arbre vers str.

        Returns :
            st_str : une représentation de l'arbre de stratégie en forme de str, objet du type str.
        """

        # Plus précisément, on utilise pour cela l'une des méthodes ci-dessous
        return self.str_alt_2()

    def str_alt(self):
        """
        Une méthode qui réalise une transformation particulière de l'arbre vers str.

        Returns :
            st_str : une représentation de l'arbre de stratégie en forme de str, objet du type str.
        """
        if self._root is None:
            return '{}'
        res = '{'
        nodes = [self._root]
        while len(nodes) > 0:
            nodes_tmp = []
            for n in nodes:
                if isinstance(n, Observation):
                    res += '%s -> {\'no\': %s, \'yes\': %s}, ' % (
                        n.get_name(),
                        n.get_no_child().get_name() if n.get_no_child() is not None else 'None',
                        n.get_yes_child().get_name() if n.get_yes_child() is not None else 'None'
                    )
                elif isinstance(n, Repair):
                    res += '%s -> {%s}, ' % (
                        n.get_name(),
                        n.get_child().get_name() if n.get_child() is not None else 'None'
                    )
                nodes_tmp.extend(n.get_list_of_children())
            nodes = nodes_tmp
        res = res[:-2] + '}'

        return res

    def str_alt_2(self):
        """
        Une autre méthode qui réalise une transformation particulière de l'arbre vers str.

        Returns :
            st_str : une représentation de l'arbre de stratégie en forme de str de manière alternative,
                objet du type str.
        """
        if self._root is None:
            return '{}'
        res = '{'
        for k in self._adj_dict.keys():
            node = self.get_node(k)
            res += '%s -> {' % node.get_name()
            if isinstance(node, Observation):
                ch1, ch2 = \
                    self._adj_dict[k][0] if len(self._adj_dict[k]) > 0 else None,\
                    self._adj_dict[k][1] if len(self._adj_dict[k]) > 1 else None
                ch1, ch2 = self.get_node(ch1), self.get_node(ch2)
                res += '\'no\': %s, \'yes\': %s}, ' % (
                    (ch1.get_name() if ch1 is not None else 'None', ch2.get_name() if ch2 is not None else 'None')
                    if ch1 == node.get_no_child() else
                    (ch2.get_name() if ch2 is not None else 'None', ch1.get_name() if ch1 is not None else 'None')
                )
            elif isinstance(node, Repair):
                ch = self.get_node(self._adj_dict[k][0] if len(self._adj_dict[k]) > 0 else None)
                res += '%s}, ' % (ch.get_name() if ch is not None else 'None')
        res = res[:-2] + '}'
        return res

    def set_root(self, root):
        """
        Un setter d'un attribut _root.

        Args :
            root : une racine à mettre en place, objet du type NodeST.
        """

        # Si une racine soumise n'était pas de bon type, on évoquerait une exception
        if root is not None:
            if not isinstance(root, NodeST):
                raise TypeError('Type of parameter "root" must be NodeST!')
        # Sinon on met à jour les attributs de l'arbre selon une racine indiquée
        self._root = root
        if root is not None and not root.get_id() in [n.get_id() for n in self._nodes]:
            self._nodes.insert(0, self._root)
            self._adj_dict[self._root.get_id()] = [n.get_id() for n in self._root.get_list_of_children()]

    def get_root(self):
        """
        Un getter d'un attribut _root.

        Returns :
            _root : une racine de l'arbre, objet du type NodeST.
        """
        return self._root

    def get_node(self, id):
        """
        Une méthode qui retourne un noeud exacte (en sens de l'objet dans mémoire vivant) de l'arbre avec id indiqué.

        Args :
            id : soit id de noeud, objet du type str, soit un noeud lui-même, objet du type NodeST, dont un clone
                (en sens d'id) il faut trouver dans l'arbre.

        Returns :
            n : un noeud de l'arbre avec la même id que celui soumis, objet du type NodeST.
        """

        if isinstance(id, NodeST):
            id = id.get_id()
        for n in self._nodes:
            if n.get_id() == id:
                return n
        # S'il n'existe pas de tel noeud dans l'arbre alors on retourne None
        return None

    def get_node_by_name(self, name):
        """
        Une méthode qui retourne tous les noeuds de l'arbre dont les noms sont égaux à celui indiqué.

        Args :
            name : un nom ou un noeud dont le nom on doit utiliser ; objet du type str ou NodeST.

        Returns :
            nodes : une liste de tous les noeuds de l'arbre qui ont le même nom que celui donné.
        """
        if isinstance(name, NodeST):
            name = name.get_name()
        nodes = []
        for n in self._nodes:
            if n.get_name() == name:
                nodes.append(n)
        return nodes

    def get_edges(self):
        """
        Une méthode qui récupère tous les arcs de l'arbres.

        Returns :
            edges : une liste de triplets où chaque celui edge correspond à un arc d'un graphe de manière que
                edge[0] est un parent, edge[1] est un enfant tandis que edge[2] est un attribut de la branche.
        """
        res = []
        # On parcourt par tous les parents et par tous leurs enfants
        for par_id in self._adj_dict.keys():
            for ch_id in self._adj_dict[par_id]:
                par = self.get_node(par_id)
                ch = self.get_node(ch_id)
                # On ajoute un triplet dans une liste
                if isinstance(par, Observation) and par.get_no_child() == ch:
                    res.append((self.get_node(par_id), self.get_node(ch_id), 'no'))
                elif isinstance(par, Observation) and par.get_yes_child() == ch:
                    res.append((self.get_node(par_id), self.get_node(ch_id), 'yes'))
                else:
                    res.append((self.get_node(par_id), self.get_node(ch_id), None))
        return res

    def add_node(self, node):
        """
        Une méthode qui permet d'ajouter un.des nouveau.x noeud.s dans l'arbre.

        Args :
            node : noeud.s à ajouter, un objet du type NodeST ou list<NodeST>.
        """
        if node is None:
            return
        if isinstance(node, NodeST):
            # Si un seul noeud a été mis en place on l'ajoute dans un arbre sous condition que ce noeud ait un id unique
            if node.get_id() in [n.get_id() for n in self._nodes]:
                raise ValueError('All the nodes\' ids must be unique!')
            self._nodes.append(node)
            self._adj_dict[node.get_id()] = [n.get_id() for n in node.get_list_of_children()]
        elif isinstance(node, list) and sum([isinstance(n, NodeST) for n in node]) == len(node):
            # Si on a passé une liste de noeuds on les ajoute tous dans un arbre sous la même condition
            for n in node:
                if n.get_id() in [_n.get_id() for _n in self._nodes]:
                    raise ValueError('All the nodes\' ids must be unique!')
            self._nodes.extend(node)
            for n in node:
                self._adj_dict[n.get_id()] = [ch.get_id() for ch in n.get_list_of_children()]
        else:
            # Si on ne peut pas reconnaître le type, on évoque une exception
            raise TypeError('Type of parameter "node" must be NodeST or list<NodeST>!')

    def add_edge(self, parent, child, child_type=None):
        """
        Une méthode qui permet d'ajouter un arc dans un arbre entre deux noeuds.

        Args :
            parent : un noeud qui doit être un parent dans cet arc, donc un noeud duquel cet arc part ; objet du type
                str ou NodeST.
            child : un noeud qui doit être un enfant dans cet arc, donc un noeud auquel cet arc entre ; objet du type
                str ou NodeST.
            child_type : un attribut de la branche du parent à laquelle il faut ajouter un enfant (par exemple si
                parent est une observation alors child_type est égal soit à 'no', soit à 'yes') ; objet du type str.
        """

        # On vérifie si les paramètres soumis ont des bons types
        if not ((isinstance(parent, str) or isinstance(parent, NodeST)) and
                (isinstance(child, str) or isinstance(child, NodeST))):
            raise TypeError('Parent and child must be either str or NodeST!')
        # On récupère un parent et un enfant dans l'arbre et s'il n'y a pas de tels noeuds, on les ajoute
        par, ch = self.get_node(parent), self.get_node(child)
        if par is None:
            self.add_node(parent)
        if ch is None:
            self.add_node(child)
        par, ch = self.get_node(parent), self.get_node(child)
        # On ajoute un arc
        self._adj_dict[par.get_id()].append(ch.get_id())
        if not ch.get_id() in [n.get_id() for n in par.get_list_of_children()]:
            if isinstance(par, Repair):
                par.set_child(ch)
            elif isinstance(par, Observation) and child_type == 'yes':
                par.set_yes_child(ch)
            elif isinstance(par, Observation) and child_type == 'no':
                par.set_no_child(ch)
            else:
                par.add_child(ch)

    def get_nodes(self):
        """
        Un getter d'un attribut _nodes.

        Returns :
            _nodes : une copie superficielle d'une liste des noeuds d'un arbre, objet du type list<NodeST>.
        """
        return self._nodes.copy()

    def get_adj_dict(self):
        """
        Un getter d'un attribut _adj_dict.

        Returns :
            _adj_dict : une copie superficielle d'un dictionnaire d'adjacence, objet du type dict<str, list<str>>.
        """
        return self._adj_dict.copy()

    def get_sub_tree(self, sub_root):
        """
        Une méthode qui retourne un sous-arbre de cet arbre-là étant donné une sous-racine à partir de laquelle ce
        sous-arbre commence.

        Args :
            sub_root : une sous-racine de l'arbre qui est une racine de sous-arbre qu'il faut retourner, objet du type
                NodeST.

        Returns :
            sub_tree : un sous-arbre de cet arbre dont la racine est sub_root ; objet du type StrategyTree.
        """
        # Pour récupérer un sous-arbre demandé on effectue un parcours de l'arbre en largeur
        # nodes : les noeuds déjà traités et alors déjà ajoutés dans un sous-arbre
        nodes = [sub_root]
        # nodes_to_process : les enfants des noeuds dans nodes
        nodes_to_process = []
        nodes_to_process.extend(self.get_node(sub_root).get_list_of_children())

        # des variables auxiliaires pour voir comment l'algo marche si on le lance pas-à-pas
        debug_n = [n.get_id() for n in nodes]
        debug_ntp = [n.get_id() for n in nodes_to_process]

        # Tant qu'il existe des noeuds à traiter
        while len(nodes_to_process) > 0:
            debug_n = [n.get_id() for n in nodes]
            # On ajoute dans nodes tous les noeuds qu'on a trouvés dans un pas précédent
            nodes.extend(nodes_to_process)
            debug_n = [n.get_id() for n in nodes]
            nodes_to_process_copy = nodes_to_process.copy()
            # On met à jour nodes_to_process en remplaçant chaque noeud par ses enfants
            for n in nodes_to_process_copy:
                debug_ntp = [n.get_id() for n in nodes_to_process]
                nodes_to_process.remove(n)
                debug_ntp = [n.get_id() for n in nodes_to_process]
                nodes_to_process.extend(self.get_node(n).get_list_of_children())
                debug_ntp = [n.get_id() for n in nodes_to_process]

        for i in range(len(nodes)):
            nodes[i] = self.get_node(nodes[i])

        return StrategyTree(root=self.get_node(sub_root), nodes=nodes)

    def get_parent(self, child):
        """
        Une méthode qui cherche et qui retourne un parent d'un noeud soumis dans l'arbre. Remarque : dans un arbre comme
        ça chaque noeud ne doit avoir qu'un seul parent.

        Args :
            child : un enfant dont le parent il faut trouver dans cet arbre-là, objet du type NodeST ou str.

        Returns :
            parent : un parent trouvé d'un noeud soumis, objet du type NodeST.
        """
        ch = self.get_node(child)
        if ch is None or ch == self._root:
            return None
        # Parcours par un dictionnaire d'adjacence pour trouver un noeud qui a child pour cet enfant
        for n in self._nodes:
            if ch.get_id() in self._adj_dict[n.get_id()]:
                return n
        return None

    def remove_sub_tree(self, sub_root):
        """
        Une méthode qui nous permet de supprimer un sous-arbre de cet arbre étant donné une sous-racine.

        Args :
            sub_root : une sous-racine d'un sous-arbre qu'il faut supprimer, objet du type str ou NodeST.

        Returns :
            flag : une variable booléen qui est égale à True si on supprime quelque chose et False sinon.
        """

        # En exécutant parcours en largeur on récupère les noeuds qu'il faudrait supprimer (cf. une méthode get_sub_tree
        # pour obtenir plus de détails sur cet algo)
        sub_root_internal = self.get_node(sub_root)
        if sub_root_internal is None:
            return False
        nodes = [sub_root_internal]
        nodes_to_process = []
        nodes_to_process.extend(sub_root_internal.get_list_of_children())

        debug_n = [n.get_id() for n in nodes]
        debug_ntp = [n.get_id() for n in nodes_to_process]

        while len(nodes_to_process) > 0:
            debug_n = [n.get_id() for n in nodes]
            nodes.extend(nodes_to_process)
            debug_n = [n.get_id() for n in nodes]
            nodes_to_process_copy = nodes_to_process.copy()
            for n in nodes_to_process_copy:
                debug_ntp = [n.get_id() for n in nodes_to_process]
                nodes_to_process.remove(n)
                debug_ntp = [n.get_id() for n in nodes_to_process]
                nodes_to_process.extend(self.get_node(n).get_list_of_children())
                debug_ntp = [n.get_id() for n in nodes_to_process]

        # À partir de noeuds qu'il faut supprimer on efface toute l'information nécessaire
        # On supprime une connexion entre sous-racine et son parent (s'il y en existe)
        if sub_root == self._root:
            self._root = None
        else:
            par_sub_root = self.get_parent(sub_root)
            if isinstance(par_sub_root, Repair):
                par_sub_root.set_child(None)
            elif isinstance(par_sub_root, Observation) and par_sub_root.get_no_child() == sub_root:
                par_sub_root.set_no_child(None)
            elif isinstance(par_sub_root, Observation) and par_sub_root.get_yes_child() == sub_root:
                par_sub_root.set_yes_child(None)

        # On efface les noeuds dans _nodes et dans _adj_dict
        for n_r in nodes:
            if n_r in self._nodes:
                self._nodes.remove(n_r)
            if n_r.get_id() in self._adj_dict.keys():
                self._adj_dict.pop(n_r.get_id())
        for n in self._nodes:
            for n_r in nodes:
                if n_r.get_id() in self._adj_dict[n.get_id()]:
                    self._adj_dict[n.get_id()].remove(n_r.get_id())
        return True

    def copy(self):
        """
        Une méthode qui retourne une copie superficielle de cet arbre.

        Returns :
            copy : une copie superficielle de cet arbre-là, objet du type StrategyTree.
        """
        other_root = self._root.copy() if self._root is not None else None
        other_nodes = []
        for _n in self._nodes:
            other_nodes.append(_n.copy())
        return StrategyTree(root=other_root, nodes=other_nodes)

    def connect(self, root_with_subtree, root_child_type=None):
        """
        Une méthode qui nous permet de connecter deux arbre, plus précisément, on connecte tout cet arbre avec celui
        soumis dans root_with_subtree en remplissant sa branche qui correspond à root_child_type.

        Args :
            root_with_subtree : un arbre vers la racine duquel on va connecter cet arbre ; objet du type StrategyTree.
            root_child_type : un attribut de la branche de la racine du root_with_subtree ; objet du type str.

        Returns :
            merged_tree : un arbre fusionné, objet du type StrategyTree.
        """
        if not isinstance(root_with_subtree, StrategyTree):
            raise TypeError('root_with_tree must be an object of type StrategyTree')
        res = root_with_subtree.copy()

        # On supprime une branche de la racine du root_with_subtree qui correspond à un attribut root_child_type
        res.remove_sub_tree(res.get_node(res._root.get_child_by_attribute(root_child_type)))
        # On ajoute tout cet arbre dans un arbre du résultat
        # Donc chaque noeud ...
        for node in self._nodes:
            if node == self._root:
                self_root = node.copy()
                res.add_node(self_root)
            else:
                res.add_node(node.copy())
        # ... ainsi que chaque arc
        for par, ch, attr in self.get_edges():
            res.add_edge(par, ch, attr)
        # On ajoute un arc entre root_with_subtree et cet arbre
        res.add_edge(res._root, self_root, root_child_type)
        return res

    def visualize(self, filename='last_best_strategy_tree.gv'):
        """
        Une méthode qui nous permet d'afficher cet arbre de stratégie via un module graphviz. L'image construit est
        sauvegardé dans un fichier 'filename.pdf'.

        Args :
            filename : un nom de fichier où il faut sauvegarder une image obtenue.
        """
        vst = Digraph()
        for node in self._nodes:
            node_type = ('Obs-Rep' if isinstance(node, Repair) and node.get_obs_rep_couples()
                         else ('Repair' if isinstance(node, Repair) else 'Observation'))
            vst.node(node.get_id(), '%s (n%s) : %s' % (node.get_name(), node.get_id(), node_type))
        for par in self._adj_dict.keys():
            for ch in self._adj_dict[par]:
                for attr in self.get_node(par).bn_labels_children_association().keys():
                    if self.get_node(ch) == self.get_node(par).bn_labels_children_association()[attr]:
                        vst.edge(par, ch, label=attr)
        vst.render(filename, view=True)
