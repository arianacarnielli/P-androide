import os
from graphviz import Digraph


class NodeST:

    """
    Représente un noeud abstrait d'un arbre de stratégie; on remarque que cette
    classe ne dispose pas d'attribut correspondant à des enfants (il n'y a pas
    un attribut qui correspond à les noeuds suivants), pourtant, on suppose que
    ses sous-classes en auront.

    Parameters
    ----------
    id : str
        Identificateur unique d'un noeud.
    cost : float
        Correspond au "coût" du noeud.
    name : str, facultatif
        Nom du noeud qui peut ne pas être unique; si rien a été soumis, on pose
        que *_name* = *_id*.

    Attributes
    ----------
    _id : str
        Identificateur unique d'un noeud.
    _cost : float
        Correspond au "coût" du noeud.
    _name : str
        Nom du noeud qui peut ne pas être unique; si rien a été soumis, on pose
        que *_name* = *_id*.
    """

    def __init__(self, id, cost, name=None):
        self._id = id
        self._cost = cost
        self._name = name if name is not None else id

    def set_id(self, id):
        """
        Setter de l'attribut *_id*.

        Parameters
        ----------
        id : str
            Nouvel identificateur du noeud en question. Doit être unique. 
        """
        self._id = id

    def get_id(self):
        """
        Getter de l'attribut *_id*.

        Returns
        -------
        _id : str
            Identificateur unique courant du noeud concerné.
        """
        return self._id

    def set_cost(self, cost):
        """
        Setter de l'attribut *_cost*.

        Parameters
        ----------
        cost : float
            Valeur du coût du noeud en question. Plus grand ou égal à zero.
        """
        self._cost = cost

    def get_cost(self):
        """
        Getter de l'attribut *_cost*.

        Returns
        -------
        _cost : float
            La valeur du coût du noeud concerné.
        """
        return self._cost

    def set_name(self, name):
        """
        Setter de l'attribut *_name*.

        Parameters
        ----------
        name : str
            Le nouveau nom du noeud en question. 
        """
        self._name = name

    def get_name(self):
        """
        Getter de l'attribut *_name*.

        Returns
        -------
        _name : str
            Le nom du noeud concerné.
        """
        return self._name

    def get_child_by_attribute(self, attr):
        """
        Méthode abstraite qui retournerait l'enfant du noeud correspondant à 
        *attr*.

        Parameters
        ----------
        attr : str
            L'attribut de l'enfant qu'il faut retourner.

        Returns
        -------
        child : NodeST
            L'enfant du noeud concerné qui correspond à l'attribut soumis.
        """
        raise NotImplementedError('Abstract method! Should be implemented in subclass!')

    def set_child_by_attribute(self, attr, child=None):
        """
        Méthode abstraite qui ajouterait un enfant correspondant à *attr* aux
        enfants du noeud concerné.

        Parameters
        ----------
        attr : str
            L'attribut (un type) de l'enfant qui va être ajouté.
        child : NodeST, facultatif
            L'enfant qu'on veut ajouter. Il faut qu'il correspond à l'attribut
            soumis.
        """
        raise NotImplementedError('Abstract method! Should be implemented in subclass!')

    def get_list_of_children(self):
        """
        Méthode abstraite qui permettrait d'obtenir la liste de tous les 
        enfants d'un noeud.

        Returns
        -------
        list_of_children : list(NodeST)
            Liste de tous les enfants d'un noeud.
        """
        raise NotImplementedError('Abstract method! Should be implemented in subclass!')

    def add_child(self, child):
        """
        Méthode abstraite qui ajouterait un enfant au noeud.

        Parameters
        ----------
        child : NodeST
            L'enfant à ajouter.
        """
        raise NotImplementedError('Abstract method! Should be implemented in subclass!')

    def copy(self):
        """
        Retournerait une copie superficielle du noeud.

        Returns
        -------
        copy : NodeST
            Copie superficielle du noeud.
        """
        raise NotImplementedError('Abstract method! Should be implemented in subclass!')

    def __eq__(self, other):
        """
        Overloading de l'opérateur __eq__ ; on dit que deux noeuds sont égaux 
        ssi ils ont les mêmes ids.

        Parameters
        ----------
        other : NodeST
            Le noeud à comparer avec le noeud concerné.

        Returns
        -------
        comp_res : bool
            True si self._id == other._id ET si self et other ont le même type.
            False, sinon.
        """
        if (self is None and other is not None) or (self is not None and other is None):
            return False
        return self._id == other._id and isinstance(other, type(self))

    def __str__(self):
        """
        Overloading de l'opérateur __str__.

        Returns
        -------
        corr_str : str
            La représentation du noeud sous la forme de str.
        """
        return '(' + self._id + ': ' + str(self._cost) + ', ' + self._name + ')'

    def bn_labels_children_association(self):
        """
        Méthode abstraite qui retournera un dictionnaire des associations entre
        les labels du réseau Bayésien et les enfants du noeud.

        Returns
        -------
        da : dict
            Le dictionnaire des associations.
        """
        raise NotImplementedError('Abstract method! Should be implemented in subclass!')


class Repair(NodeST):

    """
    Classe pour répresenter les noeuds des arbres de stratégie correspondants à 
    des actions de réparation.

    Parameters
    ----------
    id : str
        Identificateur unique du noeud.
    cost : float
        Correspond au "coût" du noeud.
    name : str, facultatif
        Nom du noeud qui peut ne pas être unique; si rien a été soumis, on pose
        que *_name* = *_id*.
    child : NodeST, facultatif
        Enfant du noeud, c'est-à-dire, le noeud suivant dans un arbre.

    Attributes
    ----------
    _id : str
        Identificateur unique du noeud.
    _cost : float
        Correspond au "coût" du noeud.
    name : str, facultatif
        Nom du noeud qui peut ne pas être unique; si rien a été soumis, on pose
        que *_name* = *_id*. 
    _child : NodeST
        Enfant du noeud, c'est-à-dire, le noeud suivant dans un arbre.
    """

    def __init__(self, id, cost, name=None, child=None):
        super().__init__(id, cost, name)
        self._child = child

    def set_child(self, child=None):
        """
        Setter de l'attribut *_child*.

        Parameters
        ----------
        child : NodeST, facultatif
            L'enfant du noeud.
        """
        self._child = child

    def get_child(self):
        """
        Getter de l'attribut *_child*.

        Returns
        -------
        _child : NodeST
            L'enfant du noeud concerné.
        """
        return self._child

    def get_child_by_attribute(self, attr):
        """
        Realisation d'une méthode abstraite de la superclass; comme ce type de
        noeud ne dispose que d'un seul enfant on retourne toujours cet enfant 
        pour n'importe quel *attr* passé en argument.

        Parameters
        ----------
        attr : str
             L'attribut de l'enfant qu'il faut retourner. Peut être n'importe
             quoi ici.

        Returns
        -------
        child : NodeST
            L'enfant du noeud concerné.
        """
        return self._child

    def set_child_by_attribute(self, attr, child=None):
        """
        Realisation d'une méthode abstraite de la superclass qui met en place 
        un enfant correspondant à *attr* au noeud concerné. 
        superclass.

        Parameters
        ----------
        attr : str
            L'attribut (un type) de l''enfant qu'il faut mettre en place.
        child : NodeST, facultatif
            L'enfant qu'on veut ajouter. Il faut qu'il correspond à l'attribut
            soumis.    
        """
        self.set_child(child)

    def get_list_of_children(self):
        """
        Retourne la liste qui contient tous les enfants du noeud; pour ce cas,
        soit une liste avec un seul élément, soit une liste vide.

        Returns
        -------
        list_of_children : list(NodeST)
            La liste de tous les enfants du noeud (ici soit une liste avec un 
            seul élément, soit une liste vide).
        """
        return [self._child] if self._child is not None else []

    def add_child(self, child):
        """
        Ajoute un enfant dans la liste des enfants du noeud.
        ATTENTION : ce méthode ne change pas un enfant qui existe déjà; pour 
        cela, veuillez utiliser set_child.

        Parameters
        ----------
        child : NodeST
            L'enfant à ajouter.
        """
        if child is None:
            return
        if self._child is not None:
            raise OverflowError('It is not permitted to add more than one child to this type of nodes!')
        self.set_child(child)

    def copy(self):
        """
        Retourne une copie superficielle du noeud.

        Returns
        -------
        copy : Repair
            Copie superficielle du noeud.
        """
        new_node = Repair(self._id, self._cost, self._name, self._child.copy() if self._child is not None else None)
        return new_node

    def bn_labels_children_association(self):
        """
        Retourne un dictionnaire des associations entre les labels d'un réseau
        Bayésien et les enfants du noeud.

        Returns
        -------
        da : dict
            Dictionnaire des associations concerné.
        """
        return {'': self.get_child()}


class Observation(NodeST):
    """
    Classe pour répresenter les noeuds des arbres de stratégie correspondants à 
    des actions d'observation.

    Parameters
    ----------
    id : str
        Identificateur unique du noeud.
    cost : float
        Correspond au "coût" du noeud.
    name : str, facultatif
        Nom du noeud qui peut ne pas être unique; si rien a été soumis, on pose
        que *_name* = *_id*.        
    yes_child : NodeST, facultatif
        Enfant du noeud qui correspond à la branche "yes".
    no_child : NodeST, facultatif
        Enfant du noeud qui correspond à la branche "no".
    obs_rep_couples : bool, facultatif
        Indique si le noeud représente un couple observation-réparation ou pas.

    Attributes
    ----------
    _id : str
        Identificateur unique du noeud.
    _cost : float
        Correspond au "coût" du noeud.
    _name : str, facultatif
        Nom du noeud qui peut ne pas être unique; si rien a été soumis, on pose
        que *_name* = *_id*.
    _yes_child : NodeST, facultatif
        Enfant du noeud qui correspond à la branche "yes".
    _no_child : NodeST, facultatif
        Enfant du noeud qui correspond à la branche "no".
    _obs_rep_couples : bool, facultatif
        Indique si le noeud représente un couple observation-réparation ou pas.
    """

    def __init__(self, id, cost, name=None, yes_child=None, no_child=None, obs_rep_couples=False):
        super().__init__(id, cost, name)
        self._yes_child = yes_child
        self._no_child = no_child
        self._obs_rep_couples = obs_rep_couples

    def set_yes_child(self, yes_child=None):
        """
        Setter de l'attribut *_yes_child*.

        Parameters
        ----------
        yes_child : NodeST, facultatif
            Enfant du noeud qui correspond à la branche "yes".
        """
        self._yes_child = yes_child

    def get_yes_child(self):
        """
        Getter de l'attribut *_yes_child*.

        Returns
        -------
        _yes_child : NodeST
            Enfant du noeud qui correspond à la branche "yes".
        """
        return self._yes_child

    def set_no_child(self, no_child=None):
        """
        Setter de l'attribut *_no_child*.

        Parameters
        ----------
        no_child : NodeST, facultatif
            Enfant du noeud qui correspond à la branche "no".
        """
        self._no_child = no_child

    def get_no_child(self):
        """
        Getter de l'attribut *_no_child*.

        Returns
        -------
        _no_child : NodeST
            Enfant du noeud qui correspond à la branche "no".
        """
        return self._no_child

    def set_obs_rep_couples(self, obs_rep_couples):
        """
        Setter de l'attribut *_obs_rep_couples*.

        Parameters
        ----------
        obs_rep_couples : bool
            Indique si le noeud représente un couple d'observation-réparation 
            ou pas.
        """
        self._obs_rep_couples = obs_rep_couples

    def get_obs_rep_couples(self):
        """
        Getter de l'attribut *_obs_rep_couples*.

        Returns
        -------
        _obs_rep_couples : bool
            Indique si le noeud représente un couple d'observation-réparation 
            ou pas.
        """
        return self._obs_rep_couples

    def get_child_by_attribute(self, attr):
        """
        Retourne l'enfant du noeud qui correspond à l'attribut passé en 
        argument, c'est-à-dire l'enfant sur branche _yes_child si "yes" est 
        passé en argument et l'enfant sur branche _no_child if "no" est passé.

        Parameters
        ----------
        attr : str
            Indique la branche voulue.
        Returns
        -------
        child : NodeST
            L'enfant correspondant à l'attribut.
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
        Met en place l'enfant correspondant à l'attribut de la branche indiqué
        par *attr*.

        Parameters
        ----------
        attr : str
            Indique la branche voulue.
        child : NodeST, facultatif
            Enfant qui va être mis en place.
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
        Retourne la liste avec tous les enfants du noeud.

        Returns
        -------
        list_of_children : list(NodeST)
            Liste avec tous les enfants du noeud.
        """
        # On ajoute les enfants
        res = [self._yes_child, self._no_child]
        # On supprime tous les valeurs de None
        while None in res:
            res.remove(None)
        return res

    def add_child(self, child):
        """
        Ajoute un enfant dans la liste des enfants du noeud.
        ATTENTION : ce méthode ne change pas un enfant qui existe déjà; pour 
        cela, veuillez utiliser set_child. Par defaut, essaie d'ajouter 
        l'enfant au branche _yes_child d'abord.

        Parameters
        ----------
        child : NodeST
            L'enfant à ajouter.
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
        Retourne une copie superficielle du noeud.

        Returns
        -------
        copy : Observation
            copie superficielle du noeud.
        """
        new_node = Observation(self._id, self._cost, self._name,
                               self._yes_child.copy() if self._yes_child is not None else None,
                               self._no_child.copy() if self._no_child is not None else None,
                               self._obs_rep_couples)
        return new_node

    def bn_labels_children_association(self):
        """
        Retourne un dictionnaire des associations entre les labels d'un réseau
        Bayésien et les enfants du noeud.

        Returns
        -------
        da : dict
            Le dictionnaire des associations concerné.
        """
        return {'no': self.get_no_child(), 'yes': self.get_yes_child()}


class StrategyTree:
    """
    Représente l'arbre de stratégie qui utilisé à la résolution du problème
    de Troubleshooting.

    Parameters
    ----------
    root : NodeST, facultatif
        Racine de l'arbre, i.e. une action pour commencer.
    nodes : list(NodeST), facultatif
        Liste des noeuds de l'arbre.

    Attributes
    ----------
    _root : NodeST
        Racine de l'arbre, i.e. une action pour commencer.
    _nodes : list(NodeST)
        Liste des noeuds de l'arbre.
    _adj_dict : dict
        Dictionnaire qui indique quels noeuds sont liés par des arcs.
    fout_newline : str
        Indique le début d'une nouvelle ligne quand on transforme cet arbre en
        un fichier de texte.
    fout_sep : str
        Séparateur d'attributs qu'on utilise quand on transforme cet arbre en 
        un fichier texte.
    """

    def __init__(self, root=None, nodes=None):

        # une initialisation par défaut qui crée un arbre vide
        self._root = root
        self._nodes = []
        self._adj_dict = {}
        self.fout_newline = '\n'
        self.fout_sep = ','

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
        Réalise la transformation de l'arbre vers str.

        Returns
        -------
        st_str : str
            Représentation de l'arbre de stratégie en forme de str.
        """

        # Plus précisément, on utilise pour cela l'une des méthodes ci-dessous
        return self.str_alt_2()

    def str_alt(self):
        """
        Réalise la transformation de l'arbre vers str.

        Returns
        -------
        st_str : str
            Représentation de l'arbre de stratégie en forme de str.
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
        Transformation alternative de l'arbre vers str.

        Returns
        -------
        st_str : str
            Représentation de l'arbre de stratégie en forme de str de manière 
            alternative.
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
        Setter de l'attribut *_root*.

        Parameters
        ----------
        root : NodeST
            Racine à mettre en place.
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
        Getter de l'attribut *_root*.

        Returns
        -------
        _root : NodeST
            Racine de l'arbre.
        """
        return self._root

    def get_node(self, id):
        """
        Retourne le noeud exacte (en sens de l'objet dans mémoire vivant) de 
        l'arbre avec *id* indiqué.

        Parameters
        ----------
        id : str / NodeST
            Soit *id* du noeud, soit un noeud lui-même dont on cherche un clone 
            (en sens d'id) dans l'arbre.

        Returns
        -------
        n : NodeST
            Noeud de l'arbre avec la *id* soumis.
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
        Retourne tous les noeuds de l'arbre dont les noms sont égaux à celui 
        indiqué.

        Parameters
        ----------
        name : str
            Un nom ou un noeud dont le nom on doit utiliser.

        Returns
        -------
        nodes : list(NodeST)
            Liste de tous les noeuds de l'arbre qui ont le même nom que celui
            indiqué.
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
        Récupère tous les arcs de l'arbre.

        Returns
        -------
        edges : list(tuple(NodeST, NodeST, str))
            Liste de triplets où chaque élement correspond à un arc d'un graphe
            de manière que tuple[0] est un parent, tuple[1] est leur enfant et 
            tuple[2] est l'attribut identifiant la branche.
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
        Permet d'ajouter un ou plusieurs nouveaux noeuds dans l'arbre.

        Parameters
        ----------
        node : NodeST / list(NodeST)
            Noeud.s à ajouter.
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
        Permet d'ajouter un arc dans un arbre entre deux noeuds.

        Parameters
        ----------
        parent : str / NodeST
            Noeud qui va être le parent. Le noeud duquel l'arc part.
        child : str / NodeST
            Noeud qui va être l'enfant. Le noeud auquel l'arc arrive.
        child_type : str, facultatif
            L'attribut de la branche du parent à laquelle il faut ajouter 
            l'enfant (par exemple si parent est une observation alors 
            child_type est égal soit à 'no', soit à 'yes').
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
        Getter de l'attribut *_nodes*.

        Returns
        -------
        _nodes : list(NodeST)
            Copie superficielle de la liste des noeuds de l'arbre.
        """
        return self._nodes.copy()

    def get_adj_dict(self):
        """
        Getter de l'attribut *_adj_dict*.

        Returns
        -------
        _adj_dict : dict
            Copie superficielle du dictionnaire d'adjacence de l'arbre.
        """
        return self._adj_dict.copy()

    def get_sub_tree(self, sub_root):
        """
        Retourne le sous-arbre qui a le noeud *sub_root* comme racine.

        Parameters
        ----------
        sub_root : NodeST
            Racine de le sous-arbre.

        Returns
        -------
        sub_tree : StrategyTree
            Sous-arbre de cet arbre dont la racine est *sub_root*.
        """
        if sub_root is None:
            return None
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
        Retourne le parent du noeud *child* dans l'arbre. Remarque : dans cette
        implémentation d'arbre chaque noeud ne peut avoir qu'un seul parent.

        Parameters
        ----------
        child : str / NodeST
            L'enfant dont le parent on cherche dans l'arbre.

        Returns
        -------
        parent : NodeST
            Parent du noeud *child*.
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
        Supprime le sous-arbre qui a comme racine *sub_root*.
        
        Parameters
        ----------
        sub_root : str / NodeST
            Racine du sous-arbre qu'il faut supprimer.

        Returns
        -------
        flag : bool
            Égale à True si la fonction a supprimé un sous-arbre, False sinon.
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
        Retourne une copie superficielle de l'arbre.

        Returns
        -------
        copy : StrategyTree
            Copie superficielle de l'arbre.
        """
        other_root = self._root.copy() if self._root is not None else None
        other_nodes = []
        for _n in self._nodes:
            other_nodes.append(_n.copy())
        return StrategyTree(root=other_root, nodes=other_nodes)

    def connect(self, root_with_subtree, root_child_type=None):
        """
        Connecte deux arbres, plus précisément, on connecte l'arbre actuel à 
        l'arbre *root_with_subtree* en remplissant la branche qui correspond à 
        root_child_type dans *root_with_subtree*.

        Parameters
        ----------
        root_with_subtree : StrategyTree
            L'arbre vers la racine duquel on va connecter l'arbre actuel.
        root_child_type : str
            L'attribut de la branche de la racine du root_with_subtree.

        Returns
        -------
        merged_tree : StrategyTree
            L'arbre fusionné.
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
        Affiche l'arbre de stratégie via le module graphviz. L'image construit 
        est sauvegardé dans le fichier *filename.pdf*.

        Parameters
        ----------
        filename : str, facultatif
            Le nom du fichier où on sauvegarde l'image.
        """
        vst = Digraph()
        for node in self._nodes:
            node_type = 'Repair' if isinstance(node, Repair) else 'Observation'
            vst.node(node.get_id(), '%s (n%s) : %s' % (node.get_name(), node.get_id(), node_type))
        for par in self._adj_dict.keys():
            for ch in self._adj_dict[par]:
                for attr in self.get_node(par).bn_labels_children_association().keys():
                    if self.get_node(ch) == self.get_node(par).bn_labels_children_association()[attr]:
                        vst.edge(par, ch, label=attr)
        vst.render(filename, view=True)

    def to_file(self, filemame='last_best_tree.txt'):
        """
        Permet de sauvegarder l'arbre de stratégie sous forme de fichier texte.
        On utilise le modèle suivant :
        1) Chaque noeud est représenté par une ligne du type :
        _id,_cost,_name,_type  
        C'est bien possible de remplacer la virgule par un séparateur 
        différent en précisant l'attribut *self.fout_sep* de la classe.
        2) Chaque arc est représenté par une ligne du type :
        _id_parent,_id_child,_attribut 
        Où _attribut est le type d'arc (par exemple 'yes' ou 'no' si parent
        est une Observation).
        3) Le fichier lui-même a la structure suivante :
        racine de l'arbre    # ligne 1
        [ligne vide]         # ligne 2
        noeud_1              # ligne 3
        noeud_2              # ligne 4
        ...
        noeud_n              # ligne n + 2
        [ligne vide]         # ligne n + 3
        arc_1                # ligne n + 4
        arc_2                # ligne n + 5
        ...
        arc_m                # ligne n + m + 3
        Cette méthode utilise également l'attribut self.fout_newline pour 
        représenter le signe qui indique le début d'une nouvelle ligne.

        Parameters
        ----------
        filemame : str, facultatif
            Le nom du fichier où on sauvegarde le texte.
        """
        fout = open(filemame, 'w')
        newline = self.fout_newline
        sep = self.fout_sep
        fout.write(
            self._root.get_id() + sep + str(self._root.get_cost()) + sep + self._root.get_name() + sep +
            ('obs' if isinstance(self._root, Observation) else 'rep') + newline
        )
        fout.write(newline)
        for node in self._nodes:
            fout.write(
                node.get_id() + sep + str(node.get_cost()) + sep + node.get_name() + sep +
                ('obs' if isinstance(node, Observation) else 'rep') + newline
            )
        fout.write(newline)
        for edge in self.get_edges():
            fout.write(
                edge[0].get_id() + sep + edge[1].get_id() + sep + (edge[2] if edge[2] is not None else 'None') +
                newline
            )
        fout.close()


def st_from_file(filename='last_best_tree.txt', sep=',', newline=None):
    """
    Permet de créer un objet du type StrategyTree à partir du fichier indiqué
    par *filename* en suivant le modèle fourni par la méthode 
    StrategyTree.to_file.

    Parameters
    ----------
    filename : str, facultatif
        Nom du fichier où l'arbre est stocké l'arbre sous forme textuelle.
    sep : str, facultatif
        Le séparateur utilisé dans le fichier.
    newline : str, facultatif
        Signe qui indique le début d'une nouvelle ligne.

    Returns
    -------
    stin : StrategyTree
        L'arbre créé à partir des paramètres passés.
    """
    if newline is None:
        newline = '\n'
    fin = open(filename, 'r')
    lines = fin.read().split(newline)
    fin.close()
    line = lines.pop(0).split(sep)
    stin = StrategyTree(root=(
        Repair(line[0], float(line[1]), line[2]) if line[3] == 'rep' else Observation(line[0], float(line[1]), line[2]))
    )
    stin.fout_newline = newline
    stin.fout_sep = sep
    idroot = line[0]

    lines.pop(0)
    line = lines.pop(0)
    while line != '':
        line = line.split(sep)
        if line[0] != idroot:
            stin.add_node(
                Repair(line[0], float(line[1]), line[2]) if line[3] == 'rep'
                else Observation(line[0], float(line[1]), line[2])
            )
        line = lines.pop(0)
    line = lines.pop(0)
    while line != '':
        line = line.split(sep)
        stin.add_edge(line[0], line[1], line[2])
        line = lines.pop(0)
    line = lines.pop(0)
    if line != '':
        ecrin = float(line)
        return stin, ecrin
    return stin,
