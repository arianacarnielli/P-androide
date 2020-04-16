class NodeST:

    def __init__(self, id, cost, name):
        self._id = id
        self._cost = cost
        self._name = name

    def set_id(self, id):
        self._id = id

    def get_id(self):
        return self._id

    def set_cost(self, cost):
        self._cost = cost

    def get_cost(self):
        return self._cost
    
    def set_name(self, name):
        self._name = name

    def get_name(self):
        return self._name

    def get_list_of_children(self):
        raise NotImplementedError('Abstract method! Should be implemented in subclass!')

    def add_child(self, child):
        raise NotImplementedError('Abstract method! Should be implemented in subclass!')

    def copy(self):
        raise NotImplementedError('Abstract method! Should be implemented in subclass!')

    def __eq__(self, other):
        if (self is None and other is not None) or (self is not None and other is None):
            return False
        return self._id == other._id


class Repair(NodeST):

    def __init__(self, id, cost, name, child=None):
        super().__init__(id, cost, name)
        self._child = child

    def set_child(self, child=None):
        self._child = child

    def get_child(self):
        return self._child

    def get_list_of_children(self):
        return [self._child] if self._child is not None else []

    def add_child(self, child):
        if child is None:
            return
        if self._child is not None:
            raise OverflowError('It is not permitted to add more than one child to this type of nodes!')
        self.set_child(child)

    def copy(self):
        new_node = Repair(self._id, self._cost, self._name, self._child)
        return new_node


class Observation(NodeST):

    def __init__(self, id, cost, name, yes_child=None, no_child=None):
        super().__init__(id, cost, name)
        self._yes_child = yes_child
        self._no_child = no_child

    def set_yes_child(self, yes_child=None):
        self._yes_child = yes_child

    def get_yes_child(self):
        return self._yes_child

    def set_no_child(self, no_child=None):
        self._no_child = no_child

    def get_no_child(self):
        return self._no_child

    def get_list_of_children(self):
        res = [self._yes_child, self._no_child]
        while None in res:
            res.remove(None)
        return res

    def add_child(self, child):
        if child is None:
            return
        if self._yes_child is None:
            self.set_yes_child(child)
        elif self._no_child is None:
            self.set_no_child(child)
        else:
            raise OverflowError('It is not permitted to add more than two children to this type of nodes!')

    def copy(self):
        new_node = Observation(self._id, self._cost, self._name, self._yes_child, self._no_child)
        return new_node

    def bn_labels_children_association(self):
        return {'no': self.get_no_child(), 'yes': self.get_yes_child()}


class StrategyTree:

    def __init__(self, root=None, nodes=None):

        self._root = root
        self._nodes = []
        self._adj_dict = {}

        if root is not None:
            if not isinstance(root, NodeST):
                raise TypeError('Type of parameter "root" must be NodeST!')
            self._adj_dict[self._root.get_id()] = [n.get_id() for n in self._root.get_list_of_children()]
            self._nodes.append(self._root)

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

    def set_root(self, root):
        if root is not None:
            if not isinstance(root, NodeST):
                raise TypeError('Type of parameter "root" must be NodeST!')
        self._root = root
        if root is not None and not root.get_id() in [n.get_id() for n in self._nodes]:
            self._nodes.insert(0, self._root)
            self._adj_dict[self._root.get_id()] = [n.get_id() for n in self._root.get_list_of_children()]

    def get_root(self):
        return self._root

    def get_node(self, id):
        if isinstance(id, NodeST):
            id = id.get_id()
        for n in self._nodes:
            if n.get_id() == id:
                return n
        return None

    def get_edges(self):
        res = []
        for k in self._adj_dict.keys():
            for ch in self._adj_dict[k]:
                res.append((self.get_node(k), self.get_node(ch)))
        return res

    def add_node(self, node):
        if node is None:
            return
        if isinstance(node, NodeST):
            if node.get_id() in [n.get_id() for n in self._nodes]:
                raise ValueError('All the nodes\' ids must be unique!')
            self._nodes.append(node)
            self._adj_dict[node.get_id()] = [n.get_id() for n in node.get_list_of_children()]
        elif isinstance(node, list) and sum([isinstance(n, NodeST) for n in node]) == len(node):
            for n in node:
                if n.get_id() in [_n.get_id() for _n in self._nodes]:
                    raise ValueError('All the nodes\' ids must be unique!')
            self._nodes.extend(node)
            for n in node:
                self._adj_dict[n.get_id()] = [ch.get_id() for ch in n.get_list_of_children()]
        else:
            raise TypeError('Type of parameter "node" must be NodeST or list<NodeST>!')

    def add_edge(self, parent, child, child_type=None):
        if not ((isinstance(parent, str) or isinstance(parent, NodeST)) and
                (isinstance(child, str) or isinstance(child, NodeST))):
            raise TypeError('Parent and child must be either str or NodeST!')
        par, ch = self.get_node(parent), self.get_node(child)
        if par is None:
            self.add_node(parent)
        if ch is None:
            self.add_node(child)
        par, ch = self.get_node(parent), self.get_node(child)
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
        return self._nodes.copy()

    def get_adj_dict(self):
        return self._adj_dict.copy()

    def get_sub_tree(self, sub_root):
        nodes = [sub_root]
        nodes_to_process = []
        nodes_to_process.extend(sub_root.get_list_of_children())

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
                nodes_to_process.extend(n.get_list_of_children())
                debug_ntp = [n.get_id() for n in nodes_to_process]

        return StrategyTree(root=sub_root, nodes=nodes)

    def copy(self):
        return StrategyTree(root=self._root, nodes=self._nodes)
