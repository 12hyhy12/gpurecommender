from .expr_functor import ExprFunctor
from . import expr as _expr
import networkx as nx
class VisualizeExpr(ExprFunctor):
    def __init__(self):
        super().__init__()
        self.graph = nx.DiGraph()
        self.counter = 0
    def viz(self, expr):
        #assert isinstance(expr, _expr.Function)
        for param in expr.params:
            self.visit(param)
        return self.visit(expr.body)
    def visit_constant(self, const): # overload this!
        pass
    def visit_var(self, var):
        name = var.name_hint
        self.graph.add_node(name)
        self.graph.nodes[name]['style'] = 'filled'
        self.graph.nodes[name]['fillcolor'] = 'mistyrose'
        return var.name_hint
    def visit_tuple_getitem(self, get_item):
        tuple = self.visit(get_item.tuple_value)
        # self.graph.nodes[tuple]
        index = get_item.index
        # import pdb; pdb.set_trace()
        return tuple
    def visit_call(self, call):
        parents = []
        for arg in call.args:
            parents.append(self.visit(arg))
        # assert isinstance(call.op, _expr.Op)
        name = "{}({})".format(call.op.name, self.counter)
        self.counter += 1
        self.graph.add_node(name)
        self.graph.nodes[name]['style'] = 'filled'
        self.graph.nodes[name]['fillcolor'] = 'turquoise'
        self.graph.nodes[name]['shape'] = 'diamond'
        edges = []
        for i, parent in enumerate(parents):
            edges.append((parent, name, { 'label': 'arg{}'.format(i) }))
        self.graph.add_edges_from(edges)
        return name
def visualize(expr,mydir="relay_ir.png"):
    viz_expr = VisualizeExpr()
    viz_expr.viz(expr)
    graph = viz_expr.graph
    dotg = nx.nx_pydot.to_pydot(graph)
    dotg.write_png(mydir)
