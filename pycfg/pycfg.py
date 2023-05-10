import ast
import re
import astunparse
import pygraphviz


class CFGNode(dict):
    # 用于为节点分配唯一ID的静态变量
    registry = 0
    # 一个以registry为键，以CFGNode对象为值的字典
    # 这允许你快速查找节点对象
    cache = {}
    # 在DFS算法中访问节点时用的堆栈
    stack = []

    def __init__(self, parents=[], ast=None):
        assert type(parents) is list
        # 当前结点的父结点列表
        self.parents = parents
        # 在当前节点上调用的函数列表
        self.calls = []
        # 当前节点的子节点的列表
        self.children = []
        # 当前节点所对应的抽象语法树（AST）节点
        self.ast_node = ast
        # 代表节点唯一ID的变量
        self.rid = CFGNode.registry
        CFGNode.cache[self.rid] = self
        CFGNode.registry += 1

    # 返回该节点所对应的代码行的编号
    # 如果没有行号，则返回 0
    def lineno(self):
        return self.ast_node.lineno if hasattr(self.ast_node, 'lineno') else 0

    # 将一个节点表示为一个字符串
    def __str__(self):
        return "id:%d line[%d] parents: %s : %s" % (self.rid, self.lineno(), str([p.rid for p in self.parents]), self.source())

    def __repr__(self):
        return str(self)

    # 添加子节点
    def add_child(self, c):
        if c not in self.children:
            self.children.append(c)

    # 比较节点之间是否有相同的ID
    def __eq__(self, other):
        return self.rid == other.rid

    def __neq__(self, other):
        return self.rid != other.rid

    # 设置父节点
    def set_parents(self, p):
        self.parents = p

    # 添加父节点
    def add_parent(self, p):
        if p not in self.parents:
            self.parents.append(p)

    # 添加多个父节点
    def add_parents(self, ps):
        for p in ps:
            self.add_parent(p)

    # 添加被调用函数
    def add_calls(self, func):
        self.calls.append(func)

    # 以字符串形式返回节点的代码
    def source(self):
        return astunparse.unparse(self.ast_node).strip()

    # 以 JSON 形式返回节点的信息
    def to_json(self):
        return {'id': self.rid, 'parents': [p.rid for p in self.parents], 'children': [c.rid for c in self.children], 'calls': self.calls, 'at': self.lineno(), 'ast': self.source()}

    # 返回一个AGraph对象，它将所有节点表示成图
    # arcs参数：接收一个要显示的边的列表。 你可以用这个参数添加行覆盖信息
    # 每个节点的标签（label）包含行号和源代码，每个边缘的颜色取决于行覆盖信息
    @classmethod
    def to_graph(cls, arcs=[]):
        # 为了将代码可视化为一张图，使节点重新命名，使它们看起来更好看
        # 例如，将 '_if' 重命名为 'if'
        def unhack(v):
            for i in ['if', 'while', 'for', 'elif']:
                v = re.sub(r'^_%s:' % i, '%s:' % i, v)
            return v

        G = pygraphviz.AGraph(directed=True)  # 给图进行初始化
        cov_lines = set(i for i, j in arcs)
        # 使用 CFGNode.cache.items() 获取所有的节点
        # 将每个节点添加到图中，添加标签和边
        for nid, cnode in CFGNode.cache.items():
            G.add_node(cnode.rid)  # 将每个节点添加到图
            n = G.get_node(cnode.rid)
            lineno = cnode.lineno()
            # 对于每个节点，添加一个标签和一条边
            # 标签包含该节点所代表的源代码的行号和语句
            n.attr['label'] = "%d: %s" % (lineno, unhack(cnode.source()))
            # 给每个节点添加一条边，将其与父节点连接起来
            for pn in cnode.parents:
                plineno = pn.lineno()
                # 如果当前的父节点是函数的调用节点，并且当前节点不是调用节点的子节点，
                # 则在该父节点和当前节点之间添加一条虚线，并进入下一次迭代。
                if hasattr(pn, 'calllink') and pn.calllink > 0 and not hasattr(cnode, 'calleelink'):
                    G.add_edge(pn.rid, cnode.rid, style='dotted', weight=100)
                    continue

                # 你可以在执行流程中检查测试覆盖率
                # arcs是一组被执行的线的图元（tuple），我们根据 arcs，对边缘进行不同的着色
                # 如果存在一个名为 arcs 的变量，并且满足以下条件之一（就是if语句），
                # 则在其父节点和当前节点之间添加一条蓝色的边
                if arcs:
                    if (plineno, lineno) in arcs:
                        G.add_edge(pn.rid, cnode.rid, color='blue')
                    elif plineno == lineno and lineno in cov_lines:
                        G.add_edge(pn.rid, cnode.rid, color='blue')
                    # child is exit and parent is covered
                    elif hasattr(cnode, 'fn_exit_node') and plineno in cov_lines:
                        G.add_edge(pn.rid, cnode.rid, color='blue')
                    # parent is exit and one of its parents is covered.
                    elif hasattr(pn, 'fn_exit_node') and len(set(n.lineno() for n in pn.parents) | cov_lines) > 0:
                        G.add_edge(pn.rid, cnode.rid, color='blue')
                    # child is a callee (has calleelink) and one of the parents is covered.
                    elif plineno in cov_lines and hasattr(cnode, 'calleelink'):
                        G.add_edge(pn.rid, cnode.rid, color='blue')
                    else:
                        G.add_edge(pn.rid, cnode.rid, color='red')
                else:
                    G.add_edge(pn.rid, cnode.rid)
        return G


class PyCFG:

    def __init__(self):
        self.founder = CFGNode(
            parents=[], ast=ast.parse('start').body[0])  # sentinel
        self.founder.ast_node.lineno = 0
        self.functions = {}
        self.functions_node = {}

    # 解析作为参数传递的源代码字符串并返回一个抽象语法树（AST）节点。
    def parse(self, src):
        return ast.parse(src)

    # 递归地遍历 AST 节点，为每个节点类型调用函数。
    # 如果该节点类型没有函数存在，则返回 myparents。
    def walk(self, node, myparents):
        if node is None:
            return
        # 根据当前节点的类名，生成一个要调用的函数名
        # 例如，如果节点的类名是 Module,，fname 将存储 "on_module"。
        fname = "on_%s" % node.__class__.__name__.lower()
        # 如果有一个与 fname 同名的属性（定义了一个与 on_module 同名的函数）
        if hasattr(self, fname):
            fn = getattr(self, fname)
            # 例子，调用 on_module 函数
            v = fn(node, myparents)
            return v
        else:
            return myparents

    # 处理 module 节点
    # 返回值是构建好的 CFG，它是一个由 CFGNode 对象组成的列表，
    # 每个 CFGNode 对象代表一个语句或者一个基本块，它包含了该语句或基本块的所有信息，如语句的 AST 表示、父节点和子节点等
    def on_module(self, node, myparents):
        """
        Module(stmt* body)
        """
        # each time a statement is executed unconditionally, make a link from
        # the result to next statement
        # 执行 Module 节点中的每一个子节点，即遍历 node.body 中的每个语句，并且将每个语句与其后面的语句连接起来
        p = myparents
        # 调用 walk 函数来处理每个语句，并将处理的结果传递给下一个语句
        for n in node.body:
            p = self.walk(n, p)
        return p

    # 处理赋值（assign）节点
    # 返回一个以 value 节点为父节点的 CFGNode 对象
    def on_assign(self, node, myparents):
        """
        Assign(expr* targets, expr value)
        TODO: AugAssign(expr target, operator op, expr value)
        -- 'simple' indicates that we annotate simple name without parens
        TODO: AnnAssign(expr target, expr annotation, expr? value, int simple)
        """
        # target: 位于赋值语句左边的变量的节点列表
        if len(node.targets) > 1:
            raise NotImplemented('Parallel assignments')

        p = [CFGNode(parents=myparents, ast=node)]
        # value: 位于赋值语句右边的表达式的节点
        p = self.walk(node.value, p)

        return p

   # 处理 pass 节点
   # 返回一个以 myparents 为父母的 CFGNode 对象
    def on_pass(self, node, myparents):
        return [CFGNode(parents=myparents, ast=node)]

    # 处理 break 节点
    # exit_nodes: 当在一个循环语句中执行 break 或continue语句时，这个列表包含了该循环语句的终点信息。这个列表包含了该循环语句的所有路径的节点。
    def on_break(self, node, myparents):
        parent = myparents[0]
        while not hasattr(parent, 'exit_nodes'):
            parent = parent.parents[0]

        assert hasattr(parent, 'exit_nodes')
        p = CFGNode(parents=myparents, ast=node)

        # make the break one of the parents of label node.
        parent.exit_nodes.append(p)

        # break doesnt have immediate children
        return []

    # 处理 continue 节点
    def on_continue(self, node, myparents):
        parent = myparents[0]
        while not hasattr(parent, 'exit_nodes'):
            parent = parent.parents[0]

        assert hasattr(parent, 'exit_nodes')
        p = CFGNode(parents=myparents, ast=node)

        # make continue one of the parents of the original test node.
        parent.add_parent(p)

        # return the parent because a continue is not the parent
        # for the just next node
        return []

    # 处理 for 节点
    def on_for(self, node, myparents):
        # node.target in node.iter: node.body
        # _test_node 对象将代表重复的条件，直到循环的结束。
        # 以 myparents 列表为父节点。有一个 AST 节点，将 node.iter 评估为 True 或 False 。
        _test_node = CFGNode(parents=myparents, ast=ast.parse(
            '_for: True if %s else False' % astunparse.unparse(node.iter).strip()).body[0])
        ast.copy_location(_test_node.ast_node, node)

        # we attach the label node here so that break can find it.
        # 用于调用 walk 函数来评估 node.iter
        _test_node.exit_nodes = []
        test_node = self.walk(node.iter, [_test_node])

        # 以 _test_node 作为它的父节点。有一个AST节点，将 node.target 分配给 node.iter 的下一个值。
        # extract_node 对象负责分配 for 循环中使用的迭代变量的值。
        extract_node = CFGNode(parents=[_test_node], ast=ast.parse('%s = %s.shift()' % (
            astunparse.unparse(node.target).strip(), astunparse.unparse(node.iter).strip())).body[0])
        ast.copy_location(extract_node.ast_node, _test_node.ast_node)

        # now we evaluate the body, one at a time.
        # 对于 node.body 中的每个句子，通过调用 self.walk 函数来执行它。
        # walk 函数接收一个 p1 列表作为参数，其中包含了以前走过的节点。
        # walk 函数创建一个新的节点，将其添加到以前走过的节点的 exit_nodes 属性中，并返回新节点的 exit_nodes 属性。
        p1 = [extract_node]
        for n in node.body:
            p1 = self.walk(n, p1)

        # the test node is looped back at the end of processing.
        # 添加 _test_node 为 p1 的父节点。这将导致循环结束，并且再次检查 _test_node 的条件。
        _test_node.add_parents(p1)

        return _test_node.exit_nodes + test_node

    # 处理 while 节点
    def on_while(self, node, myparents):
        # For a while, the earliest parent is the node.test
        # _test_node 将被用来评估循环中的条件表达式。
        # 其父节点被设置为当前节点的父节点（myparents）。
        # ast 属性添加了一个 AST 节点，用于评估循环的条件表达式。这个 AST 节点是用 ast.parse() 函数创建的。
        _test_node = CFGNode(parents=myparents, ast=ast.parse(
            '_while: %s' % astunparse.unparse(node.test).strip()).body[0])
        ast.copy_location(_test_node.ast_node, node.test)
        # exit_nodes 列表属性存储了循环退出时的节点。
        _test_node.exit_nodes = []
        test_node = self.walk(node.test, [_test_node])

        # we attach the label node here so that break can find it.

        # now we evaluate the body, one at a time.
        # 设置第一个父节点（p1）来处理 _test_node 和 node.body 节点。
        p1 = test_node
        # 迭代更新 p1 中的列表以处理 _test_node 和 node.body 节点。
        # 在这种情况下，_test_node 被用来作为评估循环的条件表达式的节点。
        for n in node.body:
            p1 = self.walk(n, p1)

        # the test node is looped back at the end of processing.
        _test_node.add_parents(p1)

        # link label node back to the condition.
        return _test_node.exit_nodes + test_node

    # 处理 if 节点
    def on_if(self, node, myparents):
        # 测试节点 _test_node: 父节点是 myparents；AST节点是一个 if 语句，表示测试节点的内容是执行条件测试语句。
        # 将 node.test 作为测试语句，并将 AST 节点添加到 _test_node.ast_node
        _test_node = CFGNode(parents=myparents, ast=ast.parse(
            '_if: %s' % astunparse.unparse(node.test).strip()).body[0])
        # 将测试节点的位置信息复制到源代码中的测试语句上
        ast.copy_location(_test_node.ast_node, node.test)
        # 对测试节点进行遍历，得到测试节点的所有子节点，存储在 test_node 中
        test_node = self.walk(node.test, [_test_node])
        # 对 if 语句的主体部分进行遍历，将遍历结果存储在 g1 中
        g1 = test_node
        for n in node.body:
            g1 = self.walk(n, g1)
        # 对 if 语句的 else 部分进行遍历，将遍历结果存储在 g2 中
        g2 = test_node
        for n in node.orelse:
            g2 = self.walk(n, g2)

        # 将 g1 和 g2 拼接起来，返回这个列表。这些节点将成为下一步控制流图的起始节点
        return g1 + g2

    # 处理二元运算符
    # 将左右操作数（left和right）分别传递给 walk 函数进行处理
    def on_binop(self, node, myparents):
        left = self.walk(node.left, myparents)
        right = self.walk(node.right, left)
        return right

    # 处理比较运算符
    # 将左操作数（left）和第一个比较器（comparators[0]）分别传递给 walk 函数进行处理
    def on_compare(self, node, myparents):
        left = self.walk(node.left, myparents)
        right = self.walk(node.comparators[0], left)
        return right

    # 处理一元运算符
    # 将操作数（operand）传递给 walk 函数进行处理
    def on_unaryop(self, node, myparents):
        return self.walk(node.operand, myparents)

    # 处理函数调用（call）节点
    def on_call(self, node, myparents):
        # 从函数调用节点中获取函数的名称。
        # 根据节点的类型分别获取名称，如果节点的类型是 ast.Call，则递归调用该函数以获取函数名称。
        def get_func(node):
            if type(node.func) is ast.Name:
                mid = node.func.id
            elif type(node.func) is ast.Attribute:
                mid = node.func.attr
            elif type(node.func) is ast.Call:
                mid = get_func(node.func)
            else:
                raise Exception(str(type(node.func)))
            return mid
            # mid = node.func.value.id

        p = myparents
        # 对于每个函数调用的参数，使用 walk 函数递归地处理它们，
        # 并将它们添加到变量 p 中。
        for a in node.args:
            p = self.walk(a, p)
        # 调用 get_func 函数，获取函数名称并将其赋值给变量 mid
        mid = get_func(node)
        # 将函数名称添加到父节点的函数调用列表中。
        myparents[0].add_calls(mid)

        # these need to be unlinked later if our module actually defines these
        # functions. Otherwsise we may leave them around.
        # during a call, the direct child is not the next
        # statement in text.
        # 将 p 中的所有节点的 calllink 属性设置为 0，以便稍后删除它们
        for c in p:
            c.calllink = 0
        return p

    # 处理表达式语句节点
    def on_expr(self, node, myparents):
        p = [CFGNode(parents=myparents, ast=node)]
        # 遍历表达式值以计算其父节点
        return self.walk(node.value, p)

    # 处理 return 节点
    def on_return(self, node, myparents):
        # 获取最近的父节点
        parent = myparents[0]

        # 调用 walk 函数处理 return 语句中的值
        val_node = self.walk(node.value, myparents)
        # on return look back to the function definition.
        # 在 parent 中查找 return_nodes 属性，该属性是函数节点定义时创建的，用于记录所有的 return 语句节点。
        # 如果没有找到该属性，则一直向上查找父节点，直到找到该属性为止。
        while not hasattr(parent, 'return_nodes'):
            parent = parent.parents[0]
        assert hasattr(parent, 'return_nodes')

        p = CFGNode(parents=val_node, ast=node)

        # make the break one of the parents of label node.
        # 将新节点 p 加入 parent 的 return_nodes 列表中，表示该 return 语句的结束
        parent.return_nodes.append(p)

        # return doesnt have immediate children
        return []

    # 处理 def 节点
    def on_functiondef(self, node, myparents):
        # a function definition does not actually continue the thread of control flow
        # 函数名、参数、内容、装饰器、返回值
        fname = node.name
        args = node.args
        returns = node.returns

        # 构造入口节点与出口节点
        enter_node = CFGNode(parents=[], ast=ast.parse('enter: %s(%s)' % (
            node.name, ', '.join([a.arg for a in node.args.args]))).body[0])  # sentinel
        enter_node.calleelink = True
        ast.copy_location(enter_node.ast_node, node)
        exit_node = CFGNode(parents=[], ast=ast.parse('exit: %s(%s)' % (
            node.name, ', '.join([a.arg for a in node.args.args]))).body[0])  # sentinel
        exit_node.fn_exit_node = True
        ast.copy_location(exit_node.ast_node, node)
        enter_node.return_nodes = []  # sentinel

        # 遍历节点并进行连接
        p = [enter_node]
        for n in node.body:
            p = self.walk(n, p)
        for n in p:
            if n not in enter_node.return_nodes:
                enter_node.return_nodes.append(n)
        for n in enter_node.return_nodes:
            exit_node.add_parent(n)
        self.functions[fname] = [enter_node, exit_node]
        self.functions_node[enter_node.lineno()] = fname
        return myparents

    # 给定节点函数名
    def get_defining_function(self, node):
        # 检查节点行号是否在function_node中
        if node.lineno() in self.functions_node:
            return self.functions_node[node.lineno()]
        if not node.parents:
            self.functions_node[node.lineno()] = ''
            return ''
        val = self.get_defining_function(node.parents[0])
        self.functions_node[node.lineno()] = val
        return val

    # 连接函数调用
    def link_functions(self):
        for nid, node in CFGNode.cache.items():
            # 检查函数调用
            if node.calls:
                for calls in node.calls:
                    if calls in self.functions:
                        enter, exit = self.functions[calls]
                        enter.add_parent(node)
                        if node.children:
                            assert node.calllink > -1
                            # 当前节点包含一个函数调用，但还没有与该调用的返回值关联
                            node.calllink += 1
                            for i in node.children:
                                i.add_parent(exit)

    # 更新函数信息
    def update_functions(self):
        for nid, node in CFGNode.cache.items():
            _n = self.get_defining_function(node)

    # 更新子节点信息
    def update_children(self):
        for nid, node in CFGNode.cache.items():
            for p in node.parents:
                p.add_child(node)

    # 产生cfg流程图
    def gen_cfg(self, src):
        """
        >>> i = PyCFG()
        >>> i.walk("100")
        5
        """
        node = self.parse(src)
        nodes = self.walk(node, [self.founder])
        self.last_node = CFGNode(parents=nodes, ast=ast.parse('stop').body[0])
        ast.copy_location(self.last_node.ast_node, self.founder.ast_node)
        self.update_children()
        self.update_functions()
        self.link_functions()


# deminator算法
def compute_dominator(cfg, start=0, key='parents'):
    dominator = {}
    dominator[start] = {start}
    all_nodes = set(cfg.keys())
    rem_nodes = all_nodes - {start}
    for n in rem_nodes:
        dominator[n] = all_nodes

    # 对于每个前驱节点，算法从其支配者集合中取出所有集合的交集，
    # 并将该节点本身加入到其中，从而得到该节点的新的支配者集合。
    # 如果该节点的支配者集合发生了变化，继续迭代直到不再发生变化。
    c = True
    while c:
        c = False
        for n in rem_nodes:
            pred_n = cfg[n][key]
            doms = [dominator[p] for p in pred_n]
            i = set.intersection(*doms) if doms else set()
            v = {n} | i
            if dominator[n] != v:
                c = True
            dominator[n] = v
    # 返回一个字典，其中每个键值对表示一个节点及其对应的支配者集合
    return dominator


# 接受文件名参数'f'
def slurp(f):
    with open(f, 'r') as f:
        return f.read()


# 控制流图生成器
def get_cfg(pythonfile):
    cfg = PyCFG()
    cfg.gen_cfg(slurp(pythonfile).strip())
    cache = CFGNode.cache
    g = {}
    for k, v in cache.items():
        j = v.to_json()
        at = j['at']
        parents_at = [cache[p].to_json()['at'] for p in j['parents']]
        children_at = [cache[c].to_json()['at'] for c in j['children']]
        if at not in g:
            g[at] = {'parents': set(), 'children': set()}
        # remove dummy nodes
        ps = set([p for p in parents_at if p != at])
        cs = set([c for c in children_at if c != at])
        g[at]['parents'] |= ps
        g[at]['children'] |= cs
        if v.calls:
            g[at]['calls'] = v.calls
        g[at]['function'] = cfg.functions_node[v.lineno()]
    return (g, cfg.founder.ast_node.lineno, cfg.last_node.ast_node.lineno)


# 计算控制流支配与前驱支配关系
def compute_flow(pythonfile):
    cfg, first, last = get_cfg(pythonfile)
    return cfg, compute_dominator(cfg, start=first), compute_dominator(cfg, start=last, key='children')


if __name__ == '__main__':
    import json
    import sys
    import argparse
    # 获取命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('pythonfile', help='The python file to be analyzed')
    parser.add_argument('-d', '--dots', action='store_true',
                        help='generate a dot file')
    parser.add_argument('-c', '--cfg', action='store_true', help='print cfg')
    parser.add_argument('-x', '--coverage', action='store',
                        dest='coverage', type=str, help='branch coverage file')
    parser.add_argument('-y', '--ccoverage', action='store',
                        dest='ccoverage', type=str, help='custom coverage file')
    args = parser.parse_args()
    # 执行命令行指令
    if args.dots:
        arcs = None
        if args.coverage:
            cdata = coverage.CoverageData()
            cdata.read_file(filename=args.coverage)
            arcs = [(abs(i), abs(j))
                    for i, j in cdata.arcs(cdata.measured_files()[0])]
        elif args.ccoverage:
            arcs = [(i, j) for i, j in json.loads(open(args.ccoverage).read())]
        else:
            arcs = []
        # 生成cfg图
        cfg = PyCFG()
        cfg.gen_cfg(slurp(args.pythonfile).strip())
        g = CFGNode.to_graph(arcs)  # 对 g 进行 DFS 搜索所有可能路径的数量
        g.draw(args.pythonfile + '.png', prog='dot')
        print(g.string(), file=sys.stderr)
    elif args.cfg:
        # 输出cfg图
        cfg, first, last = get_cfg(args.pythonfile)
        for i in sorted(cfg.keys()):
            print(i, 'parents:', cfg[i]['parents'],
                  'children:', cfg[i]['children'])
