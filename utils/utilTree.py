from nltk import Tree


def my_is_tree_same(str_input1, str_input2):
    root1 = Tree.fromstring(str_input1)
    root2 = Tree.fromstring(str_input2)
    str_output1 = my_oneline(root1)
    str_output2 = my_oneline(root2)
    if str_output1 == str_output2:
        return True
    else:
        return False


def my_delete_unary(str_input):
    tree_obj = Tree.fromstring(str_input)
    for node in my_post_order(tree_obj):
        for i in range(0, len(node)):
            child = node[i]
            if len(child) == 1 and not my_is_pre_leaf(child):
                node[i] = child[0]

    if len(tree_obj) == 1:
        tree_obj = tree_obj[0]
    return my_oneline(tree_obj)


def my_simplify_unary(str_input):
    tree_obj = Tree.fromstring(str_input)
    for node in my_post_order(tree_obj):
        if len(node) == 1 and not my_is_leaf(node[0]):
            if len(node[0]) == 1 and not my_is_leaf(node[0][0]):
                node[0] = node[0][0]

            if node.label() == node[0].label():
                tmp_node = node[0]
                node[0] = tmp_node[0]
                for i in range(1, len(tmp_node)):
                    node.append(tmp_node[i])
    return my_oneline(tree_obj)


def my_unbinarize(tree_str):
    root = Tree.fromstring(tree_str)
    for node in my_post_order(root):
        if not my_is_pre_leaf(node):
            for child_ix in range(0, len(node)):
                tag = node[child_ix].label()
                if tag[len(tag) - 1] == "*":
                    tmp_node = node[child_ix]
                    node[child_ix] = tmp_node[0]
                    for j in range(1, len(tmp_node)):
                        node.insert(child_ix + j, tmp_node[j])
    for node in my_post_order(root):
        label = node.label()
        node.set_label(label[0: len(label) - 2])
    return my_oneline(root)


def my_unbinarize2(tree_str):
    root = Tree.fromstring(tree_str)
    for node in my_post_order(root):
        if not my_is_pre_leaf(node):
            for child_ix in range(0, len(node)):
                tag = node[child_ix].label()
                if tag[len(tag) - 1] == "*":
                    if tag[0:tag.index("_")] == node.label()[0:node.label().index("_")]:
                        tmp_node = node[child_ix]
                        node[child_ix] = tmp_node[0]
                        for j in range(1, len(tmp_node)):
                            node.insert(child_ix + j, tmp_node[j])
                    else:
                        node[child_ix].set_label(tag[0:len(tag) - 1])
    for node in my_post_order(root):
        label = node.label()
        node.set_label(label[0: label.index("_")])
    return my_oneline(root)


def my_is_leaf(node):
    return not isinstance(node, Tree)


def my_is_pre_leaf(node):
    return (not my_is_leaf(node)) and all(my_is_leaf(child) for child in node)


def my_post_order(node):
    pointer_list = []
    _recursive_post_order(node, pointer_list)
    return pointer_list


def my_post_order_span_ix(node):  # not consider the leaf node
    span_list = []
    nodeAddr_2_span = {}
    index = 0
    for node in my_post_order(node):
        if my_is_pre_leaf(node):
            node_addr = id(node)
            ix_pair = (index, index + 1)
            nodeAddr_2_span[node_addr] = ix_pair
            index = index + 1
            span_list.append(ix_pair)
        else:
            node_addr = id(node)
            ix_pair = (nodeAddr_2_span.get(id(node[0]))[0],
                       nodeAddr_2_span.get(id(node[-1]))[1])
            nodeAddr_2_span[node_addr] = ix_pair
            span_list.append(ix_pair)
    return span_list


def my_post_order_span_ix_leaf(node):  # consider the leaf node
    span_list = []
    nodeAddr_2_span = {}
    index = 0
    for node in my_post_order(node):
        if my_is_pre_leaf(node):
            node_addr = id(node)
            ix_pair = (index, index + len(node[0]))
            nodeAddr_2_span[node_addr] = ix_pair
            index = index + len(node[0])
            span_list.append(ix_pair)
        else:
            node_addr = id(node)
            ix_pair = (nodeAddr_2_span.get(id(node[0]))[0],
                       nodeAddr_2_span.get(id(node[-1]))[1])
            nodeAddr_2_span[node_addr] = ix_pair
            span_list.append(ix_pair)
    return span_list


def _recursive_post_order(node, pointer_list):
    if not my_is_leaf(node):
        for child in node:
            _recursive_post_order(child, pointer_list)
        pointer_list.append(node)


def my_oneline(t):
    return t._pformat_flat(nodesep='', parens='()', quotes=False)


def my_oneline_combine(label, child_str_list: list):
    children_str = ' '.join(child_str_list)
    node_str = f'({label} {children_str})'
    return node_str


def my_oneline_split(tree_str):
    elems = tree_str[1:-1].split(' ')
    head_tag = elems[0]
    child_strs = elems[1:]
    return head_tag, child_strs

