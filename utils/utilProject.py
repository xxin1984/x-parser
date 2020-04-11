from GlobalPara import *

# recover an unary-binary tree into a tree


def my_recover_tree(root_: Tree):
    root = root_
    for node in my_post_order(root):
        i = 0
        while i < len(node):
            child = node[i]
            if my_is_leaf(child) is False and my_is_pre_leaf(child) is False:
                label = child.label()
                if label[-1] == "*":
                    node.pop(i)
                    for j in range(0, len(child)):
                        node.insert(i + j, child[j])
                    i = i + len(child) - 1
            i = i + 1
    for node in my_post_order(root):
        label = node.label().split("_")[0]
        node.set_label(label)

    return root


# def my_generate_g_mask_from_hash_table(height, length, input_data: dict):
#     mask_matrix = np.zeros([height, length, 4], dtype=np.float)
#     for key in input_data.keys():
#         i, j, k = key
#         mask_matrix[i][j][k] = 1
#     return mask_matrix
#
#
# def my_generate_h_mask_from_hash_table(height, length, input_data: dict):
#     mask_matrix = np.zeros([height, length, Paras.HEAD_CAT_SIZE], dtype=np.float)
#     for key in input_data.keys():
#         i, j, k = key
#         mask_matrix[i][j][k] = 1
#     return mask_matrix
#
#
# def my_generate_w_mask_from_hash_table(length, height, input_data: dict):
#     mask_matrix = np.zeros([length, height, Paras.MY_HLSTM_HIDDEN_DIM], dtype=np.int32)
#     for key in input_data.keys():
#         i, j = key
#         mask_matrix[i][j] = np.ones([Paras.MY_HLSTM_HIDDEN_DIM], dtype=np.int32)
#     return mask_matrix