from GlobalPara import Paras
from Evaluation import MyEvaluation
from myAlgorithm import MyAlgorithm
from utils.utilTorch import *


if __name__ == '__main__':
    CUDA = 0
    DIRNAME = "output/test"
    env = Environment(DIRNAME, CUDA)
    Paras.init()
    model = MyAlgorithm().cuda()

    check_file = "/home/xxin/self-attentive-parser/local_model_sp/output/t7051/checkpoints/e68.ckpt"
    my_load_checkpoint(check_file, model)

    model.eval()
    seg_f1, pos_f1, cst_f1, spn_prf1, preds = MyEvaluation.get_loss(model)
    content = " seg f1:" + str(seg_f1) + \
              " pos f1:" + str(pos_f1) + \
              " cst f1:" + str(cst_f1) + \
              " spn f1:" + str(spn_prf1) + "\n"
    print(content)
    env.save_print(content)
    env.save_prediction(-1, preds)
