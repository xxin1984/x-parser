import torch
import os
from utils.utilGeneral import *
import random


def my_is_NAN(input_list: list):
    for ts in input_list:
        a = torch.max(ts).item()
        if np.isnan(a):
            return True
        b = torch.min(ts).item()
        if np.isnan(b):
            return True
    return False


def my_soft_max_2(input: torch.tensor):
    denominator = torch.logsumexp(input, 1)
    denominator_ex = denominator.unsqueeze(1).expand(-1, input.size(1))
    output1 = input - denominator_ex
    return torch.exp(output1)


def my_shift_right(input: torch.tensor):
    b = input[0:len(input) - 1]
    c = torch.cat([input[len(input) - 1:len(input)], b], 0)
    return c


def my_reverse_tensor(input: torch.tensor):
    idx = [i for i in range(input.size(0)-1, -1, -1)]
    idx = torch.tensor(idx, dtype=torch.long)
    return torch.index_select(input, 0, idx)


def my_save_checkpoint(ckpt_file, model):
    print('Saving Checkpoint', ckpt_file)
    try:
        torch.save(model.state_dict(), ckpt_file)
    except Exception as err:
        print('Fail to save checkpoint', ckpt_file)
        print('Error:', err)


def my_load_checkpoint(ckpt_file, model):
    print('Loading Checkpoint', ckpt_file)
    state_dict = torch.load(ckpt_file)
    model.load_state_dict(state_dict)


def my_decay_lr(optimizer, epoch, init_lr, decay_rate):
    lr = init_lr * ((1 - decay_rate) ** epoch)
    if lr < 0.0001:
        lr = 0.0001
    print('Learning Rate is setted as:', lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def my_set_lr(optimizer, lr_input):
    print('Learning Rate is setted as:', lr_input)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_input
    return optimizer


def my_clip(model, max_norm=5.0):
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


class Environment:
    """
    存储布局
    model_dir/
        checkpoints/
            - e{e}.s{s}.ckpt
        evaluation/
            - output.txt
        prediction/
            - e{e}.dev.txt
            - e{e}.test.txt
        src/
            *.py
    """

    def __init__(self, model_dir, cuda):
        assert not os.path.exists(model_dir), f'目录已存在 {model_dir}'
        model_dir = os.path.realpath(model_dir)
        self.model_dir = model_dir  # 存储根目录
        self.ckpt_dir = os.path.join(model_dir, 'checkpoints')  # 检查点目录
        self.eval_dir = os.path.join(model_dir, 'evaluation')  # 评估结果
        self.pred_dir = os.path.join(model_dir, 'prediciton')  # 预测结果
        self.src_dir = os.path.join(model_dir, 'src')  # 运行时的源码
        self.eval_file = os.path.join(self.eval_dir, "output.txt")

        os.mkdir(self.model_dir)
        os.mkdir(self.ckpt_dir)
        os.mkdir(self.eval_dir)
        os.mkdir(self.pred_dir)
        os.mkdir(self.src_dir)

        my_write_file(self.eval_file, "")
        self.copy_src()
        print(model_dir)

        os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda)
        torch.manual_seed(2)
        np.random.seed(2)
        torch.cuda.manual_seed_all(2)
        random.seed(2)

        torch.set_default_tensor_type(torch.cuda.FloatTensor)

    def copy_src(self):
        """
        复制源码
        :return:
        """
        proj_dir = os.getcwd()
        src_files = os.path.join(proj_dir, '*.py')
        cmd = f'cp {src_files} {self.src_dir}'
        print('Copy Source Code:', cmd)
        os.system(cmd)
        #src_dir = os.path.dirname(os.path.realpath(__file__))
        src_dir = os.path.join(proj_dir, "utils")
        src_files = os.path.join(src_dir, '*.py')
        cmd = f'cp {src_files} {self.src_dir}'
        print('Copy Source Code:', cmd)
        os.system(cmd)

    def save_checkpoint(self, epoch: int, model):
        """
        保存检查点 checkpoints/e{e}.s{s}.ckpt
        """
        ckpt_file = os.path.join(self.ckpt_dir, f'e{epoch}.ckpt')
        try:
            torch.save(model.state_dict(), ckpt_file)
            print('Checkpoint:', ckpt_file)
        except Exception as err:
            print('Failed to save checkpoint', ckpt_file)
            print('Error:', err)

    def save_prediction(self, epoch: int, test_lines: list):
        """
        保存推断结果 prediction/e{e}.s{s}.{dev,test}.txt
        """
        test_lines = [l + '\n' for l in test_lines]
        test_file = os.path.join(self.pred_dir, f'e{epoch}.test.txt')
        try:
            with open(test_file, 'w', encoding='utf-8') as fp:
                fp.writelines(test_lines)
            print('Prediction saved')
        except Exception as err:
            print('Failed to save prediction', test_file)
            print('Error:', err)

    def save_print(self, content):
        my_write_file_append(self.eval_file, content)
        print("Output saved...")

