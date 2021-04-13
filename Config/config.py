import os


def create_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)


CUDA_NUMBER = 0
weight_dir = '../weights/'
test_batch_size = 1

data_type_list = ['数据集3', '数据集4', '数据集2', '数据集1']
data_type = data_type_list[2]


weight_path = os.path.join(weight_dir, 'net_%s.pth' % data_type)
test_dir = '../dataset/test/%s' % data_type

test_result_root = '../result/'
test_compare_results_dir = os.path.join(test_result_root, '%s_compare' % data_type)
test_results_dir = os.path.join(test_result_root, '%s' % data_type)


train_batch_size = 1

train_dir = '../dataset/train/%s' % data_type

train_result_root = '../result/'
train_compare_results_dir = os.path.join(train_result_root, '%s_compare' % data_type)
train_results_dir = os.path.join(train_result_root, '%s' % data_type)


create_dir(test_result_root)
create_dir(test_compare_results_dir)
create_dir(test_results_dir)

create_dir(train_result_root)
create_dir(train_compare_results_dir)
create_dir(test_results_dir)
