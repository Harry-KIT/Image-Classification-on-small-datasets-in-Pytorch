import os
import random
import shutil
from tqdm import tqdm

class SplitDataset():
    def __init__(self, dataset_dir, saved_dataset_dir, train_ratio=0.7, show_progress=False):
        self.dataset_dir = dataset_dir
        self.saved_dataset_dir = saved_dataset_dir
        self.saved_train_dir = saved_dataset_dir + "/train/"
        self.saved_val_dir = saved_dataset_dir + "/val/"
        

        self.train_ratio = train_ratio
        

        self.train_file_path = []
        self.val_file_path = []

        self.index_label_dict = {}

        self.show_progress = show_progress

        if not os.path.exists(self.saved_train_dir):
            os.mkdir(self.saved_train_dir)
        if not os.path.exists(self.saved_val_dir):
            os.mkdir(self.saved_val_dir)


    def __get_label_names(self):
        label_names = []
        for item in os.listdir(self.dataset_dir):
            item_path = os.path.join(self.dataset_dir, item)
            if os.path.isdir(item_path):
                label_names.append(item)
        return label_names

    def __get_all_file_path(self):
        all_file_path = []
        index = 0
        for file_type in self.__get_label_names():
            self.index_label_dict[index] = file_type
            index += 1
            type_file_path = os.path.join(self.dataset_dir, file_type)
            file_path = []
            for file in os.listdir(type_file_path):
                single_file_path = os.path.join(type_file_path, file)
                file_path.append(single_file_path)
            all_file_path.append(file_path)
        return all_file_path

    def __copy_files(self, type_path, type_saved_dir):
        for item in tqdm(type_path):
            src_path_list = item[1]
            dst_path = type_saved_dir + "%s/" % (item[0])
            if not os.path.exists(dst_path):
                os.mkdir(dst_path)
            for src_path in src_path_list:
                shutil.copy(src_path, dst_path)
                # if self.show_progress:
                #     print("Copying file "+src_path+" to "+dst_path)

    def __split_dataset(self):
        all_file_paths = self.__get_all_file_path()
        for index in range(len(all_file_paths)):
            file_path_list = all_file_paths[index]
            file_path_list_length = len(file_path_list)
            random.shuffle(file_path_list)

            train_num = int(file_path_list_length * self.train_ratio)

            self.train_file_path.append([self.index_label_dict[index], file_path_list[: train_num]])
            self.val_file_path.append([self.index_label_dict[index], file_path_list[train_num:]])

    def start_splitting(self):
        self.__split_dataset()
        self.__copy_files(type_path=self.train_file_path, type_saved_dir=self.saved_train_dir)
        self.__copy_files(type_path=self.val_file_path, type_saved_dir=self.saved_val_dir)


if __name__ == '__main__':
    split_dataset = SplitDataset(dataset_dir="wild-animals-original",
                                 saved_dataset_dir="dataset",  # dataset_sclip, dataset_s3
                                 show_progress=True)
    split_dataset.start_splitting()