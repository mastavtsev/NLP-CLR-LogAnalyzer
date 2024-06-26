import os
from tkinter import Tk
from tkinter.filedialog import asksaveasfilename, askopenfilename
import warnings

warnings.filterwarnings("ignore")


class FileManager:
    @staticmethod
    def get_filename(xes_file):
        base_name = os.path.basename(xes_file)
        file_name, _ = os.path.splitext(base_name)
        return file_name

    @staticmethod
    def get_save_path(default_filename, default_extension):
        '''
        :param default_filename: file name with extension
        :param default_extension: str extension with dot in front, like ".png"
        :return: path to file
        '''
        root = Tk()
        root.attributes('-topmost', True)
        root.withdraw()
        filetype = "*" + default_extension
        save_path = asksaveasfilename(
            initialfile=default_filename,
            defaultextension=default_extension,
            filetypes=[(f"{default_extension} files", filetype), ("All files", "*.*")]
        )
        root.destroy()
        return save_path

    @staticmethod
    def get_in_path(base_dir, default_extension):
        '''
        :param base_dir: directory to start searching files
        :param default_extension: str extension with dot in front, like ".txt"
        :return: path to file
        '''
        root = Tk()
        root.attributes('-topmost', True)
        root.withdraw()
        filetype = "*" + default_extension
        in_path = askopenfilename(
            initialdir=base_dir,
            defaultextension=default_extension,
            filetypes=[(f"{default_extension} files", filetype), ("All files", "*.*")]
        )
        root.destroy()
        return in_path
