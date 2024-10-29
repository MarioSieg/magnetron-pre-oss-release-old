# (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>
# GUI Viewer for MSML storage files with hex editor-like interface.

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import sys
import os

class MSMLViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MSML File Viewer")
        self.setWindowIcon(QIcon("icon.png"))
        self.resize(1920, 1080)

        # Main widget and layout
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)

        # Left - Tensor list
        self.tensor_list = QListWidget()
        self.tensor_list.setMinimumWidth(200)
        self.tensor_list.itemClicked.connect(self.show_tensor_data)

        # Right - Data display
        self.data_view = QTextEdit()
        self.data_view.setReadOnly(True)
        self.data_view.setFont(QFont("Courier", 10))

        # Add widgets to the layout
        main_layout.addWidget(self.tensor_list)
        main_layout.addWidget(self.data_view)

        # Set the main widget with layout
        self.setCentralWidget(main_widget)

        # Menu bar
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("File")
        open_action = QAction("Open", self)
        open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action)

    def open_file(self):
        # Open file dialog
        file_name, _ = QFileDialog.getOpenFileName(self, "Open MSML File", os.getcwd(), "MSML Files (*.msml);;All Files (*)")
        if file_name:
            # Load the file and populate tensor_list with tensors found in the file (placeholder code)
            self.tensor_list.clear()
            self.tensor_list.addItem("Tensor 1")
            self.tensor_list.addItem("Tensor 2")
            self.tensor_list.addItem("Tensor 3")

    def show_tensor_data(self, item):
        # Display the data of the selected tensor in hex format
        tensor_name = item.text()
        # Here, load tensor data from the file using tensor_name (placeholder example)
        hex_data = " ".join([f"{i:02X}" for i in range(256)])
        self.data_view.setText(hex_data)


def main():
    app = QApplication(sys.argv)
    viewer = MSMLViewer()
    viewer.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
