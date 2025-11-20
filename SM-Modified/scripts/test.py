import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.models import Model

def main():
    model = Model()
    model.test()

if __name__ == '__main__':
    main()