# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding='utf-8')
import config  # triggers sys.path setup
from data_loader import DataLoader
loader = DataLoader()
loader.print_status()
