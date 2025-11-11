#!/usr/bin/env python
# coding: UTF-8

# 必要なモジュールのインポート
import pandas as pd
import re
import glob
# PyPDF2をインポート
from PyPDF2 import PdfReader, PdfWriter

# 入出力のファイル名
fname_in = "input.pdf"
fname_out = "output.pdf"

print("Hello, World!")
