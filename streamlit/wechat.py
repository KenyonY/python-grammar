import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import re
import time

import itchat
from itchat.content import *
import time
# import jieba
itchat.auto_login(hotReload=True)

name = 'guang'
guang = itchat.search_friends(name)
print(guang[0])
# st.write(guang[0])