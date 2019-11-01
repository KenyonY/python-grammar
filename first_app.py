import streamlit as st
import numpy 
import pandas
import matplotlib.pyplot as plt

st.title("My first app")

st.write("Here's our first attempt at using data to create a table:")



chart_data = pandas.DataFrame(
     numpy.random.randn(20, 3),
     columns=['a', 'b', 'c'])

st.line_chart(chart_data)

if st.checkbox('Show dataframe'):
    chart_data = pandas.DataFrame(
       numpy.random.randn(20, 3),
       columns=['a', 'b', 'c'])

    st.line_chart(chart_data)

map_data = pandas.DataFrame(
    numpy.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
    columns=['lat', 'lon'])

st.map(map_data)

"""
# My first app
Here's our first attempt at using data to create a table:
"""

df = pandas.DataFrame({
  'first column': [1, 2, 3, 4],
  'second column': [10, 20, 30, 40]
})

df

# option = st.selectbox(
#     'Which number do you like best?',
#      df['first column'])

# 'You selected: ', option

option = st.sidebar.selectbox(
    'Which number do you like best?',
     df['first column'])

'You selected:', option

import time

'Starting a long computation...'
import numpy as np
progress_bar = st.progress(0)
status_text = st.empty()
chart = st.line_chart(np.random.randn(10, 2))

for i in range(100):
    # Update progress bar.
    progress_bar.progress(i)

    new_rows = np.random.randn(10, 2)

    # Update status text.
    status_text.text(
        f'The latest random number is: {new_rows[-1, 1]}')

    # Append data to the chart.
    chart.add_rows(new_rows)

    # Pretend we're doing some computation that takes time.
    time.sleep(0.1)

status_text.text('Done!')
st.balloons()
# import librosa
# import librosa.display
# import IPython.display as ipd

# y,sr = librosa.load(r'C:\Users\beidongjiedeguang\OneDrive\a_github\我的项目\a_my_deep_voice\voiceData\nikki\00.wav')
# ipd.Audio(y, rate=sr)


