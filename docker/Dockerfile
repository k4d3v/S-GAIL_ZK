FROM zaku.sys.es.osaka-u.ac.jp:10081/miyazawa/pytorch-jupyterlab-docker

RUN pip install h5py
RUN pip install gym

CMD ["jupyter", "lab", "--no-browser", "--ip=0.0.0.0","--NotebookApp.token=''"]
