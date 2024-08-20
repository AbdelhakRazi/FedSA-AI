FROM tensorflow/tensorflow
MAINTAINER Salah BinRuba "salah.binruba@cname.fr"
ENV CLIENTID=0
ENV URL = "localhost"
ENV k=1

#RUN pip install --upgrade pip && \
#    pip install Flask
WORKDIR FLClient

RUN pip install scikit-learn

COPY . .


#EXPOSE 3000
ENTRYPOINT ["python"]
CMD ["client_driver_current.py"]
#CMD ["python","envFile.py"]
#CMD ["python","mClientD.py"]