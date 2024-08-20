FROM python:3
MAINTAINER Salah BinRuba "salah.binruba@cname.fr"
ENV CLIENTID=0
ENV URL = "localhost"
ENV k=1

RUN pip install --upgrade pip && \
    pip install numpy scikit-learn
WORKDIR FLSplitter

COPY . .


#EXPOSE 3000
ENTRYPOINT ["python"]
CMD ["db_splitter.py"]
#CMD ["python","envFile.py"]
#CMD ["python","mClientD.py"]