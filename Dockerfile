FROM ubuntu:18.04

RUN apt-get update && apt-get install -y python3-pip vim wget
RUN pip3 install --upgrade pip setuptools

WORKDIR /app
COPY . . 
RUN pip3 install -r requirements.txt
RUN chmod +x testing.sh
CMD [ './testing.sh' ]
