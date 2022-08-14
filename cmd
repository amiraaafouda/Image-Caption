docker build . -t ics
docker run -v $(pwd):/app -ti -p 8000:5000 ics
