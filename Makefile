include setenv.sh

IMAGE_NAME := mldatautils_ex
build-docker-ex:
	docker build -t $(IMAGE_NAME) -f docker/Dockerfile .

run-docker-ex:
	docker run \
	-it \
	--rm \
	-p $(PORT):$(PORT) \
	-e DB_USERNAME=${DB_USERNAME} \
	-e DB_PASSWORD=${DB_PASSWORD} \
	-e DB_HOSTNAME=${DB_HOSTNAME} \
	-e DATABASE=${DATABASE} \
	-e PORT=$(PORT) \
	-e AWS_ACCESS_KEY_ID=$(AWS_ACCESS_KEY_ID) \
	-e AWS_SECRET_ACCESS_KEY=$(AWS_SECRET_ACCESS_KEY) \
	-v $(PWD):/work \
	$(IMAGE_NAME) \
	/bin/bash

run-docker-test:
	docker run \
	-it \
	--rm \
	-e DB_USERNAME=${DB_USERNAME} \
	-e DB_PASSWORD=${DB_PASSWORD} \
	-e DB_HOSTNAME=${DB_HOSTNAME} \
	-e DATABASE=${DATABASE} \
	-e AWS_ACCESS_KEY_ID=$(AWS_ACCESS_KEY_ID) \
	-e AWS_SECRET_ACCESS_KEY=$(AWS_SECRET_ACCESS_KEY) \
	-v $(PWD):/work \
	$(IMAGE_NAME) \
	pytest

upload-pypi:
	python setup.py sdist && \
	twine upload --repository-url https://upload.pypi.org/legacy/ dist/*
