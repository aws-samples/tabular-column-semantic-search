FROM public.ecr.aws/lambda/python:3.8

COPY . ${LAMBDA_TASK_ROOT}

RUN pip install --upgrade pip
RUN pip install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

CMD ["lambda_embed_payload.handler"]
