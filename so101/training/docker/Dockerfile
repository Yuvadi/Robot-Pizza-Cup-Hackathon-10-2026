# syntax=docker/dockerfile:1
# Training app image: policy YAML + resume entrypoint on the published deps layer.
#
#   bash training/scripts/build_docker.sh --push
#
# Override the base with DEPS_IMAGE=... or --build-arg DEPS_IMAGE=...

ARG DEPS_IMAGE=mnrozhkov/pizza-robo-hack:deps
FROM ${DEPS_IMAGE} AS app

WORKDIR /lerobot

COPY training/policy/ ./training/policy/
COPY training/scripts/train_entrypoint.sh /lerobot/train_entrypoint.sh
RUN chmod +x /lerobot/train_entrypoint.sh

ENTRYPOINT ["/lerobot/train_entrypoint.sh"]
CMD ["--config_path=training/policy/act/config.yaml"]
