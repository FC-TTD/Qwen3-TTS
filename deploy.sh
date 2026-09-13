#!/bin/bash
set -e

# 默认 usage 输出
usage() {
  echo "Usage: $0 [--ci] [--cd] docker | swarm | sync"
  echo "Examples:"
  echo "  $0 docker         # 同步 + 构建镜像 + 部署容器 (Compose)"
  echo "  $0 swarm          # 部署最终形态到 Docker Swarm (2x base + 1x fusion)"
  echo "  $0 --ci docker    # 仅同步 + 构建镜像"
  echo "  $0 --cd docker    # 仅部署容器"
  echo "  $0 sync           # 仅同步项目文件"
}

CI_MODE=false
CD_MODE=false
SYNC_ONLY=false
ANSIBLE_DIR="./ansible"
TARGET=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --ci)
      CI_MODE=true
      shift
      ;;
    --cd)
      CD_MODE=true
      shift
      ;;
    sync)
      SYNC_ONLY=true
      shift
      ;;
    docker|swarm)
      TARGET="$1"
      shift
      ;;
    *)
      usage
      exit 1
      ;;
  esac
done

if [ "$SYNC_ONLY" = false ] && [ -z "$TARGET" ]; then
  echo "Error: 请指定部署目标 (docker) 或使用 sync"
  echo
  usage
  exit 1
fi

if [ "$SYNC_ONLY" = true ]; then
  echo "仅同步项目文件 (sync)..."
  ansible-playbook -i "$ANSIBLE_DIR/inventory.yml" "$ANSIBLE_DIR/site.yml" --tags "sync"
  exit 0
fi

TAGS="$TARGET"
if [ "$CI_MODE" = true ] && [ "$CD_MODE" = false ]; then
  TAGS="$TAGS,sync,ci"
  echo "仅同步 + 构建镜像 ($TARGET,sync,ci)..."
elif [ "$CI_MODE" = false ] && [ "$CD_MODE" = true ]; then
  TAGS="$TAGS,cd"
  echo "仅部署服务 ($TARGET,cd)..."
else
  TAGS="$TAGS,sync,ci,cd"
  echo "完整CI/CD流水线部署 ($TARGET,sync,ci,cd)..."
fi

EXTRA_ARGS=""
if [ "$TARGET" = "docker" ]; then
  EXTRA_ARGS="--skip-tags swarm"
fi

if [ "$TARGET" = "swarm" ]; then
  echo "目标将同时发布 qwen-tts(base) 与 qwen-tts-fusion(fusion) 两个 Swarm Stack"
  EXTRA_ARGS="--skip-tags docker"
fi

ansible-playbook -i "$ANSIBLE_DIR/inventory.yml" "$ANSIBLE_DIR/site.yml" --tags "$TAGS" $EXTRA_ARGS
