#!/bin/bash
set -e

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
    docker)
      TARGET="$1"
      shift
      ;;
    *)
      echo "Usage: $0 [--ci] [--cd] docker | sync"
      echo "Examples:"
      echo "  $0 docker         # 同步 + 构建镜像 + 部署容器"
      echo "  $0 --ci docker    # 仅同步 + 构建镜像"
      echo "  $0 --cd docker    # 仅部署容器"
      echo "  $0 sync           # 仅同步项目文件"
      exit 1
      ;;
  esac
done

if [ "$SYNC_ONLY" = false ] && [ -z "$TARGET" ]; then
  echo "Error: 请指定部署目标 (docker) 或使用 sync"
  exit 1
fi

if [ "$SYNC_ONLY" = true ]; then
  echo "仅同步项目文件 (sync)..."
  ansible-playbook -i "$ANSIBLE_DIR/inventory.yml" "$ANSIBLE_DIR/site.yml" --tags "sync"
  exit 0
fi

TAGS="$TARGET"
if [ "$CI_MODE" = true ] && [ "$CD_MODE" = false ]; then
  TAGS="$TAGS,ci"
  echo "仅构建镜像 (docker,ci)..."
elif [ "$CI_MODE" = false ] && [ "$CD_MODE" = true ]; then
  TAGS="$TAGS,cd"
  echo "仅部署服务 (docker,cd)..."
else
  TAGS="$TAGS"
  echo "使用CI/CD流水线部署 (docker)..."
fi

ansible-playbook -i "$ANSIBLE_DIR/inventory.yml" "$ANSIBLE_DIR/site.yml" --tags "$TAGS"
